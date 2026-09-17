"""
soc.py

Exact (second-order-cone) chance-constrained formulation, and the variance
diagnostics needed to compare it against the L1 shortcut in ``cc.py``.

Why this module exists
----------------------
``cc.compute_L1_env_cost_mean_var`` builds a *per-process* standard deviation

    sigma_j = sqrt( sum_e ( mu_q,e^2 sigma_b,ej^2
                          + mu_b,ej^2 sigma_q,e^2
                          + sigma_q,e^2 sigma_b,ej^2 ) )

and the chance constraint then aggregates those with an L1 sum,
``sum_j s_j (mu_j + z sigma_j)``. That has two separate consequences:

1. the L1 norm over-estimates the Euclidean norm (bounded by sqrt(n), SI eq.
   S23), so the formulation is conservative by an amount nobody has measured;
2. more importantly, the *same* characterization factor ``q_e`` multiplies
   every process ``j``, so the per-process contributions are strongly
   correlated. Summing per-process variances silently assumes they are not.

Both are fixed by aggregating the biosphere row *before* characterizing.
Writing the life-cycle impact as ``X = sum_e q_e g_e`` with
``g_e = sum_j b_ej s_j`` and keeping only the assumptions the SI already makes
(``q`` independent of ``b``, entries of ``b`` mutually independent, ``q_e``
independent across ``e``):

    E[g_e]   = sum_j mu_b,ej s_j
    Var[g_e] = sum_j s_j^2 sigma_b,ej^2
    Var[X]   = sum_e [ (mu_q,e^2 + sigma_q,e^2) Var[g_e]
                       + sigma_q,e^2 E[g_e]^2 ]

which rearranges into a *diagonal* quadratic form,

    Var[X] = sum_j d_j s_j^2 + sum_e w_e y_e^2
        d_j = sum_e ( mu_q,e^2 + sigma_q,e^2 ) sigma_b,ej^2     (a constant)
        w_e = sigma_q,e^2                (nonzero only where a CF is uncertain)
        y_e = sum_j mu_b,ej s_j          (the mean inventory flow e)

so the exact deterministic counterpart

    min  mu^T s + Phi^-1(lambda) * t     s.t.  sum_j d_j s_j^2
                                              + sum_e w_e y_e^2 <= t^2,  t >= 0

is a second-order cone program with one extra variable plus one auxiliary
variable per *characterized* flow (order 10^2, not 10^4). Gurobi solves it as a
QCP directly.

The only assumption that is *not* repaired here is correlation among the
``b_ej`` themselves; ecoinvent stores none, so it cannot be. That omission
biases the variance downwards, i.e. the chance constraint remains optimistic in
that one respect - state it, do not hide it.

Nothing in ``cc.py`` is modified: the L1 path stays bit-for-bit reproducible.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pyomo.environ as pyo
import scipy.sparse as sp
import scipy.stats
import stats_arrays

from pulpo.utils import optimizer
from pulpo.utils.uncertainty.preparer import UncertaintyData, UncertaintySpec

__all__ = [
    "SOCCoefficients",
    "compute_soc_coefficients",
    "current_scaling_vector",
    "impact_mean",
    "impact_std_exact",
    "impact_std_l2_independent",
    "impact_std_l1",
    "impact_std_l1_signed",
    "impact_std_gradient",
    "apply_SOC_formulation",
    "solve_soc",
    "prepare_exact_model",
    "solve_exact",
]


# ---------------------------------------------------------------------------
# Coefficient assembly
# ---------------------------------------------------------------------------
class SOCCoefficients:
    """Everything the exact formulation needs, in matrix form.

    Attributes
    ----------
    mu_env_cost : np.ndarray, shape (n_process,)
        ``mu_j = sum_e mu_q,e mu_b,ej`` - the expected environmental cost of
        one unit of process j. Identical in meaning to the ``loc`` entries
        produced by ``cc.compute_L1_env_cost_mean_var``.
    d : np.ndarray, shape (n_process,)
        The diagonal process term of the variance.
    w : np.ndarray, shape (n_uncertain_cf,)
        ``sigma_q,e^2`` for the flows whose characterization factor is
        uncertain.
    cf_rows : np.ndarray, shape (n_uncertain_cf,)
        Biosphere row indices matching ``w``.
    B_mean_unc : scipy.sparse.csr_matrix, shape (n_uncertain_cf, n_process)
        Rows of the mean biosphere matrix needed to build ``y_e``.
    sigma_env_cost : np.ndarray, shape (n_process,)
        The *per-process* standard deviation as computed by the L1 path, kept
        here so the three norms can be compared on identical inputs.
    """

    def __init__(self, mu_env_cost, d, w, cf_rows, B_mean_unc, sigma_env_cost,
                 method):
        self.mu_env_cost = mu_env_cost
        self.d = d
        self.w = w
        self.cf_rows = cf_rows
        self.B_mean_unc = B_mean_unc
        self.sigma_env_cost = sigma_env_cost
        self.method = method

    def summary(self) -> dict:
        return {
            "processes": int(self.mu_env_cost.size),
            "processes_with_variance": int((self.d > 0).sum()),
            "uncertain_cfs": int(self.w.size),
            "cf_variance_share_possible": bool(self.w.size > 0),
        }


def _mean_and_sigma_matrices(normal_uncertainty_data: UncertaintyData,
                             lci_data: dict, method: str):
    """Build (B_mean, Sigma_b, q_mean, q_sigma) from the normal-fitted specs.

    Defaults come from the deterministic LCI; every parameter that carries a
    fitted normal overrides its deterministic counterpart, exactly as the L1
    path does.
    """
    B = lci_data["intervention_matrix"]
    n_flow, n_proc = B.shape

    # --- mean biosphere matrix -------------------------------------------
    B_mean = B.tolil(copy=True)
    rows, cols, scales = [], [], []
    for if_data in normal_uncertainty_data["If"].values():
        for (flow_idx, process_id), spec in if_data["defined"].items():
            B_mean[flow_idx, process_id] = spec["loc"]
            rows.append(flow_idx)
            cols.append(process_id)
            scales.append(spec["scale"])
    B_mean = B_mean.tocsr()

    Sigma_b = sp.coo_matrix(
        (np.asarray(scales, dtype=float),
         (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))),
        shape=(n_flow, n_proc)).tocsr()

    # --- characterization factors ----------------------------------------
    q_mean = np.asarray(lci_data["matrices"][method].diagonal(), dtype=float).ravel()
    q_sigma = np.zeros(n_flow)
    for flow_idx, spec in normal_uncertainty_data["Cf"][method]["defined"].items():
        q_mean[flow_idx] = spec["loc"]
        q_sigma[flow_idx] = spec["scale"]

    return B_mean, Sigma_b, q_mean, q_sigma


def compute_soc_coefficients(normal_uncertainty_data: UncertaintyData,
                             lci_data: dict,
                             method: str,
                             verbose: bool = True) -> SOCCoefficients:
    """Assemble the exact-variance coefficients described in the module docstring."""
    B_mean, Sigma_b, q_mean, q_sigma = _mean_and_sigma_matrices(
        normal_uncertainty_data, lci_data, method)

    # mu_j = sum_e mu_q,e mu_b,ej
    mu_env_cost = np.asarray(q_mean @ B_mean).ravel()

    # d_j = sum_e (mu_q,e^2 + sigma_q,e^2) sigma_b,ej^2
    Sigma_b_sq = Sigma_b.multiply(Sigma_b).tocsr()
    d = np.asarray((q_mean ** 2 + q_sigma ** 2) @ Sigma_b_sq).ravel()

    # The CF-sharing term only exists where a characterization factor is
    # genuinely uncertain. With the as-published CF settings this set is
    # effectively empty, which is exactly why the omission of covariance went
    # unnoticed.
    cf_rows = np.flatnonzero(q_sigma > 0)
    w = q_sigma[cf_rows] ** 2
    B_mean_unc = B_mean[cf_rows, :].tocsr()

    # The per-process sigma of the L1 path, recomputed here so that all three
    # norms are built from one consistent set of inputs.
    #   sigma_j^2 = sum_e ( mu_q,e^2 sigma_b,ej^2 + mu_b,ej^2 sigma_q,e^2
    #                       + sigma_q,e^2 sigma_b,ej^2 )
    B_mean_sq = B_mean.multiply(B_mean).tocsr()
    sigma_sq = (np.asarray((q_mean ** 2 + q_sigma ** 2) @ Sigma_b_sq).ravel()
                + np.asarray(q_sigma ** 2 @ B_mean_sq).ravel())
    sigma_env_cost = np.sqrt(np.maximum(sigma_sq, 0.0))

    coeffs = SOCCoefficients(mu_env_cost, d, w, cf_rows, B_mean_unc,
                             sigma_env_cost, method)
    if verbose:
        s = coeffs.summary()
        print("SOC coefficients assembled:")
        print(f"  processes                      : {s['processes']:,}")
        print(f"  processes with flow variance   : {s['processes_with_variance']:,}")
        print(f"  uncertain characterization f.  : {s['uncertain_cfs']:,}")
        if s["uncertain_cfs"] == 0:
            print("  NOTE: no characterization factor carries uncertainty, so the "
                  "CF-sharing covariance term is exactly zero. This is the "
                  "as-published configuration.")
    return coeffs


# ---------------------------------------------------------------------------
# Post-hoc evaluation of the three norms at a given scaling vector
# ---------------------------------------------------------------------------
def _as_vector(s, n: int) -> np.ndarray:
    """Accept a dict {process_id: value}, a pandas Series or an array."""
    if isinstance(s, np.ndarray):
        return s
    vec = np.zeros(n)
    items = s.items() if hasattr(s, "items") else enumerate(s)
    for j, value in items:
        vec[int(j)] = float(value)
    return vec


def current_scaling_vector(model_instance, n: int) -> np.ndarray:
    """Read the current scaling vector off a solved model instance.

    Use this after a solve to evaluate :func:`impact_mean` / :func:`impact_std_exact`
    / etc. at the actual optimum, e.g. because ``model.impacts[h]`` itself is left at
    the *mean* by :func:`apply_SOC_formulation` / :func:`prepare_exact_model` (the
    chance-constrained penalty lives in ``z * SOC_T`` / ``z * SOC_CUTS``, not in
    ``impacts[h]``).
    """
    s = np.zeros(n)
    for j in model_instance.PROCESS:
        value = pyo.value(model_instance.scaling_vector[j], exception=False)
        if value is not None:
            s[int(j)] = value
    return s


def impact_mean(s, coeffs: SOCCoefficients) -> float:
    """Expected life-cycle impact ``mu^T s``."""
    vec = _as_vector(s, coeffs.mu_env_cost.size)
    return float(coeffs.mu_env_cost @ vec)


def impact_std_exact(s, coeffs: SOCCoefficients) -> float:
    """Exact standard deviation, including the shared-CF covariance (R3-4)."""
    vec = _as_vector(s, coeffs.mu_env_cost.size)
    diag = float(coeffs.d @ (vec ** 2))
    if coeffs.w.size:
        y = np.asarray(coeffs.B_mean_unc @ vec).ravel()
        shared = float(coeffs.w @ (y ** 2))
    else:
        shared = 0.0
    return float(np.sqrt(max(diag + shared, 0.0)))


def impact_std_l2_independent(s, coeffs: SOCCoefficients) -> float:
    """Euclidean norm of the per-process sigmas - SI eq. S21, no covariance."""
    vec = _as_vector(s, coeffs.mu_env_cost.size)
    return float(np.sqrt(np.sum((vec * coeffs.sigma_env_cost) ** 2)))


def impact_std_l1(s, coeffs: SOCCoefficients) -> float:
    """The true L1 norm, ``sum_j sigma_j |s_j|`` - SI eq. S22/S24."""
    vec = _as_vector(s, coeffs.mu_env_cost.size)
    return float(np.abs(vec) @ coeffs.sigma_env_cost)


def impact_std_gradient(s, coeffs: SOCCoefficients) -> tuple[float, np.ndarray]:
    """Return ``(sigma(s), grad sigma(s))`` for the exact standard deviation.

    With ``Q(s) = sum_j d_j s_j^2 + sum_e w_e y_e^2`` and ``y = B_unc s``,

        grad Q     = 2 d*s + 2 B_unc^T (w * y)
        grad sigma = grad Q / (2 sigma)

    ``sigma`` is convex and positively homogeneous of degree one, so
    ``grad sigma(s)^T s = sigma(s)``: the supporting hyperplane at any point is
    exact there and underestimates everywhere else. That is what makes the
    cutting-plane solve below both valid and finitely convergent.
    """
    vec = _as_vector(s, coeffs.mu_env_cost.size)
    grad_q = 2.0 * coeffs.d * vec
    if coeffs.w.size:
        y = np.asarray(coeffs.B_mean_unc @ vec).ravel()
        grad_q = grad_q + 2.0 * np.asarray(
            coeffs.B_mean_unc.T @ (coeffs.w * y)).ravel()
        q = float(coeffs.d @ (vec ** 2) + coeffs.w @ (y ** 2))
    else:
        q = float(coeffs.d @ (vec ** 2))
    sigma = float(np.sqrt(max(q, 0.0)))
    if sigma <= 0.0:
        return 0.0, np.zeros_like(vec)
    return sigma, grad_q / (2.0 * sigma)


def impact_std_l1_signed(s, coeffs: SOCCoefficients) -> float:
    """What the LP objective actually adds: ``sum_j sigma_j s_j``, unsigned.

    The SI replaces ``sum_j sigma_j |s_j|`` by ``sum_j sigma_j s_j``, an
    identity that holds only for a non-negative scaling vector. PULPO pins
    choice alternatives to ``s >= 0`` but leaves other processes free
    (``converter.py`` defaults the lower bound to -inf), and avoided-burden or
    treatment activities do settle at ``s_j < 0``. For those the penalty enters
    with the wrong sign. Comparing this against :func:`impact_std_l1` measures
    how much uncertainty the objective silently cancels.
    """
    vec = _as_vector(s, coeffs.mu_env_cost.size)
    return float(vec @ coeffs.sigma_env_cost)


# ---------------------------------------------------------------------------
# The optimization model
# ---------------------------------------------------------------------------
_SOC_COMPONENTS = ("SOC_T", "SOC_TERM", "SOC_C", "SOC_C_CNSTR", "SOC_FLOW",
                   "SOC_Y", "SOC_Y_CNSTR", "SOC_CONE", "SOC_CUTS",
                   "SOC_C_CNSTR_index", "SOC_C_index",
                   "SOC_Y_CNSTR_index", "SOC_Y_index")

PLACEHOLDER_BOUND = 1e19


def _relax_placeholder_bounds(model_instance, threshold: float = PLACEHOLDER_BOUND,
                              replacement: float | None = None) -> int:
    """Re-scale 'effectively infinite' capacity bounds so a barrier can cope.

    PULPO case studies use 1e20 to mean 'uncapacitated'. An LP simplex solve
    never notices, but the interior-point method used for the cone does: a
    placeholder that large destroys the conditioning of the whole system.

    ``replacement`` is the finite value the placeholders are moved to. A finite
    replacement is preferred over ``inf``: making the variables genuinely free
    leaves the barrier with thousands of free directions and it stalls. The
    replacement must be large enough never to bind - verify that afterwards by
    checking no scaling factor sits at it. Returns the number of bounds moved.
    """
    n = 0
    for param_name, sign in (("UPPER_LIMIT", 1.0), ("UPPER_INV_LIMIT", 1.0),
                             ("LOWER_LIMIT", -1.0), ("LOWER_INV_LIMIT", -1.0)):
        param = getattr(model_instance, param_name, None)
        if param is None:
            continue
        updates = {}
        for index in param:
            value = pyo.value(param[index])
            if sign * value >= threshold:
                updates[index] = (sign * float("inf") if replacement is None
                                  else sign * replacement)
        if updates:
            param.store_values(updates, check=False)
            n += len(updates)
    return n


def check_placeholder_not_binding(model_instance, replacement: float,
                                  rel_tol: float = 1e-6) -> int:
    """Confirm the finite replacement bound never became an active constraint."""
    n_binding = 0
    for j in model_instance.PROCESS:
        value = pyo.value(model_instance.scaling_vector[j], exception=False)
        if value is None:
            continue
        if abs(abs(value) - replacement) <= rel_tol * replacement:
            n_binding += 1
    return n_binding


def apply_SOC_formulation(model_instance, lambda_level: float,
                          coeffs: SOCCoefficients,
                          s_ref=None,
                          var_coverage: float = 1.0 - 1e-8,
                          relax_bounds: bool = True,
                          placeholder_bound: float = PLACEHOLDER_BOUND,
                          placeholder_replacement: float | None = None,
                          rebuild: bool = False) -> dict:
    """Turn an instantiated PULPO model into the exact SOC chance-constrained one.

    The impact constraint is left at the *mean* environmental cost, so
    ``model.impacts[h]`` keeps its meaning (expected impact) and the
    uncertainty penalty appears separately as ``Phi^-1(lambda) * SOC_T``. That
    separation is also what R3-28 asks for in the figures.

    Conditioning
    ------------
    The cone is written with **unit quadratic coefficients** by introducing one
    auxiliary variable per term,

        c_k = sqrt(d_j) s_j ,   y_e = sqrt(w_e) sum_j mu_b,ej s_j
        sum_k c_k^2 + sum_e y_e^2 <= t^2 ,  t >= 0

    rather than feeding Gurobi ``sum_j d_j s_j^2 <= t^2`` directly. The d_j
    span roughly twenty orders of magnitude across an ecoinvent-scale system,
    and putting that range inside a quadratic constraint makes the barrier
    fail; moving it into the linear defining constraints does not.

    ``s_ref`` (typically the L1 solution) additionally allows dropping terms
    that carry a negligible share of the variance - the same "isolate the
    dominant uncertain parameters" argument the SI already makes for the L1
    approximation, but here with a measured and reported truncation error.

    Calling this repeatedly for a lambda sweep only rebuilds the objective.
    """
    z = float(scipy.stats.norm.ppf(lambda_level))
    method = coeffs.method
    info: dict = {}

    if rebuild:
        for name in _SOC_COMPONENTS:
            if hasattr(model_instance, name):
                model_instance.del_component(getattr(model_instance, name))

    first_build = not hasattr(model_instance, "SOC_T")
    if first_build:
        if relax_bounds:
            n_relaxed = _relax_placeholder_bounds(model_instance, placeholder_bound,
                                                  placeholder_replacement)
            info["bounds_relaxed"] = n_relaxed
            info["placeholder_replacement"] = placeholder_replacement
            if n_relaxed:
                target = ("infinity" if placeholder_replacement is None
                          else f"{placeholder_replacement:g}")
                print(f"Relaxed {n_relaxed:,} placeholder bounds (>= "
                      f"{placeholder_bound:g}) to {target} for the cone solve")

        # impacts[h] must carry the *mean* environmental cost.
        optimizer.update_env_cost(model_instance, {
            (int(j), method): float(value)
            for j, value in enumerate(coeffs.mu_env_cost)
        })

        scaling = model_instance.scaling_vector

        # --- which process terms to carry -------------------------------
        nz = np.flatnonzero(coeffs.d > 0)
        if s_ref is not None and var_coverage < 1.0:
            ref = _as_vector(s_ref, coeffs.mu_env_cost.size)
            contribution = coeffs.d[nz] * ref[nz] ** 2
            order = nz[np.argsort(contribution)[::-1]]
            csum = np.cumsum(np.sort(contribution)[::-1])
            total = csum[-1] if csum.size else 0.0
            if total > 0:
                keep = int(np.searchsorted(csum / total, var_coverage) + 1)
                dropped_share = 1.0 - csum[min(keep, csum.size) - 1] / total
                nz = order[:keep]
                info["process_terms_dropped_variance_share"] = float(dropped_share)
        d_terms = [(int(j), float(np.sqrt(coeffs.d[j]))) for j in nz]

        model_instance.SOC_T = pyo.Var(domain=pyo.NonNegativeReals,
                                       doc="Euclidean norm of the impact uncertainty")

        model_instance.SOC_TERM = pyo.Set(initialize=list(range(len(d_terms))),
                                          doc="Per-process variance terms")
        model_instance.SOC_C = pyo.Var(model_instance.SOC_TERM, domain=pyo.Reals,
                                       doc="sqrt(d_j) * s_j")

        def _c_rule(model, k):
            j, root = d_terms[k]
            return model.SOC_C[k] == root * scaling[j]

        model_instance.SOC_C_CNSTR = pyo.Constraint(model_instance.SOC_TERM,
                                                    rule=_c_rule)

        # --- shared-CF terms: y_e = sqrt(w_e) * sum_j mu_b,ej s_j -------
        w_idx = [e for e in range(coeffs.w.size) if coeffs.w[e] > 0]
        B = coeffs.B_mean_unc
        model_instance.SOC_FLOW = pyo.Set(initialize=w_idx,
                                          doc="Flows with uncertain characterization factors")
        model_instance.SOC_Y = pyo.Var(model_instance.SOC_FLOW, domain=pyo.Reals,
                                       doc="sqrt(sigma_q,e^2) * mean inventory of flow e")

        def _y_rule(model, e):
            root = float(np.sqrt(coeffs.w[e]))
            lo, hi = B.indptr[e], B.indptr[e + 1]
            if hi == lo:
                return model.SOC_Y[e] == 0.0
            cols = B.indices[lo:hi]
            vals = B.data[lo:hi]
            return model.SOC_Y[e] == pyo.quicksum(
                root * float(v) * scaling[int(j)] for j, v in zip(cols, vals))

        model_instance.SOC_Y_CNSTR = pyo.Constraint(model_instance.SOC_FLOW,
                                                    rule=_y_rule)

        def _cone_rule(model):
            quad = pyo.quicksum(model.SOC_C[k] ** 2 for k in model.SOC_TERM)
            if w_idx:
                quad += pyo.quicksum(model.SOC_Y[e] ** 2 for e in model.SOC_FLOW)
            return quad <= model.SOC_T ** 2

        model_instance.SOC_CONE = pyo.Constraint(rule=_cone_rule)
        info["process_terms"] = len(d_terms)
        info["shared_cf_terms"] = len(w_idx)
        print(f"SOC cone built: {len(d_terms):,} process terms, "
              f"{len(w_idx):,} shared-CF terms, unit quadratic coefficients")
        if "process_terms_dropped_variance_share" in info:
            print(f"  truncation: dropped terms carry "
                  f"{info['process_terms_dropped_variance_share']:.2e} of the "
                  f"reference variance")

    # Objective: expected impact + z * ||.||_2. Rebuilt for every lambda.
    model_instance.del_component(model_instance.OBJ)
    model_instance.OBJ = pyo.Objective(
        sense=pyo.minimize,
        expr=pyo.quicksum(model_instance.impacts[h] * model_instance.WEIGHTS[h]
                          for h in model_instance.INDICATOR)
        + z * model_instance.SOC_T,
    )
    print(f"Applying exact SOC chance constraint with lambda: {lambda_level} "
          f"(z = {z:.4f})")
    return info


# Barrier settings for the cone. The LP-oriented defaults used elsewhere in
# PULPO (ScaleFlag=2 in particular) hurt here; the homogeneous barrier is what
# Gurobi itself recommends for a cone this badly scaled.
SOC_SOLVER_OPTIONS = {
    "BarHomogeneous": 1,
    "NumericFocus": 3,
    "BarQCPConvTol": 1e-10,
    "FeasibilityTol": 1e-8,
    "OptimalityTol": 1e-8,
}


def solve_soc(model_instance, options: dict | None = None, tee: bool = False):
    """Solve the SOC model with Gurobi as a QCP.

    Gurobi recognises ``sum c_k^2 <= t^2, t >= 0`` as a second-order cone, so no
    non-convex flag is needed. If a build ever produces a form Gurobi rejects,
    ``NonConvex=2`` would be the escape hatch - but needing it means the cone
    was assembled wrongly and should be fixed rather than forced.
    """
    solver = pyo.SolverFactory("gurobi")
    merged = dict(SOC_SOLVER_OPTIONS)
    merged.update(options or {})
    for key, value in merged.items():
        if key != "tee":
            solver.options[key] = value
    results = solver.solve(model_instance, tee=tee, load_solutions=True)
    model_instance.solver_status = results.solver.status
    model_instance.solver_termination = results.solver.termination_condition
    print(f"SOC solved: status={results.solver.status}, "
          f"termination={results.solver.termination_condition}")
    return results, model_instance


# ---------------------------------------------------------------------------
# Exact solve by cutting planes (the method actually used on ecoinvent)
# ---------------------------------------------------------------------------
def prepare_exact_model(model_instance, coeffs: SOCCoefficients) -> None:
    """Set up the LP-with-cuts form of the exact chance-constrained problem.

    Why not hand the cone straight to Gurobi? Because an ecoinvent technosphere
    is far outside what an interior-point method tolerates: Gurobi reports a
    "large matrix coefficient range" and silently drops several hundred
    coefficients below 1e-13, and the barrier then stalls with a primal
    residual many orders of magnitude above the objective. Simplex has no such
    trouble with the same matrix.

    So the nonlinearity is peeled off instead. The only nonlinear object is the
    scalar ``sigma(s)``, which is convex and positively homogeneous, hence the
    supremum of its supporting hyperplanes:

        sigma(s) = max_k  grad sigma(s^k)^T s

    Replacing it by a variable ``T`` bounded below by a growing set of those
    hyperplanes turns each iteration into the *same LP* PULPO already solves,
    and the iteration converges to the exact cone optimum with a certified gap.
    """
    optimizer.update_env_cost(model_instance, {
        (int(j), coeffs.method): float(value)
        for j, value in enumerate(coeffs.mu_env_cost)
    })
    if not hasattr(model_instance, "SOC_T"):
        model_instance.SOC_T = pyo.Var(domain=pyo.NonNegativeReals, initialize=0.0,
                                       doc="Exact standard deviation of the impact")
        model_instance.SOC_CUTS = pyo.ConstraintList(
            doc="Supporting hyperplanes of sigma(s)")


def _set_exact_objective(model_instance, z: float) -> None:
    model_instance.del_component(model_instance.OBJ)
    model_instance.OBJ = pyo.Objective(
        sense=pyo.minimize,
        expr=pyo.quicksum(model_instance.impacts[h] * model_instance.WEIGHTS[h]
                          for h in model_instance.INDICATOR)
        + float(z) * model_instance.SOC_T,
    )


def _cut_support(grad: np.ndarray, s: np.ndarray, sigma: float,
                 tol: float) -> np.ndarray:
    """Which gradient entries to write into a cut.

    A supporting hyperplane of sigma at s^k carries one coefficient per process,
    and on an ecoinvent technosphere those span an enormous range - the largest
    is some thirty orders of magnitude above the smallest. A linear program
    whose rows have that dynamic range is not solved reliably: the solver's
    presolve drops coefficients below its own threshold, so each cut added
    changes the problem being solved rather than only tightening it, and the
    sequence of LP optima stops being monotone. Once that happens the lower and
    upper bounds can cross and nothing is certified.

    So a coefficient is written only if it can matter. "Can matter" is measured
    against sigma itself, which is available because the hyperplane is exact at
    the point it was taken from: ``grad @ s == sigma`` by homogeneity, so a term
    contributing less than ``tol * sigma`` there is below the precision the
    result is quoted to.

    **Only positive coefficients are ever dropped.** Removing a positive term
    lowers the right-hand side, which weakens the cut - and a weaker cut is
    still a valid underestimator of sigma, so the bound it produces stays sound.
    Removing a negative term would raise the right-hand side and could cut off
    the true optimum, turning a conditioning fix into a wrong answer. In this
    system the negative entries number a couple of dozen against several
    thousand, so keeping all of them costs nothing.
    """
    nz = np.flatnonzero(grad)
    if tol <= 0.0 or sigma <= 0.0 or nz.size == 0:
        return nz
    contribution = grad[nz] * s[nz]
    keep = (grad[nz] < 0.0) | (np.abs(contribution) >= tol * sigma)
    return nz[keep]


def solve_exact(model_instance, coeffs: SOCCoefficients, lambda_level: float,
                solve_fn, max_iter: int = 40, rel_gap: float = 1e-6,
                stall_patience: int = 3, cut_coeff_tol: float = 0.0,
                verbose: bool = True) -> dict:
    """Solve the exact chance-constrained problem by Kelley cutting planes.

    ``solve_fn(model)`` must solve the current LP in place (e.g. a closure over
    ``PulpoOptimizer.solve``). Cuts accumulate on the model, so sweeping lambda
    in increasing order reuses all previous work.

    Returns a dict with the optimal value, the certified relative gap and the
    iteration count. The gap is real, not nominal: every cut underestimates
    sigma, so ``mu^T s + z T`` is a lower bound while ``mu^T s + z sigma(s)``
    evaluated at the same s is an achievable upper bound.

    Both bounds are tracked as running best values, and that is not tidiness.
    Every LP optimum is a valid lower bound on the true optimum and every
    evaluated iterate a valid upper bound, so

        max_k lower_k  <=  v*  <=  min_k upper_k

    must hold. Pairing the best upper with the *latest* lower instead mixes two
    different relaxations, and on a degenerate problem the two can cross - which
    reports a "gap" that is really the size of an inconsistency, and reports it
    as though the optimum had been bracketed. A crossing is therefore measured
    and returned in ``bound_crossing`` rather than folded into the gap by an
    absolute value, and it never counts as convergence: a bracket that has
    inverted has not converged, it has failed, and the caller has to be able to
    tell those apart.
    """
    z = float(scipy.stats.norm.ppf(lambda_level))
    _set_exact_objective(model_instance, z)

    best_upper = np.inf
    best_lower = -np.inf
    best_residual = np.inf
    stalled = 0
    history = []
    for iteration in range(1, max_iter + 1):
        solve_fn(model_instance)

        s = np.zeros(coeffs.mu_env_cost.size)
        for j in model_instance.PROCESS:
            value = pyo.value(model_instance.scaling_vector[j], exception=False)
            if value is not None:
                s[int(j)] = value

        sigma, grad = impact_std_gradient(s, coeffs)
        mean = float(coeffs.mu_env_cost @ s)
        # At lambda = 0.5 the epigraph variable carries no objective weight and
        # no cuts yet, so the presolver drops it and Pyomo gets no value back.
        t_raw = pyo.value(model_instance.SOC_T, exception=False)
        t_value = 0.0 if t_raw is None else float(t_raw)

        lower = mean + z * t_value          # LP optimum: cuts underestimate sigma
        upper = mean + z * sigma            # achievable at this same s
        best_upper = min(best_upper, upper)
        best_lower = max(best_lower, lower)

        scale = max(abs(best_upper), 1e-12)
        spread = best_upper - best_lower
        # Positive spread is an optimality gap; negative is the two bounds
        # having crossed, which no amount of cutting explains and which must not
        # be reported as a gap of zero.
        gap = max(0.0, spread) / scale
        crossing = max(0.0, -spread)
        residual = abs(spread) / scale
        history.append({"iteration": iteration, "lower": lower,
                        "upper": upper, "best_lower": best_lower,
                        "best_upper": best_upper, "sigma": sigma, "T": t_value,
                        "gap": gap, "crossing": crossing})
        if verbose:
            print(f"    cut {iteration:2d}: LB={lower:.6g}  UB={upper:.6g}  "
                  f"sigma={sigma:.6g}  T={t_value:.6g}  gap={gap:.2e}"
                  + (f"  CROSSED by {crossing:.3g}" if crossing else ""))

        # Convergence needs the bracket to be both tight and intact. Testing
        # only the gap would declare success the moment the bounds crossed,
        # because a crossed bracket has a gap of exactly zero.
        if z <= 0.0 or residual <= rel_gap:
            break

        # Degenerate LPs can return the same vertex repeatedly once the cuts
        # bite; the remaining spread is then solver tolerance, not a missing cut.
        # Stop and report what was achieved instead of spinning to max_iter.
        # Progress is measured on the residual so that a widening crossing
        # counts as failure to progress rather than as an improving gap.
        if residual < best_residual * (1 - 1e-3):
            best_residual, stalled = residual, 0
        else:
            stalled += 1
            if stalled >= stall_patience:
                if verbose:
                    print(f"    stalled at residual={residual:.2e} after "
                          f"{iteration} cuts (LP degeneracy, not a missing "
                          f"cut) - stopping")
                break

        nz = _cut_support(grad, s, sigma, cut_coeff_tol)
        model_instance.SOC_CUTS.add(
            model_instance.SOC_T >= pyo.quicksum(
                float(grad[j]) * model_instance.scaling_vector[int(j)] for j in nz))

    # The certificate that actually decides optimality, and it needs nothing
    # from any other iterate. The LP is a relaxation of the true problem, so its
    # optimum never exceeds the true one; if T has risen to sigma at the point
    # the LP returned, the LP's value IS the true objective there, and a
    # feasible point attaining a value no larger than the relaxation's optimum
    # is optimal. Hence T == sigma(s*) at the final solve is sufficient on its
    # own - which is worth saying plainly, because the bracket below compares
    # bounds taken from different relaxations and on a degenerate problem those
    # can cross, certifying nothing.
    exactness = abs(t_value - sigma) / sigma if sigma > 0 else float("nan")
    return {"objective": best_upper, "lower_bound": best_lower, "gap": gap,
            "bound_crossing": crossing, "iterations": iteration,
            "sigma": sigma, "T": t_value, "exactness": exactness,
            "mean": mean, "history": history, "z": z}


def restore_linear_objective(model_instance) -> None:
    """Drop the SOC components and put back the plain weighted-impact objective."""
    for name in _SOC_COMPONENTS:
        if hasattr(model_instance, name):
            model_instance.del_component(getattr(model_instance, name))
    model_instance.del_component(model_instance.OBJ)
    model_instance.OBJ = pyo.Objective(
        sense=pyo.minimize,
        expr=pyo.quicksum(model_instance.impacts[h] * model_instance.WEIGHTS[h]
                          for h in model_instance.INDICATOR),
    )
