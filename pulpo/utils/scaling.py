"""LP equilibration for PULPO's optimization model.

Why
---
An ecoinvent technosphere spans roughly 1e-13 .. 2e+11. Infrastructure
processes have a functional unit of one whole facility -- a port consumes
2.3e11 kWh, a hydropower plant 1.7e11 kg of gravel -- while their activity
level in any solution is ~1e-10 facilities. Every LP solver scales each row by
its largest coefficient and applies its feasibility tolerance (1e-7 .. 1e-6) in
that scaled space, so the balance row of a facility product counts as satisfied
even when the facility is short-supplied by 1e-7 units: more than its whole
activity level, and enough to obtain 2e11 * 1e-7 kWh of electricity for free.
On a 52k-process model this leak was worth 0.9% of the GWP optimum; HiGHS,
Gurobi and their numeric-focus / tolerance options all returned points that
were only feasible under that tolerance, and disagreed with each other.

What
----
:func:`equilibrate_model_data` rewrites the data dictionary produced by
``combine_inputs`` / ``combine_inputs_time`` in place, substituting

    x_j = s_j * y_j          (column scaling of the scaling vector)
    row_i * r_i              (row scaling of every product balance)

with ``r`` and ``s`` chosen by alternating geometric-mean equilibration so that
all meaningful coefficients of ``diag(r) A diag(s)`` are O(1e-4 .. 1e4). The
factors are powers of two, hence the scaled LP is bit-for-bit equivalent to the
original one. The impact variables, their limits and the objective weights are
untouched, so ``instance.impacts`` reads exactly as before; after the solve,
:func:`unscale_solution` multiplies the scaling vector back.

Reading a solved instance: ``scaling_vector`` and ``slack`` hold *unscaled*
values as soon as ``solve_model`` returns. The constraint rows of the instance,
``model._env_cost`` and the limit Params stay in scaled units; the factors are
available as ``model._row_scale`` (per product) and ``model._col_scale`` (per
process).

Writing onto a scaled instance: anything that adds a term ``c * x_j`` to a
constraint or a bound ``x_j <= b`` after ``instantiate`` must express it in the
model's units, i.e. ``(c * s_j) * scaling_vector[j]`` and ``b / s_j`` in
``LOWER_LIMIT`` / ``UPPER_LIMIT``. :func:`to_scaled_coefficient`,
:func:`to_scaled_process_bound` and :func:`from_scaled_process_bound` do
that and are the identity on an unscaled model; the uncertainty formulations
(``uncertainty.soc`` and ``uncertainty.cc``) go through them. Impact and
inventory quantities are never scaled and need no conversion.
"""

import warnings

import numpy as np
import scipy.sparse as sps

#: Scaled variable bounds beyond this magnitude are treated as infinite. A
#: bound of 1e9 on a facility whose column is scaled by 1e-11 becomes 1e20,
#: which HiGHS already treats as infinity and which makes Gurobi's simplex
#: return O(1) row residuals on the scaled model.
SCALED_BOUND_CAP = 1e15

#: Gurobi options for an equilibrated model, applied by ``solve_gurobi`` unless
#: the caller overrides them. Gurobi's own scaling is switched off (it would
#: re-introduce the problem); the tightened tolerances and NumericFocus=1 keep
#: the residuals measured on the *unscaled* model at ~1e-8 without measurable
#: cost in time. Add ``"Method": 1`` (dual simplex) for bit-identical repeated
#: LP solves: the default concurrent method races primal, dual and barrier and
#: the winner depends on machine load.
GUROBI_OPTIONS_SCALED = {
    "ScaleFlag": 0,
    "FeasibilityTol": 1e-9,
    "OptimalityTol": 1e-9,
    "NumericFocus": 1,
}


def geometric_scaling(A, iters=20, stat_cut=1e-11):
    """Alternating geometric-mean row/column scaling of a sparse matrix.

    Returns ``(r, s)`` such that every row and column of ``diag(r) @ A @
    diag(s)`` is centred at 1 (``max * min == 1`` over its nonzeros). The
    factors are rounded to powers of two, so applying them is exact.

    Entries below ``stat_cut`` are ignored when computing the factors (they
    are still scaled): ecoinvent contains coefficients down to 1e-45 that are
    numerically irrelevant -- dropping everything below 1e-9 does not move the
    optimum -- and letting them drive the factors produces 1e22 row scales.
    """
    A = sps.csr_matrix(A, dtype=float)
    m, n = A.shape
    absA = abs(A)
    absA.data[absA.data < stat_cut] = 0.0
    absA.eliminate_zeros()
    r = np.ones(m)
    s = np.ones(n)
    if absA.nnz == 0:
        return r, s
    for _ in range(iters):
        B = (sps.diags(r) @ absA @ sps.diags(s)).tocoo()
        rmax = np.zeros(m)
        rmin = np.full(m, np.inf)
        np.maximum.at(rmax, B.row, B.data)
        np.minimum.at(rmin, B.row, B.data)
        ok = np.isfinite(rmin) & (rmax > 0)
        rf = np.ones(m)
        rf[ok] = 1.0 / np.sqrt(rmax[ok] * rmin[ok])
        r *= 2.0 ** np.round(np.log2(rf))
        B = (sps.diags(r) @ absA @ sps.diags(s)).tocoo()
        cmax = np.zeros(n)
        cmin = np.full(n, np.inf)
        np.maximum.at(cmax, B.col, B.data)
        np.minimum.at(cmin, B.col, B.data)
        ok = np.isfinite(cmin) & (cmax > 0)
        cf = np.ones(n)
        cf[ok] = 1.0 / np.sqrt(cmax[ok] * cmin[ok])
        s *= 2.0 ** np.round(np.log2(cf))
    return r, s


def _cap_bound(value, n_capped):
    if np.isfinite(value) and abs(value) >= SCALED_BOUND_CAP:
        n_capped[0] += 1
        return float('inf') if value > 0 else -float('inf')
    return value


def equilibrate_model_data(model_data, iters=20, stat_cut=1e-11, row_shift=1024.0):
    """Scale the ``combine_inputs`` / ``combine_inputs_time`` data in place.

    * ``TECH_MATRIX[i, j] *= r_i * s_j`` and ``FINAL_DEMAND[i] *= r_i``
    * ``ENV_COST_MATRIX[j, h] *= s_j`` and ``INV_MATRIX[g, j] *= s_j``
    * ``LOWER_LIMIT[j] /= s_j`` and ``UPPER_LIMIT[j] /= s_j`` (bounds beyond
      :data:`SCALED_BOUND_CAP` become infinite, with a warning)
    * ``LEFT_WEIGHTS / RIGHT_WEIGHTS[c, j] *= s_j``
    * time-indexed data: the same per product / process for every timestep,
      and the carry-over matrix ``K[i, i2] *= r_i / r_i2`` so that the
      carried term ``K[i, i2] * A[i2, j]`` ends up scaled by ``r_i``.

    ``row_shift`` multiplies all row factors (keep it a power of two): it
    lifts the smallest scaled coefficients further above Gurobi's 1e-13 input
    cutoff and tightens the solver's absolute row tolerance in original units.

    Stores and returns ``(row_scale, col_scale)``, dicts keyed by product and
    process id, also placed under ``model_data[None]['ROW_SCALE']`` /
    ``['COL_SCALE']`` for the instantiators to pick up.
    """
    d = model_data[None]
    tech = d['TECH_MATRIX']
    products = list(d['PRODUCT'][None])
    processes = list(d['PROCESS'][None])
    ridx = {i: k for k, i in enumerate(products)}
    cidx = {j: k for k, j in enumerate(processes)}
    time_indexed = 'TIME' in d

    keys = list(tech)
    rows = np.fromiter((ridx[i] for i, _ in keys), dtype=np.int64, count=len(keys))
    cols = np.fromiter((cidx[j] for _, j in keys), dtype=np.int64, count=len(keys))
    vals = np.fromiter((tech[k] for k in keys), dtype=float, count=len(keys))
    A = sps.csr_matrix((vals, (rows, cols)), shape=(len(products), len(processes)))

    r, s = geometric_scaling(A, iters=iters, stat_cut=stat_cut)
    r = r * float(row_shift)

    def prod_of(key):
        return key[1] if time_indexed else key

    def proc_of(key):
        return key[1] if time_indexed else key

    for k, v in zip(keys, vals * r[rows] * s[cols]):
        tech[k] = float(v)
    for key, v in d['FINAL_DEMAND'].items():
        d['FINAL_DEMAND'][key] = v * r[ridx[prod_of(key)]]
    for (j, h), v in d['ENV_COST_MATRIX'].items():
        d['ENV_COST_MATRIX'][(j, h)] = v * s[cidx[j]]
    for (g, j), v in d['INV_MATRIX'].items():
        d['INV_MATRIX'][(g, j)] = v * s[cidx[j]]
    n_capped = [0]
    for name in ('LOWER_LIMIT', 'UPPER_LIMIT'):
        for key, v in d[name].items():
            d[name][key] = _cap_bound(v / s[cidx[proc_of(key)]], n_capped)
    for name in ('LEFT_WEIGHTS', 'RIGHT_WEIGHTS'):
        for (c, j), v in d.get(name, {}).items():
            d[name][(c, j)] = v * s[cidx[j]]
    for (i, i2), v in d.get('K', {}).items():
        d['K'][(i, i2)] = v * r[ridx[i]] / r[ridx[i2]]

    if n_capped[0]:
        warnings.warn(
            f"{n_capped[0]} scaled process bounds exceeded {SCALED_BOUND_CAP:.0e} in "
            "magnitude and were treated as infinite. Finite default limits such as "
            "upper_bound=1e9 are meaningless on facility-scale processes; pass "
            "+-inf (the default) for limits that are not meant to bind.",
            UserWarning, stacklevel=3,
        )

    row_scale = {i: float(r[ridx[i]]) for i in products}
    col_scale = {j: float(s[cidx[j]]) for j in processes}
    d['ROW_SCALE'] = row_scale
    d['COL_SCALE'] = col_scale
    return row_scale, col_scale


def relax_default_bounds(lower_limit_dict, upper_limit_dict, explicit_lower, explicit_upper):
    """Set every process bound that is *not* an explicit limit to +-inf, in place.

    ``combine_inputs`` initialises all process bounds from ``default_limits``
    and overwrites the explicit ones (choice capacities, ``lower_limit`` /
    ``upper_limit``). A finite default such as ``upper_bound=1e9`` is not a
    modelling decision -- it never binds -- but divided by a column scale of
    1e-11 it becomes a finite 1e20 that Gurobi's simplex handles badly (row
    residuals of O(1) on the scaled model, or "infeasible or unbounded"), and
    capping only the largest ones does not help. Under scaling, unspecified
    bounds are therefore made truly infinite, as PULPO's own defaults are.

    ``explicit_lower`` / ``explicit_upper`` hold the dict keys whose bounds
    were set explicitly. Warns once when a finite default was replaced.
    """
    replaced = 0
    for key in lower_limit_dict:
        if key not in explicit_lower:
            replaced += np.isfinite(lower_limit_dict[key])
            lower_limit_dict[key] = -float('inf')
    for key in upper_limit_dict:
        if key not in explicit_upper:
            replaced += np.isfinite(upper_limit_dict[key])
            upper_limit_dict[key] = float('inf')
    if replaced:
        warnings.warn(
            f"scale=True: {int(replaced)} finite default process bounds "
            "(default_limits['lower_bound'] / ['upper_bound']) were replaced by +-inf. "
            "Bounds that are not meant to bind must be infinite on a scaled model; "
            "use lower_limit / upper_limit for limits that are.",
            UserWarning, stacklevel=3,
        )
    return replaced


def attach_scale(model, data):
    """Store the factors from ``data`` (a ``model_data[None]`` dict) on ``model``."""
    model._row_scale = dict(data.get('ROW_SCALE', {}))
    model._col_scale = dict(data.get('COL_SCALE', {}))
    model._solution_unscaled = False


def is_scaled(model):
    return bool(getattr(model, '_col_scale', None))


def col_factor(model, j):
    """Column factor ``s_j`` of process ``j`` (``x_j = s_j * y_j``); 1.0 if unscaled."""
    if not is_scaled(model):
        return 1.0
    return float(model._col_scale[j])


def to_scaled_coefficient(model, j, coefficient):
    """Coefficient of ``scaling_vector[j]`` for a term ``coefficient * x_j``.

    ``c * x_j == (c * s_j) * y_j``, so a row written in original units goes
    onto the model multiplied by the column factor. Identity when unscaled.
    """
    return float(coefficient) * col_factor(model, j)


def to_scaled_process_bound(model, j, bound):
    """Value to store in ``LOWER_LIMIT[j]`` / ``UPPER_LIMIT[j]`` for ``x_j <> bound``.

    ``x_j <= b`` is ``y_j <= b / s_j``. A finite scaled bound beyond
    :data:`SCALED_BOUND_CAP` becomes infinite, as ``equilibrate_model_data``
    does at build time, with the same warning. Identity when unscaled.
    """
    bound = float(bound)
    if not is_scaled(model) or not np.isfinite(bound):
        return bound
    n_capped = [0]
    value = _cap_bound(bound / col_factor(model, j), n_capped)
    if n_capped[0]:
        warnings.warn(
            f"a process bound written onto the scaled model exceeded {SCALED_BOUND_CAP:.0e} "
            "in magnitude after scaling and was treated as infinite.",
            UserWarning, stacklevel=3,
        )
    return value


def from_scaled_process_bound(model, j, value):
    """Read a ``LOWER_LIMIT[j]`` / ``UPPER_LIMIT[j]`` value back in original units."""
    return float(value) * col_factor(model, j)


def cut_row_factor(coefficients, stat_cut=1e-11):
    """Power-of-two row factor centring a constraint row at 1 (``max * min == 1``).

    Same per-row rule as :func:`geometric_scaling`, for a single row written
    after the build (a Kelley cut, for instance). Entries below ``stat_cut``
    in magnitude are ignored for the statistics; an empty row gives 1.0.
    Multiplying a whole constraint by a power of two is exact and leaves the
    LP unchanged, so this only affects the solver's conditioning.
    """
    mags = np.abs(np.asarray(coefficients, dtype=float))
    mags = mags[mags >= stat_cut]
    if mags.size == 0:
        return 1.0
    return float(2.0 ** np.round(-0.5 * np.log2(mags.max() * mags.min())))


def _apply(model, unscale):
    """Multiply (unscale) or divide (rescale) the solution by the factors."""
    if not is_scaled(model) or getattr(model, '_solution_unscaled', False) == unscale:
        return
    col_scale = model._col_scale
    row_scale = model._row_scale
    for idx in model.scaling_vector:
        j = idx[1] if isinstance(idx, tuple) else idx
        var = model.scaling_vector[idx]
        if var.value is not None:
            f = col_scale[j] if unscale else 1.0 / col_scale[j]
            # The Var keeps its scaled bounds; skip_validation silences W1002.
            var.set_value(var.value * f, skip_validation=True)
    if hasattr(model, 'slack'):
        for idx in model.slack:
            i = idx[1] if isinstance(idx, tuple) else idx
            var = model.slack[idx]
            if var.value is not None:
                f = 1.0 / row_scale[i] if unscale else row_scale[i]
                var.set_value(var.value * f, skip_validation=True)
    model._solution_unscaled = unscale


def unscale_solution(model):
    """Bring ``scaling_vector`` / ``slack`` to original units (no-op if already)."""
    _apply(model, unscale=True)


def rescale_solution(model):
    """Bring ``scaling_vector`` / ``slack`` back to scaled units before a solve."""
    _apply(model, unscale=False)
