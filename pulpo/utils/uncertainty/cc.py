"""Chance-constrained optimization in reduced space.

Why
---
With a joint reliability level ``lambda`` over ``K`` events (the impact row
and one per uncertain bound) and Boole's inequality, each event may fail with
probability ``eps_k = w_k (1 - lambda)`` (equal weights by default). Then ::

    impact row:  P(X <= z) >= lambda_z = 1 - eps_0
                 =>  mu' s + kappa sigma(s) <= z,   kappa = Phi^-1(lambda_z)   (X normal, A2)
    bound j:     P(s_j <= U_j) >= 1 - eps_k   =>  s_j <= F_Uj^-1(eps_k)        (exact quantile)

and minimizing ``z`` gives ``min mu' s + kappa sigma(s)`` subject to PULPO's
constraints with the uncertain bounds at their quantiles. In reduced space
(``pulpo.utils.reduced``) ``s = s0 + S v`` and ::

    sigma(s) = || R [1; v] ||,     R' R = G' G,     G = Q^(1/2) [s0, S]

with ``Q = diag(d) + B_u' diag(w) B_u`` from :mod:`moments`. The problem is a
second-order cone program with one column per alternative, solved directly by
Clarabel (open source, the default) or Gurobi. ``kappa >= 0`` (``lambda_z >=
1/2``) keeps it convex; lower levels are refused.

``allocation='individual'`` imposes every event at ``lambda`` on its own (a
comparison only: it controls no joint probability).
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats
import stats_arrays
from pyomo.opt import TerminationCondition

from pulpo.utils import optimizer
from pulpo.utils import reduced as _reduced
from pulpo.utils import scaling as _scaling
from pulpo.utils.uncertainty.moments import Moments
from pulpo.utils.uncertainty.preparer import UncertaintySpec, _validate

#: Below this many entries of ``Q^(1/2) [s0, S]`` the factor ``R`` comes from a
#: QR decomposition of that matrix; above, from the Gram matrix assembled with
#: ``2 (K + 1)`` solves, which never stores the rows of ``S``.
QR_ENTRIES = 2 ** 24

#: Clarabel options for the cone. The error in the optimum scales with the
#: tolerance (about 1e-6 relative at Clarabel's default of 1e-8 on an ecoinvent
#: problem); 1e-10 is the tightest setting that still converges reliably.
CLARABEL_OPTIONS = {'tol_gap_abs': 1e-10, 'tol_gap_rel': 1e-10, 'tol_feas': 1e-10}

#: Gurobi options for the cone. Its default BarQCPConvTol (1e-6) moves the
#: Pareto front by up to 1e-5 in the objective's units.
GUROBI_CONE_OPTIONS = {'BarQCPConvTol': 1e-9, 'FeasibilityTol': 1e-9, 'OptimalityTol': 1e-9}


# ---------------------------------------------------------------------------
# Risk budget and exact quantiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskBudget:
    """A total failure budget ``eps = 1 - lambda`` split across ``K`` events.

    Imposing each chance constraint at ``lambda`` individually controls no joint
    probability: with ``K`` rows each allowed to fail with probability ``eps``,
    the chance that at least one fails reaches ``K * eps``. Boole's inequality
    repairs this - allocate ``eps_k = w_k * eps`` with ``sum(w_k) = 1`` and the
    union of the failures is bounded by ``eps``, so every row holds *together*
    with probability at least ``lambda``. It needs only marginals, so no
    correlation between the events has to be estimated.

    ``weights[0]`` is the impact target; ``weights[1:]`` are the uncertain
    bounds in the order of :attr:`ChanceConstrained.events` (lower bounds
    first, then upper bounds, each by process index). The weights are
    fixed before the solve - choosing them after seeing a solution would make
    the budget a function of the decision it certifies.
    """

    lambda_level: float
    K: int
    weights: Tuple[float, ...]

    @property
    def epsilon(self) -> float:
        """The total failure budget being divided."""
        return 1.0 - self.lambda_level

    @property
    def lambda_impact(self) -> float:
        """Level for the impact target: ``1 - w_0 * eps``, replacing ``lambda``."""
        return 1.0 - self.weights[0] * self.epsilon

    def epsilon_at(self, position: int) -> float:
        """The share of the budget allocated to the event at ``position``."""
        return self.weights[position] * self.epsilon


def bonferroni_budget(lambda_level: float, K: int, weights=None) -> RiskBudget:
    """Split ``1 - lambda_level`` across ``K`` events, equally unless told otherwise.

    An equal split is the default because it is neutral: it needs no
    justification and cannot be read as tuned to produce a result. Unequal
    weights are valid - Boole's inequality only requires them to sum to one -
    but each must be positive: a weight of zero demands its event with
    certainty, which no distribution with unbounded support can meet.
    """
    if not 0.0 <= lambda_level < 1.0:
        raise ValueError(f"lambda_level must lie in [0, 1); got {lambda_level!r}.")
    if K < 1:
        raise ValueError(f"K must be at least 1; got {K!r}.")
    if weights is None:
        weights = (1.0 / K,) * K
    weights = tuple(float(w) for w in weights)
    if len(weights) != K:
        raise ValueError(f"weights has length {len(weights)} but K is {K}; one weight per "
                         "event, the first for the impact target.")
    if any(not w > 0.0 for w in weights):
        raise ValueError(f"weights must be positive; got {weights!r}.")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError(f"weights must sum to 1 for Boole's inequality to bound the joint "
                         f"failure probability by {1.0 - lambda_level:.4g}; got {sum(weights):.6g}.")
    return RiskBudget(lambda_level=float(lambda_level), K=int(K), weights=weights)


def declared_quantile(spec: UncertaintySpec, probability: float) -> float:
    """The exact ``probability``-quantile of a parameter's declared family.

    A right-hand-side-only chance constraint ``P(s <= xi) >= 1 - eps`` is
    ``s <= F^-1(eps)`` with ``F`` the declared distribution, so it needs no
    Gaussian representation. For a triangular distribution the inverse CDF is
    elementary and never leaves the support, which a moment-matched normal does
    at high reliability.
    """
    p = float(probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must lie in [0, 1]; got {p!r}.")
    utype = int(spec['uncertainty_type'])
    if utype in (stats_arrays.UndefinedUncertainty.id, stats_arrays.NoUncertainty.id):
        return float(spec['amount'])
    _validate(spec)
    if utype == stats_arrays.NormalUncertainty.id:
        return float(spec['loc'] + spec['scale'] * scipy.stats.norm.ppf(p))
    if utype == stats_arrays.UniformUncertainty.id:
        a, b = float(spec['minimum']), float(spec['maximum'])
        return a + p * (b - a)
    if utype == stats_arrays.TriangularUncertainty.id:
        a, c, b = float(spec['minimum']), float(spec['loc']), float(spec['maximum'])
        if b <= a:
            return a
        # The branch follows from where the mode sits: a large share of the
        # budget at a low level can cross it.
        if p <= (c - a) / (b - a):
            return a + np.sqrt(p * (b - a) * (c - a))
        return b - np.sqrt((1.0 - p) * (b - a) * (b - c))
    mu, sigma = float(spec['loc']), float(spec['scale'])
    if spec.get('negative', False):
        # A mirrored lognormal is decreasing in its own quantile.
        return -float(np.exp(mu + sigma * scipy.stats.norm.ppf(1.0 - p)))
    return float(np.exp(mu + sigma * scipy.stats.norm.ppf(p)))


# ---------------------------------------------------------------------------
# The cone
# ---------------------------------------------------------------------------

def solve_socp(lp, R, kappa, solver_name=None, options=None):
    """``min c'x + c0 + kappa ||R [1; x_v]||`` over the rows and bounds of ``lp``.

    ``x_v`` are the first ``R.shape[1] - 1`` columns of ``lp``. Rows, columns
    and the objective are equilibrated as in :func:`reduced.solve_lp`. Returns
    a :class:`reduced.LPSolution`; ``objective`` includes the cone term.
    """
    name = (solver_name or 'clarabel').lower()
    if name not in ('clarabel', 'gurobi'):
        raise ValueError(f"The chance-constrained problem solves with 'clarabel' or 'gurobi', not {solver_name!r}.")
    if kappa < 0:
        raise ValueError(f"kappa = {kappa:.4g} < 0 (impact level below 1/2) makes the problem non-convex.")
    options = dict(options or {})
    tee = bool(options.pop('tee', False))
    n, kv = len(lp.c), R.shape[1] - 1
    R0, R1 = R[:, 0], np.hstack([R[:, 1:], np.zeros((R.shape[0], n - kv))])
    stats = sp.vstack([lp.matrix, sp.csr_matrix(lp.c.reshape(1, -1)), sp.csr_matrix(R1)], format='csr')
    r, d = _scaling.ruiz_scaling(stats)
    r = r[:lp.matrix.shape[0]]
    M = (sp.diags(r) @ lp.matrix @ sp.diags(d)).tocsr()
    c = lp.c * d
    R1 = R1 * d
    cmax = max(np.abs(c).max() if c.size else 0.0, kappa)
    f = 2.0 ** -np.round(np.log2(cmax)) if cmax > 0 else 1.0
    with np.errstate(invalid='ignore'):
        lo, up = lp.row_lower * r, lp.row_upper * r
        clo, cup = lp.col_lower / d, lp.col_upper / d

    start = time.perf_counter()
    if name == 'clarabel':
        status, x = _solve_clarabel(c * f, kappa * f, M, lo, up, clo, cup, R0, R1, options, tee)
    else:
        status, x = _solve_gurobi_cone(c * f, kappa * f, M, lo, up, clo, cup, R0, R1, options, tee)
    seconds = time.perf_counter() - start
    if x is None:
        return _reduced.LPSolution(status, seconds=seconds)
    x = x * d
    sigma = float(np.linalg.norm(R0 + R[:, 1:] @ x[:kv]))
    return _reduced.LPSolution(status, x, float(lp.c @ x + lp.c0 + kappa * sigma), seconds)


def _linear_rows(M, lo, up, clo, cup):
    """Split ``lo <= M x <= up`` and the column bounds into equalities and ``<=`` rows."""
    n = M.shape[1]
    eq = np.flatnonzero(lo == up)
    rows_le, rhs_le = [M[np.flatnonzero((lo != up) & np.isfinite(up))]], [up[(lo != up) & np.isfinite(up)]]
    rows_le.append(-M[np.flatnonzero((lo != up) & np.isfinite(lo))])
    rhs_le.append(-lo[(lo != up) & np.isfinite(lo)])
    ident = sp.identity(n, format='csr')
    rows_le.append(ident[np.flatnonzero(np.isfinite(cup))])
    rhs_le.append(cup[np.isfinite(cup)])
    rows_le.append(-ident[np.flatnonzero(np.isfinite(clo))])
    rhs_le.append(-clo[np.isfinite(clo)])
    return M[eq], lo[eq], sp.vstack(rows_le, format='csr'), np.concatenate(rhs_le)


def _solve_clarabel(c, kappa, M, lo, up, clo, cup, R0, R1, options, tee):
    import clarabel
    n = M.shape[1]
    A_eq, b_eq, A_le, b_le = _linear_rows(M, lo, up, clo, cup)
    # Variables (x, t). Cones: x rows = b_eq; b_le - A_le x >= 0; (t, R0 + R1 x) in SOC.
    zero_col = lambda m: sp.csr_matrix((m.shape[0], 1))
    soc = sp.vstack([sp.hstack([sp.csr_matrix((1, n)), sp.csr_matrix([[-1.0]])]),
                     sp.hstack([sp.csr_matrix(-R1), zero_col(R1)])], format='csc')
    A = sp.vstack([sp.hstack([A_eq, zero_col(A_eq)]), sp.hstack([A_le, zero_col(A_le)]), soc], format='csc')
    b = np.concatenate([b_eq, b_le, [0.0], R0])
    cones = [clarabel.ZeroConeT(A_eq.shape[0]), clarabel.NonnegativeConeT(A_le.shape[0]),
             clarabel.SecondOrderConeT(1 + len(R0))]
    settings = clarabel.DefaultSettings()
    settings.verbose = tee
    for key, value in {**CLARABEL_OPTIONS, **options}.items():
        setattr(settings, key, value)
    solution = clarabel.DefaultSolver(sp.csc_matrix((n + 1, n + 1)), np.append(c, kappa), A, b,
                                      cones, settings).solve()
    status = str(solution.status)
    condition = {'Solved': TerminationCondition.optimal,
                 'PrimalInfeasible': TerminationCondition.infeasible,
                 'DualInfeasible': TerminationCondition.unbounded,
                 'MaxIterations': TerminationCondition.maxIterations,
                 'MaxTime': TerminationCondition.maxTimeLimit}.get(status, TerminationCondition.other)
    if condition != TerminationCondition.optimal:
        return condition, None
    return condition, np.asarray(solution.x[:n], dtype=float)


def _solve_gurobi_cone(c, kappa, M, lo, up, clo, cup, R0, R1, options, tee):
    import gurobipy as gp
    from gurobipy import GRB
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', int(tee))
    env.start()
    try:
        model = gp.Model(env=env)
        n = M.shape[1]
        x = model.addMVar(n, lb=np.where(np.isfinite(clo), clo, -GRB.INFINITY),
                          ub=np.where(np.isfinite(cup), cup, GRB.INFINITY))
        t = model.addVar(lb=0.0)
        y = model.addMVar(len(R0), lb=-GRB.INFINITY)
        eq = np.flatnonzero(lo == up)
        le = np.flatnonzero((lo != up) & np.isfinite(up))
        ge = np.flatnonzero((lo != up) & np.isfinite(lo))
        if len(eq):
            model.addMConstr(M[eq], x, GRB.EQUAL, lo[eq])
        if len(le):
            model.addMConstr(M[le], x, GRB.LESS_EQUAL, up[le])
        if len(ge):
            model.addMConstr(M[ge], x, GRB.GREATER_EQUAL, lo[ge])
        model.addConstr(y == R0 + sp.csr_matrix(R1) @ x)
        model.addConstr(y @ y <= t * t)
        model.setObjective(c @ x + kappa * t, GRB.MINIMIZE)
        for key, value in {**GUROBI_CONE_OPTIONS, **options}.items():
            model.setParam(key, value)
        model.optimize()
        condition = {GRB.OPTIMAL: TerminationCondition.optimal,
                     GRB.INFEASIBLE: TerminationCondition.infeasible,
                     GRB.UNBOUNDED: TerminationCondition.unbounded,
                     GRB.INF_OR_UNBD: TerminationCondition.infeasibleOrUnbounded,
                     GRB.TIME_LIMIT: TerminationCondition.maxTimeLimit}.get(model.Status, TerminationCondition.other)
        if condition != TerminationCondition.optimal:
            return condition, None
        return condition, np.asarray(x.X, dtype=float)
    finally:
        env.dispose()


# ---------------------------------------------------------------------------
# The chance-constrained problem
# ---------------------------------------------------------------------------

@dataclass
class Point:
    """One solved reliability level.

    ``adjusted`` is the chance-constrained impact ``z = mean + kappa * sigma``;
    ``mean`` and ``sigma`` are evaluated exactly at ``s`` (not read off the
    cone). ``bounds`` holds the bound imposed on each uncertain process and
    ``epsilon`` the failure probability allocated to it, both keyed by the event
    ``(kind, process)``, e.g. ``('upper', 4)``; ``size`` is the
    number of processes and the columns and rows of the reduced problem.
    """
    lambda_level: float
    lambda_impact: float
    kappa: float
    mean: float
    sigma: float
    adjusted: float
    s: np.ndarray
    v: np.ndarray
    bounds: Dict[Tuple[str, int], float]
    epsilon: Dict[Tuple[str, int], float]
    seconds: dict = field(default_factory=dict)
    rounds: int = 1
    balance_residual: float = None
    size: dict = field(default_factory=dict)


@dataclass
class Projections:
    """The impact's moments on the reduced space ``s = s0 + S v``.

    With these and :meth:`reduced.ReducedModel.linear_program` a formulation
    over ``v`` needs no further solves::

        E[X]  = m0 + m' v
        Var X = sum_{j in J} d_j (s0_J + S_J v)_j^2 + sum_e w_e (B_unc_s0 + B_unc_S v)_e^2

    ``J`` are the processes that carry variance (``d_j > 0``), ``S_J`` their
    rows of ``S``, and ``B_unc_S = E[B_u] S`` the mean flows of the uncertain
    CFs per unit of ``v``. A sampled impact of the base and of one unit of each
    alternative follows from the same arrays (see :mod:`validation`).
    """
    s0: np.ndarray
    m0: float
    m: np.ndarray
    J: np.ndarray
    S_J: np.ndarray
    B_unc_s0: np.ndarray
    B_unc_S: np.ndarray


class Front(dict):
    """``{lambda: Point}`` with a tabular view."""

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{'lambda': p.lambda_level, 'lambda_impact': p.lambda_impact,
                              'kappa': p.kappa, 'mean': p.mean, 'sigma': p.sigma,
                              'adjusted': p.adjusted, 'seconds': p.seconds.get('total')}
                             for p in self.values()]).set_index('lambda')


class ChanceConstrainedError(optimizer.SolveError):
    """A reliability level did not solve to optimality; ``results`` holds its
    :class:`reduced.ReducedResults`."""


class ChanceConstrained:
    """``min mu' s + kappa sigma(s)`` at a reliability level, in reduced space.

    Args:
        model: a :class:`reduced.ReducedModel`, or an instantiated worker (whose
            reduced model is built). Every deterministic constraint of the
            instance applies; its objective is replaced by the chance-constrained
            impact of ``moments.method``. A limit on that impact itself applies to
            its mean ``mu' s`` (it is not chance-constrained).
        moments (Moments): from :func:`moments.compute_moments`.
        upper_bounds, lower_bounds (dict): uncertain process bounds,
            ``{process: spec}`` with ``process`` an activity, a key or a process
            index and ``spec`` a declared distribution (normal, lognormal,
            uniform, triangular). They replace the instance's bounds on those
            processes. Each one is an event of the risk budget.
        allocation (str): ``'bonferroni'`` (joint, the default) or ``'individual'``.
        weights (sequence, optional): Bonferroni weights, the first for the
            impact row, then one per event in the order of :attr:`events`.
    """

    def __init__(self, model, moments: Moments, upper_bounds=None, lower_bounds=None,
                 allocation='bonferroni', weights=None):
        self.worker = None if isinstance(model, _reduced.ReducedModel) else model
        self.model = None
        self.moments = moments
        if allocation not in ('bonferroni', 'individual'):
            raise ValueError(f"allocation must be 'bonferroni' or 'individual'; got {allocation!r}.")
        self.allocation = allocation
        self._model(model if self.worker is None else None)
        events = []
        for kind, specs in (('lower', lower_bounds or {}), ('upper', upper_bounds or {})):
            for process, spec in specs.items():
                spec = dict(spec)
                _validate(spec, process)
                events.append(((kind, self._process_index(process)), spec))
        # Lower bounds first, then upper bounds, each by process index: the
        # order in which weights[1:] are read, the same as PULPO 1.8.0's.
        events.sort(key=lambda e: (e[0][0] != 'lower', e[0][1]))
        if len({e[0] for e in events}) != len(events):
            raise ValueError("A process carries the same uncertain bound twice.")
        #: The events after the impact row, in budget order: ``(kind, process)``.
        self.events = [e[0] for e in events]
        self._specs = dict(events)
        if weights is not None and allocation != 'bonferroni':
            raise ValueError("weights apply to the Bonferroni allocation only.")
        self.weights = None if weights is None else tuple(weights)
        bonferroni_budget(0.5, self.K, self.weights)        # validates the weights

    def _model(self, model=None):
        """The reduced model of the worker's current instance (rebuilt after a
        re-instantiation), checked for what the problem supports."""
        model = model or _reduced.build(self.worker)
        if model is not self.model:
            if self.model is None or model.system is not self.model.system:
                self._C_SS = None
                self._qr = None
            if self.moments.mu.shape[0] != model.n:
                raise ValueError("The moments and the model have different numbers of processes.")
            if len(model.instance.GOAL_INDICATOR):
                raise NotImplementedError("The chance-constrained objective replaces the goal objective; "
                                          "instantiate with objective='weighted_sum'.")
            self.model = model
        return model

    def _process_index(self, process):
        if isinstance(process, (int, np.integer)):
            return int(process)
        key = getattr(process, 'key', process)
        return int(self.model.lci_data['process_map'][key])

    @property
    def K(self):
        """Number of events: the impact row and every uncertain bound."""
        return 1 + len(self.events)

    @property
    def bound_specs(self):
        """``{(kind, process): spec}``: the declared distribution of each uncertain bound."""
        return {event: dict(spec) for event, spec in self._specs.items()}

    # -- levels and bounds ---------------------------------------------------

    def levels(self, lambda_level):
        """``(lambda_impact, {event: epsilon})`` at a joint level."""
        if not 0.0 < lambda_level < 1.0:
            raise ValueError(f"lambda must lie in (0, 1); got {lambda_level!r}.")
        if self.allocation == 'individual':
            return lambda_level, {event: 1.0 - lambda_level for event in self.events}
        budget = bonferroni_budget(lambda_level, self.K, self.weights)
        return budget.lambda_impact, {event: budget.epsilon_at(k + 1) for k, event in enumerate(self.events)}

    def bounds(self, lambda_level):
        """``{event: bound}``: the exact quantile imposed on each uncertain bound."""
        _, eps = self.levels(lambda_level)
        out = {}
        for (kind, j), e in eps.items():
            value = declared_quantile(self._specs[(kind, j)], e if kind == 'upper' else 1.0 - e)
            if not np.isfinite(value):
                raise ValueError(f"The {kind} bound on process {j} cannot hold with probability "
                                 f"{1 - e:.6g}: its quantile is {value}.")
            out[(kind, j)] = value
        return out

    # -- variance factor -----------------------------------------------------

    def projections(self) -> Projections:
        """The moments of the impact on the current instance's reduced space."""
        model = self._model() if self.worker is not None else self.model
        mom, system = self.moments, model.system
        s0 = system.base(model.demand()[0])
        J = np.flatnonzero(mom.d > 0)
        B_unc = mom.B_unc
        return Projections(s0=s0, m0=float(mom.mu @ s0), m=system.project_vectors([mom.mu])[0],
                           J=J, S_J=system.rows(J), B_unc_s0=np.asarray(B_unc @ s0).ravel(),
                           B_unc_S=system.project_vectors([B_unc[k] for k in range(B_unc.shape[0])]))

    def _factor(self, f_tilde):
        """``R`` with ``R' R = G' G`` for the current demand (``s0`` depends on it)."""
        mom, system = self.moments, self.model.system
        s0 = system.base(f_tilde)
        J = np.flatnonzero(mom.d > 0)
        K = system.n_free
        if (len(J) + len(mom.cf_rows)) * (K + 1) <= QR_ENTRIES:
            if self._qr is None:
                B_unc = mom.B_unc
                self._qr = (J, system.rows(J), B_unc,
                            system.project_vectors([B_unc[k] for k in range(B_unc.shape[0])]))
            J, S_J, B_unc, BS = self._qr
            sd, sw = np.sqrt(mom.d[J]), np.sqrt(mom.w)
            G = np.vstack([np.column_stack([sd * s0[J], sd[:, None] * S_J]),
                           np.column_stack([sw * (B_unc @ s0), sw[:, None] * BS])])
            if G.shape[0] == 0:
                return np.zeros((1, K + 1))
            return np.linalg.qr(G, mode='r')
        # Gram matrix C = [s0, S]' Q [s0, S] without storing rows of S:
        # C_SS from K forward and K adjoint solves (once), C_S0 from one adjoint solve.
        def Q(x):
            return mom.d[:, None] * x + mom.B_unc.T @ (mom.w[:, None] * (mom.B_unc @ x))
        if self._C_SS is None:
            C = np.zeros((K, K))
            chunk = max(1, _reduced.CHUNK_ENTRIES // max(system.n, 1))
            for start in range(0, K, chunk):
                cols = system.columns[start:start + chunk]
                rhs = np.zeros((system.n, len(cols)))
                rhs[cols, np.arange(len(cols))] = 1.0
                X = system.factorization.solve(rhs)
                Z = system.factorization.solve(Q(X), transpose=True)
                C[:, start:start + len(cols)] = Z[system.columns, :]
                system.solves += 2 * len(cols)
            self._C_SS = (C + C.T) / 2.0
        q0 = Q(s0[:, None])[:, 0]
        C = np.empty((K + 1, K + 1))
        C[0, 0] = float(s0 @ q0)
        C[1:, 0] = C[0, 1:] = system.project_vectors([q0])[0]
        C[1:, 1:] = self._C_SS
        vals, vecs = np.linalg.eigh(C)
        return np.sqrt(np.clip(vals, 0.0, None))[:, None] * vecs.T

    # -- solve -----------------------------------------------------------------

    def solve_point(self, lambda_level, solver_name=None, options=None) -> Point:
        """Solve one reliability level; raises :class:`ChanceConstrainedError` unless optimal."""
        start = time.perf_counter()
        lambda_impact, eps = self.levels(lambda_level)
        kappa = float(scipy.stats.norm.ppf(lambda_impact))
        if kappa < 0:
            raise ValueError(f"The impact row is imposed at {lambda_impact:.4g} < 1/2, which makes the "
                             "problem non-convex; raise lambda.")
        model, mom = (self._model() if self.worker is not None else self.model), self.moments
        lower, upper = model.process_bounds()
        imposed = self.bounds(lambda_level)
        for (kind, j), value in imposed.items():
            (upper if kind == 'upper' else lower)[j] = value
        K = model.system.n_free
        m = model.system.project_vectors([mom.mu])[0]
        cache = {}

        def factor(f_tilde):
            key = f_tilde.tobytes()
            if key not in cache:
                cache[key] = (self._factor(f_tilde), float(mom.mu @ model.system.base(f_tilde)))
            return cache[key]

        def transform(lp, f_tilde):
            m0 = factor(f_tilde)[1]
            lp.c = np.concatenate([m, np.zeros(len(lp.c) - K)])
            lp.c0 = m0
            if ('impact', mom.method) in lp.row_labels:
                # A limit on the uncertain impact holds for its mean.
                i = lp.row_labels.index(('impact', mom.method))
                lb, ub = _reduced._var_bounds(model.instance.impacts[mom.method])
                matrix = lp.matrix.tolil()
                matrix[i, :] = np.concatenate([m, np.zeros(len(lp.c) - K)])
                lp.matrix = matrix.tocsr()
                lp.row_lower[i], lp.row_upper[i] = lb - m0, ub - m0

        def solve(lp, f_tilde):
            return solve_socp(lp, factor(f_tilde)[0], kappa, solver_name=solver_name, options=options)

        results, s = model.optimize(solve, bounds=(lower, upper), transform=transform)
        if results.termination_condition != TerminationCondition.optimal:
            raise ChanceConstrainedError(
                f"lambda = {lambda_level}: the chance-constrained problem did not solve to optimality "
                f"({results.termination_condition}).", results)
        mean, sigma = mom.mean(s), mom.std(s)
        results.seconds['total'] = time.perf_counter() - start
        return Point(lambda_level=float(lambda_level), lambda_impact=float(lambda_impact), kappa=kappa,
                     mean=mean, sigma=sigma, adjusted=mean + kappa * sigma, s=s, v=results.v,
                     bounds=imposed, epsilon=dict(eps),
                     seconds=results.seconds, rounds=results.rounds,
                     balance_residual=results.balance_residual,
                     size={'processes': model.n, 'variables': results.n_variables, 'rows': results.n_rows})

    def solve(self, lambdas, solver_name=None, options=None) -> Front:
        """Solve every level of ``lambdas`` (one factorization serves them all)."""
        lambdas = [lambdas] if np.isscalar(lambdas) else list(lambdas)
        return Front((float(lam), self.solve_point(lam, solver_name, options)) for lam in lambdas)

    def write(self, point: Point):
        """Write a solved point onto the instance, as a solve would leave it, so
        ``extract_results()`` and the rest read it. The impacts there are the
        instance's deterministic ones; the point carries the chance-constrained ones."""
        model = self._model() if self.worker is not None else self.model
        model.write_solution(point.s, point.v)
        inst, lci = model.instance, model.lci_data
        methods = getattr(self.worker, 'method', None)
        if isinstance(methods, dict) and len(methods) > 1 and 0 in methods.values():
            optimizer.calculate_methods(inst, lci, methods)
        optimizer.calculate_inv_flows(inst, lci)


def apply_CC_formulation(model_instance, risk_budget: RiskBudget, upper_bounds=None, lower_bounds=None):
    """Write the exact-quantile bounds of a risk budget onto a Pyomo instance.

    For a full-space solve of a problem whose only uncertain rows are process
    bounds. ``upper_bounds`` / ``lower_bounds`` map process indices to declared
    distributions and take the budget's positions ``1, 2, ...``: lower bounds
    first, then upper bounds, each by process index, as in
    :class:`ChanceConstrained`. Values are
    stored in the instance's units (see :mod:`pulpo.utils.scaling`).
    """
    events = sorted([(j, 'upper', spec) for j, spec in (upper_bounds or {}).items()]
                    + [(j, 'lower', spec) for j, spec in (lower_bounds or {}).items()],
                    key=lambda e: (e[1] != 'lower', e[0]))
    if len(events) + 1 != risk_budget.K:
        raise ValueError(f"the risk budget covers K={risk_budget.K} events but {len(events)} bound(s) plus "
                         f"the impact target are {len(events) + 1}.")
    for position, (j, kind, spec) in enumerate(events, start=1):
        eps = risk_budget.epsilon_at(position)
        value = declared_quantile(spec, eps if kind == 'upper' else 1.0 - eps)
        param = model_instance.UPPER_LIMIT if kind == 'upper' else model_instance.LOWER_LIMIT
        param[j] = _scaling.to_scaled_process_bound(model_instance, j, value)
