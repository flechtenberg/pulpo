"""Reduced-space solution of PULPO's static LP: ``solve(method='reduced')``.

Why
---
PULPO's LP has one variable per process and one balance row per product. The
choices free only a few directions: one per alternative, minus one per
category (16 on a 23,569-process ecoinvent model). With the technosphere
matrix ``A`` square and invertible, every scaling vector that satisfies the
merged balances is

    s = s0 + S v,        s0 = A^-1 f~,        S = A^-1 E

where ``v_k`` is the net output on alternative ``k``'s own product row (plus
the slack of every supply product), ``E`` places ``v`` on those rows, and
``f~`` is the demand on all other rows. Conversely every ``v`` gives a
balanced ``s``. The LP over ``s`` and the LP over ``v`` are therefore the same
problem: the reduction is exact, and it holds for every static PULPO
constraint because each of them is linear in ``s``.

The reduced LP has one column per alternative and one row per constraint
that is not a balance, so it is small and dense. It is solved directly; the
full scaling vector is recovered with one forward solve, and all balances
hold to the precision of the factorization rather than to the LP solver's
feasibility tolerance.

What
----
Every constraint of the instance becomes one row in ``v``:

    category c            sum_{k in c} v_k = f_c
    bound on process j    l_j - s0_j  <=  S[j, :] v  <=  u_j - s0_j
    impact h              S' c_h  (objective, impact limits, goals)
    flow limit g          S' B_g'
    dependent constraint  S' (L_d - R_d)  <=  -(L_d - R_d)' s0

``S`` is never formed: on a large database it is a dense n x K matrix (2.6 GB
for 216,467 processes and 1,500 alternatives). A row ``m' S`` needs only the
adjoint solve ``A' x = m``, after which ``(m' S)_k = x[p_k]``; a block of
bound rows ``S[J, :]`` can instead be read off K forward solves, keeping only
the rows ``J``. :meth:`ReducedSystem.project` picks whichever needs fewer
solves. One factorization of ``A`` serves both directions and every solve
after it: across objective weights, limits, demand and repeated solves.
Every finite process bound is a row ``S[j, :]``, so the LP stays small only
while few processes are bounded, which is PULPO's default (bounds are +-inf
unless set). Finite ``default_limits`` on every process put the whole of
``S`` into the LP (see ``PulpoOptimizer.instantiate``).

Reading a solved instance: unchanged. :meth:`ReducedModel.solve` writes the
recovered scaling vector, the impacts, the flows, the supply slacks and the
transgressions onto the Pyomo instance in original units, exactly where a
full solve leaves them, so ``extract_results`` / ``summarize_results`` and
the post-processing in ``PulpoOptimizer.solve`` work as before.

Requirements, checked when the model is built: ``A`` square and non-singular;
a static instance (the time-dependent model raises ``NotImplementedError``);
no constraints, variables or objectives beyond those ``optimizer.instantiate``
creates. Changes made on the instance itself are honoured: limit Params,
variable bounds, fixed variables, deactivated optional rows.
"""

import hashlib
import time
import warnings
import weakref
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from . import scaling as _scaling

#: Dense right-hand-side blocks are solved in chunks of at most this many
#: entries (n x chunk), which bounds the memory of a projection to a few
#: hundred MB on any database size.
CHUNK_ENTRIES = 2 ** 24

#: At most this many iterative-refinement steps follow a solve, until the
#: componentwise backward error is below REFINEMENT_TOL; see Factorization.solve.
REFINEMENT_STEPS = 2
REFINEMENT_TOL = 1e-14

#: Balance residual of a recovered scaling vector (relative to the size of
#: each balance's terms) above which the solve is reported as failed instead
#: of written onto the instance.
RESIDUAL_LIMIT = 1e-8

#: HiGHS options for the reduced LP. The LP is equilibrated before it is
#: handed over, so the tolerances apply to O(1) rows; tighter tolerances than
#: the defaults cost nothing on a problem this small. ``small_matrix_value``
#: is HiGHS's minimum: the default of 1e-9 would drop the small cross terms of
#: dense rows.
HIGHS_OPTIONS = {
    'primal_feasibility_tolerance': 1e-9,
    'dual_feasibility_tolerance': 1e-9,
    'small_matrix_value': 1e-12,
}

#: Gurobi options for the reduced LP; same reasoning as :data:`HIGHS_OPTIONS`.
GUROBI_OPTIONS = {
    'FeasibilityTol': 1e-9,
    'OptimalityTol': 1e-9,
}

#: The components ``optimizer.instantiate`` creates and the reduced backend
#: understands. ``impacts_calculated`` and ``inv_flows`` are written by the
#: post-processing of a solve.
_KNOWN_VARS = {'impacts', 'scaling_vector', 'inv_vector', 'slack', 'transgression',
               'impacts_calculated', 'inv_flows'}
_KNOWN_CONSTRAINTS = {'FINAL_DEMAND_CNSTR', 'IMPACTS_CNSTR', 'INVENTORY_CNSTR',
                      'DEPENDENT_CNSTR', 'TRANSGRESSION_CNSTR'}


# ---------------------------------------------------------------------------
# Factorization
# ---------------------------------------------------------------------------

class Factorization:
    """One sparse LU of the technosphere matrix, for ``A x = b`` and ``A' x = b``.

    PARDISO (``pypardiso``, already a PULPO dependency) factorizes an
    ecoinvent-sized matrix in about a second; SciPy's SuperLU, the fallback
    where MKL is unavailable, takes tens of seconds with the
    ``MMD_AT_PLUS_A`` ordering and minutes with its default.

    ``pypardiso`` keeps a single solver instance per process, which bw2calc
    uses as well. A factorization that was replaced in between (by an LCA
    calculation, say) is detected before each solve and redone. The factors
    themselves are native objects and are not pickled: a copied or unpickled
    factorization refactorizes on its first solve.

    A singular ``A`` is refused: the reduction parametrizes the balanced
    scaling vectors only when ``A`` is invertible. SuperLU stops at an exact
    zero pivot; PARDISO instead perturbs pivots below ``1e-13 * |A|`` and
    returns the solution of a nearby matrix, so any perturbed pivot
    (``iparm(14)``) counts as singular.

    Args:
        A: square sparse matrix.
        backend (str): ``'auto'`` (PARDISO if available, else SciPy),
            ``'pardiso'`` or ``'scipy'``.
    """

    def __init__(self, A, backend='auto'):
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"The technosphere matrix must be square for the reduced "
                             f"formulation; got shape {A.shape}.")
        if backend not in ('auto', 'pardiso', 'scipy'):
            raise ValueError(f"Unknown factorization backend {backend!r}; "
                             "use 'auto', 'pardiso' or 'scipy'.")
        self.n = A.shape[0]
        self.refactorizations = 0
        self._A = sp.csr_matrix(A, dtype=np.float64, copy=True)
        self._A.sort_indices()
        self._abs_A = abs(self._A)
        self._solver = None
        self._lu = None
        self.backend = _resolve_backend(backend)
        start = time.perf_counter()
        self._factorize()
        self.seconds = time.perf_counter() - start

    def _factorize(self):
        if self.backend == 'pardiso':
            from pypardiso.scipy_aliases import pypardiso_solver
            self._solver = pypardiso_solver
            self._solver.factorize(self._A)
            perturbed = int(self._solver.get_iparm(14))
            if perturbed:
                raise ValueError(f"The technosphere matrix is singular or numerically singular "
                                 f"(PARDISO perturbed {perturbed} pivots); the reduced formulation "
                                 "needs an invertible A. Solve with method='full'.")
        else:
            try:
                self._lu = spla.splu(self._A.tocsc(), permc_spec='MMD_AT_PLUS_A')
            except RuntimeError as exc:
                raise ValueError("The technosphere matrix is singular; the reduced formulation "
                                 "needs an invertible A. Solve with method='full'.") from exc

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_solver'] = None
        state['_lu'] = None
        return state

    def solve(self, b, transpose=False):
        """Solve ``A x = b`` (or ``A' x = b``) for a vector or a dense block ``b``.

        A plain solve can leave a componentwise backward error far above
        roundoff on an ecoinvent technosphere (entries 1e-13 .. 1e11): up to
        1e-5 with SuperLU, 4e-8 with PARDISO on a 216,467-process system.
        Iterative refinement with the same factors brings it to roundoff; it
        runs only while :func:`backward_error` exceeds ``REFINEMENT_TOL``.
        """
        b = np.asarray(b, dtype=np.float64)
        A = self._A.T if transpose else self._A
        abs_A = self._abs_A.T if transpose else self._abs_A
        x = self._raw_solve(b, transpose)
        for _ in range(REFINEMENT_STEPS):
            r = b - A @ x
            if backward_error(r, abs_A @ np.abs(x) + np.abs(b)) <= REFINEMENT_TOL:
                break
            x = x + self._raw_solve(r, transpose)
        return x

    def _raw_solve(self, b, transpose):
        if self.backend == 'scipy':
            if self._lu is None:
                self.refactorizations += 1
                self._factorize()
            return self._lu.solve(b, trans='T' if transpose else 'N')
        if self._solver is None or not self._solver._is_already_factorized(self._A):
            self.refactorizations += 1
            self._factorize()
        solver = self._solver
        # PARDISO's iparm(12) selects the transposed solve with the stored
        # factors; pypardiso's own ``solve`` resets it on every call, so the
        # private entry point is used and the setting restored afterwards.
        solver.set_iparm(12, 2 if transpose else 0)
        solver.set_phase(33)
        try:
            x = solver._call_pardiso(self._A, np.asfortranarray(b))
        finally:
            solver.set_iparm(12, 0)
            solver.set_phase(13)
        return x


def backward_error(residual, size):
    """Largest ``|r_i| / size_i``: each residual relative to the size of its
    row's terms (``|A| |x| + |b|``), zero where the row is empty.

    The size is floored at machine epsilon times the largest one: a row whose
    terms are smaller than that (activities of 1e-40 next to 1e10, say) holds
    only the rounding of the larger terms, which no solve can remove.
    """
    residual, size = np.abs(np.asarray(residual, dtype=float)), np.asarray(size, dtype=float)
    size = np.maximum(size, np.finfo(float).eps * float(size.max(initial=0.0)))
    ratio = np.divide(residual, size, out=np.zeros_like(residual), where=size > 0)
    return float(ratio.max(initial=0.0))


def _resolve_backend(backend):
    if backend == 'scipy':
        return 'scipy'
    try:
        from pypardiso.scipy_aliases import pypardiso_solver  # noqa: F401
        return 'pardiso'
    except (ImportError, OSError):
        if backend == 'pardiso':
            raise
        return 'scipy'


def matrix_digest(A):
    """Content hash of a sparse matrix (shape, pattern and values)."""
    A = sp.csr_matrix(A)
    digest = hashlib.sha1()
    digest.update(np.asarray(A.shape, dtype=np.int64).tobytes())
    for array, dtype in ((A.indptr, np.int64), (A.indices, np.int64), (A.data, np.float64)):
        digest.update(np.ascontiguousarray(array, dtype=dtype).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Reduced system
# ---------------------------------------------------------------------------

class ReducedSystem:
    """The affine map ``v -> s = A^-1 (f~ + E v)`` for a fixed set of free rows.

    ``columns[k]`` is the product row whose net output is ``v_k``. This is the
    public building block for formulations on top of PULPO's reduced space:
    :meth:`project` gives ``M S`` for any linear functional ``M`` of the
    scaling vector (impact rows, flow rows, rows of ``S`` itself) and
    :meth:`base` gives ``s0``, so that ``M s = M s0 + (M S) v``.

    Projections of process rows are cached per process, so the bound rows
    are computed once for every solve that keeps ``A`` and the columns.
    """

    def __init__(self, factorization, columns):
        self.factorization = factorization
        self.columns = np.asarray(columns, dtype=np.int64)
        self.n = factorization.n
        self.solves = 0
        self._process_rows = {}
        self._vector_rows = {}

    @property
    def n_free(self):
        """Number of reduced variables ``v``."""
        return len(self.columns)

    def base(self, f_tilde):
        """``s0 = A^-1 f~``: the scaling vector at ``v = 0``."""
        self.solves += 1
        return self.factorization.solve(np.asarray(f_tilde, dtype=np.float64))

    def recover(self, f_tilde, v):
        """``s = A^-1 (f~ + E v)``: the full scaling vector of a reduced point."""
        rhs = np.array(f_tilde, dtype=np.float64)
        np.add.at(rhs, self.columns, np.asarray(v, dtype=np.float64))
        self.solves += 1
        return self.factorization.solve(rhs)

    def project(self, M):
        """``M S`` for a sparse ``(r, n)`` matrix ``M``, without forming ``S``.

        Uses ``r`` adjoint solves or ``n_free`` forward solves, whichever is
        fewer, in chunks of :data:`CHUNK_ENTRIES`.
        """
        M = sp.csr_matrix(M, dtype=np.float64)
        r, K = M.shape[0], self.n_free
        out = np.zeros((r, K))
        if r == 0 or K == 0:
            return out
        chunk = max(1, CHUNK_ENTRIES // max(self.n, 1))
        if r <= K:
            for start in range(0, r, chunk):
                block = M[start:start + chunk].T.toarray()
                x = self.factorization.solve(block, transpose=True)
                out[start:start + chunk] = x[self.columns, :].T
                self.solves += block.shape[1]
        else:
            for start in range(0, K, chunk):
                cols = self.columns[start:start + chunk]
                rhs = np.zeros((self.n, len(cols)))
                rhs[cols, np.arange(len(cols))] = 1.0
                x = self.factorization.solve(rhs)
                out[:, start:start + len(cols)] = M @ x
                self.solves += len(cols)
        return out

    def rows(self, processes):
        """``S[J, :]`` for the processes ``J`` (cached per process)."""
        processes = [int(j) for j in processes]
        missing = [j for j in dict.fromkeys(processes) if j not in self._process_rows]
        if missing:
            M = sp.csr_matrix((np.ones(len(missing)), (np.arange(len(missing)), missing)),
                              shape=(len(missing), self.n))
            for j, row in zip(missing, self.project(M)):
                self._process_rows[j] = row
        if not processes:
            return np.zeros((0, self.n_free))
        return np.vstack([self._process_rows[j] for j in processes])

    def project_vectors(self, vectors):
        """``m' S`` for each dense or sparse vector ``m`` (cached by content)."""
        vectors = [sp.csr_matrix(np.asarray(m).reshape(1, -1)) if not sp.issparse(m)
                   else sp.csr_matrix(m).reshape(1, -1) for m in vectors]
        keys = [_vector_key(m) for m in vectors]
        missing = {k: m for k, m in zip(keys, vectors) if k not in self._vector_rows}
        if missing:
            rows = self.project(sp.vstack(list(missing.values()), format='csr'))
            for k, row in zip(missing, rows):
                self._vector_rows[k] = row
        if not keys:
            return np.zeros((0, self.n_free))
        return np.vstack([self._vector_rows[k] for k in keys])


def _vector_key(m):
    m = sp.csr_matrix(m)
    m.sum_duplicates()
    m.eliminate_zeros()
    digest = hashlib.sha1()
    digest.update(m.indices.astype(np.int64).tobytes())
    digest.update(m.data.astype(np.float64).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Reduced LP and its solution
# ---------------------------------------------------------------------------

@dataclass
class ReducedLP:
    """``min c'x + c0  s.t.  row_lower <= matrix x <= row_upper,  col_lower <= x <= col_upper``.

    The variables are ``v`` (in original units), followed by any auxiliary
    columns of the formulation (the transgressions of a goal objective).
    ``row_labels`` / ``col_labels`` name what each row and column is.
    """
    c: np.ndarray
    c0: float
    matrix: sp.csr_matrix
    row_lower: np.ndarray
    row_upper: np.ndarray
    col_lower: np.ndarray
    col_upper: np.ndarray
    row_labels: list = field(default_factory=list)
    col_labels: list = field(default_factory=list)


@dataclass
class LPSolution:
    termination_condition: TerminationCondition
    x: np.ndarray = None
    objective: float = None
    seconds: float = 0.0


def solve_lp(lp, solver_name=None, options=None):
    """Equilibrate and solve a :class:`ReducedLP` with HiGHS (default) or Gurobi.

    ``options`` are passed to the solver after :data:`HIGHS_OPTIONS` /
    :data:`GUROBI_OPTIONS`, so they win; the key ``'tee'`` shows the solver log.
    The returned ``x`` and ``objective`` are in the units of ``lp``.
    """
    name = (solver_name or 'highs').lower()
    if name not in ('highs', 'gurobi'):
        raise ValueError(f"The reduced backend solves with 'highs' or 'gurobi', not {solver_name!r}.")
    options = dict(options or {})
    tee = bool(options.pop('tee', False))

    if len(lp.c) == 0:
        # Nothing to decide (no alternatives, no supply): the balances pin the
        # scaling vector, and the rows only state whether it is feasible.
        size = np.maximum(1.0, np.abs(np.where(np.isfinite(lp.row_lower), lp.row_lower, 0.0))
                          + np.abs(np.where(np.isfinite(lp.row_upper), lp.row_upper, 0.0)))
        tol = 1e-9 * size
        feasible = bool(np.all(lp.row_lower <= tol) and np.all(lp.row_upper >= -tol))
        if not feasible:
            return LPSolution(TerminationCondition.infeasible)
        return LPSolution(TerminationCondition.optimal, np.zeros(0), float(lp.c0))

    # Equilibrate the rows and columns (the objective takes part in the
    # column statistics), then bring the objective to O(1) with one factor.
    stats = sp.vstack([lp.matrix, sp.csr_matrix(lp.c.reshape(1, -1))], format='csr')
    r, d = _scaling.ruiz_scaling(stats)
    r = r[:-1]
    matrix = (sp.diags(r) @ lp.matrix @ sp.diags(d)).tocsc()
    c = lp.c * d
    cmax = np.abs(c).max() if c.size else 0.0
    obj_factor = 2.0 ** -np.round(np.log2(cmax)) if cmax > 0 else 1.0
    c = c * obj_factor
    with np.errstate(invalid='ignore'):
        row_lower, row_upper = lp.row_lower * r, lp.row_upper * r
        col_lower, col_upper = lp.col_lower / d, lp.col_upper / d

    start = time.perf_counter()
    if name == 'highs':
        status, x = _solve_highs(c, matrix, row_lower, row_upper, col_lower, col_upper, options, tee)
    else:
        status, x = _solve_gurobi(c, matrix, row_lower, row_upper, col_lower, col_upper, options, tee)
    seconds = time.perf_counter() - start
    if x is None:
        return LPSolution(status, seconds=seconds)
    x = x * d
    return LPSolution(status, x, float(lp.c @ x + lp.c0), seconds)


def _solve_highs(c, matrix, row_lower, row_upper, col_lower, col_upper, options, tee):
    import highspy
    h = highspy.Highs()
    h.setOptionValue('output_flag', tee)
    for key, value in {**HIGHS_OPTIONS, **options}.items():
        h.setOptionValue(key, value)
    lp = highspy.HighsLp()
    lp.num_col_ = len(c)
    lp.num_row_ = matrix.shape[0]
    lp.col_cost_ = c
    lp.col_lower_ = col_lower
    lp.col_upper_ = col_upper
    lp.row_lower_ = row_lower
    lp.row_upper_ = row_upper
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = matrix.indptr
    lp.a_matrix_.index_ = matrix.indices
    lp.a_matrix_.value_ = matrix.data
    h.passModel(lp)
    h.run()
    status = h.getModelStatus()
    statuses = highspy.HighsModelStatus
    condition = {
        statuses.kOptimal: TerminationCondition.optimal,
        statuses.kInfeasible: TerminationCondition.infeasible,
        statuses.kUnbounded: TerminationCondition.unbounded,
        statuses.kUnboundedOrInfeasible: TerminationCondition.infeasibleOrUnbounded,
        statuses.kTimeLimit: TerminationCondition.maxTimeLimit,
        statuses.kIterationLimit: TerminationCondition.maxIterations,
    }.get(status, TerminationCondition.other)
    if condition != TerminationCondition.optimal:
        return condition, None
    return condition, np.asarray(h.getSolution().col_value, dtype=float)


def _solve_gurobi(c, matrix, row_lower, row_upper, col_lower, col_upper, options, tee):
    import gurobipy as gp
    from gurobipy import GRB
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', int(tee))
    env.start()
    try:
        model = gp.Model(env=env)
        x = model.addMVar(len(c), lb=np.where(np.isfinite(col_lower), col_lower, -GRB.INFINITY),
                          ub=np.where(np.isfinite(col_upper), col_upper, GRB.INFINITY))
        csr = matrix.tocsr()
        eq = np.flatnonzero(row_lower == row_upper)
        lo = np.flatnonzero((row_lower != row_upper) & np.isfinite(row_lower))
        up = np.flatnonzero((row_lower != row_upper) & np.isfinite(row_upper))
        if len(eq):
            model.addMConstr(csr[eq], x, GRB.EQUAL, row_lower[eq])
        if len(lo):
            model.addMConstr(csr[lo], x, GRB.GREATER_EQUAL, row_lower[lo])
        if len(up):
            model.addMConstr(csr[up], x, GRB.LESS_EQUAL, row_upper[up])
        model.setMObjective(None, c, 0.0, sense=GRB.MINIMIZE)
        for key, value in {**GUROBI_OPTIONS, **options}.items():
            model.setParam(key, value)
        model.optimize()
        condition = {
            GRB.OPTIMAL: TerminationCondition.optimal,
            GRB.INFEASIBLE: TerminationCondition.infeasible,
            GRB.UNBOUNDED: TerminationCondition.unbounded,
            GRB.INF_OR_UNBD: TerminationCondition.infeasibleOrUnbounded,
            GRB.TIME_LIMIT: TerminationCondition.maxTimeLimit,
            GRB.ITERATION_LIMIT: TerminationCondition.maxIterations,
        }.get(model.Status, TerminationCondition.other)
        if condition != TerminationCondition.optimal:
            return condition, None
        return condition, np.asarray(x.X, dtype=float)
    finally:
        env.dispose()


# ---------------------------------------------------------------------------
# The reduced model of a PULPO instance
# ---------------------------------------------------------------------------

@dataclass
class ReducedResults:
    """Outcome of :meth:`ReducedModel.solve`.

    ``termination_condition`` is Pyomo's enum, as in the results of a full
    solve with Gurobi. ``v`` and ``s`` are in original units; ``seconds``
    splits the time into assembling the LP (projections included), solving
    it and recovering ``s``. ``balance_residual`` is the largest
    ``|A s - f~ - E v|`` of a balance relative to the size of its terms,
    ``|A| |s| + |f~ + E v|``.
    ``rounds`` counts the LP solves: more than one only when a remote bound
    side or limit (see :attr:`ReducedModel.REMOTE_BOUND`) was withheld and the
    solution violated it, or the problem was unbounded without it.
    """
    termination_condition: TerminationCondition
    objective: float = None
    solver: str = None
    v: np.ndarray = None
    s: np.ndarray = None
    seconds: dict = field(default_factory=dict)
    n_variables: int = 0
    n_rows: int = 0
    balance_residual: float = None
    rounds: int = 1


class ReducedSolveError(RuntimeError):
    """The reduced LP did not end optimal. ``results`` (a :class:`ReducedResults`)
    holds the termination condition, the size of the LP and the times."""

    def __init__(self, message, results):
        super().__init__(message)
        self.results = results


def _finite(value, default):
    return default if value is None else float(value)


def _var_bounds(var, factor=1.0, unscaled=True):
    """Bounds of a Pyomo variable in original units; a fixed variable is pinned
    at its value. ``factor`` converts the model's units to original units;
    ``unscaled`` says whether the stored *value* is already in original units
    (it is after every solve, see ``scaling.unscale_solution``)."""
    if var.fixed:
        level = float(var.value) * (1.0 if unscaled else factor)
        return level, level
    lb, ub = var.bounds
    return _finite(lb, -np.inf) * factor, _finite(ub, np.inf) * factor


class ReducedModel:
    """The reduced formulation of one instantiated (static) PULPO model.

    Built by :func:`build`. Every call to :meth:`linear_program` /
    :meth:`solve` reads the instance's current limits, demand, weights, goals,
    dependent-constraint weights, variable bounds and fixed variables, so
    changes made in place between solves (an epsilon-constraint sweep on
    ``UPPER_IMP_LIMIT``, say) are honoured, while the projections, which depend
    only on ``A``, the choices and the coefficient vectors, are reused.

    Every finite process bound is a row of the LP. A bound side far beyond
    every activity of the problem, such as a capacity of 1e10 meaning
    "unlimited", is the exception: its right-hand side would only degrade the
    conditioning of a solve (of the chance-constrained cone above all), so it
    is withheld and imposed only if a solution violates it (:meth:`optimize`
    re-solves then, which is exact). Impact and flow limits are treated alike.

    Attributes:
        system (ReducedSystem): the map ``v -> s``.
        column_labels (list): per reduced variable, ``('alternative', category,
            process)`` or ``('slack', product)``.
        categories (dict): category -> indices of its alternatives in ``v``.
    """

    #: A withheld (remote) bound or limit counts as violated beyond this
    #: tolerance, relative to the bound (absolute below 1).
    BOUND_TOL = 1e-9

    #: A bound side beyond this multiple of the problem's scale (the largest
    #: base activity or category demand) is withheld until violated.
    REMOTE_BOUND = 1e6

    def __init__(self, instance, lci_data, choices, system):
        self.instance = instance
        self.lci_data = lci_data
        self.system = system
        self.n = system.n
        self.alternatives, self.categories, self.supply_products = _free_rows(instance, lci_data, choices)
        self.column_labels = ([('alternative', cat, j) for j, cat in self.alternatives]
                              + [('slack', i) for i in self.supply_products])

    # -- reading the instance ------------------------------------------------

    def _unscaled(self):
        return bool(getattr(self.instance, '_solution_unscaled', False))

    def process_bounds(self):
        """``(lower, upper)``: every process bound in original units, a fixed
        ``scaling_vector`` entry pinned at its value."""
        inst = self.instance
        col_scale = inst._col_scale if _scaling.is_scaled(inst) else None
        unscaled = self._unscaled()
        lower = np.full(self.n, -np.inf)
        upper = np.full(self.n, np.inf)
        for j, var in inst.scaling_vector.items():
            factor = col_scale[j] if col_scale is not None else 1.0
            lower[j], upper[j] = _var_bounds(var, factor, unscaled)
        return lower, upper

    def demand(self):
        """``f~`` (demand on every non-category row) and the category demands."""
        inst = self.instance
        row_scale = inst._row_scale if _scaling.is_scaled(inst) else None
        f_tilde = np.zeros(self.n)
        f_cat = {}
        for i, value in inst.FINAL_DEMAND.extract_values().items():
            value = float(value) / (row_scale[i] if row_scale is not None else 1.0)
            if i in self.categories:
                f_cat[i] = value
            else:
                f_tilde[i] = value
        return f_tilde, f_cat

    def impact_vectors(self):
        """``{h: c_h}``: the dense environmental cost vector of each indicator, in
        original units (per unit of activity), as the impact rows hold it now."""
        inst = self.instance
        col_scale = inst._col_scale if _scaling.is_scaled(inst) else None
        vectors = {h: np.zeros(self.n) for h in inst.INDICATOR}
        for (j, h), value in inst._env_cost.items():
            if h in vectors:
                vectors[h][j] = value / (col_scale[j] if col_scale is not None else 1.0)
        return vectors

    def _dependent_vectors(self):
        inst = self.instance
        col_scale = inst._col_scale if _scaling.is_scaled(inst) else None
        weights = {d: np.zeros(self.n) for d in inst.DEPENDENT_CONSTRAINTS
                   if inst.DEPENDENT_CNSTR[d].active}
        for param, sign in ((inst.LEFT_WEIGHTS, 1.0), (inst.RIGHT_WEIGHTS, -1.0)):
            for (d, j), value in param.extract_values_sparse().items():
                if d in weights and value:
                    weights[d][j] += sign * float(value) / (col_scale[j] if col_scale is not None else 1.0)
        return weights

    def _slack_bounds(self):
        inst = self.instance
        row_scale = inst._row_scale if _scaling.is_scaled(inst) else None
        unscaled = self._unscaled()
        bounds = [_var_bounds(inst.slack[i], 1.0 / row_scale[i] if row_scale is not None else 1.0, unscaled)
                  for i in self.supply_products]
        return (np.array([b[0] for b in bounds], dtype=float),
                np.array([b[1] for b in bounds], dtype=float))

    # -- assembly --------------------------------------------------------------

    def linear_program(self, processes=None, bounds=None):
        """Assemble the reduced LP for the instance's current data.

        Args:
            processes: process ids whose bounds become rows; ``None`` (as
                :meth:`optimize` uses it) adds a row for every bounded process.
            bounds: ``(lower, upper)`` from :meth:`process_bounds`, if already read.

        Returns ``(lp, f_tilde)``; the first ``system.n_free`` columns of
        ``lp`` are ``v``.
        """
        inst = self.instance
        _check_instance(inst)
        system = self.system
        K = system.n_free
        f_tilde, f_cat = self.demand()
        s0 = system.base(f_tilde)

        blocks, lower, upper, labels = [], [], [], []

        def add(rows, lo, up, row_labels):
            blocks.append(sp.csr_matrix(rows))
            lower.append(np.asarray(lo, dtype=float))
            upper.append(np.asarray(up, dtype=float))
            labels.extend(row_labels)

        # Pooled balances: one row per category.
        cat_rows = np.zeros((len(self.categories), K))
        for r, (cat, idx) in enumerate(self.categories.items()):
            cat_rows[r, idx] = 1.0
        f_c = np.array([f_cat.get(cat, 0.0) for cat in self.categories])
        add(cat_rows, f_c, f_c, [('category', cat) for cat in self.categories])

        # Process bounds.
        p_lower, p_upper = bounds if bounds is not None else self.process_bounds()
        bounded = np.flatnonzero(np.isfinite(p_lower) | np.isfinite(p_upper))
        if processes is not None:
            bounded = np.intersect1d(bounded, np.fromiter(processes, dtype=np.int64))
        if len(bounded):
            add(system.rows(bounded), p_lower[bounded] - s0[bounded], p_upper[bounded] - s0[bounded],
                [('bound', int(j)) for j in bounded])

        # Impact rows: needed for the objective, impact limits and goals.
        impact_vectors = self.impact_vectors()
        indicators = list(impact_vectors)
        m_imp = dict(zip(indicators, system.project_vectors([impact_vectors[h] for h in indicators])))
        m0_imp = {h: float(impact_vectors[h] @ s0) for h in indicators}
        lim_h = []
        for h in indicators:
            lb, ub = _var_bounds(inst.impacts[h])
            if np.isfinite(lb) or np.isfinite(ub):
                lim_h.append((h, lb, ub))
        if lim_h:
            add(np.vstack([m_imp[h] for h, _, _ in lim_h]),
                [lb - m0_imp[h] for h, lb, _ in lim_h], [ub - m0_imp[h] for h, _, ub in lim_h],
                [('impact', h) for h, _, _ in lim_h])

        # Elementary-flow limits.
        flows = []
        for g in inst.INV:
            if not inst.INVENTORY_CNSTR[g].active:
                continue
            lb, ub = _var_bounds(inst.inv_vector[g])
            if np.isfinite(lb) or np.isfinite(ub):
                flows.append((g, lb, ub))
        if flows:
            B = sp.csr_matrix(self.lci_data['intervention_matrix'])[[g for g, _, _ in flows]]
            b0 = B @ s0
            add(system.project_vectors([B[k] for k in range(B.shape[0])]),
                [lb - b0[k] for k, (_, lb, _) in enumerate(flows)],
                [ub - b0[k] for k, (_, _, ub) in enumerate(flows)],
                [('flow', g) for g, _, _ in flows])

        # Dependent constraints: (L - R)' s <= 0.
        dependent = self._dependent_vectors()
        if dependent:
            names = list(dependent)
            add(system.project_vectors([dependent[d] for d in names]),
                np.full(len(names), -np.inf), [-float(dependent[d] @ s0) for d in names],
                [('dependent', d) for d in names])

        matrix = sp.vstack(blocks, format='csr') if blocks else sp.csr_matrix((0, K))
        row_lower = np.concatenate(lower) if lower else np.zeros(0)
        row_upper = np.concatenate(upper) if upper else np.zeros(0)
        col_lower = np.full(K, -np.inf)
        col_upper = np.full(K, np.inf)
        if self.supply_products:
            n_alt = len(self.alternatives)
            col_lower[n_alt:], col_upper[n_alt:] = self._slack_bounds()
        col_labels = list(self.column_labels)

        goals = list(inst.GOAL_INDICATOR)
        if goals:
            # Goal programming: t_h >= impacts_h / L_h - 1 (where the row is
            # active), t_h within its bounds (>= 0), minimize mean t.
            G = len(goals)
            active = [h for h in goals if inst.TRANSGRESSION_CNSTR[h].active]
            col = {h: K + k for k, h in enumerate(goals)}
            goal_rows = np.zeros((len(active), K + G))
            for r, h in enumerate(active):
                goal_rows[r, :K] = m_imp[h] / pyo.value(inst.IMP_GOALS[h])
                goal_rows[r, col[h]] = -1.0
            goal_up = [1.0 - m0_imp[h] / pyo.value(inst.IMP_GOALS[h]) for h in active]
            matrix = sp.vstack([sp.hstack([matrix, sp.csr_matrix((matrix.shape[0], G))]),
                                sp.csr_matrix(goal_rows)], format='csr')
            row_lower = np.concatenate([row_lower, np.full(len(active), -np.inf)])
            row_upper = np.concatenate([row_upper, goal_up])
            labels.extend(('goal', h) for h in active)
            t_bounds = [_var_bounds(inst.transgression[h]) for h in goals]
            col_lower = np.concatenate([col_lower, [b[0] for b in t_bounds]])
            col_upper = np.concatenate([col_upper, [b[1] for b in t_bounds]])
            col_labels.extend(('transgression', h) for h in goals)
            c = np.concatenate([np.zeros(K), np.full(G, 1.0 / G)])
            c0 = 0.0
        else:
            weights = {h: float(pyo.value(inst.WEIGHTS[h])) for h in indicators}
            c = sum((weights[h] * m_imp[h] for h in indicators), np.zeros(K))
            c0 = sum(weights[h] * m0_imp[h] for h in indicators)

        lp = ReducedLP(c=np.asarray(c, dtype=float), c0=float(c0), matrix=matrix,
                       row_lower=row_lower, row_upper=row_upper,
                       col_lower=col_lower, col_upper=col_upper,
                       row_labels=labels, col_labels=col_labels)
        return lp, f_tilde

    # -- solve -----------------------------------------------------------------

    def optimize(self, solve, bounds=None, transform=None):
        """Solve with every process bound as a row, remote sides withheld.

        Args:
            solve: ``solve(lp, f_tilde) -> LPSolution`` for an LP from
                :meth:`linear_program` (whose first ``system.n_free`` columns are
                ``v``); the chance-constrained problem passes its cone solve here.
            bounds: ``(lower, upper)`` process bounds in original units; defaults
                to :meth:`process_bounds`.
            transform: ``transform(lp, f_tilde)``, applied to every LP before it
                is solved (the chance-constrained problem replaces the objective).

        Bound sides and impact or flow limits beyond :attr:`REMOTE_BOUND` times
        the problem's scale are withheld from the solve and checked against its
        solution; one that is violated, or one without which the problem is
        unbounded, is imposed and the LP solved again.

        Returns:
            ``(results, s)``: a :class:`ReducedResults` (``v``, ``objective``,
            ``balance_residual`` set when optimal) and the full scaling vector.
            Nothing is written onto the instance, and nothing is raised: the
            caller checks ``results.termination_condition``.
        """
        start = time.perf_counter()
        K = self.system.n_free
        true_lo, true_up = self.process_bounds() if bounds is None else bounds
        # A bound far beyond every activity of the problem (a 1e10 capacity that
        # stands for "unlimited") cannot bind, but its right-hand side would
        # spoil the conditioning of a cone solve. Such sides are withheld and
        # enter only if a solution violates them.
        f_base, f_cat = self.demand()
        scale = max([1.0, float(np.abs(self.system.base(f_base)).max(initial=0.0))]
                    + [abs(value) for value in f_cat.values()])
        far = self.REMOTE_BOUND * scale
        remote_lo = np.isfinite(true_lo) & (true_lo < -far)
        remote_up = np.isfinite(true_up) & (true_up > far)
        lo = np.where(remote_lo, -np.inf, true_lo)
        up = np.where(remote_up, np.inf, true_up)

        def tolerance(value):
            return self.BOUND_TOL * np.maximum(1.0, np.abs(np.where(np.isfinite(value), value, 0.0)))

        kept_limits = set()                 # limit rows whose remote sides were violated
        seconds = {'assemble': 0.0, 'solve': 0.0, 'recover': 0.0}
        rounds = 0
        s = None
        while True:
            rounds += 1
            t = time.perf_counter()
            lp, f_tilde = self.linear_program(bounds=(lo, up))
            if transform is not None:
                transform(lp, f_tilde)
            withheld = []
            for i, label in enumerate(lp.row_labels):
                if label[0] in ('impact', 'flow') and label not in kept_limits:
                    for side, values, sign in (('lo', lp.row_lower, -1.0), ('up', lp.row_upper, 1.0)):
                        if np.isfinite(values[i]) and sign * values[i] > far:
                            withheld.append((i, label, float(values[i]), side))
                            values[i] = sign * np.inf
            seconds['assemble'] += time.perf_counter() - t
            if rounds == 1:
                self._warn_remote(np.flatnonzero(remote_lo), np.flatnonzero(remote_up), true_lo, true_up,
                                  [label for _, label, _, _ in withheld], scale)
            sol = solve(lp, f_tilde)
            seconds['solve'] += sol.seconds
            if sol.termination_condition != TerminationCondition.optimal:
                # A withheld side can be what keeps the problem bounded: impose
                # them all and solve once more.
                if (withheld or remote_lo.any() or remote_up.any()) and sol.termination_condition in (
                        TerminationCondition.unbounded, TerminationCondition.infeasibleOrUnbounded):
                    lo, up = true_lo.copy(), true_up.copy()
                    remote_lo[:] = remote_up[:] = False
                    kept_limits |= {label for _, label, _, _ in withheld}
                    continue
                break
            violated_limits = set()
            for i, label, value, side in withheld:
                activity = float((lp.matrix[i] @ sol.x)[0])
                slack = self.BOUND_TOL * max(1.0, abs(value))
                if (activity < value - slack) if side == 'lo' else (activity > value + slack):
                    violated_limits.add(label)
            if violated_limits:
                kept_limits |= violated_limits
                continue
            t = time.perf_counter()
            s = self.system.recover(f_tilde, sol.x[:K])
            seconds['recover'] += time.perf_counter() - t
            far_lo = np.flatnonzero(remote_lo & (s < true_lo - tolerance(true_lo)))
            far_up = np.flatnonzero(remote_up & (s > true_up + tolerance(true_up)))
            if far_lo.size or far_up.size:
                lo[far_lo], up[far_up] = true_lo[far_lo], true_up[far_up]
                remote_lo[far_lo] = remote_up[far_up] = False
                continue
            break
        results = ReducedResults(termination_condition=sol.termination_condition,
                                 n_variables=len(lp.c), n_rows=lp.matrix.shape[0],
                                 seconds=seconds, rounds=rounds)
        if sol.termination_condition == TerminationCondition.optimal:
            v = sol.x[:K]
            rhs = f_tilde.copy()
            np.add.at(rhs, self.system.columns, v)
            A = self.lci_data['technology_matrix']
            residual = backward_error(A @ s - rhs, abs(A) @ np.abs(s) + np.abs(rhs))
            results.balance_residual = residual
            if residual > RESIDUAL_LIMIT:
                # Not expected with an invertible A, which Factorization checks;
                # kept so that a wrong point is never reported as a solution.
                results.termination_condition = TerminationCondition.error
            else:
                results.objective = sol.objective
                results.v, results.s = v, s
        results.seconds['total'] = time.perf_counter() - start
        return results, s

    def _warn_remote(self, far_lo, far_up, lower, upper, limits, scale):
        """Tell the user about bounds that stand for "no limit" (see :meth:`optimize`)."""
        count = len(far_lo) + len(far_up) + len(set(limits))
        if not count:
            return
        names = self.lci_data.get('process_map_metadata', {})
        examples = ([f"lower bound {lower[j]:.3g} on '{names.get(int(j), j)}'" for j in far_lo[:2]]
                    + [f"upper bound {upper[j]:.3g} on '{names.get(int(j), j)}'" for j in far_up[:2]]
                    + [f"the {kind} limit on {key!r}" for kind, key in list(dict.fromkeys(limits))[:2]])
        warnings.warn(
            f"{count} bound(s) or limit(s) lie far beyond every activity of this problem (the largest is "
            f"about {scale:.3g}), e.g. {'; '.join(examples[:3])}. They cannot bind, so they are left out of the "
            "solve and imposed only if a solution reaches them. For 'no limit', use float('inf') or None.",
            UserWarning, stacklevel=5)

    def solve(self, solver_name=None, options=None):
        """Solve the reduced LP and write the solution onto the instance.

        Raises :class:`ReducedSolveError` when the LP does not end optimal (or
        the recovered scaling vector does not satisfy the balances); the
        instance then keeps its previous values.
        """
        results, s = self.optimize(lambda lp, f_tilde: solve_lp(lp, solver_name=solver_name, options=options))
        results.solver = (solver_name or 'highs').lower()
        if results.termination_condition != TerminationCondition.optimal:
            raise ReducedSolveError(
                f"The reduced LP did not solve to optimality ({results.termination_condition}); "
                "the instance keeps its previous values.", results)
        self.write_solution(s, results.v)
        return results

    def write_solution(self, s, v):
        """Write a full scaling vector (and the slacks in ``v``) onto the instance.

        Impacts, flows and transgressions are evaluated from ``s`` as the
        constraints of the full model define them. Values are in original
        units; on an equilibrated instance this is the state ``solve_model``
        leaves after a solve. Fixed variables keep their value, converted to
        original units as ``scaling.unscale_solution`` does.
        """
        inst = self.instance
        scaled = _scaling.is_scaled(inst)
        to_original = scaled and not self._unscaled()
        for j, var in inst.scaling_vector.items():
            if var.fixed:
                if to_original:
                    var.set_value(var.value * inst._col_scale[j], skip_validation=True)
            else:
                var.set_value(float(s[j]), skip_validation=True)
        impacts = {h: float(c @ s) for h, c in self.impact_vectors().items()}
        for h, value in impacts.items():
            if not inst.impacts[h].fixed:
                inst.impacts[h].set_value(value, skip_validation=True)
        if len(inst.INV):
            B = sp.csr_matrix(self.lci_data['intervention_matrix'])
            flows = B @ s
            for g in inst.INV:
                if not inst.inv_vector[g].fixed:
                    inst.inv_vector[g].set_value(float(flows[g]), skip_validation=True)
        offset = len(self.alternatives)
        for k, i in enumerate(self.supply_products):
            var = inst.slack[i]
            if var.fixed:
                if to_original:
                    var.set_value(var.value / inst._row_scale[i], skip_validation=True)
            else:
                var.set_value(float(v[offset + k]), skip_validation=True)
        for h in inst.GOAL_INDICATOR:
            var = inst.transgression[h]
            if var.fixed:
                continue
            lb = _finite(var.bounds[0], 0.0)
            if inst.TRANSGRESSION_CNSTR[h].active:
                value = max(lb, impacts[h] / pyo.value(inst.IMP_GOALS[h]) - 1.0)
            else:
                value = lb
            var.set_value(value, skip_validation=True)
        if scaled:
            inst._solution_unscaled = True


def _check_instance(inst):
    if hasattr(inst, 'TIME'):
        raise NotImplementedError("The reduced backend supports static PULPO models only; "
                                  "solve the time-dependent model with method='full'.")
    extra_vars = sorted(set(c.local_name for c in inst.component_objects(pyo.Var)) - _KNOWN_VARS)
    extra_cons = sorted(set(c.local_name for c in inst.component_objects(pyo.Constraint, active=True))
                        - _KNOWN_CONSTRAINTS)
    objectives = [o.local_name for o in inst.component_objects(pyo.Objective, active=True)]
    if extra_vars or extra_cons or objectives != ['OBJ']:
        raise NotImplementedError(
            "The reduced backend supports the components optimizer.instantiate creates; this "
            f"instance also has variables {extra_vars}, constraints {extra_cons} or "
            f"objectives {objectives}. Solve it with method='full'.")
    for name in ('FINAL_DEMAND_CNSTR', 'IMPACTS_CNSTR'):
        component = getattr(inst, name)
        if not component.active or any(not c.active for c in component.values()):
            raise NotImplementedError(f"Deactivated rows of {name} are not supported by the "
                                      "reduced backend; solve with method='full'.")


def _free_rows(instance, lci_data, choices):
    """The alternatives (in the order of ``choices``), their categories, and the
    supply products: the product rows whose net output is free in the LP.

    Mirrors ``converter.combine_inputs``: an alternative's product row is the
    row of its process, and an activity listed in two categories is pooled in
    the last one.
    """
    process_map = lci_data['process_map']
    category_of = {}
    for cat, processes in choices.items():
        for proc in processes:
            category_of[process_map[proc.key]] = cat
    products = set(instance.PRODUCT)
    missing = sorted({str(cat) for cat in category_of.values() if cat not in products})
    if missing:
        raise ValueError(f"The choice categories {missing} are not rows of the instance; "
                         "re-run instantiate() after changing the choices.")
    alternatives = list(category_of.items())
    categories = {}
    for k, (_, cat) in enumerate(alternatives):
        categories.setdefault(cat, []).append(k)
    for cat in instance.PRODUCT:
        if not isinstance(cat, (int, np.integer)) and cat not in categories:
            raise ValueError(f"The instance pools a category {cat!r} that is not in the "
                             "worker's choices; re-run instantiate().")
    supply = sorted(int(i) for i in instance.PRODUCT_SUPPLY)
    return alternatives, categories, supply


#: Per-worker caches of :func:`build`, held outside the worker so that a
#: worker can still be pickled and copied (``solve_MC`` sends it to joblib
#: processes); an entry disappears with its worker.
_CACHES = weakref.WeakKeyDictionary()


def build(worker, backend='auto'):
    """The :class:`ReducedModel` of the worker's current instance.

    The factorization of ``A`` and the projections are cached per worker and
    reused by every later call while the technosphere matrix keeps its content
    (and, for the projections, the alternatives and supply products), across
    re-instantiations with other limits, weights or demand.

    Args:
        worker (PulpoOptimizer): an instantiated optimizer.
        backend (str): factorization backend, see :class:`Factorization`.
    """
    inst = worker.instance
    if inst is None or worker.lci_data is None:
        raise ValueError("Call get_lci_data() and instantiate() before building the reduced model.")
    _check_instance(inst)
    A = worker.lci_data['technology_matrix']
    digest = matrix_digest(A)
    cache = _CACHES.get(worker)
    if cache is None or cache['digest'] != digest or (backend != 'auto' and cache['backend'] != backend):
        factorization = Factorization(A, backend=backend)
        cache = {'digest': digest, 'backend': factorization.backend, 'factorization': factorization,
                 'systems': {}, 'model': None}
        _CACHES[worker] = cache
    model = cache['model']
    if model is not None and model.instance is inst:
        return model
    if len(inst.PROCESS) != A.shape[1] or set(inst.PROCESS) != set(range(A.shape[1])):
        raise ValueError("The instance's processes do not match the columns of the technosphere matrix.")
    alternatives, _, supply = _free_rows(inst, worker.lci_data, worker.choices)
    columns = tuple([j for j, _ in alternatives] + supply)
    system = cache['systems'].get(columns)
    if system is None:
        system = ReducedSystem(cache['factorization'], columns)
        cache['systems'][columns] = system
    model = ReducedModel(inst, worker.lci_data, worker.choices, system)
    cache['model'] = model
    return model
