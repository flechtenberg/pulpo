"""Out-of-sample validation of chance-constrained decisions.

Why
---
A solved point claims reliabilities that rest on the normality of ``X`` (A2)
and on Boole's inequality (A5). Both are checked here without either: at the
fixed decision ``s*`` and the recorded target ``z*``, every declared input is
drawn from its own family (not a fitted normal) and each event is counted, ::

    impact row    X(s*) <= z*                 against lambda_z
    bound j       s*_j <= U_j  (s*_j >= L_j)  against 1 - eps_j
    joint         all of them at once         against lambda

with Wilson intervals. Every decision is priced on the same draws (common
random numbers), so the difference between two decisions is paired: the
regret of one against another is measured in the same world, not as the
difference of two independent Monte Carlo means.

Sampling
--------
``X(s) = sum_e q_e sum_j b_ej s_j``. At fixed decisions only the B entries
that carry variance there matter. Each entry's contribution to ``Var X`` is
known exactly, ``(E[q_e]^2 + w_e) Var(b_ej) s_j^2`` (its total-order index
times ``V``, see :mod:`decomposition`), so :class:`ImpactSampler` holds the
smallest of them at their means, as long as together they stay below ``tol``
of every decision's variance, and reports the share drawn (``covered``).
Entries on processes the decisions do not use cost nothing. Declared CFs are
always drawn. Undeclared parameters are deterministic, as in the model; to
validate under widened ones, pass :func:`decomposition.widen` data.

Each parameter and each uncertain bound is drawn from its own stream, keyed
by the seed and by its position (its entry of ``B``, its flow, its process).
Its draws therefore depend on the seed alone, not on which other parameters or
decisions are drawn, nor on how the draws are blocked. :class:`Validation`
records the seed, and decisions validated in separate calls with the same seed
are paired too, up to the entries held at their means (``tol``).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.stats
import stats_arrays

from pulpo.utils.uncertainty.moments import compute_moments, spec_moments
from pulpo.utils.uncertainty.preparer import UncertaintyData, _validate

#: Draws are generated in blocks of at most this many values, which bounds the
#: memory of a validation regardless of its sample size.
CHUNK_ENTRIES = 2 ** 22


# ---------------------------------------------------------------------------
# Vectorized draws per family
# ---------------------------------------------------------------------------

def _number(value):
    return np.nan if value is None else float(value)


_NORMAL_BASED = (stats_arrays.NormalUncertainty.id, stats_arrays.LognormalUncertainty.id)
_UNIFORM_BASED = (stats_arrays.UniformUncertainty.id, stats_arrays.TriangularUncertainty.id)


def _streams(seed, keys):
    """One generator per key, from the seed and the key alone."""
    return [np.random.default_rng(np.random.SeedSequence(seed, spawn_key=tuple(int(k) for k in key)))
            for key in keys]


class _SpecArrays:
    """Specs as arrays per field, for drawing many parameters at once."""

    def __init__(self, specs):
        specs = list(specs)
        for spec in specs:
            if int(spec['uncertainty_type']) > stats_arrays.NoUncertainty.id:
                _validate(spec)
        self.n = len(specs)
        self.utype = np.array([int(sp['uncertainty_type']) for sp in specs], dtype=np.int64)
        for name in ('amount', 'loc', 'scale', 'minimum', 'maximum'):
            setattr(self, name, np.array([_number(sp.get(name)) for sp in specs], dtype=float))
        self.negative = np.array([bool(sp.get('negative', False)) for sp in specs], dtype=bool)

    def draw(self, n, rng):
        """``n`` draws of every spec from one generator, family by family."""
        standard = np.zeros((self.n, n))
        for family in np.unique(self.utype):
            i = np.flatnonzero(self.utype == family)
            if family in _NORMAL_BASED:
                standard[i] = rng.standard_normal((len(i), n))
            elif family in _UNIFORM_BASED:
                standard[i] = rng.random((len(i), n))
        return self._transform(standard)

    def draw_streams(self, streams, n):
        """The next ``n`` draws of spec ``i`` from ``streams[i]``."""
        standard = np.zeros((self.n, n))
        for i, (family, stream) in enumerate(zip(self.utype, streams)):
            if family in _NORMAL_BASED:
                standard[i] = stream.standard_normal(n)
            elif family in _UNIFORM_BASED:
                standard[i] = stream.random(n)
        return self._transform(standard)

    def _transform(self, standard):
        """Standard normal (normal, lognormal) or uniform (uniform, triangular)
        variates to each spec's family."""
        out = np.empty_like(standard)
        for family in np.unique(self.utype):
            i = np.flatnonzero(self.utype == family)
            z = standard[i]
            if family in (stats_arrays.UndefinedUncertainty.id, stats_arrays.NoUncertainty.id):
                out[i] = self.amount[i, None]
            elif family == stats_arrays.NormalUncertainty.id:
                out[i] = self.loc[i, None] + self.scale[i, None] * z
            elif family == stats_arrays.LognormalUncertainty.id:
                x = np.exp(self.loc[i, None] + self.scale[i, None] * z)
                out[i] = np.where(self.negative[i, None], -x, x)
            elif family == stats_arrays.UniformUncertainty.id:
                lo, hi = self.minimum[i, None], self.maximum[i, None]
                out[i] = lo + (hi - lo) * z
            else:                                    # triangular, by its inverse CDF
                lo, mode, hi = self.minimum[i, None], self.loc[i, None], self.maximum[i, None]
                width = hi - lo
                at_mode = np.divide(mode - lo, width, out=np.zeros_like(width), where=width > 0)
                left = lo + np.sqrt(z * width * (mode - lo))
                right = hi - np.sqrt((1.0 - z) * width * (hi - mode))
                out[i] = np.where(z <= at_mode, left, right)
        return out


def sample_specs(specs, n, seed=None):
    """``n`` draws of each spec, as an array ``(len(specs), n)``.

    Families: exact and undeclared (the amount), normal, lognormal (mirrored
    for ``negative``), uniform and triangular, the same as the moments.
    ``seed`` is a seed or a ``numpy.random.Generator``.
    """
    return _SpecArrays(specs).draw(int(n), np.random.default_rng(seed))


@dataclass
class Draws:
    """Draws of the declared parameters as arrays by parameter: row ``i`` of
    ``B`` is the B entry ``(b_rows[i], b_cols[i])``, row ``i`` of ``Q`` the CF
    of flow ``q_rows[i]``, and column ``t`` is draw ``t`` in both."""
    b_rows: np.ndarray
    b_cols: np.ndarray
    B: np.ndarray
    q_rows: np.ndarray
    Q: np.ndarray


def declared_parameters(uncertainty_data: UncertaintyData, method=None):
    """``(b_rows, b_cols, b_specs), (q_rows, q_specs)``: every declared B entry
    and every declared CF of ``method`` (the data's only one by default)."""
    if method is None:
        (method,) = uncertainty_data['Cf']
    b_index, b_specs = [], []
    for block in uncertainty_data['If'].values():
        for index, spec in block['declared'].items():
            b_index.append(index)
            b_specs.append(spec)
    b_index = np.asarray(b_index, dtype=np.int64).reshape(-1, 2)
    cf = uncertainty_data['Cf'][method]['declared']
    return ((b_index[:, 0], b_index[:, 1], b_specs),
            (np.asarray(list(cf), dtype=np.int64), list(cf.values())))


def draw_parameters(uncertainty_data: UncertaintyData, n, seed=None, method=None, processes=None) -> Draws:
    """``n`` draws of every declared B entry (only those on ``processes``, if
    given) and every declared CF, by parameter.

    The building block for formulations that need the sampled inputs
    themselves, such as a scenario-based risk measure over the reduced space
    (see :class:`cc.Projections`). ``seed`` is a seed or a
    ``numpy.random.Generator``; draw in several calls with one ``Generator``
    to bound the memory.
    """
    rng = np.random.default_rng(seed)
    (b_rows, b_cols, b_specs), (q_rows, q_specs) = declared_parameters(uncertainty_data, method)
    if processes is not None:
        keep = np.isin(b_cols, np.asarray(list(processes), dtype=np.int64))
        b_rows, b_cols, b_specs = b_rows[keep], b_cols[keep], [sp for sp, k in zip(b_specs, keep) if k]
    return Draws(b_rows=b_rows, b_cols=b_cols, B=sample_specs(b_specs, n, rng),
                 q_rows=q_rows, Q=sample_specs(q_specs, n, rng))


# ---------------------------------------------------------------------------
# Realized impacts of fixed decisions
# ---------------------------------------------------------------------------

class ImpactSampler:
    """Realized impacts ``X(s_k)`` of fixed decisions, all on the same draws.

    Args:
        uncertainty_data: the configuration to draw from; it may differ from
            the one the decisions were solved for.
        lci_data: the LCI data, or a worker holding it.
        scaling_vectors: one scaling vector per decision (original units).
        seed (int, optional): keys every parameter's stream (see the module
            docstring); a fresh one is drawn and kept as ``seed`` if omitted.
        tol: the share of each decision's ``Var X`` that may be held at its
            mean (see the module docstring); 0 draws every entry that carries
            variance.
        method (str, optional): the LCIA method; the data's only one by default.

    Attributes:
        moments (Moments): the closed-form moments of ``uncertainty_data``.
        covered (ndarray): per decision, the share of ``Var X`` that is drawn.
        drawn (ndarray): ``(flow, process)`` of each B entry drawn.
        n_drawn (int): the number of B entries drawn.
    """

    def __init__(self, uncertainty_data: UncertaintyData, lci_data, scaling_vectors, seed=None,
                 tol=1e-9, method=None):
        lci = lci_data.lci_data if hasattr(lci_data, 'lci_data') else lci_data
        self.seed = int(np.random.SeedSequence().entropy if seed is None else seed)
        self.moments = mom = compute_moments(uncertainty_data, lci, method)
        S = np.atleast_2d(np.asarray(scaling_vectors, dtype=float))
        (b_rows, b_cols, b_specs), (q_rows, q_specs) = declared_parameters(uncertainty_data, mom.method)
        b_moments = np.array([spec_moments(sp) for sp in b_specs], dtype=float).reshape(-1, 2)
        weight = (mom.q_mean ** 2 + mom.q_var)[b_rows] * b_moments[:, 1]
        contribution = weight[None, :] * S[:, b_cols] ** 2
        V = np.array([mom.variance(s) for s in S])
        share = np.divide(contribution, V[:, None], out=np.zeros_like(contribution), where=V[:, None] > 0)
        # Hold the smallest entries while every decision keeps at least 1 - tol.
        order = np.argsort(share.max(axis=0, initial=0.0), kind='stable')
        n_held = int((np.cumsum(share[:, order], axis=1) <= tol).all(axis=0).sum())
        held, drawn = order[:n_held], np.sort(order[n_held:])
        self.covered = 1.0 - share[:, held].sum(axis=1)
        self.drawn = np.column_stack([b_rows[drawn], b_cols[drawn]])
        self.n_drawn = len(drawn)

        # Only characterized flows enter X.
        flows = np.union1d(np.flatnonzero(mom.q_mean != 0), q_rows)
        position = np.full(len(mom.q_mean), -1, dtype=np.int64)
        position[flows] = np.arange(len(flows))
        # The deterministic part of each decision's inventory: E[B] s without
        # the drawn entries, which are added back per draw.
        base = np.asarray(mom.B_mean @ S.T).T
        for k in range(S.shape[0]):
            base[k] -= np.bincount(b_rows[drawn], weights=b_moments[drawn, 0] * S[k, b_cols[drawn]],
                                   minlength=base.shape[1])
        self._base = base[:, flows]
        self._q_mean = mom.q_mean[flows]
        self._q_pos = position[q_rows]
        self._b_pos = position[b_rows[drawn]]
        self._weights = S[:, b_cols[drawn]]
        self._b = _SpecArrays([b_specs[i] for i in drawn])
        self._q = _SpecArrays(q_specs)
        self._b_streams = _streams(self.seed, [(0, e, j) for e, j in self.drawn])
        self._q_streams = _streams(self.seed, [(1, e) for e in q_rows])
        self._chunk = max(1, CHUNK_ENTRIES // max(len(drawn) + len(flows), 1))

    def sample(self, n):
        """The next ``n`` realizations: an array ``(decisions, n)``."""
        X = np.empty((self._base.shape[0], int(n)))
        for lo in range(0, int(n), self._chunk):
            width = min(self._chunk, int(n) - lo)
            b = self._b.draw_streams(self._b_streams, width)
            q = np.repeat(self._q_mean[:, None], width, axis=1)
            q[self._q_pos] = self._q.draw_streams(self._q_streams, width)
            X[:, lo:lo + width] = self._base @ q + self._weights @ (q[self._b_pos] * b)
        return X


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson(successes, n, level=0.95):
    """The Wilson score interval ``(low, high)`` of a binomial proportion."""
    z = float(scipy.stats.norm.ppf(0.5 + level / 2.0))
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z / denominator * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return float(centre - half), float(centre + half)


def _cvar(x, level):
    """Empirical CVaR at ``level`` (Rockafellar-Uryasev estimator)."""
    var = float(np.quantile(x, level))
    return var + float(np.maximum(x - var, 0.0).mean()) / (1.0 - level)


@dataclass
class Validation:
    """Outcome of :func:`validate`.

    Attributes:
        table (DataFrame): one row per front point (``design = 'front'``) and
            per design and front point, indexed by ``(design, lambda)``:
            the claim (``lambda_impact``, ``threshold`` = z*, ``mean``,
            ``sigma``); the audit of the sampler against the closed form of
            the drawn configuration (``mean_error_in_se``, ``std_ratio``);
            the shape of ``X`` (``skew``, ``excess_kurtosis``) and the
            normal-approximation error ``quantile_error`` (the empirical
            ``lambda_z`` quantile minus z*); ``coverage`` of the impact row,
            of each bound (``coverage <kind>:<process>``, against
            ``nominal <kind>:<process>``) and ``joint_coverage``, each with
            Wilson bounds ``_low``/``_high``; ``cvar_gaussian`` and
            ``cvar_empirical`` at ``lambda_z``; the regret against the lowest
            level of the front and against the best front point in each draw
            (mean and 95th percentile); ``covered_variance``.
        impacts (ndarray): the realized impacts, one row per decision.
        decisions (list): labels of the rows of ``impacts``.
        seed (int): reproduces the draws.
        n (int): draws per decision.
        tol (float): the variance share held at means (see :class:`ImpactSampler`).
    """
    table: pd.DataFrame
    impacts: np.ndarray
    decisions: list
    seed: int
    n: int
    level: float = 0.95
    tol: float = 1e-9
    covered: np.ndarray = field(default=None)


def validate(front, problem, uncertainty_data: UncertaintyData, n=200_000, seed=None, designs=None,
             tol=1e-9, level=0.95) -> Validation:
    """Validate the points of a front out of sample.

    Args:
        front: a :class:`cc.Front` or a list of :class:`cc.Point`.
        problem (ChanceConstrained): the problem the front was solved with; it
            gives the moments of the claim and the uncertain bounds.
        uncertainty_data: the configuration to draw from (usually the one
            solved for; a widened one tests the undeclared parameters).
        n (int): draws per decision.
        seed (int, optional): a fresh one is drawn and recorded if omitted.
        designs (dict, optional): ``{name: s}``, further decisions (e.g. the
            deterministic optimum) priced on the same draws against each
            front point's target and budget.
        tol (float): see :class:`ImpactSampler`.
        level (float): confidence level of the Wilson intervals.
    """
    points = sorted(front.values() if hasattr(front, 'values') else front, key=lambda p: p.lambda_level)
    if not points:
        raise ValueError("The front has no points.")
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be a positive number of draws; got {n}.")
    seed = int(np.random.SeedSequence().entropy if seed is None else seed)
    designs = dict(designs or {})
    labels = [f'lambda={p.lambda_level:g}' for p in points] + list(designs)
    vectors = [p.s for p in points] + [np.asarray(s, dtype=float) for s in designs.values()]

    sampler = ImpactSampler(uncertainty_data, problem.model.lci_data, vectors, seed=seed, tol=tol,
                            method=problem.moments.method)
    X = sampler.sample(n)
    events, specs = problem.events, problem.bound_specs
    U = _SpecArrays([specs[e] for e in events]).draw_streams(
        _streams(seed, [(2, kind == 'upper', j) for kind, j in events]), n)
    front_best = X[:len(points)].min(axis=0)
    world = sampler.moments
    claim = problem.moments

    rows = []
    for k, (label, s) in enumerate(zip(labels, vectors)):
        x = X[k]
        held = {(kind, j): (s[j] <= U[i]) if kind == 'upper' else (s[j] >= U[i])
                for i, (kind, j) in enumerate(events)}
        all_held = np.logical_and.reduce(list(held.values())) if held else np.ones(n, dtype=bool)
        premium, hindsight = x - X[0], x - front_best
        mean, sigma = claim.mean(s), claim.std(s)
        world_mean, world_sigma = world.mean(s), world.std(s)
        mc_mean, mc_std = float(x.mean()), float(x.std(ddof=1))
        common = {
            'mean': mean, 'sigma': sigma, 'mc_mean': mc_mean, 'mc_std': mc_std,
            'mean_error_in_se': ((mc_mean - world_mean) / (world_sigma / np.sqrt(n))
                                 if world_sigma > 0 else np.nan),
            'std_ratio': mc_std / world_sigma if world_sigma > 0 else np.nan,
            'skew': float(scipy.stats.skew(x)), 'excess_kurtosis': float(scipy.stats.kurtosis(x)),
            'regret_vs_lowest_lambda_mean': float(premium.mean()),
            'regret_vs_lowest_lambda_p95': float(np.percentile(premium, 95)),
            'regret_vs_front_best_mean': float(hindsight.mean()),
            'regret_vs_front_best_p95': float(np.percentile(hindsight, 95)),
            'covered_variance': float(sampler.covered[k])}
        for point in (points[k:k + 1] if k < len(points) else points):
            z, lz = point.adjusted, point.lambda_impact
            ok = x <= z
            row = {'design': 'front' if k < len(points) else label, 'lambda': point.lambda_level,
                   'lambda_impact': lz, 'threshold': z, **common}
            row['coverage'] = float(ok.mean())
            row['coverage_low'], row['coverage_high'] = wilson(int(ok.sum()), n, level)
            row['quantile_error'] = float(np.quantile(x, lz)) - z
            row['cvar_gaussian'] = mean + float(scipy.stats.norm.pdf(scipy.stats.norm.ppf(lz))) / (1.0 - lz) * sigma
            row['cvar_empirical'] = _cvar(x, lz)
            for (kind, j), h in held.items():
                name = f'{kind}:{j}'
                row[f'coverage {name}'] = float(h.mean())
                row[f'coverage {name}_low'], row[f'coverage {name}_high'] = wilson(int(h.sum()), n, level)
                row[f'nominal {name}'] = 1.0 - point.epsilon[(kind, j)]
            joint = ok & all_held
            row['joint_coverage'] = float(joint.mean())
            row['joint_coverage_low'], row['joint_coverage_high'] = wilson(int(joint.sum()), n, level)
            rows.append(row)
    table = pd.DataFrame(rows).set_index(['design', 'lambda'])
    return Validation(table=table, impacts=X, decisions=labels, seed=seed, n=n, level=level, tol=tol,
                      covered=sampler.covered)
