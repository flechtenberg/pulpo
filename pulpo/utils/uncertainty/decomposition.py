"""Exact variance decomposition at a fixed decision, screening of undeclared
parameters, and post-hoc diagnostics of a solved front.

Why
---
``X(s) = sum_e q_e sum_j b_ej s_j`` is a sum of products of independent
inputs (A1), so its ANOVA decomposition stops at second order. With
``V = Var X(s)`` and ``y_e = sum_j E[b_ej] s_j``,

    S1(q_e)        = w_e y_e^2 / V
    S1(b_ej)       = E[q_e]^2 Var(b_ej) s_j^2 / V
    S2(q_e, b_ej)  = w_e Var(b_ej) s_j^2 / V
    ST(q_e)        = w_e (y_e^2 + sum_j Var(b_ej) s_j^2) / V
    ST(b_ej)       = (E[q_e]^2 + w_e) Var(b_ej) s_j^2 / V

and ``sum S1 + sum S2 = 1`` with no higher-order terms. The Sobol' indices
are therefore computed, not estimated: no sampling and no confidence
intervals. Parameters of one kind (B entries among themselves, CFs among
themselves) never interact, so the first-order and total-order indices of a
family of B entries, or of CFs, are the sums over its members. The uncertain
process bounds are not inputs of ``X`` at a fixed ``s``; they shape ``s``.

Undeclared parameters are deterministic (A3). :func:`screen_undeclared`
tests that assumption: every undeclared parameter gets the same coefficient of
variation ``r`` with its mean unchanged (:func:`widen`), and the indices show
which of them would carry variance. The caller names the CFs that are exact
by definition (``exact_cfs``; for a global warming potential the CO2 flows,
since CO2 is its reference gas), and those never receive a width. No B entry is
exact by definition, so every undeclared one is widened: holding one exact
could only lower the figures, and only the few at the top of the ranking need
an expert's judgement. The distribution family of a widened parameter does not matter to
the moments; :func:`widen` uses a lognormal (mirrored for a negative amount),
so the same data also serve for validation draws and for a re-solved front.
"""

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import stats_arrays

from pulpo.utils.uncertainty.moments import Moments, compute_moments
from pulpo.utils.uncertainty.preparer import UncertaintyData


def _lci(lci_data):
    return lci_data.lci_data if hasattr(lci_data, 'lci_data') else lci_data


def _decision(s):
    """``(s, kappa)`` from a solved point, or ``(s, None)`` from an array."""
    if hasattr(s, 's') and hasattr(s, 'kappa'):
        return np.asarray(s.s, dtype=float), float(s.kappa)
    return np.asarray(s, dtype=float), None


def co2_flows(lci_data):
    """Rows of the intervention matrix whose flow name begins with "Carbon dioxide".

    Fossil, non-fossil, uptake from air, from or to soil: in a global warming
    potential each has the factor +-1 by definition, so this is the usual
    ``exact_cfs`` of :func:`screen_undeclared`, :func:`width_sensitivity` and
    :func:`widen` for such a method. Flows are matched by the name in
    ``intervention_map_metadata`` (ecoinvent's naming); check the result on
    another biosphere.
    """
    lci = _lci(lci_data)
    meta = lci.get('intervention_map_metadata') or {}
    return np.array(sorted(int(e) for e, name in meta.items()
                           if str(name).startswith('Carbon dioxide')), dtype=np.int64)


def families_by_database(lci_data):
    """A ``families`` function for :func:`decompose`: each B entry belongs to
    the database of its process."""
    lci = _lci(lci_data)
    database = np.empty(len(lci['process_map']), dtype=object)
    for key, j in lci['process_map'].items():
        database[j] = key[0]

    def families(flows, processes):
        return database[np.asarray(processes, dtype=np.int64)]
    return families


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

@dataclass
class Decomposition:
    """Sobol' indices of ``X`` at one decision (see the module docstring).

    Attributes:
        variance (float): ``V = Var X(s)``.
        parameters (DataFrame): one row per parameter that carries variance,
            by decreasing ``ST``: ``group`` (``'If'`` or ``'Cf'``), ``flow``,
            ``process`` (-1 for a CF), ``family``, ``S1``, ``ST``.
        interactions (DataFrame): ``S2(q_e, b_ej)`` for each B entry whose CF
            is uncertain, by ``flow`` and ``process``.
    """
    variance: float
    parameters: pd.DataFrame
    interactions: pd.DataFrame

    @property
    def std(self):
        return float(np.sqrt(self.variance))

    def families(self) -> pd.DataFrame:
        """``S1`` and ``ST`` of each family: the sums over its members, exact
        because a family holds either B entries or CFs."""
        return (self.parameters.groupby('family')[['S1', 'ST']].sum()
                .sort_values('ST', ascending=False))

    def top(self, n=10) -> pd.DataFrame:
        """The ``n`` parameters with the largest total-order index."""
        return self.parameters.head(n)


def decompose(s, moments: Moments, families=None) -> Decomposition:
    """Exact first-, second- and total-order Sobol' indices of ``X(s)``.

    Args:
        s: a scaling vector in original units, or a solved
            :class:`cc.Point`.
        moments (Moments): from :func:`moments.compute_moments`.
        families (callable, optional): ``families(flows, processes)`` returns
            a label for each B entry (arrays of equal length), e.g.
            :func:`families_by_database`. CFs always form the family ``'Cf'``;
            without it every B entry is in family ``'If'``.
    """
    s, _ = _decision(s)
    mom = moments
    V = mom.variance(s)
    if not V > 0:
        raise ValueError("Var X(s) = 0: there is no variance to decompose.")
    B_var = mom.B_var.tocoo()
    t = B_var.data * s[B_var.col] ** 2                     # Var(b_ej) s_j^2
    q2, w = mom.q_mean ** 2, mom.q_var
    s1_b = q2[B_var.row] * t / V
    s2 = w[B_var.row] * t / V
    y = np.asarray(mom.B_mean @ s).ravel()
    G = np.bincount(B_var.row, weights=t, minlength=len(q2))
    e = mom.cf_rows
    s1_q = mom.w * y[e] ** 2 / V
    st_q = mom.w * (y[e] ** 2 + G[e]) / V

    keep = (s1_b + s2) > 0
    rows, cols = B_var.row[keep], B_var.col[keep]
    labels = (np.full(len(rows), 'If', dtype=object) if families is None
              else np.asarray(families(rows, cols), dtype=object))
    b_frame = pd.DataFrame({'group': 'If', 'flow': rows, 'process': cols, 'family': labels,
                            'S1': s1_b[keep], 'ST': s1_b[keep] + s2[keep]})
    keep_q = st_q > 0
    q_frame = pd.DataFrame({'group': 'Cf', 'flow': e[keep_q], 'process': -1, 'family': 'Cf',
                            'S1': s1_q[keep_q], 'ST': st_q[keep_q]})
    parameters = (pd.concat([b_frame, q_frame], ignore_index=True)
                  .sort_values('ST', ascending=False, kind='stable').reset_index(drop=True))
    interacting = s2 > 0
    interactions = pd.DataFrame({'flow': B_var.row[interacting], 'process': B_var.col[interacting],
                                 'S2': s2[interacting]})
    return Decomposition(variance=V, parameters=parameters, interactions=interactions)


# ---------------------------------------------------------------------------
# Undeclared parameters
# ---------------------------------------------------------------------------

def lognormal_with_cv(amount, r):
    """A lognormal spec with mean ``amount`` and coefficient of variation ``r``,
    mirrored for a negative amount (``Var = r^2 amount^2`` either way)."""
    scale = float(np.sqrt(np.log1p(r * r)))
    return {'uncertainty_type': stats_arrays.LognormalUncertainty.id, 'amount': float(amount),
            'loc': float(np.log(abs(amount)) - scale ** 2 / 2.0), 'scale': scale,
            'shape': np.nan, 'minimum': np.nan, 'maximum': np.nan, 'negative': bool(amount < 0)}


def _width(widths, group, subgroup):
    if np.isscalar(widths):
        return float(widths)
    key = 'Cf' if group == 'Cf' else subgroup
    return float(widths.get(key, widths.get(subgroup, 0.0)))


def _check_widths(widths, uncertainty_data):
    """A key that names no subgroup would leave a group deterministic unnoticed."""
    if np.isscalar(widths):
        return
    known = set(uncertainty_data['If']) | set(uncertainty_data['Cf']) | {'Cf'}
    unknown = set(widths) - known
    if unknown:
        raise KeyError(f"Widths for {sorted(map(str, unknown))}, which are not subgroups of the data; "
                       f"name a database of {sorted(uncertainty_data['If'])} or 'Cf'.")


def widen(uncertainty_data: UncertaintyData, widths, exact_cfs) -> UncertaintyData:
    """A copy of the data in which every undeclared parameter has a width.

    Each gets :func:`lognormal_with_cv` of its amount: its mean is unchanged
    and its variance is ``r^2 amount^2``. Parameters with a zero amount carry
    nothing and stay undeclared; declared parameters are not touched.

    Args:
        uncertainty_data: from :func:`preparer.import_declared`.
        widths: one coefficient of variation ``r`` for every undeclared
            parameter, or ``{subgroup: r}`` with a database name for its B
            entries and ``'Cf'`` for the CFs; a subgroup not named keeps its
            undeclared parameters deterministic.
        exact_cfs: flow rows whose CF never receives a width, normally the
            CO2 flows, ``co2_flows(worker)``; ``()`` holds none exact.
    """
    _check_widths(widths, uncertainty_data)
    data = copy.deepcopy(uncertainty_data)
    exact = {int(e) for e in exact_cfs}
    for group, blocks in data.items():
        for subgroup, block in blocks.items():
            r = _width(widths, group, subgroup)
            if not r > 0:
                continue
            for index, spec in list(block['undefined'].items()):
                if (group == 'Cf' and int(index) in exact) or spec['amount'] == 0:
                    continue
                block['defined'][index] = lognormal_with_cv(spec['amount'], r)
                del block['undefined'][index]
    return data


def _exact_cfs(exact_cfs):
    return np.asarray(list(exact_cfs), dtype=np.int64)


@dataclass
class Screening:
    """Outcome of :func:`screen_undeclared`.

    Attributes:
        sigma (DataFrame): per width ``r``: ``sigma``, ``ratio`` to the
            declared sigma, and the undeclared parameters' share of ``sum ST``.
        ranking (DataFrame): the undeclared parameters that receive a width,
            by decreasing ``|contribution|``, their deterministic contribution
            ``E[q_e] b_ej s_j`` (``E[q_e] y_e`` for a CF) to ``E[X]``. This is
            the ranking: it does not depend on ``r``. Columns ``ST r=...``
            hold each parameter's total-order index at each width.
        top (dict): ``{r: DataFrame}``, the ``n`` parameters, declared or not,
            with the largest total-order index at width ``r``, with their
            ``share`` of ``sum ST``.
        exact_cfs (ndarray): the flows whose CF was held exact.
    """
    sigma: pd.DataFrame
    ranking: pd.DataFrame
    top: dict
    exact_cfs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def screen_undeclared(s, uncertainty_data: UncertaintyData, lci_data, *, exact_cfs, r=(0.1, 0.3), n=10,
                      families=None) -> Screening:
    """Which undeclared parameters would matter if they were uncertain.

    Every undeclared parameter gets the coefficient of variation ``r`` (see
    :func:`widen`), the CFs in ``exact_cfs`` excepted, and the exact indices of
    :func:`decompose` are evaluated at the fixed decision ``s``. For an
    undeclared B entry ``ST`` is ``(E[q_e]^2 + w_e) r^2 b_ej^2 s_j^2 / V``:
    up to its CF's factor ``1 + w_e / E[q_e]^2`` it is proportional to
    ``(E[q_e] b_ej s_j)^2``, the same at every ``r``. The ``ranking`` uses
    that deterministic contribution, so it does not depend on ``r``; the
    indices at each ``r`` sit beside it.

    Args:
        s: a scaling vector (original units) or a solved :class:`cc.Point`,
            typically the decision of the declared configuration.
        uncertainty_data: the declared configuration.
        lci_data: the LCI data, or a worker holding it.
        r: the widths to evaluate.
        n: rows of each ``top`` table.
        exact_cfs: flow rows whose CF is exact by definition and never
            receives a width, e.g. ``co2_flows(worker)`` for a GWP method.
            Undeclared B entries are all widened: none is exact by definition,
            and holding one exact could only lower the figures.
        families: as in :func:`decompose`.
    """
    s, _ = _decision(s)
    lci = _lci(lci_data)
    exact = _exact_cfs(exact_cfs)
    exact_set = set(exact.tolist())
    declared = compute_moments(uncertainty_data, lci)
    sigma0 = declared.std(s)
    y = np.asarray(declared.B_mean @ s).ravel()

    # The parameters that receive a width, with their r-independent contribution.
    entries = []
    for group, blocks in uncertainty_data.items():
        for subgroup, block in blocks.items():
            for index, spec in block['undefined'].items():
                if spec['amount'] == 0 or (group == 'Cf' and int(index) in exact_set):
                    continue
                if group == 'If':
                    e, j = index
                    contribution = declared.q_mean[e] * spec['amount'] * s[j]
                else:
                    e, j = int(index), -1
                    contribution = declared.q_mean[e] * y[e]
                entries.append({'group': group, 'subgroup': subgroup, 'flow': int(e), 'process': int(j),
                                'amount': float(spec['amount']), 'contribution': float(contribution)})
    ranking = pd.DataFrame(entries, columns=['group', 'subgroup', 'flow', 'process', 'amount', 'contribution'])
    ranking = (ranking.assign(_key=ranking['contribution'].abs())
               .sort_values('_key', ascending=False, kind='stable').drop(columns='_key')
               .reset_index(drop=True))
    undeclared_keys = set(zip(ranking['group'], ranking['flow'], ranking['process']))

    sigma_rows, top = [], {}
    for width in np.atleast_1d(r):
        width = float(width)
        mom = compute_moments(widen(uncertainty_data, width, exact), lci)
        dec = decompose(s, mom, families)
        params = dec.parameters
        total = params['ST'].sum()
        keys = list(zip(params['group'], params['flow'], params['process']))
        is_undeclared = np.array([k in undeclared_keys for k in keys], dtype=bool)
        st = dict(zip(keys, params['ST']))
        ranking[f'ST r={width:g}'] = [st.get(k, 0.0) for k in zip(ranking['group'], ranking['flow'],
                                                                 ranking['process'])]
        head = params.head(n).copy()
        head['share'] = head['ST'] / total
        head['undeclared'] = is_undeclared[:len(head)]
        top[width] = head.reset_index(drop=True)
        sigma_rows.append({'r': width, 'sigma': dec.std, 'ratio': dec.std / sigma0 if sigma0 > 0 else np.nan,
                           'undeclared_share': float(params['ST'][is_undeclared].sum() / total)})
    # A parameter that moves neither the mean nor the variance at s is no part
    # of the result. One with no contribution to the mean can still carry
    # variance: a CF whose flow nets to zero over processes with uncertain entries.
    st_columns = [c for c in ranking.columns if c.startswith('ST r=')]
    ranking = ranking[(ranking['contribution'] != 0) | (ranking[st_columns] > 0).any(axis=1)]
    ranking = ranking.reset_index(drop=True)
    return Screening(sigma=pd.DataFrame(sigma_rows).set_index('r'), ranking=ranking, top=top,
                     exact_cfs=exact)


def width_sensitivity(s, uncertainty_data: UncertaintyData, lci_data, widths, *, exact_cfs,
                      kappa=None) -> pd.DataFrame:
    """``sigma(s)`` with the undeclared parameters at each of ``widths``.

    The mean does not change with the widths, so at a fixed decision the
    chance-constrained impact rises by ``kappa * (sigma - sigma_declared)``.
    Re-optimizing at the wider setting can only do better, so that is an
    upper bound on the rise of the optimal adjusted impact (and the rise is
    not negative, since no variance decreases).

    Args:
        s: a scaling vector, or a solved :class:`cc.Point` (whose ``kappa``
            is used unless ``kappa`` is given).
        uncertainty_data: the configuration the decision was solved for.
        lci_data: the LCI data, or a worker holding it.
        widths: a list of settings, each a coefficient of variation for every
            undeclared parameter or ``{subgroup: r}`` (see :func:`widen`), e.g.
            ``{'ecoinvent-3.10-cutoff': 0.1, 'foreground': 0.3, 'Cf': 0.1}``
            with the names of the worker's databases.
        exact_cfs: flow rows whose CF never receives a width (see
            :func:`screen_undeclared`).

    Returns:
        DataFrame: per setting, the width of each subgroup (``r[...]``),
        ``sigma``, ``ratio`` to the declared sigma, ``delta_sigma`` and, with
        a ``kappa``, ``bound`` (``kappa * delta_sigma``).
    """
    s, point_kappa = _decision(s)
    kappa = point_kappa if kappa is None else kappa
    lci = _lci(lci_data)
    exact = _exact_cfs(exact_cfs)
    sigma0 = compute_moments(uncertainty_data, lci).std(s)
    subgroups = [('If', sub) for sub in uncertainty_data['If']] + [('Cf', 'Cf')]
    rows = []
    for setting in widths:
        sigma = compute_moments(widen(uncertainty_data, setting, exact), lci).std(s)
        row = {f'r[{sub}]': _width(setting, group, sub) for group, sub in subgroups}
        row.update({'sigma': sigma, 'ratio': sigma / sigma0 if sigma0 > 0 else np.nan,
                    'delta_sigma': sigma - sigma0})
        if kappa is not None:
            row['bound'] = kappa * (sigma - sigma0)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diagnostics of a solved front
# ---------------------------------------------------------------------------

def diagnostics(front, moments: Moments) -> pd.DataFrame:
    """Post-hoc figures of each solved point, all evaluated at its ``s``.

    - ``sigma_indep``: the standard deviation with the covariance through
      shared CFs dropped, and ``sigma / sigma_indep``;
    - ``cf_share``: the CFs' share of ``Var X`` (the sum of their total-order
      indices: the variance that goes away when every CF is exact);
    - the bound imposed on each uncertain process (``bound <kind>:<process>``);
    - the size of the reduced problem and the solve times.

    Args:
        front: a :class:`cc.Front` or an iterable of :class:`cc.Point`.
        moments (Moments): the moments the front was solved with.
    """
    points = front.values() if hasattr(front, 'values') else front
    q2 = moments.q_mean ** 2
    rows = []
    for point in points:
        s = np.asarray(point.s, dtype=float)
        V = moments.variance(s)
        sigma, sigma_indep = float(np.sqrt(max(V, 0.0))), moments.std_independent(s)
        no_cf = float(q2 @ (moments.B_var @ (s * s)))       # Var X with every CF exact
        row = {'lambda': point.lambda_level, 'lambda_impact': point.lambda_impact, 'kappa': point.kappa,
               'mean': point.mean, 'sigma': sigma, 'adjusted': point.adjusted,
               'sigma_indep': sigma_indep,
               'sigma_over_indep': sigma / sigma_indep if sigma_indep > 0 else np.nan,
               'cf_share': (V - no_cf) / V if V > 0 else np.nan}
        row.update({f'bound {kind}:{j}': value for (kind, j), value in point.bounds.items()})
        row.update({f'n_{key}': value for key, value in (point.size or {}).items()})
        row.update({f'seconds_{key}': value for key, value in point.seconds.items()})
        row['rounds'] = point.rounds
        rows.append(row)
    return pd.DataFrame(rows).set_index('lambda')
