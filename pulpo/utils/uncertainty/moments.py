"""Closed-form moments of the impact ``X(s) = sum_e q_e sum_j b_ej s_j``.

Why
---
The chance constraint needs the mean and the variance of ``X`` at every
decision ``s``. Both follow in closed form from the first two moments of each
input, which every supported family has in closed form too, so nothing is
sampled and no parameter has to be normal. The only normality assumption is on
the aggregate ``X`` (made by the chance constraint, not here).

What
----
Assumptions: all uncertain inputs are mutually independent (A1); undeclared
parameters are deterministic (A3); ``A`` is deterministic (A4). Then, with
``y_e = sum_j E[b_ej] s_j`` the mean inventory flow ``e``,

    E[X]  = mu' s,                    mu_j = sum_e E[q_e] E[b_ej]
    Var X = sum_j d_j s_j^2 + sum_e w_e y_e^2
            d_j = sum_e (E[q_e]^2 + Var q_e) Var b_ej,     w_e = Var q_e

The second term is the covariance that every process using flow ``e`` shares
through the one factor ``q_e``; dropping it gives :meth:`Moments.std_independent`.

Families: lognormal (also with ``negative`` amounts), normal, uniform,
triangular, and ``uncertainty_type = 1`` (an exact value). Any other family
raises.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import stats_arrays

from pulpo.utils.uncertainty.preparer import UncertaintyData, UncertaintySpec, _validate


def spec_moments(spec: UncertaintySpec):
    """``(mean, variance)`` of one parameter's declared distribution.

    An undeclared parameter (``uncertainty_type`` 0) and an exact one (1) have
    mean ``amount`` and variance 0.
    """
    utype = int(spec['uncertainty_type'])
    if utype in (stats_arrays.UndefinedUncertainty.id, stats_arrays.NoUncertainty.id):
        return float(spec['amount']), 0.0
    _validate(spec)
    if utype == stats_arrays.NormalUncertainty.id:
        return float(spec['loc']), float(spec['scale']) ** 2
    if utype == stats_arrays.UniformUncertainty.id:
        a, b = float(spec['minimum']), float(spec['maximum'])
        return (a + b) / 2.0, (b - a) ** 2 / 12.0
    if utype == stats_arrays.TriangularUncertainty.id:
        a, c, b = float(spec['minimum']), float(spec['loc']), float(spec['maximum'])
        return (a + b + c) / 3.0, max((a * a + b * b + c * c - a * b - a * c - b * c) / 18.0, 0.0)
    # Lognormal: ln|x| ~ N(loc, scale^2); a negative amount mirrors the
    # distribution, which flips the mean and keeps the variance.
    mu, sigma = float(spec['loc']), float(spec['scale'])
    sign = -1.0 if spec.get('negative', False) else 1.0
    return (sign * np.exp(mu + sigma ** 2 / 2.0),
            float(np.expm1(sigma ** 2) * np.exp(2.0 * mu + sigma ** 2)))


def compute_closed_form_moments(uncertainty_data: UncertaintyData):
    """``{group: {subgroup: {index: (mean, variance)}}}`` for every parameter,
    declared and undeclared."""
    return {group: {sub: {index: spec_moments(spec)
                          for status in ('defined', 'undefined')
                          for index, spec in block[status].items()}
                    for sub, block in blocks.items()}
            for group, blocks in uncertainty_data.items()}


@dataclass
class Moments:
    """The moment coefficients of one impact, in original units.

    Attributes:
        method (str): the LCIA method.
        mu (ndarray, n_process): ``E[c_j]``, the expected impact of one unit of process j.
        d (ndarray, n_process): the diagonal of the variance's process term.
        w (ndarray, n_cf): ``Var q_e`` of the flows whose CF is uncertain.
        cf_rows (ndarray, n_cf): the biosphere rows of ``w``.
        B_mean (csr, n_flow x n_process): ``E[B]``.
        B_var (csr, n_flow x n_process): ``Var b_ej`` (zero where undeclared).
        q_mean, q_var (ndarray, n_flow): moments of the characterization factors.
    """
    method: str
    mu: np.ndarray
    d: np.ndarray
    w: np.ndarray
    cf_rows: np.ndarray
    B_mean: sp.csr_matrix
    B_var: sp.csr_matrix
    q_mean: np.ndarray
    q_var: np.ndarray

    def summary(self) -> dict:
        """How many processes carry variance and how many CFs are uncertain."""
        return {'processes': int(self.mu.size), 'processes_with_variance': int((self.d > 0).sum()),
                'uncertain_cfs': int(self.w.size)}

    @property
    def B_unc(self):
        """Rows of ``E[B]`` whose characterization factor is uncertain."""
        return self.B_mean[self.cf_rows]

    def mean(self, s):
        """``E[X(s)] = mu' s``."""
        return float(self.mu @ np.asarray(s, dtype=float))

    def variance(self, s):
        """``Var X(s)``, the shared-factor covariance included."""
        s = np.asarray(s, dtype=float)
        y = self.B_unc @ s
        return float(self.d @ (s * s) + self.w @ (y * y))

    def std(self, s):
        return float(np.sqrt(max(self.variance(s), 0.0)))

    def process_std(self):
        """``sigma_j``: the standard deviation of ``c_j = sum_e q_e b_ej``, the
        impact of one unit of process j, ``sigma_j^2 = d_j + sum_e w_e E[b_ej]^2``."""
        var_c = self.d + np.asarray(self.q_var @ self.B_mean.multiply(self.B_mean)).ravel()
        return np.sqrt(np.clip(var_c, 0.0, None))

    def std_independent(self, s):
        """The standard deviation with the covariance through shared factors
        dropped: each process's impact ``c_j`` treated as independent."""
        s = np.asarray(s, dtype=float)
        return float(np.linalg.norm(self.process_std() * s))


def current_scaling_vector(model_instance, n=None):
    """The scaling vector a solved (static) instance holds, as an array in
    process order and original units, for evaluating :class:`Moments` at it."""
    n = len(model_instance.PROCESS) if n is None else n
    s = np.zeros(n)
    for j, var in model_instance.scaling_vector.items():
        if var.value is not None:
            s[int(j)] = var.value
    return s


def compute_moments(uncertainty_data: UncertaintyData, lci_data, method=None) -> Moments:
    """Assemble :class:`Moments` from imported (and possibly overridden) data.

    Args:
        uncertainty_data: from :func:`preparer.import_declared`.
        lci_data (dict or PulpoOptimizer): the LCI data, or a worker holding it.
        method (str, optional): defaults to the method of ``uncertainty_data``.
    """
    lci = lci_data.lci_data if hasattr(lci_data, 'lci_data') else lci_data
    if method is None:
        (method,) = uncertainty_data['Cf']
    B = sp.csr_matrix(lci['intervention_matrix'], dtype=float)
    n_flow, n_proc = B.shape

    rows, cols, means, variances = [], [], [], []
    for block in uncertainty_data['If'].values():
        for status in ('defined', 'undefined'):
            for (e, j), spec in block[status].items():
                mean, var = spec_moments(spec)
                rows.append(e)
                cols.append(j)
                means.append(mean)
                variances.append(var)
    rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    # E[B]: the stored amounts, with every imported parameter replaced by its mean.
    B_mean = B.tolil(copy=True)
    if len(rows):
        B_mean[rows, cols] = np.asarray(means)
    B_mean = B_mean.tocsr()
    B_mean.eliminate_zeros()
    B_var = sp.csr_matrix((np.asarray(variances, dtype=float), (rows, cols)), shape=(n_flow, n_proc))
    B_var.eliminate_zeros()

    q_mean = np.asarray(lci['matrices'][method].diagonal(), dtype=float).ravel().copy()
    q_var = np.zeros(n_flow)
    cf = uncertainty_data['Cf'][method]
    for status in ('defined', 'undefined'):
        for e, spec in cf[status].items():
            q_mean[e], q_var[e] = spec_moments(spec)

    mu = np.asarray(q_mean @ B_mean).ravel()
    d = np.asarray((q_mean ** 2 + q_var) @ B_var).ravel()
    cf_rows = np.flatnonzero(q_var > 0)
    return Moments(method=method, mu=mu, d=d, w=q_var[cf_rows], cf_rows=cf_rows,
                   B_mean=B_mean, B_var=B_var, q_mean=q_mean, q_var=q_var)
