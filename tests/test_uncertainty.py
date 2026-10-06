"""Tests for the uncertainty module.

- the import of declared parameters (``preparer``), and expert overrides;
- the closed-form moments of each family and of the impact, against Monte
  Carlo (within four standard errors) and against PULPO 1.8.0's coefficients;
- the risk budget and the exact bound quantiles (against SciPy, 1e-12);
- the chance-constrained problem in reduced space, against PULPO 1.8.0's
  reference front, an independent full-space cone, an analytic optimum on an
  ill-conditioned facility system, and across solvers, scaling and the two
  ways of factoring the variance;
- the bw25 extraction of uncertainty parameters in ``bw_parser.import_data``.

Reference values of PULPO 1.8.0 (tag v1.8.0) come from its own exact
chance-constrained pipeline on the SOC demo database: no parameter filter,
undeclared parameters deterministic, closed-form moments, a Bonferroni budget
with exact quantiles for the uncertain cap, and cutting planes to a relative
gap of 1e-11.
"""

import copy
import io
import contextlib
import types
import unittest
import warnings

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import scipy.sparse as sp
import scipy.stats
import stats_arrays
import bw2data as bd

from pulpo import pulpo, pulpo_unc
from pulpo.utils import bw_parser, converter, optimizer, reduced, scaling
from pulpo.utils import uncertainty as unc
from pulpo.utils.uncertainty import cc, moments as moments_module, processor
from pulpo.utils.utils import is_bw25
from pulpo.datasets.sample_database import setup_sample_db
from pulpo.datasets.soc_demo_database import setup_soc_demo_db, METHOD_KEY

with contextlib.redirect_stdout(io.StringIO()):
    setup_sample_db()
    setup_soc_demo_db()

PROJECT = "sample_project_bw25" if is_bw25() else "sample_project"
DATABASES = ["background_db", "foreground_db"]
CLIMATE_KEY = "('my project', 'climate change')"
SOC_PROJECT = "soc_demo_project_bw25" if is_bw25() else "soc_demo_project"
SOC_DBS = ['soc_demo_background_db', 'soc_demo_foreground_db']
SOC_METHOD = str(METHOD_KEY)

#: The uncertain electrolysis capacity of the reference runs.
TRI_CAP = {'uncertainty_type': 5, 'minimum': 0.005, 'loc': 0.03, 'maximum': 0.035}

#: PULPO 1.8.0 on the SOC demo: lambda -> (adjusted impact, electrolysis activity).
V18_FREE = {0.5: (1.865886926931, 0.178000003099), 0.7: (2.266162388164, 0.066156919274),
            0.9: (2.498129644517, 0.028422647633), 0.95: (2.600239384625, 0.024531214375),
            0.99: (2.788914638782, 0.020714075925)}
V18_CAP = {0.5: (2.336715722843, 0.018693063938), 0.7: (2.439943323446, 0.015606601718),
           0.9: (2.612847472715, 0.011123724357), 0.95: (2.702274329489, 0.009330127019),
           0.99: (2.876249701522, 0.006936491673)}
#: PULPO 1.8.0's moment coefficients (mu_j, d_j) by process, and w of the methane CF.
V18_MU_D = {'natural gas extraction': (0.2373365139441714, 0.003363260101492242),
            'electricity supply': (0.1699722655725929, 0.008205656484765336),
            'hydrogen SMR': (9.759025373346807, 2.1008190540748135),
            'N2 air separation': (0.0, 0.0), 'hydrogen electrolysis': (0.0, 0.0),
            'ammonia synthesis': (0.26570000327774324, 0.00894347998195326)}
V18_W_METHANE = 8.954250039306524


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def soc_worker(electrolysis_cap=float('inf'), scale=False, worker_class=pulpo.PulpoOptimizer, **kwargs):
    worker = worker_class(SOC_PROJECT, SOC_DBS, {SOC_METHOD: 1})
    quiet(worker.get_lci_data)
    get = worker.retrieve_processes
    worker.ammonia = get(processes=['ammonia synthesis'])[0]
    worker.smr = get(processes=['hydrogen SMR'])[0]
    worker.elyz = get(processes=['hydrogen electrolysis'])[0]
    choices = {'hydrogen': {worker.smr: float('inf'), worker.elyz: electrolysis_cap}}
    quiet(worker.instantiate, choices=choices, demand={worker.ammonia: 1}, scale=scale, **kwargs)
    return worker


def index(worker, activity):
    return worker.lci_data['process_map'][activity.key]


# ---------------------------------------------------------------------------
# full-space reference cone, written independently of the reduced code
# ---------------------------------------------------------------------------

def full_space_front(worker, mom, lambdas, cap_spec=None, K=1):
    """min mu's + kappa ||G s|| over the merged balances and the bounds, in full space."""
    import clarabel
    A = sp.csr_matrix(worker.lci_data['technology_matrix']).toarray()
    n = A.shape[0]
    alts = [index(worker, worker.smr), index(worker, worker.elyz)]
    keep = [i for i in range(n) if i not in alts]
    A_m = np.vstack([A[keep], A[alts].sum(axis=0)])
    f = np.zeros(n)
    f[index(worker, worker.ammonia)] = 1.0
    f_m = np.append(f[keep], 0.0)
    G = np.vstack([np.diag(np.sqrt(mom.d)), np.sqrt(mom.w)[:, None] * mom.B_unc.toarray()])
    out = {}
    for lam in lambdas:
        budget = cc.bonferroni_budget(lam, K)
        kappa = scipy.stats.norm.ppf(budget.lambda_impact)
        rows, rhs = [-np.eye(n)[alts]], [np.zeros(2)]
        if cap_spec is not None:
            cap = cc.declared_quantile(cap_spec, budget.epsilon_at(1))
            rows.append(np.eye(n)[[alts[1]]])
            rhs.append([cap])
        A_le = np.vstack(rows)
        Acone = np.vstack([np.append(np.zeros(n), -1.0), np.hstack([-G, np.zeros((G.shape[0], 1))])])
        A_all = sp.csc_matrix(np.vstack([np.hstack([A_m, np.zeros((A_m.shape[0], 1))]),
                                         np.hstack([A_le, np.zeros((A_le.shape[0], 1))]), Acone]))
        b_all = np.concatenate([f_m, np.concatenate(rhs), np.zeros(1 + G.shape[0])])
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-11
        sol = clarabel.DefaultSolver(sp.csc_matrix((n + 1, n + 1)), np.append(mom.mu, kappa), A_all, b_all,
                                     [clarabel.ZeroConeT(A_m.shape[0]), clarabel.NonnegativeConeT(A_le.shape[0]),
                                      clarabel.SecondOrderConeT(1 + G.shape[0])], settings).solve()
        s = np.asarray(sol.x[:n])
        out[lam] = (mom.mean(s) + kappa * mom.std(s), s)
    return out


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------

class TestImport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.worker = soc_worker()
        cls.data = unc.import_declared(cls.worker)

    def test_structure_and_families(self):
        self.assertEqual(set(self.data['If']), set(SOC_DBS))
        self.assertEqual(list(self.data['Cf']), [SOC_METHOD])
        families = {int(spec['uncertainty_type']) for group in self.data.values()
                    for block in group.values() for spec in block['declared'].values()}
        self.assertTrue({2, 3, 4, 5} <= families)
        cf = self.data['Cf'][SOC_METHOD]
        co2, n2o = 0, 2
        self.assertEqual(cf['declared'][co2]['uncertainty_type'], 3)      # exact: N(1, 0)
        self.assertEqual(cf['declared'][co2]['scale'], 0.0)
        self.assertIn(n2o, cf['undeclared'])

    def test_every_characterized_entry_is_a_parameter(self):
        """No filter: the parameters are exactly the nonzero, characterized entries of B."""
        B = sp.coo_matrix(self.worker.lci_data['intervention_matrix'])
        q = self.worker.lci_data['matrices'][SOC_METHOD].diagonal()
        characterized = {e for e in range(B.shape[0])}            # all three flows carry a CF here
        self.assertTrue(all(q[e] != 0 for e in characterized))
        expected = {(int(e), int(j)) for e, j, v in zip(B.row, B.col, B.data) if v != 0 and e in characterized}
        imported = {idx for block in self.data['If'].values()
                    for status in ('declared', 'undeclared') for idx in block[status]}
        self.assertEqual(imported, expected)

    def test_counts_and_undeclared(self):
        rows = unc.counts(self.data)
        total = sum(r['n'] for r in rows)
        n = sum(len(b[s]) for g in self.data.values() for b in g.values() for s in ('declared', 'undeclared'))
        self.assertEqual(total, n)
        und = unc.undeclared(self.data)
        self.assertIn(2, und['Cf'][SOC_METHOD])

    def test_one_method_only(self):
        worker = pulpo.PulpoOptimizer(PROJECT, DATABASES, {CLIMATE_KEY: 1, "('my project', 'air quality')": 1})
        quiet(worker.get_lci_data)
        with self.assertRaises(ValueError):
            unc.import_declared(worker)

    def test_override(self):
        data = copy.deepcopy(self.data)
        n2o = 2
        amount = data['Cf'][SOC_METHOD]['undeclared'][n2o]['amount']
        unc.override(data, 'Cf', SOC_METHOD, {n2o: {'uncertainty_type': 4, 'minimum': 200.0, 'maximum': 350.0}})
        spec = data['Cf'][SOC_METHOD]['declared'][n2o]
        self.assertNotIn(n2o, data['Cf'][SOC_METHOD]['undeclared'])
        self.assertEqual(spec['amount'], amount)                      # the deterministic value stays
        with self.assertRaises(KeyError):
            unc.override(data, 'Cf', SOC_METHOD, {99: {'uncertainty_type': 3, 'loc': 1.0, 'scale': 0.1}})
        with self.assertRaises(ValueError):                            # mode outside the support
            unc.override(data, 'Cf', SOC_METHOD, {n2o: {'uncertainty_type': 5, 'minimum': 1.0,
                                                        'loc': 5.0, 'maximum': 2.0}})
        with self.assertRaises(NotImplementedError):                   # Weibull: no closed form here
            unc.override(data, 'Cf', SOC_METHOD, {n2o: {'uncertainty_type': 8, 'loc': 1.0, 'scale': 1.0}})
        unc.override(data, 'Cf', SOC_METHOD, {n2o: {'uncertainty_type': 0}})
        self.assertIn(n2o, data['Cf'][SOC_METHOD]['undeclared'])

    def test_shared_entries_are_refused(self):
        """Two exchanges on one entry of B have no single declared distribution."""
        lci = dict(self.worker.lci_data)
        params = lci['intervention_params']
        lci['intervention_params'] = np.concatenate([params, params[:1]])
        fake = types.SimpleNamespace(lci_data=lci, database=SOC_DBS, method={SOC_METHOD: 1})
        with self.assertRaises(NotImplementedError):
            unc.import_declared(fake)


# ---------------------------------------------------------------------------
# moments
# ---------------------------------------------------------------------------

class TestSpecMoments(unittest.TestCase):
    """Closed-form mean and variance of each family against Monte Carlo."""

    N = 400_000

    def assert_matches(self, spec, draws):
        mean, var = unc.spec_moments(spec)
        se_mean = draws.std() / np.sqrt(len(draws))
        centred = draws - draws.mean()
        se_var = np.sqrt(((centred ** 2 - centred.var()) ** 2).mean() / len(draws))
        self.assertLess(abs(mean - draws.mean()), 4 * se_mean, f"mean of {spec}")
        self.assertLess(abs(var - draws.var()), 4 * se_var, f"variance of {spec}")

    def test_families_against_monte_carlo(self):
        rng = np.random.default_rng(20261001)
        cases = [
            ({'uncertainty_type': 2, 'loc': np.log(0.15), 'scale': 0.5, 'amount': 0.15},
             rng.lognormal(np.log(0.15), 0.5, self.N)),
            ({'uncertainty_type': 2, 'loc': np.log(2.0), 'scale': 0.3, 'amount': -2.0, 'negative': True},
             -rng.lognormal(np.log(2.0), 0.3, self.N)),
            ({'uncertainty_type': 3, 'loc': -1.5, 'scale': 0.4, 'amount': -1.5},
             rng.normal(-1.5, 0.4, self.N)),
            ({'uncertainty_type': 5, 'minimum': 0.1, 'loc': 1.0, 'maximum': 1.1, 'amount': 1.0},
             rng.triangular(0.1, 1.0, 1.1, self.N)),
            ({'uncertainty_type': 4, 'minimum': 2.0, 'maximum': 6.0, 'amount': 4.0},
             rng.uniform(2.0, 6.0, self.N)),
        ]
        for spec, draws in cases:
            with self.subTest(family=spec['uncertainty_type'], negative=spec.get('negative', False)):
                self.assert_matches(spec, draws)

    def test_exact_and_undeclared(self):
        self.assertEqual(unc.spec_moments({'uncertainty_type': 0, 'amount': 3.0}), (3.0, 0.0))
        self.assertEqual(unc.spec_moments({'uncertainty_type': 1, 'amount': -2.0}), (-2.0, 0.0))

    def test_unsupported_family_raises(self):
        with self.assertRaises(NotImplementedError):
            unc.spec_moments({'uncertainty_type': 6, 'amount': 1.0, 'loc': 1.0, 'scale': 1.0, 'shape': 2.0})


class TestImpactMoments(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.worker = soc_worker()
        cls.data = unc.import_declared(cls.worker)
        cls.mom = unc.compute_moments(cls.data, cls.worker)

    def test_coefficients_match_pulpo_1_8(self):
        names = {j: key[1] for key, j in self.worker.lci_data['process_map'].items()}
        for j, name in names.items():
            with self.subTest(process=name):
                self.assertAlmostEqual(self.mom.mu[j], V18_MU_D[name][0], places=13)
                self.assertAlmostEqual(self.mom.d[j], V18_MU_D[name][1], places=13)
        self.assertEqual(list(self.mom.cf_rows), [1])
        self.assertAlmostEqual(self.mom.w[0], V18_W_METHANE, places=12)

    def test_mean_and_variance_of_X_against_monte_carlo(self):
        """X(s) sampled parameter by parameter. The methane CF is widened and s
        weights the methane-emitting processes, so that the CF term w_e y_e^2 is
        a sizeable part of Var X and the test can tell the variance from the
        variance without it."""
        data = copy.deepcopy(self.data)
        unc.override(data, 'Cf', SOC_METHOD, {1: {'uncertainty_type': 3, 'loc': 29.7, 'scale': 15.0}})
        mom = unc.compute_moments(data, self.worker)
        names = {key[1]: j for key, j in self.worker.lci_data['process_map'].items()}
        s = np.zeros(mom.mu.size)
        s[names['natural gas extraction']], s[names['hydrogen SMR']] = 3.0, 0.5
        rng = np.random.default_rng(7)
        N = 400_000
        B = sp.csr_matrix(self.worker.lci_data['intervention_matrix']).toarray()
        q = np.asarray(self.worker.lci_data['matrices'][SOC_METHOD].diagonal(), dtype=float)

        def draw(spec):
            t = int(spec['uncertainty_type'])
            if t == 2:
                x = rng.lognormal(spec['loc'], spec['scale'], N)
                return -x if spec.get('negative') else x
            if t == 3:
                return rng.normal(spec['loc'], spec['scale'], N) if spec['scale'] > 0 else np.full(N, spec['loc'])
            if t == 4:
                return rng.uniform(spec['minimum'], spec['maximum'], N)
            if t == 5:
                return rng.triangular(spec['minimum'], spec['loc'], spec['maximum'], N)
            return np.full(N, spec['amount'])

        g = {e: np.full(N, B[e] @ s) for e in range(B.shape[0])}
        for block in data['If'].values():
            for (e, j), spec in block['declared'].items():
                g[e] += (draw(spec) - B[e, j]) * s[j]
        X = np.zeros(N)
        for e in range(B.shape[0]):
            spec = data['Cf'][SOC_METHOD]['declared'].get(e) or data['Cf'][SOC_METHOD]['undeclared'].get(e)
            X += (draw(spec) if spec is not None else q[e]) * g[e]
        se_mean = X.std() / np.sqrt(N)
        c = X - X.mean()
        se_var = np.sqrt(((c ** 2 - c.var()) ** 2).mean() / N)
        self.assertLess(abs(mom.mean(s) - X.mean()), 4 * se_mean)
        self.assertLess(abs(mom.variance(s) - X.var()), 4 * se_var)
        # Power: without the CF term w_e y_e^2 the variance is rejected.
        self.assertGreater(abs(float(mom.d @ s ** 2) - X.var()), 10 * se_var)

    def test_independent_std_drops_the_shared_factor_covariance(self):
        s = np.ones(self.mom.mu.size)
        var_c = self.mom.d + self.mom.q_var @ self.mom.B_mean.multiply(self.mom.B_mean).toarray()
        self.assertAlmostEqual(self.mom.std_independent(s), np.sqrt(var_c @ s ** 2), places=12)
        no_cf = copy.copy(self.mom)
        no_cf.w, no_cf.cf_rows = np.zeros(0), np.zeros(0, dtype=int)
        no_cf.q_var = np.zeros_like(self.mom.q_var)
        self.assertAlmostEqual(no_cf.std(s), no_cf.std_independent(s), places=12)

    def test_summary(self):
        self.assertEqual(self.mom.summary(), {'processes': 6, 'processes_with_variance': 4, 'uncertain_cfs': 1})

    def test_moments_at_a_solved_instance(self):
        quiet(self.worker.solve, formulation='reduced')
        s = unc.current_scaling_vector(self.worker.instance)
        j = index(self.worker, self.worker.ammonia)
        self.assertAlmostEqual(s[j], 1.0, places=12)
        self.assertAlmostEqual(self.mom.mean(s), float(self.mom.mu @ s), places=12)

    def test_closed_form_table(self):
        table = unc.compute_closed_form_moments(self.data)
        n = sum(len(b[st]) for g in self.data.values() for b in g.values() for st in ('declared', 'undeclared'))
        self.assertEqual(sum(len(b) for g in table.values() for b in g.values()), n)


# ---------------------------------------------------------------------------
# risk budget and quantiles
# ---------------------------------------------------------------------------

class TestRiskBudget(unittest.TestCase):

    def test_equal_split_is_the_default(self):
        budget = cc.bonferroni_budget(0.95, 4)
        self.assertEqual(budget.weights, (0.25, 0.25, 0.25, 0.25))
        self.assertAlmostEqual(budget.epsilon, 0.05, places=15)
        self.assertAlmostEqual(budget.epsilon_at(1), 0.0125, places=15)
        self.assertAlmostEqual(budget.lambda_impact, 0.9875, places=15)

    def test_single_event_reproduces_the_individual_level(self):
        budget = cc.bonferroni_budget(0.95, 1)
        self.assertAlmostEqual(budget.lambda_impact, 0.95, places=15)

    def test_validation(self):
        with self.assertRaises(ValueError):
            cc.bonferroni_budget(0.95, 4, weights=(0.25, 0.25, 0.25, 0.30))
        with self.assertRaises(ValueError):
            cc.bonferroni_budget(0.95, 4, weights=(0.5, 0.5))
        with self.assertRaises(ValueError):
            cc.bonferroni_budget(0.95, 3, weights=(1.5, -0.25, -0.25))
        for bad in (-0.1, 1.0, 1.5):
            with self.assertRaises(ValueError):
                cc.bonferroni_budget(bad, 4)
        budget = cc.bonferroni_budget(0.95, 4, weights=(0.7, 0.1, 0.1, 0.1))
        self.assertAlmostEqual(budget.lambda_impact, 0.965, places=15)


class TestDeclaredQuantile(unittest.TestCase):

    def test_triangular_against_scipy(self):
        a, c, b = 0.1, 1.0, 1.1
        spec = {'uncertainty_type': 5, 'minimum': a, 'loc': c, 'maximum': b}
        dist = scipy.stats.triang((c - a) / (b - a), loc=a, scale=b - a)
        for p in (0.0, 1e-9, 0.001, 0.0125, 0.125, 0.5, 0.9, 0.95, 0.999, 1.0):
            self.assertAlmostEqual(cc.declared_quantile(spec, p), dist.ppf(p), delta=1e-12)

    def test_other_families_against_scipy(self):
        cases = [({'uncertainty_type': 3, 'loc': 5.0, 'scale': 2.0}, scipy.stats.norm(5.0, 2.0)),
                 ({'uncertainty_type': 4, 'minimum': 2.0, 'maximum': 6.0}, scipy.stats.uniform(2.0, 4.0)),
                 ({'uncertainty_type': 2, 'loc': 0.3, 'scale': 0.5}, scipy.stats.lognorm(0.5, scale=np.exp(0.3)))]
        for spec, dist in cases:
            for p in (0.01, 0.25, 0.5, 0.9):
                self.assertAlmostEqual(cc.declared_quantile(spec, p), dist.ppf(p), delta=1e-12)
        negative = {'uncertainty_type': 2, 'loc': 0.3, 'scale': 0.5, 'negative': True}
        mirrored = scipy.stats.lognorm(0.5, scale=np.exp(0.3))
        self.assertAlmostEqual(cc.declared_quantile(negative, 0.1), -mirrored.ppf(0.9), delta=1e-12)

    def test_exact_values_and_validation(self):
        self.assertEqual(cc.declared_quantile({'uncertainty_type': 0, 'amount': 4.0}, 0.3), 4.0)
        with self.assertRaises(ValueError):
            cc.declared_quantile({'uncertainty_type': 3, 'loc': 0.0, 'scale': 1.0}, 1.5)
        with self.assertRaises(NotImplementedError):
            cc.declared_quantile({'uncertainty_type': 7, 'loc': 0.0, 'scale': 1.0}, 0.5)


# ---------------------------------------------------------------------------
# the chance-constrained problem
# ---------------------------------------------------------------------------

try:
    import gurobipy  # noqa: F401
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False


class TestChanceConstrained(unittest.TestCase):

    LAMBDAS = sorted(V18_FREE)

    @classmethod
    def setUpClass(cls):
        cls.free = soc_worker()
        cls.capped = soc_worker(electrolysis_cap=0.035)
        cls.mom = unc.compute_moments(unc.import_declared(cls.free), cls.free)

    def front(self, worker, capped, **kwargs):
        upper = {worker.elyz: TRI_CAP} if capped else None
        return unc.ChanceConstrained(worker, self.mom, upper_bounds=upper, **kwargs).solve(self.LAMBDAS)

    def test_reproduces_pulpo_1_8(self):
        for capped, reference, worker in ((False, V18_FREE, self.free), (True, V18_CAP, self.capped)):
            front = self.front(worker, capped)
            for lam, (adjusted, electrolysis) in reference.items():
                with self.subTest(capped=capped, lam=lam):
                    self.assertLess(abs(front[lam].adjusted - adjusted), 1e-6 * adjusted)
                    if capped:
                        # The cap binds: the decision is pinned and comparable.
                        self.assertAlmostEqual(front[lam].s[index(worker, worker.elyz)], electrolysis, delta=1e-8)

    def test_matches_an_independent_full_space_cone(self):
        for capped, worker in ((False, self.free), (True, self.capped)):
            front = self.front(worker, capped)
            ref = full_space_front(worker, self.mom, self.LAMBDAS, TRI_CAP if capped else None, K=2 if capped else 1)
            for lam in self.LAMBDAS:
                with self.subTest(capped=capped, lam=lam):
                    self.assertLess(abs(front[lam].adjusted - ref[lam][0]), 1e-7 * ref[lam][0])
                    # At a smooth optimum the objective is flat, so a gap of 1e-10
                    # pins the decision only to about sqrt(1e-10) relative.
                    np.testing.assert_allclose(front[lam].s, ref[lam][1], rtol=1e-4, atol=1e-8)

    def test_point_contents(self):
        front = self.front(self.capped, True)
        point = front[0.9]
        self.assertAlmostEqual(point.lambda_impact, 0.95, places=15)
        self.assertAlmostEqual(point.kappa, scipy.stats.norm.ppf(0.95), places=12)
        self.assertAlmostEqual(point.adjusted, point.mean + point.kappa * point.sigma, places=12)
        j = index(self.capped, self.capped.elyz)
        self.assertAlmostEqual(point.bounds[('upper', j)], cc.declared_quantile(TRI_CAP, 0.05), places=14)
        self.assertAlmostEqual(point.epsilon[('upper', j)], 0.05, places=14)   # keyed like bounds
        self.assertLess(point.balance_residual, 1e-12)
        table = front.table()
        self.assertEqual(list(table.index), self.LAMBDAS)

    @unittest.skipUnless(HAS_GUROBI, "gurobipy is not installed")
    def test_gurobi_agrees_with_clarabel(self):
        a = self.front(self.capped, True)
        b = unc.ChanceConstrained(self.capped, self.mom, upper_bounds={self.capped.elyz: TRI_CAP}).solve(
            self.LAMBDAS, solver_name='gurobi')
        for lam in self.LAMBDAS:
            self.assertLess(abs(a[lam].adjusted - b[lam].adjusted), 1e-8)

    def test_gram_factor_agrees_with_qr(self):
        a = self.front(self.free, False)
        original = cc.QR_ENTRIES
        cc.QR_ENTRIES = 0
        try:
            b = self.front(self.free, False)
        finally:
            cc.QR_ENTRIES = original
        for lam in self.LAMBDAS:
            self.assertLess(abs(a[lam].adjusted - b[lam].adjusted), 1e-8)

    def test_scaled_instance_gives_the_same_front(self):
        scaled = soc_worker(electrolysis_cap=0.035, scale=True)
        a = self.front(self.capped, True)
        b = self.front(scaled, True)
        for lam in self.LAMBDAS:
            self.assertLess(abs(a[lam].adjusted - b[lam].adjusted), 1e-9)

    def test_individual_allocation(self):
        ccp = unc.ChanceConstrained(self.capped, self.mom, upper_bounds={self.capped.elyz: TRI_CAP},
                                    allocation='individual')
        lam_z, eps = ccp.levels(0.9)
        self.assertEqual(lam_z, 0.9)
        self.assertAlmostEqual(list(eps.values())[0], 0.1, places=15)
        point = ccp.solve_point(0.9)
        self.assertAlmostEqual(point.s[index(self.capped, self.capped.elyz)],
                               cc.declared_quantile(TRI_CAP, 0.1), delta=1e-8)

    def test_events_weights_and_levels(self):
        ccp = unc.ChanceConstrained(self.capped, self.mom, upper_bounds={self.capped.elyz: TRI_CAP},
                                    lower_bounds={self.capped.smr: {'uncertainty_type': 4, 'minimum': 0.0,
                                                                    'maximum': 0.05}},
                                    weights=(0.5, 0.25, 0.25))
        self.assertEqual(ccp.K, 3)
        # Lower bounds first, then upper bounds (PULPO 1.8.0's order).
        self.assertEqual(ccp.events, [('lower', index(self.capped, self.capped.smr)),
                                      ('upper', index(self.capped, self.capped.elyz))])
        lam_z, eps = ccp.levels(0.9)
        self.assertAlmostEqual(lam_z, 0.95, places=15)
        bounds = ccp.bounds(0.9)
        j_smr = index(self.capped, self.capped.smr)
        self.assertAlmostEqual(bounds[('lower', j_smr)], 0.05 * (1 - 0.025), places=14)
        point = ccp.solve_point(0.9)
        self.assertGreaterEqual(point.s[j_smr], bounds[('lower', j_smr)] - 1e-9)
        with self.assertRaises(ValueError):
            unc.ChanceConstrained(self.capped, self.mom, upper_bounds={self.capped.elyz: TRI_CAP},
                                  weights=(0.5, 0.5, 0.0))

    def test_weights_follow_the_event_order(self):
        uni = lambda a, b: {'uncertainty_type': 4, 'minimum': a, 'maximum': b}
        j_smr, j_ely = index(self.free, self.free.smr), index(self.free, self.free.elyz)
        ccp = unc.ChanceConstrained(self.free, self.mom, upper_bounds={self.free.smr: uni(0.05, 0.15)},
                                    lower_bounds={self.free.elyz: uni(0.0, 0.1)}, weights=(0.5, 0.4, 0.1))
        bounds = ccp.bounds(0.9)
        self.assertAlmostEqual(bounds[('lower', j_ely)], 0.1 * (1 - 0.4 * 0.1), places=14)
        self.assertAlmostEqual(bounds[('upper', j_smr)], 0.05 + 0.1 * (0.1 * 0.1), places=14)

    def test_certain_events_are_refused(self):
        normal = {'uncertainty_type': 3, 'loc': 0.03, 'scale': 0.01}
        with self.assertRaises(ValueError):
            unc.ChanceConstrained(self.capped, self.mom, upper_bounds={self.capped.elyz: normal}, weights=(1.0, 0.0))
        with self.assertRaises(ValueError):
            cc.bonferroni_budget(0.9, 2, weights=(0.0, 1.0))

    def test_large_finite_limits_that_cannot_bind(self):
        """Capacities of 1e10 and default limits of 1e15 stand for "unlimited";
        they must not degrade the cone solve."""
        defaults = {'lower_bound': -1e15, 'upper_bound': 1e15, 'upper_inv_bound': 1e15, 'lower_inv_bound': -1e15,
                    'lower_imp_bound': -1e15, 'upper_imp_bound': 1e15}
        for label, worker in (('1e10 caps', soc_worker(electrolysis_cap=1e10)),
                              ('1e15 defaults', soc_worker(default_limits=defaults))):
            for solver in ['clarabel'] + (['gurobi'] if HAS_GUROBI else []):
                with self.subTest(case=label, solver=solver):
                    front = unc.ChanceConstrained(worker, self.mom).solve(self.LAMBDAS, solver_name=solver)
                    for lam, (adjusted, _) in V18_FREE.items():
                        self.assertLess(abs(front[lam].adjusted - adjusted), 1e-6 * adjusted)
                        self.assertEqual(front[lam].rounds, 1)

    def test_a_large_limit_that_binds_is_still_imposed(self):
        """A remote bound is withheld only until a solution violates it."""
        free = self.front(self.free, False)
        mean = free[0.9].mean
        worker = soc_worker(upper_imp_limit={SOC_METHOD: mean - 0.05})
        scaled_limit = {SOC_METHOD: (mean - 0.05)}
        point = unc.ChanceConstrained(worker, self.mom).solve_point(0.9)
        self.assertLessEqual(point.mean, scaled_limit[SOC_METHOD] + 1e-9)

    def test_reinstantiation_is_followed(self):
        worker = soc_worker()
        ccp = unc.ChanceConstrained(worker, self.mom)
        first = ccp.solve_point(0.9)
        quiet(worker.instantiate, choices={'hydrogen': {worker.smr: float('inf'), worker.elyz: 0.05}},
              demand={worker.ammonia: 2})
        again = ccp.solve_point(0.9)
        fresh = unc.ChanceConstrained(worker, self.mom).solve_point(0.9)
        self.assertNotAlmostEqual(again.adjusted, first.adjusted, places=3)
        self.assertAlmostEqual(again.adjusted, fresh.adjusted, places=9)
        ccp.write(again)
        self.assertAlmostEqual(worker.instance.scaling_vector[index(worker, worker.ammonia)].value, 2.0, places=12)

    def test_levels_below_one_half_are_refused(self):
        ccp = unc.ChanceConstrained(self.free, self.mom)
        with self.assertRaises(ValueError):
            ccp.solve_point(0.3)

    def test_goal_objective_is_refused(self):
        worker = soc_worker(objective='goal', imp_goals={SOC_METHOD: 1.0})
        with self.assertRaises(NotImplementedError):
            unc.ChanceConstrained(worker, self.mom)

    def test_limit_on_the_uncertain_impact_applies_to_its_mean(self):
        free = self.front(self.free, False)
        limit = (free[0.9].mean + free[0.5].mean) / 2
        worker = soc_worker(upper_imp_limit={SOC_METHOD: limit})
        point = unc.ChanceConstrained(worker, self.mom).solve_point(0.9)
        self.assertLessEqual(point.mean, limit + 1e-9)
        self.assertGreater(point.adjusted, free[0.9].adjusted)

    def test_infeasible_level_raises(self):
        impossible = {'uncertainty_type': 4, 'minimum': 2.0, 'maximum': 3.0}   # SMR >= 2 > demand
        ccp = unc.ChanceConstrained(self.free, self.mom, lower_bounds={self.free.smr: impossible})
        with self.assertRaises(unc.ChanceConstrainedError):
            ccp.solve_point(0.9)

    def test_write_puts_the_point_on_the_instance(self):
        ccp = unc.ChanceConstrained(self.capped, self.mom, upper_bounds={self.capped.elyz: TRI_CAP})
        point = ccp.solve_point(0.95)
        ccp.write(point)
        j = index(self.capped, self.capped.elyz)
        self.assertAlmostEqual(self.capped.instance.scaling_vector[j].value, point.s[j], places=14)
        self.assertIn('Scaling Vector', quiet(self.capped.extract_results))

    def test_facade(self):
        worker = soc_worker(electrolysis_cap=0.035, worker_class=pulpo_unc.PulpoOptimizerUnc)
        with self.assertRaises(ValueError):
            worker.moments()
        worker.import_uncertainty_data()
        worker.apply_expert_knowledge('Cf', SOC_METHOD, {2: {'uncertainty_type': 3, 'loc': 273.0, 'scale': 0.0}})
        front = worker.chance_constrained(upper_bounds={worker.elyz: TRI_CAP}).solve([0.9])
        self.assertLess(abs(front[0.9].adjusted - V18_CAP[0.9][0]), 1e-6)

    def test_facade_analyses_match_the_functions(self):
        """The façade's screen_undeclared and validate give what the functions give,
        and refuse to run before the uncertainty data is imported."""
        worker = soc_worker(electrolysis_cap=0.035, worker_class=pulpo_unc.PulpoOptimizerUnc)
        for call in (lambda: worker.screen_undeclared(None, exact_cfs=[0]),
                     lambda: worker.validate(None, None, n=10)):
            with self.assertRaisesRegex(ValueError, 'import_uncertainty_data'):
                call()
        data = worker.import_uncertainty_data()
        problem = worker.chance_constrained(upper_bounds={worker.elyz: TRI_CAP})
        front = problem.solve([0.5, 0.9])
        exact_cfs = unc.co2_flows(worker)
        screening = worker.screen_undeclared(front[0.9], exact_cfs=exact_cfs)
        reference = unc.screen_undeclared(front[0.9], data, worker, exact_cfs=exact_cfs)
        pd.testing.assert_frame_equal(screening.sigma, reference.sigma)
        pd.testing.assert_frame_equal(screening.ranking, reference.ranking)
        validation = worker.validate(front, problem, n=5_000, seed=1)
        pd.testing.assert_frame_equal(validation.table, unc.validate(front, problem, data, n=5_000, seed=1).table)

    def test_input_errors(self):
        """Each invalid input to the chance-constrained problem raises a clear error."""
        worker = self.capped
        j = index(worker, worker.elyz)
        build = lambda **kw: unc.ChanceConstrained(worker, self.mom, **kw)
        capped = dict(upper_bounds={worker.elyz: TRI_CAP})
        unbounded = {'uncertainty_type': 3, 'loc': 0.03, 'scale': float('inf')}
        cases = [
            ('lambda 0', lambda: build(**capped).solve([0.0]), ValueError, 'lambda must lie in'),
            ('lambda 1', lambda: build(**capped).solve([1.0]), ValueError, 'lambda must lie in'),
            ('lambda 1.2', lambda: build(**capped).solve([1.2]), ValueError, 'lambda must lie in'),
            ('impact level below 1/2', lambda: build(allocation='individual').solve([0.3]), ValueError,
             'non-convex'),
            ('unknown solver', lambda: build(**capped).solve([0.9], solver_name='cplex'), ValueError,
             "'clarabel' or 'gurobi'"),
            ('unknown allocation', lambda: build(allocation='banana'), ValueError, 'allocation must be'),
            ('weights with individual', lambda: build(allocation='individual', weights=(0.5, 0.5), **capped),
             ValueError, 'Bonferroni allocation only'),
            ('weights not summing to 1', lambda: build(weights=(0.5, 0.2), **capped), ValueError, 'sum to 1'),
            ('the same bound twice', lambda: build(upper_bounds={worker.elyz: TRI_CAP, j: TRI_CAP}), ValueError,
             'same uncertain bound twice'),
            ('an infinite quantile', lambda: build(upper_bounds={worker.elyz: unbounded}).solve([0.9]), ValueError,
             'cannot hold'),
        ]
        for label, call, error, message in cases:
            with self.subTest(label):
                with self.assertRaisesRegex(error, message):
                    call()

    def test_the_goal_objective_is_refused(self):
        worker = soc_worker(imp_goals={SOC_METHOD: 1.0}, objective='goal')
        with self.assertRaisesRegex(NotImplementedError, 'goal objective'):
            unc.ChanceConstrained(worker, self.mom)

    def test_facade_with_two_methods(self):
        """A worker with a second method (to limit, say) imports the impact it names."""
        air = "('my project', 'air quality')"
        worker = pulpo_unc.PulpoOptimizerUnc(PROJECT, DATABASES, {CLIMATE_KEY: 1, air: 0})
        quiet(worker.get_lci_data)
        with self.assertRaises(ValueError) as error:
            worker.import_uncertainty_data()
        self.assertIn('method=', str(error.exception))
        data = worker.import_uncertainty_data(method=CLIMATE_KEY)
        single = pulpo.PulpoOptimizer(PROJECT, DATABASES, {CLIMATE_KEY: 1})
        quiet(single.get_lci_data)
        np.testing.assert_equal(data, unc.import_declared(single))      # NaN fields compare equal


class Activity:
    """Stands in for a Brightway activity: hashes and compares like its key."""

    def __init__(self, key):
        self.key = key

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.key == getattr(other, 'key', other)


class TestFacilityChanceConstrained(unittest.TestCase):
    """An ill-conditioned system with a known optimum.

    Route D makes the demanded product and consumes 1e-10 facilities, each of
    which consumes 1e11 units of E; route A makes the same product at a certain
    impact of 11. Coefficients span 1e-10 .. 1e11, as in ecoinvent, and the
    facility's impact carries a variance of 1e20 per facility.
    """

    LAMBDA = 0.9

    def problem(self, scale):
        A = sp.csr_matrix(np.array([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, -1e11, 0.0],
                                    [-1e-10, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]]))
        B = sp.csr_matrix(np.array([[0.0, 1.0, 0.0, 11.0]]))
        acts = [Activity(('db', name)) for name in ('D', 'E', 'F', 'A')]
        lci = {'technology_matrix': A, 'intervention_matrix': B, 'matrices': {'h': sp.identity(1, format='csr')},
               'process_map': {a.key: j for j, a in enumerate(acts)}, 'intervention_map': {('bio', 'x'): 0}}
        choices = {'product': {acts[0]: float('inf'), acts[3]: float('inf')}}
        data = converter.combine_inputs(lci, {'product': 1.0}, choices, {}, {}, {}, {}, {}, {}, {'h': 1},
                                        scale=scale)
        worker = types.new_class('Worker')()
        worker.instance = quiet(optimizer.instantiate, data)
        worker.lci_data, worker.choices = lci, choices
        mom = moments_module.Moments(method='h', mu=np.array([0.0, 1.0, 0.0, 11.0]),
                                     d=np.array([0.01, 0.04, 1e20, 0.25]), w=np.zeros(0),
                                     cf_rows=np.zeros(0, dtype=int), B_mean=B, B_var=sp.csr_matrix(B.shape),
                                     q_mean=np.ones(1), q_var=np.zeros(1))
        return worker, mom

    def analytic(self, mom):
        z = scipy.stats.norm.ppf(self.LAMBDA)

        def objective(a):
            x = np.array([a, 10 * a, 1e-10 * a, 1 - a])
            return float(mom.mu @ x + z * np.sqrt(mom.d @ x ** 2))

        res = scipy.optimize.minimize_scalar(objective, bounds=(0.0, 1.0), method='bounded',
                                             options={'xatol': 1e-12})
        return res.fun, res.x

    def test_reaches_the_analytic_optimum(self):
        import scipy.optimize  # noqa: F401
        for scale in (False, True):
            with self.subTest(scale=scale):
                worker, mom = self.problem(scale)
                point = unc.ChanceConstrained(worker, mom).solve_point(self.LAMBDA)
                best, a_star = self.analytic(mom)
                self.assertAlmostEqual(point.adjusted / best, 1.0, places=9)
                self.assertAlmostEqual(point.s[0], a_star, places=5)
                self.assertAlmostEqual(point.s[2], 1e-10 * point.s[0], delta=1e-22)   # facility balance


class TestDrawUncertaintySampleSeeding(unittest.TestCase):
    """``draw_uncertainty_sample(seed=...)`` seeds every family, not only Normal."""

    @classmethod
    def setUpClass(cls):
        cls.data = unc.import_declared(soc_worker())

    def test_fixture_contains_non_normal_parameters(self):
        families = {spec['uncertainty_type'] for g in self.data.values() for b in g.values()
                    for spec in b['declared'].values()}
        self.assertTrue(families - {stats_arrays.NormalUncertainty.id})

    def test_same_seed_reproduces_despite_global_rng_use(self):
        first = processor.draw_uncertainty_sample(self.data, SOC_METHOD, seed=123)
        np.random.seed(999)
        np.random.random(17)
        second = processor.draw_uncertainty_sample(self.data, SOC_METHOD, seed=123)
        self.assertEqual(first['If'], second['If'])
        self.assertEqual(first['Cf'], second['Cf'])
        third = processor.draw_uncertainty_sample(self.data, SOC_METHOD, seed=124)
        self.assertNotEqual(first['If'], third['If'])


class TestApplyCCFormulation(unittest.TestCase):

    def test_writes_exact_quantiles_and_checks_K(self):
        worker = soc_worker(electrolysis_cap=0.035)
        j = index(worker, worker.elyz)
        budget = cc.bonferroni_budget(0.9, 2)
        cc.apply_CC_formulation(worker.instance, budget, upper_bounds={j: TRI_CAP})
        self.assertAlmostEqual(pyo.value(worker.instance.UPPER_LIMIT[j]),
                               cc.declared_quantile(TRI_CAP, 0.05), places=14)
        with self.assertRaises(ValueError):
            cc.apply_CC_formulation(worker.instance, cc.bonferroni_budget(0.9, 3), upper_bounds={j: TRI_CAP})


# ---------------------------------------------------------------------------
# bw25 extraction of uncertainty parameters
# ---------------------------------------------------------------------------

def setup_uncertainty_free_project():
    """A throwaway project with one CF and two databases of one process each: in
    ``no_uncertainty_db`` nothing is uncertain, in ``declared_db`` the CO2
    emission is lognormal. Only ``declared_db`` declares a distribution."""
    project = "sample_project_no_uncertainty"
    bd.projects.set_current(project)
    for db_name in ("no_uncertainty_db", "declared_db", "biosphere3"):
        if db_name in bd.databases:
            del bd.databases[db_name]
    co2 = ("biosphere3", "CO2")
    bd.Database("biosphere3").write({
        co2: {"name": "Carbon dioxide, fossil", "categories": ("climate change",),
              "type": "emission", "unit": "kg"},
    })
    bd.Database("no_uncertainty_db").write({
        ("no_uncertainty_db", "process"): {
            "name": "process", "unit": "kg", "location": "GLO", "reference product": "widget",
            "exchanges": [
                {"input": ("no_uncertainty_db", "process"), "amount": 1.0, "type": "production"},
                {"input": co2, "amount": 2.0, "type": "biosphere"},
            ],
        },
    })
    bd.Database("declared_db").write({
        ("declared_db", "process"): {
            "name": "declared process", "unit": "kg", "location": "GLO", "reference product": "gadget",
            "exchanges": [
                {"input": ("declared_db", "process"), "amount": 1.0, "type": "production"},
                {"input": co2, "amount": 3.0, "type": "biosphere", "uncertainty type": 2,
                 "loc": float(np.log(3.0)), "scale": 0.1},
            ],
        },
    })
    for method in list(bd.methods):
        bd.Method(method).deregister()
    method = bd.Method(("my project", "climate change"))
    method.register(unit="kg CO2eq")
    method.write([(co2, 1.0)])
    return project


@unittest.skipUnless(is_bw25(), "bw25-only: structured uncertainty-parameter arrays require bw2data >= 4")
class TestUncertaintyParamArrays(unittest.TestCase):
    """``bw_parser.import_data`` exposes structured uncertainty arrays, also for
    datapackages that declare no distribution (whose entries are then listed as
    deterministic)."""

    REQUIRED_FIELDS = ("row", "col", "amount", "uncertainty_type",
                       "loc", "scale", "shape", "minimum", "maximum", "negative")

    def test_with_uncertainty(self):
        lci_data = bw_parser.import_data(project=PROJECT, databases=DATABASES, method=CLIMATE_KEY,
                                         intervention_matrix_name="biosphere3", seed=42)
        method_key = next(iter(lci_data["matrices"]))
        int_params = lci_data["intervention_params"]
        cf_params = lci_data["characterization_params"][method_key]
        for name, arr in (("intervention_params", int_params), ("characterization_params", cf_params)):
            self.assertIsNotNone(arr.dtype.names, f"{name} must be a structured array")
            for f in self.REQUIRED_FIELDS:
                self.assertIn(f, arr.dtype.names, f"{name} missing field '{f}'")
        self.assertTrue((int_params["uncertainty_type"] > 0).any())
        self.assertTrue((cf_params["uncertainty_type"] > 0).any())
        self.assertLess(int_params["row"].max(), 10_000)
        self.assertLess(int_params["col"].max(), 10_000)
        pairs = list(zip(int_params["row"].tolist(), int_params["col"].tolist()))
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertGreater(len(set(int_params["col"].tolist())), 1)

    def test_without_uncertainty_lists_deterministic_entries(self):
        project = setup_uncertainty_free_project()
        lci_data = bw_parser.import_data(project=project, databases=["no_uncertainty_db"],
                                         method=CLIMATE_KEY, intervention_matrix_name="biosphere3", seed=42)
        method_key = next(iter(lci_data["matrices"]))
        for arr in (lci_data["intervention_params"], lci_data["characterization_params"][method_key]):
            self.assertEqual(len(arr), 1)
            self.assertEqual(int(arr["uncertainty_type"][0]), 0)
            self.assertEqual(arr["loc"][0], arr["amount"][0])
            self.assertTrue(np.isnan(arr["scale"][0]))


class TestUndeclaredData(unittest.TestCase):
    """Parameters without a distribution are imported as undeclared, on both
    Brightway stacks. A database or LCIA method that declares none (on bw25 its
    datapackage then carries no distributions at all) hides nothing else."""

    @classmethod
    def setUpClass(cls):
        cls.project = setup_uncertainty_free_project()

    def import_declared(self, databases):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lci_data = bw_parser.import_data(project=self.project, databases=databases, method=CLIMATE_KEY,
                                             intervention_matrix_name="biosphere3", seed=42)
        self.assertFalse([w for w in caught if 'uncertainty' in str(w.message)])
        method_key = next(iter(lci_data["matrices"]))
        worker = types.SimpleNamespace(lci_data=lci_data, database=databases, method={method_key: 1})
        return unc.import_declared(worker), method_key

    def test_nothing_declared(self):
        data, method = self.import_declared(["no_uncertainty_db"])
        block, cfs = data['If']['no_uncertainty_db'], data['Cf'][method]
        self.assertEqual((block['declared'], cfs['declared']), ({}, {}))
        self.assertEqual([spec['amount'] for spec in block['undeclared'].values()], [2.0])
        self.assertEqual([spec['amount'] for spec in cfs['undeclared'].values()], [1.0])

    def test_a_database_without_distributions_hides_no_other(self):
        data, method = self.import_declared(["no_uncertainty_db", "declared_db"])
        [spec] = data['If']['declared_db']['declared'].values()
        self.assertEqual((spec['uncertainty_type'], spec['amount']), (2, 3.0))
        self.assertAlmostEqual(spec['scale'], 0.1)
        self.assertEqual(data['If']['no_uncertainty_db']['declared'], {})
        self.assertEqual(len(data['If']['no_uncertainty_db']['undeclared']), 1)
        self.assertEqual(len(data['Cf'][method]['undeclared']), 1)


if __name__ == '__main__':
    unittest.main()
