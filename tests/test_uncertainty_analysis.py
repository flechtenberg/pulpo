"""Tests for the analyses on a solved chance-constrained front.

- the exact variance decomposition (``decompose``): indices sum to one, agree
  with the closed-form variance, and fall inside SALib's confidence intervals
  (SALib is a test-only dependency);
- the screening of undeclared parameters: the ranking does not depend on the
  width, CO2 CFs never receive one, widened data keep their means;
- the width sensitivity and its bound on the re-solved optimum;
- the diagnostics of a front;
- the vectorized sampler, the reduced-space projections as a building block,
  and the out-of-sample validation, whose coverage is exact when ``X`` is
  exactly normal.
"""

import copy
import unittest

import numpy as np
import scipy.stats
import stats_arrays

from pulpo.utils import uncertainty as unc
from pulpo.utils.uncertainty import cc, validation
from tests.test_uncertainty import SOC_METHOD, TRI_CAP, index, soc_worker

try:
    from SALib.analyze import sobol as salib_analyze
    from SALib.sample import sobol as salib_sample
except ImportError:                                  # pragma: no cover - test-only extra
    salib_analyze = salib_sample = None

LAMBDAS = [0.5, 0.9, 0.99]


def solved(capped=True):
    """A worker, its data and moments, the problem and its front."""
    worker = soc_worker()
    data = unc.import_declared(worker)
    mom = unc.compute_moments(data, worker)
    problem = unc.ChanceConstrained(worker, mom, upper_bounds={worker.elyz: TRI_CAP} if capped else None)
    return worker, data, mom, problem, problem.solve(LAMBDAS)


def ppf(spec, u):
    """Inverse CDF of a declared family, written independently of ``validate``."""
    utype = int(spec['uncertainty_type'])
    if utype == stats_arrays.NormalUncertainty.id:
        return scipy.stats.norm.ppf(u, spec['loc'], spec['scale'])
    if utype == stats_arrays.LognormalUncertainty.id:
        x = np.exp(spec['loc'] + spec['scale'] * scipy.stats.norm.ppf(u))
        return -x if spec.get('negative', False) else x
    if utype == stats_arrays.UniformUncertainty.id:
        return spec['minimum'] + (spec['maximum'] - spec['minimum']) * u
    a, c, b = spec['minimum'], spec['loc'], spec['maximum']
    return scipy.stats.triang.ppf(u, (c - a) / (b - a), loc=a, scale=b - a)


def declared_specs(data):
    return {('If', index_): spec for block in data['If'].values() for index_, spec in block['defined'].items()} | \
        {('Cf', e): spec for block in data['Cf'].values() for e, spec in block['defined'].items()}


# ---------------------------------------------------------------------------
# decomposition
# ---------------------------------------------------------------------------

class TestDecompose(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.worker, cls.data, cls.mom, cls.problem, cls.front = solved()
        cls.point = cls.front[0.9]

    def test_indices_sum_to_one_and_follow_the_closed_form(self):
        dec = unc.decompose(self.point, self.mom)
        self.assertAlmostEqual(dec.variance, self.mom.variance(self.point.s), places=15)
        self.assertAlmostEqual(dec.parameters['S1'].sum() + dec.interactions['S2'].sum(), 1.0, places=12)
        # ST = S1 + the second-order terms the parameter takes part in.
        params = dec.parameters.set_index(['group', 'flow', 'process'])
        for (e, j), s2 in dec.interactions.set_index(['flow', 'process'])['S2'].items():
            self.assertAlmostEqual(params.loc[('If', e, j), 'ST'] - params.loc[('If', e, j), 'S1'], s2, places=14)
        for e, row in params.xs('Cf', level='group').iterrows():
            pairs = dec.interactions[dec.interactions['flow'] == e[0]]['S2'].sum()
            self.assertAlmostEqual(row['ST'] - row['S1'], pairs, places=14)
        # Every B entry's index from first principles.
        s, V = self.point.s, dec.variance
        for (g, e, j), row in params.iterrows():
            if g == 'If':
                var_b = self.mom.B_var[e, j]
                self.assertAlmostEqual(row['S1'], self.mom.q_mean[e] ** 2 * var_b * s[j] ** 2 / V, places=14)
        self.assertTrue((np.diff(dec.parameters['ST'].to_numpy()) <= 0).all())

    def test_families_sum_their_members(self):
        dec = unc.decompose(self.point, self.mom, unc.families_by_database(self.worker))
        families = dec.families()
        self.assertEqual(set(families.index), {'soc_demo_background_db', 'soc_demo_foreground_db', 'Cf'})
        self.assertAlmostEqual(families['S1'].sum() + dec.interactions['S2'].sum(), 1.0, places=12)
        foreground = dec.parameters[dec.parameters['family'] == 'soc_demo_foreground_db']
        self.assertTrue((foreground['process'] == index(self.worker, self.worker.ammonia)).all())
        # Without CF uncertainty the CF family vanishes and nothing interacts.
        no_cf = copy.copy(self.mom)
        no_cf.q_var = np.zeros_like(self.mom.q_var)
        no_cf.w, no_cf.cf_rows = no_cf.w[:0], no_cf.cf_rows[:0]
        dec0 = unc.decompose(self.point, no_cf)
        self.assertNotIn('Cf', set(dec0.parameters['family']))
        self.assertTrue(dec0.interactions.empty)
        np.testing.assert_allclose(dec0.parameters['S1'], dec0.parameters['ST'])

    def test_zero_variance_raises(self):
        with self.assertRaises(ValueError):
            unc.decompose(np.zeros_like(self.point.s), self.mom)

    @unittest.skipIf(salib_sample is None, "SALib is not installed (pip install pulpo-dev[test])")
    def test_against_salib(self):
        s = self.point.s
        dec = unc.decompose(s, self.mom)
        params = list(zip(dec.parameters['group'], dec.parameters['flow'], dec.parameters['process']))
        specs = declared_specs(self.data)
        spec_of = [specs[('If', (e, j))] if g == 'If' else specs[('Cf', e)] for g, e, j in params]
        # names as an array: SALib 1.5.1 passes them to pd.unique, which refuses lists under pandas 3.
        problem = {'num_vars': len(params), 'names': np.array([str(p) for p in params], dtype=object),
                   'bounds': [[0.0, 1.0]] * len(params)}
        U = salib_sample.sample(problem, 2 ** 14, calc_second_order=True, seed=11)
        U = np.clip(U, 1e-12, 1.0 - 1e-12)
        values = np.column_stack([ppf(spec, U[:, k]) for k, spec in enumerate(spec_of)])
        y_mean = np.asarray(self.mom.B_mean @ s).ravel()
        q = np.tile(self.mom.q_mean, (len(U), 1))
        y = np.tile(y_mean, (len(U), 1))
        for k, (g, e, j) in enumerate(params):
            if g == 'Cf':
                q[:, e] = values[:, k]
            else:
                y[:, e] += (values[:, k] - self.mom.B_mean[e, j]) * s[j]
        Y = (q * y).sum(axis=1)
        result = salib_analyze.analyze(problem, Y, calc_second_order=True, seed=11)
        for k in range(len(params)):
            for name in ('S1', 'ST'):
                ours = dec.parameters[name].iloc[k]
                self.assertLessEqual(abs(result[name][k] - ours), result[f'{name}_conf'][k],
                                     f"{name} of {params[k]}: SALib {result[name][k]:.5f} "
                                     f"+- {result[f'{name}_conf'][k]:.5f}, exact {ours:.5f}")


# ---------------------------------------------------------------------------
# undeclared parameters
# ---------------------------------------------------------------------------

class TestScreening(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.worker, cls.data, cls.mom, cls.problem, cls.front = solved()
        cls.point = cls.front[0.9]
        # More undeclared parameters: the background's B entries and the CO2 CF.
        cls.wide = copy.deepcopy(cls.data)
        unc.override(cls.wide, 'If', 'soc_demo_background_db',
                     {i: {'uncertainty_type': 0} for i in list(cls.wide['If']['soc_demo_background_db']['defined'])
                      if i[0] != 1})
        unc.override(cls.wide, 'Cf', SOC_METHOD, {0: {'uncertainty_type': 0}})

    def test_co2_flows_are_found_by_name(self):
        np.testing.assert_array_equal(unc.co2_flows(self.worker), [0])

    def test_widen_keeps_means_and_sets_the_width(self):
        r = 0.3
        widened = unc.widen(self.wide, r, exact_cfs=[0])
        before = unc.compute_moments(self.wide, self.worker)
        after = unc.compute_moments(widened, self.worker)
        np.testing.assert_allclose(after.mu, before.mu, rtol=1e-14)
        for db, block in self.wide['If'].items():
            for (e, j), spec in block['undefined'].items():
                self.assertAlmostEqual(after.B_var[e, j] / spec['amount'] ** 2, r * r, places=13)
                self.assertNotIn((e, j), widened['If'][db]['undefined'])
        # The declared parameters are untouched.
        self.assertEqual(widened['If']['soc_demo_background_db']['defined'][(1, 0)],
                         self.wide['If']['soc_demo_background_db']['defined'][(1, 0)])
        # Per subgroup: only the named ones widen.
        only_cf = unc.compute_moments(unc.widen(self.wide, {'Cf': r}, exact_cfs=[0]), self.worker)
        np.testing.assert_array_equal(only_cf.B_var.toarray(), before.B_var.toarray())
        self.assertAlmostEqual(only_cf.q_var[2], (r * 273.0) ** 2, places=8)
        # Every width must name a subgroup, and the exact CFs must be stated.
        with self.assertRaises(KeyError):
            unc.widen(self.wide, {'background': r}, exact_cfs=[0])
        with self.assertRaises(TypeError):
            unc.widen(self.wide, r)
        # A negative amount: a mirrored lognormal with the same mean and CV.
        mean, var = unc.spec_moments(unc.lognormal_with_cv(-2.0, r))
        self.assertAlmostEqual(mean, -2.0, places=14)
        self.assertAlmostEqual(var, (r * 2.0) ** 2, places=14)

    def test_co2_cfs_never_receive_a_width(self):
        for r in (0.1, 0.3, 5.0):
            widened = unc.widen(self.wide, r, exact_cfs=unc.co2_flows(self.worker))
            self.assertIn(0, widened['Cf'][SOC_METHOD]['undefined'])
            self.assertEqual(unc.compute_moments(widened, self.worker).q_var[0], 0.0)
            scr = unc.screen_undeclared(self.point, self.wide, self.worker, exact_cfs=[0], r=(r,))
            self.assertFalse(((scr.ranking['group'] == 'Cf') & (scr.ranking['flow'] == 0)).any())
            self.assertFalse(((scr.top[r]['group'] == 'Cf') & (scr.top[r]['flow'] == 0)).any())
        # Only because they are held exact: without that the CO2 CF would widen.
        unheld = unc.widen(self.wide, 0.3, exact_cfs=())
        self.assertGreater(unc.compute_moments(unheld, self.worker).q_var[0], 0.0)

    def test_ranking_does_not_depend_on_the_width(self):
        rankings = [unc.screen_undeclared(self.point, self.wide, self.worker, exact_cfs=[0], r=(r,)).ranking
                    for r in (0.05, 0.1, 0.3, 1.0)]
        keys = ['group', 'subgroup', 'flow', 'process', 'amount', 'contribution']
        for other in rankings[1:]:
            self.assertTrue(rankings[0][keys].equals(other[keys]))
        self.assertGreaterEqual(len(rankings[0]), 6)
        # Among B entries on a flow with an exact CF, the total-order index
        # follows the ranking at every width.
        scr = unc.screen_undeclared(self.point, self.wide, self.worker, exact_cfs=[0], r=(0.1, 0.3))
        exact_cf = scr.ranking[(scr.ranking['group'] == 'If') & (scr.ranking['flow'] == 0)]
        self.assertGreaterEqual(len(exact_cf), 3)
        for column in ('ST r=0.1', 'ST r=0.3'):
            self.assertTrue((np.diff(exact_cf[column].to_numpy()) <= 0).all())

    def test_sigma_and_shares(self):
        scr = unc.screen_undeclared(self.point, self.wide, self.worker, exact_cfs=[0], r=(0.1, 0.3), n=5)
        declared = unc.compute_moments(self.wide, self.worker).std(self.point.s)
        for r in (0.1, 0.3):
            widened = unc.compute_moments(unc.widen(self.wide, r, [0]), self.worker)
            self.assertAlmostEqual(scr.sigma.loc[r, 'sigma'], widened.std(self.point.s), places=13)
            self.assertAlmostEqual(scr.sigma.loc[r, 'ratio'], widened.std(self.point.s) / declared, places=13)
            top = scr.top[r]
            self.assertEqual(len(top), 5)
            self.assertTrue((np.diff(top['ST'].to_numpy()) <= 0).all())
        self.assertGreater(scr.sigma.loc[0.3, 'undeclared_share'], scr.sigma.loc[0.1, 'undeclared_share'])

    def test_width_sensitivity_bounds_the_resolved_optimum(self):
        r = 0.3
        table = unc.width_sensitivity(self.point, self.data, self.worker,
                                      [0.0, r, {'soc_demo_foreground_db': r}, {'Cf': r}], exact_cfs=[0])
        with self.assertRaises(TypeError):
            unc.screen_undeclared(self.point, self.data, self.worker)
        self.assertEqual(table.loc[0, 'delta_sigma'], 0.0)
        np.testing.assert_allclose(table['bound'], self.point.kappa * table['delta_sigma'])
        self.assertEqual(list(table.loc[2, ['r[soc_demo_background_db]', 'r[soc_demo_foreground_db]', 'r[Cf]']]),
                         [0.0, r, 0.0])
        # Re-solving at the wider setting rises by at least 0 and at most the bound.
        widened = unc.compute_moments(unc.widen(self.data, r, unc.co2_flows(self.worker)), self.worker)
        resolved = unc.ChanceConstrained(self.worker, widened, upper_bounds={self.worker.elyz: TRI_CAP})
        rise = resolved.solve_point(0.9).adjusted - self.point.adjusted
        self.assertGreaterEqual(rise, -1e-9)
        self.assertLessEqual(rise, table.loc[1, 'bound'] + 1e-9)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics(unittest.TestCase):

    def test_front_diagnostics(self):
        worker, data, mom, problem, front = solved()
        table = unc.diagnostics(front, mom)
        self.assertEqual(list(table.index), LAMBDAS)
        B_mean, B_var = mom.B_mean.toarray(), mom.B_var.toarray()
        q, w = mom.q_mean, mom.q_var
        var_c = ((q[:, None] ** 2) * B_var + w[:, None] * B_mean ** 2 + w[:, None] * B_var).sum(axis=0)
        np.testing.assert_allclose(mom.process_std(), np.sqrt(var_c), rtol=1e-14)
        for lam, point in front.items():
            row = table.loc[lam]
            s = point.s
            self.assertAlmostEqual(row['sigma_indep'], float(np.sqrt(var_c @ s ** 2)), places=13)
            cf = unc.decompose(point, mom).parameters
            self.assertAlmostEqual(row['cf_share'], cf[cf['group'] == 'Cf']['ST'].sum(), places=13)
            self.assertAlmostEqual(row['bound upper:4'], point.bounds[('upper', 4)], places=15)
            self.assertEqual((row['n_processes'], row['n_variables']), (6, 2))
            self.assertGreater(row['seconds_total'], 0.0)
            self.assertNotIn('sigma_l1', row.index)


# ---------------------------------------------------------------------------
# sampling, projections, validation
# ---------------------------------------------------------------------------

class TestSampling(unittest.TestCase):

    SPECS = [
        {'uncertainty_type': 3, 'amount': 1.0, 'loc': 1.0, 'scale': 0.2},
        {'uncertainty_type': 2, 'amount': 2.0, 'loc': np.log(2.0), 'scale': 0.4},
        {'uncertainty_type': 2, 'amount': -2.0, 'loc': np.log(2.0), 'scale': 0.4, 'negative': True},
        {'uncertainty_type': 4, 'amount': 1.0, 'minimum': 0.5, 'maximum': 2.0},
        {'uncertainty_type': 5, 'amount': 0.03, 'minimum': 0.005, 'loc': 0.03, 'maximum': 0.035},
        {'uncertainty_type': 5, 'amount': 1.0, 'minimum': 1.0, 'loc': 1.0, 'maximum': 1.0},
        {'uncertainty_type': 1, 'amount': 7.0},
        {'uncertainty_type': 0, 'amount': -3.0},
    ]

    def test_each_family_against_its_moments_and_cdf(self):
        n = 400_000
        x = unc.sample_specs(self.SPECS, n, rng=3)
        self.assertEqual(x.shape, (len(self.SPECS), n))
        for spec, draws in zip(self.SPECS, x):
            mean, var = unc.spec_moments(spec)
            if var == 0:
                self.assertTrue((draws == mean).all())
                continue
            self.assertLess(abs(draws.mean() - mean), 4 * np.sqrt(var / n))
            self.assertLess(abs(draws.var() - var), 4 * np.std((draws - mean) ** 2) / np.sqrt(n))
            for p in (0.01, 0.3, 0.7, 0.99):
                q = cc.declared_quantile(spec, p)
                self.assertLess(abs((draws <= q).mean() - p), 4 * np.sqrt(p * (1 - p) / n))

    def test_seeded_and_unsupported(self):
        np.testing.assert_array_equal(unc.sample_specs(self.SPECS, 10, rng=5), unc.sample_specs(self.SPECS, 10, rng=5))
        with self.assertRaises(NotImplementedError):
            unc.sample_specs([{'uncertainty_type': 7, 'amount': 1.0}], 3)


class TestImpactSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.worker, cls.data, cls.mom, cls.problem, cls.front = solved()
        cls.S = np.vstack([p.s for p in cls.front.values()])

    def test_moments_of_X_against_the_closed_form(self):
        n = 400_000
        sampler = unc.ImpactSampler(self.data, self.worker, self.S, seed=1, tol=0.0)
        np.testing.assert_array_equal(sampler.covered, 1.0)
        X = sampler.sample(n)
        for k, s in enumerate(self.S):
            mean, var = self.mom.mean(s), self.mom.variance(s)
            self.assertLess(abs(X[k].mean() - mean), 4 * np.sqrt(var / n))
            self.assertLess(abs(X[k].var() - var), 4 * np.std((X[k] - mean) ** 2) / np.sqrt(n))
        # The same seed gives the same draws.
        first = unc.ImpactSampler(self.data, self.worker, self.S, seed=1, tol=0.0).sample(1000)
        np.testing.assert_array_equal(first, unc.ImpactSampler(self.data, self.worker, self.S, seed=1,
                                                               tol=0.0).sample(1000))

    def test_draws_depend_on_the_seed_alone(self):
        # A decision that leaves SMR idle draws fewer entries than the front;
        # adding the front must not change its draws, nor must the block size.
        idle = self.S[0].copy()
        idle[2] = 0.0
        alone = unc.ImpactSampler(self.data, self.worker, [idle], seed=8, tol=0.0)
        joint = unc.ImpactSampler(self.data, self.worker, [idle, *self.S], seed=8, tol=0.0)
        self.assertLess(alone.n_drawn, joint.n_drawn)
        x = alone.sample(5000)[0]
        np.testing.assert_allclose(joint.sample(5000)[0], x, rtol=1e-13)
        chunk = validation.CHUNK_ENTRIES
        try:
            validation.CHUNK_ENTRIES = 64
            blocked = unc.ImpactSampler(self.data, self.worker, [idle], seed=8, tol=0.0)
            np.testing.assert_allclose(blocked.sample(5000)[0], x, rtol=1e-13)
        finally:
            validation.CHUNK_ENTRIES = chunk
        # Successive calls continue the streams.
        again = unc.ImpactSampler(self.data, self.worker, [idle], seed=8, tol=0.0)
        np.testing.assert_allclose(np.concatenate([again.sample(2000), again.sample(3000)], axis=1)[0], x,
                                   rtol=1e-13)
        self.assertFalse(np.allclose(unc.ImpactSampler(self.data, self.worker, [idle], seed=9).sample(10), x[:10]))

    def test_held_entries_stay_below_the_tolerance(self):
        sampler = unc.ImpactSampler(self.data, self.worker, self.S, tol=0.05)
        everything = unc.ImpactSampler(self.data, self.worker, self.S, tol=0.0)
        self.assertLess(sampler.n_drawn, everything.n_drawn)
        self.assertTrue((sampler.covered >= 0.95).all())
        self.assertTrue((sampler.covered < 1.0).any())
        # The share held is the sum of the total-order indices of the held entries.
        drawn = set(map(tuple, sampler.drawn.tolist()))
        for k, s in enumerate(self.S):
            b = unc.decompose(s, self.mom).parameters.query("group == 'If'")
            held = [st for e, j, st in zip(b['flow'], b['process'], b['ST']) if (e, j) not in drawn]
            self.assertAlmostEqual(1.0 - sampler.covered[k], sum(held), places=12)
        X = sampler.sample(200_000)
        for k, s in enumerate(self.S):
            # Held entries sit at their means: the mean is still exact.
            self.assertLess(abs(X[k].mean() - self.mom.mean(s)), 4 * self.mom.std(s) / np.sqrt(200_000))

    def test_projections_rebuild_mean_variance_and_sampled_impacts(self):
        p = self.problem.projections()
        model = self.problem.model
        f_tilde = model.demand()[0]
        rng = np.random.default_rng(0)
        draws = unc.draw_parameters(self.data, 4, rng=rng, processes=p.J)
        B_mean, q_mean = self.mom.B_mean.toarray(), self.mom.q_mean
        S_J = dict(zip(p.J.tolist(), p.S_J))
        position = {e: i for i, e in enumerate(self.mom.cf_rows)}
        for _ in range(3):
            v = rng.random(model.system.n_free)
            s = model.system.recover(f_tilde, v)
            self.assertAlmostEqual(p.m0 + p.m @ v, self.mom.mean(s), places=12)
            var = (self.mom.d[p.J] * (p.s0[p.J] + p.S_J @ v) ** 2).sum() + \
                (self.mom.w * (p.B_unc_s0 + p.B_unc_S @ v) ** 2).sum()
            self.assertAlmostEqual(var, self.mom.variance(s), places=12)
            for t in range(draws.B.shape[1]):
                B, q = B_mean.copy(), q_mean.copy()
                B[draws.b_rows, draws.b_cols] = draws.B[:, t]
                q[draws.q_rows] = draws.Q[:, t]
                direct = q @ B @ s
                # The sampled impact of the base and of one unit of each alternative.
                x0 = q @ B @ p.s0
                x = p.m.copy()
                for e, i in position.items():
                    x += (q[e] - q_mean[e]) * p.B_unc_S[i]
                for e, j, b in zip(draws.b_rows, draws.b_cols, draws.B[:, t]):
                    x += q[e] * (b - B_mean[e, j]) * S_J[j]
                self.assertAlmostEqual(x0 + x @ v, direct, places=11)


class TestValidate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.worker, cls.data, cls.mom, cls.problem, cls.front = solved()

    def test_coverage_is_exact_when_X_is_exactly_normal(self):
        # Normal B entries with the declared moments and exact CFs make X normal.
        data = copy.deepcopy(self.data)
        for db, block in data['If'].items():
            specs = {}
            for idx, spec in block['defined'].items():
                mean, var = unc.spec_moments(spec)
                specs[idx] = {'uncertainty_type': 3, 'loc': mean, 'scale': np.sqrt(var)}
            unc.override(data, 'If', db, specs)
        unc.override(data, 'Cf', SOC_METHOD, {e: {'uncertainty_type': 1} for e in data['Cf'][SOC_METHOD]['defined']})
        mom = unc.compute_moments(data, self.worker)
        self.assertEqual(mom.w.size, 0)
        problem = unc.ChanceConstrained(self.worker, mom, upper_bounds={self.worker.elyz: TRI_CAP})
        front = problem.solve(LAMBDAS)
        n = 400_000
        val = unc.validate(front, problem, data, n=n, seed=2026)
        for lam, row in val.table.loc['front'].iterrows():
            lz = row['lambda_impact']
            se = np.sqrt(lz * (1 - lz) / n)
            self.assertLess(abs(row['coverage'] - lz), 4 * se, f"lambda {lam}")
            nominal = row['nominal upper:4']
            self.assertLess(abs(row['coverage upper:4'] - nominal), 4 * np.sqrt(nominal * (1 - nominal) / n))
            # Independent events: the joint coverage is the product, at least lambda.
            self.assertLess(abs(row['joint_coverage'] - lz * nominal), 4 * np.sqrt(lam * (1 - lam) / n))
            self.assertLess(abs(row['skew']), 4 * np.sqrt(6 / n))
            quantile_se = row['sigma'] * se / scipy.stats.norm.pdf(scipy.stats.norm.ppf(lz))
            self.assertLess(abs(row['quantile_error']), 4 * quantile_se)
            self.assertLess(abs(row['mean_error_in_se']), 4)
            self.assertLess(abs(row['cvar_empirical'] - row['cvar_gaussian']), 0.05 * row['sigma'])

    def test_table_regret_and_reproducibility(self):
        val = unc.validate(self.front, self.problem, self.data, n=20_000, seed=9,
                           designs={'smr': self.front[0.5].s})
        self.assertEqual(val.seed, 9)
        self.assertEqual(val.decisions, ['lambda=0.5', 'lambda=0.9', 'lambda=0.99', 'smr'])
        self.assertEqual(val.impacts.shape, (4, 20_000))
        X = val.impacts
        table = val.table
        self.assertEqual(table.loc[('front', 0.5), 'regret_vs_lowest_lambda_mean'], 0.0)
        for k, lam in enumerate(LAMBDAS):
            row = table.loc[('front', lam)]
            self.assertAlmostEqual(row['regret_vs_lowest_lambda_mean'], (X[k] - X[0]).mean(), places=12)
            self.assertAlmostEqual(row['regret_vs_front_best_p95'],
                                   np.percentile(X[k] - X[:3].min(axis=0), 95), places=12)
            self.assertGreaterEqual(row['regret_vs_front_best_mean'], 0.0)
            # The design is priced against each point's target on the same draws.
            design = table.loc[('smr', lam)]
            self.assertEqual(design['threshold'], row['threshold'])
            self.assertAlmostEqual(design['coverage'], (X[3] <= row['threshold']).mean(), places=15)
        np.testing.assert_allclose(X[3], X[0], rtol=1e-14)   # the same decision on the same draws
        again = unc.validate(self.front, self.problem, self.data, n=20_000, seed=9)
        np.testing.assert_allclose(again.impacts, X[:3], rtol=1e-14)
        fresh = unc.validate(self.front, self.problem, self.data, n=1_000)
        self.assertIsInstance(fresh.seed, int)
        np.testing.assert_array_equal(unc.validate(self.front, self.problem, self.data, n=1_000,
                                                   seed=fresh.seed).impacts, fresh.impacts)

    def test_sample_size_and_bounds_without_events(self):
        uncapped = unc.ChanceConstrained(self.worker, self.mom)
        front = uncapped.solve([0.9])
        val = unc.validate(front, uncapped, self.data, n=2e3, seed=1)
        self.assertEqual(val.n, 2000)
        self.assertEqual(val.impacts.shape, (1, 2000))
        row = val.table.loc[('front', 0.9)]
        self.assertEqual(row['joint_coverage'], row['coverage'])
        with self.assertRaises(ValueError):
            unc.validate(front, uncapped, self.data, n=0)

    def test_wilson_matches_scipy(self):
        for k, n in ((0, 50), (3, 50), (980, 1000), (1000, 1000)):
            low, high = unc.wilson(k, n, 0.95)
            ci = scipy.stats.binomtest(k, n).proportion_ci(0.95, method='wilson')
            self.assertAlmostEqual(low, ci.low, places=12)
            self.assertAlmostEqual(high, ci.high, places=12)

    def test_widened_configuration(self):
        widened = unc.widen(self.data, 0.3, unc.co2_flows(self.worker))
        val = unc.validate(self.front, self.problem, widened, n=50_000, seed=4)
        world = unc.compute_moments(widened, self.worker)
        for k, (lam, point) in enumerate(self.front.items()):
            row = val.table.loc[('front', lam)]
            self.assertEqual(row['sigma'], point.sigma)       # the claim
            self.assertLess(abs(row['mean_error_in_se']), 4)  # audited against the drawn world
            self.assertAlmostEqual(row['std_ratio'], row['mc_std'] / world.std(point.s), places=12)
