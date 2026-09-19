"""Tests for the LP equilibration in ``pulpo.utils.scaling``.

The scaled LP is meant to be *invisible*: ``instantiate(scale=True)`` must
return the same optimum, scaling vector, impacts and extracted
results as ``scale=False``, on the static and on the time-indexed model, and
the machinery around it (``update_env_cost``, re-solving the same instance,
``extract_params``) must keep reading in original units.

The reason the feature exists cannot be shown on the sample databases (their
coefficients are all O(1)); it is exercised on a synthetic model whose
technosphere mimics ecoinvent's facility processes: a product consumed in
1e11 units by a process whose own activity level is 1e-10.
"""

import unittest
import warnings

import numpy as np
import pyomo.environ as pyo
import scipy.sparse as sps

from pulpo import pulpo, pulpo_time
from pulpo.utils import optimizer, scaling
from pulpo.utils.saver import extract_params
from pulpo.datasets.sample_database import (
    setup_test_db, setup_background_db, setup_biosphere_db,
    setup_lcia_methods, setup_foreground_db,
)
from pulpo.datasets.elec_time_database import (
    setup_elec_time_db, PROJECT_NAME as TIME_PROJECT, DB_NAME as TIME_DB,
    CHARGE_PRODUCT_CHOICE, ELECTRICITY_CHOICE,
)
from pulpo.utils.utils import is_bw25

setup_biosphere_db()
setup_lcia_methods()
setup_test_db()
setup_background_db()
setup_foreground_db()
setup_elec_time_db()

project_name = "sample_project_bw25" if is_bw25() else "sample_project"
METHODS = {
    "('my project', 'climate change')": 1,
    "('my project', 'air quality')": 1,
    "('my project', 'resources')": 0,
}


def _sample_worker():
    worker = pulpo.PulpoOptimizer(project_name, 'technosphere', METHODS, '')
    worker.intervention_matrix = 'biosphere3'
    worker.get_lci_data()
    eCar = worker.retrieve_activities(reference_products='transport')
    elec = worker.retrieve_activities(reference_products='electricity')
    return worker, {eCar[0]: 1}, {'electricity': {elec[0]: 100, elec[1]: 100}}


def _scaling_vector(worker):
    return {j: worker.instance.scaling_vector[j].value for j in worker.instance.PROCESS}


class TestGeometricScaling(unittest.TestCase):
    def test_factors_are_powers_of_two_and_balance_the_matrix(self):
        rng = np.random.default_rng(0)
        # A badly scaled matrix with ecoinvent's structure: well-behaved
        # coefficients (0.1 .. 10) in units that differ by up to 1e10 per
        # product row and per process column, i.e. A = R M S with R, S diagonal.
        M = sps.random(60, 80, density=0.1, random_state=0, format='csr')
        M.data = 10.0 ** rng.uniform(-1, 1, M.nnz)
        R = sps.diags(10.0 ** rng.uniform(-5, 5, 60))
        S = sps.diags(10.0 ** rng.uniform(-5, 5, 80))
        A = (R @ M @ S).tocsr()
        r, s = scaling.geometric_scaling(A)
        self.assertTrue(np.all(np.log2(r) == np.round(np.log2(r))))
        self.assertTrue(np.all(np.log2(s) == np.round(np.log2(s))))
        B = (sps.diags(r) @ A @ sps.diags(s)).tocoo()
        before = np.log10(A.data.max() / A.data.min())
        after = np.log10(B.data.max() / B.data.min())
        self.assertGreater(before, 12)
        self.assertLess(after, 4)
        # exact: scaling and unscaling round-trips bit for bit
        C = (sps.diags(1 / r) @ B @ sps.diags(1 / s)).tocsr()
        np.testing.assert_array_equal(C.data, A.tocsr().data)

    def test_tiny_entries_do_not_drive_the_factors(self):
        A = sps.csr_matrix(np.array([[1.0, 1e-45], [1e-45, 1.0]]))
        r, s = scaling.geometric_scaling(A)
        np.testing.assert_array_equal(r, 1.0)
        np.testing.assert_array_equal(s, 1.0)

    def test_empty_matrix(self):
        r, s = scaling.geometric_scaling(sps.csr_matrix((3, 4)))
        np.testing.assert_array_equal(r, 1.0)
        np.testing.assert_array_equal(s, 1.0)


class TestEquilibrateModelData(unittest.TestCase):
    """A hand-built ``combine_inputs`` dictionary with a facility process."""

    @staticmethod
    def _data():
        # products/processes 0..2; process 2 is a "facility" consumed in
        # 1e-10 units by process 0 and consuming 1e11 units of product 1.
        tech = {(0, 0): 1.0, (1, 1): 1.0, (2, 2): 1.0,
                (1, 0): -0.5, (2, 0): -1e-10, (1, 2): -1e11}
        return {None: {
            'PRODUCT': {None: [0, 1, 2]}, 'PROCESS': {None: [0, 1, 2]},
            'TECH_MATRIX': dict(tech),
            'ENV_COST_MATRIX': {(0, 'h'): 1.0, (1, 'h'): 2.0, (2, 'h'): 3e10},
            'INV_MATRIX': {(7, 2): 5e10},
            'FINAL_DEMAND': {0: 1.0, 1: 0.0, 2: 0.0},
            'LOWER_LIMIT': {0: -float('inf'), 1: -float('inf'), 2: -1e4},
            'UPPER_LIMIT': {0: float('inf'), 1: 10.0, 2: 1e9},
            'LEFT_WEIGHTS': {('c', 2): 1.0}, 'RIGHT_WEIGHTS': {('c', 0): 2.0},
        }}

    def test_scaled_system_is_equivalent(self):
        data = self._data()
        original = {k: dict(v) if isinstance(v, dict) else v for k, v in data[None].items()}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            r, s = scaling.equilibrate_model_data(data)
        d = data[None]
        # every tech entry is a_ij * r_i * s_j; row factors carry the 1024 shift
        for (i, j), v in original['TECH_MATRIX'].items():
            self.assertEqual(d['TECH_MATRIX'][(i, j)], v * r[i] * s[j])
        self.assertTrue(all(np.log2(x / 1024.0) == round(np.log2(x / 1024.0)) for x in r.values()))
        for i, v in original['FINAL_DEMAND'].items():
            self.assertEqual(d['FINAL_DEMAND'][i], v * r[i])
        for (j, h), v in original['ENV_COST_MATRIX'].items():
            self.assertEqual(d['ENV_COST_MATRIX'][(j, h)], v * s[j])
        self.assertEqual(d['INV_MATRIX'][(7, 2)], 5e10 * s[2])
        self.assertEqual(d['UPPER_LIMIT'][1], 10.0 / s[1])
        self.assertEqual(d['LEFT_WEIGHTS'][('c', 2)], s[2])
        self.assertEqual(d['RIGHT_WEIGHTS'][('c', 0)], 2.0 * s[0])
        self.assertEqual(d['ROW_SCALE'], r)
        self.assertEqual(d['COL_SCALE'], s)
        # the facility column is scaled up by ~1e10, so its 1e11 coefficient is O(1)
        self.assertLess(abs(d['TECH_MATRIX'][(1, 2)]) / abs(d['TECH_MATRIX'][(1, 1)]), 1e3)
        # ... and its finite 1e9 bound would become ~1e19: capped to inf, with a warning
        self.assertEqual(d['UPPER_LIMIT'][2], float('inf'))
        self.assertTrue(any('treated as infinite' in str(w.message) for w in caught))

    def test_time_indexed_keys_and_carry_over(self):
        data = self._data()
        d = data[None]
        d['TIME'] = {None: ['t0', 't1']}
        d['FINAL_DEMAND'] = {(t, i): v for t in ('t0', 't1') for i, v in d['FINAL_DEMAND'].items()}
        d['LOWER_LIMIT'] = {(t, j): v for t in ('t0', 't1') for j, v in d['LOWER_LIMIT'].items()}
        d['UPPER_LIMIT'] = {(t, j): v for t in ('t0', 't1') for j, v in d['UPPER_LIMIT'].items()}
        d['K'] = {(0, 1): 0.9}
        del d['LEFT_WEIGHTS'], d['RIGHT_WEIGHTS']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r, s = scaling.equilibrate_model_data(data)
        self.assertEqual(d['FINAL_DEMAND'][('t1', 0)], 1.0 * r[0])
        self.assertEqual(d['UPPER_LIMIT'][('t0', 1)], 10.0 / s[1])
        # K[i, i2] * (scaled row i2) must equal r_i * K * (row i2)
        self.assertEqual(d['K'][(0, 1)], 0.9 * r[0] / r[1])


class TestScaledSolveEqualsUnscaled(unittest.TestCase):
    """End-to-end on the sample database: results identical either way."""

    def test_static_model(self):
        worker, demand, choices = _sample_worker()
        worker.instantiate(choices=choices, demand=demand, scale=False)
        worker.solve()
        ref_obj = worker.instance.OBJ()
        ref_x = _scaling_vector(worker)
        ref_aux = worker.instance.impacts_calculated["('my project', 'resources')"].value
        ref_results = worker.extract_results(extractparams=True)
        self.assertFalse(scaling.is_scaled(worker.instance))

        worker.instantiate(choices=choices, demand=demand, scale=True)
        self.assertTrue(scaling.is_scaled(worker.instance))
        self.assertEqual(set(worker.instance._col_scale), set(worker.instance.PROCESS))
        worker.solve()
        self.assertAlmostEqual(worker.instance.OBJ(), ref_obj, places=9)
        self.assertAlmostEqual(
            worker.instance.impacts_calculated["('my project', 'resources')"].value, ref_aux, places=9)
        for j, v in ref_x.items():
            self.assertAlmostEqual(worker.instance.scaling_vector[j].value, v, places=9)
        results = worker.extract_results(extractparams=True)
        for key in ('Scaling Vector', 'Impacts', 'Choices'):
            self.assertIn(key, results)
        np.testing.assert_allclose(
            results['Scaling Vector']['Value'].sort_index().to_numpy(),
            ref_results['Scaling Vector']['Value'].sort_index().to_numpy(), rtol=0, atol=1e-9)
        # extract_params reports the environmental costs per unit of activity
        env = extract_params(worker.instance)['ENV_COST_MATRIX']['Value'].sort_index()
        ref_env = ref_results['ENV_COST_MATRIX']['Value'].sort_index()
        np.testing.assert_allclose(env.to_numpy(), ref_env.to_numpy(), rtol=1e-12)

    def test_finite_default_limits_are_relaxed_but_explicit_limits_kept(self):
        worker, demand, choices = _sample_worker()
        elec = worker.retrieve_activities(reference_products='electricity')
        finite = {'lower_bound': -1e4, 'upper_bound': 1e9, 'upper_inv_bound': 1e9,
                  'lower_inv_bound': -1e9, 'lower_imp_bound': -1e6, 'upper_imp_bound': 1e6}
        worker.instantiate(choices=choices, demand=demand, default_limits=finite, scale=False)
        worker.solve()
        ref_obj = worker.instance.OBJ()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            worker.instantiate(choices=choices, demand=demand, default_limits=finite,
                               upper_limit={elec[0]: 100}, scale=True)
        self.assertTrue(any('replaced by +-inf' in str(w.message) for w in caught))
        inst = worker.instance
        pmap = worker.lci_data['process_map']
        j_explicit = pmap[elec[0].key]
        s = inst._col_scale
        # explicit limit kept (in scaled units), choice capacity kept, defaults infinite
        self.assertEqual(pyo.value(inst.UPPER_LIMIT[j_explicit]), 100 / s[j_explicit])
        self.assertEqual(pyo.value(inst.LOWER_LIMIT[j_explicit]), 0.0)     # choice -> lower 0
        others = [j for j in inst.PROCESS if j not in {pmap[a.key] for a in choices['electricity']}]
        self.assertTrue(all(pyo.value(inst.LOWER_LIMIT[j]) == -float('inf') for j in others))
        self.assertTrue(all(pyo.value(inst.UPPER_LIMIT[j]) == float('inf') for j in others))
        worker.solve()
        self.assertAlmostEqual(worker.instance.OBJ(), ref_obj, places=9)

    def test_resolve_same_instance(self):
        """Re-solving (as the epsilon-constraint loop and Monte Carlo do) must
        not double-scale the values left in the instance by the previous solve."""
        worker, demand, choices = _sample_worker()
        worker.instantiate(choices=choices, demand=demand, scale=True)
        worker.solve()
        first = _scaling_vector(worker)
        worker.solve()
        self.assertEqual(_scaling_vector(worker), first)

    def test_supply_slack_is_unscaled(self):
        worker, demand, choices = _sample_worker()
        elec = worker.retrieve_activities(reference_products='electricity')
        limits = {elec[0]: 0.4}
        worker.instantiate(choices=choices, demand=demand, upper_limit=limits, lower_limit=limits, scale=False)
        worker.solve()
        ref = {i: worker.instance.slack[i].value for i in worker.instance.slack}
        worker.instantiate(choices=choices, demand=demand, upper_limit=limits, lower_limit=limits, scale=True)
        worker.solve()
        for i, v in ref.items():
            self.assertAlmostEqual(worker.instance.slack[i].value, v, places=9)

    def test_update_env_cost_takes_original_units(self):
        worker, demand, choices = _sample_worker()
        h = "('my project', 'climate change')"
        worker.instantiate(choices=choices, demand=demand, scale=True)
        worker.solve()
        j = next(iter(worker.instance.PROCESS))
        # doubling one coefficient in original units doubles its reported value
        before = extract_params(worker.instance)['ENV_COST_MATRIX']['Value'][(j, h)]
        optimizer.update_env_cost(worker.instance, {(j, h): 2 * before})
        after = extract_params(worker.instance)['ENV_COST_MATRIX']['Value'][(j, h)]
        self.assertAlmostEqual(after, 2 * before, places=12)
        self.assertAlmostEqual(worker.instance._env_cost[(j, h)],
                               2 * before * worker.instance._col_scale[j], places=12)

    def test_time_model(self):
        GWP = str(("GWP", "100a"))
        time_steps = ['t0', 't1', 't2']
        names = ("solar", "coal", "battery_charge", "battery_hold", "battery_discharge", "battery_holdtm1")

        def build():
            worker = pulpo_time.PulpoOptimizerTime(TIME_PROJECT, TIME_DB, {GWP: 1}, "")
            worker.intervention_matrix = "biosphere3"
            worker.get_lci_data()
            acts = {n: worker.retrieve_activities(activities=[n])[0] for n in names}
            choices = {ELECTRICITY_CHOICE: {acts['solar']: 1e6, acts['coal']: 1e6, acts['battery_discharge']: 1e6},
                       CHARGE_PRODUCT_CHOICE: {acts['battery_charge']: 1e6, acts['battery_hold']: 1e6}}
            upper = {t: {acts['solar']: [5.0, 1.0, 0.0][k], acts['coal']: 1e6, acts['battery_charge']: 1e6,
                         acts['battery_hold']: 1e6, acts['battery_discharge']: 0.0 if k == 0 else 1e6,
                         acts['battery_holdtm1']: 0.0} for k, t in enumerate(time_steps)}
            lower = {t: {acts['battery_charge']: 0.0, acts['battery_hold']: 0.0, acts['battery_discharge']: 0.0}
                     for t in time_steps}
            demand = {t: {ELECTRICITY_CHOICE: 3.0} for t in time_steps}
            storage = [(acts['battery_holdtm1'], CHARGE_PRODUCT_CHOICE, 0.8)]
            return worker, dict(choices=choices, demand=demand, upper_limit=upper, lower_limit=lower,
                                time_steps=time_steps, storage=storage)

        worker, kwargs = build()
        worker.instantiate(scale=False, **kwargs)
        worker.solve()
        ref_obj = worker.instance.OBJ()
        ref = {idx: worker.instance.scaling_vector[idx].value for idx in worker.instance.scaling_vector}
        self.assertGreater(ref_obj, 0.0)

        worker, kwargs = build()
        worker.instantiate(scale=True, **kwargs)
        self.assertTrue(scaling.is_scaled(worker.instance))
        worker.solve()
        self.assertAlmostEqual(worker.instance.OBJ(), ref_obj, places=8)
        for idx, v in ref.items():
            self.assertAlmostEqual(worker.instance.scaling_vector[idx].value, v, places=7)


def _facility_model_data(scale, alternative=False):
    """A three-process model with an ecoinvent-style facility (see TestFacilityLeak).

    processes: 0 makes the demanded product D, 1 makes E, 2 makes the facility F.
    With ``alternative=True`` a fourth process also makes D, without the
    facility, at a higher but certain environmental cost -- which gives the
    chance-constrained formulations a genuine trade-off to decide.
    """
    tech = {(0, 0): 1.0, (1, 1): 1.0, (2, 2): 1.0,
            (2, 0): -1e-10,   # D consumes 1e-10 facilities per unit
            (1, 2): -1e11}    # a facility consumes 1e11 E
    env = {(0, 'h'): 0.0, (1, 'h'): 1.0, (2, 'h'): 0.0}   # only E carries impact
    processes = [0, 1, 2]
    if alternative:
        processes.append(3)
        tech[(0, 3)] = 1.0
        env[(3, 'h')] = 11.0
    data = {None: {
        'PRODUCT': {None: [0, 1, 2]}, 'PROCESS': {None: processes}, 'INDICATOR': {None: ['h']},
        'INV': {None: []}, 'PRODUCT_PROCESS': {None: list(tech)}, 'INV_PROCESS': {None: []},
        'TECH_MATRIX': tech, 'ENV_COST_MATRIX': env, 'INV_MATRIX': {},
        'FINAL_DEMAND': {0: 1.0, 1: 0.0, 2: 0.0}, 'SUPPLY': {0: 0, 1: 0, 2: 0},
        'LOWER_LIMIT': {j: 0.0 for j in processes}, 'UPPER_LIMIT': {j: float('inf') for j in processes},
        'UPPER_INV_LIMIT': {}, 'LOWER_INV_LIMIT': {},
        'UPPER_IMP_LIMIT': {'h': float('inf')}, 'LOWER_IMP_LIMIT': {'h': -float('inf')},
        'GOAL_INDICATOR': {None: []}, 'IMP_GOALS': {}, 'WEIGHTS': {'h': 1},
        'DEPENDENT_CONSTRAINTS': {None: []}, 'LEFT_WEIGHTS': {}, 'RIGHT_WEIGHTS': {},
    }}
    if scale:
        scaling.equilibrate_model_data(data)
    return data


class TestUncertaintyFormulationsOnScaledModels(unittest.TestCase):
    """The SOC/CC formulations write in the model's units, so a scaled instance
    gives the right answer.

    They add terms ``c * x_j`` and bounds ``x_j <= b`` after the build; on an
    equilibrated model the variable is ``y_j = x_j / s_j``, so what reaches the
    model must be ``c * s_j`` and ``b / s_j``. Checked on the facility model
    with a certain alternative route, whose column factors are far from unity
    (the sample database's coefficients are O(1), so there the factors would
    all be 1 and the test vacuous), against the analytic optimum -- the
    unscaled LP is not a valid reference here, see TestFacilityLeak.
    """

    LAMBDA = 0.9

    @classmethod
    def setUpClass(cls):
        try:
            from pulpo.utils.uncertainty import cc, soc  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("the 'uncertainty' extra is not installed")

    @staticmethod
    def _coeffs():
        from pulpo.utils.uncertainty import soc
        # Variances chosen to span the range of an ecoinvent system: the
        # facility's cost is uncertain by 1e10 per facility (1 per 1e-10 of it).
        mu = np.array([0.0, 1.0, 0.0, 11.0])
        d = np.array([0.01, 0.04, 1e20, 0.25])
        return soc.SOCCoefficients(mu, d, np.zeros(0), np.zeros(0, dtype=int),
                                   sps.csr_matrix((0, 4)), np.sqrt(d), 'h')

    @classmethod
    def _analytic_optimum(cls, coeffs):
        # x = (a, 10a, 1e-10 a, 1 - a): minimise mean + z * sigma over a in [0, 1].
        import scipy.optimize, scipy.stats
        z = scipy.stats.norm.ppf(cls.LAMBDA)

        def objective(a):
            x = np.array([a, 10 * a, 1e-10 * a, 1 - a])
            return float(coeffs.mu_env_cost @ x + z * np.sqrt(coeffs.d @ x ** 2))

        res = scipy.optimize.minimize_scalar(objective, bounds=(0.0, 1.0), method='bounded',
                                             options={'xatol': 1e-12})
        return res.fun, res.x

    def _scaled_instance(self):
        model = optimizer.instantiate(_facility_model_data(scale=True, alternative=True))
        self.assertTrue(scaling.is_scaled(model))
        self.assertTrue(any(f != 1.0 for f in model._col_scale.values()))
        return model

    def test_cutting_plane_reaches_the_analytic_optimum(self):
        from pulpo.utils.uncertainty import soc
        coeffs = self._coeffs()
        model = self._scaled_instance()
        soc.prepare_exact_model(model, coeffs)
        outcome = soc.solve_exact(model, coeffs, self.LAMBDA,
                                  lambda m: optimizer.solve_model(m), verbose=False)
        best, a_star = self._analytic_optimum(coeffs)
        self.assertAlmostEqual(outcome['objective'] / best, 1.0, places=5)
        self.assertLess(outcome['exactness'], 1e-4)     # T == sigma(s*) at the returned point
        self.assertEqual(outcome['bound_crossing'], 0.0)
        # The iterate is read back in original units.
        x = soc.current_scaling_vector(model, 4)
        self.assertAlmostEqual(x[0] + x[3], 1.0, places=9)
        self.assertAlmostEqual(x[1], 10 * x[0], places=6)
        # Kelley certifies the value, not the argmin: the objective is flat
        # around a*, so the iterate is only close to it.
        self.assertAlmostEqual(x[0], a_star, places=2)

    def test_cut_is_exact_at_its_own_point_in_the_models_units(self):
        # The identity behind the cutting planes is grad @ x == sigma. The cut
        # is written against y = x / s, so its coefficients must be grad * s;
        # writing grad itself (what the unguarded path did) breaks the identity.
        from pulpo.utils.uncertainty import soc
        coeffs = self._coeffs()
        model = self._scaled_instance()
        soc.prepare_exact_model(model, coeffs)
        optimizer.solve_model(model)
        x = soc.current_scaling_vector(model, 4)
        sigma, grad = soc.impact_std_gradient(x, coeffs)
        scaling.rescale_solution(model)
        try:
            y = {j: model.scaling_vector[j].value for j in model.PROCESS}
            written = sum(scaling.to_scaled_coefficient(model, j, grad[j]) * y[j] for j in y)
            naive = sum(grad[j] * y[j] for j in y)
        finally:
            scaling.unscale_solution(model)
        self.assertAlmostEqual(written / sigma, 1.0, places=9)
        self.assertNotAlmostEqual(naive / sigma, 1.0, places=2)

    def test_cut_row_factor_is_a_power_of_two_centring_the_row(self):
        rho = scaling.cut_row_factor([1e-8, 1.0, 1e8])
        self.assertEqual(rho, 1.0)
        rho = scaling.cut_row_factor([4.0, 1024.0])
        self.assertEqual(rho, 1.0 / 64)                  # sqrt(4 * 1024) = 64
        self.assertEqual(np.log2(scaling.cut_row_factor([3e-7, 5e9, 1.0])) % 1, 0.0)
        self.assertEqual(scaling.cut_row_factor([]), 1.0)
        self.assertEqual(scaling.cut_row_factor([1e-20]), 1.0)  # below stat_cut

    def test_cc_process_bound_is_stored_in_scaled_units(self):
        from pulpo.utils.uncertainty import cc
        import scipy.stats
        model = self._scaled_instance()
        # Chance-constrained cap on E (process 1): P(x_1 <= xi) >= lambda with
        # xi ~ N(8, 1) gives x_1 <= 8 - z, in original units.
        cc.apply_CC_formulation(model, self.LAMBDA, {}, {'upper_limit': {1: {'loc': 8.0, 'scale': 1.0}}})
        cap = 8.0 - scipy.stats.norm.ppf(self.LAMBDA)
        self.assertAlmostEqual(pyo.value(model.UPPER_LIMIT[1]) * model._col_scale[1], cap, places=12)
        optimizer.solve_model(model)
        # E = 10 a is capped, so a = cap / 10 and the rest of D comes from the
        # certain route at cost 11: impact = 10 a + 11 (1 - a) = 11 - a.
        a = cap / 10.0
        self.assertAlmostEqual(model.scaling_vector[0].value, a, places=9)
        self.assertAlmostEqual(model.scaling_vector[1].value, cap, places=9)
        self.assertAlmostEqual(model.impacts['h'].value, 11.0 - a, places=9)

    def test_direct_cone_rows_are_written_in_scaled_units(self):
        # No solver needed: the defining rows c_k == sqrt(d_j) x_j must carry
        # sqrt(d_j) * s_j on y_j = scaling_vector[j].
        from pyomo.repn import generate_standard_repn
        from pulpo.utils.uncertainty import soc
        coeffs = self._coeffs()
        model = self._scaled_instance()
        soc.apply_SOC_formulation(model, self.LAMBDA, coeffs)
        seen = set()
        for k in model.SOC_TERM:
            repn = generate_standard_repn(model.SOC_C_CNSTR[k].body)
            for var, coef in zip(repn.linear_vars, repn.linear_coefs):
                if var.parent_component() is model.scaling_vector:
                    j = var.index()
                    self.assertAlmostEqual(abs(coef) / (np.sqrt(coeffs.d[j]) * model._col_scale[j]),
                                           1.0, places=12)
                    seen.add(j)
        self.assertEqual(seen, {0, 1, 2, 3})

    def test_placeholder_relaxation_compares_and_writes_in_original_units(self):
        from pulpo.utils.uncertainty import soc
        model = self._scaled_instance()
        # A 1e20 "uncapacitated" placeholder on E, as a case study would set it.
        model.UPPER_LIMIT[1] = scaling.to_scaled_process_bound(model, 1, 1e20)
        model.UPPER_LIMIT[0] = scaling.to_scaled_process_bound(model, 0, 5.0)   # a real cap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # the 1e20 / s_j > cap warning
            n = soc._relax_placeholder_bounds(model, replacement=1e6)
        # The placeholder and the two infinite defaults are moved to the finite
        # replacement (a free variable is what the barrier copes with worst);
        # the real cap on D is left alone. All compared in original units.
        self.assertEqual(n, 3)
        for j in (1, 2, 3):
            self.assertAlmostEqual(
                scaling.from_scaled_process_bound(model, j, pyo.value(model.UPPER_LIMIT[j])), 1e6)
        self.assertAlmostEqual(
            scaling.from_scaled_process_bound(model, 0, pyo.value(model.UPPER_LIMIT[0])), 5.0)

    def test_direct_qcp_matches_the_analytic_optimum(self):
        from pulpo.utils.uncertainty import soc
        if not pyo.SolverFactory('gurobi').available(exception_flag=False):
            self.skipTest("'gurobi' Pyomo solver unavailable")
        coeffs = self._coeffs()
        model = self._scaled_instance()
        soc.apply_SOC_formulation(model, self.LAMBDA, coeffs)
        soc.solve_soc(model)
        best, a_star = self._analytic_optimum(coeffs)
        x = soc.current_scaling_vector(model, 4)   # original units: solve_soc unscales
        self.assertAlmostEqual(x[0] + x[3], 1.0, places=6)
        self.assertAlmostEqual(x[0], a_star, places=4)
        self.assertAlmostEqual(pyo.value(model.OBJ) / best, 1.0, places=5)


class TestFacilityLeak(unittest.TestCase):
    """The failure the equilibration fixes, on a model small enough to inspect.

    Product F is a facility: one unit of it is consumed per 1e-10 units of
    the demanded product, and it consumes 1e11 units of an impact-carrying
    input E. Without scaling, HiGHS's tolerance is applied to the F balance
    row after dividing it by its 1e11 coefficient, so F may be under-supplied
    by far more than its own activity level and the E consumption disappears.
    With scaling the balance holds to 1e-9 relative to the row's own terms.
    """

    def _solve(self, scale):
        model = optimizer.instantiate(_facility_model_data(scale))
        optimizer.solve_model(model)
        x = {j: model.scaling_vector[j].value for j in range(3)}
        # exact facility balance in original units: x_F - 1e-10 x_D = 0
        residual = x[2] - 1e-10 * x[0]
        return model.impacts['h'].value, residual / (1e-10 * x[0])

    def test_scaled_model_keeps_the_facility_balance(self):
        impact, rel = self._solve(scale=True)
        self.assertAlmostEqual(impact, 10.0, places=6)        # 1e-10 facilities * 1e11 E
        self.assertLess(abs(rel), 1e-7)

    def test_unscaled_model_documents_the_leak(self):
        # Not a requirement -- a record of the behaviour the scaling removes.
        # If a future solver release fixes it, this test can simply be dropped.
        impact, rel = self._solve(scale=False)
        self.assertLess(impact, 10.0 - 1e-3)


if __name__ == '__main__':
    unittest.main()
