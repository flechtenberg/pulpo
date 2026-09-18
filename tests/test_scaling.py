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


class TestFacilityLeak(unittest.TestCase):
    """The failure the equilibration fixes, on a model small enough to inspect.

    Product F is a facility: one unit of it is consumed per 1e-10 units of
    the demanded product, and it consumes 1e11 units of an impact-carrying
    input E. Without scaling, HiGHS's tolerance is applied to the F balance
    row after dividing it by its 1e11 coefficient, so F may be under-supplied
    by far more than its own activity level and the E consumption disappears.
    With scaling the balance holds to 1e-9 relative to the row's own terms.
    """

    @staticmethod
    def _model_data(scale):
        # processes: 0 makes the demanded product D, 1 makes E, 2 makes the facility F
        tech = {(0, 0): 1.0, (1, 1): 1.0, (2, 2): 1.0,
                (2, 0): -1e-10,   # D consumes 1e-10 facilities per unit
                (1, 2): -1e11}    # a facility consumes 1e11 E
        env = {(0, 'h'): 0.0, (1, 'h'): 1.0, (2, 'h'): 0.0}   # only E carries impact
        data = {None: {
            'PRODUCT': {None: [0, 1, 2]}, 'PROCESS': {None: [0, 1, 2]}, 'INDICATOR': {None: ['h']},
            'INV': {None: []}, 'PRODUCT_PROCESS': {None: list(tech)}, 'INV_PROCESS': {None: []},
            'TECH_MATRIX': tech, 'ENV_COST_MATRIX': env, 'INV_MATRIX': {},
            'FINAL_DEMAND': {0: 1.0, 1: 0.0, 2: 0.0}, 'SUPPLY': {0: 0, 1: 0, 2: 0},
            'LOWER_LIMIT': {j: 0.0 for j in range(3)}, 'UPPER_LIMIT': {j: float('inf') for j in range(3)},
            'UPPER_INV_LIMIT': {}, 'LOWER_INV_LIMIT': {},
            'UPPER_IMP_LIMIT': {'h': float('inf')}, 'LOWER_IMP_LIMIT': {'h': -float('inf')},
            'GOAL_INDICATOR': {None: []}, 'IMP_GOALS': {}, 'WEIGHTS': {'h': 1},
            'DEPENDENT_CONSTRAINTS': {None: []}, 'LEFT_WEIGHTS': {}, 'RIGHT_WEIGHTS': {},
        }}
        if scale:
            scaling.equilibrate_model_data(data)
        return data

    def _solve(self, scale):
        model = optimizer.instantiate(self._model_data(scale))
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
