"""Tests for the reduced-space backend, ``solve(formulation='reduced')``.

The full Pyomo LP is the reference. Every case solves the same instance both
ways and requires

- the same optimum (relative 1e-7),
- the same active alternatives, and
- a reduced solution that satisfies every constraint and bound of the full
  model, checked on the Pyomo instance itself (so the check does not share
  code with the reduced assembly): feasible plus the full LP's objective
  means optimal for the full LP.

Every static constraint type (process bounds, supply products, upper and
lower elementary-flow limits, impact limits, dependent constraints, the goal
objective) is tested on the sample 'technosphere' database, the sample
foreground system, the rice example and the SOC demo; the electricity toy adds
the static fallback of the time-dependent optimizer. Each limit is derived from
the unconstrained optimum and checked to move the full LP's optimum
(``assert_binds``), so no case passes without its constraint at work. Further
tests cover changes made on the instance itself (limits, fixed variables,
variable bounds, deactivated rows, an edited technosphere matrix), finite
default limits, pickling, and the building blocks against dense linear algebra.
"""

import copy
import os
import pickle
import sys
import unittest
import warnings
from unittest import mock

import numpy as np
import pyomo.environ as pyo
import scipy.sparse as sp
from pyomo.repn import generate_standard_repn

from pulpo import pulpo, pulpo_time
from pulpo.utils import optimizer, reduced, scaling
from pulpo.utils.utils import is_bw25
from pulpo.datasets.sample_database import setup_sample_db
from pulpo.datasets.rice_database import setup_rice_husk_db
from pulpo.datasets.soc_demo_database import setup_soc_demo_db, METHOD_KEY as SOC_METHOD
from pulpo.datasets.elec_time_database import (setup_elec_time_db, PROJECT_NAME as ELEC_PROJECT,
                                               DB_NAME as ELEC_DB, ELECTRICITY_CHOICE,
                                               CHARGE_PRODUCT_CHOICE)

setup_sample_db()
setup_rice_husk_db()
setup_soc_demo_db()
setup_elec_time_db()

SAMPLE_PROJECT = "sample_project_bw25" if is_bw25() else "sample_project"
SOC_PROJECT = "soc_demo_project_bw25" if is_bw25() else "soc_demo_project"
CLIMATE = "('my project', 'climate change')"
AIR = "('my project', 'air quality')"
RESOURCES = "('my project', 'resources')"
ECONOMIC = "('my project', 'economic flow')"

try:
    import gurobipy  # noqa: F401
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False


def _installed(backend):
    try:
        return reduced._resolve_backend(backend) == backend
    except (ImportError, OSError):
        return False


#: The factorization backends installed here; SciPy's is always there.
BACKENDS = [backend for backend in ('pardiso', 'umfpack', 'scipy') if _installed(backend)]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def full_model_violation(instance):
    """Largest violation of any active constraint or variable bound of the Pyomo
    model at the values it currently holds, relative to the size of the row's
    terms (a balance sums terms much larger than its result)."""
    worst = 0.0
    scaled = scaling.is_scaled(instance)
    if scaled:
        scaling.rescale_solution(instance)
    try:
        for con in instance.component_data_objects(pyo.Constraint, active=True):
            body = pyo.value(con.body)
            repn = generate_standard_repn(con.body, compute_values=True)
            size = max([1.0, abs(body)] + [abs(c * v.value) for c, v in zip(repn.linear_coefs, repn.linear_vars)])
            if con.has_lb():
                worst = max(worst, (pyo.value(con.lower) - body) / size)
            if con.has_ub():
                worst = max(worst, (body - pyo.value(con.upper)) / size)
        for var in instance.component_data_objects(pyo.Var):
            if var.value is None or var.parent_component().local_name in ('impacts_calculated', 'inv_flows'):
                continue
            size = max(1.0, abs(var.value))
            if var.lb is not None:
                worst = max(worst, (var.lb - var.value) / size)
            if var.ub is not None:
                worst = max(worst, (var.value - var.ub) / size)
    finally:
        if scaled:
            scaling.unscale_solution(instance)
    return worst


def alternative_values(worker):
    pm = worker.lci_data['process_map']
    return {pm[p.key]: worker.instance.scaling_vector[pm[p.key]].value
            for procs in worker.choices.values() for p in procs}


def active(values, tol=1e-7):
    scale = max([1.0] + [abs(v) for v in values.values()])
    return {j for j, v in values.items() if abs(v) > tol * scale}


class ParityMixin:
    """Solve full, then reduced, on the same instance and compare."""

    RTOL = 1e-7

    def assert_parity(self, worker, solver_name=None, options=None, backend='auto', unique=True,
                      free=(), feas_tol=1e-8):
        """``unique=False`` for a problem with several optimal mixes: the active
        alternatives may then differ legitimately. ``free`` lists alternatives
        (process ids) whose activity is a zero-cost direction of the LP and is
        left out of the comparison."""
        worker.solve(solver_name=solver_name, options=options)
        obj_full = pyo.value(worker.instance.OBJ)
        alt_full = alternative_values(worker)
        slack_full = {i: worker.instance.slack[i].value for i in worker.instance.PRODUCT_SUPPLY}

        if backend != 'auto':
            reduced.build(worker, backend=backend)
        results = worker.solve(formulation='reduced', solver_name=solver_name, options=options)
        self.assertEqual(str(results.termination_condition), 'optimal')
        obj_red = pyo.value(worker.instance.OBJ)
        alt_red = alternative_values(worker)
        slack_red = {i: worker.instance.slack[i].value for i in worker.instance.PRODUCT_SUPPLY}
        for j in free:
            alt_full.pop(j, None)
            alt_red.pop(j, None)

        scale = max(1.0, abs(obj_full))
        self.assertLessEqual(abs(obj_red - obj_full), self.RTOL * scale,
                             f"objective full {obj_full!r} vs reduced {obj_red!r}")
        self.assertAlmostEqual(results.objective, obj_red, delta=1e-9 * scale)
        if unique:
            self.assertEqual(active(alt_full), active(alt_red))
            for i in slack_full:
                self.assertAlmostEqual(slack_full[i], slack_red[i], delta=1e-7 * max(1.0, abs(slack_full[i])))
        self.assertLessEqual(full_model_violation(worker.instance), feas_tol)
        self.assertLessEqual(results.balance_residual, 1e-10)
        return results, obj_full

    def assert_binds(self, worker, base, constrained, rtol=1e-6):
        """The full LP's optimum moves when ``constrained`` replaces ``base``;
        leaves the worker instantiated with ``constrained``."""
        worker.instantiate(**base)
        worker.solve()
        free = pyo.value(worker.instance.OBJ)
        worker.instantiate(**constrained)
        worker.solve()
        tight = pyo.value(worker.instance.OBJ)
        self.assertGreater(abs(tight - free), rtol * max(1.0, abs(free)),
                           f"the constraint does not bind: {free!r} vs {tight!r}")

    def assert_both_infeasible(self, worker):
        """The full LP and the reduced LP both raise on an infeasible instance."""
        with self.assertRaises(RuntimeError):
            worker.solve()
        with self.assertRaises(reduced.ReducedSolveError) as caught:
            worker.solve(formulation='reduced')
        self.assertIn(str(caught.exception.results.termination_condition),
                      ('infeasible', 'infeasibleOrUnbounded'))

    def flow_index(self, worker, name):
        return worker.lci_data['intervention_map'][worker.retrieve_envflows(activities=name)[0].key]


# ---------------------------------------------------------------------------
# sample project, 'technosphere' (five processes)
# ---------------------------------------------------------------------------

class TestReducedSampleTechnosphere(ParityMixin, unittest.TestCase):

    METHODS = {CLIMATE: 1, AIR: 1, RESOURCES: 0}

    def worker(self):
        worker = pulpo.PulpoOptimizer(SAMPLE_PROJECT, 'technosphere', self.METHODS, '')
        worker.get_lci_data()
        self.ecar = worker.retrieve_activities(reference_products='transport')[0]
        self.elec = worker.retrieve_activities(reference_products='electricity')
        self.wind = worker.retrieve_activities(activities=['wind turbine'])[0]
        self.steam = worker.retrieve_activities(activities=['steam cycle'])[0]
        return worker

    def choices(self, cap=100):
        return {'electricity': {self.elec[0]: cap, self.elec[1]: cap}}

    def test_basic(self):
        worker = self.worker()
        for scale in (False, True):
            with self.subTest(scale=scale):
                worker.instantiate(choices=self.choices(), demand={self.ecar: 1}, scale=scale)
                self.assert_parity(worker)
                self.assertAlmostEqual(worker.instance.OBJ(), 0.103093, places=6)
                self.assertAlmostEqual(worker.instance.impacts_calculated[RESOURCES].value, 5.25773, places=5)

    def test_capacity_binds(self):
        worker = self.worker()
        for scale in (False, True):
            with self.subTest(scale=scale):
                worker.instantiate(choices={'electricity': {self.wind: 0.5, self.steam: 100}},
                                   demand={self.ecar: 1}, scale=scale)
                self.assert_parity(worker)

    def test_supply(self):
        worker = self.worker()
        for scale in (False, True):
            with self.subTest(scale=scale):
                worker.instantiate(choices=self.choices(), upper_limit={self.ecar: 1},
                                   lower_limit={self.ecar: 1}, scale=scale)
                self.assertEqual(len(worker.instance.PRODUCT_SUPPLY), 1)
                self.assert_parity(worker)
                self.assertAlmostEqual(worker.instance.OBJ(), 0.1, places=6)
                # No demand: the slack is the e-Car row's net output, (A s)_i.
                i = next(iter(worker.instance.PRODUCT_SUPPLY))
                s = np.array([worker.instance.scaling_vector[j].value for j in range(5)])
                output = (worker.lci_data['technology_matrix'] @ s)[i]
                self.assertAlmostEqual(worker.instance.slack[i].value, output, places=9)

    def test_process_bounds(self):
        worker = self.worker()
        oil = worker.retrieve_activities(activities=['oil extraction'])[0]
        for scale in (False, True):
            with self.subTest(scale=scale):
                worker.instantiate(choices=self.choices(), demand={self.ecar: 1},
                                   upper_limit={oil: 0.05}, lower_limit={self.steam: 0.1}, scale=scale)
                self.assert_parity(worker)

    def test_elementary_flow_limit(self):
        worker = self.worker()
        water = worker.retrieve_envflows(activities="Water, irrigation")[0]
        for scale in (False, True):
            with self.subTest(scale=scale):
                worker.instantiate(choices=self.choices(), demand={self.ecar: 1},
                                   upper_elem_limit={water: 5.2}, scale=scale)
                self.assert_parity(worker)
                self.assertAlmostEqual(worker.instance.OBJ(), 0.14237, places=5)
                self.assertAlmostEqual(worker.instance.inv_vector[3].value, 5.2, places=6)

    def test_impact_limits(self):
        worker = self.worker()
        for scale in (False, True):
            with self.subTest(scale=scale):
                worker.instantiate(choices=self.choices(), demand={self.ecar: 1},
                                   upper_imp_limit={RESOURCES: 5.2}, lower_imp_limit={AIR: 0.01},
                                   scale=scale)
                self.assertIn(RESOURCES, list(worker.instance.INDICATOR))
                self.assert_parity(worker)

    def test_dependent_constraint(self):
        worker = self.worker()
        dependent = {'wind_max_80_percent': {'left': {self.wind: 1}, 'right': {self.steam: 4}}}
        for scale in (False, True):
            with self.subTest(scale=scale):
                worker.instantiate(choices={'electricity': {self.wind: 100, self.steam: 100}},
                                   demand={self.ecar: 1}, dependent_constraints=dependent, scale=scale)
                self.assert_parity(worker)
                self.assertAlmostEqual(worker.instance.scaling_vector[3].value, 0.8247422674711002, places=6)
                self.assertAlmostEqual(worker.instance.scaling_vector[2].value, 0.20618556686777506, places=6)

    def test_goal_objective(self):
        worker = self.worker()
        # All goals met: objective 0 for every mix, so the optimum is not unique.
        cases = [({CLIMATE: 0.01, AIR: 100}, True), ({CLIMATE: 100, AIR: 100}, False), ({RESOURCES: 1}, True)]
        for goals, unique in cases:
            for scale in (False, True):
                with self.subTest(goals=goals, scale=scale):
                    worker.instantiate(choices=self.choices(), demand={self.ecar: 1},
                                       imp_goals=goals, objective='goal', scale=scale)
                    self.assert_parity(worker, unique=unique)
                    for h in worker.instance.GOAL_INDICATOR:
                        level = worker.instance.impacts[h].value / goals[h] - 1
                        self.assertAlmostEqual(worker.instance.transgression[h].value, max(0.0, level), places=9)

    def test_all_constraint_types_together(self):
        worker = self.worker()
        water = worker.retrieve_envflows(activities="Water, irrigation")[0]
        dependent = {'wind_cap': {'left': {self.wind: 1}, 'right': {self.steam: 9}}}
        for scale in (False, True):
            with self.subTest(scale=scale):
                worker.instantiate(choices={'electricity': {self.wind: 100, self.steam: 100}},
                                   demand={self.ecar: 1}, upper_elem_limit={water: 5.3},
                                   upper_imp_limit={RESOURCES: 5.3}, dependent_constraints=dependent,
                                   scale=scale)
                self.assert_parity(worker)

    @unittest.skipUnless(HAS_GUROBI, "gurobipy is not installed")
    def test_gurobi(self):
        worker = self.worker()
        worker.instantiate(choices=self.choices(), demand={self.ecar: 1})
        self.assert_parity(worker, solver_name='gurobi')

    def test_scipy_factorization(self):
        worker = self.worker()
        worker.instantiate(choices=self.choices(), demand={self.ecar: 1})
        self.assert_parity(worker, backend='scipy')
        self.assertEqual(reduced.build(worker).system.factorization.backend, 'scipy')

    def test_no_choices(self):
        """No free direction: the LP is an LCA plus a feasibility check."""
        worker = self.worker()
        worker.instantiate(demand={self.ecar: 1})
        results, obj = self.assert_parity(worker)
        self.assertEqual(results.n_variables, 0)
        worker.instantiate(demand={self.ecar: 1}, upper_limit={self.ecar: 0.5})
        self.assert_both_infeasible(worker)

    def test_infeasible_keeps_values(self):
        worker = self.worker()
        worker.instantiate(choices=self.choices(), demand={self.ecar: 1})
        worker.solve(formulation='reduced')
        before = alternative_values(worker)
        tight = {'lower_bound': -0.1, 'upper_bound': 0.1, 'upper_inv_bound': 0.1,
                 'lower_inv_bound': -0.1, 'lower_imp_bound': -0.1, 'upper_imp_bound': 0.1}
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter('always')
            worker.instantiate(choices=self.choices(), demand={self.ecar: 1}, default_limits=tight)
        self.assertTrue(any(issubclass(w.category, FutureWarning) for w in warned))
        with self.assertRaises(reduced.ReducedSolveError) as caught:
            worker.solve(formulation='reduced')
        self.assertIn(str(caught.exception.results.termination_condition),
                      ('infeasible', 'infeasibleOrUnbounded'))
        self.assertTrue(all(v is None for v in alternative_values(worker).values()))
        self.assertIsNotNone(before)

    def test_in_place_limit_change(self):
        """A limit changed in place on the instance is honoured, as by the full LP."""
        worker = self.worker()
        worker.instantiate(choices=self.choices(), demand={self.ecar: 1}, upper_imp_limit={RESOURCES: 10})
        for cap in (5.3, 5.2, 5.15):
            worker.instance.UPPER_IMP_LIMIT[RESOURCES].value = cap
            self.assert_parity(worker)
            self.assertLessEqual(worker.instance.impacts[RESOURCES].value, cap + 1e-9)

    def test_factorization_is_cached(self):
        worker = self.worker()
        worker.instantiate(choices=self.choices(), demand={self.ecar: 1})
        red = reduced.build(worker)
        worker.instantiate(choices=self.choices(cap=50), demand={self.ecar: 2})
        red2 = reduced.build(worker)
        self.assertIsNot(red, red2)
        self.assertIs(red.system, red2.system)
        worker.solve(formulation='reduced')
        worker.solve(formulation='reduced')
        self.assertIs(reduced.build(worker), red2)

    def test_custom_component_is_refused(self):
        worker = self.worker()
        worker.instantiate(choices=self.choices(), demand={self.ecar: 1})
        worker.instance.extra = pyo.Constraint(expr=worker.instance.scaling_vector[3] <= 0.5)
        with self.assertRaises(NotImplementedError):
            worker.solve(formulation='reduced')

    def test_lower_elementary_flow_limit(self):
        """A lower limit on fossil CO2, halfway between the optimum (wind) and
        steam only: it forces some steam back in."""
        worker = self.worker()
        co2 = worker.retrieve_envflows(activities="Carbon dioxide, fossil")[0]
        g = self.flow_index(worker, "Carbon dioxide, fossil")
        levels = []
        for wind_cap in (100, 0):
            worker.instantiate(choices={'electricity': {self.wind: wind_cap, self.steam: 100}},
                               demand={self.ecar: 1})
            worker.solve()
            levels.append(worker.instance.inv_flows[g].value)
        base = dict(choices={'electricity': {self.wind: 100, self.steam: 100}}, demand={self.ecar: 1},
                    upper_elem_limit={co2: 100})
        for scale in (False, True):
            with self.subTest(scale=scale):
                self.assert_binds(worker, dict(base, scale=scale),
                                  dict(base, lower_elem_limit={co2: sum(levels) / 2}, scale=scale))
                self.assert_parity(worker)

    def test_finite_default_limits_are_rows_and_warn(self):
        """Finite default limits bound every process: every bounded process is a
        row of the LP (the whole of S), and instantiate warns about it."""
        worker = self.worker()
        finite = {'lower_bound': 0.0, 'upper_bound': float('inf'), 'upper_inv_bound': float('inf'),
                  'lower_inv_bound': -float('inf'), 'lower_imp_bound': -float('inf'),
                  'upper_imp_bound': float('inf')}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            worker.instantiate(choices=self.choices(), demand={self.ecar: 1}, default_limits=finite)
        self.assertTrue(any(issubclass(w.category, FutureWarning) and 'may be deprecated' in str(w.message)
                            for w in caught))
        results, _ = self.assert_parity(worker)
        n = len(worker.instance.PROCESS)
        bound_rows = results.n_rows - 1                                      # one category row
        self.assertEqual(bound_rows, n)
        self.assertEqual(len(reduced.build(worker).system._process_rows), n)
        self.assertEqual(results.rounds, 1)

    def test_a_bound_off_the_alternatives_is_a_row_from_the_start(self):
        """A bound on a process that is not an alternative can be what keeps the
        problem bounded; it is in the LP from the first solve."""
        worker = self.worker()
        oil = worker.retrieve_activities(activities=['oil extraction'])[0]
        worker.instantiate(choices={'electricity': [self.wind, self.steam]}, demand={self.ecar: 1},
                           lower_limit={self.steam: -float('inf'), oil: 0.0})
        results, _ = self.assert_parity(worker)
        self.assertEqual(results.rounds, 1)

    def test_a_remote_bound_side_that_binds_is_imposed(self):
        """A bound side far beyond the problem's scale is withheld; when the
        problem is unbounded without it, it is imposed and the LP solved again."""
        worker = self.worker()
        oil = worker.retrieve_activities(activities=['oil extraction'])[0]
        worker.instantiate(choices={'electricity': [self.wind, self.steam]}, demand={self.ecar: 1},
                           lower_limit={self.steam: -float('inf'), oil: -1e10})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            results, _ = self.assert_parity(worker)
        self.assertEqual(results.rounds, 2)
        j = worker.lci_data['process_map'][oil.key]
        self.assertAlmostEqual(worker.instance.scaling_vector[j].value, -1e10, delta=1e-9 * 1e10)
        # It binds, so it is not reported as a bound that did not.
        self.assertFalse([w for w in caught if 'far beyond' in str(w.message)])

    def test_a_large_supply_sets_the_problem_scale(self):
        """Without demand, a fixed supply drives the model: its size counts, so a
        supply and a capacity of millions are not withheld or reported."""
        worker = self.worker()
        wind = worker.lci_data['process_map'][self.wind.key]
        for scale in (False, True):
            with self.subTest(scale=scale):
                worker.instantiate(choices={'electricity': {self.wind: 2e6, self.steam: float('inf')}},
                                   upper_limit={self.ecar: 4e6}, lower_limit={self.ecar: 4e6}, scale=scale)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    results, _ = self.assert_parity(worker)
                self.assertEqual(results.rounds, 1)
                self.assertAlmostEqual(worker.instance.scaling_vector[wind].value, 2e6, delta=1e-9 * 2e6)
                self.assertFalse([w for w in caught if 'far beyond' in str(w.message)])

    def test_none_means_no_limit(self):
        """None in a choice capacity or a limit dict reads as +-inf."""
        worker = self.worker()
        worker.instantiate(choices={'electricity': {self.wind: float('inf'), self.steam: float('inf')}},
                           demand={self.ecar: 1}, upper_limit={self.ecar: float('inf')})
        worker.solve()
        reference = worker.instance.OBJ()
        worker.instantiate(choices={'electricity': {self.wind: None, self.steam: None}},
                           demand={self.ecar: 1}, upper_limit={self.ecar: None}, lower_limit={self.wind: None})
        pmap = worker.lci_data['process_map']
        for j in (pmap[self.wind.key], pmap[self.steam.key], pmap[self.ecar.key]):
            self.assertIsNone(worker.instance.scaling_vector[j].ub)
        self.assertIsNone(worker.instance.scaling_vector[pmap[self.wind.key]].lb)
        results, objective = self.assert_parity(worker)
        self.assertAlmostEqual(objective, reference, places=9)

    def test_huge_finite_bounds_warn(self):
        worker = self.worker()
        worker.instantiate(choices={'electricity': {self.wind: 1e10, self.steam: float('inf')}},
                           demand={self.ecar: 1})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            worker.solve(formulation='reduced')
        remote = [w for w in caught if 'far beyond' in str(w.message)]
        self.assertEqual(len(remote), 1)
        self.assertIn('wind', str(remote[0].message))
        self.assertTrue(os.path.samefile(remote[0].filename, __file__))  # points at the caller
        worker.instantiate(choices={'electricity': {self.wind: None, self.steam: float('inf')}},
                           demand={self.ecar: 1})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            worker.solve(formulation='reduced')
        self.assertFalse([w for w in caught if 'far beyond' in str(w.message)])

    def test_infinite_default_limits_do_not_warn(self):
        worker = self.worker()
        infinite = {'lower_bound': -float('inf'), 'upper_bound': float('inf'), 'upper_inv_bound': 1e9,
                    'lower_inv_bound': -1e9, 'lower_imp_bound': -1e9, 'upper_imp_bound': 1e9}
        for limits in (None, infinite):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                worker.instantiate(choices=self.choices(), demand={self.ecar: 1}, default_limits=limits)
            self.assertFalse([w for w in caught if issubclass(w.category, FutureWarning)])

    def test_fixed_scaling_vector(self):
        """A fixed activity is honoured in the right units on every solve and keeps its value."""
        worker = self.worker()
        for scale in (False, True):
            with self.subTest(scale=scale):
                worker.instantiate(choices={'electricity': {self.wind: 100, self.steam: 100}},
                                   demand={self.ecar: 1}, scale=scale)
                worker.instance.scaling_vector[3].fix(0.3)      # model units: 0.3 * col_scale original
                level = 0.3 * scaling.col_factor(worker.instance, 3)
                objectives = []
                for formulation in ('reduced', 'reduced', 'full', 'reduced', 'full'):
                    worker.solve(formulation=formulation)
                    objectives.append(pyo.value(worker.instance.OBJ))
                    self.assertAlmostEqual(worker.instance.scaling_vector[3].value, level, places=12)
                self.assertLessEqual(max(objectives) - min(objectives), 1e-9)

    def test_fixed_and_bounded_auxiliary_variables(self):
        worker = self.worker()
        water = worker.retrieve_envflows(activities="Water, irrigation")[0]
        g = self.flow_index(worker, "Water, irrigation")
        for scale in (False, True):
            with self.subTest(case='fixed impact', scale=scale):
                worker.instantiate(choices=self.choices(), demand={self.ecar: 1}, scale=scale)
                worker.instance.impacts[AIR].fix(0.05)
                self.assert_parity(worker)
                self.assertEqual(worker.instance.impacts[AIR].value, 0.05)
            with self.subTest(case='fixed flow', scale=scale):
                worker.instantiate(choices=self.choices(), demand={self.ecar: 1},
                                   upper_elem_limit={water: 10}, scale=scale)
                worker.instance.inv_vector[g].fix(5.0)
                self.assert_parity(worker)
                self.assertEqual(worker.instance.inv_vector[g].value, 5.0)
            with self.subTest(case='transgression bound', scale=scale):
                worker.instantiate(choices=self.choices(), demand={self.ecar: 1},
                                   imp_goals={CLIMATE: 0.05}, objective='goal', scale=scale)
                worker.instance.transgression[CLIMATE].setub(0.5)
                self.assert_both_infeasible(worker)
            with self.subTest(case='slack bound', scale=scale):
                worker.instantiate(choices=self.choices(), upper_limit={self.ecar: 1},
                                   lower_limit={self.ecar: 1}, scale=scale)
                worker.instance.slack[next(iter(worker.instance.PRODUCT_SUPPLY))].setub(0.5)
                self.assert_both_infeasible(worker)
            with self.subTest(case='deactivated goal row', scale=scale):
                worker.instantiate(choices=self.choices(), demand={self.ecar: 1},
                                   imp_goals={CLIMATE: 0.01}, objective='goal', scale=scale)
                worker.instance.TRANSGRESSION_CNSTR[CLIMATE].deactivate()
                self.assert_parity(worker, unique=False)
                self.assertEqual(worker.instance.transgression[CLIMATE].value, 0.0)

    def test_technosphere_edited_in_place(self):
        """An in-place change of the technosphere matrix is not served from the cache."""
        worker = self.worker()
        worker.instantiate(choices={'electricity': {self.wind: 0.5, self.steam: 100}}, demand={self.ecar: 1})
        self.assert_parity(worker)
        A = worker.lci_data['technology_matrix']
        A[0, 2] = A[0, 2] * 1.8
        worker.instantiate(choices={'electricity': {self.wind: 0.5, self.steam: 100}}, demand={self.ecar: 1})
        self.assert_parity(worker)

    def test_worker_can_be_copied_after_reduced_solve(self):
        """solve_MC pickles the worker for its process pool; a reduced solve must not prevent it."""
        try:
            import cloudpickle  # a dependency of joblib >= 1.6, which no longer vendors it
        except ImportError:
            from joblib.externals import cloudpickle
        worker = self.worker()
        worker.instantiate(choices=self.choices(), demand={self.ecar: 1})
        worker.solve(formulation='reduced')
        objective = pyo.value(worker.instance.OBJ)
        clone = copy.deepcopy(worker)
        self.assertIsNotNone(cloudpickle.dumps(worker))
        clone.solve(formulation='reduced')
        self.assertAlmostEqual(pyo.value(clone.instance.OBJ), objective, places=12)
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                factorization = reduced.Factorization(worker.lci_data['technology_matrix'], backend=backend)
                restored = pickle.loads(pickle.dumps(factorization))
                b = np.arange(1.0, 6.0)
                np.testing.assert_allclose(restored.solve(b), factorization.solve(b), rtol=1e-12)
                self.assertEqual(restored.refactorizations, 1)

    def test_unknown_method(self):
        worker = self.worker()
        worker.instantiate(choices=self.choices(), demand={self.ecar: 1})
        with self.assertRaises(ValueError):
            worker.solve(formulation='banana')
        with self.assertRaises(ValueError):
            worker.solve(formulation='reduced', solver_name='cplex')

    def test_highs_options(self):
        """HiGHS options reach the solver in both formulations; a bad name or value raises."""
        worker = self.worker()
        worker.instantiate(choices=self.choices(), demand={self.ecar: 1})
        for formulation in ('full', 'reduced'):
            for options in ({'TimeLimit': 10}, {'time_limit': 'abc'}, {'primal_feasibility_tolerance': -1.0}):
                with self.subTest(formulation=formulation, options=options):
                    with self.assertRaisesRegex(ValueError, 'HiGHS option'):
                        worker.solve(formulation=formulation, options=options)
            worker.solve(formulation=formulation, options={'time_limit': 600.0, 'presolve': 'off'})
            self.assertAlmostEqual(worker.instance.OBJ(), 0.103093, places=6)
        # An iteration limit of 0 stops either solve before the optimum.
        stop = {'simplex_iteration_limit': 0, 'presolve': 'off'}
        with self.assertRaisesRegex(optimizer.SolveError, 'maxIterations'):
            worker.solve(formulation='full', options=stop)
        with self.assertRaises(reduced.ReducedSolveError):
            worker.solve(formulation='reduced', options=stop)

    def test_options_are_not_passed_to_neos(self):
        worker = self.worker()
        worker.instantiate(choices=self.choices(), demand={self.ecar: 1})
        with mock.patch.object(optimizer, 'solve_neos', return_value=(None, worker.instance)),                 warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            optimizer.solve_model(worker.instance, solver_name='cplex', options={'threads': 1})
        self.assertTrue(any('not passed to NEOS' in str(w.message) for w in caught))


# ---------------------------------------------------------------------------
# sample project, background + foreground (methanol and ozone)
# ---------------------------------------------------------------------------

class TestReducedSampleForeground(ParityMixin, unittest.TestCase):

    def worker(self, methods=None):
        worker = pulpo.PulpoOptimizer(SAMPLE_PROJECT, ['background_db', 'foreground_db'],
                                      methods or {CLIMATE: 1}, '')
        worker.get_lci_data()
        get = worker.retrieve_processes
        self.methanol = get(reference_products='methanol')[0]
        self.ozone = get(reference_products='ozone')[0]
        self.electricity = get(processes=['wind electricity', 'natural gas electricity'])
        self.hydrogen = get(processes=['hydrogen SMR', 'hydrogen electrolysis'])
        self.oxygen = get(processes=['O2-market', 'O2 ASU'])
        self.byproduct = get(processes=['O2-byproduct'])[0]
        return worker

    def choices(self, caps=(float('inf'),) * 6):
        return {
            'Electricity': {self.electricity[0]: caps[0], self.electricity[1]: caps[1]},
            'Hydrogen': {self.hydrogen[0]: caps[2], self.hydrogen[1]: caps[3]},
            'Oxygen': {self.oxygen[0]: caps[4], self.oxygen[1]: caps[5]},
        }

    def test_methanol_ozone(self):
        worker = self.worker()
        for scale in (False, True):
            with self.subTest(scale=scale):
                worker.instantiate(choices=self.choices(), demand={self.methanol: 1, self.ozone: 2},
                                   lower_limit={self.byproduct: 0}, scale=scale)
                self.assert_parity(worker)
                self.assertAlmostEqual(worker.instance.OBJ(), 1.760427, places=6)

    def base(self, **kwargs):
        return dict(dict(choices=self.choices(), demand={self.methanol: 1, self.ozone: 2},
                         lower_limit={self.byproduct: 0}), **kwargs)

    def by_name(self, name):
        return next(p for p in self.electricity + self.hydrogen + self.oxygen if p['name'] == name)

    def test_capacities(self):
        worker = self.worker()
        wind, elyz, asu = self.by_name('wind electricity'), self.by_name('hydrogen electrolysis'), self.by_name('O2 ASU')
        for caps in ({wind: 5.0}, {elyz: 0.1, asu: 1.9}):
            choices = {cat: {p: caps.get(p, float('inf')) for p in procs} for cat, procs in self.choices().items()}
            for scale in (False, True):
                with self.subTest(caps=sorted(p['name'] for p in caps), scale=scale):
                    self.assert_binds(worker, self.base(scale=scale), self.base(choices=choices, scale=scale))
                    self.assert_parity(worker)

    def test_flow_and_impact_limits(self):
        worker = self.worker(methods={CLIMATE: 1, AIR: 0})
        water = worker.retrieve_envflows(activities="Water, irrigation")[0]
        worker.instantiate(**self.base())
        worker.solve()
        level = worker.instance.inv_flows[self.flow_index(worker, "Water, irrigation")].value
        cases = {'upper flow limit': dict(upper_elem_limit={water: 0.9 * level}),
                 'lower impact limit': dict(lower_imp_limit={AIR: 0.01})}
        for name, extra in cases.items():
            for scale in (False, True):
                with self.subTest(case=name, scale=scale):
                    self.assert_binds(worker, self.base(scale=scale), self.base(scale=scale, **extra))
                    self.assert_parity(worker)

    def test_dependent_goal_supply(self):
        worker = self.worker()
        smr, elyz = self.by_name('hydrogen SMR'), self.by_name('hydrogen electrolysis')
        dependent = {'electrolysis_share': {'left': {elyz: 1}, 'right': {smr: 0.5}}}
        for scale in (False, True):
            with self.subTest(case='dependent', scale=scale):
                self.assert_binds(worker, self.base(scale=scale),
                                  self.base(dependent_constraints=dependent, scale=scale))
                self.assert_parity(worker)
            with self.subTest(case='goal', scale=scale):
                worker.instantiate(**self.base(imp_goals={CLIMATE: 0.9}, objective='goal', scale=scale))
                self.assert_parity(worker)
                self.assertGreater(worker.instance.transgression[CLIMATE].value, 0.5)
            with self.subTest(case='supply', scale=scale):
                worker.instantiate(**self.base(demand={self.ozone: 2}, upper_limit={self.methanol: 1},
                                               lower_limit={self.byproduct: 0, self.methanol: 1}, scale=scale))
                self.assertEqual(len(worker.instance.PRODUCT_SUPPLY), 1)
                self.assert_parity(worker)


# ---------------------------------------------------------------------------
# rice husk example
# ---------------------------------------------------------------------------

class TestReducedRice(ParityMixin, unittest.TestCase):

    def worker(self, methods="('my project', 'climate change')"):
        worker = pulpo.PulpoOptimizer('rice_husk_example', 'rice_husk_example_db', methods, '')
        worker.get_lci_data()
        get = worker.retrieve_processes
        self.factory = get(reference_products='Processed rice (in Mt)')[0]
        self.collections = get(processes=[f"Rice husk collection {i}" for i in range(1, 6)])
        self.boilers = get(processes=["Natural gas boiler", "Wood pellet boiler", "Rice husk boiler"])
        self.auxiliar = get(processes=["Rice husk market", "Burning of rice husk"])
        self.transport = get(processes=["Transportation by truck"])[0]
        self.pellets = get(processes=["Wood pellet supply"])[0]
        return worker

    def free(self, worker):
        """The husk market consumes one unit of its own pooled category per unit
        it produces: a zero-cost cycle whose level the LP leaves arbitrary."""
        return (worker.lci_data['process_map'][self.auxiliar[0].key],)

    def choices(self, husk_cap=float('inf'), aux_cap=10.0):
        # With the notebook's auxiliary capacity of 1e10 an LP may park the free
        # husk-market cycle at 1e10, and every activity then carries ~1e10 * eps
        # of cancellation noise; test_notebook_capacity covers that setting.
        return {'Rice Husk (Mt)': {c: husk_cap for c in self.collections},
                'Thermal Energy (TWh)': {b: float('inf') for b in self.boilers},
                'Auxiliar': {a: aux_cap for a in self.auxiliar}}

    def test_notebook_sequence(self):
        worker = self.worker()
        cases = [
            dict(choices=self.choices(), demand={self.factory: 1}),
            dict(choices=self.choices(0.03), demand={self.factory: 1}),
            dict(choices=self.choices(0.03), demand={self.factory: 1}, upper_limit={self.transport: 0.37}),
            dict(choices=self.choices(0.03), demand={self.factory: 1}, upper_limit={self.pellets: 0.1}),
            dict(choices=self.choices(), demand={self.factory: 1, 'Thermal Energy (TWh)': 10}),
        ]
        for k, case in enumerate(cases):
            for scale in (False, True):
                with self.subTest(case=k, scale=scale):
                    worker.instantiate(scale=scale, **case)
                    self.assert_parity(worker, free=self.free(worker))

    def test_notebook_capacity(self):
        worker = self.worker()
        for husk_cap in (1e10, 0.03):
            with self.subTest(husk_cap=husk_cap):
                worker.instantiate(choices=self.choices(husk_cap, aux_cap=1e10), demand={self.factory: 1})
                # A vertex with the free cycle at 1e10 holds every activity to
                # ~1e10 * eps = 2e-6 only, in either formulation.
                self.assert_parity(worker, unique=False, feas_tol=1e-5)

    def test_demand_sweep(self):
        worker = self.worker()
        for demand in (0.05, 0.4, 1.0):
            with self.subTest(demand=demand):
                worker.instantiate(choices=self.choices(0.03), demand={self.factory: demand},
                                   upper_limit={self.pellets: 0.1})
                self.assert_parity(worker, free=self.free(worker))

    def test_limits_dependent_goal_supply(self):
        """Cost limits halfway between the GWP optimum and the cost optimum, as an
        impact limit and as a limit on the economic elementary flow."""
        worker = self.worker(methods={CLIMATE: 1, ECONOMIC: 0})
        boiler = {p['name']: p for p in self.boilers}
        base = dict(choices=self.choices(0.03), demand={self.factory: 1})
        g = self.flow_index(worker, "Economic Flow")
        cost, flow = {}, {}
        for objective in ('gwp', 'cost'):
            worker.method = {CLIMATE: 1, ECONOMIC: 0} if objective == 'gwp' else {CLIMATE: 0, ECONOMIC: 1}
            worker.instantiate(**base)
            worker.solve()
            values = {h: worker.instance.impacts[h].value for h in worker.instance.impacts}
            values.update({h: v.value for h, v in getattr(worker.instance, 'impacts_calculated', {}).items()})
            cost[objective] = values[ECONOMIC]
            flow[objective] = worker.instance.inv_flows[g].value
        worker.method = {CLIMATE: 1, ECONOMIC: 0}
        economic = worker.retrieve_envflows(activities="Economic Flow")[0]
        cases = {
            'impact limit': dict(upper_imp_limit={ECONOMIC: (cost['gwp'] + cost['cost']) / 2}),
            'flow limit': dict(upper_elem_limit={economic: (flow['gwp'] + flow['cost']) / 2}),
            'dependent': dict(dependent_constraints={'pellets_vs_husk': {
                'left': {boiler['Wood pellet boiler']: 1}, 'right': {boiler['Rice husk boiler']: 0.5}}}),
        }
        for name, extra in cases.items():
            for scale in (False, True):
                with self.subTest(case=name, scale=scale):
                    self.assert_binds(worker, dict(base, scale=scale), dict(base, scale=scale, **extra))
                    self.assert_parity(worker, free=self.free(worker))
        for scale in (False, True):
            with self.subTest(case='goal', scale=scale):
                worker.instantiate(**dict(base, imp_goals={CLIMATE: 0.5}, objective='goal', scale=scale))
                self.assert_parity(worker, free=self.free(worker))
                self.assertGreater(worker.instance.transgression[CLIMATE].value, 0.5)
            with self.subTest(case='supply', scale=scale):
                worker.instantiate(**dict(base, upper_limit={self.transport: 0.45},
                                          lower_limit={self.transport: 0.45}, scale=scale))
                self.assertEqual(len(worker.instance.PRODUCT_SUPPLY), 1)
                self.assert_parity(worker, free=self.free(worker))

    def test_epsilon_constraint(self):
        worker = self.worker(methods={CLIMATE: 0, ECONOMIC: 1})
        worker.instantiate(choices=self.choices(0.03), demand={self.factory: 1},
                           upper_limit={self.pellets: 0.1})
        worker.solve()
        gwp_free = worker.instance.impacts_calculated[CLIMATE].value
        worker.instantiate(choices=self.choices(0.03), demand={self.factory: 1},
                           upper_limit={self.pellets: 0.1}, upper_imp_limit={CLIMATE: gwp_free})
        for frac in (1.0, 0.9, 0.8):
            with self.subTest(frac=frac):
                worker.instance.UPPER_IMP_LIMIT[CLIMATE].value = frac * gwp_free
                self.assert_parity(worker, free=self.free(worker))


# ---------------------------------------------------------------------------
# SOC demo (hydrogen route choice)
# ---------------------------------------------------------------------------

class TestReducedSocDemo(ParityMixin, unittest.TestCase):

    def worker(self):
        worker = pulpo.PulpoOptimizer(SOC_PROJECT, ['soc_demo_background_db', 'soc_demo_foreground_db'],
                                      {str(SOC_METHOD): 1}, '')
        worker.get_lci_data()
        get = worker.retrieve_processes
        self.ammonia = get(processes=['ammonia synthesis'])[0]
        self.smr = get(processes=['hydrogen SMR'])[0]
        self.elyz = get(processes=['hydrogen electrolysis'])[0]
        return worker

    def base(self, elyz_cap=float('inf'), **kwargs):
        return dict(dict(choices={'hydrogen': {self.smr: float('inf'), self.elyz: elyz_cap}},
                         demand={self.ammonia: 1}), **kwargs)

    def test_route_choice(self):
        """Electrolysis is the unconstrained choice; every constraint below moves
        the optimum towards SMR."""
        worker = self.worker()
        methane = worker.retrieve_envflows(activities="Methane, fossil")[0]
        worker.instantiate(**self.base(elyz_cap=0.0))
        worker.solve()
        ch4_smr = worker.instance.inv_flows[self.flow_index(worker, "Methane, fossil")].value
        cases = {
            'capacity': self.base(elyz_cap=0.09),
            'dependent': self.base(dependent_constraints={'d': {'left': {self.elyz: 1}, 'right': {self.smr: 1}}}),
            'lower flow limit': self.base(upper_elem_limit={methane: 1e3},
                                          lower_elem_limit={methane: 0.5 * ch4_smr}),
        }
        for name, constrained in cases.items():
            for scale in (False, True):
                with self.subTest(case=name, scale=scale):
                    self.assert_binds(worker, self.base(scale=scale), dict(constrained, scale=scale))
                    self.assert_parity(worker)

    def test_goal_and_supply(self):
        worker = self.worker()
        for scale in (False, True):
            with self.subTest(case='goal', scale=scale):
                worker.instantiate(**self.base(elyz_cap=0.09, imp_goals={str(SOC_METHOD): 1.0},
                                               objective='goal', scale=scale))
                self.assert_parity(worker)
                self.assertGreater(worker.instance.transgression[str(SOC_METHOD)].value, 0.5)
            with self.subTest(case='supply', scale=scale):
                worker.instantiate(**self.base(elyz_cap=0.09, demand={}, upper_limit={self.ammonia: 1},
                                               lower_limit={self.ammonia: 1}, scale=scale))
                self.assertEqual(len(worker.instance.PRODUCT_SUPPLY), 1)
                self.assert_parity(worker)


# ---------------------------------------------------------------------------
# electricity toy, static fallback of the time-dependent optimizer
# ---------------------------------------------------------------------------

class TestReducedElecStatic(ParityMixin, unittest.TestCase):

    def worker(self):
        worker = pulpo_time.PulpoOptimizerTime(ELEC_PROJECT, ELEC_DB, {str(("GWP", "100a")): 1}, '')
        worker.intervention_matrix = 'biosphere3'
        worker.get_lci_data()
        self.acts = {name: worker.retrieve_activities(activities=[name])[0]
                     for name in ("solar", "coal", "battery_charge", "battery_hold",
                                  "battery_discharge", "battery_holdtm1")}
        return worker

    def test_static_dispatch(self):
        worker = self.worker()
        a = self.acts
        choices = {ELECTRICITY_CHOICE: {a['solar']: 4.0, a['coal']: float('inf'), a['battery_discharge']: float('inf')},
                   CHARGE_PRODUCT_CHOICE: {a['battery_charge']: float('inf'), a['battery_hold']: float('inf')}}
        upper = {a['battery_holdtm1']: 0.0}
        lower = {a['battery_charge']: 0.0, a['battery_hold']: 0.0, a['battery_discharge']: 0.0}
        for scale in (False, True):
            with self.subTest(scale=scale):
                worker.instantiate(choices=choices, demand={ELECTRICITY_CHOICE: 10.0},
                                   upper_limit=upper, lower_limit=lower, scale=scale)
                self.assert_parity(worker)
                self.assertAlmostEqual(worker.instance.OBJ(), 6.0, places=7)

    def test_time_dependent_is_refused(self):
        worker = self.worker()
        a = self.acts
        steps = [0, 1]
        choices = {ELECTRICITY_CHOICE: {a['solar']: float('inf'), a['coal']: float('inf'), a['battery_discharge']: float('inf')},
                   CHARGE_PRODUCT_CHOICE: {a['battery_charge']: float('inf'), a['battery_hold']: float('inf')}}
        worker.instantiate(choices=choices, demand={t: {ELECTRICITY_CHOICE: 1.0} for t in steps},
                           time_steps=steps)
        with self.assertRaises(NotImplementedError):
            worker.solve(formulation='reduced')
        with self.assertRaises(NotImplementedError):
            reduced.build(worker)


# ---------------------------------------------------------------------------
# far limits and bounds: withheld in round 1, imposed in round 2
# ---------------------------------------------------------------------------

class TestReducedRemoteRounds(ParityMixin, unittest.TestCase):
    """The second round of the reduced solve, with a problem built for it.

    Routes A and B make product P, of which consumer C needs one unit. A uses
    less land (1 vs 2) but emits more CO2 (2e8 vs 1e8 per unit, the size of
    ecoinvent's large characterized flows) and draws 1e7 units of a bulk input
    X. The consumer is linked to B, so the reduced system's base point runs
    B, and the problem's scale is about 1: a CO2 limit of 1.5e8 or a bound of
    5e6 on X lies far beyond it. Both are withheld in round 1, whose optimum
    (all A) violates them, and must be imposed in round 2.
    """

    PROJECT = 'reduced_remote_rounds'
    LAND, CO2 = "('remote', 'land')", "('remote', 'co2')"

    @classmethod
    def setUpClass(cls):
        import bw2data as bd
        bd.projects.set_current(cls.PROJECT)
        bd.Database('biosphere3').write({
            ('biosphere3', 'co2'): {'name': 'CO2', 'type': 'emission', 'unit': 'kg', 'categories': ('air',)},
            ('biosphere3', 'land'): {'name': 'land', 'type': 'natural resource', 'unit': 'm2',
                                     'categories': ('land',)},
        })

        def act(code, name, product, exchanges):
            return {'name': name, 'reference product': product, 'unit': 'unit', 'location': 'GLO',
                    'exchanges': [{'input': ('far', code), 'amount': 1.0, 'type': 'production'}] + exchanges}

        co2, land = ('biosphere3', 'co2'), ('biosphere3', 'land')
        bd.Database('far').write({
            ('far', 'X'): act('X', 'bulk input', 'bulk', []),
            ('far', 'A'): act('A', 'route A', 'P', [
                {'input': co2, 'amount': 2e8, 'type': 'biosphere'},
                {'input': land, 'amount': 1.0, 'type': 'biosphere'},
                {'input': ('far', 'X'), 'amount': 1e7, 'type': 'technosphere'}]),
            ('far', 'B'): act('B', 'route B', 'P', [
                {'input': co2, 'amount': 1e8, 'type': 'biosphere'},
                {'input': land, 'amount': 2.0, 'type': 'biosphere'}]),
            # Linked to B, so the reduced system's base point runs B and X stays out of the scale.
            ('far', 'C'): act('C', 'consumer', 'Q', [{'input': ('far', 'B'), 'amount': 1.0, 'type': 'technosphere'}]),
        })
        for name, flow in (('land', land), ('co2', co2)):
            method = bd.Method(('remote', name))
            method.register()
            method.write([(flow, 1.0)])

    def worker(self):
        worker = pulpo.PulpoOptimizer(self.PROJECT, 'far', {self.LAND: 1, self.CO2: 0}, '')
        worker.get_lci_data()
        get = worker.retrieve_processes
        self.a, self.b, self.c, self.x = (get(processes=[name])[0]
                                          for name in ('route A', 'route B', 'consumer', 'bulk input'))
        return worker

    def solve(self, worker, **limits):
        worker.instantiate(choices={'P': [self.a, self.b]}, demand={self.c: 1}, **limits)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            results, objective = self.assert_parity(worker)
        # Imposed in round 2, so it is not reported as a bound that did not bind.
        self.assertFalse([w for w in caught if 'far beyond' in str(w.message)])
        return results, objective

    def test_unconstrained_takes_one_round(self):
        results, objective = self.solve(self.worker())
        self.assertEqual(results.rounds, 1)
        self.assertAlmostEqual(objective, 1.0, places=9)                          # all A

    def test_a_far_impact_limit_that_binds_is_imposed(self):
        worker = self.worker()
        results, objective = self.solve(worker, upper_imp_limit={self.CO2: 1.5e8})
        self.assertEqual(results.rounds, 2)
        self.assertAlmostEqual(objective, 1.5, places=9)                          # half A, half B
        self.assertAlmostEqual(worker.instance.impacts_calculated[self.CO2].value, 1.5e8, delta=1e-6 * 1.5e8)

    def test_a_far_process_bound_that_is_reached_is_imposed(self):
        worker = self.worker()
        results, objective = self.solve(worker, upper_limit={self.x: 5e6})
        self.assertEqual(results.rounds, 2)
        self.assertAlmostEqual(objective, 1.5, places=9)
        j = worker.lci_data['process_map'][self.x.key]
        self.assertAlmostEqual(worker.instance.scaling_vector[j].value, 5e6, delta=1e-9 * 5e6)


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

class TestReducedSystem(unittest.TestCase):
    """The projections against dense linear algebra on a random sparse matrix."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        n = 300
        A = sp.random(n, n, density=0.02, random_state=4, format='csr')
        cls.A = (A + sp.eye(n) * 4).tocsr()
        cls.columns = np.array([5, 17, 42, 99, 250])
        cls.Ainv = np.linalg.inv(cls.A.toarray())
        cls.S = cls.Ainv[:, cls.columns]
        cls.rng = rng

    def systems(self):
        for backend in BACKENDS:
            yield backend, reduced.ReducedSystem(reduced.Factorization(self.A, backend=backend), self.columns)

    def test_project_adjoint_and_forward(self):
        few = sp.random(3, self.A.shape[0], density=0.1, random_state=5, format='csr')
        many = sp.random(12, self.A.shape[0], density=0.1, random_state=6, format='csr')
        for backend, system in self.systems():
            with self.subTest(backend=backend):
                np.testing.assert_allclose(system.project(few), few @ self.S, atol=1e-12)    # adjoint path
                np.testing.assert_allclose(system.project(many), many @ self.S, atol=1e-12)  # forward path

    def test_rows_base_recover(self):
        f = self.rng.standard_normal(self.A.shape[0])
        f[self.columns] = 0.0
        v = self.rng.standard_normal(len(self.columns))
        for backend, system in self.systems():
            with self.subTest(backend=backend):
                np.testing.assert_allclose(system.rows([7, 250, 7]), self.S[[7, 250, 7]], atol=1e-12)
                s0 = system.base(f)
                np.testing.assert_allclose(s0, self.Ainv @ f, atol=1e-12)
                np.testing.assert_allclose(system.recover(f, v), s0 + self.S @ v, atol=1e-12)

    def test_chunked_projection(self):
        original = reduced.CHUNK_ENTRIES
        reduced.CHUNK_ENTRIES = 2 * self.A.shape[0]   # two columns per block
        try:
            many = sp.random(20, self.A.shape[0], density=0.1, random_state=7, format='csr')
            few = sp.random(4, self.A.shape[0], density=0.1, random_state=8, format='csr')
            for backend, system in self.systems():
                with self.subTest(backend=backend):
                    np.testing.assert_allclose(system.project(many), many @ self.S, atol=1e-12)
                    np.testing.assert_allclose(system.project(few), few @ self.S, atol=1e-12)
        finally:
            reduced.CHUNK_ENTRIES = original

    def test_badly_scaled_matrix_is_solved_to_roundoff(self):
        """Rows and columns scaled over 12 orders of magnitude, as in an ecoinvent
        technosphere. Unrefined SuperLU leaves a componentwise backward error of
        about 4e-2 in the transposed solve here; every backend must reach roundoff."""
        rng = np.random.default_rng(0)
        n = 2000
        T = sp.random(n, n, density=0.002, random_state=0, format='csr')
        T.data *= 0.9 / abs(T).sum(axis=0).max()
        A = (sp.diags(10.0 ** rng.uniform(-6, 6, n)) @ (sp.eye(n) - T)
             @ sp.diags(10.0 ** rng.uniform(-6, 6, n))).tocsr()
        b = rng.standard_normal((n, 3))

        def backward_error(M, x):
            return (np.abs(M @ x - b) / (abs(M) @ np.abs(x) + np.abs(b))).max()

        for backend in BACKENDS:
            with self.subTest(backend=backend):
                f = reduced.Factorization(A, backend=backend)
                self.assertLess(backward_error(A, f.solve(b)), 1e-14)
                self.assertLess(backward_error(A.T.tocsr(), f.solve(b, transpose=True)), 1e-14)

    def test_auto_backend_order(self):
        """PARDISO first, then UMFPACK, then SciPy's SuperLU, by what is installed."""
        self.assertEqual(reduced._resolve_backend('auto'), BACKENDS[0])
        no_pardiso = {'pypardiso': None, 'pypardiso.scipy_aliases': None}
        no_umfpack = {'scikits': None, 'scikits.umfpack': None}
        with mock.patch.dict(sys.modules, no_pardiso):
            self.assertEqual(reduced._resolve_backend('auto'), 'umfpack' if 'umfpack' in BACKENDS else 'scipy')
            with self.assertRaises(ImportError):
                reduced._resolve_backend('pardiso')
        with mock.patch.dict(sys.modules, {**no_pardiso, **no_umfpack}):
            self.assertEqual(reduced._resolve_backend('auto'), 'scipy')
            with self.assertRaises(ImportError):
                reduced._resolve_backend('umfpack')

    def test_non_square_is_refused(self):
        with self.assertRaises(ValueError):
            reduced.Factorization(sp.eye(4, 5, format='csr'))

    def test_singular_is_refused(self):
        A = sp.csr_matrix(np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 1.0, 3.0]]))
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                with self.assertRaises(ValueError):
                    reduced.Factorization(A, backend=backend)


if __name__ == '__main__':
    unittest.main()
