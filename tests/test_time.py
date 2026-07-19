"""Tests for the time-indexed PULPO formulation (``pulpo.pulpo_time``).

Systematic version of ``notebooks/elec_time_toy.ipynb``: the same toy
electricity system (solar / coal / four-activity battery) is dispatched over

- one day at hourly resolution (intra-day storage shifting), and
- two weeks at daily resolution (multi-day storage propagation),

each without a battery and with lossy / near-ideal round-trip retention
factors ``K``. The optimal total CO2 of every scenario is asserted against
the reference values reproduced by the notebook, together with the physical
consistency of the resulting dispatch (demand coverage, solar capacity,
storage energy balance). Result extraction and saving are checked on a small
three-step instance.
"""

import os
import unittest
from tempfile import TemporaryDirectory

import pandas as pd

from pulpo import pulpo_time
from pulpo.datasets.elec_time_database import (
    setup_elec_time_db,
    PROJECT_NAME,
    DB_NAME,
    CHARGE_PRODUCT_CHOICE,
    ELECTRICITY_CHOICE,
)

setup_elec_time_db()

GWP = str(("GWP", "100a"))

ACTIVITY_NAMES = (
    "solar", "coal", "battery_charge", "battery_hold",
    "battery_discharge", "battery_holdtm1",
)


def build_worker():
    worker = pulpo_time.PulpoOptimizerTime(PROJECT_NAME, DB_NAME, {GWP: 1}, "")
    worker.intervention_matrix = "biosphere3"
    worker.get_lci_data()
    return worker


def solve_scenario(K, time_steps, solar_cap, demand_kwh):
    """Solve one dispatch scenario (the notebook's ``build_inputs`` +
    ``solve_scenario`` without the plotting).

    Returns the solved worker and a per-timestep DataFrame with the scaling
    of the five dispatchable activities plus the per-step CO2 impact.
    """
    worker = build_worker()
    acts = {name: worker.retrieve_activities(activities=[name])[0]
            for name in ACTIVITY_NAMES}
    solar, coal = acts["solar"], acts["coal"]
    charge, hold = acts["battery_charge"], acts["battery_hold"]
    discharge, holdtm1 = acts["battery_discharge"], acts["battery_holdtm1"]

    choices = {
        ELECTRICITY_CHOICE: {solar: 1e6, coal: 1e6, discharge: 1e6},
        CHARGE_PRODUCT_CHOICE: {charge: 1e6, hold: 1e6},
    }
    upper_limit = {
        t: {
            solar: solar_cap[t],
            coal: 1e6,
            charge: 1e6,
            hold: 1e6,
            # battery starts empty -> nothing to discharge at t=0
            discharge: 0.0 if t == time_steps[0] else 1e6,
            holdtm1: 0.0,  # phantom, never actually produced
        }
        for t in time_steps
    }
    # battery flows must be non-negative (default lower bound is -inf)
    lower_limit = {t: {charge: 0.0, hold: 0.0, discharge: 0.0} for t in time_steps}
    demand = {t: {ELECTRICITY_CHOICE: demand_kwh[t]} for t in time_steps}
    storage = [(holdtm1, CHARGE_PRODUCT_CHOICE, K)] if K is not None else None

    worker.instantiate(
        choices=choices,
        demand=demand,
        upper_limit=upper_limit,
        lower_limit=lower_limit,
        time_steps=time_steps,
        storage=storage,
    )
    worker.solve()

    pmap = worker.lci_data["process_map"]
    dispatch = pd.DataFrame(
        {name: [worker.instance.scaling_vector[t, pmap[acts[name].key]].value
                for t in time_steps]
         for name in ("solar", "coal", "battery_charge", "battery_hold",
                      "battery_discharge")},
        index=time_steps,
    )
    dispatch["co2"] = [worker.instance.impacts[t, GWP].value for t in time_steps]
    return worker, dispatch


class DispatchConsistencyMixin:
    """Physical-consistency checks shared by both dispatch scenarios."""

    TOL = 1e-6
    time_steps: list
    solar_cap: dict
    demand_kwh: dict

    def assert_dispatch_consistent(self, dispatch, K):
        first_step = self.time_steps[0]
        for t in self.time_steps:
            row = dispatch.loc[t]
            # electricity balance: solar + coal + discharge feed demand plus
            # the battery-charging draw
            net_supply = (row["solar"] + row["coal"] + row["battery_discharge"]
                          - row["battery_charge"])
            self.assertGreaterEqual(net_supply, self.demand_kwh[t] - self.TOL,
                                    f"demand not covered at t={t}")
            self.assertLessEqual(row["solar"], self.solar_cap[t] + self.TOL,
                                 f"solar capacity exceeded at t={t}")
            for col in ("battery_charge", "battery_hold", "battery_discharge"):
                self.assertGreaterEqual(row[col], -self.TOL,
                                        f"negative {col} at t={t}")
        # battery starts empty
        self.assertAlmostEqual(dispatch.loc[first_step, "battery_discharge"],
                               0.0, places=6)
        if K is None:
            return
        # storage balance: DISCHARGE draws from fresh charge plus held carry,
        # and HOLD is fed by the K-decayed net charge-product of the previous
        # step (see pulpo.utils.time_extension)
        for i, t in enumerate(self.time_steps):
            row = dispatch.loc[t]
            self.assertLessEqual(
                row["battery_discharge"],
                row["battery_charge"] + row["battery_hold"] + self.TOL,
                f"discharge exceeds available charge product at t={t}",
            )
            if i > 0:
                prev = dispatch.loc[self.time_steps[i - 1]]
                net_prev = (prev["battery_charge"] + prev["battery_hold"]
                            - prev["battery_discharge"])
                self.assertLessEqual(
                    row["battery_hold"], K * net_prev + self.TOL,
                    f"hold exceeds carried-over storage state at t={t}",
                )


###############################################
#### Example 1: one day, hourly resolution ####
###############################################

class TestIntradayDispatch(DispatchConsistencyMixin, unittest.TestCase):
    """24 hourly steps: bell-shaped solar, double-peak demand.

    Storage shifts the midday solar surplus into the evening peak.
    """

    K_LOSSY = 0.95
    K_IDEAL = 0.999
    time_steps = list(range(24))
    solar_cap = dict(enumerate([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 31.1, 60.0, 84.9, 103.9, 115.9,
        120.0, 115.9, 103.9, 84.9, 60.0, 31.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]))
    demand_kwh = dict(enumerate([
        25, 20, 20, 20, 25, 35, 55, 70, 70, 55, 45, 40,
        40, 40, 45, 55, 65, 75, 80, 75, 60, 45, 35, 30,
    ]))

    @classmethod
    def setUpClass(cls):
        _, cls.no_battery = solve_scenario(
            None, cls.time_steps, cls.solar_cap, cls.demand_kwh)
        _, cls.lossy = solve_scenario(
            cls.K_LOSSY, cls.time_steps, cls.solar_cap, cls.demand_kwh)
        _, cls.ideal = solve_scenario(
            cls.K_IDEAL, cls.time_steps, cls.solar_cap, cls.demand_kwh)

    def test_total_co2_reference_values(self):
        self.assertAlmostEqual(self.no_battery["co2"].sum(), 622.8, places=4)
        self.assertAlmostEqual(self.lossy["co2"].sum(), 332.815126, places=4)
        self.assertAlmostEqual(self.ideal["co2"].sum(), 248.9, places=4)

    def test_battery_reduces_emissions(self):
        self.assertLess(self.lossy["co2"].sum(), self.no_battery["co2"].sum())
        self.assertLess(self.ideal["co2"].sum(), self.lossy["co2"].sum())

    def test_dispatch_consistency(self):
        for dispatch, K in ((self.no_battery, None),
                            (self.lossy, self.K_LOSSY),
                            (self.ideal, self.K_IDEAL)):
            self.assert_dispatch_consistent(dispatch, K)

    def test_battery_serves_evening_peak(self):
        # After sunset (18h+) the only alternative to coal is discharging the
        # battery; with storage available a substantial share of the evening
        # demand must be served from it.
        evening = [t for t in self.time_steps if t >= 18]
        self.assertAlmostEqual(
            self.no_battery.loc[evening, "battery_discharge"].sum(), 0.0,
            places=6)
        self.assertGreater(self.lossy.loc[evening, "battery_discharge"].sum(), 1.0)
        self.assertGreater(self.ideal.loc[evening, "battery_discharge"].sum(), 1.0)


################################################
#### Example 2: two weeks, daily resolution ####
################################################

class TestMultidayDispatch(DispatchConsistencyMixin, unittest.TestCase):
    """14 daily steps: sunny week followed by an overcast week.

    Storage must propagate week-1 surplus across several days (via the HOLD
    activity) to cover week-2 demand.
    """

    K_LOSSY = 0.85
    K_IDEAL = 0.999
    time_steps = list(range(14))
    solar_cap = dict(enumerate([120, 110, 95, 40, 100, 130, 100,
                                10, 5, 50, 5, 10, 30, 5]))
    demand_kwh = dict(enumerate([50, 55, 60, 50, 45, 30, 30,
                                 55, 60, 65, 60, 50, 35, 35]))

    @classmethod
    def setUpClass(cls):
        _, cls.no_battery = solve_scenario(
            None, cls.time_steps, cls.solar_cap, cls.demand_kwh)
        _, cls.lossy = solve_scenario(
            cls.K_LOSSY, cls.time_steps, cls.solar_cap, cls.demand_kwh)
        _, cls.ideal = solve_scenario(
            cls.K_IDEAL, cls.time_steps, cls.solar_cap, cls.demand_kwh)

    def test_total_co2_reference_values(self):
        self.assertAlmostEqual(self.no_battery["co2"].sum(), 255.0, places=4)
        self.assertAlmostEqual(self.lossy["co2"].sum(), 75.617395, places=4)
        self.assertAlmostEqual(self.ideal["co2"].sum(), 0.0, places=4)

    def test_battery_reduces_emissions(self):
        self.assertLess(self.lossy["co2"].sum(), self.no_battery["co2"].sum())
        self.assertLess(self.ideal["co2"].sum(), self.lossy["co2"].sum())

    def test_ideal_battery_eliminates_coal(self):
        # Week-1 surplus suffices to cover all of week 2 when carrying energy
        # over is (nearly) free.
        self.assertAlmostEqual(self.ideal["coal"].sum(), 0.0, places=4)

    def test_battery_bridges_into_overcast_week(self):
        # Multi-day propagation: with storage, week 2 (days 7-13) must be
        # partly served from carried-over week-1 solar.
        week2 = [t for t in self.time_steps if t >= 7]
        self.assertAlmostEqual(
            self.no_battery.loc[week2, "battery_discharge"].sum(), 0.0,
            places=6)
        self.assertGreater(self.lossy.loc[week2, "battery_discharge"].sum(), 1.0)
        self.assertGreater(self.ideal.loc[week2, "battery_discharge"].sum(), 1.0)

    def test_dispatch_consistency(self):
        for dispatch, K in ((self.no_battery, None),
                            (self.lossy, self.K_LOSSY),
                            (self.ideal, self.K_IDEAL)):
            self.assert_dispatch_consistent(dispatch, K)


#############################################
#### Result extraction, saving, fallback ####
#############################################

class TestTimeResultExtraction(unittest.TestCase):
    """Extraction and saving of time-indexed results (three-step instance)."""

    @classmethod
    def setUpClass(cls):
        cls.time_steps = [0, 1, 2]
        solar_cap = {t: 1.0 for t in cls.time_steps}
        demand_kwh = {t: 1.0 for t in cls.time_steps}
        cls.worker, _ = solve_scenario(0.9, cls.time_steps, solar_cap, demand_kwh)

    def test_extract_results_time_columns(self):
        result = self.worker.extract_results()

        scaling = result["Scaling Vector"]
        self.assertIn("Time", scaling.index.names)
        self.assertEqual(set(scaling.index.get_level_values("Time")), set(self.time_steps))

        impacts = result["Impacts"]
        self.assertIn("Time", impacts.index.names)
        self.assertEqual(set(impacts.index.get_level_values("Time")), set(self.time_steps))

        choices_dict = result["Choices"]
        self.assertIn(ELECTRICITY_CHOICE, choices_dict)
        electricity_df = choices_dict[ELECTRICITY_CHOICE]
        self.assertIn("Time", electricity_df.index.names)
        self.assertEqual(set(electricity_df.index.get_level_values("Time")), set(self.time_steps))

    def test_extract_results_intervention_and_demand(self):
        result = self.worker.extract_results()

        intervention = result["Intervention Vector"]
        self.assertIn("Time", intervention.index.names)

        demand = result["Demand"]
        self.assertIn("Time", demand.index.names)
        self.assertEqual(set(demand.index.get_level_values("Time")), set(self.time_steps))

    def test_save_results(self):
        with TemporaryDirectory() as temp_dir:
            file_name = os.path.join(temp_dir, "test_time_results.xlsx")
            self.worker.save_results(file_name)
            self.assertTrue(os.path.exists(file_name))

            df = pd.read_excel(file_name, sheet_name="Scaling Vector")
            self.assertIn("Time", df.columns)


#################################################
#### Goal programming on aggregated impacts  ####
#################################################

class TestTimeGoalObjective(unittest.TestCase):
    """Goal objective on the time-aggregated impacts (yearly-budget style).

    Uses the intraday no-battery scenario, whose minimum total CO2 is 622.8
    (asserted in TestIntradayDispatch). With a single goal category the
    minimum average transgression is max(0, 622.8 / budget - 1).
    """

    MIN_TOTAL_CO2 = 622.8
    time_steps = TestIntradayDispatch.time_steps
    solar_cap = TestIntradayDispatch.solar_cap
    demand_kwh = TestIntradayDispatch.demand_kwh

    def solve_goal_scenario(self, budget):
        worker = build_worker()
        acts = {name: worker.retrieve_activities(activities=[name])[0]
                for name in ACTIVITY_NAMES}
        choices = {ELECTRICITY_CHOICE: {acts["solar"]: 1e6, acts["coal"]: 1e6}}
        upper_limit = {t: {acts["solar"]: self.solar_cap[t], acts["coal"]: 1e6}
                       for t in self.time_steps}
        demand = {t: {ELECTRICITY_CHOICE: self.demand_kwh[t]} for t in self.time_steps}
        worker.instantiate(
            choices=choices, demand=demand, upper_limit=upper_limit,
            time_steps=self.time_steps,
            imp_goals={GWP: budget}, objective='goal',
        )
        worker.solve()
        return worker

    def test_transgressed_budget(self):
        budget = 300.0
        worker = self.solve_goal_scenario(budget)
        expected = self.MIN_TOTAL_CO2 / budget - 1
        self.assertAlmostEqual(worker.instance.OBJ(), expected, places=4)
        self.assertAlmostEqual(worker.instance.transgression[GWP].value, expected, places=4)
        total_co2 = sum(worker.instance.impacts[t, GWP].value for t in self.time_steps)
        self.assertAlmostEqual(total_co2, self.MIN_TOTAL_CO2, places=4)

    def test_satisfied_budget(self):
        worker = self.solve_goal_scenario(700.0)
        self.assertAlmostEqual(worker.instance.OBJ(), 0.0, places=6)
        self.assertAlmostEqual(worker.instance.transgression[GWP].value, 0.0, places=6)

    def test_extract_transgressions_time_indexed(self):
        budget = 300.0
        worker = self.solve_goal_scenario(budget)
        transgressions = worker.extract_results()["Transgressions"]
        self.assertEqual(list(transgressions.index), [GWP])
        row = transgressions.loc[GWP]
        self.assertAlmostEqual(row["Impact"], self.MIN_TOTAL_CO2, places=4)
        self.assertAlmostEqual(row["Goal"], budget, places=6)
        self.assertAlmostEqual(row["TL"], self.MIN_TOTAL_CO2 / budget, places=4)
        self.assertAlmostEqual(row["Transgression"], self.MIN_TOTAL_CO2 / budget - 1, places=4)

    def test_goal_validation_in_time_path(self):
        worker = build_worker()
        solar = worker.retrieve_activities(activities=["solar"])[0]
        coal = worker.retrieve_activities(activities=["coal"])[0]
        time_steps = [0, 1]
        choices = {ELECTRICITY_CHOICE: {solar: 1e6, coal: 1e6}}
        demand = {t: {ELECTRICITY_CHOICE: 1.0} for t in time_steps}
        with self.assertRaises(ValueError):
            worker.instantiate(choices=choices, demand=demand,
                               time_steps=time_steps, objective='goal')
        with self.assertRaises(ValueError):
            worker.instantiate(choices=choices, demand=demand,
                               time_steps=time_steps,
                               imp_goals={GWP: -5}, objective='goal')


class TestStaticFallbackAndErrors(unittest.TestCase):
    def test_instantiate_without_time_steps_falls_back_to_static(self):
        worker = build_worker()
        acts = {name: worker.retrieve_activities(activities=[name])[0]
                for name in ACTIVITY_NAMES}
        # cap solar (via its choice capacity, which is what bounds choice
        # members in the static formulation) below the demand so that
        # 0.6 kWh of coal (1 kg CO2/kWh) is unavoidable
        choices = {ELECTRICITY_CHOICE: {acts["solar"]: 0.4, acts["coal"]: 1e6}}
        demand = {ELECTRICITY_CHOICE: 1.0}
        # battery flows must be non-negative (the default lower bound is -inf,
        # under which running the battery backwards creates free electricity)
        lower_limit = {acts["battery_charge"]: 0.0, acts["battery_hold"]: 0.0,
                       acts["battery_discharge"]: 0.0}
        worker.instantiate(choices=choices, demand=demand,
                           lower_limit=lower_limit)
        worker.solve()
        self.assertIsNone(worker.time_steps)
        self.assertAlmostEqual(worker.instance.OBJ(), 0.6, places=6)

    def test_dependent_constraints_rejected_in_time_path(self):
        worker = build_worker()
        solar = worker.retrieve_activities(activities=["solar"])[0]
        coal = worker.retrieve_activities(activities=["coal"])[0]
        time_steps = [0, 1]
        choices = {ELECTRICITY_CHOICE: {solar: 1e6, coal: 1e6}}
        demand = {t: {ELECTRICITY_CHOICE: 1.0} for t in time_steps}
        dependent_constraints = {"solar_cap": {"left": {solar: 1}, "right": {coal: 4}}}
        with self.assertRaises(NotImplementedError):
            worker.instantiate(choices=choices, demand=demand,
                               time_steps=time_steps,
                               dependent_constraints=dependent_constraints)
