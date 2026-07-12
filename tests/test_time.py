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


class TestTimeExtension(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.time_steps = [0, 1, 2]

        cls.worker = pulpo_time.PulpoOptimizerTime(PROJECT_NAME, DB_NAME, {GWP: 1}, "")
        cls.worker.intervention_matrix = "biosphere3"
        cls.worker.get_lci_data()

        solar = cls.worker.retrieve_activities(activities=["solar"])[0]
        coal = cls.worker.retrieve_activities(activities=["coal"])[0]
        charge = cls.worker.retrieve_activities(activities=["battery_charge"])[0]
        hold = cls.worker.retrieve_activities(activities=["battery_hold"])[0]
        discharge = cls.worker.retrieve_activities(activities=["battery_discharge"])[0]
        holdtm1 = cls.worker.retrieve_activities(activities=["battery_holdtm1"])[0]

        choices = {
            ELECTRICITY_CHOICE: {solar: 1e6, coal: 1e6, discharge: 1e6},
            CHARGE_PRODUCT_CHOICE: {charge: 1e6, hold: 1e6},
        }
        upper_limit = {
            t: {
                solar: 1.0,
                coal: 1e6,
                charge: 1e6,
                hold: 1e6,
                # battery starts empty -> nothing to discharge at t=0
                discharge: 0.0 if t == cls.time_steps[0] else 1e6,
                holdtm1: 0.0,  # phantom, never actually produced
            }
            for t in cls.time_steps
        }
        lower_limit = {t: {charge: 0.0, hold: 0.0, discharge: 0.0} for t in cls.time_steps}
        demand = {t: {ELECTRICITY_CHOICE: 1.0} for t in cls.time_steps}
        storage_spec = [(holdtm1, CHARGE_PRODUCT_CHOICE, 0.9)]

        cls.worker.instantiate(
            choices=choices,
            demand=demand,
            upper_limit=upper_limit,
            lower_limit=lower_limit,
            time_steps=cls.time_steps,
            storage=storage_spec,
        )
        cls.worker.solve()

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


if __name__ == "__main__":
    unittest.main()
