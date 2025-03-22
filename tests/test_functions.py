import os
from tempfile import TemporaryDirectory
import pandas as pd
from pandas.testing import assert_frame_equal

from pulpo import pulpo
from pulpo.utils.bw_parser import import_data, retrieve_methods, retrieve_env_interventions, retrieve_processes
from pulpo.utils.saver import extract_flows, extract_slack, extract_impacts, extract_choices, extract_demand, extract_constraints, save_results

import tests as tests
from tests.sample_database import sample_lcia, setup_test_db
import unittest

setup_test_db()

###############################
#### Test the PARSER  #####
###############################

class TestParser(unittest.TestCase):
    def test_database_import(self):
        result = sample_lcia()
        self.assertEqual(result, [4.22308, 1.5937, 11.1567])  # Run a sample LCA and check the results

    def test_import_data(self):
        # Test if the import works and has the expected structure
        methods = {
            "('my project', 'climate change')": 1,
            "('my project', 'air quality')": 1,
            "('my project', 'resources')": 1,
        }

        result = import_data('sample_project', 'technosphere', methods, 'biosphere')
        self.assertEqual([idx for idx in result], ['matrices', 'intervention_matrix', 'technology_matrix', 'process_map', 'intervention_params', 'characterization_params', 'intervention_map', 'intervention_map_metadata', 'process_map_metadata'])
        self.assertEqual(result['technology_matrix'].shape, (5, 5))
        self.assertEqual(result['intervention_matrix'].shape, (4, 5))
        #self.assertEqual(result['process_map'], {('technosphere', 'oil extraction'): 0, ('technosphere', 'lignite extraction'): 1, ('technosphere', 'steam cycle'): 2, ('technosphere', 'wind turbine'): 3, ('technosphere', 'e-Car'): 4, 2: 'steam cycle | electricity | GLO', 1: 'lignite extraction | lignite | GLO', 0: 'oil extraction | oil | GLO', 3: 'wind turbine | electricity | GLO', 4: 'e-Car | transport | GLO'})

        # Test invalid database
        with self.assertRaises(ValueError) as context:
            import_data('sample_project', 'nothing', methods, 'biosphere')
        self.assertIn("Database 'nothing' does not exist", str(context.exception))

        # Test invalid method
        invalid_methods = {
            "('my project', 'nonexistent method')": 1,
            "('my project', 'air quality')": 1,
        }
        with self.assertRaises(ValueError) as context:
            import_data('sample_project', 'technosphere', invalid_methods, 'biosphere')
        self.assertIn("The following methods do not exist", str(context.exception))

    def test_retrieve_activities(self):
        key = retrieve_processes('sample_project', 'technosphere', keys=["('technosphere', 'wind turbine')"])
        name = retrieve_processes('sample_project', 'technosphere', activities=['e-Car'])
        location = retrieve_processes('sample_project', 'technosphere', locations=['GLO'])
        self.assertEqual(key[0]['name'], "wind turbine")
        self.assertEqual(name[0]['name'], "e-Car")
        self.assertEqual(len(location), 5)

    def test_retrieve_methods(self):
        single_result = retrieve_methods('sample_project', ['climate'])
        multi_result = retrieve_methods('sample_project', ['project'])
        self.assertEqual(single_result, [('my project', 'climate change')])
        self.assertEqual(multi_result, [('my project', 'climate change'), ('my project', 'air quality'), ('my project', 'resources')])

    def test_retrieve_envflows(self):
        result = retrieve_env_interventions('sample_project', intervention_matrix='biosphere', keys="('biosphere', 'PM')")
        self.assertEqual(result[0]['name'], 'Particulate matter, industrial')

###############################
#### Test the BASE PULPO  #####
###############################

class TestPULPO(unittest.TestCase):
    def setUp(self):
        """
        Common setup for all tests in this class.
        Creates a worker object and retrieves necessary activities.
        """
        self.project = 'sample_project'
        self.database = 'technosphere'
        self.methods = {
            "('my project', 'climate change')": 1,
            "('my project', 'air quality')": 1,
            "('my project', 'resources')": 0,
        }
        self.worker = pulpo.PulpoOptimizer(self.project, self.database, self.methods, '')
        self.worker.intervention_matrix = 'biosphere'
        self.worker.get_lci_data()

        # Retrieve activities
        self.eCar = self.worker.retrieve_activities(reference_products='transport')
        self.elec = self.worker.retrieve_activities(reference_products='electricity')
        self.water = self.worker.retrieve_envflows(activities="Water, irrigation")

    def test_invalid_project(self):
        with self.assertRaises(ValueError) as context:
            pulpo.PulpoOptimizer('nonexistent_project', self.database, self.methods, '')
        self.assertIn("Project 'nonexistent_project' does not exist", str(context.exception))

    def test_basic_pulpo(self):
        """
        Test basic PULPO functionality with choices and demand.
        """
        # Define demand and choices
        demand = {self.eCar[0]: 1}
        choices = {'electricity': {self.elec[0]: 100, self.elec[1]: 100}}

        # Instantiate and solve
        self.worker.instantiate(choices=choices, demand=demand)
        self.worker.solve()

        # Assert results
        result_obj = round(self.worker.instance.OBJ(), 6)
        result_aux = round(self.worker.instance.impacts_calculated["('my project', 'resources')"].value, 5)
        self.assertEqual(result_obj, 0.103093)
        self.assertEqual(result_aux, 5.25773)

    def test_supply_specification(self):
        # Define demand, choices, and limits
        demand = {self.eCar[0]: 1}
        choices = {'electricity': {self.elec[0]: 100, self.elec[1]: 100}}
        upper_limit = {self.eCar[0]: 1}
        lower_limit = {self.eCar[0]: 1}

        # Instantiate and solve
        self.worker.instantiate(choices=choices, demand=demand, upper_limit=upper_limit, lower_limit=lower_limit)
        self.worker.solve()

        # Assert results
        result_obj = round(self.worker.instance.OBJ(), 6)
        result_res = round(self.worker.instance.impacts_calculated["('my project', 'resources')"].value, 5)
        self.assertEqual(result_obj, 0.1)
        self.assertEqual(result_res, 5.1)

    def test_elementary_flow_constraint(self):
        # Define demand, choices, and element limits
        demand = {self.eCar[0]: 1}
        choices = {'electricity': {self.elec[0]: 100, self.elec[1]: 100}}
        upper_elem_limit = {self.water[0]: 5.2}

        # Instantiate and solve
        self.worker.instantiate(choices=choices, demand=demand, upper_elem_limit=upper_elem_limit)
        self.worker.solve()

        # Assert results
        result_obj = round(self.worker.instance.OBJ(), 6)
        result_res = round(self.worker.instance.impacts_calculated["('my project', 'resources')"].value, 5)
        self.assertEqual(result_obj, 0.14237)
        self.assertEqual(result_res, 5.2)

##########################
#### Test the SAVER  #####
##########################

class TestSaver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use the same worker object as in test_pulpo
        project = 'sample_project'
        database = 'technosphere'
        methods = {"('my project', 'climate change')": 1,
                   "('my project', 'air quality')": 1,
                   "('my project', 'resources')": 0}
        cls.worker = pulpo.PulpoOptimizer(project, database, methods, '')
        cls.worker.intervention_matrix = 'biosphere'
        cls.worker.get_lci_data()
        eCar = cls.worker.retrieve_activities(reference_products='transport')
        demand = {eCar[0]: 1}
        elec = cls.worker.retrieve_activities(reference_products='electricity')
        upper_limit = {eCar[0]: 1}
        lower_limit = {eCar[0]: 1}
        choices = {'electricity': {elec[0]: 100, elec[1]: 100}}
        water = cls.worker.retrieve_envflows(activities="Water, irrigation")
        upper_elem_limit = {water[0]: 5.2}
        cls.worker.instantiate(choices=choices, demand=demand, upper_limit=upper_limit, lower_limit=lower_limit, upper_elem_limit=upper_elem_limit)
        cls.worker.solve()

    def test_extract_flows_scaling(self):
        process_map = self.worker.lci_data['process_map']
        process_map_metadata = self.worker.lci_data['process_map_metadata']
        result = extract_flows(self.worker.instance, process_map, process_map_metadata, flow_type='scaling')

        # Define the expected DataFrame
        expected = pd.DataFrame({
            'Key': [
                ('technosphere', 'wind turbine'),
                ('technosphere', 'e-Car'),
                ('technosphere', 'oil extraction'),
                ('technosphere', 'lignite extraction'),
                ('technosphere', 'steam cycle')
            ],
            'Metadata': ["wind turbine | electricity | GLO",
                         "e-Car | transport | GLO",
                         "oil extraction | oil | GLO",
                         "lignite extraction | lignite | GLO",
                        "steam cycle | electricity | GLO"],  # Replace with actual metadata if available
            'Value': [1.0, 1.0, -0.0, -0.0, -0.0]
        }, index=[3, 4, 0, 1, 2]).rename_axis("ID")

        # Ensure the index dtype matches the result
        expected.index = expected.index.astype(result.index.dtype)

        # Assert the result matches the expected DataFrame
        assert_frame_equal(result, expected, check_exact=False, rtol=1e-5)

    def test_extract_flows_intervention(self):
        intervention_map = self.worker.lci_data['intervention_map']
        intervention_map_metadata = self.worker.lci_data['intervention_map_metadata']
        result = extract_flows(self.worker.instance, intervention_map, intervention_map_metadata, flow_type='intervention')

        # Define the expected DataFrame
        expected = pd.DataFrame({
            'Key': [
                ('biosphere', 'H2O_irrigation'),
                ('biosphere', 'PM'),
                ('biosphere', 'CO2'),
                ('biosphere', 'CH4')
            ],
            'Metadata': ["Water, irrigation | ('water use', 'irrigation')",
                         "Particulate matter, industrial | ('air quality', 'particulate matter')",
                         "Carbon dioxide, fossil | ('climate change', 'GWP 100a')",
                         "Methane, agricultural | ('climate change', 'GWP 100a')"],  # Replace with actual metadata if available
            'Value': [5.1, 1.6, 0.1, 0.0]
        }, index=[3, 2, 0, 1]).rename_axis("ID")
        # Assert the result matches the expected DataFrame
        assert_frame_equal(result, expected, check_exact=False, rtol=1e-5)

    def test_extract_slack(self):
        result = extract_slack(self.worker.instance)

        # Define the expected DataFrame
        expected = pd.DataFrame({
            'Value': [-0.00, -0.00, -0.00, -0.03]
        }, index=[0, 1, 'electricity', 4])

        # Assert the result matches the expected DataFrame
        assert_frame_equal(result, expected, check_exact=False, rtol=1e-5)

    def test_extract_impacts(self):
        result = extract_impacts(self.worker.instance)

        # Define the expected DataFrame
        expected = pd.DataFrame({
            'Weight': [1, 1],
            'Value': [-0.0, 0.1]
        }, index=["('my project', 'air quality')", "('my project', 'climate change')"]).rename_axis("Method")

        # Assert the result matches the expected DataFrame
        assert_frame_equal(result, expected, check_exact=False, rtol=1e-5)

    def test_extract_choices(self):
        process_map = self.worker.lci_data['process_map']
        process_map_metadata = self.worker.lci_data['process_map_metadata']
        result = extract_choices(self.worker.instance, self.worker.choices, process_map, process_map_metadata)

        # Define the expected dictionary of DataFrames
        expected = {
            'electricity': pd.DataFrame({
                'Value': [1.0, -0.0],
                'Capacity': [100, 100]
            }, index=['wind turbine | electricity | GLO', 'steam cycle | electricity | GLO']).rename_axis("Metadata")
        }

        # Assert the result matches the expected dictionary
        for key, df in expected.items():
            assert_frame_equal(result[key], df, check_exact=False, rtol=1e-5)


    def test_extract_demand(self):
        # Call the function
        result = extract_demand(self.worker.demand)

        # Define the expected DataFrame
        expected = pd.DataFrame({"Value": [1.0]}, index=pd.MultiIndex.from_tuples([("transport", "e-Car", "GLO")], names=["Reference Product", "Activity Name", "Location"]))

        # Ensure the dtypes match
        expected["Value"] = expected["Value"].astype(result["Value"].dtype)

        # Assert the result matches the expected DataFrame
        assert_frame_equal(result, expected, check_exact=False, rtol=1e-5)


    def test_extract_constraints(self):
        # Define the inputs for the function
        constraints = self.worker.upper_limit  # Use the upper_limit constraints from the worker
        mapping = self.worker.lci_data['process_map']  # Process mapping
        metadata = self.worker.lci_data['process_map_metadata']  # Metadata for processes
        constraint_type = 'scaling'  # Specify the type of constraint

        # Call the function
        result = extract_constraints(self.worker.instance, constraints, mapping, metadata, constraint_type)
        expected = pd.DataFrame({"Key": [('technosphere', 'e-Car')], "Metadata": ["e-Car | transport | GLO"], "Value": [1.0], "Limit": [1]}, index=[4]).rename_axis("ID")

        # Assert the result matches the expected DataFrame
        assert_frame_equal(result, expected, check_exact=False, rtol=1e-5)
    
    def test_save_results(self):
        with TemporaryDirectory() as temp_dir:
            # Define the file path for the saved results
            file_name = os.path.join(temp_dir, 'test_results.xlsx')

            # Save the results
            save_results(self.worker, file_name)

            # Assert the file was created
            self.assertTrue(os.path.exists(file_name))

            # Define the expected sheets
            expected_sheets = ['Scaling Vector', 'Intervention Vector', 'Slack', 'Impacts', 'Choices']

            # Verify the contents of each sheet
            for sheet_name in expected_sheets:
                # Read the sheet
                df = pd.read_excel(file_name, sheet_name=sheet_name, index_col=0)
