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

from pulpo.utils.utils import is_bw25
project_name = "sample_project_bw25" if is_bw25() else "sample_project"

###########################
#### Test the PARSER  #####
###########################

class TestParser(unittest.TestCase):
    def test_database_import(self):
        result = sample_lcia()
        self.assertEqual(result, [4.22308, 1.5937, 11.1567])  # Example assertion

    def test_import_data(self):
        # Test if the import works and has the expected structure
        methods = {
            "('my project', 'climate change')": 1,
            "('my project', 'air quality')": 1,
            "('my project', 'resources')": 1,
        }

        result = import_data(project_name, 'technosphere', methods, 'biosphere')

        # Define the expected keys
        expected_keys = [
            'matrices',
            'intervention_matrix',
            'technology_matrix',
            'process_map',
            'intervention_map',
            'intervention_params',
            'characterization_params',
            'intervention_map_metadata',
            'process_map_metadata'
        ]

        # Assert that the keys in the result match the expected keys
        self.assertEqual(sorted(result.keys()), sorted(expected_keys))

        # Assert the shapes of specific matrices
        self.assertEqual(result['technology_matrix'].shape, (5, 5))
        self.assertEqual(result['intervention_matrix'].shape, (4, 5))

        # Assert the process map
        expected_process_map = {
            ('technosphere', 'oil extraction'): 0,
            ('technosphere', 'lignite extraction'): 1,
            ('technosphere', 'steam cycle'): 2,
            ('technosphere', 'wind turbine'): 3,
            ('technosphere', 'e-Car'): 4,
        }

        self.assertEqual(result['process_map'], expected_process_map)

        # Test invalid database
        with self.assertRaises(ValueError) as context:
            import_data(project_name, 'nothing', methods, 'biosphere')
        self.assertIn("Database 'nothing' does not exist", str(context.exception))

        # Test invalid method
        invalid_methods = {
            "('my project', 'nonexistent method')": 1,
            "('my project', 'air quality')": 1,
        }
        with self.assertRaises(ValueError) as context:
            import_data(project_name, 'technosphere', invalid_methods, 'biosphere')
        self.assertIn("The following methods do not exist", str(context.exception))

    def test_uncertainty_import(self):
        # Test if the uncertainty import works
        methods = {
            "('my project', 'climate change')": 1,
            "('my project', 'air quality')": 1,
            "('my project', 'resources')": 1,
        }

        result = import_data(project_name, 'technosphere', methods, 'biosphere', seed=42)

        # Check one element in each matrix
        self.assertAlmostEqual(result['technology_matrix'][0, 0], 1.0, places=6)
        if is_bw25():
           self.assertAlmostEqual(result['intervention_matrix'][0, 2], 0.8275082141783688, places=6)
           self.assertAlmostEqual(result['matrices']["('my project', 'climate change')"][0, 0], 1.0647688547752003, places=6)
        else:
            self.assertAlmostEqual(result['intervention_matrix'][0, 2], 1.0647688547752003, places=6)
            self.assertAlmostEqual(result['matrices']["('my project', 'climate change')"][0, 0], 1.049671416041285, places=6)

    def test_retrieve_activities(self):
        key = retrieve_processes(project_name, 'technosphere', keys=["('technosphere', 'wind turbine')"])
        name = retrieve_processes(project_name, 'technosphere', activities=['e-Car'])
        location = retrieve_processes(project_name, 'technosphere', locations=['GLO'])
        self.assertEqual(key[0]['name'], "wind turbine")
        self.assertEqual(name[0]['name'], "e-Car")
        self.assertEqual(len(location), 5)

    def test_retrieve_methods(self):
        single_result = retrieve_methods(project_name, ['climate'])
        multi_result = retrieve_methods(project_name, ['project'])
        self.assertEqual(single_result, [('my project', 'climate change')])
        self.assertEqual(multi_result, [('my project', 'climate change'), ('my project', 'air quality'), ('my project', 'resources')])

    def test_retrieve_envflows(self):
        result = retrieve_env_interventions(project_name, intervention_matrix='biosphere', keys="('biosphere', 'PM')")
        self.assertEqual(result[0]['name'], 'Particulate matter, industrial')

###############################
#### Test the BASE PULPO  #####
###############################

class TestPULPO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = project_name
        cls.database = 'technosphere'
        cls.methods = {
            "('my project', 'climate change')": 1,
            "('my project', 'air quality')": 1,
            "('my project', 'resources')": 0
        }

    def test_invalid_project(self):
        with self.assertRaises(ValueError) as context:
            pulpo.PulpoOptimizer('nonexistent_project', self.database, self.methods, '')
        self.assertIn("Project 'nonexistent_project' does not exist", str(context.exception))

    def test_basic_pulpo(self):
        worker = pulpo.PulpoOptimizer(self.project, self.database, self.methods, '')
        worker.intervention_matrix = 'biosphere'
        worker.get_lci_data()
        eCar = worker.retrieve_activities(reference_products='transport')
        demand = {eCar[0]: 1}
        elec = worker.retrieve_activities(reference_products='electricity')
        choices = {'electricity': {elec[0]: 100, elec[1]: 100}}
        worker.instantiate(choices=choices, demand=demand)
        worker.solve()
        result_obj = round(worker.instance.OBJ(), 6)
        result_aux = round(worker.instance.impacts_calculated["('my project', 'resources')"].value, 5)
        self.assertEqual(result_obj, 0.103093)
        self.assertEqual(result_aux, 5.25773)

    def test_supply_specification(self):
        worker = pulpo.PulpoOptimizer(self.project, self.database, self.methods, '')
        worker.intervention_matrix = 'biosphere'
        worker.get_lci_data()
        eCar = worker.retrieve_activities(reference_products='transport')
        elec = worker.retrieve_activities(reference_products='electricity')
        choices = {'electricity': {elec[0]: 100, elec[1]: 100}}
        upper_limit = {eCar[0]: 1}
        lower_limit = {eCar[0]: 1}
        worker.instantiate(choices=choices, upper_limit=upper_limit, lower_limit=lower_limit)
        worker.solve()
        result_obj = round(worker.instance.OBJ(), 6)
        result_aux = round(worker.instance.impacts_calculated["('my project', 'resources')"].value, 5)
        self.assertEqual(result_obj, 0.1)
        self.assertEqual(result_aux, 5.1)

    def test_elementary_intervention_flow_constraint(self):
        worker = pulpo.PulpoOptimizer(self.project, self.database, self.methods, '')
        worker.intervention_matrix = 'biosphere'
        worker.get_lci_data()
        eCar = worker.retrieve_activities(reference_products='transport')
        demand = {eCar[0]: 1}
        elec = worker.retrieve_activities(reference_products='electricity')
        choices = {'electricity': {elec[0]: 100, elec[1]: 100}}
        water = worker.retrieve_envflows(activities="Water, irrigation")
        upper_elem_limit = {water[0]: 5.2}
        worker.instantiate(choices=choices, demand=demand, upper_elem_limit=upper_elem_limit)
        worker.solve()
        result_obj = round(worker.instance.OBJ(), 6)
        result_aux = round(worker.instance.impacts_calculated["('my project', 'resources')"].value, 5)
        self.assertEqual(result_obj, 0.14237)
        self.assertEqual(result_aux, 5.2)
        self.assertEqual(round(worker.instance.inv_flows[3].value, 3), 5.200)
        self.assertEqual(round(worker.instance.inv_flows[2].value, 3), 1.659)
    
    def test_gams_solver(self):
        """Test solving the optimization problem using the GAMS solver."""
        gams_path = os.getenv('GAMS_PULPO')  # Ensure GAMS_PULPO is set in the environment
        print(gams_path)
        if not gams_path:
            self.skipTest(
                "GAMS_PULPO environment variable is not set. Skipping GAMS test. "
                "To set it:\n"
                "- On Windows: Run 'setx GAMS_PULPO \"C:\\path\\to\\gams\"' in Command Prompt (requires reopening the terminal).\n"
                "- On macOS/Linux: Run 'export GAMS_PULPO=/path/to/gams' in the terminal."
            )

        worker = pulpo.PulpoOptimizer(self.project, self.database, self.methods, '')
        worker.intervention_matrix = 'biosphere'
        worker.get_lci_data()
        eCar = worker.retrieve_activities(reference_products='transport')
        demand = {eCar[0]: 1}
        elec = worker.retrieve_activities(reference_products='electricity')
        choices = {'electricity': {elec[0]: 100, elec[1]: 100}}
        worker.instantiate(choices=choices, demand=demand)

        # Subtests for different solvers
        for solver_name in ['cplex', 'baron', 'xpress']:
            with self.subTest(solver=solver_name):
                # Solve using GAMS with the specified solver
                if solver_name == 'cplex':
                    worker.solve(GAMS_PATH=gams_path)
                else:
                    worker.solve(GAMS_PATH=gams_path, solver_name=solver_name)

                # Assert the objective value
                result_obj = round(worker.instance.OBJ(), 6)
                self.assertEqual(result_obj, 0.103093)

    def test_neos_solver(self):
        """Test solving the optimization problem using the NEOS solver."""
        if 'NEOS_EMAIL' not in os.environ:
            self.skipTest(
                "NEOS_EMAIL environment variable is not set. Skipping NEOS test. "
                "To set it follow instructions on: https://www.twilio.com/en-us/blog/how-to-set-environment-variables-html"
            )
        worker = pulpo.PulpoOptimizer(self.project, self.database, self.methods, '')
        worker.intervention_matrix = 'biosphere'
        worker.get_lci_data()
        eCar = worker.retrieve_activities(reference_products='transport')
        demand = {eCar[0]: 1}
        elec = worker.retrieve_activities(reference_products='electricity')
        choices = {'electricity': {elec[0]: 100, elec[1]: 100}}
        worker.instantiate(choices=choices, demand=demand)

        # Solve using GAMS
        worker.solve(solver_name='cplex')

        # Assert the objective value
        result_obj = round(worker.instance.OBJ(), 6)
        self.assertEqual(result_obj, 0.103093)
    
    def test_monte_carlo(self):
        """Test the Monte Carlo simulation."""
        worker = pulpo.PulpoOptimizer(self.project, self.database, self.methods, '')
        worker.intervention_matrix = 'biosphere'
        worker.get_lci_data()
        eCar = worker.retrieve_activities(reference_products='transport')
        demand = {eCar[0]: 1}
        elec = worker.retrieve_activities(reference_products='electricity')
        choices = {'electricity': {elec[0]: 100, elec[1]: 100}}
        worker.instantiate(choices=choices, demand=demand)

        # Run Monte Carlo simulation
        results = worker.solve_MC(n_it=10)
        if is_bw25():
            self.assertEqual(sum(results)/10, 0.09328144132911008)
        else:
            self.assertEqual(sum(results)/10, 0.10209749581609506)
    
    def test_gsa(self):
        worker = pulpo.PulpoOptimizer(self.project, self.database, self.methods, '')
        worker.intervention_matrix = 'biosphere'
        worker.get_lci_data()
        eCar = worker.retrieve_activities(reference_products='transport')
        demand = {eCar[0]: 1}
        elec = worker.retrieve_activities(reference_products='electricity')
        choices = {'electricity': {elec[0]: 100, elec[1]: 100}}
        worker.instantiate(choices=choices, demand=demand)
        worker.solve()
        worker.run_gsa()
        

##########################
#### Test the SAVER  #####
##########################

class TestSaver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use the same worker object as in test_pulpo
        project = project_name
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
        # --- Test case 1: Brightway activity as demand key ---
        result = extract_demand(self.worker.demand)
        expected_index = pd.MultiIndex.from_tuples(
            [("transport", "e-Car", "GLO")],
            names=["Reference Product", "Activity Name", "Location"]
        )
        expected = pd.DataFrame({"Value": [1.0]}, index=expected_index)
        expected["Value"] = expected["Value"].astype(result["Value"].dtype)
        assert_frame_equal(result, expected, check_exact=False, rtol=1e-5)

        # --- Test case 2: Simple string as demand key ---
        self.worker.demand = {"electricity": 1}
        result = extract_demand(self.worker.demand)
        expected_index = pd.MultiIndex.from_tuples(
            [("electricity", "electricity", "Unknown")],
            names=["Reference Product", "Activity Name", "Location"]
        )
        expected = pd.DataFrame({"Value": [1.0]}, index=expected_index)
        expected["Value"] = expected["Value"].astype(result["Value"].dtype)
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
                # Check if the sheet is empty
                if df.empty:
                    self.fail(f"The sheet '{sheet_name}' is empty.")

