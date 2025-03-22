import pulpo as pulpo
import unittest
from tests.sample_database import sample_lcia, setup_test_db
from pulpo.utils.bw_parser import import_data, retrieve_methods, retrieve_env_interventions, retrieve_processes
from pulpo import pulpo

setup_test_db()

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

        result = import_data('sample_project', 'technosphere', methods, 'biosphere')

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

class TestPULPO(unittest.TestCase):

    def test_pulpo(self):
        project = 'sample_project'
        database = 'technosphere'
        methods = {"('my project', 'climate change')": 1,
                   "('my project', 'air quality')": 1,
                   "('my project', 'resources')": 0}
        
        # Test invalid project
        with self.assertRaises(ValueError) as context:
            pulpo.PulpoOptimizer('nonexistent_project', database, methods, '')
        self.assertIn("Project 'nonexistent_project' does not exist", str(context.exception))

        # Test basic PULPO:
        worker = pulpo.PulpoOptimizer(project, database, methods, '')
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
        # Test supply specification:
        upper_limit = {eCar[0]: 1}
        lower_limit = {eCar[0]: 1}
        worker.instantiate(choices=choices, upper_limit=upper_limit, lower_limit=lower_limit)
        worker.solve()
        result_obj = round(worker.instance.OBJ(), 6)
        result_aux = round(worker.instance.impacts_calculated["('my project', 'resources')"].value, 5)
        self.assertEqual(result_obj, 0.1)
        self.assertEqual(result_aux, 5.1)
        # Test elementary / intervention flow constraint:
        water = worker.retrieve_envflows(activities="Water, irrigation")
        upper_elem_limit = {water[0]: 5.2}
        worker.instantiate(choices=choices, demand=demand, upper_elem_limit=upper_elem_limit)
        worker.solve()
        result_obj = round(worker.instance.OBJ(), 6)
        result_aux = round(worker.instance.impacts_calculated["('my project', 'resources')"].value, 5)
        self.assertEqual(result_obj, 0.14237)
        self.assertEqual(result_aux, 5.2)




