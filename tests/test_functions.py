import pulpo as pulpo
import unittest
from tests.sample_database import sample_lcia, setup_test_db
from pulpo.utils.bw_parser import import_data, retrieve_methods, retrieve_envflows, retrieve_activities
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

        self.assertEqual([idx for idx in result], ['matrices', 'biosphere', 'technosphere', 'activity_map', 'elem_map'])
        self.assertEqual(result['technosphere'].shape, (5, 5))
        self.assertEqual(result['biosphere'].shape, (4, 5))
        self.assertEqual(result['activity_map'], {('technosphere', 'oil extraction'): 0, ('technosphere', 'lignite extraction'): 1, ('technosphere', 'steam cycle'): 2, ('technosphere', 'wind turbine'): 3, ('technosphere', 'e-Car'): 4, 2: 'steam cycle | electricity | GLO', 1: 'lignite extraction | lignite | GLO', 0: 'oil extraction | oil | GLO', 3: 'wind turbine | electricity | GLO', 4: 'e-Car | transport | GLO'})

    def test_retrieve_activities(self):
        key = retrieve_activities('sample_project', 'technosphere', keys=["('technosphere', 'wind turbine')"])
        name = retrieve_activities('sample_project', 'technosphere', activities=['e-Car'])
        location = retrieve_activities('sample_project', 'technosphere', locations=['GLO'])
        self.assertEqual(key[0]['name'], "wind turbine")
        self.assertEqual(name[0]['name'], "e-Car")
        self.assertEqual(len(location), 5)

    def test_retrieve_methods(self):
        single_result = retrieve_methods('sample_project', ['climate'])
        multi_result = retrieve_methods('sample_project', ['project'])
        self.assertEqual(single_result, [('my project', 'climate change')])
        self.assertEqual(multi_result, [('my project', 'climate change'), ('my project', 'air quality'), ('my project', 'resources')])

    def test_retrieve_envflows(self):
        result = retrieve_envflows('sample_project', biosphere='biosphere', keys="('biosphere', 'PM')")
        self.assertEqual(result[0]['name'], 'Particulate matter, industrial')

class TestPULPO(unittest.TestCase):

    def test_pulpo(self):
        project = 'sample_project'
        database = 'technosphere'
        methods = {"('my project', 'climate change')": 1,
                   "('my project', 'air quality')": 1,
                   "('my project', 'resources')": 0}

        worker = pulpo.PulpoOptimizer(project, database, methods, '')
        worker.biosphere = 'biosphere'
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

        upper_limit = {eCar[0]: 1}
        lower_limit = {eCar[0]: 1}
        worker.instantiate(choices=choices, upper_limit=upper_limit, lower_limit=lower_limit)
        worker.solve()
        result_obj = round(worker.instance.OBJ(), 6)
        result_aux = round(worker.instance.impacts_calculated["('my project', 'resources')"].value, 5)
        self.assertEqual(result_obj, 0.1)
        self.assertEqual(result_aux, 5.1)



