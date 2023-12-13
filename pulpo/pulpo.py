from pulpo.utils import optimizer, bw_parser, converter, saver
from typing import List, Union
import webbrowser

class PulpoOptimizer:
    def __init__(self, project: str, database: str, method: Union[str, List[str], dict], directory: str):
        self.project = project
        self.database = database
        self.biosphere = 'biosphere3'
        self.method = converter.convert_to_dict(method)
        self.directory = directory
        self.lci_data = None
        self.instance = None

    def get_lci_data(self):
        self.lci_data = bw_parser.import_data(self.project, self.database, self.method, self.biosphere)

    def instantiate(self, choices={}, demand={}, upper_limit={},lower_limit={}, upper_elem_limit={}):
        # Instantiate only for those methods that are part of the objecitve
        methods = {h: self.method[h] for h in self.method if self.method[h] !=0}
        data = converter.combine_inputs(self.lci_data, demand, choices, upper_limit, lower_limit, upper_elem_limit, methods)
        self.instance = optimizer.instantiate(data)

    def solve(self, GAMS_PATH=False, options=None):
        results, self.instance = optimizer.solve_model(self.instance, GAMS_PATH, options=options)
        # Post calculate additional methods, in case several methods have been specified and one of them is 0
        if not isinstance(self.method, str):
            if len(self.method) > 1 and 0 in [self.method[x] for x in self.method]:
                self.instance = optimizer.calculate_methods(self.instance, self.lci_data, self.method)
        self.instance = optimizer.calculate_env_flows(self.instance, self.lci_data)
        return results

    def retrieve_activities(self, keys=None, activities=None, reference_products=None, locations=None):
        activities = bw_parser.retrieve_activities(self.project, self.database, keys, activities, reference_products, locations)
        return activities

    def retrieve_envflows(self, keys=None, activities=None, categories=None):
        activities = bw_parser.retrieve_envflows(project=self.project, biosphere=self.biosphere, keys=keys, activities=activities, categories=categories)
        return activities

    def retrieve_methods(self, string=""):
        methods = bw_parser.retrieve_methods(self.project, string)
        return methods

    def save_results(self, choices={}, constraints={}, demand={}, name='results.xlxs'):
        saver.save_results(self.instance, self.project, self.database, choices, constraints, demand, self.lci_data['activity_map'], self.lci_data['elem_map'], self.directory, name)

    def summarize_results(self, choices={}, constraints={}, demand={}, zeroes=False):
            saver.summarize_results(self.instance, self.project, self.database, choices, constraints, demand, self.lci_data['activity_map'], zeroes)

def electricity_showcase():
    github_url = 'https://github.com/flechtenberg/pulpo/blob/develop/notebooks/electricity_showcase.ipynb'  # Replace with your GitHub URL
    nbviewer_url = 'https://nbviewer.jupyter.org/github/' + github_url.split('github.com/')[1]
    webbrowser.open(nbviewer_url)

def hydrogen_showcase():
    github_url = 'https://github.com/flechtenberg/pulpo/blob/develop/notebooks/hydrogen_showcase.ipynb'  # Replace with your GitHub URL
    nbviewer_url = 'https://nbviewer.jupyter.org/github/' + github_url.split('github.com/')[1]
    webbrowser.open(nbviewer_url)