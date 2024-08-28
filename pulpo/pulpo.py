from pulpo.utils import optimizer, bw_parser, converter, saver
from typing import List, Union
import webbrowser


class PulpoOptimizer:
    def __init__(self, project: str, database: str, method: Union[str, List[str], dict], directory: str):
        """
        Initializes the PulpoOptimizer with project, database, method, and directory.

        Args:
            project (str): Name of the project.
            database (str): Name of the database.
            method (Union[str, List[str], dict]): Method(s) for optimization.
            directory (str): Directory for saving results.
        """
        self.project = project
        self.database = database
        self.intervention_matrix = 'biosphere3'
        self.method = converter.convert_to_dict(method)
        self.directory = directory
        self.lci_data = None
        self.instance = None

    def get_lci_data(self):
        """
        Imports LCI data for the project using the specified database and method.
        """
        self.lci_data = bw_parser.import_data(self.project, self.database, self.method, self.intervention_matrix)

    def instantiate(self, choices={}, demand={}, upper_limit={}, lower_limit={}, upper_elem_limit={},
                    upper_imp_limit={}):
        """
        Combines inputs and instantiates the optimization model.

        Args:
            choices (dict): Choices for the model.
            demand (dict): Demand data.
            upper_limit (dict): Upper limit constraints.
            lower_limit (dict): Lower limit constraints.
            upper_elem_limit (dict): Upper elemental limit constraints.
            upper_imp_limit (dict): Upper impact limit constraints.
        """
        # Instantiate only for those methods that are part of the objective or the limits
        methods = {h: self.method[h] for h in self.method if self.method[h] != 0 or h in upper_imp_limit}
        data = converter.combine_inputs(self.lci_data, demand, choices, upper_limit, lower_limit, upper_elem_limit,
                                        upper_imp_limit, methods)
        self.instance = optimizer.instantiate(data)

    def solve(self, GAMS_PATH=False, options=None):
        """
        Solves the optimization model and calculates additional methods and inventory flows if needed.

        Args:
            GAMS_PATH (bool): Path to GAMS if needed.
            options (dict): Additional options for the solver.

        Returns:
            results: Results of the optimization.
        """
        results, self.instance = optimizer.solve_model(self.instance, GAMS_PATH, options=options)

        # Post calculate additional methods, in case several methods have been specified and one of them is 0
        if not isinstance(self.method, str):
            if len(self.method) > 1 and 0 in [self.method[x] for x in self.method]:
                self.instance = optimizer.calculate_methods(self.instance, self.lci_data, self.method)

        self.instance = optimizer.calculate_inv_flows(self.instance, self.lci_data)
        return results

    def retrieve_activities(self, keys=None, activities=None, reference_products=None, locations=None):
        """
        Retrieves activities from the database based on given filters.

        Args:
            keys (list): List of keys to filter activities.
            activities (list): List of activities to filter.
            reference_products (list): List of reference products to filter.
            locations (list): List of locations to filter.

        Returns:
            activities: Filtered activities from the database.
        """
        activities = bw_parser.retrieve_processes(self.project, self.database, keys, activities, reference_products,
                                                  locations)
        return activities

    def retrieve_envflows(self, keys=None, activities=None, categories=None):
        """
        Retrieves environmental flows from the database based on given filters.

        Args:
            keys (list): List of keys to filter environmental flows.
            activities (list): List of activities to filter.
            categories (list): List of categories to filter.

        Returns:
            activities: Filtered environmental flows from the database.
        """
        activities = bw_parser.retrieve_env_interventions(project=self.project,
                                                          intervention_matrix=self.intervention_matrix, keys=keys,
                                                          activities=activities, categories=categories)
        return activities

    def retrieve_methods(self, string=""):
        """
        Retrieves methods from the database based on a search string.

        Args:
            string (str): Search string for methods.

        Returns:
            methods: List of methods that match the search string.
        """
        methods = bw_parser.retrieve_methods(self.project, string)
        return methods

    def save_results(self, choices={}, constraints={}, demand={}, name='results.xlxs'):
        """
        Saves the results of the optimization to a file.

        Args:
            choices (dict): Choices for the model.
            constraints (dict): Constraints applied during optimization.
            demand (dict): Demand data used in optimization.
            name (str): Name of the file to save results.
        """
        saver.save_results(self.instance, self.project, self.database, choices, constraints, demand,
                           self.lci_data['process_map'], self.lci_data['intervention_map'], self.directory, name)

    def summarize_results(self, choices={}, constraints={}, demand={}, zeroes=False):
        """
        Summarizes the results of the optimization.

        Args:
            choices (dict): Choices for the model.
            constraints (dict): Constraints applied during optimization.
            demand (dict): Demand data used in optimization.
            zeroes (bool): Whether to include zero values in the summary.
        """
        saver.summarize_results(self.instance, choices, constraints, demand,
                                self.lci_data['process_map'], zeroes)


def electricity_showcase():
    """
    Opens the electricity showcase notebook in the web browser.
    """
    github_url = 'https://github.com/flechtenberg/pulpo/blob/develop/notebooks/electricity_showcase.ipynb'
    nbviewer_url = 'https://nbviewer.jupyter.org/github/' + github_url.split('github.com/')[1]
    webbrowser.open(nbviewer_url)


def hydrogen_showcase():
    """
    Opens the hydrogen showcase notebook in the web browser.
    """
    github_url = 'https://github.com/flechtenberg/pulpo/blob/develop/notebooks/hydrogen_showcase.ipynb'
    nbviewer_url = 'https://nbviewer.jupyter.org/github/' + github_url.split('github.com/')[1]
    webbrowser.open(nbviewer_url)
