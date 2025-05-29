from pulpo.utils import optimizer, bw_parser, converter, saver, uncertainty
from typing import List, Union
import webbrowser
from tests.rice_database import setup_rice_husk_db
from tests.generic_database import setup_generic_db

class PulpoOptimizer:
    def __init__(self, project: str, database: Union[str, List[str]], method: Union[str, List[str], dict], directory: str):
        """
        Initializes the PulpoOptimizer with project, databases, method, and directory.

        Args:
            project (str): Name of the project.
            database (Union[str, List[str]]): Name of the database or list of two databases
                                               (e.g. foreground and linked background).
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
        self.choices = {}
        self.demand = {}
        self.upper_limit = {}
        self.lower_limit = {}
        self.upper_elem_limit = {}
        self.upper_imp_limit = {}

        bw_parser.set_project(project)

    def get_lci_data(self, seed=None):
        """
        Imports LCI data for the project using the specified database and method.
        """
        self.lci_data = bw_parser.import_data(self.project, self.database, self.method, self.intervention_matrix, seed)

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
        self.choices = choices
        self.demand = demand
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.upper_elem_limit = upper_elem_limit
        self.upper_imp_limit = upper_imp_limit

    def solve(self, GAMS_PATH=False, solver_name=None, options:dict={}):
        """
        Solves the optimization model and calculates additional methods and inventory flows if needed.

        Args:
            GAMS_PATH (bool): Path to GAMS if needed.
            options (dict): Additional options for the solver.

        Returns:
            results: Results of the optimization.
        """
        results, self.instance = optimizer.solve_model(self.instance, GAMS_PATH, solver_name=solver_name, options=options)

        # Post calculate additional methods, in case several methods have been specified and one of them is 0
        if not isinstance(self.method, str):
            if len(self.method) > 1 and 0 in [self.method[x] for x in self.method]:
                self.instance = optimizer.calculate_methods(self.instance, self.lci_data, self.method)

        self.instance = optimizer.calculate_inv_flows(self.instance, self.lci_data)
        return results
    
    def solve_MC(self, n_it=100, GAMS_PATH=False, solver_name=None, options=None):
        """
        Solves the optimization model using Monte Carlo simulation.

        Args:
            n_it (int): Number of Monte Carlo iterations.
            GAMS_PATH (bool): Path to GAMS if needed.
            options (dict): Additional options for the solver.

        Returns:
            results: Results of the optimization.
        """
        # TODO: Analyse also the choices made in each iteration, parallelize? ...
        results = uncertainty.solve_model_MC(self, n_it, GAMS_PATH, solver_name=solver_name, options=options)

        return results

    def run_gsa(self):
        """
        Runs a global sensitivity analysis on the optimization model.

        Returns:
            results: Results of the sensitivity analysis.
        """
        results = uncertainty.run_gsa(self)
        return results
    
    def retrieve_processes(self, keys=None, processes=None, reference_products=None, locations=None):
        """
        Retrieves processes from the database based on given filters.

        Args:
            keys (list): List of keys to filter activities.
            processes (list): List of processes to filter.
            reference_products (list): List of reference products to filter.
            locations (list): List of locations to filter.

        Returns:
            activities: Filtered activities from the database.
        """
        processes = bw_parser.retrieve_processes(self.project, self.database, keys, processes, reference_products,
                                                  locations)
        return processes

    def retrieve_activities(self, keys=None, activities=None, reference_products=None, locations=None):
        """
        Works the same as "retrieve_processes" but with a different name. Will be obsolete in future versions.
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

    def save_results(self, name='results'):
        """
        Saves the results of the optimization to a file.

        Args:
            name (str): Name of the file to save results.
        """
        saver.save_results(self, name)

    def summarize_results(self, zeroes=False):
        """
        Summarizes the results of the optimization.
        """
        saver.summarize_results(self, zeroes)

    def extract_results(self, extractparams:bool=False):
        """
        Summarizes the results of the optimization.
        """
        return saver.extract_results(self, extractparams=extractparams)

def electricity_showcase():
    """
    Opens the electricity showcase notebook in the web browser.
    """
    github_url = 'https://github.com/flechtenberg/pulpo/blob/master/notebooks/electricity_showcase.ipynb'
    nbviewer_url = 'https://nbviewer.jupyter.org/github/' + github_url.split('github.com/')[1]
    webbrowser.open(nbviewer_url)


def hydrogen_showcase():
    """
    Opens the hydrogen showcase notebook in the web browser.
    """
    github_url = 'https://github.com/flechtenberg/pulpo/blob/master/notebooks/hydrogen_showcase.ipynb'
    nbviewer_url = 'https://nbviewer.jupyter.org/github/' + github_url.split('github.com/')[1]
    webbrowser.open(nbviewer_url)


def plastic_showcase():
    """
    Opens the plastic showcase notebook in the web browser.
    """
    github_url = 'https://github.com/flechtenberg/pulpo/blob/master/notebooks/plastic_showcase.ipynb'
    nbviewer_url = 'https://nbviewer.jupyter.org/github/' + github_url.split('github.com/')[1]
    webbrowser.open(nbviewer_url)


def install_rice_husk_db():
    """
    Sets up the rice husk example database.
    """
    setup_rice_husk_db()

def install_generic_db(project="generic_db_project", database="generic_db", n_prod=5, n_proc=3, n_reg=3, n_inputs=4, n_flows=4, n_methods=2, seed=None, return_data=False):
    """
    Sets up the generic LCI database in Brightway2 with specified parameters.

    Args:
        project (str): Name of the Brightway2 project to create or use. Defaults to "generic_db_project".
        database (str): Name of the database to create or use. Defaults to "generic_db".
        n_prod (int): Number of products to generate. Defaults to 5.
        n_proc (int): Maximum number of processes per product. Defaults to 3.
        n_reg (int): Number of regions where processes can be active. Defaults to 3.
        n_inputs (int): Maximum number of inputs per process. Defaults to 4.
        n_flows (int): Number of environmental flows to generate. Defaults to 4.
        n_methods (int): Number of impact assessment methods to create. Defaults to 2.
        seed (int, optional): Seed for reproducibility of random data generation. Defaults to None.
        return_data (bool): If True, returns the generated matrices (technosphere, biosphere, and
            characterization). Defaults to False.

    Returns:
        tuple: If `return_data` is True, returns a tuple containing:
            - technosphere_matrix (np.ndarray): The technosphere matrix.
            - biosphere_matrix (np.ndarray): The biosphere matrix.
            - characterization_matrices (dict): A dictionary of characterization factor matrices.
    """
    return setup_generic_db(project=project, database=database, n_prod=n_prod, n_proc=n_proc, n_reg=n_reg,
                            n_inputs=n_inputs, n_flows=n_flows, n_methods=n_methods,
                            seed=seed, return_data=return_data)

