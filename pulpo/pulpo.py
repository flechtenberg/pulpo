from pulpo.utils import optimizer, bw_parser, converter, saver
from pulpo.utils.uncertainty import preparer, processor, gsa, plots, monte_carlo
from pulpo.utils.uncertainty.preparer import UncertaintySpec
from pulpo.utils.saver import ResultDataDict
from typing import List, Union, Literal, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import array
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
        self.uncertainty_data = None
        self.lci_data = None
        self.instance = None
        self.choices:dict = {}
        self.demand:dict = {}
        self.upper_limit:dict = {}
        self.lower_limit:dict = {}
        self.upper_elem_limit:dict = {}
        self.upper_imp_limit:dict = {}

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

    def solve(self, GAMS_PATH=False, solver_name=None, options=None, neos_email=None):
        """
        Solves the optimization model and calculates additional methods and inventory flows if needed.

        Args:
            GAMS_PATH (bool): Path to GAMS if needed.
            options (dict): Additional options for the solver.

        Returns:
            results: Results of the optimization.
        """
        results, self.instance = optimizer.solve_model(self.instance, GAMS_PATH, solver_name=solver_name, options=options, neos_email=neos_email)

        # Post calculate additional methods, in case several methods have been specified and one of them is 0
        if not isinstance(self.method, str):
            if len(self.method) > 1 and 0 in [self.method[x] for x in self.method]:
                self.instance = optimizer.calculate_methods(self.instance, self.lci_data, self.method)

        self.instance = optimizer.calculate_inv_flows(self.instance, self.lci_data)
        return results
    
    def solve_MC(
        self,
        n_it=100,
        GAMS_PATH=False,
        solver_name=None,
        options=None,
        resample=("A", "B", "Q"),
        n_jobs=-1,
        seed=None,
    ):
        """
        Runs Monte Carlo simulation using pre-sampled LCI data for safe parallelization.
        """
        if self.lci_data is None:
            raise Exception("No LCI data found. Please run get_lci_data() before solve_MC().")

        # Step 1: Pre-sample all LCI variants sequentially
        print(f"Pre-sampling {n_it} LCI matrix sets...")
        samples = monte_carlo.pre_sample_lci_matrices(
            project=self.project,
            databases=self.database,
            method=self.method,
            intervention_matrix_name=self.intervention_matrix,
            n_samples=n_it,
            resample=resample,
            seed=seed,
        )

        # Step 2: Solve each sample in parallel
        print(f"Solving {n_it} Monte Carlo optimizations in parallel...")
        results = monte_carlo.solve_model_MC_pre_sampled(
            pulpo_optimizer=self,
            samples=samples,
            GAMS_PATH=GAMS_PATH,
            solver_name=solver_name,
            options=options,
            n_jobs=n_jobs,
        )
        return results
    
    def run_mc_from_uncertainty(
        self,
        n_samples: int,
        seed: int | None = None,
        n_jobs: int = -1,
        GAMS_PATH=False,
        solver_name: str | None = None,
        options=None,
    ):
        """
        Monte-Carlo optimization driven by prepared uncertainty distributions (no Brightway resample).
        """
        if self.uncertainty_data is None:
            raise Exception("No uncertainty data found. Run import_and_filter_uncertainty_data + apply_uncertainty_strategies first.")
        #if processor.check_missing_uncertainty_data(self.uncertainty_data):
        #    raise Exception("uncertainty_data still contains 'undefined' entries. Fill or drop them before MC.")

        overlays = monte_carlo.pre_sample_from_uncertainty(
            pulpo_optimizer=self, n_samples=n_samples, seed=seed
        )

        return monte_carlo.solve_model_MC_pre_sampled_uncertainty(
            pulpo_optimizer=self,
            overlays=overlays,
            GAMS_PATH=GAMS_PATH,
            solver_name=solver_name,
            options=options,
            n_jobs=n_jobs,
        )

    def solve_CC_problem(
        self,
        lambda_level:float|List, 
        normal_metadata_env_cost:Dict[Tuple[int,str], UncertaintySpec], 
        normal_metadata_var_bounds:Dict[str, Dict[int, UncertaintySpec]],
        gams_path=False, 
        solver_name:Optional[str]=None, 
        options=None, 
        neos_email=None,
        cutoff_value:float=0.01,
        plot_results:bool=False,
        bbox_to_anchor:tuple=(0.65,-3.5),
        cmap_name:str='tab20'
        ) -> Dict[float, ResultDataDict]:
        """
        Solve a (set of) Pareto point(s) for the specified lambda level(s).
        Solves one point if lambda_level is a float, or multiple points if lambda_level is an array of floats.

        Args:
            model_instance (ConcreteModel): 
                The Pyomo model instance.
            lambda_level (float):
                Target confidence/risk threshold (e.g., 0.95 for 95% quantile).
            normal_metadata_env_cost (Dict[Tuple[int,str], UncertaintySpec]): 
                Mean and standard deviation of the environmental costs.
            normal_metadata_var_bounds (Dict[str, Dict[int,UncertaintySpec]]): 
                Standard deviation of the variable bounds.
            gams_path (str or bool, optional): 
                Path to the GAMS solver or True to use the environment variable.
            solver_name (str, optional): 
                The solver to use (e.g. 'cplex', 'baron', or 'xpress').
            options (list, optional): 
                Additional options for the solver.
            neos_email (str, optional): 
                Email for NEOS solver authentication.
            cutoff_value (float, optional): 
                Cutoff value for plotting the Pareto front.
            plot_results (bool, optional): 
                If True, plots the Pareto front after solving. Default is False.
            bbox_to_anchor (tuple, optional): 
                Positioning for the legend in the plot. Default is (1.40, .05).
            cmap_name (str, optional): 
                Colormap name for the plot. Default is 'tab20'.
            
        Returns:
            results (Dict[float,ResultDataDict]): 
                returns a dictionary where each key is a lambda value and
                each value is a dictionary containing the results for that lambda level.
        """
        results = {}
        if isinstance(lambda_level, float):
            optimizer.apply_CC_formulation(self.instance, lambda_level, normal_metadata_env_cost, normal_metadata_var_bounds)
            self.solve(GAMS_PATH=gams_path, solver_name=solver_name, options=options, neos_email=neos_email)
            results[lambda_level] = self.extract_results()
        elif isinstance(lambda_level, np.ndarray) or isinstance(lambda_level, list) or isinstance(lambda_level, array.array):
            for lambda_ in lambda_level:
                print(f'solving CC problem for lambda_QB = {lambda_}')
                # ATTN: We need to store the environmental costs in the results_data and not extract seperate
                # This used to be the case before the results_data was changed, this formulation will run into errors.
                optimizer.apply_CC_formulation(self.instance, lambda_, normal_metadata_env_cost, normal_metadata_var_bounds)
                self.solve(GAMS_PATH=gams_path, solver_name=solver_name, options=options, neos_email=neos_email)
                results[lambda_] = self.extract_results(extractparams=True)
        else:
            raise Exception('lambda_level datatype not implemented, needs to be an array or a float.')
        if plot_results:
            if self.lci_data is None:
                raise Exception('No LCI data found. Please run get_lci_data method first.')
            plots.plot_pareto_front(
                result_data_CC=results, 
                cutoff_value=cutoff_value, 
                method="\n".join(next(iter(self.method)).split("'")[1::2]), 
                process_map_metadata=self.lci_data['process_map_metadata'], 
                bbox_to_anchor=bbox_to_anchor,
                cmap_name=cmap_name
                )
        return results

    
    def import_and_filter_uncertainty_data(
            self, 
            cutoff:float=0,
            scaling_vector_strategy:Literal['naive', 'constructed_demand']='naive',
            result_data:dict={}, 
            plot_results:bool=False,
            plot_n_top_processes:int=10,
        ):
        """
        Imports the uncertain parameter information from the underlying
        databases for the interventions flows (B) and the characterization
        factors (Q) and the variable bounds. Applies filter to reduce the 
        amnount of uncertain parameters for quicker uncertainty assessment
        in subsequent methods, e.g., GSA or CC-optimization.

        Args:
            cutoff (float) - optional:
                (Default: 0., i.e., no filter) cutoff factor to compute minimum contribution value to retain 
                an intervention flow. Multiplied with the LCA score, i.e., a percentage 
                of the total LCA score. The main filtering parameter, the higher the cutoff the more parameters
                will be filtered out.
            scaling_vector_strategy (Literal['naive', 'constructed_demand']) - optional: 
                How to compute scaling vector: 'naive' or 'constructed_demand'. Default: 'naive'
            result_data (dict) - optional: 
                 Solver output dict. Only needed if scaling_vector_strategy is 'constructed_demand', default: None
            plot_results (bool) - optional:
                defaulf False, Set to True if the plot main characterized processes should be created and shown.
            plot_n_top_processes (int) - optional: 
                Number of top items to display in top contribution process plot (default: 10).
        """
        if len(self.method) > 1:
            raise Exception('The uncertainty data import currently only works with a single LCIA method. Please specify a single LCIA method.')
        if self.lci_data is None:
            raise Exception('No LCI data found. Please run get_lci_data method first.')
        # Get the method name
        method = next(iter(self.method))
        paramfilter = preparer.ParameterFilter(
            lci_data=self.lci_data, 
            choices = self.choices,
            demand = self.demand,
            method = method
        )
        filtered_inventory_indcs, filtered_characterization_indcs = paramfilter.apply_filter(
            scaling_vector_strategy=scaling_vector_strategy,
            cutoff=cutoff,
            plot_results=plot_results,
            plot_n_top_processes=plot_n_top_processes,
            result_data=result_data
        )
        # import the uncertainty data for the filtered uncertain parameters
        uncertainty_importer = preparer.UncertaintyImporter(
            lci_data=self.lci_data, 
            bw_databases=self.database, 
            LCIA_method=method,
        )
        self.uncertainty_data = uncertainty_importer.import_uncertainty_data(
            if_indcs=filtered_inventory_indcs,
            cf_indcs=filtered_characterization_indcs,
            choices=self.choices,
            upper_limit=self.upper_limit,
            lower_limit=self.lower_limit,
            upper_elem_limit=self.upper_elem_limit,
            upper_imp_limit=self.upper_imp_limit,
        )

    def apply_uncertainty_strategies(self, strategies:List[processor.UncertaintyStrategyBase]=[], drop_undefined=False, scaling_factor_if=0.5, scaling_factor_cf=0.3, scaling_factor_var_bounds=0.2, **strategy_options):
        """
        Applies the uncertainty gap filling and updating strategies.
        Wrapper for utils.uncertainty.processor.apply_uncertainty_strategies.

        Args:
            strategies (List[UncertaintyStrategyBase]):
                All strategies as instatialized classes for which will manipulate the uncertainty_data
            drop_undefined (bool):
                If True, drops all uncertain parameters with undefined uncertainty types after applying the strategies.
                Default is False.
            scaling_factor_if (float):
                If no strategies are passed the missing uncerainty information in the intervention flows will be 
                filled using triangular strategy either with interpolation or this scaling factor:
                The scaling factor which will be used in the TriangluarBaseStrategy for the intervention flows
                if more than 50% of the intervention flows in the database have no uncertainty information.
                Default is 0.5, meaning that the min and max of the triangular distribution will be set to:
                min = amount - 0.5 * abs(amount)
                max = amount + 0.5 * abs(amount)
            scaling_factor_cf (float):
                If no strategies are passed the missing uncerainty information in the characterization factors will be 
                filled using triangular strategy with this scaling factor.
                The scaling factor which will be used in the TriangluarBaseStrategy for the characterization
                factors. Default is 0.3, meaning that the min and max of the triangular distribution will be set to:
                min = amount - 0.3 * abs(amount)
                max = amount + 0.3 * abs(amount)
            scaling_factor_var_bounds (float):
                If no strategies are passed the missing uncerainty information in the variable bounds will be 
                filled using triangular strategy with this scaling factor.
                The scaling factor which will be used in the TriangluarBaseStrategy for the variable bounds.
                Default is 0.2, meaning that the min and max of the triangular distribution will be set to:
                min = amount - 0.2 * abs(amount)
                max = amount + 0.2 * abs(amount)
            strategy_options:
                Additional options to be passed to the strategy assign methods, e.g., plot_results (bool), which
                shows a figure of the computed average bounds statistics when using the TriangularBaseStrategy.

        """
        if self.uncertainty_data is None:
            raise Exception('No uncertainty data found. Please run import_and_filter_uncertainty_data method first.')
        # If no strategies are provided, use default strategies
        if strategies is None or len(strategies) == 0:
            print('Applying default uncertainty strategies.')
            strategies = processor.uncertainty_strategy_base_case(
                databases=self.database if isinstance(self.database, list) else [self.database],
                method=next(iter(self.method)),
                uncertainty_data=self.uncertainty_data,
                scaling_factor_if=scaling_factor_if,
                scaling_factor_cf=scaling_factor_cf,
                scaling_factor_var_bounds=scaling_factor_var_bounds
            )
        processor.apply_uncertainty_strategies(self.uncertainty_data, strategies, **strategy_options)
        processor.check_missing_uncertainty_data(self.uncertainty_data)
        if drop_undefined:
            self.uncertainty_data = processor.drop_undefined_uncertainty_data(self.uncertainty_data)

    def run_gsa(self, result_data:dict, sample_method, SA_method, sample_size:int, plot_gsa_results:bool=False, top_sensitivity_amt:int=10) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs a global sensitivity analysis on the optimization model.

        Args:
            result_data (dict): 
                Results from the optimization model.
            sample_method:
                Sampling method from SALib.sample (e.g., SALib.sample.saltelli)
            SA_method:
                Sensitivity analysis method from SALib.analyze (e.g., SALib.analyze.sobol)
            sample_size (int):
                Sample size for the GSA.
            plot_gsa_results (bool):
                If True, plots the GSA results.
            top_sensitivity_amt (int):
                Number of top sensitivity indices to plot.

        Returns:
            total_Si (pd.DataFrame): 
                Total sensitivity indices.
            sensitivity_indices (pd.DataFrame):
                First order and second order sensitivity indices (SaLib results).
        """
        if self.uncertainty_data is None:
            raise Exception('No uncertainty data found. Please run import_and_filter_uncertainty_data method first.')
        if processor.check_missing_uncertainty_data(self.uncertainty_data, unc_types=['If', 'Cf']):
            raise Exception('The uncertainty data contains undefined uncertainty types. Please define all uncertainty types before running the GSA.')
        # Run the GSA
        gsa_study = gsa.GlobalSensitivityAnalysis(
            result_data=result_data,
            lci_data=self.lci_data,
            uncertainty_data=self.uncertainty_data,
            sampler=sample_method,
            analyser=SA_method,
            sample_size=sample_size,
            method=next(iter(self.method)),
            plot_gsa_results=plot_gsa_results,
            top_sensitivity_amt=top_sensitivity_amt
        )
        total_Si, sensitivity_indices = gsa_study.perform_gsa()
        return total_Si, sensitivity_indices
        
    def create_CC_formulation(
            self, 
            CC_env_cost:bool=True, 
            CC_var_bounds:List[Literal['upper_imp_limit', 'lower_limit', 'upper_elem_limit', 'upper_limit']] = [],
            plot_analysis_support_plots:bool=False,
            normal_transformation_sample_size:int=100
            ) -> tuple[Dict[Tuple[int,str], UncertaintySpec], Dict[str, Dict[int,UncertaintySpec]]]:
        """
        Creates the data needed for the CC formulation, i.e., the mean and standard deviation of
        the environmental costs based on the L1 norm and the standard deviation of the variable bounds.

        Args:
            CC_env_cost (bool):
                If True, computes the mean and standard deviation of the environmental costs.
            CC_var_bounds (List[Literal['upper_imp_limit', 'lower_limit', 'upper_elem_limit', 'upper_limit']]):
                List of variable bounds to extract the standard deviation for. 
                Options are 'upper_imp_limit', 'lower_limit', 'upper_elem_limit', 'upper_limit'.
            plot_analysis_support_plots (bool):
                If true, shows additional plots to support the analysis. Default is False.
            normal_transformation_sample_size (int):
                Sample size for the normal distribution transformation. Default is 100.

        Returns:
            normal_metadata_env_cost (Dict[Tuple[int,str], UncertaintySpec]):
                Mean and standard deviation of the environmental costs.
            normal_metadata_var_bounds (Dict[str, Dict[int,UncertaintySpec]]):
                Standard deviation of the variable bounds.
        """
        # Determine which uncertainty types to check for missing data
        if CC_env_cost is True and len(CC_var_bounds) > 0:
            unc_types = ['If', 'Cf', 'Var_bounds']
        elif CC_env_cost is True and len(CC_var_bounds) == 0:
            unc_types = ['If', 'Cf']
        elif CC_env_cost is False and  len(CC_var_bounds) > 0:
            unc_types = ['Var_bounds']
        else:
            raise Exception('No CC formulation specified. Please set at least one of the arguments CC_env_cost or CC_var_bounds to True or a non-empty list, respectively.')
        if self.uncertainty_data is None or processor.check_missing_uncertainty_data(self.uncertainty_data, unc_types=unc_types):
            raise Exception('None or incomplete uncertainty data found. Please run import_and_filter_uncertainty_data and apply_uncertainty_strategies methods first.')
        # Transform the uncertainty data to normal distributions
        normal_uncertainty_data = processor.transform_to_normal(
            self.uncertainty_data,
            sample_size=normal_transformation_sample_size, 
            plot_distribution=plot_analysis_support_plots,
            unc_types=unc_types
            )
        # Calculate the mean and standard deviation of the environmental costs based on the L1 norm
        if CC_env_cost:
            normal_metadata_env_cost = optimizer.compute_L1_env_cost_mean_var(
                    normal_uncertainty_data= normal_uncertainty_data,
                    lci_data=self.lci_data,
                    method=next(iter(self.method)),
                    plot_analysis_support_plots=plot_analysis_support_plots
                )
        else:
            normal_metadata_env_cost = {}
        # Extract the standard deviation of the variable bounds
        if CC_var_bounds:
            normal_metadata_var_bounds = {var_bound:normal_uncertainty_data['Var_bounds'][var_bound]['defined'] for var_bound in CC_var_bounds}
        else:
            normal_metadata_var_bounds = {} 
        return normal_metadata_env_cost, normal_metadata_var_bounds

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

