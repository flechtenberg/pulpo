"""
stats.py

Module for defining and solving life cycle assessment (LCA) case studies with PULPO,
performing uncertainty analysis (filtering, sampling, fitting),
running global sensitivity analyses, and computing Pareto fronts
via epsilon‐constraint and adaptive‐sampling solvers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import stats_arrays
import scipy.stats
import pandas as pd
import numpy as np
import os
from pulpo import pulpo
import scipy.sparse as sparse
from time import time
import stats_arrays
import matplotlib.pyplot as plt
import matplotlib as mpl
import textwrap
import bw2data
import bw2calc
import ast
import array
from typing import Union, List, Optional, Dict, Tuple

# === Case Studies ===
class BaseCaseStudy:
    """
    Abstract base class for a PULPO‐based LCA case study.

    Attributes:
        project (str): PULPO project name.
        database (str or list of str): Name(s) of the inventory database.
        method (str, list, or dict): LCIA method specification.
        directory (str): Working directory for inputs/outputs.
        pulpo_worker (pulpo.PulpoOptimizer): Solver instance, created by create_pulpo_worker.
        demand (dict): Functional unit demands (process → amount).
        choices (dict): Choices (process → capacity) in the model.
    """

    def __init__(self, project:str, database: Union[str, List[str]], method:Union[str, List[str], dict], directory :str ):
        """
        Initialize the case study with project, database, method, and directory.

        Args:
            project: Name of the PULPO project.
            database: Inventory database(s) to load.
            method: LCIA method(s) to apply.
            directory: Path to working directory for saving results.
        """
        self.project = project
        self.database = database
        self.method = method
        self.directory = directory
        self.demand:dict = {}
        self.choices:dict = {}

    def create_pulpo_worker(self):
        """
        Instantiate a PulpoOptimizer and import LCI data.

        Creates `self.pulpo_worker` and calls its get_lci_data() to load
        life cycle inventory matrices and metadata.
        """
        # Create a **PulpoOptimizer** instance. This class is used to interact with the LCI database and solve the optimization problem. It is specified by the project, database, method and directory.
        self.pulpo_worker = pulpo.PulpoOptimizer(self.project, self.database, self.method, self.directory)
        # Import LCI data. After initializing the PulpoOptimizer instance, the LCI data is imported from the database.
        self.pulpo_worker.get_lci_data()

    def solve_and_summarize(self, file_name) -> dict:
        """
        Solve the optimization model and summarize results.

        Args:
            file_name: Filename for saving results (ignored if saver used directly).

        Returns:
            result_data (dict): Extracted results dictionary.
        """

        # Instantiat and solve the optimization model
        # self.pulpo_worker.solve()
        # options = {'NEOS_EMAIL':'b.j.p.m.haussling.lowgren@cml.leidenuniv.nl'}
        # self.pulpo_worker.solve(solver_name='cplex', options=options)
        self.pulpo_worker.solve()
        # Save and summarize the results
        result_data = self.pulpo_worker.extract_results()
        self.pulpo_worker.summarize_results(zeroes=True)
        # self.pulpo_worker.save_results(result_data, file_name) # ATTN: This still does not work with the saver code probably still a mistake in there
        return result_data


class RiceHuskCase(BaseCaseStudy):
    """
    Case study for rice‐husk processing.

    Defines the functional unit (processed rice), choice sets
    (husk supply, boilers, auxiliary), and instantiates the PULPO model.
    """
    def __init__(self):
        """
        Set up default project, database, method, and directory
        for the rice‐husk example.
        """
        # Set the parameters for the rise husk example to instancialize PULPO
        self.project = "rice_husk_example" 
        if self.project not in bw2data.projects: #ATTN: test
            pulpo.install_rice_husk_db()
        self.database = "rice_husk_example_db"
        self.method = {"('my project', 'climate change')":1}
        self.directory = os.path.join(os.path.dirname(os.getcwd()), 'develop_tests/data')

    def define_problem(self):
        """
        Specify the functional unit, define choice options and capacities,
        and instantiate the PULPO model with self.pulpo_worker.instantiate().
        """
        # Specify the **functional unit**. In this case, the functional unit is 1 Mt of processed rice. PULPO implements a search function (```retrieve_processes```) to find the processes that match the specified reference products (alternatively: keys, process name, region).
        rice_factory = self.pulpo_worker.retrieve_processes(reference_products='Processed rice (in Mt)')
        self.demand = {rice_factory[0]: 1}
        # Specify the **choices**. Here, the choices are regional 🌐 choices for rise husk collections, and technological ⛏ choices for boiler type selection.
        # The auxiliar choices are needed to resolve the issue that rice, when not used in the boiler must be burned instead. 
        # (*At this point, just accept. If you are curious about how this multi-functionality is technically adressed, refer to the paper, or reach out.*)
        ## Rise husk collection
        rice_husk_processes = ["Rice husk collection 1",
                    "Rice husk collection 2",
                    "Rice husk collection 3",
                    "Rice husk collection 4",
                    "Rice husk collection 5",]
        rice_husk_collections = self.pulpo_worker.retrieve_processes(processes=rice_husk_processes)
        ## Boilers
        boiler_processes = ["Natural gas boiler",
                            "Wood pellet boiler",
                            "Rice husk boiler"]
        boilers = self.pulpo_worker.retrieve_processes(processes=boiler_processes)
        ## Auxiliar 
        auxiliar_processes = ["Rice husk market",
                            "Burning of rice husk"]
        auxiliar = self.pulpo_worker.retrieve_processes(processes=auxiliar_processes)
        ## Combine to create the choices dictionary
        ## For each kind of choice, assign a 'label' (e.g. 'boilers')
        ## To each possible choice, assign a process capacity. In the 'unconstrained' case, set this value very high (e.g. 1e10, but depends on the scale of the functional unit)
        self.choices = {'Rice Husk (Mt)': {rice_husk_collections[0]: 0.03,
                                    rice_husk_collections[1]: 0.03,
                                    rice_husk_collections[2]: 0.03,
                                    rice_husk_collections[3]: 0.03,
                                    rice_husk_collections[4]: 0.03},
                'Thermal Energy (TWh)': {boilers[0]: 1e10,
                                            boilers[1]: 1e10,
                                            boilers[2]: 1e10},
                'Auxiliar': {auxiliar[0]: 1e10,
                                auxiliar[1]: 1e10}}
        self.pulpo_worker.instantiate(choices=self.choices, demand=self.demand)


class ElectricityCase(BaseCaseStudy):
    """
    Case study for the electricity showcase problem.

    Defines the functional unit (processed rice), choice sets
    (husk supply, boilers, auxiliary), and instantiates the PULPO model.
    """
    def __init__(self):
        """
        Set up default project, database, method, and directory
        for the electricity showcase problem.
        """
        self.project = "pulpo"
        self.database = "cutoff38"
        self.method = {"('IPCC 2013', 'climate change', 'GWP 100a')": 1,
                "('ReCiPe Endpoint (E,A)', 'resources', 'total')": 0,
                "('ReCiPe Endpoint (E,A)', 'human health', 'total')": 0,
                "('ReCiPe Endpoint (E,A)', 'ecosystem quality', 'total')": 0,
                "('ReCiPe Midpoint (E) V1.13', 'ionising radiation', 'IRP_HE')": 0}
        self.directory = os.path.join( os.path.dirname(os.getcwd()), 'develop_tests/data')

    def define_problem(self):
        """
        Specify the functional unit, define choice options and capacities,
        and instantiate the PULPO model with self.pulpo_worker.instantiate().
        """
        # Retrieve the electricity market
        activities = ["market for electricity, high voltage"]
        reference_products = ["electricity, high voltage"]
        locations = ["DE"]
        electricity_market = self.pulpo_worker.retrieve_activities(activities=activities,
                                                            reference_products=reference_products,
                                                            locations=locations)
        # Specify the functional unit as demand dictionary
        self.demand = {electricity_market[0]: 1.28819e+11}
        # Retrieve the choices
        activities = ["electricity production, lignite", 
                    "electricity production, hard coal",
                    "electricity production, nuclear, pressure water reactor",
                    "electricity production, wind, 1-3MW turbine, onshore"]
        reference_products = ["electricity, high voltage"]
        locations = ["DE"]
        electricity_activities = self.pulpo_worker.retrieve_activities(activities=activities,
                                                                reference_products=reference_products,
                                                                locations=locations)
        # Specify the choices dictionary
        self.choices  = {'electricity': {electricity_activities[0]: 1e16,
                                    electricity_activities[1]: 1e16,
                                    electricity_activities[2]: 1e16,
                                    electricity_activities[3]: 1e16}}
        # Instantiate and solve the problem (here with HiGHS)
        self.pulpo_worker.instantiate(choices=self.choices, demand=self.demand)


class AmmoniaCase(BaseCaseStudy):
    """
    Case study for the reducts Ammonia case study.

    Defines the functional unit, choice sets, and instantiates the PULPO model.
    """
    def __init__(self):
        """
        Set up default project, database, method, and directory
        for the Ammonia case study.
        """
        self.project = "ammonia_reduced"
        self.database = ["ecoinvent-3.10-cutoff", "ammonia-reduced"]
        self.method = "('IPCC 2021', 'climate change', 'GWP 100a, incl. H and bio CO2')"
        self.directory = os.path.join(os.path.dirname(os.getcwd()), 'develop_tests/data')

    def create_pulpo_worker(self):
        """
        Instantiate a PulpoOptimizer and import LCI data.

        Creates `self.pulpo_worker` and calls its get_lci_data() to load
        life cycle inventory matrices and metadata.
        """
        # Create a **PulpoOptimizer** instance. This class is used to interact with the LCI database and solve the optimization problem. It is specified by the project, database, method and directory.
        self.pulpo_worker = pulpo.PulpoOptimizer(self.project, self.database, self.method, self.directory)
        self.pulpo_worker.intervention_matrix="ecoinvent-3.10-biosphere"
        # Import LCI data. After initializing the PulpoOptimizer instance, the LCI data is imported from the database.
        self.pulpo_worker.get_lci_data()

    def define_problem(self):
        """
        Specify the functional unit, define choice options and capacities,
        and instantiate the PULPO model with self.pulpo_worker.instantiate().
        """
        
        choices_biomethane_CS = ["biogas upgrading to biomethane, chemical scrubbing"]
        choices_biomethane_CSwCCS = ["biogas upgrading to biomethane, chemical scrubbing w/ CCS"]
        choices_biomethane_WS = ["biogas upgrading to biomethane, water scrubbing"]
        choices_biomethane_WSwCCS = ["biogas upgrading to biomethane, water scrubbing w/ CCS"]

        choices_methane_market = [
            "market for bio methane",
            "market group for natural gas, high pressure"
        ]

        choices_hydrogen_SMR = ["hydrogen production, steam methane reforming fg"]
        choices_hydrogen_SMRwCCS = ["hydrogen production, steam methane reforming, w/ CCS"]
        choices_hydrogen_PEM = ["hydrogen production, PEM electrolysis, yellow"]
        choices_hydrogen_plastic = ["hydrogen production, plastics gasification"]
        choices_hydrogen_plasticCCS = ["hydrogen production, plastics gasification, w/ CCS"]

        choices_hydrogen_market = [
            "market for hydrogen",
            "market for hydrogen, gaseous, low pressure"
        ]

        choices_heat_H2 = ["heat from hydrogen"]
        choices_heat_CH4wCCS = ["heat from methane, w/ CCS"]
        choices_heat_CH4 = ["heat from methane"]

        choices_ammonia = [
            "ammonia production, steam methane reforming",
            # "ammonia production, steam methane reforming, w/ CCS",
            "ammonia production, from nitrogen and hydrogen"
        ]

        choices_ammonia_market = [
            "market for ammonia",
            "market for ammonia, anhydrous, liquid"
        ]

        # Retrieve activities for each category and assign to appropriately named variables
        # Retrieve activities for each category and assign to appropriately named variables
        # Biomethane upgrading
        biomethane_activities_CS = self.pulpo_worker.retrieve_activities(activities=choices_biomethane_CS, locations=["RER", "Europe without Switzerland"])
        biomethane_activities_CSwCCS = self.pulpo_worker.retrieve_activities(activities=choices_biomethane_CSwCCS, locations=["RER", "Europe without Switzerland"])
        biomethane_activities_WS = self.pulpo_worker.retrieve_activities(activities=choices_biomethane_WS, locations=["RER", "Europe without Switzerland"])
        biomethane_activities_WSwCCS = self.pulpo_worker.retrieve_activities(activities=choices_biomethane_WSwCCS, locations=["RER", "Europe without Switzerland"])

        methane_market_activities = self.pulpo_worker.retrieve_activities(activities=choices_methane_market, locations=["RER", "Europe without Switzerland"])
        # Hydrogen
        hydrogen_activities_PEM = self.pulpo_worker.retrieve_activities(activities=choices_hydrogen_PEM, locations=["RER", "Europe without Switzerland"])
        hydrogen_activities_plastic = self.pulpo_worker.retrieve_activities(activities=choices_hydrogen_plastic, locations=["RER", "Europe without Switzerland"])
        hydrogen_activities_plasticCCS = self.pulpo_worker.retrieve_activities(activities=choices_hydrogen_plasticCCS, locations=["RER", "Europe without Switzerland"])
        hydrogen_activities_SMR = self.pulpo_worker.retrieve_activities(activities=choices_hydrogen_SMR, locations=["RER", "Europe without Switzerland"])
        hydrogen_activities_SMRwCCS = self.pulpo_worker.retrieve_activities(activities=choices_hydrogen_SMRwCCS, locations=["RER", "Europe without Switzerland"])
        hydrogen_market_activities = self.pulpo_worker.retrieve_activities(activities=choices_hydrogen_market, locations=["RER", "Europe without Switzerland"])
        # Heat
        heat_activities_CH4 = self.pulpo_worker.retrieve_activities(activities=choices_heat_CH4, locations=["RER", "Europe without Switzerland"])
        heat_activities_CH4wCCS = self.pulpo_worker.retrieve_activities(activities=choices_heat_CH4wCCS, locations=["RER", "Europe without Switzerland"])
        heat_activities_H2 = self.pulpo_worker.retrieve_activities(activities=choices_heat_H2, locations=["RER", "Europe without Switzerland"])
        # Ammonia
        ammonia_activities = self.pulpo_worker.retrieve_activities(activities=choices_ammonia, locations=["RER", "Europe without Switzerland"])
        ammonia_market_activities = self.pulpo_worker.retrieve_activities(activities=choices_ammonia_market, locations=["RER", "Europe without Switzerland"])
        
        # Set the demand and other global parameters for the problem definition
        demand_value = 3000e6 
        manure_biogas = 1.8e9 # Setting the manure biomgas availability for 2030
        scale_by_demand = False
        if scale_by_demand:
            scaling_value = demand_value
        else:
            scaling_value = 1

        # Choices as constraints
        self.choices = {
            "ammonia": {x: 1e20 for x in ammonia_activities},
            "methane_market": {x: 1e20 for x in methane_market_activities},
            "hydrogen_market": {x: 1e20 for x in hydrogen_market_activities},
            "ammonia_market": {x: 1e20 for x in ammonia_market_activities},
        }
        self.choices["hydrogen"] = {
            hydrogen_activities_PEM[0] : .3*.1e9/scaling_value, # 0.1 Mt H2 -- 30% of PEM to NH3
            hydrogen_activities_SMR[0] : 1e20, # SMR
            hydrogen_activities_SMRwCCS[0] : .3*.5e9/scaling_value, # 0.5 Mt H2 -- 30% of SMR w CCS H2 for NH3
            hydrogen_activities_plastic[0] : .3*40e6/scaling_value, # 40 kt H2 -- 30% of plastic gasification for NH3
            hydrogen_activities_plasticCCS[0] : .3*1e6/scaling_value, # 10 kt H2 -- 30% of plastic gasification w CCS for NH3
        }
        self.choices["heat"] = {
            heat_activities_CH4[0]: 1e20,
            heat_activities_CH4wCCS[0]: .1*1.4e10/scaling_value, # Assuming 10% of the heat currently needed for Ammonia produciton
            heat_activities_H2[0]:1*1.4e10/scaling_value, # Assuming 10% of the heat currently needed for Ammonia produciton
        }
        self.choices["biomethane"] = {
            biomethane_activities_CS[0]: 1e20, # Assumingly the same potential as WS
            biomethane_activities_CSwCCS[0]: .1*.2*manure_biogas/scaling_value, # Guess: the same as WS wCCS
            biomethane_activities_WS[0]: 1e20, # Assumingly the base case technology
            biomethane_activities_WSwCCS[0]: .1*.2*manure_biogas/scaling_value #  20% (biomethane for NH3) of 10% (CCS for biomethane )of 1.8 mt (biomethane) for which is the max capacity 2030
        }
        
        # Additional constraints
        anaerobic_digestion = self.pulpo_worker.retrieve_activities(activities='anaerobic digestion of animal manure, with biogenic carbon uptake', locations=["RER", "Europe without Switzerland"])
        upper_bound = {
            anaerobic_digestion[0]:.2*1.8e9/scaling_value # 20% of 1.8 mt which is the max capacity 2030
            } 

        # Demand
        ammonia_market = self.pulpo_worker.retrieve_activities(activities="market for ammonia")
        self.demand = {"ammonia_market": demand_value/scaling_value} # Ammonia production capacity Germany
        
        # Instantiate the pulpo instance
        self.pulpo_worker.instantiate(choices=self.choices, demand=self.demand, upper_limit=upper_bound)



class CCFormulationBase:
    """
    Main class for Chance Constraint formulation containing only Normal distributed uncertain parameters,
    directly formulates the neccessary data for the Pareto problem upon initialization.

    Subclasses ca override `formulate` and `update_problem` to define a
    concrete Pyomo model and its λ‐level updates.
    """
    def __init__(self,
                 unc_metadata: Dict[str,pd.DataFrame], # ATTN: It would be nicer to have the layer conatining the uncertainty data either in stast_arrays numbpy format or as dict
                 pulpo_worker,
                 method:str,
                 choices:dict,
                 plot_normal_fit_distribution:bool=False,
                 sample_size_normal_fit:int=1000000, 
                 ):
        """
        Initialize the chance-constraint formulation.

        Stores metadata for the uncertain parameters, needed for the chance constrain formulations.
        Transforms the uncertainty information into normal distributions.
        then calls `formulate` to assemble the base model.

        Args:
            unc_metadata (dict[str:pd.DataFrame]):
                Uncertainty metadata containing the uncertainty informtaion about the uncertain parameters
                for all constraints which are to be chance constained.
                    { '[name of uncertain parameter as defined in pyomo model]':
                            pd.DataFrame(
                                rows: parameters - indexed as in pyomo model
                                columns: statistical params - as defined in stats_arrays, e.g., 'scale', 'loc', ...
                            ),
                        '...': pd.DataFrame(...),
                    }
            pulpo_worker:
                The initialized deterministic Pyomo model.
            method (str):
                LCIA method name (used in impact calculations). 
                Needed to later compare the CC optimization results.
            choices (dict):
                Optimization choices defined for the case study.
                Needed to later compare the CC optimization results.
            plot_distributions (bool):
                If True, display a histogram + fitted-normal curve for each parameter.
                Defaults to False.
            sample_size (int):
                Number of random draws per parameter when fitting normal distribution
                to uncertain parameters. Defaults to 1_000_000.
        """
        self.unc_metadata = unc_metadata
        self.normal_metadata = self._transform_to_normal(sample_size=sample_size_normal_fit, plot_distribution=plot_normal_fit_distribution)
        self.pulpo_worker = pulpo_worker
        self.method = method
        self.choices = choices
        self.formulate()
    
    def formulate(self) -> None:
        """
        Build the initial chance constraint formulation (to be overridden).

        Returns:
            None
        """
        pass

    def _transform_to_normal(self, sample_size:int=100000, plot_distribution:bool=False) -> Dict[str,pd.DataFrame]:
        """
        Fit Normal distributions to all CF and IF uncertainty metadata.

        Uses the UncertaintyProcessor to convert any non‐normal uncertainty
        definitions into equivalent Normal distributions.

        Args:
            plot_distributions (bool):
                If True, display a histogram + fitted-normal curve for each parameter.
                Defaults to False.
            sample_size (int):
                Number of random draws per parameter when fitting normal distribution
                to uncertain parameters. Defaults to 1_000_000.

        Returns:
            normal_metadata (Dict[str,pd.DataFrame]): Fitted Normal loc/scale for parameters in chance constaints (e.g., "cf", "if").
        """
        normal_metadata = {}
        for var_name, metadata_df in self.unc_metadata.items():
            normal_metadata[var_name] = UncertaintyProcessor.fit_normals(metadata_df, sample_size=sample_size, plot_distributions=plot_distribution)
        # ATTN: Check if the fit_normals runs through with 0 as standard deviations
        return normal_metadata

    def update_problem(self, lambda_level:float):
        """
        Inject or update the ε‐constraint for a given risk level (to be overridden).

        Modifies the existing Pyomo model to enforce that the specified
        chance constraint (e.g. P{impact ≤ threshold} ≥ λ) is satisfied
        at the current `lambda_level`. This supports tracing the Pareto front.

        Args:
            lambda_level (float):
                Target confidence/risk threshold (e.g., 0.95 for 95% quantile).
        """
        pass
class CCFormulationObjL1(CCFormulationBase):
    """
    Implements an individual chance‐constraint formulation on the objective using the L1 norm on normally distributed uncertainties.

    This subclass approximates all uncertain intervention flows and characterization factors as Normal(μ,σ²),
    computes the aggregated standard deviation of total environmental cost under an L1 norm, and then
    traces Pareto‐optimal solutions by varying the confidence level (λ).
    """

    def formulate(self):
        """
        Prepare the variance‐based chance‐constraint formulation.

        1. Transform CF and IF metadata into fitted Normal distributions.
        2. Compute the standard deviation of environmental cost contributions.
        3. Compute the mean environmental cost.
        4. Check that the variance‐based z‐values are within acceptable bounds.
        """
        self.envcost_std = self.compute_envcost_variance(self.normal_metadata['cf'], self.normal_metadata['if'])
        self.envcost_mean = self.compute_envcost_mean()
        self.check_envcost_variance(self.envcost_std)


    def _extract_process_ids_and_intervention_flows_for_env_cost_variance(self) -> tuple[array.array, pd.DataFrame]:
        """
        Identify which processes and flows feed into the environmental-cost variance.

        Extracts the array of process IDs that have uncertain intervention flows or CFs,
        then computes per-process cost standard deviations and z-scores, printing any
        outliers and plotting the z-value distribution for inspection.

        Returns:
            process_id_uncertain_if (array.array): IDs of processes with uncertain IF contributions.
            envcost_std_mean (pd.DataFrame): DataFrame indexed by process ID with columns
                ['std', 'mean', 'z', 'metadata'] summarizing variance diagnostics.
        """
        # To Compute the variance of the environmental costs we must extract all processes which contain:
        # - an uncertain intervention flow
        process_id_uncertain_if = self.unc_metadata["if"].index.get_level_values(1).values
        # - an intervention flow associated with an uncertain characterization factor
        process_id_associated_cf = self.pulpo_worker.lci_data['intervention_matrix'][self.unc_metadata["cf"].index,:].nonzero()[1]
        process_ids = np.unique(np.append(process_id_associated_cf, process_id_uncertain_if))
        # Get the intervention flows to the uncertain characterization factors
        intervention_flows_extracted = pd.DataFrame.sparse.from_spmatrix(
            self.pulpo_worker.lci_data['intervention_matrix'][self.unc_metadata["cf"].index.values,:][:,process_ids],
            index=self.unc_metadata["cf"].index,
            columns=process_ids
        )
        return process_ids, intervention_flows_extracted


    def compute_envcost_variance(self, cf_normal_metadata_df, if_normal_metadata_df) -> pd.Series:
        """
        Calculate the standard deviation of total environmental cost across processes.

        Uses the fitted Normal distributions for CFs and IFs to derive per-process
        cost variances, then aggregates them (under independence) to obtain the
        overall cost standard deviations.
        $$
        \sigma_{q_hb_j} =\sqrt{\sum_e \big(\mu_{q_{h,e}}^2\sigma_{b_{e,j}}^2 + \mu_{b_{e,j}}^2\sigma_{q_{h,e}}^2 + \sigma_{b_{e,j}}^2 \sigma_{q_{h,e}}^2\big)}
        $$

        Args:
            cf_normal_metadata_df (pd.DataFrame):
                normal distribution information of the characterization factors from "unc_metadata['cf']"
            if_normal_metadata_df (pd.DataFrame):
                normal distribution information of the inventory flows from "unc_metadata['if']"      

        Returns:
            envcost_std (pd.Series):
                Indexed by process ID, with each value equal to the standard deviation
                of that process’s total cost contribution.
        """
        process_ids, intervention_flows_extracted = self._extract_process_ids_and_intervention_flows_for_env_cost_variance()
        envcost_std = {}
        for process_id in process_ids:
            # compute the mu_{q_{h,e}}^2 * sigma_{b_{e,j}}^2
            if process_id in if_normal_metadata_df.index.get_level_values(level=1):
                intervention_flow_std = if_normal_metadata_df.xs(process_id, level=1, axis=0, drop_level=True)['scale']
                characterization_factor_mean = pd.Series(
                    self.pulpo_worker.lci_data["matrices"][self.method].diagonal()[
                        intervention_flow_std.index.get_level_values(level=0)
                        ],
                    index=intervention_flow_std.index.get_level_values(level=0)
                )
                # Reindex so that we can perform a matrix multiplication on all intervention flows
                characterization_factor_mean = characterization_factor_mean.reindex(intervention_flow_std.index, axis=0, level=0)
                mu_q2_sigma_b2 = characterization_factor_mean.pow(2).mul(intervention_flow_std.pow(2), axis=0)
            else:
                mu_q2_sigma_b2 = pd.Series([0])
            # compute the mu_{b_{e,j}}^2 * sigma_{q_{h,e}}^2
            if (intervention_flows_extracted[process_id] > 0).any():
                characterization_factor_std = cf_normal_metadata_df['scale']
                # Reindex so that we can perform a matrix multiplication on all characterization factors
                intervention_flow_mean = intervention_flows_extracted[process_id]
                sigma_q2_mu_b2 = characterization_factor_std.pow(2).mul(intervention_flow_mean.pow(2), axis=0)
            else:
                sigma_q2_mu_b2 = pd.Series([0])
            # compute the sigma_{b_{e,j}}^2 * sigma_{q_{h,e}}^2
            if (intervention_flows_extracted[process_id] > 0).any() and process_id in if_normal_metadata_df.index.get_level_values(level=1):
                sigma_q2_sigma_b2 = characterization_factor_std.pow(2).mul(intervention_flow_std.pow(2))
            else:
                sigma_q2_sigma_b2 = pd.Series([0])
            # Take the sqrt of the sum over the std terms and the intervention flows
            envcost_std[process_id] = np.sqrt(mu_q2_sigma_b2.sum() + sigma_q2_sigma_b2.sum() + sigma_q2_mu_b2.sum())
        return envcost_std

    def compute_envcost_mean(self) -> dict:
        """
        Compute the expected (mean) total environmental cost per process.

        Returns:
            pd.Series:
                Indexed by process ID, with each value equal to the expected cost
                contribution of that process.
        """
        # Compute the mean of the environmental costs to be used together with the standard deviation to update the uncertain parameters in line with chance constraint formulation
        envcost_raw = self.pulpo_worker.lci_data['matrices'][self.method].diagonal() @ self.pulpo_worker.lci_data['intervention_matrix']
        # ATTN: The env_cost_raw must be updated with the potentially different means after fitting the normal distribution
        envcost_mean = pd.Series(envcost_raw).to_dict()
        return envcost_mean

    def check_envcost_variance(self, envcost_std:dict):
        """
        Validate z-scores (std/mean) for environmental cost contributions.

        Computes z = std/|mean| for each process and raises a warning or error
        if any z exceed reasonable thresholds (indicating potential numerical issues).

        Args:
            envcost_std (dict):
                Standard deviation per process (from `compute_envcost_variance`).
        """
        envcost_mean = self.compute_envcost_mean()
        # ATTN: For the environmental costs with very large z-value we should check if they come from interpolated values or from database uncertainty
        envcost_std_mean = pd.DataFrame.from_dict(envcost_std, orient='index', columns=['std'])
        envcost_std_mean['metadata'] = envcost_std_mean.index.map(self.pulpo_worker.lci_data['process_map_metadata'])
        if envcost_std_mean['std'].isna().any():
            raise Exception('There are NaNs in the standard deviation')
        envcost_std_mean['mean'] = envcost_std_mean.index.map(envcost_mean)
        envcost_std_mean['z'] = envcost_std_mean['std'] / envcost_std_mean['mean']
        if (envcost_std_mean['z'] > 0.5).any():
            print('These environmental costs have a standard deviation larger than 50% of their mean:\n')
            print(envcost_std_mean[envcost_std_mean['z'] > 0.5].sort_values('z', ascending=False))
            # raise Exception('There are z-values greater than 0.5 this is improbable')
        envcost_std_mean['z'].sort_values(ascending=False).iloc[5:].plot.box()
        print('The following points were excluded from the boxplot:')
        print(envcost_std_mean['z'].sort_values(ascending=False).iloc[:5])

    def update_problem(self, lambda_env_cost):
        """
        Update the Pyomo model’s ENV_COST_MATRIX for a given chance‐constraint level.

        1. Compute the normal‐distribution quantile (PPF) for the risk threshold λ.
        2. Scale each process’s cost standard deviation by this quantile.
        3. Store the updated values back into pulpo_worker.instance.ENV_COST_MATRIX.

        Args:
            lambda_env_cost (float): Confidence level (e.g. 0.95 for 95%).
        """
        super(CCFormulationObjL1, self).update_problem(lambda_env_cost)
        ppf_lambda_QB = scipy.stats.norm.ppf(lambda_env_cost)
        environmental_cost_updated = {(process_id, self.method): self.envcost_mean[process_id] + ppf_lambda_QB * self.envcost_std[process_id] for process_id in self.envcost_std.keys()}
        self.pulpo_worker.instance.ENV_COST_MATRIX.store_values(environmental_cost_updated, check=True)
class CCFormulationVarBounds(CCFormulationBase):
    """
    Implements an individual chance‐constraint formulation on the variable bounds
    using normal distributed upper and/or lower bound parameters. 
    The possible variable bounds to be chance constraint are:
        - UPPER_LIMIT: the upper bound on the scaling vector
        - LOWER_LIMIT: the lower bound on the scaling vector
        - UPPER_IMP_LIMIT: the upper bound on the environmental impacts
        - UPPER_INV_LIMIT: the upper bound on inventories emissions

    This subclass approximates all variable bounds specifief in the `unc_metadata` dictionary as Normal(μ,σ²),
    and then traces Pareto‐optimal solutions by varying the confidence level (λ).

    For all variable which are indexed in the dataframe to each of the specified bounds (above) (keys in `unc_metadata`),
    The bounds are chance constrained.
    """
    
    def update_problem(self, lambda_level):
        """
        Update the upper/lower bound parameters in the Pyomo model’s 
        for a given chance‐constraint level, based on the chance constraint mathematic formulation
        derived in the supplementary material. 

        1. Compute the normal‐distribution quantile (PPF) for the risk threshold λ.
        2. Scale each bounds standard deviation by this quantile.
        3. Store the updated values back into pulpo_worker.instance.{bound_name}.

        Args:
            lambda_level (float): Confidence level (e.g. 0.95 for 95%).
        """
        super(CCFormulationVarBounds, self).update_problem(lambda_level)
        ppf_lambda = scipy.stats.norm.ppf(lambda_level)
        for bound_name, metadata_df in self.normal_metadata.items():
            match bound_name:
                case 'UPPER_LIMIT':
                    bound_updated = (metadata_df['loc'] - ppf_lambda * metadata_df['scale']).to_dict()
                case 'LOWER_LIMIT':
                    bound_updated = (metadata_df['loc'] + ppf_lambda * metadata_df['scale']).to_dict()
                case 'UPPER_IMP_LIMIT': # ATTN: Not tested
                    bound_updated = (metadata_df['loc'] - ppf_lambda * metadata_df['scale']).to_dict()
                case 'UPPER_INV_LIMIT': # ATTN: Not tested
                    bound_updated = (metadata_df['loc'] - ppf_lambda * metadata_df['scale']).to_dict()
                case 'cf' | 'if':
                    continue # 'cf' and 'if' are keys in `normal_metadata` if the Objective is chance constrained, therefore they are skipped
                case _:
                    raise Exception('has not been implemented yet.')
            pyomo_bound = getattr(self.pulpo_worker.instance, bound_name)
            pyomo_bound.store_values(bound_updated, check=True)


class CCFormulationObjL1VarBound(CCFormulationObjL1, CCFormulationVarBounds):
    """
    Implements an individual chance‐constraint formulation on the objective using the L1 norm and 
    on the variable bounds with normally distributed uncertainties.

    This subclass approximates all uncertain intervention flows, characterization factors, and variable bounds as Normal(μ,σ²),
    computes the aggregated standard deviation of total environmental cost under an L1 norm, and then
    traces Pareto‐optimal solutions by varying the confidence level (λ).
    """

    def update_problem(self, lambda_level):
        """
        Update the Pyomo model’s ENV_COST_MATRIX, and variable bounds,
        e.g., UPPER_LIMIT, and UPPER_LIMIT 
        for a given chance‐constraint level (`lambda_level`).

        1. Compute the normal‐distribution quantile (PPF) for the risk threshold λ.
        2. Scale each process’s cost standard deviation in the metadata by this quantile.
        3. Store the updated values back into pulpo_worker.instance.ENV_COST_MATRIX.
        4. Scale each variable bound in the metadata by this quantile.
        5. Store the updated values back into pulpo_worker.instance.UPPER_LIMIT, LOWER_LIMIT.

        Args:
            lambda_level (float): Confidence level (e.g. 0.95 for 95%).
        """
        super(CCFormulationObjL1VarBound, self).update_problem(lambda_level)
        

class CCFormulationObjL2(CCFormulationBase):

    def formulate(self):
        """
        Needs to add pyomo formulations of the L2 norm with the standard deviation of the if and cf 
        """
        pass

    def update_problem(self):
        """
        Might need to change multiple things in the pyomo model
        """
        pass


class BaseParetoSolver:
    """
    Abstract base class for solvers that compute Pareto fronts using chance-constrained formulations.

    This class provides common methods for solving optimization problems under chance constraints
    with a given CCFormulationBase instance. Currently, a single lambda level is applied
    across all chance constraints.
    """
    def __init__(self, cc_formulation:CCFormulationBase):
        """
        Initialize the solver with a chance constraint formulation.

        Args:
            cc_formulation (CCFormulationBase): The formulation instance containing the
                problem definition, parameter settings, and solver configuration.
        """
        self.cc_formulation = cc_formulation

    def solve_single_pareto_point(self, lambda_level) -> dict:
        """
        Solve a single Pareto point for a specified lambda level.

        Args:
            lambda_level (float): The lambda (epsilon) level to impose in the chance constraints.

        Returns:
            dict: A dictionary containing the results extracted from the solver,
                including variable values, objective metrics, and metadata.
        """
        self.cc_formulation.update_problem(lambda_level)
        self.cc_formulation.pulpo_worker.solve()
        result_data = self.cc_formulation.pulpo_worker.extract_results()
        return result_data

    
    def compare_subsequent_paretosolutions(self, result_data_CC):
        """
        Compare impacts and decision choices across multiple Pareto solutions.

        Args:
            result_data_CC (dict of float to dict): Mapping from each lambda level
                to its corresponding solver result dictionary.
        """
        try:
            from IPython.display import display
        except ImportError:
            display = globals()['print']
        impacts = {}
        print(self.cc_formulation.method)
        for lambda_QB, result_data in result_data_CC.items():
            impacts[lambda_QB] = result_data['Impacts'].loc[self.cc_formulation.method,'Value']
            print('{}: {}'.format(lambda_QB, impacts[lambda_QB]))
        # The changs in the choices of the optimizer
        choices_results = {}
        for i_CC, (lambda_QB, result_data) in enumerate(result_data_CC.items()):
            for choice in self.cc_formulation.choices.keys():
                if i_CC == 0:
                    choices_results[choice] = result_data['Choices'][choice][['Capacity']]
                choices_results[choice] = choices_results[choice].join(result_data['Choices'][choice]['Value'].rename(lambda_QB), how='left')
        for choice, choice_result in choices_results.items():
            display(choice)
            display(choice_result)

        # Changes in the scaling vector and the characterized and scaled inventories
        lambda_array = list(result_data_CC.keys())
        for lambda_1, lambda_2 in zip(lambda_array[:len(lambda_array)-1], lambda_array[1:len(lambda_array)]):
            print(f'lambda_1: {lambda_1}\nlambda_2: {lambda_2}\n')
            scaling_vector_diff = ((result_data_CC[lambda_1]['Scaling Vector']['Value'] - result_data_CC[lambda_2]['Scaling Vector']['Value']))
            scaling_vector_ratio = (scaling_vector_diff / result_data_CC[lambda_1]['Scaling Vector']['Value']).abs().sort_values(ascending=False)
            environmental_cost_mean = {env_cost_index[0]: env_cost['Value'] for env_cost_index, env_cost in result_data_CC[lambda_1]['ENV_COST_MATRIX'].iterrows()}
            characterized_scaling_vector_diff = (scaling_vector_diff * pd.Series(environmental_cost_mean).reindex(scaling_vector_diff.index)).abs()
            characterized_scaling_vector_diff_relative = (characterized_scaling_vector_diff / result_data_CC[lambda_1]['Impacts'].loc[self.cc_formulation.method, 'Value']).abs().sort_values(ascending=False)

            print('Amount of process scaling variables that changed:\n{}: >1% \n{}: >10%\n{}: >100%\n{}: >1000%\n'.format((scaling_vector_ratio > 0.01).sum(), (scaling_vector_ratio > 0.1).sum(), (scaling_vector_ratio > 1).sum(), (scaling_vector_ratio > 10).sum()))
            print('Amount of process characterized scaling variables (impacts per process) that changed:\n{}: >1% \n{}: >10%\n{}: >100%\n{}: >1000%\n'.format((characterized_scaling_vector_diff_relative > 0.01).sum(), (characterized_scaling_vector_diff_relative > 0.1).sum(), (characterized_scaling_vector_diff_relative > 1).sum(), (characterized_scaling_vector_diff_relative > 10).sum()))
            print('{:.5e}: is the maximum impact change in one process\n{:.5e}: is the total impact change\n'.format(characterized_scaling_vector_diff_relative.max(), characterized_scaling_vector_diff_relative.sum()))

            amount_of_rows_for_visiualization = 10
            # print('The relative change of the scaling vector (s_lambda_1 - s_lambda_2)/s_lambda_1:\n')
            # display(scaling_vector_ratio.iloc[:amount_of_rows_for_visiualization].rename(result_data_CC[lambda_2]['Scaling Vector']['Metadata']).sort_values(ascending=False))
            # print('\n---\n')
            print('The relative change of the characterized scaling vector (s_lambda_1 - s_lambda_2)*QB_s / QBs:\n')
            display(characterized_scaling_vector_diff_relative.iloc[:amount_of_rows_for_visiualization].rename(result_data_CC[lambda_2]['Scaling Vector']['Metadata']))
            print('\n---\n')

    def create_data_for_plots(self, result_data_CC:dict, cutoff_value:float) -> pd.DataFrame:
        """
        Create the data for the Pareto front plots, by computing the process impacts and 
        selecting the top processes per Pareto Point based on the cut off and then concatting
        all main contributing processes across Pareto Points to show changes.

        Args:
            result_data_CC (dict of float to dict): Mapping from each lambda level
                to its corresponding solver result dictionary.
            cutoff_value (float): Relative threshold for filtering main decision variables
                to include in the bar plot.

        Return:
            data_QBs_main_df (pd.DataFrame): 
                Containing the main contributing processes to the impact per Pareto Point, 
                returned from `create_data_for_plots` method.
        """
        data_QBs_main_list = []
        # data_QBs_list = []
        for lamnda_QBs, result_data in result_data_CC.items():
            environmental_cost_mean = {env_cost_index[0]: env_cost['Value'] for env_cost_index, env_cost in result_data_CC[lamnda_QBs]['ENV_COST_MATRIX'].iterrows()}
            QBs = result_data['Scaling Vector']['Value'] * pd.Series(environmental_cost_mean).reindex(result_data['Scaling Vector']['Value'].index)
            # data_QBs_list.append(QBs)
            QBs_main = QBs[QBs.abs() > cutoff_value*QBs.abs().sum()]
            QBs_main.name = lamnda_QBs
            data_QBs_main_list.append(QBs_main)
            print('With a cutoff value of {}, we keep {} process to an error of {:.2%}'.format(cutoff_value, len(QBs_main), abs(1 - QBs_main.sum()/QBs.sum())))
        data_QBs_main_df = pd.concat(data_QBs_main_list, axis=1)
        # ATTN: Best case would be to fill NaN which appear when concating the "maion contributing processes datasets" with the data from the QBs, but currently we dont have all ENV_COST
        data_QBs_main_df = data_QBs_main_df.fillna(0.)
        # Rename the index to contain the main contributing processes.
        data_QBs_main_df = data_QBs_main_df.rename(index={process_id: self.cc_formulation.pulpo_worker.lci_data['process_map_metadata'][process_id] for process_id in data_QBs_main_df.index})
        return data_QBs_main_df

    def plot_pareto_front(self, result_data_CC:dict, cutoff_value:float, bbox_to_anchor:Tuple[float, float] = (0.65, -1.)):
        """
        Plot the Pareto front and highlight main contributing variables.

        Args:
            result_data_CC (dict of float to dict): Mapping from each lambda level
                to its corresponding solver result dictionary.
            cutoff_value (float): Relative threshold for filtering main decision variables
                to include in the bar plot.
            bbox_to_anchor (tuple): 
                Tuple holding the bbox anchor points for the legend.
                Default value is (0.65, -1.).
        """
        data_QBs_main_df = self.create_data_for_plots(result_data_CC, cutoff_value)
        plot_pareto_solution_normalized_bar_plots(data_QBs_main_df, self.cc_formulation.method, bbox_to_anchor=bbox_to_anchor)
        plot_pareto_solution_bar_plots(data_QBs_main_df, self.cc_formulation.method, bbox_to_anchor=bbox_to_anchor)

    
class EpsilonConstraintSolver(BaseParetoSolver):
    """
    Solver implementing the epsilon-constraint method for Pareto front approximation.

    This solver iterates over a sequence of epsilon levels (lambda values), updates
    the chance-constrained formulation for each level, and solves to generate points
    on the Pareto frontier.
    """
    def solve(self, lambda_epislons: array.array) -> dict: # ATTN create types the result data and add in dict here
        """
        Solve the optimization problem for each epsilon (lambda) level.

        Args:
            lambda_epislons (array.array): Sequence of epsilon levels to impose
                as constraints in separate optimization runs.

        Returns:
            result_data_CC (dict of float to dict): Mapping from each lambda level to the corresponding
                solution result dictionary extracted by `extract_results()`.
        """
        result_data_CC = {}
        for lambda_level in lambda_epislons:
            print(f'solving CC problem for lambda_QB = {lambda_level}')
            # ATTN: We need to store the environmental costs in the results_data and not extract seperate
            # This used to be the case before the results_data was changed, this formulation will run into errors.
            self.cc_formulation.update_problem(lambda_level)
            self.cc_formulation.pulpo_worker.solve()
            result_data_CC[lambda_level] = self.cc_formulation.pulpo_worker.extract_results(extractparams=True)
        return result_data_CC


class AdaptiveSamplingSolver(BaseParetoSolver):
    """
    Solver for adaptive sampling based Pareto approximation.
    """
    def solve(self, cc_formulation, **kwargs):
        pass

