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

    

    
    

    

    
class EpsilonConstraintSolver(BaseParetoSolver):
    """
    Solver implementing the epsilon-constraint method for Pareto front approximation.

    This solver iterates over a sequence of epsilon levels (lambda values), updates
    the chance-constrained formulation for each level, and solves to generate points
    on the Pareto frontier.
    """



class AdaptiveSamplingSolver(BaseParetoSolver):
    """
    Solver for adaptive sampling based Pareto approximation.
    """
    def solve(self, cc_formulation, **kwargs):
        pass

