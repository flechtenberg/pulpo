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
from typing import Union, List, Optional, Dict

# === Plots ===
def set_size(width, height, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.
 
    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy
 
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction    
 
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
 
    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2
 
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if height: #if height is specified
        fig_height_pt = height * fraction
        fig_height_in = fig_height_pt * inches_per_pt
    else:
        fig_height_in = fig_width_in * golden_ratio
 
    fig_dim = (fig_width_in, fig_height_in)
 
    return fig_dim

def plot_contribution_barplot(data:pd.DataFrame, metadata:pd.DataFrame, impact_category:str, colormap:pd.Series=pd.Series([]), bbox_to_anchor_lower:float = -0.6, bbox_to_anchor_center:float=0.5):
    """
        Barplot of the contributional variance of the parameters in an objective

        args:
            data:       series with impacts as values and processes as index
            metadata:   metadataframe with bar_names and same indices as data
            colormap:   Series with color codes to each data index     `colormap = pd.Series(mpl.cm.tab20.colors[:data.shape[0]], index=data.index)`
            bbox_to_anchor_lower: negative float, scaled how much the legend is under the plot
    """
    # width = 180
    # height = 180
    width = 4.77*72.4#600
    height = None
    _, ax = plt.subplots(1, 1, figsize=set_size(width,height))

    # Data
    data = data.sort_values(ascending=False)
    heights = data.values * 100
    bars = [textwrap.fill(string, 50) for string in metadata.reindex(data.index)]
    y_pos = range(len(bars))
    
    for height, y_po, indx in zip(heights, y_pos, data.index):
        ax.bar(y_po, height, capsize=5, ecolor="gray", color=colormap[indx], alpha=0.9)
    ax.set_xticks([])
    if (data<=1).all() and (data>=0).all():
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
        # For the minor ticks, use no labels; default NullFormatter.
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
    ax.legend(bars, loc='lower center', bbox_to_anchor=(bbox_to_anchor_center, bbox_to_anchor_lower), borderpad=1)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dotted')
    ax.set_xlabel("Main environmental parameters")
    ax.set_ylabel("Contribution to total {} in [\%]".format(impact_category))
    return ax

def plot_contribution_barplot_with_err(data:pd.DataFrame, metadata:pd.DataFrame, colormap:pd.Series=pd.Series([]), bbox_to_anchor_lower:float = -0.6, bbox_to_anchor_center:float=0.5):
    """
        Barplot of the contributional variance of the parameters in an objective

        args:
            data:       dataframe with columns: "ST" and "ST_conf"
            metadata:   metadataframe with "bar_names" column and same indices as data
            colormap:   Series with color codes to each data index     `colormap = pd.Series(mpl.cm.tab20.colors[:data.shape[0]], index=data.index)`
            bbox_to_anchor_lower: negative float, scaled how much the legend is under the plot
    """
    # width = 180
    # height = 180
    width = 4.77*72.4#600
    height = None
    _, ax = plt.subplots(1, 1, figsize=set_size(width,height))

    # Data
    data = data.sort_values(["ST"], ascending=False)
    heights = data["ST"].values * 100
    yerrs = data["ST_conf"].values * 100
    bars = [textwrap.fill(string, 50) for string in metadata["bar_names"].reindex(data.index)]
    y_pos = range(len(bars))
    
    for height, y_po, yerr, indx in zip(heights, y_pos, yerrs, data.index):
        ax.bar(y_po, height, yerr=yerr, capsize=5, ecolor="gray", color=colormap[indx], alpha=0.9)
    ax.set_xticks([])
    if (data["ST"]<=1).all() and (data["ST"]>=0).all():
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
        # For the minor ticks, use no labels; default NullFormatter.
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
    ax.legend(bars, loc='lower center', bbox_to_anchor=(bbox_to_anchor_center, bbox_to_anchor_lower), borderpad=1)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dotted')
    return ax

def plot_linked_contribution_barplot(data:pd.DataFrame,  metadata:pd.DataFrame, impact_category:str, colormap_base:tuple, colormap_linked:pd.Series=pd.Series([]), savefig:Optional[bool]=None, bbox_to_anchor_center:float=0.7, bbox_to_anchor_lower:float=0.7):
    """
        Barplot of the contributional variance of the parameters in total cost objective

        args:
            data:       dataframe with columns: "ST" and "ST_conf"
            metadata:   metadataframe with "bar_names" column and same indices as data
            impact_category:    name of environmental impact category
            colormap_base:      The colormap which should be used for the plot, use the same as underlying the colormap_linked if it is specified
            colormap_linked:    If there is a colormap from another plot where the variables shown in this plot should refer to if they appear in both
            savefig:    if true saves fig into specified path
    """
    if colormap_linked.empty:
        colormap = pd.Series(colormap_base[:data.shape[0]], index=data.index)
    else:
        # act_indcs = [index for index in colormap.index if type(index[1]) == int]
        # colormap_red = pd.Series(colormap[act_indcs].values, index=[indcs[1] for indcs in act_indcs])
        colormap_red = colormap_linked.loc[colormap_linked.index.isin(data.index)]
        addtional_incs = data.index[~data.index.isin(colormap_red.index)]
        additional_colormap = pd.Series(
            colormap_base[colormap_linked.shape[0]:colormap_linked.shape[0]+len(addtional_incs)], 
            index = addtional_incs
            )
        colormap = pd.concat([colormap_red, additional_colormap])


    ax = plot_contribution_barplot_with_err(data, metadata, colormap=colormap, bbox_to_anchor_lower=bbox_to_anchor_lower, bbox_to_anchor_center=bbox_to_anchor_center)    
    ax.set_xlabel("Main environmental parameters")
    ax.set_ylabel("Contribution to total {} in [\%]".format(impact_category))

    # Save figure
    if savefig:
        raise Exception('not implemented yet')
        # plt.savefig(r"C:\Users\admin\OneDrive - Carbon Minds GmbH\Dokumente\13 Students\MA_Bartolomeus_Löwgren\02_code\03_optimization_framework\04_case_studies\02_plots\total_env_impact_barplot" + ".{}".format(fileformat), format=fileformat, bbox_inches='tight')

def plot_CC_pareto_solution_bar_plots(data:pd.DataFrame, y_label:str, bbox_to_anchor:tuple=(1.40, .05)):
    """
        args:
            data:       columns: lambdas, rows: QBs grouped by something
    """
    
    # set figure and plot
    width = 6.
    height = 6.
    _, ax = plt.subplots(1, 1, figsize=(width,height))

    data_cleaned = data.copy()
    data_cleaned_scaled = data_cleaned.abs().divide(data_cleaned.abs().sum())
    data_cumsum = data_cleaned_scaled.cumsum(axis=0)
    width = .8
    labels = ["{:.3f}".format(label) for label in data.columns.astype(float).values]
    bottom_data = np.zeros(len(labels))
    for i_row, (type, row_data) in enumerate(data_cumsum.iterrows()):
        ax.bar(labels, row_data.values-bottom_data, width, bottom=bottom_data, label=type, color=mpl.cm.tab20.colors[i_row])
        bottom_data = row_data.values
    ax.axhline(y=0, color='k')
    ax.set_xlabel("probability level ($\lambda$)")
    ax.set_ylabel("{} in [\%]".format(y_label))
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    ax2 = ax.twinx()
    ax2.plot(labels, data.sum().values/1e9, "kx-", label="total GWP", linewidth=1)
    ax2.set_ylabel(y_label)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='lower center', bbox_to_anchor=bbox_to_anchor, borderpad=1, facecolor="None")
    ax.set_facecolor("None")




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


# === Parameter Filter ===
class ParameterFilter:
    """
    Filter out parameters whose uncertainty contributions
    to LCIA are negligible.

    Uses the  objective contribution to rank parameters.
    """
    def __init__(self, result_data:dict, lci_data: dict, choices: dict, demand:dict, method:str):
        """
        Store optimization results and LCI data for filtering.

        Args:
            result_data: Solver output dict.
            lci_data: PULPO LCI data (matrices, maps).
            choices: Choice dict from case study.
            demand: Demand dict from case study.
            method: LCIA method used.
        """
        self.result_data = result_data
        self.lci_data = lci_data # From pulpo_worker.lci_data
        self.choices = choices # From CaseStudy
        self.demand = demand # From CaseStduy
        self.method = method # From the result_data


    def prepare_sampling(self,  scaling_vector_strategy:str='naive') -> pd.Series:
        """
        Build the combined parameter matrix (B and Q) and scaling vector.
        We only consider uncertainty in the $B$ and $Q$ parameter matrizes. The scaling vector is given by the optimal solution.
        We will look at the contribution of the parameters to the environmental impact objective: 
        Q·B·s

        Args:
            scaling_vector_strategy: How to compute scaling vector: 'naive' or 'constructed_demand'.

        Returns:
            scaling_vector_series: Series of scaling factors (optimal s).
        """
        # Define the scaling vector for the subsequent analysis as the optimization results
        # put the scaling vector returned from the optimization into the same order as the process map
        match scaling_vector_strategy:
            case 'naive':
                scaling_vector_series = self.result_data['Scaling Vector']['Value'].sort_index()
            case 'constructed_demand':
                scaling_vector_series = self.construct_scaling_vector_from_choices()
            case _:
                raise Exception('Case not implemented.')
        return scaling_vector_series
        
    def construct_scaling_vector_from_choices(self) -> pd.Series:
        """
        Construct a scaling vector from the model's choice decisions.

        Returns:
            pd.Series: Scaling factors for each parameter, indexed by parameter name,
                       derived from the selected capacities.
        """
        # Define the scaling vector for the subsequent analysis as a constructed demand vector
        # Create a demand vector which includes all alternatives in the demand use use the corresponding scaling vector of that LCA for the subsequent GSA and preparation steps, the idea is to include all relevant processes in the LCIA calculation instead of just those chosen by the optimizer at on Pareto point
        demand = {}
        for product, alternatives in self.choices.items():
            # demand_amount = self.result_data['Choices'][product]["Value"].sum()
            for alternative in alternatives:
                # self.demand[alternative] = demand_amount
                demand[alternative] = 1
        # Compute the scaling vector for the set constructed demand
        method_tuple = ast.literal_eval(self.method)
        lca = bw2calc.LCA(demand, method_tuple)
        lca.lci()
        # Map the scaling vector results of the LCI calculation back to the optimization results index structure
        index_mapper_df = pd.concat(
            [
                pd.DataFrame.from_dict(self.lci_data['process_map'], orient='index', columns=['opt_problem']),
                pd.DataFrame.from_dict(lca.product_dict, orient='index', columns=['lca'])
            ],
            axis=1
        ).set_index('opt_problem')
        reindex_supply_array_df = index_mapper_df.merge(pd.DataFrame(lca.supply_array,  columns=['supply_array']), how='left', left_on='lca', right_index=True)
        scaling_vector_series = reindex_supply_array_df['supply_array']
        return scaling_vector_series



    def compute_LCI_LCIA(
            self, 
            scaling_vector_series:pd.Series, 
            ) -> tuple[float, scipy.sparse.sparray]:
        """
        Compute per-parameter LCI and LCIA contributions.

        Args:
            characterization_matrix: Characterization Q matrix, since there are multiple available in the .
            scaling_vector_series: Series of parameter scaling factors (s).

        Returns:
            lca_score (float): the summed lcia score for the specific scaling vector.
            characterized_inventory (scipy.sparse.sparray): B·(Q·s) for each parameter (impact after characterization).
        """
        characterization_matrix = self.lci_data["matrices"][self.method]
        print('chosen environmental impact method: {}'.format(self.method))
        # LCI calculatiom
        count = len(self.lci_data["process_map"])
        inventory = self.lci_data['intervention_matrix'] * \
        sparse.spdiags([scaling_vector_series.values], [0], count, count)
        # LCIA calculation
        characterized_inventory = \
        characterization_matrix * inventory
        lca_score = characterized_inventory.sum()
        print('The total impact is: {:e}'.format(lca_score))
        return lca_score, characterized_inventory

    def filter_inventoryflows(
            self, 
            characterized_inventory:scipy.sparse.sparray, 
            lca_score:float, 
            cutoff:float
            ) -> list:
        """
        Select inventory-flow parameters whose LCIA contributions exceed a threshold.

        Args:
            characterized_inventory (scipy.sparse.sparray): B·(Q·s) for each parameter (impact after characterization).
            lca_score (float): the summed lcia score for the specific scaling vector.
            cutoff (float): cutoff factor to compute minimum contribution value to retain a parameter.

        Returns:
            characterized_inventory_indices (list): Subset of inventory flows indices returned from filtering.
        """
        #ATTN: Add a simple optimization loop to find the cutoff which results in an absolute change of around 1%
        # Filters the inventory flows
        start = time()
        print('Characterized inventory:', characterized_inventory.shape, characterized_inventory.nnz)
        finv = characterized_inventory.multiply(abs(characterized_inventory) > abs(lca_score*cutoff))
        print('Filtered characterized inventory:', finv.shape, finv.nnz)
        characterized_inventory_indices = list(zip(*finv.nonzero()))
        # Since if negative and positive characterized inventories are cut away the explained fraction (finv.sum() / lca_score) can also be greater than 1
        deviation_from_lca_score = abs(1 - finv.sum() / lca_score)
        print('Deviation from LCA score:', deviation_from_lca_score)
        print('inventory {} filtering resulted in {} of {} exchanges ({}% of total impact) and took {} seconds.'.format(
            characterized_inventory.shape,
            finv.nnz,
            characterized_inventory.nnz,
            np.round(100-deviation_from_lca_score * 100, 2),
            np.round(time() - start, 3),
        ))
        return characterized_inventory_indices


    def filter_characterization_factors(self, characterized_inventory_indices:list) -> list:
        """
        Select characterization-factor parameters which characterize inventory flows returned from filtering.

        Args:
            characterized_inventory_indices (list): list of intervention flows returned from the filtering process.

        Returns:
            reduced_characterization_matrix_ids (list): Subset of characterization factors indices returned from filtering.
        """
        # Filter characterization matrix
        characterization_matrix = self.lci_data["matrices"][self.method]
        characterization_matrix_ids = characterization_matrix.diagonal().nonzero()[0]
        reduced_characterization_matrix_ids = []
        for (bio_i, ex_i) in characterized_inventory_indices:
            if bio_i in characterization_matrix_ids and bio_i not in reduced_characterization_matrix_ids:
                reduced_characterization_matrix_ids.append(bio_i)
        print('CHARACTERIZATION MATRIX {} filtering resulted in {} of {} characterization factors'.format(
            characterization_matrix.diagonal().shape,
            len(reduced_characterization_matrix_ids),
            len(characterization_matrix_ids)
        ))
        return reduced_characterization_matrix_ids

    def plot_top_processes(self, characterized_inventory:scipy.sparse.sparray, top_amount:int=10):
        """
        Plot the top-N contributing processes or parameters as a bar chart.

        Args:
            characterized_inventory (scipy.sparse.sparray): B·(Q·s) for each parameter (impact after characterization).
            top_amount (int): Number of top items to display (default: 10).

        Returns:
            None: Displays a matplotlib bar plot of the highest contributors.
        """
        # Plot the highest contributing processes
        impact_df = pd.DataFrame(
            characterized_inventory.sum(axis=0).T,
            index=list(range(characterized_inventory.shape[1])),
            columns=['impact']
        )
        impact_df['process name'] = impact_df.index.map(self.lci_data['process_map_metadata'])
        impact_df = impact_df.reindex(impact_df['impact'].abs().sort_values(ascending=False).index)
        impact_df_red = impact_df.iloc[:top_amount,:]
        impact_rest = impact_df.iloc[top_amount:,:].sum()
        impact_rest['process name'] = 'Rest'
        impact_df_red = pd.concat([impact_df_red, impact_rest.to_frame().T], axis=0)        
        impact_df_red['impact'] = impact_df_red['impact'] / impact_df['impact'].sum()
        colormap = pd.Series(mpl.cm.tab20.colors[:impact_df_red.shape[0]], index=impact_df_red.index)
        plot_contribution_barplot(impact_df_red['impact'], metadata=impact_df_red['process name'], impact_category=self.method, colormap=colormap,  bbox_to_anchor_center=1.7, bbox_to_anchor_lower=-.6)
        plt.show()


# === Uncertainty Module ===
class UncertaintyImporter:
    """
    Extract/Import and index uncertainty metadata for intervention flows and characterization factors.
    """
    def __init__(self, lci_data):
        self.lci_data = lci_data # from pulpo_worker.lci_data

    def get_intervention_indcs_to_db(self, db_name, intervention_indices:List[tuple]) -> List[tuple]:
        """
        fetches the inventory indices to a specified bw database 
        and if specified the intersection to a given list of interventions flow indices

        Args:
            db_name (str): name of the BW database for which the indices are fetched for
            intervention_indices (list) - optional: intervention flow indices for which the metadata will be extracted
        
        Returns:
            db_indcs (list): The interventions flow indices to the specified BW database and intersection to intervention_indices
        """
        db_indcs = [process_indx for (db, _), process_indx in self.lci_data['process_map'].items() if db == db_name]
        intervention_indices_in_db = [(intevention_indx, process_index) for (intevention_indx, process_index) in intervention_indices if process_index in db_indcs]
        return intervention_indices_in_db


    def get_intervention_meta(self, inventory_indices:List[tuple]) -> pd.DataFrame:
        """
        Extract intervention‐flow uncertainty metadata for given indices.

        Args:
            inventory_indices (List[tuple]): list of intervention flow indices (process_indx, intervention_indx) for which the metadata will be extracted 

        Returns:
            intervention_metadata_df (pd.DataFrame): DataFrame indexed by (row, col) with uncertainty info.
        """
        intervention_metadata_df = pd.DataFrame(self.lci_data['intervention_params'])
        intervention_metadata_df = intervention_metadata_df.set_index(["row", "col"])
        intervention_metadata_df = intervention_metadata_df.loc[inventory_indices]
        return intervention_metadata_df

    def get_cf_meta(self, method:str, characterization_indices:list) -> pd.DataFrame:
        """
        Extract uncertainty metadata for characterization factors for a method.

        Args:
            method: LCIA method key.
            characterization_indices: List of characterization row indices.
        """
        characterization_params = self.lci_data["characterization_params"][method]
        characterization_metadata_df = pd.DataFrame(characterization_params).set_index('row')
        characterization_metadata_df = characterization_metadata_df.loc[characterization_indices]
        return characterization_metadata_df

    def separate(self, uncertainty_metadata_df:pd.DataFrame) -> tuple[dict, list]:
        """
        Split metadata into defined (type>0) and undefined (type=0) entries.

        Returns:
            defined: dict index→metadata row
            undefined: list of indices without defined distributions
        """
        defined, undefined = {}, []
        for idx, row in uncertainty_metadata_df.iterrows():
            if row['uncertainty_type'] > 0:
                defined[idx] = row.to_dict()
            else:
                undefined.append(idx)
        print("Parameters with uncertainty information: {} \nParameters without uncertainty information: {}".format(len(defined), len(undefined)))
        return defined, undefined


class UncertaintyStrategyBase:
    """
    Base class for uncertainty distribution strategies.

    This abstract base defines the interface and common functionality for assigning probability
    distributions to parameters lacking predefined uncertainty metadata. Subclasses must implement
    methods to derive distribution parameters for these unspecified uncertainties.

    Attributes:
        metadata_df (pd.DataFrame): DataFrame containing parameter metadata.
        defined_uncertainty_metadata (dict): Mapping of parameter indices to their defined uncertainty metadata.
        undefined_uncertainty_indices (list): List of indices needing distribution assignment.
    """
    def __init__(self, metadata_df:pd.DataFrame, defined_uncertainty_metadata:dict, undefined_uncertainty_indices:list, *args, **kwargs):
        """
        Initialize the UncertaintyStrategyBase with metadata and index lists.

        Args:
            metadata_df (pd.DataFrame): The full metadata DataFrame for parameters.
            defined_uncertainty_metadata (dict): Dictionary mapping indices to existing uncertainty metadata.
            undefined_uncertainty_indices (list): List of parameter indices without defined uncertainties.
            *args: additional arguments passed to the assign method
            **kwargs: additional optional arguments passed to the assign method
        """
        self.metadata_df = metadata_df # ATTN: rename to param_metadata_df
        self.defined_uncertainty_metadata = defined_uncertainty_metadata
        self.undefined_uncertainty_indices = undefined_uncertainty_indices
        self.metadata_assigned_df = self.assign(*args, **kwargs)

    def add_random_noise_to_scaling_factor(self, scaling_factor:Union[float, list], low:float, high:float) -> list:
        """
        Adds random noise from uniform distribution to scaling factors, to avoid unrealistic structure in data.
        Multiplies the scaling vector with 1-low and 1+high to generate noisy scaling vectors given by the interval [1-low, 1+high].

        Args:
            scaling_factor (float, array, list): the scaling factor which will get noise added to it
            low (float): the lower bound (1-low) for which the noise will be sampled and multiplied to the scaling_factor.
            high (float): the lower bound (1+high) for which the noise will be sampled and multiplied to the scaling_factor.

        Returns:
            scaling_factor_randomized (list): The scaling factor, now a list superposed with the random noise

        """
        if isinstance(scaling_factor, float):
            scaling_factor = [scaling_factor] * len(self.undefined_uncertainty_indices)
        rng = np.random.default_rng(seed=161)
        random_noise = rng.uniform(1-low, 1+high, len(self.undefined_uncertainty_indices))
        scaling_factor_randomized = random_noise * np.array(scaling_factor)
        return scaling_factor_randomized.tolist()

    def assign(self, *args, **kwargs) -> pd.DataFrame:
        """
        Assign distribution parameters to parameters without predefined uncertainty.

        Args:
            <<Depending on the strategy>>

        Returns:
            pd.DataFrame: Updated metadata DataFrame for targeted parameters.
        """
        return pd.DataFrame([])

class UniformBaseStrategy(UncertaintyStrategyBase):
    """
    Strategy that assigns uniform distributions to parameters with undefined uncertainty information.

    For each parameter index in undefined_uncertainty_indices, the staretgy sets min and and max based on
    configurable scaling factors. 
    """
    def __init__(
            self, 
            metadata_df, 
            defined_uncertainty_metadata, 
            undefined_uncertainty_indices, 
            upper_scaling_factor, 
            lower_scaling_factor, 
            noise_interval:Dict[str,float]={'min':0., 'max':0.}
            ) -> None:
        """
        Initialize the UniformBaseStrategy with metadata and index lists and scaling factors.

        Args:
            metadata_df (pd.DataFrame): The full metadata DataFrame for parameters.
            defined_uncertainty_metadata (dict): Dictionary mapping indices to existing uncertainty metadata.
            undefined_uncertainty_indices (list): List of parameter indices without defined uncertainties.
            upper_scaling_factor (float): the scaling factor multiplied with the amount 
                to get the maximum value for the uniform distribution
            lower_scaling_factor (float): the scaling factor multiplied with the amount 
                to get the minimum value for the uniform distribution
            noise_interval (Dict[str,float]): Dict containing "min" and "max" keywords 
                holding the upper and lower bound of the noise generated with a uniform distribution 
                and multiplied with the scaling factor vector as (1-min) and (1+max)
        """
        super().__init__(
            metadata_df, 
            defined_uncertainty_metadata, 
            undefined_uncertainty_indices, 
            upper_scaling_factor, 
            lower_scaling_factor, 
            noise_interval=noise_interval
            )

    def _compute_uniform_dist_params(
            self,
            upper_scaling_factor:float, 
            lower_scaling_factor:float,
            noise_interval:Dict[str,float]={'min':0., 'max':0.}
            ) -> pd.DataFrame:
        """
        Compute uniform distribution parameters to parameters without predefined uncertainty.

        Args:
            upper_scaling_factor (float): Scaling factor to determine the upper bound relative to the median.
            lower_scaling_factor (float): Scaling factor to determine the lower bound relative to the median.

        Returns:
            pd.DataFrame: Updated metadata DataFrame including 'minimum', 'maximum',
                          and 'uncertainty_type' set to 4 (uniform) for targeted parameters.
        """
        metadata_df = self.metadata_df.copy()
        # Create a scaling factor array if scaling_factors are floats and randomize it if the noise interval has min and max greater 0.
        upper_scaling_factor_randomized = self.add_random_noise_to_scaling_factor(upper_scaling_factor, noise_interval['min'], noise_interval['max'])
        lower_scaling_factor_randomized = self.add_random_noise_to_scaling_factor(lower_scaling_factor, noise_interval['min'], noise_interval['max'])
        # For each undefined parameter, set loc=median, bounds = ±factor·|median|
        for undefined_indx, upper_scaling_factor, lower_scaling_factor in zip(self.undefined_uncertainty_indices, upper_scaling_factor_randomized, lower_scaling_factor_randomized):
            amount = metadata_df.loc[undefined_indx].amount
            metadata_df.loc[undefined_indx, 'loc'] = np.NaN
            if amount > 0:
                metadata_df.loc[undefined_indx, 'maximum'] = amount + upper_scaling_factor * abs(amount)
                metadata_df.loc[undefined_indx, 'minimum'] = amount - lower_scaling_factor * abs(amount)
            elif amount < 0:
                metadata_df.loc[undefined_indx, 'maximum'] = amount + lower_scaling_factor * abs(amount)
                metadata_df.loc[undefined_indx, 'minimum'] = amount - upper_scaling_factor * abs(amount)
            metadata_df.loc[undefined_indx, 'uncertainty_type'] = 4,
        # Check for negative‐median cases and adjust skew mapping
        if ((metadata_df.loc[self.undefined_uncertainty_indices,'maximum'] - metadata_df.loc[self.undefined_uncertainty_indices,'minimum']) <= 0).any():
            raise Exception('There is a parameter with where the asigned minimum value is equal or larger than the asigned maximum value')
        return metadata_df
    
    def assign(self, *args, **kwargs):
        metadata_asigned_df = self._compute_uniform_dist_params(*args, **kwargs)
        return metadata_asigned_df

class TriangluarBaseStrategy(UncertaintyStrategyBase):
    """
    Strategy that assigns triangular distributions to parameters with undefined uncertainty information.

    For each parameter index in undefined_uncertainty_indices, this strategy sets the median
    (loc) from the 'amount' field of metadata_df and defines min and and max based on
    configurable scaling factors.

    The min is computed as loc - lower_scaling_factor * abs(loc), and the max
    as loc + upper_scaling_factor * abs(loc).

    Methods:
        _compute_triag_dist_params: Computes scaling factors (upper and lower) based on given scaling_factors.
        assign: Applies computed scaling factors to assign 'loc', 'minimum', 'maximum', and 'uncertainty_type'.

    Attributes:
        Inherits metadata_df, defined_uncertainty_metadata, and undefined_uncertainty_indices from base class.
    """
    def __init__(
            self, 
            metadata_df, 
            defined_uncertainty_metadata, 
            undefined_uncertainty_indices, 
            upper_scaling_factor:float, 
            lower_scaling_factor:float, 
            noise_interval:Dict[str,float]={'min':0., 'max':0.}
            ) -> None:
        """
        Initialize the TriangluarBaseStrategy with metadata and index lists and scaling factors.

        Args:
            metadata_df (pd.DataFrame): The full metadata DataFrame for parameters.
            defined_uncertainty_metadata (dict): Dictionary mapping indices to existing uncertainty metadata.
            undefined_uncertainty_indices (list): List of parameter indices without defined uncertainties.
            upper_scaling_factor (float): the scaling factor multiplied with the amount 
                to get the maximum value for the triangular distribution
            lower_scaling_factor (float): the scaling factor multiplied with the amount 
                to get the minimum value for the triangular distribution
            noise_interval (Dict[str,float]): Dict containing "min" and "max" keywords 
                holding the upper and lower bound of the noise generated with a uniform distribution 
                and multiplied with the scaling factor vector as (1-min) and (1+max)
        """
        super().__init__(
            metadata_df, 
            defined_uncertainty_metadata, 
            undefined_uncertainty_indices, 
            upper_scaling_factor, 
            lower_scaling_factor, 
            noise_interval=noise_interval
            )


    def _compute_triag_dist_params(
            self,
            upper_scaling_factor:float, 
            lower_scaling_factor:float,
            noise_interval:Dict[str,float]={'min':0., 'max':0.}
            ) -> pd.DataFrame:
        """
        Compute triangular distribution parameters to parameters without predefined uncertainty.

        Args:
            upper_scaling_factor (float): Scaling factor to determine the upper bound relative to the median.
            lower_scaling_factor (float): Scaling factor to determine the lower bound relative to the median.
            noise_interval (Dict[str,float]): Dict containing "min" and "max" keywords 
                holding the upper and lower bound of the noise generated with a uniform distribution 
                and multiplied with the scaling factor vector as (1-min) and (1+max)

        Returns:
            pd.DataFrame: Updated metadata DataFrame including 'loc', 'minimum', 'maximum',
                          and 'uncertainty_type' set to 5 (triangular) for targeted parameters.
        """
        metadata_df = self.metadata_df.copy()
        # Create a scaling factor array if scaling_factors are floats and randomize it if the noise interval has min and max greater 0.
        upper_scaling_factor_randomized = self.add_random_noise_to_scaling_factor(upper_scaling_factor, noise_interval['min'], noise_interval['max'])
        lower_scaling_factor_randomized = self.add_random_noise_to_scaling_factor(lower_scaling_factor, noise_interval['min'], noise_interval['max'])
        # For each undefined parameter, set loc=median, bounds = ±factor·|median|
        for undefined_indx, upper_scaling_fac, lower_scaling_fac in zip(self.undefined_uncertainty_indices, upper_scaling_factor_randomized, lower_scaling_factor_randomized):
            amount = metadata_df.loc[undefined_indx].amount
            metadata_df.loc[undefined_indx, 'loc'] = amount
            if amount > 0:
                metadata_df.loc[undefined_indx, 'maximum'] = amount + upper_scaling_fac * abs(amount)
                metadata_df.loc[undefined_indx, 'minimum'] = amount - lower_scaling_fac * abs(amount)
            elif amount < 0:
                metadata_df.loc[undefined_indx, 'maximum'] = amount + lower_scaling_fac * abs(amount)
                metadata_df.loc[undefined_indx, 'minimum'] = amount - upper_scaling_fac * abs(amount)
            metadata_df.loc[undefined_indx, 'uncertainty_type'] = 5,
        # Check for negative‐median cases and adjust skew mapping
        if ((metadata_df.loc[self.undefined_uncertainty_indices,'maximum'] - metadata_df.loc[self.undefined_uncertainty_indices,'minimum']) <= 0).any():
            raise Exception('There is a parameter with where the asigned minimum value is equal or larger than the asigned maximum value')
        # There can be negative flows and their upper and lower bounds need to be considered in detail!
        print('uncertain parameters with negative median value:')
        print(metadata_df.loc[self.undefined_uncertainty_indices].loc[metadata_df.loc[self.undefined_uncertainty_indices,'loc'] < 0])
        return metadata_df
    
    def assign(self, *args, **kwargs):
        metadata_asigned_df = self._compute_triag_dist_params(*args, **kwargs)
        return metadata_asigned_df
    
class TriangularBoundInterpolationStrategy(TriangluarBaseStrategy):
    """
    Strategy that assigns triangular distributions to parameters with undefined uncertainty information.

    For each parameter index in undefined_uncertainty_indices, this strategy sets the median
    (loc) from the 'amount' field of metadata_df and defines min and max based on
    configurable scaling factors derived from existing uncertainty metadata statistics,
    using the bounds information.

    The min is computed as loc - lower_scaling_factor * abs(loc), and the max
    as loc + upper_scaling_factor * abs(loc).

    Methods:
        _get_bounds: Computes the bounds of the parameters with defined uncertainty information
        _compute_bounds_statistics: Computes scaling factors (upper and lower) based on statistical analysis of defined uncertainties.
        assign: Applies computed scaling factors to assign 'loc', 'minimum', 'maximum', and 'uncertainty_type'.

    Attributes:
        Inherits metadata_df, defined_uncertainty_metadata, and undefined_uncertainty_indices from base class.
    """
    def __init__(self, metadata_df, defined_uncertainty_metadata, undefined_uncertainty_indices, noise_interval:Dict[str,float]={'min':0., 'max':0.}):
        """
        Initialize the TriangularBoundInterpolationStrategy with metadata and index lists.

        Args:
            metadata_df (pd.DataFrame): The full metadata DataFrame for parameters.
            defined_uncertainty_metadata (dict): Dictionary mapping indices to existing uncertainty metadata.
            undefined_uncertainty_indices (list): List of parameter indices without defined uncertainties.
            noise_interval (Dict[str,float]): Dict containing "min" and "max" keywords 
                holding the upper and lower bound of the noise generated with a uniform distribution 
                and multiplied with the scaling factor vector as (1-min) and (1+max)
        """
        UncertaintyStrategyBase.__init__(self, metadata_df, defined_uncertainty_metadata, undefined_uncertainty_indices, noise_interval=noise_interval)

    def _get_bounds(self):
        """
        Compute min/max bounds for all parameters via UncertaintyProcessor.
        Raises if no metadata defined.
        """
        if not self.defined_uncertainty_metadata:
            raise Exception('There are no uncertain parameters with defined uncertainty, as needed to interpolate the bouds.')
        self.uncertainty_bounds = UncertaintyProcessor.compute_bounds(self.defined_uncertainty_metadata)
            

    def _compute_bounds_statistics(self) -> tuple[float, float]:
        """
        Computes the scaling factors from the the bounds of the uncertain parameters with known distribution
        Assumes that the bounds of the median of 95% confidence interval can be used to compute scaling factors.
        
        Returns:
            upper_scaling_factor: upper scaling factor to be multiplied with a central moment to get the max value for a distribution, e.g., triangular or uniform
            lower_scaling_factor: lower scaling factor to be multiplied with a central moment to get the min value for a distribution, e.g., triangular or uniform
        """
        self._get_bounds()
        if len(self.uncertainty_bounds) < 3:
            raise Exception('There are only three uncertain parameters with uncertainty bounds, not enough to compute bounds statistics for interpolation')
        lower_spread = (self.uncertainty_bounds['amount'] - self.uncertainty_bounds['lower']).abs() / self.uncertainty_bounds['amount'].abs()
        upper_spread = (self.uncertainty_bounds['amount'] - self.uncertainty_bounds['upper']).abs() / self.uncertainty_bounds['amount'].abs()
        ax = lower_spread.hist(bins=30, label='lower spread')
        upper_spread.hist(bins=30, label='upper spread', ax=ax, alpha=0.5)
        ax.legend()
        print('upper spread statistics')
        print('mean: {:.4f}\nmode: {}\nmedian: {:.4f}\nstd: {:.4f}\nmin: {:.4f}\nmax: {:.4f}\n'.format(upper_spread.mean(), upper_spread.mode(), upper_spread.median(), upper_spread.std(), upper_spread.min(), upper_spread.max()))
        print('\nlower spread statistics')
        print('mean: {:.4f}\nmode: {}\nmedian: {:.4f}\nstd: {:.4f}\nmin: {:.4f}\nmax: {:.4f}\n'.format(lower_spread.mean(), lower_spread.mode(), lower_spread.median(), lower_spread.std(), lower_spread.min(), lower_spread.max()))
        # based on the statistics below, I decided to use the median of the amount fraction of the upper and lower bound, since the distribution of the spreads contains multiple modes and many "outliers" which will distort the mean greatly.
        # **ATTN:**
        # There are multiple modes in the spread statistics, which means there seems to be a few 'groups' or 'types' of intervention flows which have very different spreads, it might be good to analyze which these are to make the extrapolation more accurate.
        upper_scaling_factor = upper_spread.median()
        lower_scaling_factor = lower_spread.median()
        print('The upper spread scaling factor for intervention flows is: {}\nThe lower spread scaling factor for intervention flows is: {}'.format(upper_scaling_factor, lower_scaling_factor)) 
        return upper_scaling_factor, lower_scaling_factor
    
    def assign(self, **kwargs) -> pd.DataFrame:
        """
        Assign triangular distribution parameters derived averaged bounds, to parameters without predefined uncertainty.
        """
        upper_scaling_factor, lower_scaling_factor = self._compute_bounds_statistics()
        metadata_asigned_df = self._compute_triag_dist_params(upper_scaling_factor, lower_scaling_factor, **kwargs)
        return metadata_asigned_df    


class UncertaintyProcessor:
    """
    Processes uncertainty metadata by fitting non-normal distributions to normal approximations
    and computing statistical bounds for each parameter.
    """

    @staticmethod
    def fit_normals(uncertainty_metadata_df:pd.DataFrame, plot_distributions:bool=False, sample_size:int=1000000) -> pd.DataFrame:
        """
        Fit normal distributions to parameters defined with non-normal uncertainty types.

        For each row in `uncertainty_metadata_df`, this method:
          1. Draws `sample_size` samples from the parameter’s defined distribution.
          2. Uses the sample’s percentile-point function (PPF) to fit a normal (via mean and std).
          3. Optionally plots the histogram of raw samples against the fitted normal PDF.
          4. Returns a new DataFrame where each parameter’s `loc` and `scale` reflect the
             fitted normal, and `uncertainty_type` is set to 3 (normal).

        Args:
            uncertainty_metadata_df (pd.DataFrame):
                Indexed by parameter ID, with columns specifying the original distribution
                type and its parameters (e.g. for lognormal, triangular, etc.).
            plot_distributions (bool):
                If True, display a histogram + fitted-normal curve for each parameter.
                Defaults to False.
            sample_size (int):
                Number of random draws per parameter when fitting. Defaults to 1_000_000.

        Returns:
            pd.DataFrame:
                Indexed by parameter ID, with columns:
                  - `loc` (float): Mean of the fitted normal distribution.
                  - `scale` (float): Standard deviation of the fitted normal.
                  - `uncertainty_type` (int): Always 3, indicating “normal” type.
        """
        normal_uncertainty_metadata_df = uncertainty_metadata_df.copy()
        print('{} parameters with non normal distribution are transformed into normal distributions via max likelihood approximation'.format((uncertainty_metadata_df['uncertainty_type'] != 3).sum()))
        # For each parameter:
        #   - generate random samples from its original distribution
        #   - estimate mean and std via max likelihood fit of the percent‐point function samples (ppf)
        #   - replace in returned DataFrame
        for param_index, metadata in uncertainty_metadata_df[uncertainty_metadata_df['uncertainty_type'] != 3].iterrows():
            if metadata['uncertainty_type'] == 1:
                raise Exception('The intervention flow has the "no uncertainty" distribution type. This is not allowed')
            metadata_uncertainty_array = stats_arrays.UncertaintyBase.from_dicts(metadata.to_dict())
            uncertainty_choice = stats_arrays.uncertainty_choices[metadata['uncertainty_type']]
            # Sample the non-normal distribution 
            param_samples = uncertainty_choice.random_variables(metadata_uncertainty_array, sample_size)
            # Calculate the ppf values to 
            percentages = np.expand_dims(np.linspace(0.001, 0.999, 1000, axis=0), axis=0)
            x = uncertainty_choice.ppf(metadata_uncertainty_array, percentages=percentages)
            x, y = uncertainty_choice.pdf(metadata_uncertainty_array, xs=x)
            # Fit a normal distribution to the sampled data
            loc_norm, scale_norm = scipy.stats.norm.fit(param_samples.T)
            if plot_distributions:
                _, ax = plt.subplots(1, 1)
                # plot the histrogram of the samples
                ax.hist(param_samples.T, density=True, bins='auto', histtype='stepfilled', alpha=0.2, label='{} samples'.format(uncertainty_choice.description))
                # plot the lognormal pdf
                ax.plot(x, y, 'k-', lw=2, label='frozen {} pdf'.format(uncertainty_choice.description))
                # Plot the fitted normal distibution
                ax.plot(x, scipy.stats.norm.pdf(x,  loc=loc_norm, scale=scale_norm), 'b-', lw=2, label='fitted normal pdf')
                ax.set_title(str(param_index))
                ax.legend(loc='best', frameon=False)
            # Overwrite the lognormal distribution statistics with the fitted normal 
            normal_uncertainty_metadata = {
                'scale':scale_norm,
                'loc':loc_norm,
                'uncertainty_type':stats_arrays.NormalUncertainty.id
            }
            normal_uncertainty_metadata_df.loc[param_index] = normal_uncertainty_metadata
        if plot_distributions:
            plt.show()
        return normal_uncertainty_metadata_df
    
    @staticmethod
    def compute_bounds(uncertainty_metadata:dict, return_type:str='df') -> Union[pd.DataFrame, dict]:
        """
        Compute mean, median (or mode), and 95% CI bounds for each parameter.

        Iterates over a dictionary mapping parameter IDs to uncertainty definitions
        (in the format accepted by `stats_arrays.UncertaintyBase`). For each parameter,
        it computes:
          - `mean`
          - `median` (or mode, depending on distribution)
          - `lower` and `upper` bounds of the 95% confidence interval
          - preserves the original `amount` value

        Args:
            uncertainty_metadata (dict):
                {param_id: {‘uncertainty_type’: int, …distribution params…}}
            return_type (str):
                - `'df'`   → return a pandas.DataFrame indexed by param_id with columns
                  `['mean', 'median', 'lower', 'upper', 'amount']`
                - `'dict'` → return a dict[param_id] = {same keys & values}

        Returns:
            Union[pd.DataFrame, dict]:
                Computed bounds as specified by `return_type`.

        Raises:
            ValueError: If any parameter’s computed `upper` ≤ `lower`.
        """
        uncertainty_bounds = {}
        for indx, uncertainty_dict in uncertainty_metadata.items():
            uncertainty_array = stats_arrays.UncertaintyBase.from_dicts(uncertainty_dict)
            uncertainty_choice = stats_arrays.uncertainty_choices[uncertainty_dict['uncertainty_type']]
            parameter_statistics = uncertainty_choice.statistics(uncertainty_array)
            # ATTN: for some reason doe the uniform distribution give out the statistic in a 2d array, therefore we are unpacking them here
            if not isinstance(parameter_statistics['mean'], float):
                parameter_statistics = {key: value[0][0] for key, value in parameter_statistics.items()}
            uncertainty_bounds[indx] = parameter_statistics
            uncertainty_bounds[indx]['amount'] = uncertainty_dict['amount']
        uncertainty_bounds_df = pd.DataFrame(uncertainty_bounds).T
        # Test if the bounds are valid upperbound > lowerbound
        if ((uncertainty_bounds_df['upper'] - uncertainty_bounds_df['lower']) <= 0).any():
            raise Exception('There is one bound where the lower bound which is equal or larger than the upper bound')
        match return_type:
            case 'df':
                return uncertainty_bounds_df
            case 'dict':
                return uncertainty_bounds
            case _:
                raise Exception(f'Not defined return_type: {return_type}')



# class UncertaintyModule:
#     """Orchestrate uncertainty prep for GSA or CC workflows"""
#     def __init__(self, intervention_meta_df, cf_meta_df, bounds_df, strategies):
#         self.interv_meta_raw = intervention_meta_df
#         self.cf_meta_raw = cf_meta_df
#         self.bounds_df = bounds_df
#         self.strategies = strategies
#         self.processor = UncertaintyProcessor()
#         self.interv_meta, self.cf_meta = None, None

#     def prepare(self):
#         sep_i, sep_c = UncertaintySeparator(self.interv_meta_raw), UncertaintySeparator(self.cf_meta_raw)
#         def_i, undef_i = sep_i.separate(); def_c, undef_c = sep_c.separate()

#     def apply_uncertainty_strategies(self):
#         for strat in self.strategies:
#             strat.assign(sep_i.meta, undef_i); strat.assign(sep_c.meta, undef_c)




# === Global Sensitivity Analysis ===
class GlobalSensitivityAnalysis:
    """
    Performs a global sensitivity analysis (e.g., Sobol) on environmental impacts computed by PULPO.

    This class orchestrates the preparation of uncertainty bounds, generation of parameter samples,
    model evaluation on those samples, and calculation of Sobol sensitivity indices using SALib.
    It handles both intervention flows (IF) and characterization factors (CF).

    Attributes:
        result_data (dict):
            Deterministic PULPO results, including 'impacts' and 'Scaling Vector'.
        lci_data (dict):
            Life-Cycle Inventory matrices and metadata for processes and interventions.
        unc_metadata (dict[str:pd.DataFrame]):
            Uncertainty metadata containing uncertainty informtaion for characterization factors and intervention flows (must have no undefined types).
        sampler:
            SALib sampling function (e.g., saltelli.sample).
        analyser:
            SALib analysis function (e.g., sobol.analyze).
        sample_size (int):
            Number of parameter sets to draw for the sensitivity analysis.
        sample_impacts (pd.Series or pd.DataFrame):
            Placeholder for total impacts of each sample (initialized to None).
        sample_characterized_inventories (pd.DataFrame):
            Placeholder for characterized inventory samples (initialized to None).
        sensitivity_indices (pd.DataFrame):
            Placeholder for computed sensitivity indices (initialized to None).
    """
    def __init__(
        self,
        result_data: dict,
        lci_data: dict,
        unc_metadata: dict,
        sampler,
        analyser,
        sample_size: int,
        method:str
    ):
        """
        Initialize the global sensitivity analysis.

        Args:
            result_data (dict):
                Deterministic model output with keys 'impacts' (labels & values)
                and 'Scaling Vector' (for final impact calculation).
            lci_data (dict):
                Contains 'process_map_metadata' and 'intervention_map_metadata'
                for reconstructing labels in plots.
            unc_metadata (dict[str:pd.DataFrame]):
                Uncertainty metadata containing uncertainty informtaion for characterization factors 
                and intervention flows (must have no undefined types), contains "cf" and "if" keys.
            sampler:
                SALib sampling function (e.g., SALib.sample.saltelli).
            analyser:
                SALib analysis function (e.g., SALib.analyze.sobol).
            sample_size (int):
                Number of samples to generate for the analysis.
        """
        self.result_data = result_data # This is the optimization solution at which we compute the GSA
        self.method = method # ATTN: This might generate errors in the future
        self.lci_data = lci_data # from pulpo_worker
        self.unc_metadata = unc_metadata
        if (unc_metadata["if"].uncertainty_type == 0).any():
            raise Exception('There are still intervention flows with undefined uncertainty information')
        if (unc_metadata["cf"].uncertainty_type == 0).any():
            raise Exception('There are still characterization factors with undefined uncertainty information')
        self.sampler = sampler # from SALib.sample
        self.analyser = analyser # from SALib.analyze
        self.sample_size = sample_size
        self.sample_impacts = None
        self.sample_characterized_inventories = None
        self.sensitivity_indices = None

    def perform_gsa(self) -> None:
        """
        Calls all relevant methods including plots to perform a full GSA with initialized data
        """
        gsa_problem, all_bounds_indx_dict = self.define_problem()
        sample_data_if, sample_data_cf = self.sample(gsa_problem, all_bounds_indx_dict)
        sample_impacts, sample_characterized_inventories = self.run_model(sample_data_if, sample_data_cf)
        total_Si = self.analyze(gsa_problem, sample_impacts)
        total_Si_metadata = self.generate_Si_metadata(all_bounds_indx_dict, total_Si)
        colormap_base, colormap_SA_barplot = self.plot_top_total_sensitivity_indices(total_Si, total_Si_metadata)
        self.plot_total_env_impact_contribution(
            sample_characterized_inventories, 
            total_Si_metadata, 
            colormap_base=colormap_base, 
            colormap_SA_barplot=colormap_SA_barplot,
        )


    def _compute_bounds(self) -> tuple[dict, dict]:
        """
        Compute 95%-CI bounds for both IF and CF parameters.

        Uses `UncertaintyProcessor.compute_bounds` to turn metadata into
        dictionaries of lower/upper bounds for SALib.

        Returns:
            Tuple[dict, dict]:
                - if_bounds: mapping IF parameter names → {'lower', 'upper', 'mean', 'median', 'amount'}
                - cf_bounds: mapping CF parameter names → same structure
        """
        if_bounds = UncertaintyProcessor.compute_bounds(self.unc_metadata['if'].T.to_dict(), return_type='dict')
        cf_bounds = UncertaintyProcessor.compute_bounds(self.unc_metadata["cf"].T.to_dict(), return_type='dict')
        return if_bounds, cf_bounds

    def define_problem(self) -> tuple[dict, dict]:
        """
        Build the SALib problem definition combining IF and CF.

        Returns:
            problem (dict): {
                'num_vars': int,
                'names': List[str],
                'bounds': List[List[lower, upper]]
            }
            index_map (dict): {
                'if_start': 0,
                'cf_start': index where CF parameters begin
            }
        """
        if_bounds, cf_bounds = self._compute_bounds()
        all_bounds = if_bounds | cf_bounds
        all_bounds_indx_dict = {
            "if_start": 0,
            "cf_start":len(if_bounds),
        }
        problem = {
            'num_vars': len(all_bounds),
            'names': list(all_bounds.keys()),
            'bounds': [[bound['lower'], bound['upper']]for bound in all_bounds.values()]
        }
        print('problem includes:\n{} uncertain intervention flows\n{} uncertain characterization factors'.format(len(if_bounds), len(cf_bounds)))
        return problem, all_bounds_indx_dict

    def sample(self, problem:dict, all_bounds_indx_dict:dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate parameter samples for intervention flows and characterization factors.

        Args:
            problem (dict): SALib problem dict from `define_problem`.
            all_bounds_indx_dict (dict): Contains 'cf_start' to split the sample matrix.

        Returns:
            sample_data_if: DataFrame of IF samples (sparse).
            sample_data_cf: DataFrame of CF samples (sparse).
        """
        sample_data = self.sampler.sample(problem, self.sample_size)
        sample_data_if = pd.DataFrame.sparse.from_spmatrix(
            scipy.sparse.csr_matrix(sample_data[:,:all_bounds_indx_dict['cf_start']]), 
            columns=problem['names'][:all_bounds_indx_dict['cf_start']]
            # columns=pd.MultiIndex.from_tuples(problem['names'][:all_bounds_indx_dict['intervention_flows_end']])
            )
        sample_data_cf = pd.DataFrame.sparse.from_spmatrix(
            scipy.sparse.csr_matrix(sample_data[:,all_bounds_indx_dict['cf_start']:]), 
            columns=problem['names'][all_bounds_indx_dict['cf_start']:]
            )
        return sample_data_if, sample_data_cf

    def _compute_env_cost(
        self,
        sample_data_if: pd.DataFrame,
        sample_data_cf: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate environmental cost contributions for each sample.

        Aligns CF samples to IF structure and computes element-wise cost: Q·B.

        Args:
            sample_data_if (pd.DataFrame): IF parameter samples.
            sample_data_cf (pd.DataFrame): CF parameter samples.

        Returns:
            sample_env_cost: cost contributions per sample.
            level_index_if: DataFrame mapping multi-index for inventory labeling.
        """
        # Compute the environmental costs $Q \cdot B$ by reindexing the chracterization factors sample based on the intervention flow sample, so we can do dot product between for each characterization factors corresponding to the intervnetnion flow
        level_index_if= pd.DataFrame.from_records(sample_data_if.columns.values)
        sample_data_cf_expanded = sample_data_cf.reindex(level_index_if[0].values, axis=1)
        sample_data_if.columns = level_index_if[0].values
        sample_env_cost = sample_data_cf_expanded * sample_data_if
        sample_env_cost.columns = level_index_if[1].values
        return sample_env_cost, level_index_if
    
    def _compute_env_impact(self, sample_env_cost):
        """
        Compute total environmental impacts from cost contributions.

        Args:
            sample_env_cost (pd.DataFrame): Environmental cost per sample.

        Returns:
            sample_characterized_inventories (pd.DataFrame): inventory flows × samples.
            sample_impacts (pd.Series): total impact per sample.
        """
        # Compute the environmental impact using a dot product of the reindex scaling vector
        # Set the columns values to match the intervention columns
        scaling_vector_expanded = self.result_data['Scaling Vector']['Value'].reindex(sample_env_cost.columns)
        sample_characterized_inventories = sample_env_cost * scaling_vector_expanded
        sample_impacts = sample_env_cost @ scaling_vector_expanded
        return sample_characterized_inventories, sample_impacts
    
    def run_model(self, sample_data_if:pd.DataFrame, sample_data_cf:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute the LCI and LCIA for all samples.

        Performs cost & impact calculations, prints summary stats, and plots distribution.

        Args:
            sample_data_if (pd.DataFrame): IF samples.
            sample_data_cf (pd.DataFrame): CF samples.

        Returns:
            sample_impacts (pd.Series): total impact per sample.
            sample_characterized_inventories (pd.DataFrame): flows × samples.
        """
        sample_env_cost, level_index_if = self._compute_env_cost(sample_data_if, sample_data_cf)
        sample_characterized_inventories, sample_impacts = self._compute_env_impact(sample_env_cost)
        # Since multiindex columns are to slow to compute, rename the characterized inventory columns again after they have been computed to match the inventory flows indices
        sample_characterized_inventories.columns = pd.MultiIndex.from_frame(level_index_if)
        print(f'The statistics of the the sample impacts: {self.method}') # ATTN: if we have more than one method in the results data this might become a problem
        print(sample_impacts.sparse.to_dense().describe())
        print('The deterministic impact is {}'.format('\n'.join(['{} : {:e}'.format(values[0], values[1]) for values in self.result_data['Impacts'].values])))
        # Show the z-value and the distribution of the output
        sample_impacts.plot.hist(bins=50)
        print(sample_impacts.shape)
        # The z-value of the total environmental impact
        print('the z-value of the total impact: {}'.format(sample_impacts.sparse.to_dense().std()/abs(sample_impacts.mean())))
        return sample_impacts, sample_characterized_inventories

    def analyze(self, problem:dict, sample_impacts:pd.DataFrame):
        """
        Calculate Sobol sensitivity indices from sampled impacts.

        Args:
            problem (dict): SALib problem definition.
            sample_impacts (pd.Series): Impact per sample.

        Returns:
            pd.DataFrame: DataFrame of total Sobol indices 'ST' and 'ST_conf'
                          indexed by parameter names.
        """
        sensitivity_indices = self.analyser.analyze(problem, sample_impacts.sparse.to_dense().values, parallel=True)
        # total_Si, first_Si, second_Si = sensitivity_indices.to_df()
        total_Si = pd.DataFrame([sensitivity_indices['ST'].T, sensitivity_indices['ST_conf'].T], index=['ST', 'ST_conf'], columns=problem['names']).T
        # Calculate total explained variance
        print("The total explained variance is \n{:.4}%".format(total_Si["ST"].sum()*100))
        return total_Si

    def generate_Si_metadata(
        self,
        all_bounds_indx_dict: dict,
        total_Si: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create human-readable labels for sensitivity indices plots.

        Args:
            all_bounds_indx_dict (dict): Contains 'cf_start' to split IF/CF.
            total_Si (pd.DataFrame): DataFrame of sensitivity indices.

        Returns:
            pd.DataFrame: Single-column DataFrame ('bar_names') mapping parameters
                          to "Intervention --- Process" labels.
        """
        # Generate the data and the names for the contribution plot
        metadata_dict = {}
        for (intervention_index, process_index) in total_Si.index[:all_bounds_indx_dict['cf_start']]:
            metadata_dict[(intervention_index, process_index)] = '{} --- {}'.format(self.lci_data['process_map_metadata'][process_index], self.lci_data['intervention_map_metadata'][intervention_index])
        for intervention_index in total_Si.index[all_bounds_indx_dict['cf_start']:]:
            metadata_dict[intervention_index] = '{} --- {}'.format(self.lci_data['intervention_map_metadata'][intervention_index], self.method)
        total_Si_metadata = pd.DataFrame([metadata_dict], index=['bar_names']).T
        return total_Si_metadata

    def plot_top_total_sensitivity_indices(self, total_Si:pd.DataFrame, total_Si_metadata:pd.DataFrame, top_amount:int=10) -> tuple[pd.Series, pd.Series]:
        """
        Plot the top contributors to total variance (Sobol ST).

        Args:
            total_Si (pd.DataFrame): Contains 'ST' and 'ST_conf' columns.
            total_Si_metadata (pd.DataFrame): 'bar_names' labels.
            top_amount (int): Number of top parameters to display.

        Returns:
            colormap_base: list of colors used.
            colormap_SA_barplot: pd.Series mapping params → colors.
        """
        # Plot the contribution to variance
        top_total_Si = total_Si.sort_values('ST', ascending=False).iloc[:top_amount,:]
        top_total_Si_metadata = total_Si_metadata.loc[top_total_Si.index]
        colormap_base = mpl.cm.tab20.colors
        colormap_SA_barplot = pd.Series(colormap_base[:top_total_Si.shape[0]], index=top_total_Si.index)
        plot_contribution_barplot_with_err(data=top_total_Si, metadata=top_total_Si_metadata, colormap=colormap_SA_barplot, bbox_to_anchor_center=1.7, bbox_to_anchor_lower=-.6)
        return colormap_base, colormap_SA_barplot
        
    def plot_total_env_impact_contribution(
            self,
            sample_characterized_inventories: pd.DataFrame,
            total_Si_metadata: pd.DataFrame,
            top_amount: int = 10,
            colormap_base: pd.Series = pd.Series([]),
            colormap_SA_barplot: pd.Series = pd.Series([])
        ) -> pd.DataFrame:
        """
        Plot each process's share of the total environmental impact.

        Args:
            sample_characterized_inventories (pd.DataFrame):
                Characterized inventory flows per sample.
            total_Si_metadata (pd.DataFrame):
                'bar_names' for labeling processes.
            top_amount (int): Number of top processes to include.
            colormap_base (pd.Series): Base colormap mapping (optional).
            colormap_SA_barplot (pd.Series): Sensitivity-plot colormap mapping (optional).

        Returns:
            pd.DataFrame: Data prepared for the linked impact contribution plot.
        """        
        #Plot the main contributing variables to the total environmental impact
        # Generate the data
        top_characterized_inventories_indcs = sample_characterized_inventories.mean().abs().sort_values(ascending=False).iloc[:top_amount].index
        data_plot = pd.DataFrame([])
        characterized_inventories_scaled = (sample_characterized_inventories.T / sample_characterized_inventories.T.sum()).T #sample_characterized_inventories.div(sample_characterized_inventories.sum(axis=1), axis=0)
        data_plot["ST"] = characterized_inventories_scaled.mean()[top_characterized_inventories_indcs]
        data_plot["ST_conf"] = characterized_inventories_scaled.sparse.to_dense().std()[top_characterized_inventories_indcs]
        data_plot.index = data_plot.index.to_flat_index()
        metadata_plot = total_Si_metadata.loc[data_plot.index,['bar_names']]
        # Plot the total environmental impact for the top processes
        if not colormap_base:
            colormap_base = pd.Series(mpl.cm.tab20.colors[:data_plot.shape[0]], index=data_plot.index)
        plot_linked_contribution_barplot(data_plot, metadata=metadata_plot, impact_category=self.method, colormap_base=colormap_base, colormap_linked=colormap_SA_barplot, savefig=False, bbox_to_anchor_center=1.7, bbox_to_anchor_lower=-.6)
        return data_plot


class CCFormulationBase:
    """
    Main class for Chance Constraint formulation, directly formulates the Pareto problem upon initialization

    Subclasses must override `formulate` and `update_problem` to define a
    concrete Pyomo model and its λ‐level updates.
    """
    def __init__(self,
                 # ATTN: Maybe make the metadata_df quasi arguments with zeros as default value, to allow different formulations
                 unc_metadata: Dict[str,pd.DataFrame],
                 pulpo_worker,
                 method:str,
                 choices:dict,
                 demand:dict,
                 ):
        """
        Initialize the chance-constraint formulation.

        Stores characterization‐factor and intervention‐flow metadata, the
        PULPO model constructor, impact method, normative choices, and demand,
        then calls `formulate` to assemble the base model.

        Args:
            unc_metadata (dict[str:pd.DataFrame]):
                Uncertainty metadata containing uncertainty informtaion for paramters in chance constaints
            (must have no undefined types)
            pulpo_worker:
                Factory or callable that constructs the deterministic Pyomo model.
            method (str):
                LCIA method name (used in impact calculations).
            choices (dict):
                Formulation options (e.g. whether to use L1 vs. L2 norms).
            demand (dict):
                Demand vector mapping each intervention flow to its required amount.
        """
        self.unc_metadata = unc_metadata
        self.method = method
        self.pulpo_worker = pulpo_worker
        self.choices = choices
        self.demand = demand
        self.formulate()
    
    def formulate(self) -> None:
        """
        Build the initial chance constraint formulation (to be overridden).

        Returns:
            None
        """
        pass

    def update_problem(self, lambda_level:float) -> None:
        """
        Inject or update the ε‐constraint for a given risk level (to be overridden).

        Modifies the existing Pyomo model to enforce that the specified
        chance constraint (e.g. P{impact ≤ threshold} ≥ λ) is satisfied
        at the current `lambda_level`. This supports tracing the Pareto front.

        Args:
            lambda_level (float):
                Target confidence/risk threshold (e.g., 0.95 for 95% quantile).
        
        Returns:
            None
        """
        pass

class CCFormulationObjIndividualNormalL1(CCFormulationBase):
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
        normal_metadata = self.transform_to_normal()
        self.envcost_std = self.compute_envcost_variance(normal_metadata['cf'], normal_metadata['if'])
        self.envcost_mean = self.compute_envcost_mean()
        self.check_envcost_variance(self.envcost_std)


    def transform_to_normal(self) -> Dict[str,pd.DataFrame]:
        """
        Fit Normal distributions to all CF and IF uncertainty metadata.

        Uses the UncertaintyProcessor to convert any non‐normal uncertainty
        definitions into equivalent Normal distributions.

        Returns:
            normal_metadata (Dict[str,pd.DataFrame]): Fitted Normal loc/scale for parameters in chance constaints (e.g., "cf", "if").
        """
        normal_metadata = {}
        for var_name, metadata_df in self.unc_metadata.items():
            normal_metadata[var_name] = UncertaintyProcessor.fit_normals(metadata_df)
        # ATTN: Check if the fit_normals runs through with 0 as standard deviations
        return normal_metadata


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
            if process_id in if_normal_metadata_df.index.get_level_values(level='col'):
                intervention_flow_std = if_normal_metadata_df.xs(process_id, level='col', axis=0, drop_level=True)['scale']
                characterization_factor_mean = pd.Series(
                    self.pulpo_worker.lci_data["matrices"][self.method].diagonal()[
                        intervention_flow_std.index.get_level_values(level='row')
                        ],
                    index=intervention_flow_std.index.get_level_values(level='row')
                )
                # Reindex so that we can perform a matrix multiplication on all intervention flows
                characterization_factor_mean = characterization_factor_mean.reindex(intervention_flow_std.index, axis=0, level='row')
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
            if (intervention_flows_extracted[process_id] > 0).any() and process_id in if_normal_metadata_df.index.get_level_values(level='col'):
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
        ppf_lambda_QB = scipy.stats.norm.ppf(lambda_env_cost)
        environmental_cost_updated = {(process_id, self.method): self.envcost_mean[process_id] + ppf_lambda_QB * self.envcost_std[process_id] for process_id in self.envcost_std.keys()}
        self.pulpo_worker.instance.ENV_COST_MATRIX.store_values(environmental_cost_updated, check=True)

class CCFormulationObjVBIndividualNormalL1(CCFormulationObjIndividualNormalL1):
    """
    Implements an individual chance‐constraint formulation on the objective using the L1 norm and 
    on the variable bounds with normally distributed uncertainties.

    This subclass approximates all uncertain intervention flows, characterization factors, and variable bounds as Normal(μ,σ²),
    computes the aggregated standard deviation of total environmental cost under an L1 norm, and then
    traces Pareto‐optimal solutions by varying the confidence level (λ).
    """

class CCFormulationObjIndividualNormalL2(CCFormulationBase):

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
        result_data = self.extract_results()
        return result_data

    def extract_results(self):
        """
        Extract the results from the chance-constrained solver after execution.

        Returns:
            dict: A dictionary containing the extracted results.
        """
        result_data = pulpo.saver.extract_results(self.cc_formulation.pulpo_worker.instance, self.cc_formulation.pulpo_worker.project, self.cc_formulation.pulpo_worker.database, self.cc_formulation.choices, {}, self.cc_formulation.demand,
                                        self.cc_formulation.pulpo_worker.lci_data['process_map'], self.cc_formulation.pulpo_worker.lci_data['process_map_metadata'],
                                        self.cc_formulation.pulpo_worker.lci_data['intervention_map'], self.cc_formulation.pulpo_worker.lci_data['intervention_map_metadata']) # ATTN: this should be wrapped in the pulpo module similar to the save_results method
        return result_data
    
    def compare_subsequent_paretosolutions(self, result_data_CC):
        """
        Compare impacts and decision choices across multiple Pareto solutions.

        Args:
            result_data_CC (dict of float to dict): Mapping from each lambda level
                to its corresponding solver result dictionary.
        """
        impacts = {}
        print(self.cc_formulation.method)
        for lambda_QB, result_data in result_data_CC.items():
            impacts[lambda_QB] = result_data['Impacts'].set_index('Key').loc[self.cc_formulation.method,'Value']
            print('{}: {}'.format(lambda_QB, impacts[lambda_QB]))
        # The changs in the choices of the optimizer
        choices_results = {}
        for i_CC, (lambda_QB, result_data) in enumerate(result_data_CC.items()):
            for choice in self.cc_formulation.choices.keys():
                if i_CC == 0:
                    choices_results[choice] = result_data['Choices'].xs(tuple(self.cc_formulation.choices.keys()), axis=1)[['Process', 'Capacity']].dropna()
                choices_results[choice] = choices_results[choice].join(result_data['Choices'].xs(tuple(self.cc_formulation.choices.keys()), axis=1)['Value'].rename(lambda_QB), how='left')
        for choice, choice_result in choices_results.items():
            print(choice)
            print(choice_result)

        # Changes in the scaling vector and the characterized and scaled inventories
        lambda_array = list(result_data_CC.keys())
        for lambda_1, lambda_2 in zip(lambda_array[:len(lambda_array)-1], lambda_array[1:len(lambda_array)]):
            print(f'lambda_1: {lambda_1}\nlambda_2: {lambda_2}\n')
            scaling_vector_diff = ((result_data_CC[lambda_1]['Scaling Vector']['Value'] - result_data_CC[lambda_2]['Scaling Vector']['Value']))
            scaling_vector_ratio = (scaling_vector_diff / result_data_CC[lambda_1]['Scaling Vector']['Value']).abs().sort_values(ascending=False)
            environmental_cost_mean = {env_cost_index[0]: env_cost for env_cost_index, env_cost in result_data_CC[lambda_1]['ENV_COST_MATRIX']['ENV_COST_MATRIX'].items()}
            characterized_scaling_vector_diff = (scaling_vector_diff * pd.Series(environmental_cost_mean).reindex(scaling_vector_diff.index)).abs()
            characterized_scaling_vector_diff_relative = (characterized_scaling_vector_diff / result_data_CC[lambda_1]['impacts'].set_index('Key').loc[self.cc_formulation.method, 'Value']).abs().sort_values(ascending=False)

            print('Amount of process scaling variables that changed:\n{}: >1% \n{}: >10%\n{}: >100%\n{}: >1000%\n'.format((scaling_vector_ratio > 0.01).sum(), (scaling_vector_ratio > 0.1).sum(), (scaling_vector_ratio > 1).sum(), (scaling_vector_ratio > 10).sum()))
            print('Amount of process characterized scaling variables (impacts per process) that changed:\n{}: >1% \n{}: >10%\n{}: >100%\n{}: >1000%\n'.format((characterized_scaling_vector_diff_relative > 0.01).sum(), (characterized_scaling_vector_diff_relative > 0.1).sum(), (characterized_scaling_vector_diff_relative > 1).sum(), (characterized_scaling_vector_diff_relative > 10).sum()))
            print('{:.5e}: is the maximum impact change in one process\n{:.5e}: is the total impact change\n'.format(characterized_scaling_vector_diff_relative.max(), characterized_scaling_vector_diff_relative.sum()))

            amount_of_rows_for_visiualization = 10
            print('The relative change of the scaling vector (s_lambda_1 - s_lambda_2)/s_lambda_1:\n')
            print(scaling_vector_ratio.iloc[:amount_of_rows_for_visiualization].rename(result_data_CC[lambda_2]['Scaling Vector']['Process metadata']).sort_values(ascending=False))
            print('\n---\n')
            print('The relative change of the characterized scaling vector (s_lambda_1 - s_lambda_2)*QB_s / QBs:\n')
            print(characterized_scaling_vector_diff_relative.iloc[:amount_of_rows_for_visiualization].rename(result_data_CC[lambda_2]['Scaling Vector']['Process metadata']))
            print('\n---\n')

    def plot_pareto_front(self, result_data_CC:dict, cutoff_value:float):
        """
        Plot the Pareto front and highlight main contributing variables.

        Args:
            result_data_CC (dict of float to dict): Mapping from each lambda level
                to its corresponding solver result dictionary.
            cutoff_value (float): Relative threshold for filtering main decision variables
                to include in the bar plot.
        """
        data_QBs_list = []
        for lamnda_QBs, result_data in result_data_CC.items():
            environmental_cost_mean = {env_cost_index[0]: env_cost for env_cost_index, env_cost in result_data_CC[lamnda_QBs]['ENV_COST_MATRIX']['ENV_COST_MATRIX'].items()}
            QBs = result_data['Scaling Vector']['Value'] * pd.Series(environmental_cost_mean).reindex(result_data['Scaling Vector']['Value'].index)
            QBs_main = QBs[QBs.abs() > cutoff_value*QBs.abs().sum()]
            QBs_main.name = lamnda_QBs
            data_QBs_list.append(QBs_main)
            print('With a cutoff value of {}, we keep {} process to an error of {:.2%}'.format(cutoff_value, len(QBs_main), abs(1 - QBs_main.sum()/QBs.sum())))
        data_QBs = pd.concat(data_QBs_list, axis=1)
        data_QBs = data_QBs.rename(index={process_id: self.cc_formulation.pulpo_worker.lci_data['process_map_metadata'][process_id] for process_id in data_QBs.index})
        bbox_to_anchor = (0.65, -1.)
        plot_CC_pareto_solution_bar_plots(data_QBs, self.cc_formulation.method, bbox_to_anchor=bbox_to_anchor)


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
            self.cc_formulation.update_problem(lambda_level)
            self.cc_formulation.pulpo_worker.solve()
            result_data_CC[lambda_level] = self.extract_results()
        return result_data_CC


class AdaptiveSamplingSolver(BaseParetoSolver):
    """
    Solver for adaptive sampling based Pareto approximation.
    """
    def solve(self, cc_formulation, **kwargs):
        pass

