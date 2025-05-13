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
from typing import Union, List, Optional

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
    width = 6
    height = 6
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
    """Abstract base for case studies (Rice husk, Electricity, Ammonia)."""
    def __init__(self, project:str, database: Union[str, List[str]], method:Union[str, List[str], dict], directory :str ):
        self.project = project
        self.database = database
        self.method = method
        self.directory = directory
        self.pulpo_worker:pulpo.PulpoOptimizer = None
        self.demand:dict = {}
        self.choices:dict = {}

    def create_pulpo_worker(self):
        # Create a **PulpoOptimizer** instance. This class is used to interact with the LCI database and solve the optimization problem. It is specified by the project, database, method and directory.
        self.pulpo_worker = pulpo.PulpoOptimizer(self.project, self.database, self.method, self.directory)
        # Import LCI data. After initializing the PulpoOptimizer instance, the LCI data is imported from the database.
        self.pulpo_worker.get_lci_data()

    def solve_and_summarize(self, file_name) -> dict:
        # Instantiat and solve the optimization model
        self.pulpo_worker.solve()
        # Save and summarize the results
        result_data = pulpo.saver.extract_results(self.pulpo_worker.instance, self.pulpo_worker.project, self.pulpo_worker.database, self.choices, {}, self.demand,
                                    self.pulpo_worker.lci_data['process_map'], self.pulpo_worker.lci_data['process_map_metadata'],
                                    self.pulpo_worker.lci_data['intervention_map'], self.pulpo_worker.lci_data['intervention_map_metadata']) # ATTN: this should be wrapped in the pulpo module similar to the save_results method
        self.pulpo_worker.summarize_results(choices=self.choices, demand=self.demand, zeroes=True)
        # self.pulpo_worker.save_results(result_data, file_name) # ATTN: This still does not work with the saver code probably still a mistake in there
        return result_data


class RiceHuskCase(BaseCaseStudy):
    """1.1 Rice husk problem"""
    def __init__(self):
        # Set the parameters for the rise husk example to instancialize PULPO
        self.project = "rice_husk_example" 
        if self.project not in bw2data.projects: #ATTN: test
            pulpo.install_rice_husk_db()
        self.database = "rice_husk_example_db"
        self.method = {"('my project', 'climate change')":1}
        self.directory = os.path.join(os.path.dirname(os.getcwd()), 'develop_tests/data')

    def define_problem(self):
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
    """1.2 Electricity showcase problem"""
    def __init__(self):
        self.project = "pulpo"
        self.database = "cutoff38"
        self.method = {"('IPCC 2013', 'climate change', 'GWP 100a')": 1,
                "('ReCiPe Endpoint (E,A)', 'resources', 'total')": 0,
                "('ReCiPe Endpoint (E,A)', 'human health', 'total')": 0,
                "('ReCiPe Endpoint (E,A)', 'ecosystem quality', 'total')": 0,
                "('ReCiPe Midpoint (E) V1.13', 'ionising radiation', 'IRP_HE')": 0}
        self.directory = os.path.join( os.path.dirname(os.getcwd()), 'develop_tests/data')

    def define_problem(self):
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
    """1.3 Ammonia case study"""
    def __init__(self):
        self.project = "pulpo-ammonia"
        self.database = ["nc-inventories-ei310-all", "ecoinvent-3.10-cutoff"]
        self.method = {"('IPCC 2021', 'climate change', 'GWP 100a, incl. H and bio CO2')":1}
        self.directory = os.path.join(os.path.dirname(os.getcwd()), 'develop_tests/data')

    def define_problem(self):
        choices_biogas = [
            "anaerobic digestion of animal manure, with biogenic carbon uptake",
            "anaerobic digestion of agricultural residues, with biogenic carbon uptake",
            "treatment of sewage sludge by anaerobic digestion, cut-off with biogenic carbon uptake",
            "treatment of industrial wastewater by anaerobic digestion, cut-off with biogenic carbon uptake",
            "treatment of biowaste by anaerobic digestion, cut-off with biogenic carbon uptake",
            "anaerobic digestion of sequential crop, with biogenic carbon uptake"
        ]
        choices_hydrogen = [
            "hydrogen production, biomass gasification",
            "hydrogen production, biomass gasification, with CCS",
            "hydrogen production, steam methane reforming of biomethane",
            "hydrogen production, steam methane reforming of biomethane, with CCS",
            "hydrogen production, steam methane reforming of natural gas, with CCS",
            "hydrogen production, PEM electrolysis, green",
            "green hydrogen",
            "hydrogen production, plastics gasification",
            "hydrogen production, plastics gasification, with CCS"
        ]
        choices_heat = [
            "heat from biomethane",
            "heat from biomethane, with CCS",
            "heat from hydrogen",
            "heat from natural gas, with CCS"
        ]
        choices_ammonia = [
            "ammonia production, steam methane reforming of biomethane",
            "ammonia production, steam methane reforming of biomethane, with CCS",
            "ammonia production, steam methane reforming of natural gas, with CCS",
            "ammonia production, from nitrogen and hydrogen"
        ]
        choices_biomethane = [
            "biogas upgrading to biomethane, chemical scrubbing",
            "biogas upgrading to biomethane, chemical scrubbing w/ CCS",
            "biogas upgrading to biomethane, membrane",
            "biogas upgrading to biomethane, membrane w/ CCS",
            "biogas upgrading to biomethane, pressure swing adsorption",
            "biogas upgrading to biomethane, pressure swing adsorption w/ CCS",
            "biogas upgrading to biomethane, water scrubbing",
            "biogas upgrading to biomethane, water scrubbing w/ CCS"
        ]

        # Retrieve activities for each category
        biogas_activities = self.pulpo_worker.retrieve_activities(activities=choices_biogas)
        hydrogen_activities = self.pulpo_worker.retrieve_activities(activities=choices_hydrogen)
        heat_activities = self.pulpo_worker.retrieve_activities(activities=choices_heat)
        biomethane_activities = self.pulpo_worker.retrieve_activities(activities=choices_biomethane)

        ammonia_activities = self.pulpo_worker.retrieve_activities(activities=choices_ammonia)
        # Add BAU Ammonia from ecoinvent as choice
        ammonia_activities.append(self.pulpo_worker.retrieve_activities(reference_products="ammonia, anhydrous, liquid", activities="ammonia production, steam reforming, liquid", locations="RER w/o RU")[0])
        # Choices
        self.choices = {
            "biogas": {x: 1e10 for x in biogas_activities},
            "hydrogen": {x: 1e10 for x in hydrogen_activities},
            "heat": {x: 1e10 for x in heat_activities},
            "biomethane": {x: 1e10 for x in biomethane_activities},
            "ammonia": {x: 1e10 for x in ammonia_activities},
        }
        # Demand
        ammonia_market = self.pulpo_worker.retrieve_activities(activities="new market for ammonia")
        self.demand = {ammonia_market[0]: 1}
        self.pulpo_worker.instantiate(choices=self.choices, demand=self.demand)


# === Parameter Filter ===
class ParameterFilter:
    """2. Filtering out negligible uncertain parameters"""
    def __init__(self, result_data:dict, lci_data: dict, choices: dict, demand:dict, method:str):
        self.result_data = result_data
        self.lci_data = lci_data # From pulpo_worker.lci_data
        self.choices = choices # From CaseStudy
        self.demand = demand # From CaseStduy
        self.method = method # From the result_data


    def prepare_sampling(self,  scaling_vector_strategy:str='naive') -> tuple[scipy.sparse.sparray, pd.Series]:
        # From notebook cells #### 2.0.1 & 2.0.2
        # We only consider uncertainty in the $B$ and $Q$ parameter matrizes. The scaling vector is given by the optimal solution.
        # We will look at the contribution of the parameters to the environmental impact objective:
        #     e(Q, B) =  Q \cdot B \cdot s
        characterization_matrix = self.lci_data["matrices"][self.method]
        print('chosen environmental impact method: {}'.format(self.method))
        # Define the scaling vector for the subsequent analysis as the optimization results
        # put the scaling vector returned from the optimization into the same order as the process map
        match scaling_vector_strategy:
            case 'naive':
                scaling_vector_series = self.result_data["scaling_vector"].set_index('ID')['Value'].sort_index()
            case 'constructed_demand':
                scaling_vector_series = self.construct_scaling_vector_from_choices()
            case _:
                raise Exception('Case not implemented.')
        return characterization_matrix, scaling_vector_series
        
    def construct_scaling_vector_from_choices(self) -> pd.Series:
        # Define the scaling vector for the subsequent analysis as a constructed demand vector
        # Create a demand vector which includes all alternatives in the demand use use the corresponding scaling vector of that LCA for the subsequent GSA and preparation steps, the idea is to include all relevant processes in the LCIA calculation instead of just those chosen by the optimizer at on Pareto point
        for product, alternatives in self.choices.items():
            demand_amount = self.result_data['choices'][product]["Value"].sum()
            for alternative in alternatives:
                self.demand[alternative] = demand_amount
        # Compute the scaling vector for the set constructed demand
        method_tuple = ast.literal_eval(self.method)
        lca = bw2calc.LCA(self.demand, method_tuple)
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



    def compute_LCI_LCIA(self, scaling_vector_series:pd.Series, characterization_matrix:scipy.sparse.sparray) -> tuple[float, scipy.sparse.sparray]:
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

    def filter_biosphereflows(self, characterized_inventory:scipy.sparse.sparray, lca_score:float, cutoff:float) -> list:
        #ATTN: Add a simple optimization loop to find the cutoff which results in an absolute change of around 1%
        # Filters the biosphere flows
        start = time()
        print('Characterized inventory:', characterized_inventory.shape, characterized_inventory.nnz)
        finv = characterized_inventory.multiply(abs(characterized_inventory) > abs(lca_score*cutoff))
        print('Filtered characterized inventory:', finv.shape, finv.nnz)
        characterized_inventory_indices = list(zip(*finv.nonzero()))
        # Since if negative and positive characterized inventories are cut away the explained fraction (finv.sum() / lca_score) can also be greater than 1
        deviation_from_lca_score = abs(1 - finv.sum() / lca_score)
        print('Deviation from LCA score:', deviation_from_lca_score)
        print('BIOSPHERE {} filtering resulted in {} of {} exchanges ({}% of total impact) and took {} seconds.'.format(
            characterized_inventory.shape,
            finv.nnz,
            characterized_inventory.nnz,
            np.round(100-deviation_from_lca_score * 100, 2),
            np.round(time() - start, 3),
        ))
        return characterized_inventory_indices


    def filter_characterization_factors(self, characterization_matrix:scipy.sparse.sparray, characterized_inventory_indices:list) -> list:
        # Filter characterization matrix
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
    """Extract/import uncertainty metadata from data sources"""
    def __init__(self, lci_data):
        self.lci_data = lci_data # from pulpo_worker.lci_data

    def get_intervention_meta(self, inventory_indices:list) -> pd.DataFrame:
        intervention_metadata_df = pd.DataFrame(self.lci_data['intervention_params'])
        intervention_metadata_df = intervention_metadata_df.set_index(["row", "col"])
        intervention_metadata_df = intervention_metadata_df.loc[inventory_indices]
        return intervention_metadata_df

    def get_cf_meta(self, method:str, characterization_indices:list) -> pd.DataFrame:
        characterization_params = self.lci_data["characterization_params"][method]
        characterization_metadata_df = pd.DataFrame(characterization_params).set_index('row')
        characterization_metadata_df = characterization_metadata_df.loc[characterization_indices]
        return characterization_metadata_df

    def separate(self, uncertainty_metadata_df:pd.DataFrame) -> tuple[dict, list]:
        # From notebook cells 91 & 94
        defined, undefined = {}, []
        for idx, row in uncertainty_metadata_df.iterrows():
            if row['uncertainty_type'] > 0:
                defined[idx] = row.to_dict()
            else:
                undefined.append(idx)
        print("Parameters with uncertainty information: {} \nParameters without uncertainty information: {}".format(len(defined), len(undefined)))
        return defined, undefined


class UncertaintyStrategyBase:
    """Base for strategies assigning uncertainty to undefined parameters"""
    def __init__(self, metadata_df:pd.DataFrame, defined_uncertainty_metadata:dict, undefined_uncertainty_indices:list):
        self.metadata_df = metadata_df # ATTN: rename to param_metadata_df
        self.defined_uncertainty_metadata = defined_uncertainty_metadata
        self.undefined_uncertainty_indices = undefined_uncertainty_indices


class TriangularStrategy(UncertaintyStrategyBase):
    """Assign triangular distribution based on bounds and scaling factors"""

    def _get_bounds(self):
        # Calls UncertaintyProcessor.compute_bounds
        if not self.defined_uncertainty_metadata:
            raise Exception('There are no uncertain parameters with defined uncertainty, as needed to interpolate the bouds.')
        self.uncertainty_bounds = UncertaintyProcessor.compute_bounds(self.defined_uncertainty_metadata)
            

    def compute_bounds_statistics(self) -> tuple[float, float]:
        ''' 
        Assumes that the bounds of the median of 95% confidence interval can be used to compute scaling factors.
        
        When these scaling factors are used to get the min and max of the triangular distribution, then the bounds returned from the triangular distribution will be smaller than the median computed here, since the 95% confidence interval of the triangular distribution lies within the mi and max value.
        '''
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

        
    def assign(self, upper_scaling_factor:float, lower_scaling_factor:float) -> pd.DataFrame:
        ''' The scaling factors are multiplied to the deterministic parameter amount and will give the maximum and minimum value of the triangular distribution '''
        # **ATTN For negative flows the skewness might need to be inversed!!**
        metadata_df = self.metadata_df.copy()
        for undefined_indx in self.undefined_uncertainty_indices:
            amount = metadata_df.loc[undefined_indx].amount
            # ATTN: BHL: If we have negative values than the skewdness which mostly is poisitve for positive flows will now be positive for negative flows (remain right skewed) while in reality negative flows might be left skewed (tail going away from zero not towards zero as now)
            metadata_df.loc[undefined_indx, 'loc'] = amount,
            metadata_df.loc[undefined_indx, 'maximum'] = amount + upper_scaling_factor * abs(amount),
            metadata_df.loc[undefined_indx, 'minimum'] = amount - lower_scaling_factor * abs(amount),
            metadata_df.loc[undefined_indx, 'uncertainty_type'] = 5,
        if ((metadata_df.loc[self.undefined_uncertainty_indices,'maximum'] - metadata_df.loc[self.undefined_uncertainty_indices,'minimum']) <= 0).any():
            raise Exception('There is a parameter with where the asigned minimum value is equal or larger than the asigned maximum value')
        # There can be negative flows and their upper and lower bounds need to be considered in detail!
        print('uncertain parameters with negative median value:')
        print(metadata_df.loc[self.undefined_uncertainty_indices].loc[metadata_df.loc[self.undefined_uncertainty_indices,'loc'] < 0])
        return metadata_df

class UncertaintyProcessor:
    """Compute uncertainties: sampling defined & fitting to normal"""

    @staticmethod
    def fit_normals(uncertainty_metadata_df:pd.DataFrame, plot_distributions:bool=False, sample_size:int=1000000) -> pd.DataFrame:
        normal_uncertainty_metadata_df = uncertainty_metadata_df.copy()
        print('{} parameters with non normal distribution are transformed into normal distributions via max likelihood approximation'.format((uncertainty_metadata_df['uncertainty_type'] != 3).sum()))
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
                'uncertainty_type':3
            }
            normal_uncertainty_metadata_df.loc[param_index] = normal_uncertainty_metadata
        if plot_distributions:
            plt.show()
        return normal_uncertainty_metadata_df
    
    @staticmethod
    def compute_bounds(uncertainty_metadata:dict, return_type:str='df') -> Union[pd.DataFrame, dict]:
        ''' Build a dictionary of mean, mode, median, and 95% confidence interval upper and lower values. '''
        uncertainty_bounds = {}
        for indx, uncertainty_dict in uncertainty_metadata.items():
            uncertainty_array = stats_arrays.UncertaintyBase.from_dicts(uncertainty_dict)
            uncertainty_choice = stats_arrays.uncertainty_choices[uncertainty_dict['uncertainty_type']]
            parameter_statistics = uncertainty_choice.statistics(uncertainty_array)
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
    def __init__(self, result_data:dict, lci_data:dict, cf_metadata_df:pd.DataFrame, if_metadata_df:pd.DataFrame, sampler, analyser, sample_size:int):
        self.result_data = result_data # This is the optimization solution at which we compute the GSA
        self.method = self.result_data['impacts']['Key'][0] # ATTN: This might generate errors in the future
        self.lci_data = lci_data # from pulpo_worker
        self.cf_metadata_df = cf_metadata_df
        if (if_metadata_df.uncertainty_type == 0).any():
            raise Exception('There are still intervention flows with undefined uncertainty information')
        self.if_metadata_df = if_metadata_df
        if (cf_metadata_df.uncertainty_type == 0).any():
            raise Exception('There are still characterization factors with undefined uncertainty information')
        self.sampler = sampler # from SALib.sample
        self.analyser = analyser # from SALib.analyze
        self.sample_size = sample_size
        self.sample_impacts = None
        self.sample_characterized_inventories = None
        self.sensitivity_indices = None

    def _compute_bounds(self) -> tuple[dict, dict]:
        if_bounds = UncertaintyProcessor.compute_bounds(self.if_metadata_df.T.to_dict(), return_type='dict')
        cf_bounds = UncertaintyProcessor.compute_bounds(self.cf_metadata_df.T.to_dict(), return_type='dict')
        return if_bounds, cf_bounds

    def define_problem(self) -> tuple[dict, dict]:
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

    def _compute_env_cost(self, sample_data_if, sample_data_cf) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Compute the environmental costs $Q \cdot B$ by reindexing the chracterization factors sample based on the intervention flow sample, so we can do dot product between for each characterization factors corresponding to the intervnetnion flow
        level_index_if= pd.DataFrame.from_records(sample_data_if.columns.values)
        sample_data_cf_expanded = sample_data_cf.reindex(level_index_if[0].values, axis=1)
        sample_data_if.columns = level_index_if[0].values
        sample_env_cost = sample_data_cf_expanded * sample_data_if
        sample_env_cost.columns = level_index_if[1].values
        return sample_env_cost, level_index_if
    
    def _compute_env_impact(self, sample_env_cost):
        # Compute the environmental impact using a dot product of the reindex scaling vector
        # Set the columns values to match the intervention columns
        scaling_vector_expanded = self.result_data['scaling_vector']['Value'].reindex(sample_env_cost.columns)
        sample_characterized_inventories = sample_env_cost * scaling_vector_expanded
        sample_impacts = sample_env_cost @ scaling_vector_expanded
        return sample_characterized_inventories, sample_impacts
    
    def run_model(self, sample_data_if:pd.DataFrame, sample_data_cf:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        sample_env_cost, level_index_if = self._compute_env_cost(sample_data_if, sample_data_cf)
        sample_characterized_inventories, sample_impacts = self._compute_env_impact(sample_env_cost)
        # Since multiindex columns are to slow to compute, rename the characterized inventory columns again after they have been computed to match the inventory flows indices
        sample_characterized_inventories.columns = pd.MultiIndex.from_frame(level_index_if)
        print(f'The statistics of the the sample impacts: {self.method}') # ATTN: if we have more than one method in the results data this might become a problem
        print(sample_impacts.sparse.to_dense().describe())
        print('The deterministic impact is {}'.format('\n'.join(['{} : {:e}'.format(values[0], values[1]) for values in self.result_data['impacts'].values])))
        # Show the z-value and the distribution of the output
        sample_impacts.plot.hist(bins=50)
        print(sample_impacts.shape)
        # The z-value of the total environmental impact
        print('the z-value of the total impact: {}'.format(sample_impacts.sparse.to_dense().std()/abs(sample_impacts.mean())))
        return sample_impacts, sample_characterized_inventories

    def analyze(self, problem:dict, sample_impacts:pd.DataFrame):
        sensitivity_indices = self.analyser.analyze(problem, sample_impacts.sparse.to_dense().values, parallel=True)
        # total_Si, first_Si, second_Si = sensitivity_indices.to_df()
        total_Si = pd.DataFrame([sensitivity_indices['ST'].T, sensitivity_indices['ST_conf'].T], index=['ST', 'ST_conf'], columns=problem['names']).T
        # Calculate total explained variance
        print("The total explained variance is \n{:.4}%".format(total_Si["ST"].sum()*100))
        return total_Si

    def generate_Si_metadata(self,  all_bounds_indx_dict:dict, total_Si:pd.DataFrame) -> pd.DataFrame:
        # Generate the data and the names for the contribution plot
        metadata_dict = {}
        for (intervention_index, process_index) in total_Si.index[:all_bounds_indx_dict['cf_start']]:
            metadata_dict[(intervention_index, process_index)] = '{} --- {}'.format(self.lci_data['process_map_metadata'][process_index], self.lci_data['intervention_map_metadata'][intervention_index])
        for intervention_index in total_Si.index[all_bounds_indx_dict['cf_start']:]:
            metadata_dict[intervention_index] = '{} --- {}'.format(self.lci_data['intervention_map_metadata'][intervention_index], self.method)
        total_Si_metadata = pd.DataFrame([metadata_dict], index=['bar_names']).T
        return total_Si_metadata

    def plot_top_total_sensitivity_indices(self, total_Si:pd.DataFrame, total_Si_metadata:pd.DataFrame, top_amount:int=10) -> tuple[pd.Series, pd.Series]:
        # Plot the contribution to variance
        top_total_Si = total_Si.sort_values('ST', ascending=False).iloc[:top_amount,:]
        top_total_Si_metadata = total_Si_metadata.loc[top_total_Si.index]
        colormap_base = mpl.cm.tab20.colors
        colormap_SA_barplot = pd.Series(colormap_base[:top_total_Si.shape[0]], index=top_total_Si.index)
        plot_contribution_barplot_with_err(data=top_total_Si, metadata=top_total_Si_metadata, colormap=colormap_SA_barplot, bbox_to_anchor_center=1.7, bbox_to_anchor_lower=-.6)
        return colormap_base, colormap_SA_barplot
        
    def plot_total_env_impact_contribution(self, sample_characterized_inventories:pd.DataFrame, total_Si_metadata:pd.DataFrame, top_amount:int=10, colormap_base:pd.Series=pd.Series([]), colormap_SA_barplot:pd.Series=pd.Series([])) -> None:
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

    Responsibilities:
      - transform_to_normal: convert to Normal(mu, sigma^2) (Section 6.2, cell 181)
      - compute_cost_variance: compute σ_X (Section 6.3, code around cell 198)
      - compute_bounds_variance: stub for future bounds CC
      - apply_chance_constraints: inject L1/L2 constraints into Pyomo model (Section 6.3)
      - solve_pareto: generate Pareto front via ε‑constraint or adaptive sampling (Notebook §1.3 in LCO_math.md)
    """
    def __init__(self,
                 # ATTN: Maybe make the metadata_df quasi arguments with zeros as default value, to allow different formulations
                 cf_metadata_df: pd.DataFrame,
                 if_metadata_df: pd.DataFrame,
                 # ATTN: Add bounds uncertainty here probably
                 pulpo_worker,
                 method:str,
                 choices:dict,
                 demand:dict,
                 ):
        """
        uncertainty data and pulpo worker
        """
        self.cf_metadata_df = cf_metadata_df
        self.if_metadata_df = if_metadata_df
        self.method = method
        self.pulpo_worker = pulpo_worker
        self.choices = choices
        self.demand = demand
        self.formulate()
    
    def formulate(self) -> None:
        pass

    def update_problem(self, lambda_level:float) -> None:
        pass

class CCFormulationIndividualNormalL1(CCFormulationBase):
    """
    This class creates the formulation for linear individual CC formulation of the LCO problem with normal distributed uncertain parameters.

    It assumes L1 Norm for the joint standard distributions of the intervention flows and characterization factors.
    """

    def formulate(self):
        cf_normal_metadata_df, if_normal_metadata_df = self.transform_to_normal()
        self.envcost_std = self.compute_envcost_variance(cf_normal_metadata_df, if_normal_metadata_df)
        self.envcost_mean = self.compute_envcost_mean()
        self.check_envcost_variance(self.envcost_std)


    def transform_to_normal(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform raw uncertainties to Normal distributions:
        """
        cf_normal_metadata_df = UncertaintyProcessor.fit_normals(self.cf_metadata_df)
        if_normal_metadata_df = UncertaintyProcessor.fit_normals(self.if_metadata_df)
        # ATTN: Add the fit_normals for the bound uncertainty
        # ATTN: Check if the fit_normals runs through with 0 as standard deviations
        return cf_normal_metadata_df, if_normal_metadata_df
        

    def _extract_process_ids_and_intervention_flows_for_env_cost_variance(self) -> tuple[array.array, pd.DataFrame]:
        # To Compute the variance of the environmental costs we must extract all processes which contain:
        # - an uncertain intervention flow
        process_id_uncertain_if = self.if_metadata_df.index.get_level_values(1).values
        # - an intervention flow associated with an uncertain characterization factor
        process_id_associated_cf = self.pulpo_worker.lci_data['intervention_matrix'][self.cf_metadata_df.index,:].nonzero()[1]
        process_ids = np.unique(np.append(process_id_associated_cf, process_id_uncertain_if))
        # Get the intervention flows to the uncertain characterization factors
        intervention_flows_extracted = pd.DataFrame.sparse.from_spmatrix(
            self.pulpo_worker.lci_data['intervention_matrix'][self.cf_metadata_df.index.values,:][:,process_ids],
            index=self.cf_metadata_df.index,
            columns=process_ids
        )
        return process_ids, intervention_flows_extracted


    def compute_envcost_variance(self, cf_normal_metadata_df, if_normal_metadata_df) -> pd.Series:
        """
        Compute the standard deviation of the environmental costs using the uncertain intervention flows and characterization factors
        
        $$
        \sigma_{q_hb_j} =\sqrt{\sum_e \big(\mu_{q_{h,e}}^2\sigma_{b_{e,j}}^2 + \mu_{b_{e,j}}^2\sigma_{q_{h,e}}^2 + \sigma_{b_{e,j}}^2 \sigma_{q_{h,e}}^2\big)}
        $$
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
        # Compute the mean of the environmental costs to be used together with the standard deviation to update the uncertain parameters in line with chance constraint formulation
        envcost_raw = self.pulpo_worker.lci_data['matrices'][self.method].diagonal() @ self.pulpo_worker.lci_data['intervention_matrix']
        envcost_mean = pd.Series(envcost_raw).to_dict()
        return envcost_mean

    def check_envcost_variance(self, envcost_std:dict):
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
        ppf_lambda_QB = scipy.stats.norm.ppf(lambda_env_cost)
        environmental_cost_updated = {(process_id, self.method): self.envcost_mean[process_id] + ppf_lambda_QB * self.envcost_std[process_id] for process_id in self.envcost_std.keys()}
        self.pulpo_worker.instance.ENV_COST_MATRIX.store_values(environmental_cost_updated, check=True)

class CCFormulationIndividualNormalL2(CCFormulationBase):

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
    Abstract base for Pareto front solvers.
    Currently only using one Lambda level for all chance constraints.
    """
    def __init__(self, cc_formulation:CCFormulationBase):
        self.cc_formulation = cc_formulation

    def solve_single_pareto_point(self, lambda_level):
        self.cc_formulation.update_problem(lambda_level)
        self.cc_formulation.pulpo_worker.solve()
        result_data = self.extract_results()
        return result_data

    def extract_results(self):
        result_data = pulpo.saver.extract_results(self.cc_formulation.pulpo_worker.instance, self.cc_formulation.pulpo_worker.project, self.cc_formulation.pulpo_worker.database, self.cc_formulation.choices, {}, self.cc_formulation.demand,
                                        self.cc_formulation.pulpo_worker.lci_data['process_map'], self.cc_formulation.pulpo_worker.lci_data['process_map_metadata'],
                                        self.cc_formulation.pulpo_worker.lci_data['intervention_map'], self.cc_formulation.pulpo_worker.lci_data['intervention_map_metadata']) # ATTN: this should be wrapped in the pulpo module similar to the save_results method
        return result_data
    
    def compare_subsequent_paretosolutions(self, result_data_CC):
        impacts = {}
        print(self.cc_formulation.method)
        for lambda_QB, result_data in result_data_CC.items():
            impacts[lambda_QB] = result_data['impacts'].set_index('Key').loc[self.cc_formulation.method,'Value']
            print('{}: {}'.format(lambda_QB, impacts[lambda_QB]))
        # The changs in the choices of the optimizer
        choices_results = {}
        for i_CC, (lambda_QB, result_data) in enumerate(result_data_CC.items()):
            for choice in self.cc_formulation.choices.keys():
                if i_CC == 0:
                    choices_results[choice] = result_data['choices'].xs(tuple(self.cc_formulation.choices.keys()), axis=1)[['Process', 'Capacity']].dropna()
                choices_results[choice] = choices_results[choice].join(result_data['choices'].xs(tuple(self.cc_formulation.choices.keys()), axis=1)['Value'].rename(lambda_QB), how='left')
        for choice, choice_result in choices_results.items():
            print(choice)
            print(choice_result)

        # Changes in the scaling vector and the characterized and scaled inventories
        lambda_array = list(result_data_CC.keys())
        for lambda_1, lambda_2 in zip(lambda_array[:len(lambda_array)-1], lambda_array[1:len(lambda_array)]):
            print(f'lambda_1: {lambda_1}\nlambda_2: {lambda_2}\n')
            scaling_vector_diff = ((result_data_CC[lambda_1]['scaling_vector'].set_index('ID')['Value'] - result_data_CC[lambda_2]['scaling_vector'].set_index('ID')['Value']))
            scaling_vector_ratio = (scaling_vector_diff / result_data_CC[lambda_1]['scaling_vector'].set_index('ID')['Value']).abs().sort_values(ascending=False)
            environmental_cost_mean = {env_cost_index[0]: env_cost for env_cost_index, env_cost in result_data_CC[lambda_1]['ENV_COST_MATRIX']['ENV_COST_MATRIX'].items()}
            characterized_scaling_vector_diff = (scaling_vector_diff * pd.Series(environmental_cost_mean).reindex(scaling_vector_diff.index)).abs()
            characterized_scaling_vector_diff_relative = (characterized_scaling_vector_diff / result_data_CC[lambda_1]['impacts'].set_index('Key').loc[self.cc_formulation.method, 'Value']).abs().sort_values(ascending=False)

            print('Amount of process scaling variables that changed:\n{}: >1% \n{}: >10%\n{}: >100%\n{}: >1000%\n'.format((scaling_vector_ratio > 0.01).sum(), (scaling_vector_ratio > 0.1).sum(), (scaling_vector_ratio > 1).sum(), (scaling_vector_ratio > 10).sum()))
            print('Amount of process characterized scaling variables (impacts per process) that changed:\n{}: >1% \n{}: >10%\n{}: >100%\n{}: >1000%\n'.format((characterized_scaling_vector_diff_relative > 0.01).sum(), (characterized_scaling_vector_diff_relative > 0.1).sum(), (characterized_scaling_vector_diff_relative > 1).sum(), (characterized_scaling_vector_diff_relative > 10).sum()))
            print('{:.5e}: is the maximum impact change in one process\n{:.5e}: is the total impact change\n'.format(characterized_scaling_vector_diff_relative.max(), characterized_scaling_vector_diff_relative.sum()))

            amount_of_rows_for_visiualization = 10
            print('The relative change of the scaling vector (s_lambda_1 - s_lambda_2)/s_lambda_1:\n')
            print(scaling_vector_ratio.iloc[:amount_of_rows_for_visiualization].rename(result_data_CC[lambda_2]['scaling_vector'].set_index('ID')['Process metadata']).sort_values(ascending=False))
            print('\n---\n')
            print('The relative change of the characterized scaling vector (s_lambda_1 - s_lambda_2)*QB_s / QBs:\n')
            print(characterized_scaling_vector_diff_relative.iloc[:amount_of_rows_for_visiualization].rename(result_data_CC[lambda_2]['scaling_vector'].set_index('ID')['Process metadata']))
            print('\n---\n')

    def plot_pareto_front(self, result_data_CC:dict, cutoff_value:float):
        data_QBs_list = []
        for lamnda_QBs, result_data in result_data_CC.items():
            environmental_cost_mean = {env_cost_index[0]: env_cost for env_cost_index, env_cost in result_data_CC[lamnda_QBs]['ENV_COST_MATRIX']['ENV_COST_MATRIX'].items()}
            QBs = result_data['scaling_vector'].set_index('ID')['Value'] * pd.Series(environmental_cost_mean).reindex(result_data['scaling_vector']['Value'].index)
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
    See epsilon constraint derivation in the mathematical fomulation
    """
    def solve(self, lambda_epislons: array.array) -> dict: # ATTN create types the result data and add in dict here
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

