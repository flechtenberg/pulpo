"""
plots.py

Module that contains the plots used in the uncertainty modules.
It is split up in "Plots Wrapper" and "General Plots"
which are the methods that prepare the data and then call 
a general plot method.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import pandas as pd
import numpy as np
import matplotlib as mpl
import textwrap
from typing import Union, List, Optional, Dict, Tuple, Literal
from pulpo.utils.saver import ResultDataDict
import seaborn as sns

# === Plots Wrapper ===

def plot_top_characterized_processes(   
        process_map_metadata:dict[int, str],
        characterized_inventory:scipy.sparse.sparray, 
        method:str, 
        top_amount:int=10
        ):
    """
    Plot the top-N contributing processes or parameters as a bar chart.

    Args:
        process_map_metadata (dict):
            Dictionary mapping the process index (keys) to the metadata, i.e., description (values)
        characterized_inventory (scipy.sparse.sparray): 
            B·(Q·s) for each parameter (impact after characterization).
        method (str):
            The LCIA method used to compute the characterized inventory
        top_amount (int) - optional: 
            Number of top items to display (default: 10).

    Returns:
        None: Displays a matplotlib bar plot of the highest contributors.
    """
    # Plot the highest contributing processes
    impact_df = pd.DataFrame(
        characterized_inventory.sum(axis=0).T,
        index=list(range(characterized_inventory.shape[1])),
        columns=['impact']
    )
    impact_df['process name'] = impact_df.index.map(process_map_metadata)
    impact_df = impact_df.reindex(impact_df['impact'].abs().sort_values(ascending=False).index)
    impact_df_red = impact_df.iloc[:top_amount,:]
    impact_rest = impact_df.iloc[top_amount:,:].sum(numeric_only=True)
    impact_rest['process name'] = 'Rest'
    impact_df_red = pd.concat([impact_df_red, impact_rest.to_frame().T], axis=0)        
    impact_df_red['impact'] = impact_df_red['impact'] / impact_df['impact'].sum()
    colormap = mpl.colormaps['tab20']
    colormap_ser = pd.Series(colormap.colors[:impact_df_red.shape[0]], index=impact_df_red.index)
    plot_contribution_barplot(impact_df_red['impact'], metadata=impact_df_red['process name'], impact_category=method, colormap=colormap_ser,  bbox_to_anchor_center=1.7, bbox_to_anchor_lower=-.6)
    plt.show()

def plot_top_total_sensitivity_indices(total_Si:pd.DataFrame, total_Si_metadata:pd.DataFrame, top_amount:int=10, cmap_name:str='tab20') -> tuple[pd.Series, pd.Series]:
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
        colormap_base = discrete_cmap(top_total_Si.shape[0].shape[0], cmap_name).colors
        colormap_SA_barplot = pd.Series(colormap_base, index=top_total_Si.index)
        plot_contribution_barplot_with_err(data=top_total_Si, metadata=top_total_Si_metadata, colormap=colormap_SA_barplot, bbox_to_anchor_center=1.7, bbox_to_anchor_lower=-.6)
        return colormap_base, colormap_SA_barplot


        
def plot_total_env_impact_contribution(
        sample_characterized_inventories: pd.DataFrame,
        total_Si_metadata: pd.DataFrame,
        method: str,
        top_amount: int = 10,
        colormap_base: pd.Series = pd.Series([]),
        colormap_SA_barplot: pd.Series = pd.Series([]),
        cmap_name: str = 'tab20'
    ) -> pd.DataFrame:
    """
    Plot each process's share of the total environmental impact.

    Args:
        sample_characterized_inventories (pd.DataFrame):
            Characterized inventory flows per sample.
        total_Si_metadata (pd.DataFrame):
            'bar_names' for labeling processes.
        method (str):
            LCIA method used.
        top_amount (int): 
            Number of top processes to include.
        colormap_base (pd.Series): 
            Base colormap mapping (optional).
        colormap_SA_barplot (pd.Series): 
            Sensitivity-plot colormap mapping (optional).
        cmap_name (str): 
            Name of the colormap to use if colormap_base is not provided. Default is 'tab20'.

    Returns:
        data_plot (pd.DataFrame): 
            Data prepared for the linked impact contribution plot.
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
        cmap = discrete_cmap(data_plot.shape[0], cmap_name)
        colormap_base = pd.Series(cmap.colors, index=data_plot.index)
    plot_linked_contribution_barplot(data_plot, metadata=metadata_plot, impact_category=method, colormap_base=colormap_base, colormap_linked=colormap_SA_barplot, savefig=False, bbox_to_anchor_center=1.7, bbox_to_anchor_lower=-.6)
    return data_plot

def create_choices_data_for_CC_plots(result_data_CC:Dict[float,ResultDataDict], group_act_by:Optional[Literal['process', 'product', 'location']]=None) -> Dict[str,pd.DataFrame]:
    # The changs in the choices of the optimizer
    choices_results = {}
    for i_CC, (lambda_QB, result_data) in enumerate(result_data_CC.items()):
        for choice, choice_data in result_data['Choices'].items():
            if i_CC == 0:
                # choices_results[choice] = choice_data[['Capacity']]
                choices_results[choice] = pd.DataFrame(index=choice_data.index)
            # Join the data of each Pareto Point to the dataframe   
            choices_results[choice] = choices_results[choice].join(choice_data['Value'].rename(lambda_QB), how='left')
    # Delete all rows that are all zero
    for choice in choices_results.keys():
        if group_act_by:
            metadata_df = pd.Series(choices_results[choice].index, index=choices_results[choice].index).str.split(' | ', expand=True, regex=False)
            metadata_labels = ['process', 'product', 'location']
            metadata_df.columns = metadata_labels
            choices_results[choice] = choices_results[choice].merge(metadata_df, left_index=True, right_index=True, how='left')
            choices_results[choice].set_index(metadata_labels, inplace=True)
            # Group by product to see the contribution of each product group
            choices_results[choice] = choices_results[choice].groupby(group_act_by).sum()        
        choices_results[choice] = choices_results[choice].loc[~(choices_results[choice]==0).all(axis=1)]
    return choices_results

def create_data_for_CC_plots(result_data_CC:Dict[float,ResultDataDict], cutoff_value:float, process_map_metadata:dict, group_act_by:Optional[Literal['process', 'product', 'location']]=None) -> pd.DataFrame:
        """
        Create the data for the Pareto front plots, by computing the process impacts and 
        selecting the top processes per Pareto Point based on the cut off and then concatting
        all main contributing processes across Pareto Points to show changes.

        Args:
            result_data_CC (Dict[float,ResultDataDict]): Mapping from each lambda level
                to its corresponding solver result dictionary.
            cutoff_value (float): Relative threshold for filtering main decision variables
                to include in the bar plot.
            process_map_metadata (dict):
                Dictionary mapping the process index (keys) to the metadata, i.e., description (values)
                from `lci_data['process_map_metadata']`
            group_act_by (str, optional):
                If specified, groups the activities by the given metadata field.
                Options are 'process', 'product', or 'location'.
                Default is None, which means no grouping.

        Return:
            data_QBs_main_df (pd.DataFrame): 
                Containing the main contributing processes to the impact per Pareto Point, 
                returned from `create_data_for_plots` method.
        """
        data_QBs_main_list = []
        data_QBs_list = []
        for lamnda_QBs, result_data in result_data_CC.items():
            environmental_cost_mean = {env_cost_index[0]: env_cost['Value'] for env_cost_index, env_cost in result_data_CC[lamnda_QBs]['ENV_COST_MATRIX'].iterrows()}
            QBs = result_data['Scaling Vector']['Value'] * pd.Series(environmental_cost_mean).reindex(result_data['Scaling Vector']['Value'].index)
            # data_QBs_list.append(QBs)
            QBs.name = lamnda_QBs  
            data_QBs_list.append(QBs)
            QBs_main = QBs[QBs.abs() > cutoff_value*QBs.abs().sum()].sort_values(ascending=False)
            data_QBs_main_list.append(QBs_main)
            print('With a cutoff value of {}, we keep {} process to an error of {:.2%}'.format(cutoff_value, len(QBs_main), abs(1 - QBs_main.sum()/QBs.sum())))
        data_QBs_df = pd.concat(data_QBs_list, axis=1)
        # Take all main contributing processes across all Pareto Points
        main_processes = pd.Index([process_id for data_QBs_main in data_QBs_main_list for process_id in data_QBs_main.index]).unique()
        data_QBs_main_df = data_QBs_df.loc[main_processes]
        # Comput the remaining impacts as "rest" and "negative rest"
        data_QBs_rest = data_QBs_df.drop(index=main_processes)
        data_QBs_main_df.loc['negative rest'] = data_QBs_rest[data_QBs_rest<0].sum()
        data_QBs_main_df.loc['positive rest'] = data_QBs_rest[data_QBs_rest>0].sum()        
        # Rename the index to contain the main contributing processes.
        if group_act_by:
            process_map_metadata_df = pd.Series(process_map_metadata).str.split(' | ', expand=True, regex=False)
            metadata_labels = ['process', 'product', 'location']
            process_map_metadata_df.columns = metadata_labels
            data_QBs_main_df = data_QBs_main_df.merge(process_map_metadata_df, left_index=True, right_index=True, how='left')
            data_QBs_main_df.loc['negative rest', metadata_labels] = ['negative rest']*len(metadata_labels)
            data_QBs_main_df.loc['positive rest', metadata_labels] = ['positive rest']*len(metadata_labels)
            data_QBs_main_df.set_index(metadata_labels, inplace=True)
            # Group by product to see the contribution of each product group
            data_QBs_main_df = data_QBs_main_df.groupby(group_act_by).sum()
            print('by grouping the processes by product, we reduce the number of variables to {}'.format(data_QBs_main_df.shape[0]))
        else:
            process_map_updated = process_map_metadata.copy()
            process_map_updated.update({'negative rest': 'negative rest', 'positive rest': 'positive rest'})
            data_QBs_main_df = data_QBs_main_df.rename(index={process_id: process_map_updated[process_id] for process_id in data_QBs_main_df.index})
        return data_QBs_main_df

def create_cmap_for_CC_plots(CC_bar_plot_labels:pd.Index, choices_labels:Optional[pd.Index]=None, cmap_name:str='tab20', create_legend:bool=False) -> dict|Tuple[dict,List[mpl.patches.Patch]]:
    """Create a colormap for the CC plots based on the main contributing processes.

    Args:
        CC_bar_plot_labels (pd.Index): Index containing the main contributing processes.
        cmap_name (str): Name of the colormap to use.

    Returns:
        cmap (dict): Dictionary mapping process IDs to their corresponding colors.
        legend_elements (list): List of legend elements for the colormap (if create_legend is True).
    """
    if choices_labels is not None:
        all_labels = CC_bar_plot_labels.union(choices_labels).unique()
    else:
        all_labels = CC_bar_plot_labels.unique()
    # Create metadata out of the index for better color grouping
    if pd.Series(all_labels, index=all_labels).str.contains('|', regex=False).any():
        all_labels_metadata = pd.Series(all_labels, index=all_labels).str.split(' | ', expand=True, regex=False)
        all_labels_metadata.columns = ['process', 'product', 'location']
        # Sort by product to have similar colors for similar products
        all_labels_metadata = all_labels_metadata.sort_values(by='product')
    else:
        all_labels_metadata = pd.Series(all_labels, index=all_labels).sort_index()
    cmap_list = sns.color_palette(cmap_name, n_colors=len(all_labels_metadata.index))  # a list of RGB tuples
    cmap = {idx: cmap_val for idx, cmap_val in zip(all_labels_metadata.index, cmap_list)}
    # cmap_list = discrete_cmap(l en(all_labels), base_cmap=cmap_name)
    # cmap = {idx: cmap_val for idx, cmap_val in zip(all_labels, cmap_list.colors)}
    if create_legend:
        legend_elements = [mpl.patches.Patch(facecolor=color, label=label) for label, color in cmap.items()]
    return cmap, legend_elements

def plot_pareto_front(
        result_data_CC:Dict[float,ResultDataDict], 
        cutoff_value:float, 
        method:str, 
        process_map_metadata:dict,
        rel_abs:Literal['relative', 'absolute'] = 'absolute',
        bbox_to_anchor:Tuple[float, float] = (0.65, -1.),
        cmap_name:str = 'gist_ncar',
        group_act_by:Optional[Literal['process', 'product', 'location']]=None
        ):
    """
    Plot the Pareto front and highlight main contributing variables.

    Args:
        result_data_CC (Dict[float,ResultDataDict]): Mapping from each lambda level
            to its corresponding solver result dictionary.
        cutoff_value (float): Relative threshold for filtering main decision variables
            to include in the bar plot.
        method (str):
            The LCIA method used to compute the characterized inventory
        process_map_metadata (dict):
                Dictionary mapping the process index (keys) to the metadata, i.e., description (values)
                from `lci_data['process_map_metadata']`
        rel_abs (Literal['relative', 'absolute']): 
            Specifies whether to plot relative or absolute contributions. Default is 'absolute'.
            'relative' normalizes contributions to [0,1], while 'absolute' shows actual values.
        bbox_to_anchor (tuple): 
            Tuple holding the bbox anchor points for the legend.
            Default value is (0.65, -1.).
        cmap_name (str):
            Name of the colormap to use for the plots. Default is 'gist_ncar'.
    """
    data_QBs_main_df = create_data_for_CC_plots(result_data_CC, cutoff_value, process_map_metadata, group_act_by=group_act_by)
    choices_data = create_choices_data_for_CC_plots(result_data_CC, group_act_by=group_act_by)
    choices_labels = pd.Index([idx for choice in choices_data.keys() for idx in choices_data[choice].index])
    cmap, legend_elements = create_cmap_for_CC_plots(data_QBs_main_df.index, choices_labels=choices_labels, cmap_name=cmap_name, create_legend=True)
    match rel_abs:
        case 'relative':
            _, axs = plt.subplots(len(choices_data.keys())+1, 1, figsize=(6,6), dpi=300, sharex=True, gridspec_kw={'hspace': 0.1}, height_ratios= [8]+[1]*len(choices_data.keys()))
            plot_pareto_solution_normalized_bar_plots(data_QBs_main_df, method, bbox_to_anchor=bbox_to_anchor, cmap_name=cmap_name, cmap=cmap, ax=axs[0], legend_elements=legend_elements)
            plot_choices_pareto_solutions_bar_plots(choices_data, cmap=cmap, shared_xaxis=axs[1:])
        case 'absolute':
            _, axs2 = plt.subplots(len(choices_data.keys())+1, 1, figsize=(6,6), dpi=300, sharex=True, gridspec_kw={'hspace': 0.1}, height_ratios= [8]+[1]*len(choices_data.keys()))
            plot_pareto_solution_bar_plots(data_QBs_main_df, method, bbox_to_anchor=bbox_to_anchor, cmap_name=cmap_name, cmap=cmap, ax=axs2[0], legend_elements=legend_elements)
            plot_choices_pareto_solutions_bar_plots(choices_data, cmap=cmap, shared_xaxis=axs2[1:])
# === General Plots ===

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = mpl.colormaps.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    return plt.cm.colors.ListedColormap(color_list, color_list, N)

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
    width = 6*72.4
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
    width = 6*72.4
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

def plot_pareto_solution_normalized_bar_plots(data:pd.DataFrame, y_label:str, bbox_to_anchor:tuple=(1.40, .05), cmap_name:str='gist_ncar', cmap:Optional[Dict]=None, ax:Optional[mpl.axes.Axes]=None,legend_elements:Optional[List[mpl.patches.Patch]]=None) -> None:
    """
    Plots normalized bar plots [0-1] stacked by the rows in the `data` dataframe and bars based on the columns.
    Normalized the bars based on the sum of each column.

    args:
        data (pd.DataFrame):
            Dataframe with: columns as Pareto solutions, e.g., Lambdas, rows: impacts of groups or processes
        y_label (str):
            Name of the variable in in the values in the dataframe, e.g., impact category
        bbox_to_anchor (tuple): 
                Tuple holding the bbox anchor points for the legend.
                Default value is (1.40, .05).
        cmap_name (str):
            Name of the colormap to use for the bars. Default is 'gist_ncar'.
        cmap (Optional[Dict]):
            Colormap to use for the bars. If None, a default colormap will be used.
        ax (Optional[mpl.axes.Axes]):
            Matplotlib Axes to plot on. If None, a new figure and axes will be created.
        legend_elements (Optional[List[mpl.patches.Patch]]):
            List of legend elements to use for the legend. If None, no legend will be added.
    """
    
    width = 6*72.4
    height = None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=set_size(width,height), gridspec_kw={'hspace': 0.3})
    # Normalize the data and create bar plot data format
    data_cleaned = data.copy()
    data_cleaned_scaled = data_cleaned.abs().divide(data_cleaned.abs().sum())
    data_cumsum = data_cleaned_scaled.cumsum(axis=0)
    # Set the bar plot style
    bar_width = .8
    labels = ["{:.3f}".format(label) for label in data.columns.astype(float).values]
    bottom_data = np.zeros(len(labels))
    if not cmap:
        cmap_list = discrete_cmap(data_cumsum.shape[0], cmap_name)
        cmap = {idx: cmap_val for idx, cmap_val in zip(data_cumsum.index, cmap_list.colors)}
    # Plot the bars
    for type, row_data in data_cumsum.iterrows():
        ax.bar(labels, row_data.values-bottom_data, bar_width, bottom=bottom_data, label=type, color=cmap[type])
        bottom_data = row_data.values
    ax.axhline(y=0, color='k')
    ax.set_xlabel("probability level ($\lambda$)")
    ax.set_ylabel("relative {}".format(y_label))
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    # Plot the Pareto front 
    ax2 = ax.twinx()
    ax2.plot(labels, data.sum().values/1e9, "kx-", label="total GWP", linewidth=1)
    ax2.set_ylabel("{}".format(y_label))
    # Set the legend
    lines2, labels2 = ax2.get_legend_handles_labels()
    if legend_elements:
        ax.legend(handles=legend_elements+lines2, loc='lower center', bbox_to_anchor=bbox_to_anchor, borderpad=1) 
    else:
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='lower center', bbox_to_anchor=bbox_to_anchor, borderpad=1, facecolor="None")
    ax.set_facecolor("None")

def plot_pareto_solution_bar_plots(data:pd.DataFrame, y_label:str, bbox_to_anchor:tuple=(1.40, .05), save_fig_name:str="", cmap_name:str='gist_ncar', cmap:Optional[Dict]=None, ax:Optional[mpl.axes.Axes]=None, legend_elements:Optional[List[mpl.patches.Patch]]=None) -> None:
    """
    args:
        data (pd.DataFrame):
            Dataframe with: columns as Pareto solutions, e.g., Lambdas, rows: impacts of groups or processes
        y_label (str):
            Name of the variable in in the values in the dataframe, e.g., impact category
        bbox_to_anchor (tuple): 
            Tuple holding the bbox anchor points for the legend.
            Default value is (1.40, .05).
        save_fig_name (str):
            Currently not in use
        cmap_name (str):
            Name of the colormap to use for the bars. Default is 'gist_ncar'.
        cmap (Optional[Dict]):
            Colormap to use for the bars. If None, a default colormap will be used.
    """
    
    # set figure and plot
    width = 6*72.4
    height = None
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=set_size(width,height))
    data_cleaned = data.copy()
    data_cleaned[data.abs() < data.abs().sum()/500] = 0
    data_cleaned = data_cleaned.drop(index = data_cleaned.index[(data_cleaned == 0).all(axis=1)])
    data_positive_cumsum = data_cleaned[(data_cleaned>=0).all(axis=1)].cumsum(axis=0)
    if not cmap:
        cmap_list = discrete_cmap(data_cleaned.shape[0], cmap_name)
        cmap = {idx: cmap_val for idx, cmap_val in zip(data_cleaned.index, cmap_list.colors)}   
    width = .8
    labels = ["{:.3f}".format(label) for label in data.columns.astype(float).values]
    bottom_data = np.zeros(len(labels))
    for type, row_data in data_positive_cumsum.iterrows():
        ax.bar(labels, row_data.values-bottom_data, width, bottom=bottom_data, label=type, color=cmap[type])
        bottom_data = row_data.values
    data_negative_cumsum = data_cleaned[(data_cleaned<=0).all(axis=1)].cumsum(axis=0)
    bottom_data = np.zeros(len(labels))
    for type, row_data in data_negative_cumsum.iterrows():
        ax.bar(labels, row_data.values-bottom_data, width, bottom=bottom_data, label=type, color=cmap[type])
        bottom_data = row_data.values
    ax.plot(labels, data.sum().values, "kx-", markersize=6, linewidth=1.)
    ax.axhline(y=0, color='k')
    plt.xticks(rotation = 45) 
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=bbox_to_anchor, borderpad=1)
    else:
        ax.legend(loc='lower center', bbox_to_anchor=bbox_to_anchor, borderpad=1)

    ax.set_xlabel("Pareto points, represented by alpha")
    ax.set_ylabel(y_label)
    # if save_fig_name != "":
    #     plt.savefig(r"C:\Users\admin\OneDrive - Carbon Minds GmbH\Dokumente\13 Students\MA_Bartolomeus_Löwgren\02_code\03_optimization_framework\04_case_studies\02_plots" + "\{}.{}".format(save_fig_name, fileformat), format=fileformat, bbox_inches='tight')


def plot_choices_pareto_solutions_bar_plots(
        choices_results:Dict[str,pd.DataFrame], 
        save_fig_name:str="", 
        cmap:Optional[Dict]=None, 
        cmap_name:str='gist_ncar',
        shared_xaxis:Optional[List[mpl.axes.Axes]]=None,
        ):
    """
    args:
        choices_results (Dict[str,pd.DataFrame]):
            Dictionary with the choice name as key and a dataframe as value.
            The dataframe has the Pareto solutions as columns and the decision variables as index.
        save_fig_name (str):
            Currently not in use
        cmap (Optional[Dict]):
            Colormap to use for the bars. If None, a default colormap will be used.
        cmap_name (str):
            Name of the colormap to use for the bars. Default is 'gist_ncar'.
        shared_xaxis (Optional[List[mpl.axes.Axes]]):
            List of axis which the choices subplots will be plotted on. 
            If the figure and axis are already created outside of this method.
            e.g., axs[1:] if the first plot is a Pareto front plot.
            If None, new axis will be created.
    """
    # set figure and plot
    width = 6*72.4
    height = None
    if shared_xaxis is None:
        _, axs = plt.subplots(len(choices_results), 1, figsize=set_size(width,height), sharex=True, sharey=True)
    else:
        axs = shared_xaxis
    if not cmap:
        choices_labels = [idx for choice in choices_results.keys() for idx in choices_results[choice].index]
        cmap_list = discrete_cmap(len(choices_labels), cmap_name)
        cmap = {idx: cmap_val for idx, cmap_val in zip(choices_labels, cmap_list.colors)}   
    for i_choice, (choice, data) in enumerate(choices_results.items()):
        # Normalize the data 
        data_norm = data.abs().divide(data.abs().sum()).fillna(0.)
        # Sort data by first Pareto soltution and create bar plot data format
        data_cumsum = data_norm.sort_values(by=data.columns.tolist()[1]).cumsum(axis=0)
        # Set the bar plot style
        # cmap = discrete_cmap(data_cumsum.shape[0], cmap_name)
        width_bars = 1.5*(plt.gcf().get_size_inches()[0] / data.shape[1])
        labels = ["{:.3f}".format(label) for label in data.columns.astype(float).values]
        bottom_data = np.zeros(len(labels))
        for type, row_data in data_cumsum.iterrows():
            axs[i_choice].bar(labels, row_data.values-bottom_data, width_bars, bottom=bottom_data, label=type, color=cmap[type])#, color=cmap.colors[i_row+data_cumsum.shape[0]])
            # axs[i_choice].bar(np.array(labels, dtype=float)+i_choice*width-width*len(choices_results)/2, row_data.values-bottom_data, width, bottom=bottom_data, label="{}: {}".format(choice, type), color=cmap.colors[i_row])
            bottom_data = row_data.values
        ax2 = axs[i_choice].twinx()
        ax2.plot(labels, data.sum().values, "kx-", markersize=4, linewidth=.5)
        # ax2.set_ylabel("Total {}".format(choice), rotation=0, labelpad=20, va='center', ha='left')
        # Set the font size of the second y-axis to 6 and format to not use scientific notation
        ax2.set_yticks(ax2.get_yticks(), labels=[f"{tick:.3e}" for tick in ax2.get_yticks()], fontsize=6)
        if i_choice == 0:
            axs[i_choice].set_xticks(ticks=axs[i_choice].get_xticks(), labels=["{:.3f}".format(label) for label in data.columns.astype(float).values], rotation=45)
            axs[i_choice].set_xlabel("Pareto points, represented by alpha")
        axs[i_choice].set_yticks([])
        axs[i_choice].set_ylabel(choice, rotation=0, labelpad=30, va='center')
        # axs[i_choice].legend(loc='lower center', bbox_to_anchor=bbox_to_anchor, borderpad=1)
    # if save_fig_name != "":
#     plt.savefig(r"C:\Users\admin\OneDrive - Carbon Minds GmbH\Dokumente\
    # Set shared x-axis and format
    plt.subplots_adjust(wspace=0, hspace=0.1)

