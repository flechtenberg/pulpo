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
from pulpo.utils.saver import ResultDataDict

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

def plot_top_total_sensitivity_indices(total_Si:pd.DataFrame, total_Si_metadata:pd.DataFrame, top_amount:int=10) -> tuple[pd.Series, pd.Series]:
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
        colormap_base = mpl.colormaps['tab20'].colors
        colormap_SA_barplot = pd.Series(colormap_base[:top_total_Si.shape[0]], index=top_total_Si.index)
        plot_contribution_barplot_with_err(data=top_total_Si, metadata=top_total_Si_metadata, colormap=colormap_SA_barplot, bbox_to_anchor_center=1.7, bbox_to_anchor_lower=-.6)
        return colormap_base, colormap_SA_barplot


        
def plot_total_env_impact_contribution(
        sample_characterized_inventories: pd.DataFrame,
        total_Si_metadata: pd.DataFrame,
        method: str,
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
        method (str):
            LCIA method used.
        top_amount (int): 
            Number of top processes to include.
        colormap_base (pd.Series): 
            Base colormap mapping (optional).
        colormap_SA_barplot (pd.Series): 
            Sensitivity-plot colormap mapping (optional).

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
        colormap_base = pd.Series(mpl.colormaps['tab20'].colors[:data_plot.shape[0]], index=data_plot.index)
    plot_linked_contribution_barplot(data_plot, metadata=metadata_plot, impact_category=method, colormap_base=colormap_base, colormap_linked=colormap_SA_barplot, savefig=False, bbox_to_anchor_center=1.7, bbox_to_anchor_lower=-.6)
    return data_plot

def create_data_for_plots(result_data_CC:Dict[float,ResultDataDict], cutoff_value:float, process_map_metadata:dict) -> pd.DataFrame:
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
        data_QBs_main_df = data_QBs_main_df.rename(index={process_id: process_map_metadata[process_id] for process_id in data_QBs_main_df.index})
        return data_QBs_main_df

def plot_pareto_front(result_data_CC:Dict[float,ResultDataDict], cutoff_value:float, method:str, process_map_metadata:dict, bbox_to_anchor:Tuple[float, float] = (0.65, -1.)):
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
        bbox_to_anchor (tuple): 
            Tuple holding the bbox anchor points for the legend.
            Default value is (0.65, -1.).
    """
    data_QBs_main_df = create_data_for_plots(result_data_CC, cutoff_value, process_map_metadata)
    plot_pareto_solution_normalized_bar_plots(data_QBs_main_df, method, bbox_to_anchor=bbox_to_anchor)
    plot_pareto_solution_bar_plots(data_QBs_main_df, method, bbox_to_anchor=bbox_to_anchor)

# === General Plots ===

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

def plot_pareto_solution_normalized_bar_plots(data:pd.DataFrame, y_label:str, bbox_to_anchor:tuple=(1.40, .05)):
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
    """
    
    width = 6*72.4
    height = None
    _, ax = plt.subplots(1, 1, figsize=set_size(width,height))
    # Normalize the data and create bar plot data format
    data_cleaned = data.copy()
    data_cleaned_scaled = data_cleaned.abs().divide(data_cleaned.abs().sum())
    data_cumsum = data_cleaned_scaled.cumsum(axis=0)
    # Set the bar plot style
    bar_width = .8
    labels = ["{:.3f}".format(label) for label in data.columns.astype(float).values]
    bottom_data = np.zeros(len(labels))
    # Plot the bars
    for i_row, (type, row_data) in enumerate(data_cumsum.iterrows()):
        ax.bar(labels, row_data.values-bottom_data, bar_width, bottom=bottom_data, label=type, color=mpl.cm.tab20.colors[i_row])
        bottom_data = row_data.values
    ax.axhline(y=0, color='k')
    ax.set_xlabel("probability level ($\lambda$)")
    ax.set_ylabel("{} in [\%]".format(y_label))
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    # Plot the Pareto front 
    ax2 = ax.twinx()
    ax2.plot(labels, data.sum().values/1e9, "kx-", label="total GWP", linewidth=1)
    ax2.set_ylabel(y_label)
    # Set the legend 
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='lower center', bbox_to_anchor=bbox_to_anchor, borderpad=1, facecolor="None")
    ax.set_facecolor("None")

def plot_pareto_solution_bar_plots(data:pd.DataFrame, y_label:str, bbox_to_anchor:tuple=(1.40, .05), save_fig_name:str=""):
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
    """
    
    # set figure and plot
    width = 6*72.4
    height = None
    _, ax = plt.subplots(1, 1, figsize=set_size(width,height))

    data_cleaned = data.copy()
    data_cleaned[data.abs() < data.abs().sum()/500] = 0
    data_cleaned = data_cleaned.drop(index = data_cleaned.index[(data_cleaned == 0).all(axis=1)])
    data_positive_cumsum = data_cleaned[(data_cleaned>=0).all(axis=1)].cumsum(axis=0)
    width = .8
    labels = ["{:.3f}".format(label) for label in data.columns.astype(float).values]
    bottom_data = np.zeros(len(labels))
    for i_row, (type, row_data) in enumerate(data_positive_cumsum.iterrows()):
        ax.bar(labels, row_data.values-bottom_data, width, bottom=bottom_data, label=type, color=mpl.cm.tab20.colors[i_row])
        bottom_data = row_data.values
    data_negative_cumsum = data_cleaned[(data_cleaned<=0).all(axis=1)].cumsum(axis=0)
    bottom_data = np.zeros(len(labels))
    for i_row, (type, row_data) in enumerate(data_negative_cumsum.iterrows()):
        ax.bar(labels, row_data.values-bottom_data, width, bottom=bottom_data, label=type, color=mpl.cm.tab20.colors[i_row+data_positive_cumsum.shape[0]])
        bottom_data = row_data.values
    ax.plot(labels, data.sum().values, "kx-", markersize=6, linewidth=1.)
    ax.axhline(y=0, color='k')
    plt.xticks(rotation = 45) 
    ax.legend(loc='lower center', bbox_to_anchor=bbox_to_anchor, borderpad=1)

    ax.set_xlabel("Pareto points, represented by alpha")
    ax.set_ylabel(y_label)
    # if save_fig_name != "":
    #     plt.savefig(r"C:\Users\admin\OneDrive - Carbon Minds GmbH\Dokumente\13 Students\MA_Bartolomeus_Löwgren\02_code\03_optimization_framework\04_case_studies\02_plots" + "\{}.{}".format(save_fig_name, fileformat), format=fileformat, bbox_inches='tight')



