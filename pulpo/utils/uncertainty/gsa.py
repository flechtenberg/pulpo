
"""
gsa.py

Module for the global sensitivity analysis based on SaLib library. 
It currently only looks at sensitivity of an LCA, i.e., an optimal LCO solution.
It also only considers uncertainty in the Biosphere matrix (invervention flows, B)
and in the Characterization matrix (characterization factors, Q).
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
            colormap_base = pd.Series(mpl.cm.tab20.colors[:data_plot.shape[0]], index=data_plot.index)
        plot_linked_contribution_barplot(data_plot, metadata=metadata_plot, impact_category=self.method, colormap_base=colormap_base, colormap_linked=colormap_SA_barplot, savefig=False, bbox_to_anchor_center=1.7, bbox_to_anchor_lower=-.6)
        return data_plot

