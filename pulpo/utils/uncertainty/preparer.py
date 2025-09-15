
"""
preparation.py

utils uncertainty Module which containing the uncertainty functions that are dependent on `pulpo_worker` attributes.
This module filter and imports uncertainty data to then be used for uncertainty asessments.
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

class UncertaintyImporter:
    """
    Extract/Import and index uncertainty metadata for intervention flows and characterization factors.
    """
    def __init__(self, lci_data):
        """
        Initiates uncertainty importer with the LCI data.

        Args:
            lci_data (dict): 
                PULPO LCI data (matrices, maps).
        """
        self.lci_data = lci_data # from pulpo_worker.lci_data

    def get_intervention_indcs_to_db(self, db_name, intervention_indices:List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        fetches the inventory indices to a specified bw database 
        and if specified the intersection to a given list of interventions flow indices

        Args:
            db_name (str): 
                name of the BW database for which the indices are fetched for
            intervention_indices (list): 
                intervention flow indices for which the metadata will be extracted
        
        Returns:
            intervention_indices_in_db (list): 
                The interventions flow indices to the specified BW database and intersection to intervention_indices
        """
        # To increase the speed of the search the process_map and the intervention_indices are transformed into arrays (pandas)
        process_map_df = pd.DataFrame(zip(*self.lci_data['process_map'].keys(), self.lci_data['process_map'].values()), index=['db', 'key']).T
        db_process_indcs = process_map_df.index[process_map_df['db'] == db_name]
        intervention_indices_df = pd.DataFrame(zip(*intervention_indices), index=['intervention_indx', 'process_indx']).T
        intervention_indices_in_db_df = intervention_indices_df[intervention_indices_df['process_indx'].isin(db_process_indcs)]
        intervention_indices_in_db = intervention_indices_in_db_df.to_records(index=False).tolist()
        # db_process_indcs = [process_indx for (db, _), process_indx in self.lci_data['process_map'].items() if db == db_name]
        # intervention_indices_in_db = [(intevention_indx, process_index) for (intevention_indx, process_index) in intervention_indices if process_index in db_process_indcs]
        return intervention_indices_in_db


    def get_if_meta(self, inventory_indices:List[tuple]) -> pd.DataFrame:
        """
        Extract intervention‐flow uncertainty metadata for given indices.
        The data is taken from the `lci_data` object of the `pulpo_worker` instance.
        The link to the pyomo instance is that the parameters are identically indexed 
        in the `lci_data` and pyomo instance.

        Args:
            inventory_indices (List[tuple]): 
                list of intervention flow indices (process_indx, intervention_indx) 
                for which the metadata will be extracted 

        Returns:
            intervention_metadata_df (pd.DataFrame): 
                DataFrame containing the uncertainty information of the inventory flows
                indexed by (row, col).
        """
        intervention_metadata_df = pd.DataFrame(self.lci_data['intervention_params'])
        intervention_metadata_df = intervention_metadata_df.set_index(["row", "col"])
        intervention_metadata_df = intervention_metadata_df.loc[inventory_indices]
        return intervention_metadata_df

    def get_cf_meta(self, method:str, characterization_indices:List[int]) -> pd.DataFrame:
        """
        Extract uncertainty metadata for characterization factors for a LCIA method.
        The data is taken from the `lci_data` object of the `pulpo_worker` instance.
        The link to the pyomo instnace is that the parameters are identically indexed 
        in the `lci_data` and pyomo instance.

        Args:
            method (str): 
                LCIA method key.
            characterization_indices (list[int]): 
                List of characterization row indices.
        
        Return:
            characterization_metadata_df (pd.DataFrame):
                metadata of the characterization factors containing uncertainty information 
                indexed by .
        """
        characterization_params = self.lci_data["characterization_params"][method]
        characterization_metadata_df = pd.DataFrame(characterization_params).set_index('row')
        characterization_metadata_df = characterization_metadata_df.loc[characterization_indices]
        return characterization_metadata_df

    def separate(self, uncertainty_metadata_df:pd.DataFrame) -> tuple[dict, list]:
        """
        Split metadata into defined (type>0) and undefined (type=0) entries.

        Returns:
            defined (dict): 
                dictionary indexed by the parameter index, containing the the uncertainty metadata 
                for parameter which have uncertainty information, 
            undefined (list): 
                list of indices without defined distributions
        """
        defined, undefined = {}, []
        for idx, row in uncertainty_metadata_df.iterrows():
            if row['uncertainty_type'] > 0:
                defined[idx] = row.to_dict()
            else:
                undefined.append(idx)
        print("Parameters with uncertainty information: {} \nParameters without uncertainty information: {}".format(len(defined), len(undefined)))
        return defined, undefined
    
    def get_pyomo_param_meta(self, instance, pyomo_param_name:str, param_indcs:List[int]) -> pd.DataFrame:
        """
        Generates the uncertainty metadata structre for parameters in the pyomo model 
        which are not intervention flows or characterization factors.
        i.e., uncertainty information to parameters not given in Brightway (LCI databases),
        e.g., variable bounds, impact bounds, addtional constraints, etc.
        Since these parameters at the current state of implementation do not have any 
        uncertainty information specified, they will only get the attribute key "amount" and "uncertainty_type"

        The indexing of these parameters are based on the initialized pyomo model.

        Args:
            instance (pyomo.instance):
                The pyomo instance of the case study, used to extract the "amount"
            pyomo_param_name (str):
                Name of the pyomo parameter to which the uncertainty metadata is to be generated for
            param_indcs (list):
                List of the parameter indices for which uncertainty metadata is to be extracted, 
                as indexed in the pyomo instance

        Returns:
            param_metadata_df (pd.DataFrame):
                Dataframe with the param indices as index and "amount" and "uncertainty_type" as columns
            
        """
        param_metadata:dict = {}
        # ATTN: Change this method that it gets the pyomo parameter amount from the dicts used to instantiate the worker
        #   from: upper_limit, lower_limit, upper_elem_limit, upper_imp_limit
        #   These must then be arguments into the method, probably as **kwargs
        #   passed by the wrapper method in the pulpo worker
        param_data = getattr(instance, pyomo_param_name).extract_values()
        for param_indx in param_indcs:
            param_metadata[param_indx] = {}
            param_metadata[param_indx]['amount'] = param_data[param_indx]
            param_metadata[param_indx]['uncertainty_type'] = 0
        param_metadata_df = pd.DataFrame(param_metadata).T
        return param_metadata_df



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

    def apply_filter(self, scaling_vector_strategy:str, cutoff:float, plot_n_top_processes:int=10) -> Tuple[list,list]:
         """
         Applies the filtering steps:
         1. Prepare the scaling vector used to subselect the most contributing paramters to the impact results
         2. Compute the LCI and LCIA results with the chosen scaling vector
         3. Plot the main contributing processes to the impact result
         4. Filters out the intervention flows whose characterized results 
            are smaller than the cutoff multiplied with the LCA score
         5. Filter out the chcharacterization factors which are not connected to any intervention flows after step 4.

         Args:
            scaling_vector_strategy (str): 
                How to compute scaling vector: 'naive' or 'constructed_demand'.
            cutoff (float): 
                cutoff factor to compute minimum contribution value to retain an intervention flow. 
                Multiplied with the LCA score, i.e., a percentage of the total LCA score
            plot_n_top_processes (int): 
                Number of top items to display in top contribution process plot (default: 10).

        Returns:
            filtered_inventory_indcs (list): 
                Subset of inventory flows indices returned from filtering.
            filtered_characterization_indcs (list): 
                Subset of characterization factors indices returned from filtering.
         """
         scaling_vector_series = self.prepare_scaling_vector(scaling_vector_strategy=scaling_vector_strategy)
         lca_score, characterized_inventory = self.compute_LCI_LCIA(scaling_vector_series)
         self.plot_top_processes(characterized_inventory, top_amount=plot_n_top_processes)
         filtered_inventory_indcs = self.filter_inventoryflows(characterized_inventory, lca_score, cutoff)
         filtered_characterization_indcs = self.filter_characterization_factors(filtered_inventory_indcs)
         return filtered_inventory_indcs, filtered_characterization_indcs

    def prepare_scaling_vector(self,  scaling_vector_strategy:str='naive') -> pd.Series:
        """
        Prepares the scaling vector which will be used to compute the LCIA contributions per inventory flow.
        The scaling vector can be created from the determinisitc optimum ('naive') 
        or computed based on all possible choices ('constructed_demand').

        Args:
            scaling_vector_strategy (str): 
                How to compute scaling vector: 'naive' or 'constructed_demand' (default: 'naive')

        Returns:
            scaling_vector_series: Series of scaling factors (optimal s).
        """
        match scaling_vector_strategy:
            case 'naive':
            # put the scaling vector returned from the optimization into the same order as the process map
                scaling_vector_series = self.result_data['Scaling Vector']['Value'].sort_index()
            case 'constructed_demand':
                scaling_vector_series = self.construct_scaling_vector_from_choices()
            case _:
                raise Exception('Case not implemented.')
        return scaling_vector_series
        
    def construct_scaling_vector_from_choices(self) -> pd.Series:
        """
        Compute a scaling vector from a constructed demand, with the idea that it contains the impact linked to all possible choices.
        The demand vector is constructed that the demand for each choice is equal to one.

        Returns:
            pd.Series: Scaling factors based on the constructed demand.
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
        Compute per-inventory LCI and LCIA contributions.

        Args:
            scaling_vector_series (pd.Series):
                Series of parameter scaling factors (s) from 'prepare_scaling_vector' method.

        Returns:
            lca_score (float): 
                the summed lcia score for the specific scaling vector.
            characterized_inventory (scipy.sparse.sparray): 
                B·(Q·s) for each parameter (impact after characterization).
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
            characterized_inventory (scipy.sparse.sparray): 
                B·(Q·s) for each parameter (impact after characterization).
            lca_score (float): 
                the summed lcia score for the specific scaling vector.
            cutoff (float): 
                cutoff factor to compute minimum contribution value to retain a parameter. 
                Multiplied with the LCA score, i.e., a percentage of the total LCA score

        Returns:
            characterized_inventory_indices (list): 
                Subset of inventory flows indices returned from filtering.
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
            characterized_inventory_indices (list): 
                list of intervention flows returned from the filtering process.

        Returns:
            reduced_characterization_matrix_ids (list): 
                Subset of characterization factors indices returned from filtering.
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
            characterized_inventory (scipy.sparse.sparray): 
                B·(Q·s) for each parameter (impact after characterization).
            top_amount (int): 
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
