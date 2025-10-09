
"""
gsa.py

Module for the global sensitivity analysis based on SaLib library. 
It currently only looks at sensitivity of an LCA, i.e., an optimal LCO solution.
It also only considers uncertainty in the Biosphere matrix (invervention flows, B)
and in the Characterization matrix (characterization factors, Q).
"""

import pandas as pd
import scipy.sparse
import warnings

from pulpo.utils.uncertainty import plots, processor
from pulpo.utils.uncertainty.preparer import UncertaintyData


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
        uncertainty_data (dict[str:pd.DataFrame]):
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
        uncertainty_data: UncertaintyData,
        sampler,
        analyser,
        sample_size: int,
        method:str,
        plot_gsa_results:bool=False,
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
            uncertainty_data (UncertaintyData):
                Uncertainty metadata containing uncertainty informtaion for characterization factors 
                and intervention flows (must have no undefined types), contains "cf" and "if" keys.
            sampler:
                SALib sampling function (e.g., SALib.sample.saltelli).
            analyser:
                SALib analysis function (e.g., SALib.analyze.sobol).
            sample_size (int):
                Number of samples to generate for the analysis.
            plot_gsa_results (bool):
                Whether to generate plots of the GSA results.
        """
        self.result_data = result_data # This is the optimization solution at which we compute the GSA
        self.method = method # ATTN: This might generate errors in the future
        self.lci_data = lci_data # from pulpo_worker
        self.uncertainty_data = uncertainty_data
        if processor.check_missing_uncertainty_data(uncertainty_data):
            warnings.warn("The uncertainty data contains undefined uncertainty types. Please define all uncertainty types before running the GSA.")
        self.sampler = sampler # from SALib.sample
        self.analyser = analyser # from SALib.analyze
        self.sample_size = sample_size
        self.sample_impacts = None
        self.sample_characterized_inventories = None
        self.sensitivity_indices = None
        self.plot_gsa_results_bool = plot_gsa_results

    def perform_gsa(self) -> tuple[pd.DataFrame, dict]:
        """
        Calls all relevant methods including plots to perform a full GSA with initialized data

        Returns:
            total_Si (pd.DataFrame):
                DataFrame of total Sobol indices 'ST' and 'ST_conf' indexed by parameter names.
            sensitivity_indices (dict):
                Full SALib sensitivity indices output.
        """
        gsa_problem, all_bounds_indx_dict = self.define_problem()
        sample_data_if, sample_data_cf = self.sample(gsa_problem, all_bounds_indx_dict)
        sample_impacts, sample_characterized_inventories = self.run_model(sample_data_if, sample_data_cf)
        total_Si, sensitivity_indices = self.analyze(gsa_problem, sample_impacts)
        if self.plot_gsa_results_bool:
            self.plot_gsa_results(all_bounds_indx_dict, total_Si, sample_characterized_inventories)
        return total_Si, sensitivity_indices


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
        All_if_uncertainty_data = {}
        for database in self.uncertainty_data['If'].keys():
            All_if_uncertainty_data.update(self.uncertainty_data['If'][database]['defined'])
        if_bounds = processor.compute_bounds(All_if_uncertainty_data, return_type='dict')
        cf_bounds = processor.compute_bounds(self.uncertainty_data['Cf'][self.method]['defined'], return_type='dict')
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

    def analyze(self, problem:dict, sample_impacts:pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Calculate Sobol sensitivity indices from sampled impacts.

        Args:
            problem (dict): SALib problem definition.
            sample_impacts (pd.Series): Impact per sample.

        Returns:
            total_Si (pd.DataFrame): 
                DataFrame of total Sobol indices 'ST' and 'ST_conf' indexed by parameter names.
            sensitivity_indices (dict):
                Full SALib sensitivity indices output.
        """
        sensitivity_indices = self.analyser.analyze(problem, sample_impacts.sparse.to_dense().values, parallel=True)
        # total_Si, first_Si, second_Si = sensitivity_indices.to_df()
        total_Si = pd.DataFrame([sensitivity_indices['ST'].T, sensitivity_indices['ST_conf'].T], index=['ST', 'ST_conf'], columns=problem['names']).T
        # Calculate total explained variance
        print("The total explained variance is \n{:.4}%".format(total_Si["ST"].sum()*100))
        return total_Si, sensitivity_indices

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

    def plot_gsa_results(self, all_bounds_indx_dict:dict, total_Si:pd.DataFrame, sample_characterized_inventories:pd.DataFrame):
        """
        Generate plots for the GSA results. Must be called after `analyze`. 
        Args:
            all_bounds_indx_dict (dict): 
                Contains 'cf_start' to split IF/CF.
            total_Si (pd.DataFrame): 
                DataFrame of sensitivity indices.
            sample_characterized_inventories (pd.DataFrame): 
                flows × samples.
        """
        total_Si_metadata = self.generate_Si_metadata(all_bounds_indx_dict, total_Si)
        colormap_base, colormap_SA_barplot = plots.plot_top_total_sensitivity_indices(total_Si, total_Si_metadata)
        plots.plot_total_env_impact_contribution(
            sample_characterized_inventories, 
            total_Si_metadata, 
            self.method, 
            top_amount=10,
            colormap_base=colormap_base, 
            colormap_SA_barplot=colormap_SA_barplot,
        )

    

