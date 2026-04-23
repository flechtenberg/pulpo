"""
cc.py

Chance-constrained formulation helpers. Moved out of pulpo.utils.optimizer so
that the core optimizer module has no hard dependency on stats_arrays or on
the uncertainty sub-package.

These functions are only imported by the uncertainty-enabled optimizer facade
(`pulpo.pulpo_unc`).
"""

import array
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats
import stats_arrays

from pulpo.utils.uncertainty.preparer import UncertaintyData, UncertaintySpec


def compute_L1_env_cost_mean_var(
        normal_uncertainty_data: UncertaintyData,
        lci_data: dict,
        method: str,
        plot_analysis_support_plots: bool = False,
        ) -> Dict[Tuple[int, str], UncertaintySpec]:
    """
    Computes the environmental cost mean and variance associated with
    the uncertain intervention and characterization flows specified in uncertainty_data.

    This is a shortcut approach to implement an individual chance-constraint formulation
    on the objective using the L1 norm on normally distributed uncertainties.
    """
    def _check_all_uncertainty_is_normal(uncertainty_data: UncertaintyData, method: str):
        normal_id = stats_arrays.NormalUncertainty.id
        for if_data in uncertainty_data['If'].values():
            for spec in if_data['defined'].values():
                if spec.get('uncertainty_type', None) != normal_id:
                    raise ValueError("All 'If' uncertainty specs must be Normal distributions.")
        for spec in uncertainty_data['Cf'][method]['defined'].values():
            if spec.get('uncertainty_type', None) != normal_id:
                raise ValueError("All 'Cf' uncertainty specs must be Normal distributions.")

    def _extract_process_ids_and_intervention_flows_for_env_cost_variance(
            uncertainty_data: UncertaintyData, lci_data: dict, method: str
    ) -> tuple[array.array, pd.DataFrame]:
        process_id_uncertain_if = []
        for if_unc_data in uncertainty_data['If'].values():
            process_id_uncertain_if += [if_indx for (_, if_indx) in if_unc_data['defined'].keys()]
        Cf_indcs = list(uncertainty_data['Cf'][method]['defined'].keys())
        process_id_associated_cf = lci_data['intervention_matrix'][Cf_indcs, :].nonzero()[1]
        process_ids = np.unique(np.append(process_id_associated_cf, process_id_uncertain_if))
        intervention_flows_extracted = pd.DataFrame.sparse.from_spmatrix(
            lci_data['intervention_matrix'][Cf_indcs, :][:, process_ids],
            index=Cf_indcs,
            columns=process_ids,
        )
        intervention_flows_extracted_stacked = intervention_flows_extracted.stack().astype('float')
        for If_db in uncertainty_data['If'].keys():
            normal_means = pd.DataFrame.from_dict(uncertainty_data['If'][If_db]['defined']).T['loc']
            intervention_flows_extracted_stacked.update(normal_means)
        intervention_flows_extracted = intervention_flows_extracted_stacked.unstack()
        return process_ids, intervention_flows_extracted

    def _extract_characterization_factors_for_env_cost_variance(
            uncertainty_data: UncertaintyData, lci_data: dict, method: str
    ) -> pd.Series:
        characterization_factor_mean = pd.Series(lci_data["matrices"][method].diagonal())
        normal_means = pd.DataFrame.from_dict(uncertainty_data['Cf'][method]['defined']).T['loc']
        characterization_factor_mean.update(normal_means)
        return characterization_factor_mean

    def _compute_envcost_variance(normal_uncertainty_data: UncertaintyData, lci_data, method) -> dict:
        if_unc_dict = {}
        for if_uncertainty_data in normal_uncertainty_data['If'].values():
            if_unc_dict.update(if_uncertainty_data['defined'])
        if_normal_metadata_df = pd.DataFrame(if_unc_dict).T
        cf_normal_metadata_df = pd.DataFrame(normal_uncertainty_data['Cf'][method]['defined']).T
        process_ids, intervention_flows_extracted = _extract_process_ids_and_intervention_flows_for_env_cost_variance(
            normal_uncertainty_data, lci_data, method
        )
        characterization_factor_extracted = _extract_characterization_factors_for_env_cost_variance(
            normal_uncertainty_data, lci_data, method
        )
        envcost_std = {}
        for process_id in process_ids:
            if process_id in if_normal_metadata_df.index.get_level_values(level=1):
                intervention_flow_std = if_normal_metadata_df.xs(process_id, level=1, axis=0, drop_level=True)['scale']
                characterization_factor_mean = characterization_factor_extracted[
                    intervention_flow_std.index.get_level_values(level=0)
                ]
                characterization_factor_mean = characterization_factor_mean.reindex(
                    intervention_flow_std.index, axis=0, level=0
                )
                mu_q2_sigma_b2 = characterization_factor_mean.pow(2).mul(intervention_flow_std.pow(2), axis=0)
            else:
                mu_q2_sigma_b2 = pd.Series([0])
            if (intervention_flows_extracted[process_id] > 0).any():
                characterization_factor_std = cf_normal_metadata_df['scale']
                intervention_flow_mean = intervention_flows_extracted[process_id]
                sigma_q2_mu_b2 = characterization_factor_std.pow(2).mul(intervention_flow_mean.pow(2), axis=0)
            else:
                sigma_q2_mu_b2 = pd.Series([0])
            if (intervention_flows_extracted[process_id] > 0).any() and process_id in if_normal_metadata_df.index.get_level_values(level=1):
                sigma_q2_sigma_b2 = characterization_factor_std.pow(2).mul(intervention_flow_std.pow(2))
            else:
                sigma_q2_sigma_b2 = pd.Series([0])
            envcost_std[process_id] = np.sqrt(mu_q2_sigma_b2.sum() + sigma_q2_sigma_b2.sum() + sigma_q2_mu_b2.sum())
        return envcost_std

    def _compute_envcost_mean(lci_data: dict, normal_uncertainty_data: UncertaintyData, method: str) -> dict:
        Cf_means = _extract_characterization_factors_for_env_cost_variance(normal_uncertainty_data, lci_data, method)
        intervention_flows_extracted = pd.DataFrame.sparse.from_spmatrix(lci_data['intervention_matrix'])
        intervention_flows_extracted_stacked = intervention_flows_extracted.stack().astype('float')
        for If_db in normal_uncertainty_data['If'].keys():
            normal_means = pd.DataFrame.from_dict(normal_uncertainty_data['If'][If_db]['defined']).T['loc']
            intervention_flows_extracted_stacked.update(normal_means)
        If_means = intervention_flows_extracted_stacked.unstack()
        envcost_mean = (Cf_means @ If_means).to_dict()
        return envcost_mean

    def _check_envcost_variance(envcost_std: dict, envcost_mean: dict, lci_data: dict, plot_details: bool = False):
        envcost_std_mean = pd.DataFrame.from_dict(envcost_std, orient='index', columns=['std'])
        envcost_std_mean['metadata'] = envcost_std_mean.index.map(lci_data['process_map_metadata'])
        if envcost_std_mean['std'].isna().any():
            raise Exception('There are NaNs in the standard deviation')
        envcost_std_mean['mean'] = envcost_std_mean.index.map(envcost_mean)
        envcost_std_mean['z'] = envcost_std_mean['std'] / envcost_std_mean['mean']
        if (envcost_std_mean['z'] > 0.5).any():
            if plot_details:
                print('These environmental costs have a standard deviation larger than 50% of their mean:\n')
                print(envcost_std_mean[envcost_std_mean['z'] > 0.5].sort_values('z', ascending=False))
        if plot_details:
            envcost_std_mean['z'].sort_values(ascending=False).iloc[5:].plot.box()
            print('The following points were excluded from the boxplot:')
            print(envcost_std_mean['z'].sort_values(ascending=False).iloc[:5])

    _check_all_uncertainty_is_normal(normal_uncertainty_data, method)
    envcost_std = _compute_envcost_variance(normal_uncertainty_data, lci_data, method)
    envcost_mean = _compute_envcost_mean(lci_data, normal_uncertainty_data, method)
    _check_envcost_variance(envcost_std, envcost_mean, lci_data, plot_details=plot_analysis_support_plots)
    normal_metadata_env_cost: Dict[Tuple[int, str], UncertaintySpec] = {
        (process_id, method): {
            'loc': envcost_mean[process_id],
            'scale': envcost_std[process_id],
            'uncertainty_type': stats_arrays.NormalUncertainty.id,
            'amount': np.NaN,
            'maximum': np.NaN,
            'minimum': np.NaN,
            'shape': np.NaN,
        } for process_id in envcost_std.keys()
    }
    return normal_metadata_env_cost


def apply_CC_formulation(
        model_instance,
        lambda_level: float,
        normal_metadata_env_cost: Dict[Tuple[int, str], UncertaintySpec] = {},
        normal_metadata_var_bounds: Dict[str, Dict[int, UncertaintySpec]] = {},
        ):
    """
    Inject or update the epsilon-constraint for a given risk level.
    """
    ppf_lambda = scipy.stats.norm.ppf(lambda_level)
    if normal_metadata_env_cost:
        print(f'Applying CC constraints to the environmental cost calculation with lambda: {lambda_level}')
        environmental_cost_updated = {
            env_cost_indx: env_cost_data['loc'] + ppf_lambda * env_cost_data['scale']
            for env_cost_indx, env_cost_data in normal_metadata_env_cost.items()
        }
        model_instance.ENV_COST_MATRIX.store_values(environmental_cost_updated, check=True)
    for bound_name, metadata_vb in normal_metadata_var_bounds.items():
        if metadata_vb:
            print(f'Applying CC constraints to the {bound_name} constraint with lambda: {lambda_level}')
            match bound_name:
                case 'upper_limit':
                    pyomo_var_name = 'UPPER_LIMIT'
                    bound_updated = {indx: (unc_data['loc'] - ppf_lambda * unc_data['scale'])
                                     for indx, unc_data in metadata_vb.items()}
                case 'lower_limit':
                    pyomo_var_name = 'LOWER_LIMIT'
                    bound_updated = {indx: (unc_data['loc'] + ppf_lambda * unc_data['scale'])
                                     for indx, unc_data in metadata_vb.items()}
                case 'upper_imp_limit':
                    pyomo_var_name = 'UPPER_IMP_LIMIT'
                    bound_updated = {indx: (unc_data['loc'] - ppf_lambda * unc_data['scale'])
                                     for indx, unc_data in metadata_vb.items()}
                case 'upper_inv_limit':
                    pyomo_var_name = 'UPPER_INV_LIMIT'
                    bound_updated = {indx: (unc_data['loc'] - ppf_lambda * unc_data['scale'])
                                     for indx, unc_data in metadata_vb.items()}
                case _:
                    raise Exception('has not been implemented yet.')
            pyomo_bound = getattr(model_instance, pyomo_var_name)
            pyomo_bound.store_values(bound_updated, check=True)
