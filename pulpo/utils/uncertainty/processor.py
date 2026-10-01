"""
processor.py

Helpers on the uncertainty data: completeness checks, statistics of declared
distributions, readable parameter names, and the sampler that draws one
realization of every declared parameter.
"""

import numpy as np
import pandas as pd
import stats_arrays
from typing import Union, List, Dict, Tuple, Literal

from pulpo.utils.uncertainty.preparer import UncertaintyData

def check_missing_uncertainty_data(uncertainty_data: UncertaintyData, unc_types: List[Literal['If', 'Cf']] = ('If', 'Cf')) -> bool:
    """
    Check if there are any undefined uncertainty data in the uncertainty_data dict.
    
    Args:
        uncertainty_data (UncertaintyData): 
            Dictionary containing metadata about uncertain intervention flows (IF) and characterization factors (CF).
        unc_types (List[Literal['If', 'Cf']]):
            The groups to check. Defaults to both.
    
    Returns:
        missing_unc_data (bool): 
            True if there is any undefined uncertainty data, False otherwise.
    """
    missing_unc_data = False
    for unc_type, unc_type_data in uncertainty_data.items():
        # Only check the specified uncertainty types as some CC formulations might just require a subset
        if unc_type not in unc_types:
            continue
        for unc_subgroup, unc_subgroup_data in unc_type_data.items():
            if len(unc_subgroup_data['undefined']):
                missing_unc_data = True
                print('{} - {} \n \t {} parameters without uncertainty information'.format(unc_type, unc_subgroup, len(unc_subgroup_data['undefined'])))
    if not missing_unc_data:
        print('No uncertainty data missing.')
    return missing_unc_data

def drop_undefined_uncertainty_data(uncertainty_data:UncertaintyData) -> UncertaintyData:
    """
    Drops any undefined uncertainty data in the uncertainty_data dict.
    
    Args:
        uncertainty_data (UncertaintyData): 
            Dictionary containing metadata about uncertain intervention flows (IF) and characterization factors (CF).
    Returns:
        cleaned_uncertainty_data (UncertaintyData):
            Dictionary containing metadata about uncertain intervention flows (IF) and characterization factors (CF),
            with any undefined uncertainty data removed.
    """
    cleaned_uncertainty_data:UncertaintyData = {}
    for unc_type, unc_type_data in uncertainty_data.items():
        cleaned_uncertainty_data[unc_type] = {}
        for unc_subgroup, unc_subgroup_data in unc_type_data.items():
            cleaned_uncertainty_data[unc_type][unc_subgroup] = {}
            cleaned_uncertainty_data[unc_type][unc_subgroup]['defined'] = unc_subgroup_data['defined']
            cleaned_uncertainty_data[unc_type][unc_subgroup]['undefined'] = {}
            if len(unc_subgroup_data['undefined']):
                print('Dropping {} undefined uncertainty parameters for {} - {}'.format(len(unc_subgroup_data['undefined']), unc_type, unc_subgroup))
    return cleaned_uncertainty_data


def compute_bounds(uncertainty_metadata:dict, return_type:str='df') -> Union[pd.DataFrame, dict]:
    """
    Compute mean, median (or mode), and 95% CI bounds for each parameter using the stats_array package.

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

def rename_metadata_index(metadata_df, lci_data:dict, param_type:str):
        """
        Changes the index of the metadata_df from the matrix index to a readable name based on the metadata of the underlying parameters.
        Currently implemented for "intervention_flow" and "characterization_factor".

        Args:
            metadata_df (pd.dataframe):
                The metadata dataframe containing the index to be renamed as dataframe index
            lci_data (dict):
                The lci_data containing the "..._map_metadata" dicts needed to rename the index, from pulpo_worker.
            param_type (str):
                The parameter name contained in the "metadata_df", 
                options are: "intervention_flow", "characterization_factor" and "process".
        
        Returns:
            metadata_df (pd.DataFrame):
                The uncertainty metadata frame with descriptive indices.
        """
        match param_type:
            case 'intervention_flow':
                if_index_map = {
                    (interv_indx, process_indx):  '{} --- {}'.format(
                        lci_data['process_map_metadata'][process_indx], lci_data['intervention_map_metadata'][interv_indx]
                        ) for interv_indx, process_indx in metadata_df.index
                        }
                flat_index = metadata_df.index.to_flat_index()
                metadata_df = metadata_df.reset_index()
                metadata_df.index = flat_index
                metadata_df = metadata_df.rename(index=if_index_map)
            case 'characterization_factor':
                cf_index_map = {interv_indx:  '{} '.format(lci_data['intervention_map_metadata'][interv_indx]) for interv_indx in metadata_df.index}
                metadata_df = metadata_df.reset_index()
                metadata_df.index = metadata_df['index']
                metadata_df = metadata_df.rename(index=cf_index_map)
            case 'process':
                process_index_map = {process_indx:  '{} '.format(lci_data['process_map_metadata'][process_indx]) for process_indx in metadata_df.index}
                metadata_df = metadata_df.reset_index()
                metadata_df.index = metadata_df['index']
                metadata_df = metadata_df.rename(index=process_index_map)
            case _:
                raise Exception(f'"rename_metadata_index" to <<{param_type}>> as "uncertainty_var_name" has not been implemented')
        return metadata_df

# --- Sampler


def _merge_defined_blocks(unc_data: dict, top_key: str) -> Dict[Union[Tuple[int,int],int], dict]:
    """Collect & merge all 'defined' blocks under unc_data[top_key]."""
    out = {}
    for block in unc_data.get(top_key, {}).values():
        out.update(block.get('defined', {}))
    return out


def _sample_one_spec(spec: dict, rng: np.random.Generator) -> float:
    """
    Sample a single uncertainty spec. An exact value (type 1) returns its
    amount, a normal is drawn with numpy, every other family by stats_arrays.

    ``rng`` must reach *both* branches. stats_arrays' ``random_variables``
    falls back to the legacy global ``np.random`` when ``seeded_random`` is
    omitted, so leaving it out made ``draw_uncertainty_sample(seed=...)``
    reproducible for Normal parameters only -- every lognormal, triangular and
    uniform parameter silently ignored the seed and consumed the global stream
    instead. Each of the four families stats_arrays dispatches to calls only
    ``normal``/``lognormal``/``triangular``/``uniform``, all of which exist on
    a ``Generator``, so the same object serves both branches.
    """
    utype = spec.get("uncertainty_type", None)
    if utype == stats_arrays.NoUncertainty.id:
        return float(spec['amount'])
    if utype == stats_arrays.NormalUncertainty.id:
        loc = float(spec.get("loc", 0.0) or 0.0)
        scale = float(spec.get("scale", 0.0) or 0.0)
        # scale may be 0 for degenerate normals -> returns loc deterministically
        return float(rng.normal(loc, scale)) if scale > 0 else loc
    # generic fallback for triangular/lognormal/etc.
    ua = stats_arrays.UncertaintyBase.from_dicts(spec)
    choice = stats_arrays.uncertainty_choices[utype]
    return float(np.asarray(choice.random_variables(ua, 1, seeded_random=rng)).ravel()[0])

def draw_uncertainty_sample(
    uncertainty_data: dict,
    method: str,
    seed: int | None = None,
) -> dict:
    """
    One draw of every declared parameter: ``{'If': {(e, j): value}, 'Cf': {e: value}}``.
    Undeclared parameters keep their amounts and are not part of the draw.
    """
    rng = np.random.default_rng(seed)
    if_draw = {k: _sample_one_spec(v, rng) for k, v in _merge_defined_blocks(uncertainty_data, 'If').items()}
    cf_defined = uncertainty_data.get('Cf', {}).get(method, {}).get('defined', {})
    cf_draw = {k: _sample_one_spec(v, rng) for k, v in cf_defined.items()}
    return {'If': if_draw, 'Cf': cf_draw}
