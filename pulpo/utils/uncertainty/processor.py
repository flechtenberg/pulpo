"""
processor.py

Module for processing the uncertainty data, by filling in missing data,
updating data or comupting metrics from the uncertainty data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stats_arrays
import scipy.stats
import pandas as pd
import numpy as np
import stats_arrays
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Dict, Tuple, Literal

from pulpo.utils.uncertainty.preparer import UncertaintyData, UncertaintySpec

class UncertaintyStrategyBase:
    """
    Base class for uncertainty distribution specification strategies.

    This abstract base defines the interface and common functionality for assigning probability
    distributions to parameters lacking predefined uncertainty metadata. Subclasses must implement
    methods to derive distribution parameters for these unspecified uncertainties.

    """
    def __init__(
            self, 
            uncertain_param_type:Literal['If', 'Cf', 'Var_bounds'], 
            uncertain_param_subgroup:str,
        ):
        """
        Initialize the UncertaintyStrategyBase with metadata and index lists.

        Args:
            uncertain_param_type (str): 
                The uncertainty parameter type: 'If', 'Cf', or 'Var_bounds'
            uncertain_param_subgroup (str):
                The subgroup of the uncertain_param_type, depends on the 
                uncertain_param_type, the BW databse name for 'If', the 
                LCIA method for 'Cf' and the types of variable bounds for 
                'Var_bounds'
        """
        self.uncertain_param_type = uncertain_param_type
        self.uncertain_param_subgroup = uncertain_param_subgroup

    def add_random_noise_to_scaling_factor(self, undefined_param_amt:int, scaling_factor:Union[float, list], low:float, high:float) -> list:
        """
        Adds random noise from uniform distribution to scaling factors, to avoid structured randomness when scaling the data.
        Multiplies the scaling factors in the array of scaling factors with 1-low and 1+high 
        to generate noisy scaling vectors given by the interval [1-low, 1+high].

        Args:
            scaling_factor (float, array, list): 
                the scaling factor which will get noise added to it
            undefined_param_amt (int):
                The amount of undefined parameters to create the scaling_factor_randomized shape.
            low (float): 
                the lower bound (1-low) for which the noise will be sampled and multiplied to the scaling_factor.
            high (float): 
                the lower bound (1+high) for which the noise will be sampled and multiplied to the scaling_factor.

        Returns:
            scaling_factor_randomized (list): 
                The scaling factor, now a list superposed with the random noise

        """
        if isinstance(scaling_factor, float):
            scaling_factor = [scaling_factor] * undefined_param_amt
        rng = np.random.default_rng(seed=161)
        random_noise = rng.uniform(1-low, 1+high, undefined_param_amt)
        scaling_factor_randomized = random_noise * np.array(scaling_factor)
        return scaling_factor_randomized.tolist()

    def assign(self, *args):
        """
        Assign distribution parameters to parameters without predefined uncertainty.
        """
        pass

class ExpertKnowledgeStrategy(UncertaintyStrategyBase):
    """
    Adds specific probability distribution informaiton to uncertain parameters based on expert judgement.
    """
    def __init__(
            self,
            uncertain_param_type:Literal['If', 'Cf', 'Var_bounds'], 
            uncertain_param_subgroup:str,
            prob_metadata:Dict[int, Dict[str,Union[int, float]]]
            ):
        """
        Initialize the ExpertKnowledgeStrategy with 'prob_metadata' containing the expert knowledge 
        and index lists with the indexes of the parameter who's uncertainty information should be
        replaced with the data in prob_metadata.

        Args:
            uncertain_param_type (str): 
                The uncertainty parameter type: 'If', 'Cf', or 'Var_bounds'
            uncertain_param_subgroup (str):
                The subgroup of the uncertain_param_type, depends on the 
                uncertain_param_type, the BW databse name for 'If', the 
                LCIA method for 'Cf' and the types of variable bounds for 
                'Var_bounds'
            prob_metadata (Dict[int, Dict[str,Union[int, float]]]): 
                the export knowledge uncertainty metadata of the parameters which are to be
                replaced in the metadata_df, the dictionary is structured as:
                { 
                    index of 1. uncertain parameter (int or tuple): {
                        uncertainty attribute (str), e.g., 'loc', 'scale', 'minimum', 'uncertainty_type': 
                            attribute value (float or int) 
                    },
                    index of 2. uncertain parameter (int or tuple): {
                        ...
                    },
                    ...
                }
        """
        super().__init__(uncertain_param_type, uncertain_param_subgroup)
        self.prob_metadata = prob_metadata  


    def insert_expert_knowledge(
            self, 
            uncertainty_data:UncertaintyData
            ):
        """
        Updated the parameters uncertainty information in the metadata_df
        with expert uncertainty information found in the prob_metadata dict.

        Args:
            uncertainty_data (dict): 
                The uncertainty data containing the defined and undefined 
                uncertainty information, returned from:
                `preparer.UncertaintyImporter.import_uncertainty_data()`
        """
        # Roughly checks the probability metadata by creating an uncertainty object in stats_array 
        # which will only take the correct keys out of the prob_metadata dict
        # It can still have missing data or wrong values
        prob_metadata_stats_array = stats_arrays.UncertaintyBase.from_dicts(*self.prob_metadata.values())
        # Writes the expert uncertainty information into the uncertainty_data
        for indx, prob_metadata_arr in zip(self.prob_metadata.keys(), prob_metadata_stats_array):
            prob_metadata = dict(zip(prob_metadata_stats_array.dtype.names, prob_metadata_arr))
            if indx not in uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined'].keys():
                raise Exception(f'{indx} is not found in "defined" uncertainty data of {self.uncertain_param_subgroup} in {self.uncertain_param_type}.')
            uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined'][indx].update(prob_metadata)
    
    def assign(self, uncertainty_data:UncertaintyData):
        self.insert_expert_knowledge(uncertainty_data)
class UniformBaseStrategy(UncertaintyStrategyBase):
    """
    Strategy that assigns uniform distributions to parameters with undefined uncertainty information.

    For each parameter index in undefined_uncertainty_indices, the staretgy sets min and and max based on
    the specified scaling factors. 
    """
    def __init__(
            self, 
            uncertain_param_type:Literal['If', 'Cf', 'Var_bounds'], 
            uncertain_param_subgroup:str,
            upper_scaling_factor:float, 
            lower_scaling_factor:float,
            noise_interval:Dict[str,float]={'min':0., 'max':0.}
            ) -> None:
        """
        Initialize the UniformBaseStrategy with metadata and index lists and scaling factors.

        Args:
            uncertain_param_type (str): 
                The uncertainty parameter type: 'If', 'Cf', or 'Var_bounds'
            uncertain_param_subgroup (str):
                The subgroup of the uncertain_param_type, depends on the 
                uncertain_param_type, the BW databse name for 'If', the 
                LCIA method for 'Cf' and the types of variable bounds for 
                'Var_bounds'
            upper_scaling_factor (float): 
                The scaling factor multiplied with the amount (deterministic values)
                to get the maximum value for the uniform distribution
            lower_scaling_factor (float): 
                The scaling factor multiplied with the amount (deterministic values)
                to get the minimum value for the uniform distribution
            noise_interval (Dict[str,float]): Dict containing "min" and "max" keywords 
                holding the upper and lower bound of the noise generated with a uniform distribution 
                and multiplied with the scaling factor vector as (1-min) and (1+max)
        """
        super().__init__(uncertain_param_type, uncertain_param_subgroup)
        self.upper_scaling_factor = upper_scaling_factor
        self.lower_scaling_factor = lower_scaling_factor
        self.noise_interval = noise_interval

    def _compute_uniform_dist_params(
            self,
            uncertainty_data:UncertaintyData, 
            ) -> pd.DataFrame:
        """
        Compute uniform distribution parameters to parameters without predefined uncertainty.

        Args:
            uncertainty_data (dict): 
                The uncertainty data containing the defined and undefined 
                uncertainty information, returned from:
                `preparer.UncertaintyImporter.import_uncertainty_data()`

        Returns:
            pd.DataFrame: Updated metadata DataFrame including 'minimum', 'maximum',
                          and 'uncertainty_type' set to 4 (uniform) for targeted parameters.
        """
        # Create a scaling factor array if scaling_factors are floats and randomize it if the noise interval has min and max greater 0.
        undefined_param_amt = len(uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['undefined'])
        upper_scaling_factor_randomized = self.add_random_noise_to_scaling_factor(undefined_param_amt, self.upper_scaling_factor, self.noise_interval['min'], self.noise_interval['max'])
        lower_scaling_factor_randomized = self.add_random_noise_to_scaling_factor(undefined_param_amt, self.lower_scaling_factor, self.noise_interval['min'], self.noise_interval['max'])
        # For each undefined parameter, set loc=median, bounds = ±factor·|median|
        undefined_uncertainty_indices = list(uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['undefined'].keys())
        for undefined_indx, upper_scaling_factor, lower_scaling_factor in zip(undefined_uncertainty_indices, upper_scaling_factor_randomized, lower_scaling_factor_randomized):
            undefined_dict = uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['undefined'].pop(undefined_indx)
            uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined'][undefined_indx] = undefined_dict
            amount = undefined_dict['amount']
            uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined'][undefined_indx]['loc'] = np.NaN
            if amount > 0:
                uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined'][undefined_indx]['maximum'] = amount + upper_scaling_factor * abs(amount)
                uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined'][undefined_indx]['minimum'] = amount - lower_scaling_factor * abs(amount)
            elif amount < 0:
                uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined'][undefined_indx]['maximum'] = amount + lower_scaling_factor * abs(amount)
                uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined'][undefined_indx]['minimum'] = amount - upper_scaling_factor * abs(amount)
            uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined'][undefined_indx]['uncertainty_type'] = 4
        # Check for negative‐median cases and adjust skew mapping
        # ATTN: fix this Exception based on the new uncertainty data type
        # if ((metadata_df.loc[self.undefined_uncertainty_indices,'maximum'] - metadata_df.loc[self.undefined_uncertainty_indices,'minimum']) <= 0).any():
        #     raise Exception('There is a parameter with where the asigned minimum value is equal or larger than the asigned maximum value')
    
    def assign(self, uncertainty_data:UncertaintyData):
        self._compute_uniform_dist_params(uncertainty_data)

class TriangluarBaseStrategy(UncertaintyStrategyBase):
    """
    Strategy that assigns triangular distributions with specified scaling facotrs
    to parameters with undefined uncertainty information.

    For each undefined uncertain parameter, this strategy sets the median
    (loc) equal to the amount and computes min and and max based on
    the speficied scaling factors.

    The min is computed as loc - lower_scaling_factor * abs(loc), and the max
    as loc + upper_scaling_factor * abs(loc).

    Methods:
        _compute_triag_dist_params: Computes scaling factors (upper and lower) based on given scaling_factors.
        assign: Applies computed scaling factors to assign 'loc', 'minimum', 'maximum', and 'uncertainty_type'.
    """
    def __init__(
            self, 
            uncertain_param_type:Literal['If', 'Cf', 'Var_bounds'], 
            uncertain_param_subgroup:str,
            upper_scaling_factor:float, 
            lower_scaling_factor:float, 
            noise_interval:Dict[str,float]={'min':0., 'max':0.}
            ) -> None:
        """
        Initialize the TriangluarBaseStrategy with metadata and index lists and scaling factors.

        Args:
            uncertain_param_type (str): 
                The uncertainty parameter type: 'If', 'Cf', or 'Var_bounds'
            uncertain_param_subgroup (str):
                The subgroup of the uncertain_param_type, depends on the 
                uncertain_param_type, the BW databse name for 'If', the 
                LCIA method for 'Cf' and the types of variable bounds for 
                'Var_bounds'
            upper_scaling_factor (float): 
                the scaling factor multiplied with the amount to get the maximum value 
                for the triangular distribution
            lower_scaling_factor (float): 
                the scaling factor multiplied with the amount to get the minimum value 
                for the triangular distribution
            noise_interval (Dict[str,float]): 
                Dict containing "min" and "max" keywords holding the upper and lower bound 
                of the noise generated with a uniform distribution and multiplied with 
                the scaling factor vector as (1-min) and (1+max)
        """
        super().__init__(uncertain_param_type, uncertain_param_subgroup)
        self.upper_scaling_factor = upper_scaling_factor
        self.lower_scaling_factor = lower_scaling_factor
        self.noise_interval = noise_interval


    def _compute_triag_dist_params(self, uncertainty_data:UncertaintyData):
        """
        Compute triangular distribution parameters to parameters without predefined uncertainty.

        Args:
            uncertainty_data (dict): 
                The uncertainty data containing the defined and undefined 
                uncertainty information, stored outside the class, initially returned from:
                `preparer.UncertaintyImporter.import_uncertainty_data()`

        Returns:
            metadata_df (pd.DataFrame(): 
                Updated metadata DataFrame including 'loc', 'minimum', 'maximum',
                and 'uncertainty_type' set to 5 (triangular) for targeted parameters.
        """
        # Create a scaling factor array if scaling_factors are floats and randomize it if the noise interval has min and max greater 0.
        undefined_param_amt = len(uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['undefined'])
        upper_scaling_factor_randomized = self.add_random_noise_to_scaling_factor(undefined_param_amt, self.upper_scaling_factor, self.noise_interval['min'], self.noise_interval['max'])
        lower_scaling_factor_randomized = self.add_random_noise_to_scaling_factor(undefined_param_amt, self.lower_scaling_factor, self.noise_interval['min'], self.noise_interval['max'])
        # For each undefined parameter, set loc=median, bounds = ±factor·|median|
        undefined_uncertainty_indices = list(uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['undefined'].keys())
        for undefined_indx, upper_scaling_fac, lower_scaling_fac in zip(undefined_uncertainty_indices, upper_scaling_factor_randomized, lower_scaling_factor_randomized):
            undefined_dict = uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['undefined'].pop(undefined_indx)
            uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined'][undefined_indx] = undefined_dict
            amount = undefined_dict['amount']
            uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined'][undefined_indx]['loc'] = amount
            if amount > 0:
                uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined'][undefined_indx]['maximum'] = amount + upper_scaling_fac * abs(amount)
                uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined'][undefined_indx]['minimum'] = amount - lower_scaling_fac * abs(amount)
            elif amount < 0:
                uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined'][undefined_indx]['maximum'] = amount + lower_scaling_fac * abs(amount)
                uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined'][undefined_indx]['minimum'] = amount - upper_scaling_fac * abs(amount)
            uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined'][undefined_indx]['uncertainty_type'] = 5
        # Check for negative‐median cases and adjust skew mapping
        # ATTN: fix this Exception based on the new uncertainty data type
        # if ((metadata_df.loc[self.undefined_uncertainty_indices,'maximum'] - metadata_df.loc[self.undefined_uncertainty_indices,'minimum']) <= 0).any():
        #     raise Exception('There is a parameter with where the asigned minimum value is equal or larger than the asigned maximum value')
        # There can be negative flows and their upper and lower bounds need to be considered in detail!
        # ATTN: fix this check based on the new uncertainty data type
        # print('uncertain parameters with negative median value:')
        # print(metadata_df.loc[self.undefined_uncertainty_indices].loc[metadata_df.loc[self.undefined_uncertainty_indices,'loc'] < 0])
    
    def assign(self, uncertainty_data:UncertaintyData):
        self._compute_triag_dist_params(uncertainty_data)
    
class TriangularBoundInterpolationStrategy(TriangluarBaseStrategy):
    """
    Strategy that assigns triangular distributions based on interpolated parameter bounds
    to parameters with undefined uncertainty information.

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
    """
    def __init__(
            self, 
            uncertain_param_type:Literal['If', 'Cf', 'Var_bounds'], 
            uncertain_param_subgroup:str,
            noise_interval:Dict[str,float]={'min':0., 'max':0.}):
        """
        Initialize the TriangularBoundInterpolationStrategy with metadata and index lists.

        Args:
            uncertain_param_type (str): 
                The uncertainty parameter type: 'If', 'Cf', or 'Var_bounds'
            uncertain_param_subgroup (str):
                The subgroup of the uncertain_param_type, depends on the 
                uncertain_param_type, the BW databse name for 'If', the 
                LCIA method for 'Cf' and the types of variable bounds for 
                'Var_bounds'
            noise_interval (Dict[str,float]): 
                Dict containing "min" and "max" keywords 
                holding the upper and lower bound of the noise generated with a uniform distribution 
                and multiplied with the scaling factor vector as (1-min) and (1+max)
        """
        super().__init__(uncertain_param_type, uncertain_param_subgroup, np.NaN, np.NaN, noise_interval)

    def _get_bounds(
            self,
            uncertainty_data:UncertaintyData,
            ) -> pd.DataFrame:
        """
        Compute min/max bounds for all parameters via UncertaintyProcessor.
        Raises if no metadata defined.
        """
        if not uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined']:
            raise Exception('There are no uncertain parameters with defined uncertainty, as needed to interpolate the bouds.')
        uncertainty_bounds = compute_bounds(uncertainty_data[self.uncertain_param_type][self.uncertain_param_subgroup]['defined'])
        return uncertainty_bounds
            

    def _compute_bounds_statistics(self, uncertainty_bounds:pd.DataFrame, bound_statistic_fig:bool=False) -> tuple[float, float]:
        """
        Computes the scaling factors from the the bounds of the uncertain parameters with known distribution
        Assumes that the bounds of the median of 95% confidence interval can be used to compute scaling factors.
        
        Args:
            uncertainty_bounds (pd.DataFrame):
                Dataframe contaning the "lower" and "upper" bounds and other statistics as columns
                and parameter indices as rows.
            bound_statistic_fig (bool) - optional:
                default: False, Set True if a descriptive figure for the bound computations should be shown.

        Returns:
            upper_scaling_factor: upper scaling factor to be multiplied with a central moment to get the max value for a distribution, e.g., triangular or uniform
            lower_scaling_factor: lower scaling factor to be multiplied with a central moment to get the min value for a distribution, e.g., triangular or uniform
        """
        if len(uncertainty_bounds) < 3:
            raise Exception('There are only three uncertain parameters with uncertainty bounds, not enough to compute bounds statistics for interpolation')
        lower_spread = (uncertainty_bounds['amount'] - uncertainty_bounds['lower']).abs() / uncertainty_bounds['amount'].abs()
        upper_spread = (uncertainty_bounds['amount'] - uncertainty_bounds['upper']).abs() / uncertainty_bounds['amount'].abs()
        if bound_statistic_fig:
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
    
    def assign(self, uncertainty_data:UncertaintyData):
        """
        Assign triangular distribution parameters derived averaged bounds, to parameters without predefined uncertainty.
        """
        uncertainty_bounds = self._get_bounds(uncertainty_data)
        self.upper_scaling_factor, self.lower_scaling_factor = self._compute_bounds_statistics(uncertainty_bounds)
        self._compute_triag_dist_params(uncertainty_data)

def apply_uncertainty_strategies(uncertainty_data: UncertaintyData, strategies: List[UncertaintyStrategyBase]):
    """
    Applies the strategies, by passing the strategies as instatialized classes and then performs the assign method.

    Args:
        uncertainty_data (dict): 
            The uncertainty data containing the defined and undefined 
            uncertainty information, stored outside the class, initially returned from:
            `preparer.UncertaintyImporter.import_uncertainty_data()`
        strategies (List[UncertaintyStrategyBase]):
            All strategies as instatialized classes for which will manipulate the uncertainty_data
    """
    did = id(uncertainty_data)
    for s in strategies:
        print('Applying uncertainy strategy {}, for {} in {}'.format(s.__class__.__name__, s.uncertain_param_subgroup, s.uncertain_param_type))
        s.assign(uncertainty_data)
        # guardrail: catch accidental rebinds
        assert id(uncertainty_data) == did, "Strategy must not rebind the data dict"

def uncertainty_strategy_base_case(
        databases:Union[str, List[str]], 
        method:str, 
        uncertainty_data:UncertaintyData, 
        scaling_factor_if:float=0.5, 
        scaling_factor_cf:float=0.3,
        scaling_factor_var_bounds:float=0.2
        ) -> List[TriangluarBaseStrategy]:
    """
    Creates a list of uncertainty strategies which can be used as a base case for uncertainty analysis.
    The strategies are:
        - TriangularBoundInterpolationStrategy for all intervention flows in the database(s)
        - TriangluarBaseStrategy for the characterization factors in the method
        - TriangluarBaseStrategy for the variable bounds of the variables 'upper_limit', 'lower_limit', 'upper_elem_limit', 'upper_imp_limit'
    
    Args:
        databases (List[str]):
            The name of the database or a list of database names for which the intervention flows will get
            the TriangularBoundInterpolationStrategy assigned to them.
        method (Dict[str, str]):
            The LCIA method for which the characterization factors will get the TriangluarBaseStrategy assigned to them.
        uncertainty_data (UncertaintyData):
            The uncertainty data containing the defined and undefined 
            uncertainty information.
        scaling_factor_if (float):
            The scaling factor which will be used in the TriangluarBaseStrategy for the intervention flows
            if more than 50% of the intervention flows in the database have no uncertainty information.
            Default is 0.5, meaning that the min and max of the triangular distribution will be set to:
            min = amount - 0.5 * abs(amount)
            max = amount + 0.5 * abs(amount)
        scaling_factor_cf (float):
            The scaling factor which will be used in the TriangluarBaseStrategy for the characterization
            factors. Default is 0.3, meaning that the min and max of the triangular distribution will be set to:
            min = amount - 0.3 * abs(amount)
            max = amount + 0.3 * abs(amount)
        scaling_factor_var_bounds (float):
            The scaling factor which will be used in the TriangluarBaseStrategy for the variable bounds.
            Default is 0.2, meaning that the min and max of the triangular distribution will be set to:
            min = amount - 0.2 * abs(amount)
            max = amount + 0.2 * abs(amount)
    Returns:
        strategies (List[UncertaintyStrategyBase]):
            A list of instatialized uncertainty strategies which can be used in the 
            `apply_uncertainty_strategies` function.
    """
    If_strategies = []
    print('Creating base case uncertainty strategies for intervention flows')
    for database in databases:
        if len(uncertainty_data['If'][database]['defined']) ==0 or len(uncertainty_data['If'][database]['undefined'])/len(uncertainty_data['If'][database]['defined']) > 0.5:
            print('More than 50% of the intervention flows in the database {database} have no uncertainty information, the scaling factors are set to: {scaling_factor}.')
            print('\tCreating triangular bound base strategy for intervention flows in database: {}'.format(database))
            If_strategies.append(
                TriangluarBaseStrategy(
                    uncertain_param_type='If',
                    uncertain_param_subgroup=database,
                    upper_scaling_factor = scaling_factor_if,
                    lower_scaling_factor = scaling_factor_if,
                    noise_interval={'min':.1, 'max':.1}
                )
            )
        else:
            print('\tCreating triangular bound interpolation strategy for intervention flows in database: {}'.format(database))
            If_strategies.append(
                TriangularBoundInterpolationStrategy(
                    uncertain_param_type='If',
                    uncertain_param_subgroup=database,
                    noise_interval={'min':.1, 'max':.1}
                )
            )
    print('Creating triangular bound base strategy for Characterization factors: {}'.format(next(iter(method))))
    Cf_strategies = [
        TriangluarBaseStrategy(
            uncertain_param_type='Cf',
            uncertain_param_subgroup=method,
            upper_scaling_factor = scaling_factor_cf,
            lower_scaling_factor = scaling_factor_cf,
            noise_interval={'min':.1, 'max':.1}
        )
    ]
    print('Creating triangular bound base strategy for variable bounds')
    Var_strategies = [
            TriangluarBaseStrategy(
            uncertain_param_type='Var_bounds',
            uncertain_param_subgroup=var_bound,
            upper_scaling_factor=scaling_factor_var_bounds,
            lower_scaling_factor=scaling_factor_var_bounds,
            noise_interval={'min':.2, 'max':.1}
        ) for var_bound in ['upper_limit', 'lower_limit', 'upper_elem_limit', 'upper_imp_limit']
    ]
    strategies = If_strategies + Cf_strategies + Var_strategies
    return strategies

def check_missing_uncertainty_data(uncertainty_data: UncertaintyData) -> bool:
    """
    Check if there are any undefined uncertainty data in the uncertainty_data dict.
    
    Args:
        uncertainty_data (UncertaintyData): 
            Dictionary containing metadata about uncertain intervention flows (IF) and characterization factors (CF).
    
    Returns:
        missing_unc_data (bool): 
            True if there is any undefined uncertainty data, False otherwise.
    """
    missing_unc_data = False
    for unc_type, unc_type_data in uncertainty_data.items():
        for unc_subgroup, unc_subgroup_data in unc_type_data.items():
            if len(unc_subgroup_data['undefined']):
                missing_unc_data = True
                print('{} - {} \n \t {} parameters without uncertainty information'.format(unc_type, unc_subgroup, len(unc_subgroup_data['undefined'])))
    if not missing_unc_data:
        print('No uncertainty data missing.')
    return missing_unc_data


def transform_to_normal(uncertainty_data:UncertaintyData, sample_size:int=100000, plot_distribution:bool=False) -> UncertaintyData:
    """
    Fit Normal distributions to all CF and IF uncertainty metadata.

    Uses the UncertaintyProcessor to convert any non‐normal uncertainty
    definitions into equivalent Normal distributions.

    Args:
        uncertainty_data (UncertaintyData): 
            Dictionary containing metadata about uncertain intervention flows (IF) and characterization factors (CF).
        sample_size (int):
            Number of random draws per parameter when fitting normal distribution
            to uncertain parameters. Defaults to 1_000_000.
        plot_distributions (bool):
            If True, display a histogram + fitted-normal curve for each parameter.
            Defaults to False.
    Returns:
        normal_metadata (UncertaintyData): 
            Fitted Normal loc/scale for parameters in chance constaints (e.g., "cf", "if").
    """
    if check_missing_uncertainty_data(uncertainty_data):
        raise Exception('There is undefined uncertainty data, you can only compute the env. cost statistics when all uncertainty data is defined')
    normal_metadata:UncertaintyData = {}
    for param_type, params_metadata in uncertainty_data.items():
        normal_metadata[param_type] = {}
        for var_name, var_metadata in params_metadata.items():
            normal_metadata[param_type][var_name] = {}
            normal_metadata[param_type][var_name]['defined'] = fit_normals(var_metadata['defined'], sample_size=sample_size, plot_distributions=plot_distribution)
    # ATTN: Check if the fit_normals runs through with 0 as standard deviations
    return normal_metadata

def fit_normals(
        uncertainty_metadata:Dict[Union[Tuple[int,int],int], UncertaintySpec], 
        plot_distributions:bool=False, 
        sample_size:int=1000000
        ) -> Dict[Union[Tuple[int,int],int], UncertaintySpec]:
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
    normal_uncertainty_metadata_dict = {}
    if uncertainty_metadata:
        print('{} parameters with non normal distribution are transformed into normal distributions via max likelihood approximation'.format((pd.DataFrame(uncertainty_metadata).T['uncertainty_type'] != 3).sum()))
    # For each parameter:
    #   - generate random samples from its original distribution
    #   - estimate mean and std via max likelihood fit of the percent‐point function samples (ppf)
    #   - replace in returned DataFrame
    for param_index, metadata in uncertainty_metadata.items():
        if metadata['uncertainty_type'] == 1:
            raise Exception('The intervention flow has the "no uncertainty" distribution type. This is not allowed')
        if metadata['uncertainty_type'] == 3:
            # If the distribution is already normal, just copy the metadata
            normal_uncertainty_metadata_dict[param_index] = {
                'scale':metadata['scale'],
                'loc':metadata['loc'],
                'uncertainty_type':stats_arrays.NormalUncertainty.id
            }
            continue
        metadata_uncertainty_array = stats_arrays.UncertaintyBase.from_dicts(metadata)
        uncertainty_choice = stats_arrays.uncertainty_choices[metadata['uncertainty_type']]
        # Sample the non-normal distribution 
        param_samples = uncertainty_choice.random_variables(metadata_uncertainty_array, sample_size)
        # Calculate the ppf values to 
        percentages = np.expand_dims(np.linspace(0.001, 0.999, 1000, axis=0), axis=0)
        x = uncertainty_choice.ppf(metadata_uncertainty_array, percentages=percentages)
        x, y = uncertainty_choice.pdf(metadata_uncertainty_array, xs=x)
        # ATTN: There is an error in stats_arrays when computing the trinagular pdf, bug is reported (#18), here is the fix.
        if metadata['uncertainty_type'] == 5:
            _, scale = uncertainty_choice.rescale(metadata_uncertainty_array)
            y = y/scale
        # Fit a normal distribution to the sampled data
        loc_norm, scale_norm = scipy.stats.norm.fit(param_samples.T)
        if plot_distributions:
            _, ax = plt.subplots(1, 1)
            # plot the histrogram of the samples
            ax.hist(param_samples.T, density=True, bins='auto', histtype='stepfilled', alpha=0.2, label='{} samples'.format(uncertainty_choice.description))
            # plot the lognormal pdf
            ax.plot(x.flatten(), y.flatten(), 'k-', lw=2, label='frozen {} pdf'.format(uncertainty_choice.description))
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
        normal_uncertainty_metadata_dict[param_index] = normal_uncertainty_metadata
    if plot_distributions:
        plt.show()
    return normal_uncertainty_metadata_dict

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
                options are: "intervention_flow" and "characterization_factor".
        
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
                metadata_df.rename(index=cf_index_map)
            case 'process':
                process_index_map = {process_indx:  '{} '.format(lci_data['process_map_metadata'][process_indx]) for process_indx in metadata_df.index}
                metadata_df = metadata_df.rename(index=process_index_map)
            case _:
                raise Exception(f'"rename_metadata_index" to <<{param_type}>> as "uncertainty_var_name" has not been implemented')
        return metadata_df
