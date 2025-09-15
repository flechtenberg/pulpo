"""
processor.py

Module for processing the uncertainty data, by filling in missing data,
updating data or comupting metrics from the uncertainty data.
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

class UncertaintyStrategyBase:
    """
    Base class for uncertainty distribution specification strategies.

    This abstract base defines the interface and common functionality for assigning probability
    distributions to parameters lacking predefined uncertainty metadata. Subclasses must implement
    methods to derive distribution parameters for these unspecified uncertainties.

    Attributes:
        metadata_df (pd.DataFrame): 
            DataFrame containing parameter metadata.
        defined_uncertainty_metadata (dict): 
            Mapping of parameter indices to their defined uncertainty metadata.
        undefined_uncertainty_indices (list): 
            List of indices needing distribution assignment.
    """
    def __init__(self, metadata_df:pd.DataFrame, undefined_uncertainty_indices:list, *args, **kwargs):
        """
        Initialize the UncertaintyStrategyBase with metadata and index lists.

        Args:
            metadata_df (pd.DataFrame): 
                The full metadata DataFrame for parameters.
            undefined_uncertainty_indices (list): 
                List of parameter indices without defined uncertainties.
            *args: 
                additional arguments passed to the assign method
            **kwargs: 
                additional optional arguments passed to the assign method
        """
        self.metadata_df = metadata_df # ATTN: rename to param_metadata_df
        self.undefined_uncertainty_indices = undefined_uncertainty_indices
        self.metadata_assigned_df = self.assign(*args, **kwargs)

    def add_random_noise_to_scaling_factor(self, scaling_factor:Union[float, list], low:float, high:float) -> list:
        """
        Adds random noise from uniform distribution to scaling factors, to avoid structured randomness when scaling the data.
        Multiplies the scaling factors in the array of scaling factors with 1-low and 1+high 
        to generate noisy scaling vectors given by the interval [1-low, 1+high].

        Args:
            scaling_factor (float, array, list): 
                the scaling factor which will get noise added to it
            low (float): 
                the lower bound (1-low) for which the noise will be sampled and multiplied to the scaling_factor.
            high (float): 
                the lower bound (1+high) for which the noise will be sampled and multiplied to the scaling_factor.

        Returns:
            scaling_factor_randomized (list): 
                The scaling factor, now a list superposed with the random noise

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
        metadata_asigned_df = pd.DataFrame([])
        return metadata_asigned_df

class ExpertKnowledgeStrategy(UncertaintyStrategyBase):
    """
    Adds specific probability distribution informaiton to uncertain parameters based on expert judgement.
    """
    def __init__(self, metadata_df, undefined_uncertainty_indices, prob_metadata:Dict[int, Dict[str,Union[int, float]]]):
        """
        Initialize the ExpertKnowledgeStrategy with 'prob_metadata' containing the expert knowledge 
        and index lists with the indexes of the parameter who's uncertainty information should be
        replaced with the data in prob_metadata.

        Args:
            metadata_df (pd.DataFrame): 
                The metadata DataFrame for the uncertainty parameters, for which the selected
                parameters in 'undefined_uncertainty_indices' will get an updated uncertainty information
            undefined_uncertainty_indices (list): 
                List of parameter indices without defined uncertainties.
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
        super().__init__(metadata_df, undefined_uncertainty_indices, prob_metadata)  


    def assign(self, prob_metadata):
        """
        Updated the parameters uncertainty information in the metadata_df
        with expert uncertainty information found in the prob_metadata dict.

        Args:
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

        Returns:
            metadata_asigned_df: 
                The updated self.metadata_df containing the expert uncertainty information.
        """
        metadata_asigned_df = self.metadata_df.copy()
        # Roughly checks the probability metadata by creating an uncertainty object in stats_array 
        # which will only take the correct keys out of the prob_metadata dict
        # It can still have missing data or wrong values
        prob_metadata_stats_array = stats_arrays.UncertaintyBase.from_dicts(*prob_metadata.values())
        prob_metadata_df= pd.DataFrame(prob_metadata_stats_array, index=prob_metadata.keys())
        # Writes the expert uncertainty information into the metadata_asigned_df
        for indx in self.undefined_uncertainty_indices:
            for unc_attr in prob_metadata[indx].keys():
                metadata_asigned_df.loc[indx, unc_attr] = prob_metadata_df.loc[indx,unc_attr]
        return metadata_asigned_df
class UniformBaseStrategy(UncertaintyStrategyBase):
    """
    Strategy that assigns uniform distributions to parameters with undefined uncertainty information.

    For each parameter index in undefined_uncertainty_indices, the staretgy sets min and and max based on
    the specified scaling factors. 
    """
    def __init__(
            self, 
            metadata_df, 
            undefined_uncertainty_indices, 
            upper_scaling_factor, 
            lower_scaling_factor, 
            noise_interval:Dict[str,float]={'min':0., 'max':0.}
            ) -> None:
        """
        Initialize the UniformBaseStrategy with metadata and index lists and scaling factors.

        Args:
            metadata_df (pd.DataFrame):
                The full metadata DataFrame for parameters.
            undefined_uncertainty_indices (list): 
                List of parameter indices without defined uncertainties.
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
        super().__init__(
            metadata_df, 
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
            upper_scaling_factor (float): 
                Scaling factor to determine the upper bound relative to the median.
            lower_scaling_factor (float): 
                Scaling factor to determine the lower bound relative to the median.

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

    Attributes:
        Inherits metadata_df, defined_uncertainty_metadata, and undefined_uncertainty_indices from base class.
    """
    def __init__(
            self, 
            metadata_df, 
            undefined_uncertainty_indices, 
            upper_scaling_factor:float, 
            lower_scaling_factor:float, 
            noise_interval:Dict[str,float]={'min':0., 'max':0.}
            ) -> None:
        """
        Initialize the TriangluarBaseStrategy with metadata and index lists and scaling factors.

        Args:
            metadata_df (pd.DataFrame): 
                The full metadata DataFrame for parameters.
            undefined_uncertainty_indices (list): 
                List of parameter indices without defined uncertainties.
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
        super().__init__(
            metadata_df, 
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
            upper_scaling_factor (float): 
                Scaling factor to determine the upper bound relative to the median.
            lower_scaling_factor (float): 
                Scaling factor to determine the lower bound relative to the median.
            noise_interval (Dict[str,float]): 
                Dict containing "min" and "max" keywords holding the upper and lower bound 
                of the noise generated with a uniform distribution and multiplied with 
                the scaling factor vector as (1-min) and (1+max)

        Returns:
            metadata_df (pd.DataFrame(): 
                Updated metadata DataFrame including 'loc', 'minimum', 'maximum',
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

    Attributes:
        Inherits metadata_df and undefined_uncertainty_indices from base class.
        defined_uncertainty_metadata set in this class upon initilialization
    """
    def __init__(self, metadata_df, undefined_uncertainty_indices, defined_uncertainty_metadata, noise_interval:Dict[str,float]={'min':0., 'max':0.}):
        """
        Initialize the TriangularBoundInterpolationStrategy with metadata and index lists.

        Args:
            metadata_df (pd.DataFrame): 
                The full metadata DataFrame for parameters.
            undefined_uncertainty_indices (list): 
                List of parameter indices without defined uncertainties.
            defined_uncertainty_metadata (dict): 
                Dictionary mapping indices to existing uncertainty metadata.
            noise_interval (Dict[str,float]): 
                Dict containing "min" and "max" keywords 
                holding the upper and lower bound of the noise generated with a uniform distribution 
                and multiplied with the scaling factor vector as (1-min) and (1+max)
        """
        self.defined_uncertainty_metadata = defined_uncertainty_metadata
        UncertaintyStrategyBase.__init__(self, metadata_df, undefined_uncertainty_indices, noise_interval=noise_interval)

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

        Returns:
            metadata_asigned_df (pd.DataFrame):
                Updated metadata DataFrame with the interpolated triangular uncertainty information.
                
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
        normal_uncertainty_metadata_dict = {}
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
        return pd.DataFrame(normal_uncertainty_metadata_dict).T
    
    @staticmethod
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

    @staticmethod
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
