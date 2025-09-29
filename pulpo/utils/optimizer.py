import os
import pandas as pd
import numpy as np
import scipy
import array
from typing import Dict, Tuple
import pyomo.environ as pyo
from pyomo.contrib import appsi
import stats_arrays
from .saver import extract_flows
from .uncertainty import processor
from .uncertainty.preparer import UncertaintyData, UncertaintySpec, DefUndefBlock

def create_model():
    """
    Builds an abstract model on top of the ecoinvent database.

    Returns:
        AbstractModel: The Pyomo abstract model for optimization.
    """
    model = pyo.AbstractModel()

    # Sets
    model.PRODUCT = pyo.Set(doc='Set of intermediate products (or technosphere exchanges), indexed by i')
    model.PROCESS = pyo.Set(doc='Set of processes (or activities), indexed by j')
    model.ENV_COST = pyo.Set(doc='Set of environmental cost flows, indexed by e')
    model.INDICATOR = pyo.Set(doc='Set of impact assessment indicators, indexed by h')
    model.INV = pyo.Set(doc='Set of intervention flows, indexed by g')
    model.ENV_COST_PROCESS = pyo.Set(within=model.PROCESS * model.INDICATOR, doc='Relation set between environmental cost flows and processes')
    model.ENV_COST_IN = pyo.Set(model.INDICATOR, within=model.ENV_COST)
    model.PROCESS_IN = pyo.Set(model.PROCESS, within=model.PRODUCT)
    model.PROCESS_OUT = pyo.Set(model.PRODUCT, within=model.PROCESS)
    model.PRODUCT_PROCESS = pyo.Set(within=model.PRODUCT * model.PROCESS, doc='Relation set between intermediate products and processes')
    model.INV_PROCESS = pyo.Set(within=model.INV * model.PROCESS, doc='Relation set between environmental flows and processes')
    model.INV_OUT = pyo.Set(model.INV, within=model.PROCESS)

    # Parameters
    model.UPPER_LIMIT = pyo.Param(model.PROCESS, mutable=True, within=pyo.Reals, doc='Maximum production capacity of process j')
    model.LOWER_LIMIT = pyo.Param(model.PROCESS, mutable=True, within=pyo.Reals, doc='Minimum production capacity of process j')
    model.UPPER_INV_LIMIT = pyo.Param(model.INV, mutable=True, within=pyo.Reals, doc='Maximum intervention flow g')
    model.UPPER_IMP_LIMIT = pyo.Param(model.INDICATOR, mutable=True, within=pyo.Reals, doc='Maximum impact on category h')
    model.ENV_COST_MATRIX = pyo.Param(model.ENV_COST_PROCESS, mutable=True, doc='Enviornmental cost matrix Q*B describing the environmental cost flows e associated to process j')
    model.INV_MATRIX = pyo.Param(model.INV_PROCESS, mutable=True, doc='Intervention matrix B describing the intervention flow g entering/leaving process j')
    model.FINAL_DEMAND = pyo.Param(model.PRODUCT, mutable=True, within=pyo.Reals, doc='Final demand of intermediate product flows (i.e., functional unit)')
    model.SUPPLY = pyo.Param(model.PRODUCT, mutable=True, within=pyo.Binary, doc='Binary parameter which specifies whether or not a supply has been specified instead of a demand')
    model.TECH_MATRIX = pyo.Param(model.PRODUCT_PROCESS, mutable=True, doc='Technology matrix A describing the intermediate product i produced/absorbed by process j')
    model.WEIGHTS = pyo.Param(model.INDICATOR, mutable=True, within=pyo.NonNegativeReals, doc='Weighting factors for the impact assessment indicators in the objective function')

    # Variables
    model.impacts = pyo.Var(model.INDICATOR, bounds=(-1e24, 1e24), doc='Environmental impact on indicator h evaluated with the established LCIA method')
    model.scaling_vector = pyo.Var(model.PROCESS, bounds=(-1e24, 1e24), doc='Activity level of each process to meet the final demand')
    model.inv_vector = pyo.Var(model.INV, bounds=(-1e24, 1e24), doc='Intervention flows')
    model.slack = pyo.Var(model.PRODUCT, bounds=(-1e24, 1e24), doc='Supply slack variables')

    # Rule functions
    def populate_env(model):
        """Relates the environmental flows to the processes."""
        for j, h in model.ENV_COST_PROCESS:
            if j not in model.ENV_COST_IN[h]:
                model.ENV_COST_IN[h].add(j)

    def populate_in_and_out(model):
        """Relates the inputs of an activity to its outputs."""
        for i, j in model.PRODUCT_PROCESS:
            model.PROCESS_OUT[i].add(j)
            model.PROCESS_IN[j].add(i)

    def populate_inv(model):
        """Relates the impacts to the environmental flows"""
        for a, j in model.INV_PROCESS:
            model.INV_OUT[a].add(j)

    # Building rules for sets
    model.Env_in_out = pyo.BuildAction(rule=populate_env)
    model.Process_in_out = pyo.BuildAction(rule=populate_in_and_out)
    model.Inv_in_out = pyo.BuildAction(rule=populate_inv)

    def demand_constraint(model, i):
        """Fixes a value in the demand vector"""
        return sum(model.TECH_MATRIX[i, j] * model.scaling_vector[j] for j in model.PROCESS_OUT[i]) == model.FINAL_DEMAND[i] + model.slack[i]

    def impact_constraint(model, h):
        """Calculates all the impact categories"""
        return model.impacts[h] == sum(model.ENV_COST_MATRIX[j, h] * model.scaling_vector[j] for j in model.ENV_COST_IN[h])

    def inventory_constraint(model, g):
        """Calculates the environmental flows"""
        return model.inv_vector[g] == sum(model.INV_MATRIX[g, j] * model.scaling_vector[j] for j in model.INV_OUT[g])

    def upper_constraint(model, j):
        """Ensures that variables are within capacities (Maximum production constraint) """
        return model.scaling_vector[j] <= model.UPPER_LIMIT[j]

    def lower_constraint(model, j):
        """ Minimum production constraint """
        return model.scaling_vector[j] >= model.LOWER_LIMIT[j]

    def upper_env_constraint(model, g):
        """Ensures that variables are within capacities (Maximum production constraint) """
        return model.inv_vector[g] <= model.UPPER_INV_LIMIT[g]

    def upper_imp_constraint(model, h):
        """ Imposes upper limits on selected impact categories """
        return model.impacts[h] <= model.UPPER_IMP_LIMIT[h]

    def slack_upper_constraint(model, j):
        """ Slack variable upper limit for activities where supply is specified instead of demand """
        return model.slack[j] <= 1e20 * model.SUPPLY[j]

    def slack_lower_constraint(model, j):
        """ Slack variable upper limit for activities where supply is specified instead of demand """
        return model.slack[j] >= -1e20 * model.SUPPLY[j]

    def objective_function(model):
        """Objective is a sum over all indicators with weights. Typically, the indicator of study has weight 1, the rest 0"""
        return sum(model.impacts[h] * model.WEIGHTS[h] for h in model.INDICATOR)

    # Constraints
    model.FINAL_DEMAND_CNSTR = pyo.Constraint(model.PRODUCT, rule=demand_constraint)
    model.IMPACTS_CNSTR = pyo.Constraint(model.INDICATOR, rule=impact_constraint)
    model.INVENTORY_CNSTR = pyo.Constraint(model.INV, rule=inventory_constraint)
    model.UPPER_CNSTR = pyo.Constraint(model.PROCESS, rule=upper_constraint)
    model.LOWER_CNSTR = pyo.Constraint(model.PROCESS, rule=lower_constraint)
    model.SLACK_UPPER_CNSTR = pyo.Constraint(model.PRODUCT, rule=slack_upper_constraint)
    model.SLACK_LOWER_CNSTR = pyo.Constraint(model.PRODUCT, rule=slack_lower_constraint)
    model.INV_CNSTR = pyo.Constraint(model.INV, rule=upper_env_constraint)
    model.IMP_CNSTR = pyo.Constraint(model.INDICATOR, rule=upper_imp_constraint)

    # Objective function
    model.OBJ = pyo.Objective(sense=pyo.minimize, rule=objective_function)

    return model





def calculate_methods(instance, lci_data, methods):
    """
    Calculates the impacts if a method with weight 0 has been specified.

    Args:
        instance: The Pyomo model instance.
        lci_data (dict): LCI data containing matrices and mappings.
        methods (dict): Methods for environmental impact assessment.

    Returns:
        instance: The updated Pyomo model instance with calculated impacts.
    """
    # Filter matrices for specified methods
    matrices = {h: lci_data['matrices'][h] for h in lci_data['matrices'] if str(h) in methods}
    intervention_matrix = lci_data['intervention_matrix']

    # Calculate environmental costs
    env_cost = {h: matrices[h] @ intervention_matrix for h in matrices}

    # Extract scaling vector
    scaling_vector = extract_flows(instance, lci_data['process_map'], lci_data['process_map_metadata'], 'scaling').sort_index()
    scaling_values = scaling_vector['Value'].to_numpy()

    # Calculate impacts
    impacts = {h: (env_cost[h] @ scaling_values).sum() for h in matrices}

    # Update or create impacts_calculated variable
    if hasattr(instance, 'impacts_calculated'):
        for h, value in impacts.items():
            instance.impacts_calculated[h].value = value
    else:
        instance.impacts_calculated = pyo.Var(impacts.keys(), initialize=impacts)

    return instance

def compute_L1_env_cost_mean_var(
        normal_uncertainty_data:UncertaintyData, 
        lci_data:dict, 
        method:str, 
        plot_analysis_support_plots=False
        ) -> Dict[Tuple[int,str], UncertaintySpec]:
    """
    Computes the environmental cost mean and variance associated with 
    the uncertain intervention and characterization flows specified in uncertainty_data

    This is a shortcut approach to implement an individual chance‐constraint formulation 
    on the objective using the L1 norm on normally distributed uncertainties.

    This method approximates all uncertain intervention flows and characterization factors as Normal(μ,σ²),
    computes the aggregated standard deviation of total environmental cost under an L1 norm.

    Args:
        normal_uncertainty_data (UncertaintyData): 
            Dictionary containing metadata about uncertain intervention flows (IF) and characterization factors (CF).
            All uncertain paramters need to have been transformed to normal distributions.
        lci_data (dict): 
            Dictionary containing life cycle inventory data, including matrices and mappings.
        method (str): 
            The impact assessment method for which to compute environmental cost statistics.
        sample_size_normal_fit (int, optional): 
            Number of samples to use when fitting normal distributions to uncertainty data. Default is 100000.
        plot_analysis_support_plots (bool, optional): 
            If True, plots the fitted normal distributions for inspection. Default is False.

    Returns:
        normal_metadata_env_cost (Dict[Tuple[int,str], UncertaintySpec]):
            dictionary holding the mean, std as loc and scale for all environmental cost subjected to uncertainty.
            With (method, process_id) as keys, as indexed in pyomo.
    """
    def _check_all_uncertainty_is_normal(uncertainty_data:UncertaintyData, method:str):
        """
        Checks that all uncertainty specs in 'If' and 'Cf' have Normal uncertainty type.

        Args:
            uncertainty_data (UncertaintyData): 
                Dictionary containing metadata about uncertain intervention flows (IF) and characterization factors (CF).
            method (str): 
                The impact assessment method for which to check CF uncertainty.

        Raises:
            ValueError: If any uncertainty spec is not of Normal type.
        """
        normal_id = stats_arrays.NormalUncertainty.id
        # Check 'If'
        for if_data in uncertainty_data['If'].values():
            for spec in if_data['defined'].values():
                if spec.get('uncertainty_type', None) != normal_id:
                    raise ValueError("All 'If' uncertainty specs must be Normal distributions.")
        # Check 'Cf'
        for spec in uncertainty_data['Cf'][method]['defined'].values():
            if spec.get('uncertainty_type', None) != normal_id:
                raise ValueError("All 'Cf' uncertainty specs must be Normal distributions.")
    
    def _extract_process_ids_and_intervention_flows_for_env_cost_variance(uncertainty_data:UncertaintyData, lci_data:dict, method:str) -> tuple[array.array, pd.DataFrame]:
        """
        Identify which processes and flows feed into the environmental-cost variance.

        Extracts the array of process IDs that have uncertain intervention flows or CFs,
        then computes per-process cost standard deviations and z-scores, printing any
        outliers and plotting the z-value distribution for inspection.

        Args:
            uncertainty_data (UncertaintyData): 
                Dictionary containing metadata about uncertain intervention flows (IF) and characterization factors (CF).
            lci_data (dict): 
                Dictionary containing life cycle inventory data, including matrices and mappings.
            method (str): 
                The impact assessment method for which to compute environmental cost statistics.

        Returns:
            process_id_uncertain_if (array.array): 
                IDs of processes with uncertain IF contributions.
            envcost_std_mean (pd.DataFrame): 
                DataFrame indexed by process ID with columns
                ['std', 'mean', 'z', 'metadata'] summarizing variance diagnostics.
        """
        # To Compute the variance of the environmental costs we must extract all processes which contain:
        # - an uncertain intervention flow
        process_id_uncertain_if = []
        for if_unc_data in uncertainty_data['If'].values():
            process_id_uncertain_if += [if_indx for (_, if_indx) in if_unc_data['defined'].keys()]
        # - an intervention flow associated with an uncertain characterization factor
        Cf_indcs = list(uncertainty_data['Cf'][method]['defined'].keys())
        process_id_associated_cf = lci_data['intervention_matrix'][Cf_indcs,:].nonzero()[1]
        process_ids = np.unique(np.append(process_id_associated_cf, process_id_uncertain_if))
        # Get the intervention flows to the uncertain characterization factors
        intervention_flows_extracted = pd.DataFrame.sparse.from_spmatrix(
            lci_data['intervention_matrix'][Cf_indcs,:][:,process_ids],
            index=Cf_indcs,
            columns=process_ids
        )
        return process_ids, intervention_flows_extracted

    def _compute_envcost_variance(normal_uncertainty_data:UncertaintyData, lci_data, method) -> dict:
        """
        Calculate the standard deviation of total environmental cost across processes.
        This calculation assumes that we can use the L1 norm to compute the variance
        of the total environmental cost, which is a sum of products of uncertain IFs and CFs.

        Uses the fitted Normal distributions for CFs and IFs to derive per-process
        cost variances, then aggregates them (under independence) to obtain the
        overall cost standard deviations.
        $$
        \sigma_{q_hb_j} =\sqrt{\sum_e \big(\mu_{q_{h,e}}^2\sigma_{b_{e,j}}^2 + \mu_{b_{e,j}}^2\sigma_{q_{h,e}}^2 + \sigma_{b_{e,j}}^2 \sigma_{q_{h,e}}^2\big)}
        $$

        Args:
            uncertainty_data (UncertaintyData): 
                Dictionary containing metadata about uncertain intervention flows (IF) and characterization factors (CF).
            lci_data (dict): 
                Dictionary containing life cycle inventory data, including matrices and mappings.
            method (str): 
                The impact assessment method for which to compute environmental cost statistics.    

        Returns:
            envcost_std (dict):
                Indexed by process ID, with each value equal to the standard deviation
                of that process’s total cost contribution.
        """
        # Create dataframes for the uncertainty data If and Cf, as this method was written with another uncertainty_data sturcture
        # ATTN: rewrite this method to fully be based on the UncertaintyData structure
        if_unc_dict = {}
        for if_uncertainty_data in normal_uncertainty_data['If'].values():
            if_unc_dict.update(if_uncertainty_data['defined'])
        if_normal_metadata_df = pd.DataFrame(if_unc_dict).T
        cf_normal_metadata_df = pd.DataFrame(normal_uncertainty_data['Cf'][method]['defined']).T
        # Get the process and intervention flow Id's for the env costs.
        process_ids, intervention_flows_extracted = _extract_process_ids_and_intervention_flows_for_env_cost_variance(normal_uncertainty_data, lci_data, method)
        envcost_std = {}
        for process_id in process_ids:
            # compute the mu_{q_{h,e}}^2 * sigma_{b_{e,j}}^2
            if process_id in if_normal_metadata_df.index.get_level_values(level=1):
                intervention_flow_std = if_normal_metadata_df.xs(process_id, level=1, axis=0, drop_level=True)['scale']
                characterization_factor_mean = pd.Series(
                    lci_data["matrices"][method].diagonal()[
                        intervention_flow_std.index.get_level_values(level=0)
                        ],
                    index=intervention_flow_std.index.get_level_values(level=0)
                )
                # Reindex so that we can perform a matrix multiplication on all intervention flows
                characterization_factor_mean = characterization_factor_mean.reindex(intervention_flow_std.index, axis=0, level=0)
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
            if (intervention_flows_extracted[process_id] > 0).any() and process_id in if_normal_metadata_df.index.get_level_values(level=1):
                sigma_q2_sigma_b2 = characterization_factor_std.pow(2).mul(intervention_flow_std.pow(2))
            else:
                sigma_q2_sigma_b2 = pd.Series([0])
            # Take the sqrt of the sum over the std terms and the intervention flows
            envcost_std[process_id] = np.sqrt(mu_q2_sigma_b2.sum() + sigma_q2_sigma_b2.sum() + sigma_q2_mu_b2.sum())
        return envcost_std

    def _compute_envcost_mean(lci_data:dict) -> dict:
        """
        Compute the expected (mean) total environmental cost per process.

        Args:
            lci_data (dict): 
                Dictionary containing life cycle inventory data, including matrices and mappings.

        Returns:
            envcost_mean (pd.Series):
                environmental cost mean values, Indexed by process ID, with each value equal to the expected cost
                contribution of that process.
        """
        # Compute the mean of the environmental costs to be used together with the standard deviation to update the uncertain parameters in line with chance constraint formulation
        envcost_raw = lci_data['matrices'][method].diagonal() @ lci_data['intervention_matrix']
        # ATTN: The env_cost_raw must be updated with the potentially different means after fitting the normal distribution
        envcost_mean = pd.Series(envcost_raw).to_dict()
        return envcost_mean

    def _check_envcost_variance(envcost_std:dict, envcost_mean:dict, lci_data:dict, box_plots:bool=False):
        """
        Validate z-scores (std/mean) for environmental cost contributions.

        Computes z = std/|mean| for each process and raises a warning or error
        if any z exceed reasonable thresholds (indicating potential numerical issues).

        Args:
            envcost_std (dict):
                Standard deviation per process (from `compute_envcost_variance`).
            envcost_mean (dict):
                environmental cost mean values, Indexed by process ID, with each value equal to the expected cost
                contribution of that process.
            lci_data (dict):
                Dictionary containing life cycle inventory data, including matrices and mappings.
            box_plots (bool, optional):
                If true show the box plot of the z-values computed for the standard deviations
        """
        # ATTN: For the environmental costs with very large z-value we should check if they come from interpolated values or from database uncertainty
        envcost_std_mean = pd.DataFrame.from_dict(envcost_std, orient='index', columns=['std'])
        envcost_std_mean['metadata'] = envcost_std_mean.index.map(lci_data['process_map_metadata'])
        if envcost_std_mean['std'].isna().any():
            raise Exception('There are NaNs in the standard deviation')
        envcost_std_mean['mean'] = envcost_std_mean.index.map(envcost_mean)
        envcost_std_mean['z'] = envcost_std_mean['std'] / envcost_std_mean['mean']
        if (envcost_std_mean['z'] > 0.5).any():
            print('These environmental costs have a standard deviation larger than 50% of their mean:\n')
            print(envcost_std_mean[envcost_std_mean['z'] > 0.5].sort_values('z', ascending=False))
            # raise Exception('There are z-values greater than 0.5 this is improbable')
        if box_plots:
            envcost_std_mean['z'].sort_values(ascending=False).iloc[5:].plot.box()
            print('The following points were excluded from the boxplot:')
            print(envcost_std_mean['z'].sort_values(ascending=False).iloc[:5])
    
    """
    Prepare the variance‐based chance‐constraint formulation.

    1. Check that CF and IF are Normally distributed.
    2. Compute the standard deviation of environmental cost contributions.
    3. Compute the mean environmental cost.
    4. Check that the variance‐based z‐values are within acceptable bounds.
    """
    _check_all_uncertainty_is_normal(normal_uncertainty_data, method)
    envcost_std = _compute_envcost_variance(normal_uncertainty_data, lci_data, method)
    envcost_mean = _compute_envcost_mean(lci_data)
    _check_envcost_variance(envcost_std, envcost_mean, lci_data, box_plots=plot_analysis_support_plots)
    normal_metadata_env_cost: Dict[Tuple[int,str], UncertaintySpec] = {
        (process_id, method): {
            'loc':envcost_mean[process_id], 
            'scale':envcost_std[process_id],
            'uncertainty_type':stats_arrays.NormalUncertainty.id,
            'amount':np.NaN,
            'maximum':np.NaN,
            'minimum':np.NaN,
            'shape':np.NaN,
        } for process_id in envcost_std.keys()
    }
    return normal_metadata_env_cost

def apply_CC_formulation(
        model_instance, 
        lambda_level:float, 
        normal_metadata_env_cost:Dict[Tuple[int,str], UncertaintySpec]={}, 
        normal_metadata_var_bounds:Dict[str, Dict[int, UncertaintySpec]]={}
        ):
    """
    Inject or update the ε‐constraint for a given risk level.

    Modifies the existing Pyomo model to enforce that the specified
    chance constraint (e.g. P{impact ≤ threshold} ≥ λ) is satisfied
    at the current `lambda_level`. This supports tracing the Pareto front.

    Args:
        model_instance (ConcreteModel): 
            The Pyomo model instance.
        lambda_level (float):
            Target confidence/risk threshold (e.g., 0.95 for 95% quantile).
        normal_metadata_env_cost (Dict[Tuple[int,str], UncertaintySpec]):
            Dictionary holding the mean, std as loc and scale for all environmental cost subjected to uncertainty.
            With (method, process_id) as keys, as indexed in pyomo.
        normal_metadata_var_bounds (str, UncertaintySpec]):
            Dictionary holding the mean, std as loc and scale for all variable bounds subjected to uncertainty.
            With ('upper_limit' or 'lower_limit', process_id) as keys, as indexed in pyomo.
        
    """
    ppf_lambda = scipy.stats.norm.ppf(lambda_level)
    if normal_metadata_env_cost:
        print(f'Applying CC constraints to the environmental cost calculation with lambda: {lambda_level}')
        environmental_cost_updated = {env_cost_indx: env_cost_data['loc'] + ppf_lambda * env_cost_data['scale'] for env_cost_indx, env_cost_data in normal_metadata_env_cost.items()}
        model_instance.ENV_COST_MATRIX.store_values(environmental_cost_updated, check=True)
    for bound_name, metadata_vb in normal_metadata_var_bounds.items():
        if metadata_vb:
            print(f'Applying CC constraints to the {bound_name} constraint with lambda: {lambda_level}')
            match bound_name:
                case 'upper_limit':
                    pyomo_var_name = 'UPPER_LIMIT'
                    bound_updated = {indx: (unc_data['loc'] - ppf_lambda * unc_data['scale']) for indx, unc_data in metadata_vb.items()}
                case 'lower_limit':
                    pyomo_var_name = 'LOWER_LIMIT'
                    bound_updated = {indx: (unc_data['loc'] + ppf_lambda * unc_data['scale']) for indx, unc_data in metadata_vb.items()}
                case 'upper_imp_limit': # ATTN: Not tested
                    pyomo_var_name = 'UPPER_IMP_LIMIT'
                    bound_updated = {indx: (unc_data['loc'] - ppf_lambda * unc_data['scale']) for indx, unc_data in metadata_vb.items()}
                case 'upper_inv_limit': # ATTN: Not tested
                    pyomo_var_name = 'UPPER_INV_LIMIT'
                    bound_updated = {indx: (unc_data['loc'] - ppf_lambda * unc_data['scale']) for indx, unc_data in metadata_vb.items()}
                case _:
                    raise Exception('has not been implemented yet.')
            pyomo_bound = getattr(model_instance, pyomo_var_name)
            pyomo_bound.store_values(bound_updated, check=True)

def calculate_inv_flows(instance, lci_data):
    """
    Calculates elementary flows post-optimization.

    Args:
        instance: The Pyomo model instance.
        lci_data (dict): LCI data containing matrices and mappings.

    Returns:
        instance: The updated Pyomo model instance with calculated intervention flows.
    """
    # Extract intervention matrix and scaling vector
    intervention_matrix = lci_data['intervention_matrix']
    scaling_vector = extract_flows(instance, lci_data['process_map'], lci_data['process_map_metadata'], 'scaling').sort_index()
    scaling_values = scaling_vector['Value'].to_numpy()

    # Calculate intervention flows
    flows = intervention_matrix @ scaling_values

    # Update or create inv_flows variable
    if hasattr(instance, 'inv_flows'):
        for i, flow_value in enumerate(flows):
            instance.inv_flows[i].value = flow_value
    else:
        instance.inv_flows = pyo.Var(range(len(flows)), initialize=dict(enumerate(flows)))

    return instance


def instantiate(model_data):
    """
    Builds an instance of the optimization model with specific data and objective function.

    Args:
        model_data (dict): Data dictionary for the optimization model.

    Returns:
        ConcreteModel: The instantiated Pyomo model.
    """
    print('Creating Instance')
    model = create_model()
    problem = model.create_instance(model_data, report_timing=False)
    print('Instance created')
    return problem


def get_cplex_options(options):
    # ATTN: Write some instructions on how to tune these parameters. For now, they are set to work for standard ecoinvent problems with CPLEX.
    """Return the default XPLEX options if none are provided."""
    default_options = [
        'option optcr = 1e-15;',
        'option reslim = 3600;',  # Time limit
        'GAMS_MODEL.optfile = 1;',
        '$onecho > cplex.opt',
        'workmem=4096',
        'scaind=1',
        '$offecho',
    ]
    return options if options is not None else default_options

def solve_highspy(model_instance):
    """Solve the model using Highspy."""
    opt = appsi.solvers.Highs()
    results = opt.solve(model_instance)
    if results.termination_condition == appsi.base.TerminationCondition.optimal: 
        print('optimal solution found: ', results.best_feasible_objective) 
        results.solution_loader.load_vars() 
    elif results.best_feasible_objective is not None: 
        print('sub-optimal but feasible solution found: ', results.best_feasible_objective) 
    elif results.termination_condition in {appsi.base.TerminationCondition.maxIterations, appsi.base.TerminationCondition.maxTimeLimit}: 
        print('No feasible solution was found. The best lower bound found was ', results.best_objective_bound) 
    else: 
        print('The following termination condition was encountered: ', results.termination_condition) 
        print('Optimization problem solved using Highspy')
    return results, model_instance

def solve_neos(model_instance, solver_name, options, neos_email):
    """Solve the model using NEOS."""
    if neos_email is not None:
        os.environ['NEOS_EMAIL'] = neos_email

    if 'NEOS_EMAIL' not in os.environ:
        print("'NEOS_EMAIL' environment variable is not set. \n")
        print("To use the NEOS solver, please set the 'NEOS_EMAIL' environment variable as explained here:\n")
        print("https://www.twilio.com/en-us/blog/how-to-set-environment-variables-html \n")
        print("If you do not have a NEOS account, please create one at https://neos-server.org/neos/ \n")
        print("Alternatively, you can pass the 'neos_email' argument to the solve function. \n")
        return None, model_instance
    solver_manager = pyo.SolverManagerFactory('neos')
    # ATTN: deleted the 'options' use as kwargs, since I do not think it makes sense, it holds options for the PULPO solver and for the pyomo solver_manager, 
    # it needs to be either different options or completely differently structured. Now I have hard programmed the seetings.
    #  Also solver_name is a solver_manager option, it kind of does not make sense
    results = solver_manager.solve(model_instance, opt=solver_name, tee=True)
    if not results.solver.termination_condition == pyo.TerminationCondition.optimal:
        raise Exception('Could not find an optimal solutions to the problem.')

    print("Optimization problem solved using NEOS")
    return results, model_instance

def solve_gams(model_instance, gams_path, options, solver_name=None):
    """Solve the model using GAMS with either CPLEX or an alternative solver."""
    if gams_path is True:
        gams_path = os.getenv('GAMS_PULPO')
        if gams_path:
            print('GAMS path retrieved from GAMS_PULPO environment variable:', gams_path)
        else:
            print("GAMS path not found. Set the 'GAMS_PULPO' environment variable to your GAMS path or pass it explicitly.")
            return None, model_instance

    solver = pyo.SolverFactory('gams')
    if not solver.available():
        print("GAMS solver is not available. Ensure GAMS is installed and the path is correct.")
        return None, model_instance

    io_options = {'solver': solver_name or 'CPLEX'}
    options = get_cplex_options(options) if solver_name is None else options

    results = solver.solve(
        model_instance,
        keepfiles=False,
        symbolic_solver_labels=True,
        tee=False,
        report_timing=False,
        io_options=io_options,
        add_options=options,
    )
    print('Optimization problem solved using GAMS')
    return results, model_instance


def solve_gurobi(model_instance, options=None):
    """
    Solve the given Pyomo ConcreteModel using Gurobi.
    Captures:
      - model_instance.solver_status
      - model_instance.solver_termination
      - model_instance.best_feasible_obj (if available)
      - model_instance.best_obj_bound    (if available)
    Then, if truly optimal, the Pyomo vars are already loaded (no extra loader needed).
    """
    # Create the Gurobi solver plugin
    solver = pyo.SolverFactory('gurobi')

    """
    Recommended Gurobi tweaks for high-precision LP/QP runs

        options = {
            "FeasibilityTol": 1e-9,   # < tighter constraints (default 1e-6)
            "OptimalityTol" : 1e-9,   # < tighter dual/primal gap
            "BarConvTol"    : 1e-9,   # < stricter barrier convergence
            "NumericFocus"  : 3,      # > robust numerics (quad pivots, careful cuts)
            "ScaleFlag"     : 2       # > geometric scaling for better conditioning
        }

    These values keep residuals ~1×10⁻⁹ (enough for 6-8 significant-digit LCA
    results) while guarding against ill-scaled data.  Add extras like
    `"TimeLimit": 600` or `"MIPGap": 1e-8` to the same dict.
    """
    
    if options:
        for key, val in options.items():
            solver.options[key] = val

    # Solve. The results object is a standard Pyomo SolverResults.
    results = solver.solve(
        model_instance,
        tee=False,               
        load_solutions=True      
    )

    # Capture solver status and termination condition on the model instance:
    model_instance.solver_status      = results.solver.status
    model_instance.solver_termination = results.solver.termination_condition

    # If Gurobi found a feasible or optimal solution, you can also read:
    try:
        obj_val = results.problem.lower_bound if model_instance.OBJ.sense == pyo.minimize else results.problem.upper_bound
        model_instance.best_obj_bound = obj_val
    except Exception:
        model_instance.best_obj_bound = None

    try:
        model_instance.best_feasible_obj = results.problem.upper_bound if model_instance.OBJ.sense == pyo.minimize else results.problem.lower_bound
    except Exception:
        model_instance.best_feasible_obj = None

    print("Optimization problem solved using gurobi")
    print(f"status={results.solver.status}, termination={results.solver.termination_condition}")
    return results, model_instance

def solve_model(model_instance, gams_path=False, solver_name=None, options=None, neos_email=None):
    """
    Solves the instance of the optimization model using Highspy, NEOS, or GAMS.

    Args:
        model_instance (ConcreteModel): The Pyomo model instance.
        gams_path (str or bool, optional): Path to the GAMS solver or True to use the environment variable.
        solver_name (str, optional): The solver to use (e.g. 'cplex', 'baron', or 'xpress').
        options (list, optional): Additional options for the solver.
        neos_email (str, optional): Email for NEOS solver authentication.

    Returns:
        tuple: Results of the optimization and the updated model instance.
    """
    # ATTN: Cases may be too convoluted. Tidy up the logic eventually.
    # Case 1: Use Highspy if no GAMS path is provided and the solver is either not specified or is 'highs'
    if gams_path is False and (solver_name is None or 'highs' in solver_name.lower()):
        return solve_highspy(model_instance)
    
    # Case 2: Gurobi if no GAMS and solver_name == "gurobi"
    if gams_path is False and solver_name and solver_name.lower() == "gurobi":
        return solve_gurobi(model_instance, options=options)

    # Case 3: Use NEOS if a solver_name is provided (and it is not Highspy) and no GAMS path is provided
    if gams_path is False and solver_name and ('highs' not in solver_name.lower()):
        return solve_neos(model_instance, solver_name, options, neos_email)

    # Case 4: Use GAMS if gams_path is specified (either as a path or True)
    if gams_path:
        return solve_gams(model_instance, gams_path, options)

    # Default case: Return None if no valid solver configuration is found
    print("No valid solver configuration found.")
    return None, model_instance
