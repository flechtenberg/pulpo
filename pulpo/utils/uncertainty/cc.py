"""
cc.py

Chance-constrained formulation helpers. Moved out of pulpo.utils.optimizer so
that the core optimizer module has no hard dependency on stats_arrays or on
the uncertainty sub-package.

These functions are only imported by the uncertainty-enabled optimizer facade
(`pulpo.pulpo_unc`).
"""

import array
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats
import stats_arrays

from pulpo.utils import optimizer, scaling as _scaling
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
        intervention_matrix_updated = lci_data['intervention_matrix'].tolil(copy=True)
        for If_db in normal_uncertainty_data['If'].keys():
            normal_means = pd.DataFrame.from_dict(normal_uncertainty_data['If'][If_db]['defined']).T['loc']
            for (intervention_idx, process_id), mean_value in normal_means.items():
                intervention_matrix_updated[intervention_idx, process_id] = mean_value
        envcost_mean_array = np.asarray(Cf_means.values) @ intervention_matrix_updated.tocsr()
        envcost_mean_values = np.asarray(envcost_mean_array).ravel()
        envcost_mean = {process_id: float(value) for process_id, value in enumerate(envcost_mean_values)}
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
            'amount': np.nan,
            'maximum': np.nan,
            'minimum': np.nan,
            'shape': np.nan,
        } for process_id in envcost_std.keys()
    }
    return normal_metadata_env_cost


@dataclass(frozen=True)
class RiskBudget:
    """A total failure budget ``eps = 1 - lambda`` split across ``K`` events.

    Imposing each chance constraint at ``lambda`` individually controls no joint
    probability: with ``K`` rows each allowed to fail with probability ``eps``,
    the chance that at least one fails reaches ``K * eps``. Boole's inequality
    repairs this - allocate ``eps_k = w_k * eps`` with ``sum(w_k) = 1`` and the
    union of the failures is bounded by ``eps``, so every row holds *together*
    with probability at least ``lambda``.

    The union bound needs only marginals, so no correlation between the events
    has to be estimated, and nothing about the problem class changes: each row
    is still a constant on the right-hand side, computed before the solve.

    ``weights[0]`` is the impact target; ``weights[1:]`` are the variable-bound
    rows in the deterministic order ``apply_CC_formulation`` assigns them. The
    weights are parameters fixed before the solve - choosing them after seeing a
    solution would make the budget a function of the decision it certifies.
    """

    lambda_level: float
    K: int
    weights: Tuple[float, ...]

    @property
    def epsilon(self) -> float:
        """The total failure budget being divided."""
        return 1.0 - self.lambda_level

    @property
    def lambda_impact(self) -> float:
        """Level for the impact target: ``1 - w_0 * eps``, replacing ``lambda``."""
        return 1.0 - self.weights[0] * self.epsilon

    def epsilon_at(self, position: int) -> float:
        """The share of the budget allocated to the event at ``position``."""
        return self.weights[position] * self.epsilon


def bonferroni_budget(lambda_level: float, K: int,
                      weights=None) -> RiskBudget:
    """Split ``1 - lambda_level`` across ``K`` events, equally unless told otherwise.

    An equal split is the default because it is neutral: it needs no
    justification and cannot be read as tuned to produce a result. Unequal
    weights are valid - Boole's inequality only requires them to sum to one -
    and are worth running as a sensitivity, because budget spent on a constraint
    that never binds returns nothing while the impact target is tight at every
    solve by construction.
    """
    if not 0.0 <= lambda_level < 1.0:
        raise ValueError(
            f"lambda_level must lie in [0, 1); got {lambda_level!r}.")
    if K < 1:
        raise ValueError(f"K must be at least 1; got {K!r}.")
    if weights is None:
        weights = (1.0 / K,) * K
    weights = tuple(float(w) for w in weights)
    if len(weights) != K:
        raise ValueError(
            f"weights has length {len(weights)} but K is {K}; one weight per "
            f"event, the first for the impact target.")
    if any(w < 0.0 for w in weights):
        raise ValueError(f"weights must be non-negative; got {weights!r}.")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError(
            f"weights must sum to 1 for Boole's inequality to bound the joint "
            f"failure probability by {1.0 - lambda_level:.4g}; got "
            f"{sum(weights):.6g}.")
    return RiskBudget(lambda_level=float(lambda_level), K=int(K),
                      weights=weights)


def declared_quantile(spec: UncertaintySpec, probability: float) -> float:
    """The exact ``probability``-quantile of a parameter's *declared* family.

    A right-hand-side-only chance constraint ``P(s <= xi) >= 1 - eps`` is
    equivalent to ``s <= F^-1(eps)`` with ``F`` the declared distribution, so it
    needs no Gaussian representation at all. Normality is required where an
    uncertain coefficient multiplies a decision variable and a sum has to be
    reduced to a closed form; a bare bound carries no such requirement.

    For triangular(a, b, c) the inverse CDF is elementary, which makes the bound
    cheaper than the ``Phi^-1`` it replaces - and bounded below by the support
    floor ``a``, which a moment-matched normal is not: at high reliability the
    fitted normal eventually demands a negative availability.
    """
    utype = int(spec['uncertainty_type'])
    p = float(probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must lie in [0, 1]; got {p!r}.")

    if utype == stats_arrays.NormalUncertainty.id:
        return float(spec['loc'] + spec['scale'] * scipy.stats.norm.ppf(p))
    if utype == stats_arrays.UniformUncertainty.id:
        minimum, maximum = float(spec['minimum']), float(spec['maximum'])
        return minimum + p * (maximum - minimum)
    if utype == stats_arrays.TriangularUncertainty.id:
        a, c = float(spec['minimum']), float(spec['maximum'])
        b = float(spec['loc'])
        if c <= a:
            return a
        # Which branch applies is decided by where the mode sits, not assumed:
        # an unequal split with a large weight at low lambda can cross it.
        threshold = (b - a) / (c - a)
        if p <= threshold:
            return a + np.sqrt(p * (c - a) * (b - a))
        return c - np.sqrt((1.0 - p) * (c - a) * (c - b))
    if utype == stats_arrays.LognormalUncertainty.id:
        mu, sigma = float(spec['loc']), float(spec['scale'])
        sign = -1.0 if spec.get('negative', False) else 1.0
        # A negated lognormal is decreasing in p, so the quantile mirrors.
        q = p if sign > 0 else 1.0 - p
        return sign * float(np.exp(mu + sigma * scipy.stats.norm.ppf(q)))
    raise NotImplementedError(
        f"declared_quantile has no closed-form inverse CDF for "
        f"uncertainty_type={utype}; supported types are Normal(3), Uniform(4), "
        f"Triangular(5), Lognormal(2).")


# Which Pyomo parameter each bound block writes to, and whether the bound is an
# upper one. An upper bound needs F^-1(eps); a lower bound needs F^-1(1 - eps).
_BOUND_BLOCKS = {
    'upper_limit': ('UPPER_LIMIT', True),
    'upper_imp_limit': ('UPPER_IMP_LIMIT', True),
    'upper_inv_limit': ('UPPER_INV_LIMIT', True),
    'lower_limit': ('LOWER_LIMIT', False),
}


def _bound_positions(normal_metadata_var_bounds) -> Dict[Tuple[str, int], int]:
    """Assign each bound row its position in the weight vector, deterministically.

    Position 0 is the impact target, so the rows start at 1. Sorted rather than
    insertion-ordered: the weights are part of the reported configuration, and
    which row received which weight must not depend on dictionary construction
    order.
    """
    positions: Dict[Tuple[str, int], int] = {}
    position = 1
    for bound_name in sorted(normal_metadata_var_bounds):
        for indx in sorted(normal_metadata_var_bounds[bound_name]):
            positions[(bound_name, indx)] = position
            position += 1
    return positions


def apply_CC_formulation(
        model_instance,
        lambda_level: float,
        normal_metadata_env_cost: Dict[Tuple[int, str], UncertaintySpec] = {},
        normal_metadata_var_bounds: Dict[str, Dict[int, UncertaintySpec]] = {},
        risk_budget: "RiskBudget" = None,
        bound_quantile: str = 'gaussian',
        ):
    """
    Inject or update the epsilon-constraint for a given risk level.

    With ``risk_budget=None`` (the default) every row is imposed at
    ``lambda_level`` individually, which is the historical behaviour and is
    preserved exactly. Passing a :class:`RiskBudget` instead allocates a share
    of the total failure budget to each row, so that they hold *jointly* at
    ``lambda_level``; ``bound_quantile='exact'`` additionally reads each bound
    off its declared distribution rather than off a moment-matched normal.

    The two are separable on purpose: ``risk_budget`` with the default
    ``'gaussian'`` marginals and unit weights reproduces the individual
    formulation term for term, which is what makes the change testable.
    """
    # The bound quantiles are written straight into the limit Params, which on
    # an equilibrated model are in scaled units. See scaling.py.
    _scaling.require_unscaled(model_instance, "The CC formulation")
    if risk_budget is not None and not np.isclose(risk_budget.lambda_level,
                                                  lambda_level):
        raise ValueError(
            f"risk_budget was built for lambda={risk_budget.lambda_level!r} but "
            f"apply_CC_formulation was called with lambda={lambda_level!r}.")
    if bound_quantile not in ('gaussian', 'exact'):
        raise ValueError(
            f"bound_quantile must be 'gaussian' or 'exact'; got {bound_quantile!r}.")
    if bound_quantile == 'exact' and risk_budget is None:
        raise ValueError(
            "bound_quantile='exact' needs a risk_budget: the exact quantile is "
            "evaluated at the share of the failure budget allocated to each row.")

    ppf_lambda = scipy.stats.norm.ppf(lambda_level)
    if normal_metadata_env_cost:
        # Under a budget the impact target is imposed at 1 - w_0 * eps, not at
        # lambda: it is one of the K events the budget is divided among.
        ppf_impact = (ppf_lambda if risk_budget is None
                      else scipy.stats.norm.ppf(risk_budget.lambda_impact))
        print(f'Applying CC constraints to the environmental cost calculation with lambda: {lambda_level}')
        environmental_cost_updated = {
            env_cost_indx: env_cost_data['loc'] + ppf_impact * env_cost_data['scale']
            for env_cost_indx, env_cost_data in normal_metadata_env_cost.items()
        }
        optimizer.update_env_cost(model_instance, environmental_cost_updated)

    positions = _bound_positions(normal_metadata_var_bounds)
    if risk_budget is not None and len(positions) + 1 != risk_budget.K:
        raise ValueError(
            f"the risk budget covers K={risk_budget.K} events but the model "
            f"imposes {len(positions)} chance-constrained bound(s) plus the "
            f"impact target, i.e. {len(positions) + 1}. K is claimed before the "
            f"solve and must match the rows actually imposed; an unbounded "
            f"alternative should carry no bound at all rather than a sentinel.")

    upper_branch = 0
    for bound_name, metadata_vb in normal_metadata_var_bounds.items():
        if not metadata_vb:
            continue
        if bound_name not in _BOUND_BLOCKS:
            raise Exception(f'{bound_name} has not been implemented yet.')
        pyomo_var_name, is_upper = _BOUND_BLOCKS[bound_name]
        print(f'Applying CC constraints to the {bound_name} constraint with lambda: {lambda_level}')

        bound_updated = {}
        for indx, unc_data in metadata_vb.items():
            if risk_budget is None:
                # Unchanged from the individual formulation, term for term.
                bound_updated[indx] = (unc_data['loc'] + (ppf_lambda if not is_upper
                                                          else -ppf_lambda)
                                       * unc_data['scale'])
                continue
            epsilon_k = risk_budget.epsilon_at(positions[(bound_name, indx)])
            probability = epsilon_k if is_upper else 1.0 - epsilon_k
            if bound_quantile == 'exact':
                source = unc_data.get('source')
                if source is None:
                    raise ValueError(
                        f"bound_quantile='exact' needs the declared distribution "
                        f"under 'source' for {bound_name}[{indx}]; recompute the "
                        f"moments with processor.compute_closed_form_moments.")
                value = declared_quantile(source, probability)
                if (int(source['uncertainty_type'])
                        == stats_arrays.TriangularUncertainty.id):
                    a, c = float(source['minimum']), float(source['maximum'])
                    b = float(source['loc'])
                    if c > a and probability > (b - a) / (c - a):
                        upper_branch += 1
            else:
                value = (unc_data['loc']
                         + unc_data['scale'] * scipy.stats.norm.ppf(probability))
            bound_updated[indx] = value

        pyomo_bound = getattr(model_instance, pyomo_var_name)
        pyomo_bound.store_values(bound_updated, check=True)

    if risk_budget is not None:
        print(f'  risk budget: eps={risk_budget.epsilon:.6g} split over '
              f'K={risk_budget.K} events, weights={risk_budget.weights}, '
              f'impact target at lambda={risk_budget.lambda_impact:.6g}')
        if upper_branch:
            # Reported rather than asserted: the upper branch is correct, it
            # merely signals a split lopsided enough to push a row past its mode.
            print(f'  note: {upper_branch} triangular bound(s) evaluated on the '
                  f'upper branch of the inverse CDF (eps_k above (b-a)/(c-a))')
