"""Uncertainty of an LCA impact, and chance-constrained optimization over it.

As of now the uncertainty of the LCA data is considered only in the biosphere
flows and the characterization factors; technosphere exchanges are
deterministic. The closed-form moments and the reduced space both rely on it.

The pieces, in the order a study uses them::

    from pulpo.utils import uncertainty as unc

    data  = unc.import_declared(worker)                   # parameters of the impact
    unc.override(data, 'If', 'foreground', expert_specs)  # elicited distributions
    mom   = unc.compute_moments(data, worker)             # closed-form mean and variance
    ccp   = unc.ChanceConstrained(worker, mom, upper_bounds={activity: spec})
    front = ccp.solve([0.5, 0.9, 0.99])                   # reliability sweep, reduced space
    dec   = unc.decompose(front[0.5], mom)                # exact Sobol' indices at a decision
    scr   = unc.screen_undeclared(front[0.5], data, worker, exact_cfs=unc.co2_flows(worker))
    val   = unc.validate(front, ccp, data, n=200_000, seed=1)   # out-of-sample coverage

:mod:`preparer` imports the data, :mod:`moments` assembles the moments,
:mod:`cc` holds the risk budget, the exact bound quantiles and the cone,
:mod:`decomposition` the variance decomposition, the screening of undeclared
parameters and the diagnostics, :mod:`validation` the vectorized sampler and the
coverage statistics, and :mod:`processor` single draws and helpers.
``pulpo.pulpo_unc.PulpoOptimizerUnc`` offers the same steps as methods of a
worker.
"""

from pulpo.utils.uncertainty.preparer import (UncertaintyData, UncertaintySpec, counts,
                                              import_declared, override, undeclared)
from pulpo.utils.uncertainty.moments import (Moments, compute_closed_form_moments, compute_moments,
                                             current_scaling_vector, spec_moments)
from pulpo.utils.uncertainty.cc import (ChanceConstrained, ChanceConstrainedError, Front, Point,
                                        Projections, RiskBudget, bonferroni_budget, declared_quantile)
from pulpo.utils.uncertainty.decomposition import (Decomposition, Screening, co2_flows, decompose,
                                               diagnostics, families_by_database, lognormal_with_cv,
                                               screen_undeclared, widen, width_sensitivity)
from pulpo.utils.uncertainty.validation import (Draws, ImpactSampler, Validation, declared_parameters,
                                              draw_parameters, sample_specs, validate, wilson)
from pulpo.utils.uncertainty.processor import draw_uncertainty_sample

__all__ = [
    'UncertaintyData', 'UncertaintySpec', 'import_declared', 'override', 'undeclared', 'counts',
    'Moments', 'compute_moments', 'compute_closed_form_moments', 'current_scaling_vector', 'spec_moments',
    'ChanceConstrained', 'ChanceConstrainedError', 'Front', 'Point', 'Projections', 'RiskBudget',
    'bonferroni_budget', 'declared_quantile',
    'Decomposition', 'decompose', 'families_by_database', 'Screening', 'screen_undeclared', 'widen',
    'lognormal_with_cv', 'width_sensitivity', 'co2_flows', 'diagnostics',
    'Draws', 'ImpactSampler', 'Validation', 'declared_parameters', 'draw_parameters', 'sample_specs',
    'validate', 'wilson', 'draw_uncertainty_sample',
]
