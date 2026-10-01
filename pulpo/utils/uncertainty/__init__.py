"""Uncertainty of an LCA impact, and chance-constrained optimization over it.

The pieces, in the order a study uses them::

    from pulpo.utils import uncertainty as unc

    data  = unc.import_declared(worker)                   # parameters of the impact
    unc.override(data, 'If', 'foreground', expert_specs)  # elicited distributions
    mom   = unc.compute_moments(data, worker)             # closed-form mean and variance
    ccp   = unc.ChanceConstrained(worker, mom, upper_bounds={activity: spec})
    front = ccp.solve([0.5, 0.9, 0.99])                   # reliability sweep, reduced space

:mod:`preparer` imports the data, :mod:`moments` assembles the moments,
:mod:`cc` holds the risk budget, the exact bound quantiles and the cone, and
:mod:`processor` the sampler. ``pulpo.pulpo_unc.PulpoOptimizerUnc`` offers the
same steps as methods of a worker.
"""

from pulpo.utils.uncertainty.preparer import (UncertaintyData, UncertaintySpec, counts,
                                              import_declared, override, undeclared)
from pulpo.utils.uncertainty.moments import (Moments, compute_closed_form_moments, compute_moments,
                                             current_scaling_vector, spec_moments)
from pulpo.utils.uncertainty.cc import (ChanceConstrained, ChanceConstrainedError, Front, Point,
                                        RiskBudget, bonferroni_budget, declared_quantile)
from pulpo.utils.uncertainty.processor import draw_uncertainty_sample

__all__ = [
    'UncertaintyData', 'UncertaintySpec', 'import_declared', 'override', 'undeclared', 'counts',
    'Moments', 'compute_moments', 'compute_closed_form_moments', 'current_scaling_vector', 'spec_moments',
    'ChanceConstrained', 'ChanceConstrainedError', 'Front', 'Point', 'RiskBudget',
    'bonferroni_budget', 'declared_quantile', 'draw_uncertainty_sample',
]
