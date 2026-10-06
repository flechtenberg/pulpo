"""
pulpo_unc.py

A PulpoOptimizer that holds its uncertainty data::

    from pulpo import pulpo_unc
    from pulpo.utils import uncertainty
    worker = pulpo_unc.PulpoOptimizerUnc(project, databases, method)
    worker.get_lci_data()
    worker.instantiate(choices=..., demand=...)
    worker.import_uncertainty_data()
    worker.apply_expert_knowledge('If', 'my_foreground_db', expert_specs)
    problem = worker.chance_constrained(upper_bounds={activity: spec})
    front = problem.solve([0.5, 0.9, 0.99])
    worker.screen_undeclared(front[0.5], exact_cfs=uncertainty.co2_flows(worker))
    worker.validate(front, problem, seed=1)

Each method delegates to :mod:`pulpo.utils.uncertainty`, which works with a
plain ``PulpoOptimizer`` as well. The other analyses (``decompose``,
``diagnostics``, ``width_sensitivity``, ``widen``) are called from there, with
``worker.uncertainty_data`` as the data. As of now, uncertainty is considered
only in the biosphere flows and the characterization factors; technosphere
exchanges are deterministic.
"""

from pulpo.pulpo import PulpoOptimizer
from pulpo.utils import uncertainty


class PulpoOptimizerUnc(PulpoOptimizer):
    """PulpoOptimizer plus the uncertainty steps, with ``uncertainty_data`` kept on the worker."""

    def import_uncertainty_data(self, method=None):
        """Import the parameters of the impact of ``method``, which may be omitted when
        the worker has a single LCIA method (see ``uncertainty.import_declared``)."""
        self.uncertainty_data = uncertainty.import_declared(self, method=method)
        return self.uncertainty_data

    def apply_expert_knowledge(self, group, subgroup, specs):
        """Declare or replace distributions (see ``uncertainty.override``)."""
        self._require_data()
        return uncertainty.override(self.uncertainty_data, group, subgroup, specs)

    def moments(self):
        """The closed-form moments of the impact (see ``uncertainty.compute_moments``)."""
        self._require_data()
        return uncertainty.compute_moments(self.uncertainty_data, self)

    def chance_constrained(self, upper_bounds=None, lower_bounds=None, allocation='bonferroni',
                           weights=None):
        """The chance-constrained problem on the current instance (see
        ``uncertainty.ChanceConstrained``)."""
        return uncertainty.ChanceConstrained(self, self.moments(), upper_bounds=upper_bounds,
                                             lower_bounds=lower_bounds, allocation=allocation,
                                             weights=weights)

    def screen_undeclared(self, s, *, exact_cfs, widths=(0.1, 0.3), n=10, families=None):
        """Rank the undeclared parameters at a decision (see ``uncertainty.screen_undeclared``)."""
        self._require_data()
        return uncertainty.screen_undeclared(s, self.uncertainty_data, self, exact_cfs=exact_cfs, widths=widths, n=n,
                                             families=families)

    def validate(self, front, problem, n=200_000, seed=None, designs=None, tol=1e-9, level=0.95):
        """Out-of-sample coverage of a front (see ``uncertainty.validate``)."""
        self._require_data()
        return uncertainty.validate(front, problem, self.uncertainty_data, n=n, seed=seed, designs=designs,
                                    tol=tol, level=level)

    def _require_data(self):
        if self.uncertainty_data is None:
            raise ValueError("No uncertainty data; call import_uncertainty_data() first.")
