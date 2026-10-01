"""
pulpo_unc.py

A PulpoOptimizer that holds its uncertainty data::

    from pulpo import pulpo_unc
    worker = pulpo_unc.PulpoOptimizerUnc(project, databases, method, directory)
    worker.get_lci_data()
    worker.instantiate(choices=..., demand=...)
    worker.import_uncertainty_data()
    worker.apply_expert_knowledge('If', 'foreground', expert_specs)
    front = worker.chance_constrained(upper_bounds={activity: spec}).solve([0.5, 0.9, 0.99])

Each method delegates to :mod:`pulpo.utils.uncertainty`, which works with a
plain ``PulpoOptimizer`` as well.
"""

from pulpo.pulpo import PulpoOptimizer
from pulpo.utils import uncertainty


class PulpoOptimizerUnc(PulpoOptimizer):
    """PulpoOptimizer plus the uncertainty steps, with ``uncertainty_data`` kept on the worker."""

    def import_uncertainty_data(self):
        """Import the parameters of the impact (see ``uncertainty.import_declared``)."""
        self.uncertainty_data = uncertainty.import_declared(self)
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

    def _require_data(self):
        if self.uncertainty_data is None:
            raise ValueError("No uncertainty data; call import_uncertainty_data() first.")
