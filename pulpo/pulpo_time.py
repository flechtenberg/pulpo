"""
pulpo_time.py

Time-dependent façade for PULPO. Users opt into the time-indexed formulation
by writing::

    from pulpo import pulpo_time
    worker = pulpo_time.PulpoOptimizerTime(project, db, method, directory)
    worker.get_lci_data()
    worker.instantiate(
        choices=..., demand=..., upper_limit=...,
        time_steps=[0, 1, 2, ...],
        storage=[(stored_product, producing_activity, factor), ...],
    )
    worker.solve()

The base package (``from pulpo import pulpo``) stays purely static; all
time-coupling logic lives here and in :mod:`pulpo.utils.time_extension`.
"""

from __future__ import annotations

from typing import List, Optional

from pulpo.pulpo import PulpoOptimizer
from pulpo.utils import time_extension
from pulpo.datasets.elec_time_database import setup_elec_time_db


class PulpoOptimizerTime(PulpoOptimizer):
    """PulpoOptimizer + time-indexed formulation with optional storage carry-over."""

    def __init__(self, project, database, method, directory):
        super().__init__(project, database, method, directory)
        self.time_steps: Optional[list] = None
        self.storage: list = []
        self.upper_imp_agg_limit: dict = {}

    def instantiate(
        self,
        choices: Optional[dict] = None,
        demand: Optional[dict] = None,
        upper_limit: Optional[dict] = None,
        lower_limit: Optional[dict] = None,
        upper_elem_limit: Optional[dict] = None,
        upper_imp_limit: Optional[dict] = None,
        lower_elem_limit: Optional[dict] = None,
        lower_imp_limit: Optional[dict] = None,
        dependent_constraints: Optional[dict] = None,
        default_limits=None,
        time_steps: Optional[List] = None,
        storage: Optional[list] = None,
        upper_imp_agg_limit: Optional[dict] = None,
    ):
        """
        Build the time-indexed Pyomo instance.

        When ``time_steps`` is omitted, falls back to the static formulation
        from :class:`pulpo.pulpo.PulpoOptimizer`.

        Each of ``demand``, ``choices`` and the limit dicts may be supplied
        either as a static dict (broadcast across all timesteps) or as
        ``{t: dict}``. ``dependent_constraints`` is not yet supported in the
        time-dependent path.

        Args:
            storage (list, optional): Carry-over specification. List of triples
                ``(stored_product, producing_activity, factor)`` so that
                charging at *t-1* contributes ``factor * scaling[t-1]`` units
                of the stored product towards demand at *t*.
            upper_imp_agg_limit (dict, optional): Bound on the *sum* of an
                indicator's impact across all timesteps.
        """
        if time_steps is None:
            return super().instantiate(
                choices=choices, demand=demand,
                upper_limit=upper_limit, lower_limit=lower_limit,
                upper_elem_limit=upper_elem_limit, upper_imp_limit=upper_imp_limit,
                lower_elem_limit=lower_elem_limit, lower_imp_limit=lower_imp_limit,
                dependent_constraints=dependent_constraints,
                default_limits=default_limits,
            )

        choices = choices or {}
        demand = demand or {}
        upper_limit = upper_limit or {}
        lower_limit = lower_limit or {}
        upper_elem_limit = upper_elem_limit or {}
        upper_imp_limit = upper_imp_limit or {}
        lower_elem_limit = lower_elem_limit or {}
        lower_imp_limit = lower_imp_limit or {}
        dependent_constraints = dependent_constraints or {}

        if dependent_constraints:
            raise NotImplementedError(
                "`dependent_constraints` is not yet supported in the "
                "time-dependent path. Open an issue if you need this."
            )

        # Keep only methods that contribute to objective or any (per-step or
        # aggregate) impact constraint.
        def _h_in_per_step(limits, h):
            if not isinstance(limits, dict):
                return False
            return any(h in (limits.get(t, limits) if isinstance(limits, dict) else {})
                       for t in time_steps)

        methods = {
            h: self.method[h] for h in self.method
            if self.method[h] != 0
            or _h_in_per_step(upper_imp_limit, h)
            or _h_in_per_step(lower_imp_limit, h)
            or h in (upper_imp_agg_limit or {})
        }

        data = time_extension.combine_inputs_time(
            self.lci_data, demand, choices, upper_limit, lower_limit,
            upper_elem_limit, upper_imp_limit, lower_elem_limit, lower_imp_limit,
            methods, time_steps,
            storage=storage, upper_imp_agg_limit=upper_imp_agg_limit,
            default_limits=default_limits,
        )
        self.instance = time_extension.instantiate_time(data)

        self.choices = choices
        self.demand = demand
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.upper_elem_limit = upper_elem_limit
        self.upper_imp_limit = upper_imp_limit
        self.lower_elem_limit = lower_elem_limit
        self.lower_imp_limit = lower_imp_limit
        self.dependent_constraints = dependent_constraints
        self.time_steps = list(time_steps)
        self.storage = list(storage) if storage else []
        self.upper_imp_agg_limit = dict(upper_imp_agg_limit) if upper_imp_agg_limit else {}

    def solve(self, GAMS_PATH=False, solver_name=None, options=None, neos_email=None):
        """
        Solve the model. Mirrors ``PulpoOptimizer.solve()``'s post-processing
        (auxiliary zero-weight methods, elementary flows), generalized to the
        per-timestep variable layout so that ``extract_results()``/
        ``save_results()``/``summarize_results()`` work unchanged on a
        time-indexed instance.
        """
        if self.time_steps is None:
            return super().solve(
                GAMS_PATH=GAMS_PATH, solver_name=solver_name,
                options=options, neos_email=neos_email,
            )

        from pulpo.utils import optimizer
        results, self.instance = optimizer.solve_model(
            self.instance, GAMS_PATH, solver_name=solver_name,
            options=options, neos_email=neos_email,
        )

        if not isinstance(self.method, str):
            if len(self.method) > 1 and 0 in [self.method[x] for x in self.method]:
                self.instance = optimizer.calculate_methods(
                    self.instance, self.lci_data, self.method, time_steps=self.time_steps
                )

        self.instance = optimizer.calculate_inv_flows(
            self.instance, self.lci_data, time_steps=self.time_steps
        )
        return results


def install_elec_time_db():
    """Set up the toy time-dependent electricity database in Brightway2."""
    setup_elec_time_db()
