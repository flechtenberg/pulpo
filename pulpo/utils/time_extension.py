"""Time-dependent extension for PULPO (feat/time-dependent).

This module provides a parallel data-preparation and pyomo-model pipeline that
adds a ``TIME`` dimension to PULPO without modifying the existing single-step
implementation in :mod:`pulpo.utils.converter` and :mod:`pulpo.utils.optimizer`.

Public API:
    - :func:`combine_inputs_time` -- build a time-indexed pyomo data dict.
    - :func:`create_time_model`   -- abstract pyomo model with TIME index.
    - :func:`instantiate_time`    -- materialize a concrete instance.

Storage / carry-over
--------------------
The time-coupling mechanism is expressed via an optional ``storage`` argument
of :func:`combine_inputs_time`, given as a list of triples::

    storage = [
        (stored_product_activity, producing_activity, factor),
        ...
    ]

Each triple says: *the scaling of ``producing_activity`` at time t-1
contributes ``factor`` units of the stored product towards demand
satisfaction at time t.* This generalises a battery / storage tank: charging
(non-zero scaling of the charge process) at t-1 enables discharge at t.

Aggregated impact bounds
------------------------
``upper_imp_agg_limit`` constrains the *sum* of an indicator's impacts across
all timesteps -- e.g. an annual CO2 budget independent of per-step bounds.

Backwards compatibility
-----------------------
The static, single-timestep API is unchanged. To opt into the time-dependent
formulation, pass ``time_steps=[...]`` to
:meth:`pulpo.pulpo.PulpoOptimizer.instantiate`.
"""

from __future__ import annotations

import pyomo.environ as pyo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_time_indexed(d, time_steps):
    """Return True if dict ``d`` is keyed by the supplied timestep labels."""
    if not isinstance(d, dict) or not d:
        return False
    return set(d.keys()) == set(time_steps)


def _broadcast_over_time(d, time_steps):
    """Convert a (possibly static) input dict into ``{t: dict}`` form."""
    if d is None:
        d = {}
    if not isinstance(d, dict):
        raise TypeError(f"Expected a dict, got {type(d).__name__}")
    if _is_time_indexed(d, time_steps):
        for t, sub in d.items():
            if not isinstance(sub, dict):
                raise TypeError(
                    f"Time-indexed input must map each timestep to a dict; "
                    f"got {type(sub).__name__} for t={t!r}"
                )
        return {t: dict(d[t]) for t in time_steps}
    return {t: dict(d) for t in time_steps}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def combine_inputs_time(
    lci_data,
    demand,
    choices,
    upper_limit,
    lower_limit,
    upper_inv_limit,
    upper_imp_limit,
    lower_inv_limit,
    lower_imp_limit,
    methods,
    time_steps,
    storage=None,
    upper_imp_agg_limit=None,
    default_limits=None,
):
    """Build the time-indexed pyomo data dictionary.

    Mirrors :func:`pulpo.utils.converter.combine_inputs` and adds three
    time-related arguments:

    Args:
        time_steps: Ordered list of timestep labels. Must be non-empty.
        storage: Optional list of triples ``(stored_product, producing_activity,
            factor)``. Each triple couples ``producing_activity`` at t-1 to the
            demand of ``stored_product`` at t. Activity arguments accept either
            Brightway activities or their ``.key`` tuples.
        upper_imp_agg_limit: Optional ``{indicator: bound}`` constraining the
            *sum* of an indicator's impact across all timesteps.

    Each of ``demand``, ``choices``, ``upper_limit``, ``lower_limit``,
    ``upper_inv_limit``, ``upper_imp_limit``, ``lower_inv_limit``,
    ``lower_imp_limit`` may be a static dict (broadcast across timesteps)
    or already in ``{t: dict}`` form.
    """
    if not time_steps:
        raise ValueError("`time_steps` must be a non-empty list.")
    time_steps = list(time_steps)

    if default_limits is None:
        default_limits = {
            'lower_bound': -1e20,
            'upper_bound': 1e20,
            'upper_inv_bound': 1e24,
            'lower_inv_bound': -1e24,
            'lower_imp_bound': -1e24,
            'upper_imp_bound': 1e24,
            'upper_imp_agg_bound': 1e24,
        }

    demand_t = _broadcast_over_time(demand, time_steps)
    choices_t = _broadcast_over_time(choices, time_steps)
    upper_limit_t = _broadcast_over_time(upper_limit, time_steps)
    lower_limit_t = _broadcast_over_time(lower_limit, time_steps)
    upper_inv_limit_t = _broadcast_over_time(upper_inv_limit, time_steps)
    upper_imp_limit_t = _broadcast_over_time(upper_imp_limit, time_steps)
    lower_inv_limit_t = _broadcast_over_time(lower_inv_limit, time_steps)
    lower_imp_limit_t = _broadcast_over_time(lower_imp_limit, time_steps)

    matrices = lci_data['matrices']
    intervention_matrix = lci_data['intervention_matrix']
    technology_matrix = lci_data['technology_matrix']
    process_map = lci_data['process_map']
    intervention_map = lci_data['intervention_map']

    matrices = {h: matrices[h] for h in matrices if str(h) in methods}

    env_cost = {h: matrices[h].diagonal() @ intervention_matrix for h in matrices}
    env_cost_dict = {(j, h): env_cost[h][j] for h in matrices for j in range(len(env_cost[h]))}

    technology_matrix_dict = {
        (i - 1, technology_matrix.indices[j]): technology_matrix.data[j]
        for i in range(1, technology_matrix.shape[0] + 1)
        for j in range(technology_matrix.indptr[i - 1], technology_matrix.indptr[i])
    }

    # Choice topology is time-invariant; only capacity bounds may differ per t.
    keys = {}
    product_ids = []
    union_choices = {}
    for choices_at_t in choices_t.values():
        for choice_label, processes in choices_at_t.items():
            union_choices.setdefault(choice_label, {}).update(processes)
    for choice_label, processes in union_choices.items():
        for proc in processes:
            product_id = process_map[proc.key]
            keys[product_id] = choice_label
            product_ids.append(product_id)
    for product, process in list(technology_matrix_dict):
        if product in product_ids:
            new_key = (keys[product], process)
            if new_key not in technology_matrix_dict:
                technology_matrix_dict[new_key] = technology_matrix_dict[(product, process)]
            else:
                technology_matrix_dict[new_key] += technology_matrix_dict[(product, process)]
            del technology_matrix_dict[(product, process)]

    constrained_inv_keys = set()
    for t in time_steps:
        constrained_inv_keys.update(upper_inv_limit_t[t].keys())
        constrained_inv_keys.update(lower_inv_limit_t[t].keys())
    inv_to_consider = [intervention_map[g.key] if hasattr(g, 'key') else intervention_map[g]
                       for g in constrained_inv_keys]
    inv_dict = {
        (g, intervention_matrix.indices[j]): intervention_matrix.data[j]
        for g in inv_to_consider
        for j in range(intervention_matrix.indptr[g], intervention_matrix.indptr[g + 1])
    }

    PRODUCTS = {None: list({i[0] for i in technology_matrix_dict})}
    PROCESS = {None: list({i[1] for i in technology_matrix_dict})}
    PRODUCT_PROCESS = {None: list({(i[0], i[1]) for i in technology_matrix_dict})}
    ENV_COST = {None: list({i[0] for i in env_cost_dict})}
    ENV_COST_PROCESS = {None: list({i for i in env_cost_dict})}
    INV = {None: list({i[0] for i in inv_dict})}
    INV_PROCESS = {None: list({(i[0], i[1]) for i in inv_dict})}
    INDICATOR = {None: list({h for h in matrices})}
    TIME = {None: list(time_steps)}

    # Resolve storage specification into pyomo-friendly indexable form.
    storage = storage or []
    storage_pairs = {}
    storable_products = set()
    for spec in storage:
        if not (isinstance(spec, (list, tuple)) and len(spec) == 3):
            raise ValueError(
                "Each `storage` entry must be a (stored_product, producing_activity, factor) triple."
            )
        stored_act, producing_act, factor = spec
        stored_key = stored_act.key if hasattr(stored_act, 'key') else stored_act
        producing_key = producing_act.key if hasattr(producing_act, 'key') else producing_act
        if stored_key not in process_map:
            raise KeyError(f"Stored product activity {stored_key!r} not in process_map.")
        if producing_key not in process_map:
            raise KeyError(f"Producing activity {producing_key!r} not in process_map.")
        i = process_map[stored_key]
        j = process_map[producing_key]
        storage_pairs[(i, j)] = float(factor)
        storable_products.add(i)
    PRODUCT_STOR = {None: list(storable_products)}
    PRODUCT_PROCESS_STOR = {None: list(storage_pairs.keys())}

    demand_dict = {(t, prod): 0 for t in time_steps for prod in PRODUCTS[None]}
    for t in time_steps:
        for dem, value in demand_t[t].items():
            if dem in process_map:
                demand_dict[(t, process_map[dem])] = value
            elif dem in union_choices:
                demand_dict[(t, dem)] = value
            else:
                raise ValueError(f"'{dem}' is not found in process_map keys or values.")

    lower_limit_dict = {(t, p): default_limits['lower_bound'] for t in time_steps for p in PROCESS[None]}
    upper_limit_dict = {(t, p): default_limits['upper_bound'] for t in time_steps for p in PROCESS[None]}
    for t in time_steps:
        for choice_label, processes in choices_t[t].items():
            for proc, capacity in processes.items():
                lower_limit_dict[(t, process_map[proc])] = 0
                upper_limit_dict[(t, process_map[proc])] = capacity
        for proc, value in lower_limit_t[t].items():
            lower_limit_dict[(t, process_map[proc])] = value
        for proc, value in upper_limit_t[t].items():
            upper_limit_dict[(t, process_map[proc])] = value

    supply_dict = {(t, prod): 0 for t in time_steps for prod in PRODUCTS[None]}
    for t in time_steps:
        common = lower_limit_t[t].keys() & upper_limit_t[t].keys()
        for proc in common:
            if lower_limit_t[t][proc] == upper_limit_t[t][proc]:
                supply_dict[(t, process_map[proc])] = 1

    upper_inv_limit_dict = {(t, g): default_limits['upper_inv_bound'] for t in time_steps for g in INV[None]}
    lower_inv_limit_dict = {(t, g): default_limits['lower_inv_bound'] for t in time_steps for g in INV[None]}
    for t in time_steps:
        for inv, value in upper_inv_limit_t[t].items():
            key = inv.key if hasattr(inv, 'key') else inv
            upper_inv_limit_dict[(t, intervention_map[key])] = value
        for inv, value in lower_inv_limit_t[t].items():
            key = inv.key if hasattr(inv, 'key') else inv
            lower_inv_limit_dict[(t, intervention_map[key])] = value

    upper_imp_limit_dict = {(t, h): default_limits['upper_imp_bound'] for t in time_steps for h in INDICATOR[None]}
    lower_imp_limit_dict = {(t, h): default_limits['lower_imp_bound'] for t in time_steps for h in INDICATOR[None]}
    for t in time_steps:
        for imp, value in upper_imp_limit_t[t].items():
            upper_imp_limit_dict[(t, imp)] = value
        for imp, value in lower_imp_limit_t[t].items():
            lower_imp_limit_dict[(t, imp)] = value

    upper_imp_agg_limit = upper_imp_agg_limit or {}
    upper_imp_agg_limit_dict = {h: default_limits['upper_imp_agg_bound'] for h in INDICATOR[None]}
    for h, value in upper_imp_agg_limit.items():
        upper_imp_agg_limit_dict[h] = value

    k_param_dict = {pair: factor for pair, factor in storage_pairs.items()}

    weights = {method: 1 for method in matrices} if methods == {} else methods

    model_data = {
        None: {
            'TIME': TIME,
            'PRODUCT': PRODUCTS,
            'PROCESS': PROCESS,
            'ENV_COST': ENV_COST,
            'INDICATOR': INDICATOR,
            'INV': INV,
            'PRODUCT_PROCESS': PRODUCT_PROCESS,
            'ENV_COST_PROCESS': ENV_COST_PROCESS,
            'INV_PROCESS': INV_PROCESS,
            'PRODUCT_STOR': PRODUCT_STOR,
            'PRODUCT_PROCESS_STOR': PRODUCT_PROCESS_STOR,
            'TECH_MATRIX': technology_matrix_dict,
            'ENV_COST_MATRIX': env_cost_dict,
            'INV_MATRIX': inv_dict,
            'K': k_param_dict,
            'FINAL_DEMAND': demand_dict,
            'SUPPLY': supply_dict,
            'LOWER_LIMIT': lower_limit_dict,
            'UPPER_LIMIT': upper_limit_dict,
            'UPPER_INV_LIMIT': upper_inv_limit_dict,
            'LOWER_INV_LIMIT': lower_inv_limit_dict,
            'UPPER_IMP_LIMIT': upper_imp_limit_dict,
            'LOWER_IMP_LIMIT': lower_imp_limit_dict,
            'UPPER_IMP_AGG_LIMIT': upper_imp_agg_limit_dict,
            'WEIGHTS': weights,
        }
    }
    return model_data


# ---------------------------------------------------------------------------
# Pyomo abstract model (time-indexed)
# ---------------------------------------------------------------------------

def create_time_model():
    """Build the abstract time-indexed model."""
    model = pyo.AbstractModel()

    # Sets
    model.TIME = pyo.Set(ordered=True, doc='Set of timesteps, indexed by t')
    model.PRODUCT = pyo.Set(doc='Set of intermediate products, indexed by i')
    model.PROCESS = pyo.Set(doc='Set of processes, indexed by j')
    model.ENV_COST = pyo.Set(doc='Set of environmental cost flows, indexed by e')
    model.INDICATOR = pyo.Set(doc='Set of impact assessment indicators, indexed by h')
    model.INV = pyo.Set(doc='Set of intervention flows, indexed by g')
    model.ENV_COST_PROCESS = pyo.Set(within=model.PROCESS * model.INDICATOR)
    model.ENV_COST_IN = pyo.Set(model.INDICATOR, within=model.ENV_COST)
    model.PROCESS_IN = pyo.Set(model.PROCESS, within=model.PRODUCT)
    model.PROCESS_OUT = pyo.Set(model.PRODUCT, within=model.PROCESS)
    model.PRODUCT_PROCESS = pyo.Set(within=model.PRODUCT * model.PROCESS)
    model.INV_PROCESS = pyo.Set(within=model.INV * model.PROCESS)
    model.INV_OUT = pyo.Set(model.INV, within=model.PROCESS)
    model.PRODUCT_STOR = pyo.Set(within=model.PRODUCT, doc='Storable products')
    model.PRODUCT_PROCESS_STOR = pyo.Set(
        within=model.PRODUCT * model.PROCESS, doc='Storage carry-over relations',
    )

    # Parameters: per-timestep
    model.UPPER_LIMIT = pyo.Param(model.TIME, model.PROCESS, mutable=True, within=pyo.Reals)
    model.LOWER_LIMIT = pyo.Param(model.TIME, model.PROCESS, mutable=True, within=pyo.Reals)
    model.UPPER_INV_LIMIT = pyo.Param(model.TIME, model.INV, mutable=True, within=pyo.Reals)
    model.LOWER_INV_LIMIT = pyo.Param(model.TIME, model.INV, mutable=True, within=pyo.Reals)
    model.UPPER_IMP_LIMIT = pyo.Param(model.TIME, model.INDICATOR, mutable=True, within=pyo.Reals)
    model.LOWER_IMP_LIMIT = pyo.Param(model.TIME, model.INDICATOR, mutable=True, within=pyo.Reals)
    model.FINAL_DEMAND = pyo.Param(model.TIME, model.PRODUCT, mutable=True, within=pyo.Reals)
    model.SUPPLY = pyo.Param(model.TIME, model.PRODUCT, mutable=True, within=pyo.Binary)
    # Parameters: time-invariant
    model.ENV_COST_MATRIX = pyo.Param(model.ENV_COST_PROCESS, mutable=True)
    model.INV_MATRIX = pyo.Param(model.INV_PROCESS, mutable=True)
    model.TECH_MATRIX = pyo.Param(model.PRODUCT_PROCESS, mutable=True)
    model.K = pyo.Param(model.PRODUCT_PROCESS_STOR, mutable=True, default=0)
    model.WEIGHTS = pyo.Param(model.INDICATOR, mutable=True, within=pyo.NonNegativeReals)
    model.UPPER_IMP_AGG_LIMIT = pyo.Param(model.INDICATOR, mutable=True, within=pyo.Reals)

    # Variables
    model.impacts = pyo.Var(model.TIME, model.INDICATOR, doc='Impact h at time t')
    model.scaling_vector = pyo.Var(model.TIME, model.PROCESS, doc='Activity level at time t')
    model.inv_vector = pyo.Var(model.TIME, model.INV, doc='Intervention flow g at time t')
    model.slack = pyo.Var(model.TIME, model.PRODUCT, doc='Supply slack at time t')

    # Build helpers (identical structure to the static model).
    def populate_env(model):
        for j, h in model.ENV_COST_PROCESS:
            if j not in model.ENV_COST_IN[h]:
                model.ENV_COST_IN[h].add(j)

    def populate_in_and_out(model):
        for i, j in model.PRODUCT_PROCESS:
            model.PROCESS_OUT[i].add(j)
            model.PROCESS_IN[j].add(i)

    def populate_inv(model):
        for a, j in model.INV_PROCESS:
            model.INV_OUT[a].add(j)

    model.Env_in_out = pyo.BuildAction(rule=populate_env)
    model.Process_in_out = pyo.BuildAction(rule=populate_in_and_out)
    model.Inv_in_out = pyo.BuildAction(rule=populate_inv)

    # Constraint rules
    def demand_constraint(model, t, i):
        """Demand balance at time t for product i, with optional carry-over from t-1."""
        tech = sum(model.TECH_MATRIX[i, j] * model.scaling_vector[t, j]
                   for j in model.PROCESS_OUT[i])
        if i in model.PRODUCT_STOR:
            time_list = list(model.TIME.ordered_data())
            idx = time_list.index(t)
            carry = 0
            if idx > 0:
                t_prev = time_list[idx - 1]
                carry = sum(
                    model.K[i, j] * model.scaling_vector[t_prev, j]
                    for (ii, j) in model.PRODUCT_PROCESS_STOR if ii == i
                )
            return tech + carry == model.FINAL_DEMAND[t, i] + model.slack[t, i]
        return tech == model.FINAL_DEMAND[t, i] + model.slack[t, i]

    def impact_constraint(model, t, h):
        return model.impacts[t, h] == sum(
            model.ENV_COST_MATRIX[j, h] * model.scaling_vector[t, j]
            for j in model.ENV_COST_IN[h]
        )

    def inventory_constraint(model, t, g):
        return model.inv_vector[t, g] == sum(
            model.INV_MATRIX[g, j] * model.scaling_vector[t, j]
            for j in model.INV_OUT[g]
        )

    def upper_constraint(model, t, j):
        return model.scaling_vector[t, j] <= model.UPPER_LIMIT[t, j]

    def lower_constraint(model, t, j):
        return model.scaling_vector[t, j] >= model.LOWER_LIMIT[t, j]

    def upper_env_constraint(model, t, g):
        return model.inv_vector[t, g] <= model.UPPER_INV_LIMIT[t, g]

    def lower_env_constraint(model, t, g):
        return model.inv_vector[t, g] >= model.LOWER_INV_LIMIT[t, g]

    def upper_imp_constraint(model, t, h):
        return model.impacts[t, h] <= model.UPPER_IMP_LIMIT[t, h]

    def lower_imp_constraint(model, t, h):
        return model.impacts[t, h] >= model.LOWER_IMP_LIMIT[t, h]

    def upper_imp_agg_constraint(model, h):
        return sum(model.impacts[t, h] for t in model.TIME) <= model.UPPER_IMP_AGG_LIMIT[h]

    def slack_upper_constraint(model, t, j):
        return model.slack[t, j] <= 1e20 * model.SUPPLY[t, j]

    def slack_lower_constraint(model, t, j):
        return model.slack[t, j] >= -1e20 * model.SUPPLY[t, j]

    def objective_function(model):
        return sum(
            model.impacts[t, h] * model.WEIGHTS[h]
            for t in model.TIME for h in model.INDICATOR
        )

    # Constraints
    model.FINAL_DEMAND_CNSTR = pyo.Constraint(model.TIME, model.PRODUCT, rule=demand_constraint)
    model.IMPACTS_CNSTR = pyo.Constraint(model.TIME, model.INDICATOR, rule=impact_constraint)
    model.INVENTORY_CNSTR = pyo.Constraint(model.TIME, model.INV, rule=inventory_constraint)
    model.UPPER_CNSTR = pyo.Constraint(model.TIME, model.PROCESS, rule=upper_constraint)
    model.LOWER_CNSTR = pyo.Constraint(model.TIME, model.PROCESS, rule=lower_constraint)
    model.SLACK_UPPER_CNSTR = pyo.Constraint(model.TIME, model.PRODUCT, rule=slack_upper_constraint)
    model.SLACK_LOWER_CNSTR = pyo.Constraint(model.TIME, model.PRODUCT, rule=slack_lower_constraint)
    model.INV_CNSTR = pyo.Constraint(model.TIME, model.INV, rule=upper_env_constraint)
    model.LOWER_INV_CNSTR = pyo.Constraint(model.TIME, model.INV, rule=lower_env_constraint)
    model.IMP_CNSTR = pyo.Constraint(model.TIME, model.INDICATOR, rule=upper_imp_constraint)
    model.LOWER_IMP_CNSTR = pyo.Constraint(model.TIME, model.INDICATOR, rule=lower_imp_constraint)
    model.IMP_AGG_CNSTR = pyo.Constraint(model.INDICATOR, rule=upper_imp_agg_constraint)

    model.OBJ = pyo.Objective(sense=pyo.minimize, rule=objective_function)
    return model


def instantiate_time(model_data):
    """Build a concrete instance of the time-indexed abstract model."""
    print('Creating time-indexed instance')
    model = create_time_model()
    problem = model.create_instance(model_data, report_timing=False)
    print('Time-indexed instance created')
    return problem
