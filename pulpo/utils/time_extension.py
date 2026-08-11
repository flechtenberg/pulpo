"""Time-dependent extension for PULPO (feat/time-dependent).

This module provides a parallel data-preparation and pyomo-model pipeline that
adds a ``TIME`` dimension to PULPO without modifying the existing single-step
implementation in :mod:`pulpo.utils.converter` and :mod:`pulpo.utils.optimizer`.

Public API:
    - :func:`combine_inputs_time` -- build a time-indexed pyomo data dict.
    - :func:`instantiate_time`    -- build the concrete time-indexed model.

Storage / carry-over
--------------------
The time-coupling mechanism is expressed via an optional ``storage`` argument
of :func:`combine_inputs_time`, given as a list of triples::

    storage = [
        (target_product, source_product, factor),
        ...
    ]

Each triple says: *the net production of ``source_product`` at time t-1
contributes ``factor`` units to the balance of ``target_product`` at time t.*
Formally it sets ``K[target_product, source_product] = factor`` in a
product-by-product carry-over matrix; the demand balance at t becomes

    A_i · s[t]  +  Σ_{i2} K[i, i2] · ( A_{i2} · s[t-1] )  ≥ / =  d[t, i].

Products that appear as ``target_product`` (and ``source_product``) in any
storage triple are added to the set ``PRODUCT_STOR`` and use a ``>=``
balance instead of ``==``, allowing within-step over-production (the
classical battery slack: charge[t] need not be discharged at t).

This matches the four-activity (CHARGE / HOLD / HOLD t-1 / DISCHARGE)
pattern used by larger ESM integrations: there the ``source`` is
*"Charge product"* (or "Hold-t-1 product"), and ``K`` is the round-trip
efficiency / self-discharge factor.

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

from collections import defaultdict

import pyomo.environ as pyo
from pyomo.core.expr.numeric_expr import LinearExpression

from pulpo.utils.optimizer import _group_env_cost_rows
from pulpo.utils.utils import broadcast_over_time as _broadcast_over_time


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
    imp_goals=None,
):
    """Build the time-indexed pyomo data dictionary.

    Mirrors :func:`pulpo.utils.converter.combine_inputs` and adds three
    time-related arguments:

    Args:
        time_steps: Ordered list of timestep labels. Must be non-empty.
        storage: Optional list of triples ``(target_product, source_product,
            factor)``. Each triple sets ``K[target, source] = factor`` in a
            product×product carry-over matrix and adds both products to
            ``PRODUCT_STOR`` (relaxes their balance from ``==`` to ``>=``).
            Activity arguments are accepted and resolved to their self-product
            via ``process_map``.
        upper_imp_agg_limit: Optional ``{indicator: bound}`` constraining the
            *sum* of an indicator's impact across all timesteps.
        imp_goals: Optional ``{indicator: limit}`` goal-programming soft limits
            on the impacts aggregated over all timesteps (e.g. a yearly
            budget); used with the 'goal' objective of
            :func:`instantiate_time`. Only categories listed here receive a
            transgression slack.
        default_limits: Optional custom default limits. If None, uses
            standard values. Required keys: 'lower_bound', 'upper_bound',
            'upper_inv_bound', 'lower_inv_bound', 'lower_imp_bound',
            'upper_imp_bound', 'upper_imp_agg_bound' (the last one not
            present in :func:`pulpo.utils.converter.combine_inputs`'s
            6-key set). Categories listed in ``imp_goals`` ignore
            'lower_imp_bound'/'upper_imp_bound'/'upper_imp_agg_bound' (the
            goal is a soft limit, not a hard Var bound) unless also given an
            explicit upper_imp_limit/lower_imp_limit/upper_imp_agg_limit.

    Each of ``demand``, ``choices``, ``upper_limit``, ``lower_limit``,
    ``upper_inv_limit``, ``upper_imp_limit``, ``lower_inv_limit``,
    ``lower_imp_limit`` may be a static dict (broadcast across timesteps)
    or already in ``{t: dict}`` form.
    """
    if not time_steps:
        raise ValueError("`time_steps` must be a non-empty list.")
    time_steps = list(time_steps)

    # Unspecified limits must be truly infinite: huge finite defaults
    # (e.g. ±1e20) make HiGHS log "treated as ±Infinity" warnings for every
    # variable batch, which deadlocks pyomo>=6.6's appsi output capture on
    # Windows (GIL held during addVars while the capture pipe fills).
    if default_limits is None:
        default_limits = {
            'lower_bound': -float('inf'),
            'upper_bound': float('inf'),
            'upper_inv_bound': float('inf'),
            'lower_inv_bound': -float('inf'),
            'lower_imp_bound': -float('inf'),
            'upper_imp_bound': float('inf'),
            'upper_imp_agg_bound': float('inf'),
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
    INV = {None: list({i[0] for i in inv_dict})}
    INV_PROCESS = {None: list({(i[0], i[1]) for i in inv_dict})}
    INDICATOR = {None: list({h for h in matrices})}
    TIME = {None: list(time_steps)}

    # Resolve storage specification into a product×product carry-over matrix.
    # Each triple (target_product, source_product, factor) sets
    #   K[target_idx, source_idx] = factor
    # and marks both products as "storable" (>= balance instead of ==).
    # Products may be either Brightway activities (resolved via process_map)
    # or PULPO choice-label strings (e.g. 'charge_product').
    def _resolve_product(arg):
        key = arg.key if hasattr(arg, 'key') else arg
        if isinstance(key, str) and key in union_choices:
            return key  # choice label is already used as a product index
        if key in process_map:
            product_id = process_map[key]
            # If this product belongs to a choice group, the technology-matrix
            # rename above has replaced ``product_id`` with the choice label in
            # PRODUCTS — the storage reference must follow the same remapping
            # or the product index will be missing from the PRODUCT set.
            if product_id in keys:
                return keys[product_id]
            return product_id
        raise KeyError(
            f"Storage product reference {key!r} not found in process_map "
            f"or in choice labels {sorted(union_choices)!r}."
        )

    storage = storage or []
    storage_pairs = {}
    storable_products = set()
    for spec in storage:
        if not (isinstance(spec, (list, tuple)) and len(spec) == 3):
            raise ValueError(
                "Each `storage` entry must be a (target_product, source_product, factor) triple."
            )
        target, source, factor = spec
        i = _resolve_product(target)
        i2 = _resolve_product(source)
        storage_pairs[(i, i2)] = float(factor)
        storable_products.add(i)
        storable_products.add(i2)
    PRODUCT_STOR = {None: list(storable_products)}
    PRODUCT_PRODUCT = {None: list(storage_pairs.keys())}

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
                prod_id = process_map[proc]
                # Skip products that were rewired into a choice label;
                # locking a single option to 0 does not mean the choice
                # supply is fixed.
                if prod_id in keys:
                    continue
                supply_dict[(t, prod_id)] = 1

    upper_inv_limit_dict = {(t, g): default_limits['upper_inv_bound'] for t in time_steps for g in INV[None]}
    lower_inv_limit_dict = {(t, g): default_limits['lower_inv_bound'] for t in time_steps for g in INV[None]}
    for t in time_steps:
        for inv, value in upper_inv_limit_t[t].items():
            key = inv.key if hasattr(inv, 'key') else inv
            upper_inv_limit_dict[(t, intervention_map[key])] = value
        for inv, value in lower_inv_limit_t[t].items():
            key = inv.key if hasattr(inv, 'key') else inv
            lower_inv_limit_dict[(t, intervention_map[key])] = value

    # Goal-programming soft limits on the time-aggregated impacts: only
    # categories with a goal get a transgression slack. Computed before the
    # impact limit dicts below, since goal categories are excluded from the
    # generic default impact bound there.
    imp_goals = imp_goals or {}
    # Sorted independently of INDICATOR[None]'s own (hash-randomized set) order,
    # so the objective's summation order -- and hence its floating-point result
    # -- is reproducible across runs/processes.
    goal_indicator = {None: sorted(h for h in INDICATOR[None] if h in imp_goals)}
    imp_goals_dict = {h: imp_goals[h] for h in goal_indicator[None]}

    # A category with a goal defaults to unbounded per-step and aggregate
    # impact: the goal is a SOFT limit enforced via the transgression penalty
    # in the objective, not a hard Var bound, so a generic default_limits
    # value must not silently cap it. Explicit upper_imp_limit/
    # lower_imp_limit/upper_imp_agg_limit for the same category still apply
    # (a deliberate "goal + hard ceiling" combination).
    upper_imp_limit_dict = {
        (t, h): (float('inf') if h in imp_goals else default_limits['upper_imp_bound'])
        for t in time_steps for h in INDICATOR[None]
    }
    lower_imp_limit_dict = {
        (t, h): (-float('inf') if h in imp_goals else default_limits['lower_imp_bound'])
        for t in time_steps for h in INDICATOR[None]
    }
    for t in time_steps:
        for imp, value in upper_imp_limit_t[t].items():
            upper_imp_limit_dict[(t, imp)] = value
        for imp, value in lower_imp_limit_t[t].items():
            lower_imp_limit_dict[(t, imp)] = value

    upper_imp_agg_limit = upper_imp_agg_limit or {}
    upper_imp_agg_limit_dict = {
        h: (float('inf') if h in imp_goals else default_limits['upper_imp_agg_bound'])
        for h in INDICATOR[None]
    }
    for h, value in upper_imp_agg_limit.items():
        upper_imp_agg_limit_dict[h] = value

    k_param_dict = {pair: factor for pair, factor in storage_pairs.items()}

    weights = {method: 1 for method in matrices} if methods == {} else methods

    model_data = {
        None: {
            'TIME': TIME,
            'PRODUCT': PRODUCTS,
            'PROCESS': PROCESS,
            'INDICATOR': INDICATOR,
            'INV': INV,
            'PRODUCT_PROCESS': PRODUCT_PROCESS,
            'INV_PROCESS': INV_PROCESS,
            'PRODUCT_STOR': PRODUCT_STOR,
            'PRODUCT_PRODUCT': PRODUCT_PRODUCT,
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
            'GOAL_INDICATOR': goal_indicator,
            'IMP_GOALS': imp_goals_dict,
            'WEIGHTS': weights,
        }
    }
    return model_data


# ---------------------------------------------------------------------------
# Pyomo model (time-indexed)
# ---------------------------------------------------------------------------

def instantiate_time(model_data, objective='weighted_sum'):
    """Build a concrete instance of the time-indexed model.

    With ``objective='goal'`` the model minimizes the average transgression
    level of the *time-aggregated* impacts,

        (1/K) * sum_h max(0, sum_t impacts[t, h] / IMP_GOALS[h] - 1),

    over the K categories in ``GOAL_INDICATOR`` -- i.e. each goal is a total
    (e.g. yearly) budget across the whole horizon, not a per-timestep limit.

    Mirrors :func:`pulpo.utils.optimizer.instantiate`: the model is assembled as
    a ConcreteModel directly from the data dictionary. The time-invariant
    technology matrix, intervention matrix, and carry-over matrix K are embedded
    as plain float coefficients in LinearExpression constraint rows rather than
    as Pyomo Params, which makes instantiation several times faster on
    ecoinvent-scale data. This includes the environmental cost matrix: its
    dense dictionary is kept on the model as ``model._env_cost``, and code
    that needs different coefficients (the chance-constrained formulation)
    rebuilds the impact constraints via
    :func:`pulpo.utils.optimizer.update_env_cost`. Only the per-timestep
    parameters that may be updated in place between solves remain
    mutable Params. Production capacities as well as intervention-flow and
    impact limits enter as variable bounds; the bounds reference the mutable
    limit Params, so they are re-evaluated whenever the model is passed to a
    solver again. Slack variables exist only for the
    (t, product) pairs where a supply is specified (SUPPLY == 1); changing the
    supply pattern requires re-instantiating the model.
    """
    print('Creating time-indexed instance')
    data = model_data[None]
    tech = data['TECH_MATRIX']
    env = data['ENV_COST_MATRIX']
    inv = data['INV_MATRIX']
    k_matrix = data['K']
    times = list(data['TIME'][None])
    processes = data['PROCESS'][None]
    storable = set(data['PRODUCT_STOR'][None])

    # Group the sparse matrix entries by constraint row
    tech_rows = defaultdict(lambda: ([], []))  # product i -> ([process j], [A[i, j]])
    for (i, j), value in tech.items():
        row = tech_rows[i]
        row[0].append(j)
        row[1].append(value)
    env_rows = _group_env_cost_rows(env)
    inv_rows = defaultdict(lambda: ([], []))  # intervention g -> ([process j], [B[g, j]])
    for (g, j), value in inv.items():
        row = inv_rows[g]
        row[0].append(j)
        row[1].append(value)

    # Carry-over sources per target product for O(1) lookup in the demand rule
    carryover_sources = defaultdict(list)  # target i -> [source i2]
    for (i, i2) in k_matrix:
        carryover_sources[i].append(i2)
    prev_time = dict(zip(times[1:], times[:-1]))

    model = pyo.ConcreteModel()
    # Dense environmental cost dictionary (Q*B), kept for update_env_cost and
    # for the saver (extract_params); the constraints embed only the nonzeros.
    model._env_cost = dict(env)

    # Sets
    model.TIME = pyo.Set(initialize=times, ordered=True, doc='Set of timesteps, indexed by t')
    model.PRODUCT = pyo.Set(initialize=data['PRODUCT'][None], doc='Set of intermediate products, indexed by i')
    model.PROCESS = pyo.Set(initialize=processes, doc='Set of processes, indexed by j')
    model.INDICATOR = pyo.Set(initialize=data['INDICATOR'][None], doc='Set of impact assessment indicators, indexed by h')
    model.INV = pyo.Set(initialize=data['INV'][None], doc='Set of intervention flows, indexed by g')
    model.PRODUCT_STOR = pyo.Set(initialize=data['PRODUCT_STOR'][None], doc='Storable products (use >= balance)')
    model.GOAL_INDICATOR = pyo.Set(initialize=data['GOAL_INDICATOR'][None], within=model.INDICATOR,
                                   doc='Impact categories with a goal-programming soft limit on the aggregated impact')
    supply_pairs = [ti for ti, flag in data['SUPPLY'].items() if flag]
    model.PRODUCT_SUPPLY = pyo.Set(initialize=supply_pairs, dimen=2, doc='(t, product) pairs with a specified supply (slack active)')

    # Parameters: per-timestep (mutable: may be updated in place between solves)
    model.UPPER_LIMIT = pyo.Param(model.TIME, model.PROCESS, initialize=data['UPPER_LIMIT'], mutable=True, within=pyo.Reals)
    model.LOWER_LIMIT = pyo.Param(model.TIME, model.PROCESS, initialize=data['LOWER_LIMIT'], mutable=True, within=pyo.Reals)
    model.UPPER_INV_LIMIT = pyo.Param(model.TIME, model.INV, initialize=data['UPPER_INV_LIMIT'], mutable=True, within=pyo.Reals)
    model.LOWER_INV_LIMIT = pyo.Param(model.TIME, model.INV, initialize=data['LOWER_INV_LIMIT'], mutable=True, within=pyo.Reals)
    model.UPPER_IMP_LIMIT = pyo.Param(model.TIME, model.INDICATOR, initialize=data['UPPER_IMP_LIMIT'], mutable=True, within=pyo.Reals)
    model.LOWER_IMP_LIMIT = pyo.Param(model.TIME, model.INDICATOR, initialize=data['LOWER_IMP_LIMIT'], mutable=True, within=pyo.Reals)
    model.FINAL_DEMAND = pyo.Param(model.TIME, model.PRODUCT, initialize=data['FINAL_DEMAND'], mutable=True, within=pyo.Reals)
    # Parameters: time-invariant
    model.WEIGHTS = pyo.Param(model.INDICATOR, initialize=data['WEIGHTS'], mutable=True, within=pyo.NonNegativeReals)
    model.UPPER_IMP_AGG_LIMIT = pyo.Param(model.INDICATOR, initialize=data['UPPER_IMP_AGG_LIMIT'], mutable=True, within=pyo.Reals)
    model.IMP_GOALS = pyo.Param(model.GOAL_INDICATOR, initialize=data['IMP_GOALS'], mutable=True, within=pyo.PositiveReals,
                                doc='Soft limit (goal) L_h on the time-aggregated impact of category h')

    # Variables. Capacity and slack-activation limits are variable bounds rather
    # than constraints; they reference the mutable Params, so updated limits take
    # effect on the next solve.
    model.impacts = pyo.Var(model.TIME, model.INDICATOR,
                            bounds=lambda model, t, h: (model.LOWER_IMP_LIMIT[t, h], model.UPPER_IMP_LIMIT[t, h]),
                            doc='Impact h at time t')
    model.scaling_vector = pyo.Var(model.TIME, model.PROCESS,
                                   bounds=lambda model, t, j: (model.LOWER_LIMIT[t, j], model.UPPER_LIMIT[t, j]),
                                   doc='Activity level at time t')
    model.inv_vector = pyo.Var(model.TIME, model.INV,
                               bounds=lambda model, t, g: (model.LOWER_INV_LIMIT[t, g], model.UPPER_INV_LIMIT[t, g]),
                               doc='Intervention flow g at time t')
    model.slack = pyo.Var(model.PRODUCT_SUPPLY, bounds=(None, None),
                          doc='Supply slack (only (t, product) pairs with a specified supply)')
    model.transgression = pyo.Var(model.GOAL_INDICATOR, within=pyo.NonNegativeReals,
                                  doc='Transgression level max(0, sum_t impacts[t, h] / IMP_GOALS_h - 1) of goal category h')

    scaling = {(t, j): model.scaling_vector[t, j] for t in times for j in processes}
    supply_set = set(supply_pairs)

    # Constraint rules
    def demand_constraint(model, t, i):
        """Demand balance at time t for product i.

        Within-step contribution from all producers/consumers of i:

            tech_t = Σ_j A[i, j] · s[t, j]

        Carry-over from t-1 (only for products that appear as a target in K):

            prev_t = Σ_{i2 : (i, i2) ∈ K} K[i, i2] · Σ_j A[i2, j] · s[t-1, j]

        For *storable* products (i ∈ PRODUCT_STOR) the balance is relaxed to
        ``>=``: the optimizer may over-produce within a step, which is the
        natural slack for batteries (charge at t need not be discharged at t).
        For non-storable products the standard ``==`` balance with the supply
        slack mechanism applies.
        """
        procs, coefs = tech_rows[i]
        lhs_coefs = list(coefs)
        lhs_vars = [scaling[t, j] for j in procs]

        t_prev = prev_time.get(t)
        if t_prev is not None:
            for i2 in carryover_sources.get(i, ()):
                k = k_matrix[(i, i2)]
                procs2, coefs2 = tech_rows[i2]
                lhs_coefs.extend(k * c for c in coefs2)
                lhs_vars.extend(scaling[t_prev, j] for j in procs2)

        lhs = LinearExpression(constant=0, linear_coefs=lhs_coefs, linear_vars=lhs_vars)
        if i in storable:
            return lhs >= model.FINAL_DEMAND[t, i]
        if (t, i) in supply_set:
            return lhs == model.FINAL_DEMAND[t, i] + model.slack[t, i]
        return lhs == model.FINAL_DEMAND[t, i]

    def impact_constraint(model, t, h):
        processes, coefs = env_rows[h]
        lhs = LinearExpression(constant=0, linear_coefs=coefs,
                               linear_vars=[scaling[t, j] for j in processes])
        return model.impacts[t, h] == lhs

    def inventory_constraint(model, t, g):
        procs, coefs = inv_rows[g]
        lhs = LinearExpression(constant=0, linear_coefs=coefs, linear_vars=[scaling[t, j] for j in procs])
        return model.inv_vector[t, g] == lhs

    # Constraints
    model.FINAL_DEMAND_CNSTR = pyo.Constraint(model.TIME, model.PRODUCT, rule=demand_constraint)
    model.IMPACTS_CNSTR = pyo.Constraint(model.TIME, model.INDICATOR, rule=impact_constraint)
    model.INVENTORY_CNSTR = pyo.Constraint(model.TIME, model.INV, rule=inventory_constraint)
    model.IMP_AGG_CNSTR = pyo.Constraint(model.INDICATOR, rule=lambda model, h: pyo.quicksum(model.impacts[t, h] for t in model.TIME) <= model.UPPER_IMP_AGG_LIMIT[h])

    def transgression_constraint(model, h):
        """Aggregated transgression slack: t_h >= (sum_t impacts[t, h]) / L_h - 1 (t_h >= 0 via domain)"""
        return model.transgression[h] >= pyo.quicksum(model.impacts[t, h] for t in model.TIME) / model.IMP_GOALS[h] - 1
    model.TRANSGRESSION_CNSTR = pyo.Constraint(model.GOAL_INDICATOR, rule=transgression_constraint)

    if objective == 'goal':
        # Objective: average transgression level of the time-aggregated impacts.
        K = len(model.GOAL_INDICATOR)
        if K == 0:
            raise ValueError(
                "objective='goal' requires at least one category in GOAL_INDICATOR "
                "(model_data['GOAL_INDICATOR']/'IMP_GOALS'); got none."
            )
        model.OBJ = pyo.Objective(sense=pyo.minimize,
                                  expr=pyo.quicksum(model.transgression[h] for h in model.GOAL_INDICATOR) / K)
    else:
        model.OBJ = pyo.Objective(sense=pyo.minimize, expr=pyo.quicksum(
            model.impacts[t, h] * model.WEIGHTS[h] for t in times for h in model.INDICATOR
        ))

    print('Time-indexed instance created')
    return model
