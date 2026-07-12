import os
from collections import defaultdict
import pandas as pd
import numpy as np
import scipy
import pyomo.environ as pyo
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.contrib import appsi
from .saver import extract_flows



def calculate_methods(instance, lci_data, methods, time_steps=None):
    """
    Calculates the impacts if a method with weight 0 has been specified.

    Args:
        instance: The Pyomo model instance.
        lci_data (dict): LCI data containing matrices and mappings.
        methods (dict): Methods for environmental impact assessment.
        time_steps (list, optional): Timestep labels for a time-indexed instance.
            When given, `impacts_calculated` is populated per (t, method) instead
            of per method.

    Returns:
        instance: The updated Pyomo model instance with calculated impacts.
    """
    # Filter matrices for specified methods
    matrices = {h: lci_data['matrices'][h] for h in lci_data['matrices'] if str(h) in methods}
    intervention_matrix = lci_data['intervention_matrix']

    # Calculate environmental costs
    env_cost = {h: matrices[h] @ intervention_matrix for h in matrices}

    # Extract scaling vector
    scaling_vector = extract_flows(instance, lci_data['process_map'], lci_data['process_map_metadata'], 'scaling')

    if time_steps is None:
        scaling_values = scaling_vector.sort_index()['Value'].to_numpy()

        # Calculate impacts
        impacts = {h: (env_cost[h] @ scaling_values).sum() for h in matrices}

        # Update or create impacts_calculated variable
        if hasattr(instance, 'impacts_calculated'):
            for h, value in impacts.items():
                instance.impacts_calculated[h].value = value
        else:
            instance.impacts_calculated = pyo.Var(impacts.keys(), initialize=impacts)

        return instance

    # Time-indexed: compute impacts per (t, method) from that timestep's scaling slice.
    impacts = {}
    for t in time_steps:
        scaling_values = scaling_vector.xs(t, level='Time').sort_index()['Value'].to_numpy()
        for h in matrices:
            impacts[(t, h)] = (env_cost[h] @ scaling_values).sum()

    if hasattr(instance, 'impacts_calculated'):
        for key, value in impacts.items():
            instance.impacts_calculated[key].value = value
    else:
        instance.impacts_calculated = pyo.Var(time_steps, list(matrices.keys()), initialize=impacts)

    return instance

def calculate_inv_flows(instance, lci_data, time_steps=None):
    """
    Calculates elementary flows post-optimization.

    Args:
        instance: The Pyomo model instance.
        lci_data (dict): LCI data containing matrices and mappings.
        time_steps (list, optional): Timestep labels for a time-indexed instance.
            When given, `inv_flows` is populated per (t, flow) instead of per flow.

    Returns:
        instance: The updated Pyomo model instance with calculated intervention flows.
    """
    # Extract intervention matrix and scaling vector
    intervention_matrix = lci_data['intervention_matrix']
    scaling_vector = extract_flows(instance, lci_data['process_map'], lci_data['process_map_metadata'], 'scaling')

    if time_steps is None:
        scaling_values = scaling_vector.sort_index()['Value'].to_numpy()

        # Calculate intervention flows
        flows = intervention_matrix @ scaling_values

        # Update or create inv_flows variable
        if hasattr(instance, 'inv_flows'):
            for i, flow_value in enumerate(flows):
                instance.inv_flows[i].value = flow_value
        else:
            instance.inv_flows = pyo.Var(range(len(flows)), initialize=dict(enumerate(flows)))

        return instance

    # Time-indexed: compute the full intervention-flow vector per timestep from
    # that timestep's scaling slice.
    n_inv = intervention_matrix.shape[0]
    init = {}
    for t in time_steps:
        scaling_values = scaling_vector.xs(t, level='Time').sort_index()['Value'].to_numpy()
        flows = intervention_matrix @ scaling_values
        for g, flow_value in enumerate(flows):
            init[(t, g)] = flow_value

    if hasattr(instance, 'inv_flows'):
        for key, value in init.items():
            instance.inv_flows[key].value = value
    else:
        instance.inv_flows = pyo.Var(time_steps, range(n_inv), initialize=init)

    return instance


def _group_env_cost_rows(env_cost):
    """
    Group a dense {(process j, indicator h): value} environmental cost
    dictionary into constraint rows, keeping only nonzero coefficients.

    Returns a defaultdict mapping indicator h -> ([process j], [coefficient]);
    an indicator with an all-zero row yields an empty row (empty impact sum).
    """
    env_rows = defaultdict(lambda: ([], []))
    for (j, h), value in env_cost.items():
        if value:
            row = env_rows[h]
            row[0].append(j)
            row[1].append(value)
    return env_rows


def update_env_cost(model, new_values):
    """
    Updates the environmental cost coefficients of an instantiated model.

    The coefficients are embedded as plain floats in the impact constraints,
    so unlike a mutable Param they cannot be changed in place: this merges
    ``new_values`` (keyed ``(process j, indicator h)``) into the dense
    ``model._env_cost`` dictionary and reconstructs the IMPACTS_CNSTR
    component from the merged values. Works for both the plain and the
    time-indexed model. Rebuilding the few impact constraint rows is cheap
    compared to a solve; the next solver call picks up the new component.

    Args:
        model (ConcreteModel): An instance built by ``instantiate`` (or the
            time extension's ``instantiate``).
        new_values (dict): Mapping ``(process j, indicator h) -> value`` with
            the coefficients to overwrite.
    """
    unknown = [key for key in new_values if key not in model._env_cost]
    if unknown:
        raise KeyError(f"Unknown environmental cost indices: {unknown[:5]}"
                       + (" ..." if len(unknown) > 5 else ""))
    model._env_cost.update(new_values)
    env_rows = _group_env_cost_rows(model._env_cost)

    model.del_component(model.IMPACTS_CNSTR)
    # Remove the implicit index set Pyomo creates for multi-set constraints
    if hasattr(model, 'IMPACTS_CNSTR_index'):
        model.del_component(model.IMPACTS_CNSTR_index)

    scaling = model.scaling_vector
    if hasattr(model, 'TIME'):
        def impact_constraint(model, t, h):
            processes, coefs = env_rows[h]
            lhs = LinearExpression(constant=0, linear_coefs=coefs,
                                   linear_vars=[scaling[t, j] for j in processes])
            return model.impacts[t, h] == lhs
        model.add_component('IMPACTS_CNSTR', pyo.Constraint(model.TIME, model.INDICATOR, rule=impact_constraint))
    else:
        def impact_constraint(model, h):
            processes, coefs = env_rows[h]
            lhs = LinearExpression(constant=0, linear_coefs=coefs,
                                   linear_vars=[scaling[j] for j in processes])
            return model.impacts[h] == lhs
        model.add_component('IMPACTS_CNSTR', pyo.Constraint(model.INDICATOR, rule=impact_constraint))


def instantiate(model_data):
    """
    Builds an instance of the optimization model with specific data and objective function.

    The model is assembled as a ConcreteModel directly from the data dictionary.
    The technology, intervention, and environmental cost matrices are embedded
    as plain float coefficients in LinearExpression constraint rows rather than
    as Pyomo Params: skipping their per-entry Param (and relation-set)
    components makes instantiation several times faster on ecoinvent-scale
    data. The dense environmental cost dictionary is kept on the model as
    ``model._env_cost``; code that needs different coefficients (the
    chance-constrained formulation) updates that dictionary and rebuilds the
    impact constraints via :func:`update_env_cost`. Only parameters that are
    updated in place between solves (limits, demand, weights) are mutable
    Params.
    Production capacities as well as intervention-flow and impact limits enter
    as variable bounds instead of explicit constraints; the bounds reference
    the mutable limit Params, so they are re-evaluated whenever the model is
    passed to a solver again.
    Slack variables exist only for the (typically few) products where a supply
    is specified (identical lower and upper limit, SUPPLY == 1); for all other
    products the slack would be fixed to zero, so it is not created at all.
    Consequently the supply pattern is baked in at construction time: changing
    which products are supplies requires re-instantiating the model.

    Args:
        model_data (dict): Data dictionary for the optimization model.

    Returns:
        ConcreteModel: The instantiated Pyomo model.
    """
    print('Creating Instance')
    data = model_data[None]
    tech = data['TECH_MATRIX']
    env = data['ENV_COST_MATRIX']
    inv = data['INV_MATRIX']

    # Group the sparse matrix entries by constraint row (product, indicator, flow)
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

    model = pyo.ConcreteModel()
    # Dense environmental cost dictionary (Q*B), kept for update_env_cost and
    # for the saver (extract_params); the constraints embed only the nonzeros.
    model._env_cost = dict(env)

    # Sets
    model.PRODUCT = pyo.Set(initialize=data['PRODUCT'][None], doc='Set of intermediate products (or technosphere exchanges), indexed by i')
    model.PROCESS = pyo.Set(initialize=data['PROCESS'][None], doc='Set of processes (or activities), indexed by j')
    model.INDICATOR = pyo.Set(initialize=data['INDICATOR'][None], doc='Set of impact assessment indicators, indexed by h')
    model.INV = pyo.Set(initialize=data['INV'][None], doc='Set of intervention flows, indexed by g')
    model.DEPENDENT_CONSTRAINTS = pyo.Set(initialize=data['DEPENDENT_CONSTRAINTS'][None], doc='Set of dependent constraint names')
    supply_products = [i for i in data['PRODUCT'][None] if data['SUPPLY'][i]]
    model.PRODUCT_SUPPLY = pyo.Set(initialize=supply_products, within=model.PRODUCT, doc='Products for which a supply is specified instead of a demand (slack active)')

    # Parameters (mutable: updated in place by the chance-constrained and Monte Carlo code)
    model.UPPER_LIMIT = pyo.Param(model.PROCESS, initialize=data['UPPER_LIMIT'], mutable=True, within=pyo.Reals, doc='Maximum production capacity of process j')
    model.LOWER_LIMIT = pyo.Param(model.PROCESS, initialize=data['LOWER_LIMIT'], mutable=True, within=pyo.Reals, doc='Minimum production capacity of process j')
    model.UPPER_INV_LIMIT = pyo.Param(model.INV, initialize=data['UPPER_INV_LIMIT'], mutable=True, within=pyo.Reals, doc='Maximum intervention flow g')
    model.LOWER_INV_LIMIT = pyo.Param(model.INV, initialize=data['LOWER_INV_LIMIT'], mutable=True, within=pyo.Reals, doc='Minimum intervention flow g')
    model.UPPER_IMP_LIMIT = pyo.Param(model.INDICATOR, initialize=data['UPPER_IMP_LIMIT'], mutable=True, within=pyo.Reals, doc='Maximum impact on category h')
    model.LOWER_IMP_LIMIT = pyo.Param(model.INDICATOR, initialize=data['LOWER_IMP_LIMIT'], mutable=True, within=pyo.Reals, doc='Minimum impact on category h')
    model.FINAL_DEMAND = pyo.Param(model.PRODUCT, initialize=data['FINAL_DEMAND'], mutable=True, within=pyo.Reals, doc='Final demand of intermediate product flows (i.e., functional unit)')
    model.WEIGHTS = pyo.Param(model.INDICATOR, initialize=data['WEIGHTS'], mutable=True, within=pyo.NonNegativeReals, doc='Weighting factors for the impact assessment indicators in the objective function')
    model.LEFT_WEIGHTS = pyo.Param(model.DEPENDENT_CONSTRAINTS, model.PROCESS, initialize=data['LEFT_WEIGHTS'], mutable=True, default=0, doc='Left side weights for dependent constraints')
    model.RIGHT_WEIGHTS = pyo.Param(model.DEPENDENT_CONSTRAINTS, model.PROCESS, initialize=data['RIGHT_WEIGHTS'], mutable=True, default=0, doc='Right side weights for dependent constraints')

    # Variables. Capacity and slack-activation limits are variable bounds rather
    # than constraints; they reference the mutable Params, so updated limits take
    # effect on the next solve.
    model.impacts = pyo.Var(model.INDICATOR, bounds=lambda model, h: (model.LOWER_IMP_LIMIT[h], model.UPPER_IMP_LIMIT[h]),
                            doc='Environmental impact on indicator h evaluated with the established LCIA method')
    model.scaling_vector = pyo.Var(model.PROCESS, bounds=lambda model, j: (model.LOWER_LIMIT[j], model.UPPER_LIMIT[j]),
                                   doc='Activity level of each process to meet the final demand')
    model.inv_vector = pyo.Var(model.INV, bounds=lambda model, g: (model.LOWER_INV_LIMIT[g], model.UPPER_INV_LIMIT[g]),
                               doc='Intervention flows')
    model.slack = pyo.Var(model.PRODUCT_SUPPLY, bounds=(-1e20, 1e20),
                          doc='Supply slack variables (only products with a specified supply)')

    scaling = {j: model.scaling_vector[j] for j in data['PROCESS'][None]}
    supply_set = set(supply_products)

    def demand_constraint(model, i):
        """Fixes a value in the demand vector"""
        processes, coefs = tech_rows[i]
        lhs = LinearExpression(constant=0, linear_coefs=coefs, linear_vars=[scaling[j] for j in processes])
        if i in supply_set:
            return lhs == model.FINAL_DEMAND[i] + model.slack[i]
        return lhs == model.FINAL_DEMAND[i]

    def impact_constraint(model, h):
        """Calculates all the impact categories"""
        processes, coefs = env_rows[h]
        lhs = LinearExpression(constant=0, linear_coefs=coefs, linear_vars=[scaling[j] for j in processes])
        return model.impacts[h] == lhs

    def inventory_constraint(model, g):
        """Calculates the environmental flows"""
        processes, coefs = inv_rows[g]
        lhs = LinearExpression(constant=0, linear_coefs=coefs, linear_vars=[scaling[j] for j in processes])
        return model.inv_vector[g] == lhs

    def dependent_constraint(model, constraint_name):
        """Dependent constraint: sum of left side weights * scaling <= sum of right side weights * scaling"""
        left_sum = pyo.quicksum(model.LEFT_WEIGHTS[constraint_name, j] * scaling[j] for j in model.PROCESS)
        right_sum = pyo.quicksum(model.RIGHT_WEIGHTS[constraint_name, j] * scaling[j] for j in model.PROCESS)
        return left_sum <= right_sum

    # Constraints
    model.FINAL_DEMAND_CNSTR = pyo.Constraint(model.PRODUCT, rule=demand_constraint)
    model.IMPACTS_CNSTR = pyo.Constraint(model.INDICATOR, rule=impact_constraint)
    model.INVENTORY_CNSTR = pyo.Constraint(model.INV, rule=inventory_constraint)
    model.DEPENDENT_CNSTR = pyo.Constraint(model.DEPENDENT_CONSTRAINTS, rule=dependent_constraint)

    # Objective: a weighted sum over all indicators. Typically, the indicator of study has weight 1, the rest 0.
    model.OBJ = pyo.Objective(sense=pyo.minimize, expr=pyo.quicksum(model.impacts[h] * model.WEIGHTS[h] for h in model.INDICATOR))

    print('Instance created')
    return model


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

    These values keep residuals ~1Ã—10â»â¹ (enough for 6-8 significant-digit LCA
    results) while guarding against ill-scaled data.  Add extras like
    `"TimeLimit": 600` or `"MIPGap": 1e-8` to the same dict.
    """
    
    tee = False

    if options:
        for key, val in options.items():
            if key != "tee":
                solver.options[key] = val
            else:
                tee = val

    # Solve. The results object is a standard Pyomo SolverResults.
    results = solver.solve(
        model_instance,
        tee=tee,               
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
