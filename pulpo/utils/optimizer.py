import pyomo.environ as pyo
from pyomo.contrib import appsi
from .saver import extract_flows
import os

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

    # Building rules for sets
    model.Env_in_out = pyo.BuildAction(rule=populate_env)
    model.Process_in_out = pyo.BuildAction(rule=populate_in_and_out)
    model.Inv_in_out = pyo.BuildAction(rule=populate_inv)

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
    print('Optimization problem solved using Highspy')
    return results, model_instance

def solve_neos(model_instance, solver_name, options):
    """Solve the model using NEOS."""
    # ATTN: Perhaps enable passing down the mail address as an argument
    if 'NEOS_EMAIL' not in os.environ:
        print("'NEOS_EMAIL' environment variable is not set. \n")
        print("To use the NEOS solver, please set the 'NEOS_EMAIL' environment variable as explained here:\n")
        print("https://www.twilio.com/en-us/blog/how-to-set-environment-variables-html \n")
        print("If you do not have a NEOS account, please create one at https://neos-server.org/neos/ \n")
        return None, model_instance

    solver_manager = pyo.SolverManagerFactory('neos')
    kwargs = {'solver': solver_name}
    if options:
        kwargs['options'] = options
    results = solver_manager.solve(model_instance, **kwargs)

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


def solve_model(model_instance, gams_path=False, solver_name=None, options=None):
    """
    Solves the instance of the optimization model using Highspy, NEOS, or GAMS.

    Args:
        model_instance (ConcreteModel): The Pyomo model instance.
        gams_path (str or bool, optional): Path to the GAMS solver or True to use the environment variable.
        solver_name (str, optional): The solver to use (e.g. 'cplex', 'baron', or 'xpress').
        options (list, optional): Additional options for the solver.

    Returns:
        tuple: Results of the optimization and the updated model instance.
    """
    # ATTN: Cases may be too convoluted. Tidy up the logic eventually.
    # Case 1: Use Highspy if no GAMS path is provided and the solver is either not specified or is 'highs'
    if gams_path is False and (solver_name is None or 'highs' in solver_name.lower()):
        return solve_highspy(model_instance)

    # Case 2: Use NEOS if a solver_name is provided (and it is not Highspy) and no GAMS path is provided
    if gams_path is False and solver_name and ('highs' not in solver_name.lower()):
        return solve_neos(model_instance, solver_name, options)

    # Case 3: Use GAMS if gams_path is specified (either as a path or True)
    if gams_path:
        return solve_gams(model_instance, gams_path, options)

    # Default case: Return None if no valid solver configuration is found
    print("No valid solver configuration found.")
    return None, model_instance
