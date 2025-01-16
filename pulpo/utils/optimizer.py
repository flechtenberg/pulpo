import pyomo.environ as pyo
from pyomo.contrib import appsi

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
    model.ENV_COST = pyo.Set(doc='Set of environmental cost flows, indexed by e')
    model.INDICATOR = pyo.Set(doc='Set of impact assessment indicators, indexed by h')
    model.INV = pyo.Set(doc='Set of intervention flows, indexed by g')
    model.ENV_COST_PROCESS = pyo.Set(within=model.ENV_COST * model.PROCESS * model.INDICATOR, doc='Relation set between environmental cost flows and processes')
    model.ENV_COST_IN = pyo.Set(model.INDICATOR, within=model.ENV_COST)
    model.ENV_COST_OUT = pyo.Set(model.ENV_COST * model.INDICATOR, within=model.PROCESS)
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
    for i, j, h in model.ENV_COST_PROCESS:
        if i not in model.ENV_COST_IN[h]:
            model.ENV_COST_IN[h].add(i)
        model.ENV_COST_OUT[i, h].add(j)


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
    return model.impacts[h] == sum(sum(model.ENV_COST_MATRIX[i, j, h] * model.scaling_vector[j] for j in model.ENV_COST_OUT[i, h]) for i in model.ENV_COST_IN[h])

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
    import scipy.sparse as sparse
    import numpy as np
    matrices = lci_data['matrices']
    intervention_matrix = lci_data['intervention_matrix']
    matrices = {h: matrices[h] for h in matrices if str(h) in methods}
    env_cost = {h: sparse.csr_matrix.dot(matrices[h], intervention_matrix) for h in matrices}
    try:
        scaling_vector = np.array([instance.scaling_vector[x].value for x in instance.scaling_vector])
    except:
        scaling_vector = np.array([instance.scaling_vector[x] for x in instance.scaling_vector])

    impacts = {}
    for h in matrices:
        impacts[h] = sum(env_cost[h].dot(scaling_vector))

    # Check if instance.impacts_calculated exists
    if hasattr(instance, 'impacts_calculated'):
        # Update values if it already exists
        for h in impacts:
            instance.impacts_calculated[h].value = impacts[h]
    else:
        # Create instance.impacts_calculated
        instance.impacts_calculated = pyo.Var([h for h in impacts], initialize=impacts)
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
    import numpy as np
    intervention_matrix = lci_data['intervention_matrix']
    try:
        scaling_vector = np.array([instance.scaling_vector[x].value for x in instance.scaling_vector])
    except:
        scaling_vector = np.array([instance.scaling_vector[x] for x in instance.scaling_vector])
    flows = intervention_matrix.dot(scaling_vector)
    # Check if inv_flows already exists in the model
    # @TODO: Consider adding "inv_flows" directly as variable to the model and skip this check.
    if hasattr(instance, 'inv_flows'):
        # Update the values of the existing variable
        for i in range(intervention_matrix.shape[0]):
            instance.inv_flows[i].value = flows[i]
    else:
        # Create the variable if it does not exist
        instance.inv_flows = pyo.Var(range(0, intervention_matrix.shape[0]), initialize=dict(enumerate(flows)))
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


def solve_model(model_instance, gams_path=None, solver_name=None, options=None):
    """
    Solves the instance of the optimization model.

    Args:
        model_instance (ConcreteModel): The Pyomo model instance.
        gams_path (str, optional): Path to the GAMS solver. If None, GAMS will not be used.
        solver_name (str, optional): The solver to use ('highs', 'gams', or 'ipopt'). Defaults to 'highs' unless gams_path is provided.
        options (list, optional): Additional options for the solver.

    Returns:
        tuple: Results of the optimization and the updated model instance.
    """
    results = None

    # Use GAMS if gams_path is specified
    if gams_path and (solver_name is None or solver_name.lower() == 'gams'):
        pyo.pyomo.common.Executable('gams').set_path(gams_path)
        solver = pyo.SolverFactory('gams')
        print('GAMS solvers library availability:', solver.available())
        print('Solver path:', solver.executable())

        io_options = {
            'mtype': 'lp',  # Type of problem (lp, nlp, mip, minlp)
            'solver': 'CPLEX',  # Name of solver
        }

        if options is None:
            options = [
                'option optcr = 1e-15;',
                'option reslim = 3600;',
                'GAMS_MODEL.optfile = 1;',
                '$onecho > cplex.opt',
                'workmem=4096',
                'scaind=1',
                #'numericalemphasis=1',
                #'epmrk=0.99',
                #'eprhs=1E-9',
                '$offecho',
            ]

        results = solver.solve(
            model_instance,
            keepfiles=True,
            symbolic_solver_labels=True,
            tee=False,
            report_timing=False,
            io_options=io_options,
            add_options=options
        )

        model_instance.solutions.load_from(results)

    # Use IPOPT if explicitly specified
    elif solver_name and solver_name.lower() == 'ipopt':
        opt = pyo.SolverFactory('ipopt')
        if options:
            for option in options:
                opt.options[option] = True
        results = opt.solve(model_instance)

    # Default to HiGHS if no solver specified or if solver_name is 'highs'
    else:
        opt = appsi.solvers.Highs()
        results = opt.solve(model_instance)

    return results, model_instance


    print('Optimization problem solved')
    ## TODO: Add a check for infeasibility and other solver errors

    return results, model_instance