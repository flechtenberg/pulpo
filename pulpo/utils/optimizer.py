import pyomo.environ as pyo

def create_model():
    """Builds an abstract model on top of the ecoinvent database."""
    model = pyo.AbstractModel()

    # Sets
    model.PRODUCT = pyo.Set(doc='Set of intermediate products (or technosphere exchanges), indexed by i')
    model.PROCESS = pyo.Set(doc='Set of processes (or activities), indexed by p')
    model.ELEMENTARY = pyo.Set(doc='Set of elementary flows, indexed by e')
    model.INDICATOR = pyo.Set(doc='Set of impact assessment indicators, indexed by h')
    model.ENV = pyo.Set(doc='Set of environmental flows, indexed by a')
    model.ELEMENTARY_PROCESS = pyo.Set(within=model.ELEMENTARY * model.PROCESS * model.INDICATOR, doc='Relation set between elementary flows and processes')
    model.ELEMENTARY_IN = pyo.Set(model.INDICATOR, within=model.ELEMENTARY)
    model.ELEMENTARY_OUT = pyo.Set(model.ELEMENTARY * model.INDICATOR, within=model.PROCESS)
    model.PROCESS_IN = pyo.Set(model.PROCESS, within=model.PRODUCT, doc='Subset of intermediate products i produced/absorbed by process p')
    model.PROCESS_OUT = pyo.Set(model.PRODUCT, within=model.PROCESS, doc='Subset of processes p that produce/absorb intermediate product i')
    model.PRODUCT_PROCESS = pyo.Set(within=model.PRODUCT * model.PROCESS, doc='Relation set between intermediate products and processes')
    model.ENV_PROCESS = pyo.Set(within=model.ENV * model.PROCESS, doc='Relation set between environmental flows and processes')
    model.ENV_OUT = pyo.Set(model.ENV, within=model.PROCESS, doc='Subset of environmental flows a related processes p')

    # Parameters
    model.UPPER_LIMIT = pyo.Param(model.PROCESS, mutable=True, within=pyo.Reals, doc='Maximum production capacity of process p')
    model.LOWER_LIMIT = pyo.Param(model.PROCESS, mutable=True, within=pyo.Reals, doc='Minimum production capacity of process p')
    model.UPPER_ENV_LIMIT = pyo.Param(model.ENV, mutable=True, within=pyo.Reals, doc='Maximum elementary flow a')
    model.ELEMENTARY_MATRIX = pyo.Param(model.ELEMENTARY_PROCESS, mutable=True, doc='Biosphere Impact matrix Q*B describing the elementary flow e entering/leaving process p')
    model.ENV_MATRIX = pyo.Param(model.ENV_PROCESS, mutable=True, doc='Biosphere Impact matrix B describing the elementary flow e entering/leaving process p')
    model.FINAL_DEMAND = pyo.Param(model.PRODUCT, mutable=True, within=pyo.Reals, doc='Final demand of intermediate flows (i.e., functional unit)')
    model.SUPPLY = pyo.Param(model.PRODUCT, mutable=True, within=pyo.Binary, doc='Binary parameter which specifies whether or not a supply has been specified instead of a demand')
    model.TECH_MATRIX = pyo.Param(model.PRODUCT_PROCESS, mutable=True, doc='Technology matrix A describing the intermediate product i produced/absorbed by process p')
    model.WEIGHTS = pyo.Param(model.INDICATOR, mutable=True, within=pyo.NonNegativeReals, doc='Weighting factors for the impact assessment indicators in the objective function')

    # Variables
    model.impacts = pyo.Var(model.INDICATOR, bounds=(-1e24, 1e24), doc='Environmental impact on indicator h evaluated with the established LCIA method')
    model.scaling_vector = pyo.Var(model.PROCESS, bounds=(-1e24, 1e24), doc='Activity level of each process to meet the final demand')
    model.env_vector = pyo.Var(model.ENV, bounds=(-1e24, 1e24), doc='Environmental flows')
    model.slack = pyo.Var(model.PRODUCT, bounds=(0, 1e24), doc='Supply slack variables')

    # Building rules for sets
    model.Elem_in_out = pyo.BuildAction(rule=populate_elem)
    model.Process_in_out = pyo.BuildAction(rule=populate_in_and_out)
    model.Env_in_out = pyo.BuildAction(rule=populate_env)

    # Constraints
    model.FINAL_DEMAND_CNSTR = pyo.Constraint(model.PRODUCT, rule=demand_constraint)
    model.IMPACTS_CNSTR = pyo.Constraint(model.INDICATOR, rule=impact_constraint)
    model.INVENTORY_CNSTR = pyo.Constraint(model.ENV, rule=inventory_constraint)
    model.UPPER_CNSTR = pyo.Constraint(model.PROCESS, rule=upper_constraint)
    model.LOWER_CNSTR = pyo.Constraint(model.PROCESS, rule=lower_constraint)
    model.SLACK_CNSTR = pyo.Constraint(model.PRODUCT, rule=slack_constraint)
    model.ENV_CNSTR = pyo.Constraint(model.ENV, rule=upper_env_constraint)

    # Objective function
    model.OBJ = pyo.Objective(sense=pyo.minimize, rule=objective_function)

    return model


# Rule functions
def populate_elem(model):
    """Relates the environmental flows to the processes."""
    for i, j, h in model.ELEMENTARY_PROCESS:
        if i not in model.ELEMENTARY_IN[h]:
            model.ELEMENTARY_IN[h].add(i)
        model.ELEMENTARY_OUT[i, h].add(j)


def populate_in_and_out(model):
    """Relates the inputs of an activity to its outputs."""
    for i, j in model.PRODUCT_PROCESS:
        model.PROCESS_OUT[i].add(j)
        model.PROCESS_IN[j].add(i)

def populate_env(model):
    """Relates the impacts to the environmental flows"""
    for a, j in model.ENV_PROCESS:
        model.ENV_OUT[a].add(j)

def demand_constraint(model, i):
    """Fixes a value in the demand vector"""
    return sum(model.TECH_MATRIX[i, p] * model.scaling_vector[p] for p in model.PROCESS_OUT[i]) == model.FINAL_DEMAND[i] + model.slack[i]


def impact_constraint(model, h):
    """Calculates all the impact categories"""
    return model.impacts[h] == sum(sum(model.ELEMENTARY_MATRIX[i, j, h] * model.scaling_vector[j] for j in model.ELEMENTARY_OUT[i, h]) for i in model.ELEMENTARY_IN[h])

def inventory_constraint(model, a):
    """Calculates the environmental flows"""
    return model.env_vector[a] == sum(model.ENV_MATRIX[a, j] * model.scaling_vector[j] for j in model.ENV_OUT[a])


def upper_constraint(model, p):
    """Ensures that variables are within capacities (Maximum production constraint) """
    return model.scaling_vector[p] <= model.UPPER_LIMIT[p]


def lower_constraint(model, p):
    """ Minimum production constraint """
    return model.scaling_vector[p] >= model.LOWER_LIMIT[p]

def upper_env_constraint(model, a):
    """Ensures that variables are within capacities (Maximum production constraint) """
    return model.env_vector[a] <= model.UPPER_ENV_LIMIT[a]

def slack_constraint(model, p):
    """ Slack variable upper limit for activities where supply is specified instead of demand """
    return model.slack[p] <= 1e20 * model.SUPPLY[p]


def objective_function(model):
    """Objective is a sum over all indicators with weights. Typically, the indicator of study has weight 1, the rest 0"""
    return sum(model.impacts[h] * model.WEIGHTS[h] for h in model.INDICATOR)

def calculate_methods(instance, lci_data, methods):
    '''
    This function calculates the impacts if a method with weight 0 has been specified
    '''
    import scipy.sparse as sparse
    import numpy as np
    matrices = lci_data['matrices']
    biosphere = lci_data['biosphere']
    matrices = {h: matrices[h] for h in matrices if str(h) in methods}
    env_cost = {h: sparse.csr_matrix.dot(matrices[h], biosphere) for h in matrices}
    try:
        scaling_vector = np.array([instance.scaling_vector[x].value for x in instance.scaling_vector])
    except:
        scaling_vector = np.array([instance.scaling_vector[x] for x in instance.scaling_vector])
    impacts = {}
    for h in matrices:
        impacts[h] = sum(env_cost[h].dot(scaling_vector))
    instance.impacts_calculated = pyo.Var([h for h in impacts], initialize=impacts)
    return instance

def calculate_env_flows(instance, lci_data):
    '''
    This function calculates elementary flows post-calculation
    '''
    import numpy as np
    biosphere = lci_data['biosphere']
    try:
        scaling_vector = np.array([instance.scaling_vector[x].value for x in instance.scaling_vector])
    except:
        scaling_vector = np.array([instance.scaling_vector[x] for x in instance.scaling_vector])
    flows = biosphere.dot(scaling_vector)
    instance.elem_flows = pyo.Var(range(0, biosphere.shape[0]), initialize=flows)
    return instance



def instantiate(model_data):
    """ This function builds an instance of the optimization model with specific data and objective function"""
    print('Creating Instance')
    model = create_model()
    problem = model.create_instance(model_data, report_timing=False)
    print('Instance created')
    return problem


def solve_model(model_instance, gams_path, options=None):
    """ TODO enable ipopt or other free solver application! """
    """Solves the instance of the optimization model.

    Parameters:
    ------------
    model_instance: pyomo.ConcreteModel
        The instance of the Pyomo model to solve.
    gams_path: str
        The directory path of GAMS.
    options: list of str, optional
        Additional solver options.

    Returns:
    ------------
    pyomo.SolverResults, pyomo.ConcreteModel
        The solver results and the updated instance of the Pyomo model.
    """
    if gams_path is not False:
        pyo.pyomo.common.Executable('gams').set_path(gams_path)
        solver = pyo.SolverFactory('gams')
        print('GAMS solvers library availability:', solver.available())
        print('Solver path:', solver.executable())

        io_options = {
            'mtype': 'lp',                      # Type of problem (lp, nlp, mip, minlp)
            'solver': 'CPLEX',                  # Name of solver
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
            tee=True,
            report_timing=True,
            io_options=io_options,
            add_options=options
        )

        model_instance.solutions.load_from(results)

    else:
        from pyomo.contrib import appsi
        opt = appsi.solvers.Highs()
        results = opt.solve(model_instance)

    return results, model_instance


