from pulpo.utils import saver

def solve_model_MC(worker, n_it=100, gams_path=False, solver_name=None, options=None):
    """
    Perform very simple Monte Carlo (MC) simulations to solve a model multiple times, each time sampling A, B, and Q again.

    Args:
        worker (object): Object managing the model, with methods `get_lci_data(seed)`, `instantiate(...)`, and `solve(...)`.
        n_it (int, optional): Number of Monte Carlo iterations. Default is 100.
        gams_path (str or bool, optional): Path to GAMS executable or `False`. Default is `False`.
        solver_name (str, optional): Solver name. Default is `None`.
        options (dict, optional): Solver options. Default is `None`.

    Returns:
        list: Objective values from each Monte Carlo iteration.
    """
    
    results = []
    for i in range(n_it):
        worker.get_lci_data(seed=i+1)
        worker.instantiate(choices=worker.choices, demand=worker.demand, upper_limit=worker.upper_limit, lower_limit=worker.lower_limit, upper_elem_limit=worker.upper_elem_limit, upper_imp_limit=worker.upper_imp_limit)   
        worker.solve(gams_path, solver_name=solver_name, options=options)
        results.append(worker.instance.OBJ())
    return results

def run_gsa(worker):
    1+1




