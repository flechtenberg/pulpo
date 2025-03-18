import scipy.sparse as sparse


def combine_inputs(lci_data, demand, choices, upper_limit, lower_limit, upper_inv_limit, upper_imp_limit, methods):
    """
    Combines all the inputs into a dictionary as an input for the optimization model.

    Args:
        lci_data (dict): LCI data containing matrices and mappings.
        demand (dict): Demand data.
        choices (dict): Choices for the model.
        upper_limit (dict): Upper limit constraints.
        lower_limit (dict): Lower limit constraints.
        upper_inv_limit (dict): Upper intervention limit constraints.
        upper_imp_limit (dict): Upper impact limit constraints.
        methods (dict): Methods for environmental impact assessment.

    Returns:
        dict: Combined data dictionary for the optimization model.
    """

    # Load LCI data matrices and mappings
    matrices = lci_data['matrices']
    intervention_matrix = lci_data['intervention_matrix']
    technology_matrix = lci_data['technology_matrix']
    process_map = lci_data['process_map']
    intervention_map = lci_data['intervention_map']

    # Remove matrices that are not part of the objective
    matrices = {h: matrices[h] for h in matrices if str(h) in methods}

    # Prepare the environmental impact matrices Q[h]*B
    env_cost = {h: matrices[h].diagonal() @ intervention_matrix for h in matrices}
    env_cost_dict = {(j, h): env_cost[h][j] for h in matrices for j in range(len(env_cost[h]))}

    # Convert sparse csr technology matrix to dictionary
    technology_matrix_dict = {(i - 1, technology_matrix.indices[j]): technology_matrix.data[j]
                              for i in range(1, technology_matrix.shape[0] + 1)
                              for j in range(technology_matrix.indptr[i - 1], technology_matrix.indptr[i])}

    # Convert sparse csr intervention flow matrix to dictionary
    inv_to_consider = [intervention_map[g] for g in upper_inv_limit]
    inv_dict = {(g, intervention_matrix.indices[j]): intervention_matrix.data[j]
                for g in inv_to_consider
                for j in range(intervention_matrix.indptr[g], intervention_matrix.indptr[g + 1])}

    # Make technology matrix rectangular and update keys and product_ids
    keys = {}  # Store the activity keys of the choices
    product_ids = []  # Store the product IDs of the choices
    for key, processes in choices.items():
        for proc in processes:
            product_id = process_map[proc.key]
            keys[product_id] = key
            product_ids.append(product_id)

    for product, process in list(technology_matrix_dict):
        if product in product_ids:
            if (keys[product], process) not in technology_matrix_dict:
                technology_matrix_dict[(keys[product], process)] = technology_matrix_dict[(product, process)]
            else:
                technology_matrix_dict[(keys[product], process)] += technology_matrix_dict[(product, process)]
            del technology_matrix_dict[(product, process)]

    # Create sets using set comprehensions for better performance
    PRODUCTS = {None: list({i[0] for i in technology_matrix_dict})}
    PROCESS = {None: list({i[1] for i in technology_matrix_dict})}
    PRODUCT_PROCESS = {None: list({(i[0], i[1]) for i in technology_matrix_dict})}
    ENV_COST = {None: list({i[0] for i in env_cost_dict})}
    ENV_COST_PROCESS = {None: list({i for i in env_cost_dict})}
    INV = {None: list({i[0] for i in inv_dict})}
    INV_PROCESS = {None: list({(i[0], i[1]) for i in inv_dict})}
    INDICATOR = {None: list({h for h in matrices})}

    # Specify the demand
    demand_dict = {prod: 0 for prod in PRODUCTS[None]}
    for dem in demand:
        if dem in process_map:
            # Case when dem is a key in process_map
            demand_dict[process_map[dem]] = demand[dem]
        elif dem in choices:
            # Case when dem is already one of the elements in process_map
            demand_dict[dem] = demand[dem]
        else:
            # Case when dem is neither a key nor a value in process_map
            raise ValueError(f"'{dem}' is not found in process_map keys or values.")

    # Specify the lower limit
    lower_limit_dict = {proc: -1e20 for proc in PROCESS[None]}
    for choice in choices:
        for proc in choices[choice]:
            lower_limit_dict[process_map[proc]] = 0
    for proc in lower_limit:
        lower_limit_dict[process_map[proc]] = lower_limit[proc]

    # Specify the upper limit
    upper_limit_dict = {proc: 1e20 for proc in PROCESS[None]}
    for proc in upper_limit:
        upper_limit_dict[process_map[proc]] = upper_limit[proc]
    for choice in choices:
        for proc in choices[choice]:
            upper_limit_dict[process_map[proc]] = choices[choice][proc]

    # Check if a supply has been specified
    supply_dict = {prod: 0 for prod in PRODUCTS[None]}
    for proc in list(lower_limit.keys() & upper_limit.keys()):
        supply_dict[process_map[proc]] = 1 if lower_limit[proc] == upper_limit[proc] else 0

    # Specify the upper elementary flow limit
    upper_inv_limit_dict = {elem: 1e24 for elem in INV[None]}
    for inv in upper_inv_limit:
        upper_inv_limit_dict[intervention_map[inv.key]] = upper_inv_limit[inv]

    # Specify the upper impact category limit
    upper_imp_limit_dict = {imp: 1e24 for imp in INDICATOR[None]}
    for imp in upper_imp_limit:
        upper_imp_limit_dict[imp] = upper_imp_limit[imp]

    # Create weights
    weights = {method: 1 for method in matrices} if methods == {} else methods

    # Assemble the final data dictionary
    model_data = {
        None: {
            'PRODUCT': PRODUCTS,
            'PROCESS': PROCESS,
            'ENV_COST': ENV_COST,
            'INDICATOR': INDICATOR,
            'INV': INV,
            'PRODUCT_PROCESS': PRODUCT_PROCESS,
            'ENV_COST_PROCESS': ENV_COST_PROCESS,
            'INV_PROCESS': INV_PROCESS,
            'TECH_MATRIX': technology_matrix_dict,
            'ENV_COST_MATRIX': env_cost_dict,
            'INV_MATRIX': inv_dict,
            'FINAL_DEMAND': demand_dict,
            'SUPPLY': supply_dict,
            'LOWER_LIMIT': lower_limit_dict,
            'UPPER_LIMIT': upper_limit_dict,
            'UPPER_INV_LIMIT': upper_inv_limit_dict,
            'UPPER_IMP_LIMIT': upper_imp_limit_dict,
            'WEIGHTS': weights,
        }
    }
    return model_data


def convert_to_dict(input_data):
    """
    Checks if the passed method is in the correct format and converts it accordingly.

    Args:
        input_data (Union[str, list, dict]): Input data to be converted.

    Returns:
        dict: Converted dictionary with methods and their weights.

    Raises:
        ValueError: If the input data is not in the expected format.
    """

    # Check if the input is a string
    if isinstance(input_data, str):
        return {input_data: 1}

    # Check if the input is a list of strings
    elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
        return {item: 1 for item in input_data}

    # Check if the input is a dictionary with numerical values
    elif isinstance(input_data, dict) and all(isinstance(value, (int, float)) for value in input_data.values()):
        return input_data

    # If the input is of any other type, raise an error
    else:
        raise ValueError(
            "Input should be either a string, a list of strings, or a dictionary with string indices and numerical values")
