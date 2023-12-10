import scipy.sparse as sparse

def combine_inputs(lci_data, demand, choices, upper_limit, lower_limit, upper_elem_limit, methods):
    ''' This function combines all the inputs to a dictionary as an input for the optimization model'''
    ''' Load LCIA methods into a list of matrices'''
    matrices = lci_data['matrices']
    biosphere = lci_data['biosphere']
    technosphere = lci_data['technosphere']
    activity_map = lci_data['activity_map']
    elem_map = lci_data['elem_map']


    ''' Remove matrices that are not part of the objective '''
    matrices = {h: matrices[h] for h in matrices if str(h) in methods}

    ''' Prepare the characterization factor matrix Q'''
    char_dict = {(h,e): matrices[h][e,e]
                      for e in range(0, biosphere.shape[0])
                      for h in matrices if matrices[h][e,e] != 0}

    ''' Convert sparse csr biosphere impact matrix to dictionary'''
    biosphere_dict = {(e - 1, biosphere.indices[j]): biosphere.data[j]
                         for e in range(1, biosphere.shape[0] + 1)
                         for j in range(biosphere.indptr[e - 1], biosphere.indptr[e])}

    ''' Convert sparse csr technosphere matrix to dictionary'''
    technosphere_dict = {(i - 1, technosphere.indices[j]): technosphere.data[j]
                         for i in range(1, technosphere.shape[0] + 1)
                         for j in range(technosphere.indptr[i - 1], technosphere.indptr[i])}

    ''' Make technosphere matrix rectangular and update keys and product_ids'''
    capacity_dict = {}
    keys = {}  # store the activity keys of the choices
    product_ids = []  # store the product IDs of the choices
    for key, activities in choices.items():
        for act in activities:
            product_id = activity_map[act.key]
            capacity_dict[product_id] = choices[key][act]
            keys[product_id] = key
            product_ids.append(product_id)

    for product, process in list(technosphere_dict):
        if product in product_ids:
            if (keys[product], process) not in technosphere_dict:
                technosphere_dict[(keys[product], process)] = technosphere_dict[(product, process)]
            else:
                technosphere_dict[(keys[product], process)] += technosphere_dict[(product, process)]
            del technosphere_dict[(product, process)]

    ''' Create sets using set comprehensions for better performance'''
    PRODUCTS = {None: list({i[0] for i in technosphere_dict})}
    PROCESS = {None: list({i[1] for i in technosphere_dict})}
    ELEMENTARY = {None: list({k[0] for k in biosphere_dict})}
    INDICATOR = {None: list({h for h in matrices})}
    PRODUCT_PROCESS = {None: list({(i[0], i[1]) for i in technosphere_dict})}
    ELEMENTARY_PROCESS = {None: list({(i[0], i[1]) for i in biosphere_dict})}
    INDICATOR_ELEMENTARY = {None: list({(i[0], i[1]) for i in char_dict})}

    ''' Specify the demand'''
    demand_dict = {prod: 0 for prod in PRODUCTS[None]}
    for dem in demand:
        demand_dict[activity_map[dem]] = demand[dem]

    ''' Specify the lower limit '''
    lower_limit_dict = {proc: -1e20 for proc in PROCESS[None]}
    for act in lower_limit:
        lower_limit_dict[activity_map[act]] = lower_limit[act]
    for choice in choices:
        for act in choices[choice]:
            lower_limit_dict[activity_map[act]] = 0

    ''' Specify the upper limit '''
    upper_limit_dict = {proc: 1e20 for proc in PROCESS[None]}
    for act in upper_limit:
        upper_limit_dict[activity_map[act]] = upper_limit[act]

    ''' Specify the upper elementary flow limit '''
    upper_elem_limit_dict = {elem: 1e24 for elem in ELEMENTARY[None]}
    for elem in upper_elem_limit:
        upper_elem_limit_dict[elem_map[elem.key]] = upper_elem_limit[elem]

    ''' Check if a supply has been specified '''
    supply_dict = {prod: 0 for prod in PRODUCTS[None]}
    for act in list(lower_limit.keys() & upper_limit.keys()):
        supply_dict[activity_map[act]] = 1 if lower_limit[act] == upper_limit[act] else 0

    ''' Create weights'''
    weights = {method: 1 for method in matrices} if methods == {} else methods

    ''' Assemble the final data dictionary'''
    model_data = {
        None: {
            'PRODUCT': PRODUCTS,
            'PROCESS': PROCESS,
            'ELEMENTARY': ELEMENTARY,
            'INDICATOR': INDICATOR,
            'PRODUCT_PROCESS': PRODUCT_PROCESS,
            'ELEMENTARY_PROCESS': ELEMENTARY_PROCESS,
            'INDICATOR_ELEMENTARY': INDICATOR_ELEMENTARY,
            'TECH_MATRIX': technosphere_dict,
            'ELEMENTARY_MATRIX': biosphere_dict,
            'CHAR_MATRIX': char_dict,
            'FINAL_DEMAND': demand_dict,
            'SUPPLY': supply_dict,
            'LOWER_LIMIT': lower_limit_dict,
            'UPPER_LIMIT': upper_limit_dict,
            'UPPER_ELEM_LIMIT': upper_elem_limit_dict,
            'WEIGHTS': weights,
        }
    }
    return model_data

def convert_to_dict(input_data):
    ''' Checks if the passed method is in the correct format '''
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

