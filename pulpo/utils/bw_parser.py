from typing import List, Union, Dict, Any
import bw2calc as bc
import bw2data as bd
from pulpo.utils.tests import is_bw25

def import_data(project: str, database: str, method: Union[str, List[str], dict[str, int]],
                intervention_matrix_name: str) -> Dict[str, Union[dict, Any]]:
    """
    Main function to import LCI data for a project from a database.

    Args:
        project (str): Name of the project.
        database (str): Name of the database.
        method (Union[str, List[str], dict[str, int]]): Method(s) for data retrieval.
        intervention_matrix_name (str): Name of the intervention matrix.

    Returns:
        Dict[str, Union[dict, Any]]: Dictionary containing imported LCI data.
    """

    # Set project and get database
    bd.projects.set_current(project)
    eidb = bd.Database(database)

    # Prepare methods
    if isinstance(method, str):
        method = [method]  # Convert single string to list of strings
    method = sorted(method)
    methods = retrieve_methods(project, method)

    rand_act = eidb.random()

    bw25 = is_bw25()
    if bw25:
        # Process with bw25 logic
        functional_units_1 = {"act1": {rand_act.id: 1}}
        config_1 = {"impact_categories": methods}

        data_objs_1 = bd.get_multilca_data_objs(functional_units=functional_units_1, method_config=config_1)

        lca = bc.MultiLCA(demands=functional_units_1, method_config=config_1, data_objs=data_objs_1)

        lca.load_lci_data()
        lca.load_lcia_data()

        # Store the characterization (C) matrix for the method
        characterization_matrices = {str(method): lca.characterization_matrices[method] for method in methods}

    else:
        characterization_matrices = {}
        for method in methods:
            # Set up the LCA calculation with a functional unit of the random activity and the current method
            lca = bc.LCA({rand_act: 1}, method)

            # Load the LCI and LCIA data
            lca.lci()
            lca.lcia()

            # Store the characterization (C) matrix for the method
            characterization_matrices[str(method)] = lca.characterization_matrix

    # Extract A (Technosphere) and B (Biosphere) matrices from the LCA
    technology_matrix = lca.technosphere_matrix  # A matrix
    intervention_matrix = lca.biosphere_matrix  # B matrix

    if bw25:
        # Create activity map key --> ID | ID --> description
        process_map = {act.key: lca.dicts.product[act.id] for act in eidb}  # TODO: This is adherring to old ways of storring data with keys ... how to work with IDs instead?
    else:
        process_map = lca.product_dict

    for act in eidb:
        process_map[process_map[act.key]] = str(act['name']) + ' | ' + str(act['reference product']) + ' | ' + str(
            act['location'])

    if intervention_matrix_name in bd.databases:
        eidb_bio = bd.Database(intervention_matrix_name)
        if bw25:
            intervention_map = {act.key: lca.dicts.biosphere[act.id] for act in eidb_bio if act.id in lca.dicts.biosphere}  # TODO: This is adherring to old ways of storring data with keys ... how to work with IDs instead?
        else:
            intervention_map = lca.biosphere_dict
        for act in eidb_bio:
            if act.key in intervention_map:
                intervention_map[intervention_map[act.key]] = act['name'] + ' | ' + str(act['categories'])
    else:
        print(
            "The name of the biosphere is not '" + intervention_matrix_name + "'. Please specify the correct biosphere.")
        return {}

    lci_data = {
        'matrices': characterization_matrices,
        'intervention_matrix': intervention_matrix,
        'technology_matrix': technology_matrix,
        'process_map': process_map,
        'intervention_map': intervention_map,
    }

    return lci_data


def retrieve_processes(project: str, database: str, keys=None, activities=None, reference_products=None,
                       locations=None):
    """
    Retrieve activities from a database based on specified keys, activities, reference products, and locations.

    Args:
        project (str): Name of the project.
        database (str): Name of the database.
        keys (list, optional): List of keys to filter activities.
        activities (list, optional): List of activity names to filter.
        reference_products (list, optional): List of reference products to filter.
        locations (list, optional): List of locations to filter.

    Returns:
        list: List of matching activities from the database.
    """

    # Set project and get database
    bd.projects.set_current(project)
    eidb = bd.Database(database)

    # Filter by keys if provided
    if keys is not None:
        if isinstance(keys, str):
            keys = [keys]
        keys = [eval(key) for key in keys]
        return [proc for proc in eidb if proc.key in keys]

    matching_processes = []

    # Filter by activities, reference products, and locations
    for proc in eidb:
        if (activities is None or proc['name'] in activities) and \
                (reference_products is None or proc['reference product'] in reference_products) and \
                (locations is None or proc['location'] in locations):
            matching_processes.append(proc)

    if not matching_processes:
        print("No activities match the given specifications or the input format is incorrect.")
    else:
        return matching_processes


def retrieve_env_interventions(project: str = '', intervention_matrix: str = 'biosphere3', keys=None, activities=None,
                               categories=None):
    """
    Retrieve environmental flows from the biosphere database based on specified keys, activities, and categories.

    Args:
        project (str, optional): Name of the project.
        intervention_matrix (str): Name of the intervention matrix.
        keys (list, optional): List of keys to filter environmental flows.
        activities (list, optional): List of activity names to filter.
        categories (list, optional): List of categories to filter.

    Returns:
        list: List of matching environmental flows from the database.
    """

    # Set project and get database
    bd.projects.set_current(project)
    eidb = bd.Database(intervention_matrix)

    # Filter by keys if provided
    if keys is not None:
        if isinstance(keys, str):
            keys = [keys]
        keys = [eval(key) for key in keys]
        return [flow for flow in eidb if flow.key in keys]

    matching_flows = []

    # Filter by activities and categories
    for flow in eidb:
        if (activities is None or flow['name'] in activities) and \
                (categories is None or str(flow['categories']) in categories):
            matching_flows.append(flow)

    if not matching_flows:
        print("No flows match the given specifications or the input format is incorrect.")
    else:
        return matching_flows


def retrieve_methods(project: str, sub_string: List[str]) -> List[str]:
    """
    Retrieve all methods that contain the specified list of substrings.

    Args:
        project (str): Name of the project.
        sub_string (List[str]): List of substrings to search for in method names.

    Returns:
        List[str]: List of methods that match the substrings.
    """
    bd.projects.set_current(project)
    return [method for method in bd.methods if any([x.lower() in str(method).lower() for x in sub_string])]

#if __name__ == '__main__':
#    if is_bw25():
#        project = "pulpo_bw25"
#        biosphere = "ecoinvent-3.8-biosphere"
#        methods = {"('ecoinvent-3.8', 'IPCC 2013', 'climate change', 'GWP 100a')": 1,
#                   "('ecoinvent-3.8', 'IPCC 2013', 'climate change', 'GWP 20a')": 0,
#                   }
#    else:
#        project = "pulpo"
#        biosphere = "biosphere3"
#        methods = {"('IPCC 2013', 'climate change', 'GWP 100a')": 1,
#                   "('IPCC 2013', 'climate change', 'GWP 20a')": 0,
#                   }
#
#    database = "ecoinvent-3.8-cutoff"
#
#
#    import_data(project, database, methods, intervention_matrix_name=biosphere)