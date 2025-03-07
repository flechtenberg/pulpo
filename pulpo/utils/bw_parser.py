from typing import List, Union, Dict, Any
import bw2calc as bc
import bw2data as bd
from pulpo.utils.utils import get_bw_version

def import_data(project: str, databases: Union[str, List[str]], method: Union[str, List[str], Dict[str, int]],
                intervention_matrix_name: str) -> Dict[str, Union[dict, Any]]:
    """
    Main function to import LCI data for a project from one or more databases.

    Args:
        project (str): Name of the project.
        databases (Union[str, List[str]]): Name of the primary database or a list of databases
                                           (foreground, background).
        method (Union[str, List[str], Dict[str, int]]): Method(s) for data retrieval.
        intervention_matrix_name (str): Name of the intervention matrix.

    Returns:
        Dict[str, Union[dict, Any]]: Dictionary containing imported LCI data.
    """


    # Normalize databases input to a list
    if isinstance(databases, str):
        databases = [databases]

    # Validate each specified database
    for db in databases:
        if db not in bd.databases:
            raise ValueError(
                f"Database '{db}' does not exist in the project '{project}'. "
                f"Available databases: {list(bd.databases.keys())}"
            )

    # Prepare methods
    if isinstance(method, str):
        method = [method]  # Convert single string to list of strings
    method = sorted(method)

    # Retrieve and validate methods
    methods = retrieve_methods(project, method)
    invalid_methods = [m for m in method if m not in [str(mt) for mt in bd.methods]]
    if invalid_methods:
        raise ValueError(
            f"The following methods do not exist in the project '{project}': {invalid_methods}. "
            f"Available methods: {[str(mt) for mt in bd.methods]}"
        )

    # Initialize database objects
    eidbs = []
    lcas = []
    for database in databases:
        eidbs.append(bd.Database(database))

    bw_version = get_bw_version()
    characterization_matrices = {}
    process_map = {}
    
    match bw_version:
        case 'bw25':
            for eidb in eidbs:
                functional_units = {"act1": {eidb.random().id: 1}}
                config = {"impact_categories": methods}
                data_objs = bd.get_multilca_data_objs(functional_units=functional_units, method_config=config)
                lca = bc.MultiLCA(demands=functional_units, method_config=config, data_objs=data_objs)
                lca.load_lci_data()
                lca.load_lcia_data()
                # Store the characterization matrices for methods
                characterization_matrices = {str(method): lca.characterization_matrices[method] for method in methods}
                lcas.append(lca)
                process_map.update({act.key: lca.dicts.product[act.id] for act in eidb})
        case 'bw2':
            for eidb in eidbs:
                for method in methods:
                    lca = bc.LCA({eidb.random(): 1}, method)
                    lca.load_lci_data()
                    lca.load_lcia_data()
                    lcas.append(lca)
                    characterization_matrices[str(method)] = lca.characterization_matrix
                lcas.append(lca)
                process_map.update(lca.product_dict)
    # Extract A (Technosphere) and B (Biosphere) matrices from the LCA
    technology_matrix = lcas[0].technosphere_matrix  # A matrix
    intervention_matrix = lcas[0].biosphere_matrix  # B matrix

    # Add descriptive strings to the process map for both primary and secondary databases
    process_map_metadata = {}
    for eidb in eidbs:
        for act in eidb:
            process_map_metadata[process_map[act.key]] = (
                f"{act['name']} | {act.get('reference product', '')} | {act.get('location', '')}"
            )

    # ATTN: BHL could probbly ask BW what the biosphere matrix is and then move this code into the cases further up
    if intervention_matrix_name in bd.databases:
        eidb_bio = bd.Database(intervention_matrix_name)
        match bw_version:
            case 'bw25':
                intervention_map = {act.key: lca.dicts.biosphere[act.id] for act in eidb_bio if act.id in lca.dicts.biosphere}  # TODO: This is adherring to old ways of storring data with keys ... how to work with IDs instead?
            case 'bw2':
                intervention_map = lca.biosphere_dict
        intervention_map_metadata = {}
        for act in eidb_bio:
            if act.key in intervention_map:
                intervention_map_metadata[intervention_map[act.key]] = act['name'] + ' | ' + str(act['categories'])
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
        'intervention_map_metadata':intervention_map_metadata,
        'process_map_metadata':process_map_metadata,
    }

    return lci_data


def retrieve_processes(project: str, databases: Union[str, List[str]], keys=None, activities=None,
                       reference_products=None, locations=None):
    """
    Retrieve activities from one or more databases based on specified keys, activities, reference products, and locations.

    Args:
        project (str): Name of the project.
        databases (Union[str, List[str]]): Name of the primary database or a list of databases (foreground, background).
        keys (list, optional): List of keys to filter activities.
        activities (list, optional): List of activity names to filter.
        reference_products (list, optional): List of reference products to filter.
        locations (list, optional): List of locations to filter.

    Returns:
        list: List of matching activities from the specified databases.
    """
    # Set project
    bd.projects.set_current(project)

    # Normalize databases to a list
    if isinstance(databases, str):
        databases = [databases]

    # Ensure filters are lists
    if activities is not None and not isinstance(activities, list):
        activities = [activities]
    if reference_products is not None and not isinstance(reference_products, list):
        reference_products = [reference_products]
    if locations is not None and not isinstance(locations, list):
        locations = [locations]

    matching_processes = []

    # Search each database
    for db_name in databases:
        eidb = bd.Database(db_name)

        # Preprocess keys for fast lookup if provided
        if keys is not None:
            if isinstance(keys, str):
                keys = [keys]
            keys_set = set(eval(key) for key in keys)  # Use a set for faster lookup
            matching_processes.extend([proc for proc in eidb if proc.key in keys_set])
        else:
            # Preprocess filters for fast lookup
            activity_set = set(activities) if activities is not None else None
            reference_product_set = set(reference_products) if reference_products is not None else None
            location_set = set(locations) if locations is not None else None

            # Filter processes efficiently
            matching_processes.extend([
                proc for proc in eidb
                if (activity_set is None or proc['name'] in activity_set) and
                   (reference_product_set is None or proc['reference product'] in reference_product_set) and
                   (location_set is None or proc['location'] in location_set)
            ])

    if not matching_processes:
        print("No activities match the given specifications or the input format is incorrect.")
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