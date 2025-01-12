from typing import List, Union, Dict, Any
import bw2calc as bc
import bw2data as bd
from pulpo.utils.utils import is_bw25

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

    # Set project and check if it exists
    if project not in bd.projects:
        raise ValueError(f"Project '{project}' does not exist. Please check the project name.")

    bd.projects.set_current(project)

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
    eidb = bd.Database(databases[0])  # Foreground database
    eidb_secondary = bd.Database(databases[1]) if len(databases) > 1 else None  # Secondary background database

    rand_act = eidb.random()  # Random activity from foreground database

    bw25 = is_bw25()
    characterization_matrices = {}
    secondary_lca = None  # Placeholder for secondary LCA object

    if bw25:
        # Process with bw25 logic for foreground database
        functional_units_1 = {"act1": {rand_act.id: 1}}
        config_1 = {"impact_categories": methods}
        data_objs_1 = bd.get_multilca_data_objs(functional_units=functional_units_1, method_config=config_1)

        lca = bc.MultiLCA(demands=functional_units_1, method_config=config_1, data_objs=data_objs_1)
        lca.load_lci_data()
        lca.load_lcia_data()

        # Store the characterization matrices for methods
        characterization_matrices = {str(method): lca.characterization_matrices[method] for method in methods}

        # Perform LCA for the secondary database, if specified
        if eidb_secondary:
            functional_units_2 = {"act2": {eidb_secondary.random().id: 1}}
            config_2 = {"impact_categories": methods}
            data_objs_2 = bd.get_multilca_data_objs(functional_units=functional_units_2, method_config=config_2)

            secondary_lca = bc.MultiLCA(demands=functional_units_2, method_config=config_2, data_objs=data_objs_2)
            secondary_lca.load_lci_data()  # Only need the product dictionary
            # secondary_lca.products_dict will be available later

    else:
        # Process with bw2 logic for foreground database
        for method in methods:
            lca = bc.LCA({rand_act: 1}, method)
            lca.lci()
            lca.lcia()
            characterization_matrices[str(method)] = lca.characterization_matrix

        # Perform LCA for the secondary database, if specified
        if eidb_secondary:
            rand_act_bg = eidb_secondary.random()
            for method in methods:
                secondary_lca = bc.LCA({rand_act_bg: 1}, method)
                secondary_lca.lci()  # Load LCI to access product dictionary
                # Stop after loading one method since we're only interested in secondary_lca.products_dict
                break

    # Extract A (Technosphere) and B (Biosphere) matrices from the LCA
    technology_matrix = lca.technosphere_matrix  # A matrix
    intervention_matrix = lca.biosphere_matrix  # B matrix

    # Initialize the process map
    process_map = {}

    if bw25:
        # Create activity map for the primary database (eidb)
        process_map.update({act.key: lca.dicts.product[act.id] for act in eidb})

        # Add secondary database (secondary_eidb) activities, if available
        if eidb_secondary and secondary_lca:
            process_map.update({act.key: secondary_lca.dicts.product[act.id] for act in eidb_secondary})
    else:
        # Use product_dict for the primary database (eidb)
        process_map.update(lca.product_dict)

        # Add secondary database (secondary_eidb) product_dict, if available
        if eidb_secondary and secondary_lca:
            process_map.update(secondary_lca.product_dict)

    # Add descriptive strings to the process map for both primary and secondary databases
    for act in eidb:
        process_map[process_map[act.key]] = (
            f"{act['name']} | {act.get('reference product', '')} | {act.get('location', '')}"
        )

    if eidb_secondary:
        for act in eidb_secondary:
            process_map[process_map[act.key]] = (
                f"{act['name']} | {act.get('reference product', '')} | {act.get('location', '')}"
            )

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