from typing import List, Union, Dict, Any
from pathlib import Path
import pickle
import brightway2 as bw
from scipy import sparse

def import_data(project: str, database: str, method: Union[str, List[str], dict[str, int]], biosphere: str) -> Dict[str, Union[dict, Any]]:
    """ TODO can we find a faster way to import the data without having to save it locally? """
    """
    Main function to import LCI data for a project from a database.

    :param project: The name of the Ecoinvent project.
    :param database: The name of the Ecoinvent database.
    :param method: The method(s) to evaluate. Can be a single method name or a list of method names.
    :return: A dictionary with the relevant LCI data.

    :rtype: dict[str, Union[dict, Any]]
    """
    # Set project and get database
    bw.projects.set_current(project)
    eidb = bw.Database(database)
    # Prepare methods
    if isinstance(method, str):
        method = [method]  # Convert single string to list of strings
    method = sorted(method)
    methods = retrieve_methods(project, method)
    # Get data
    matrices = {}
    rand_act = eidb.random()
    for method in methods:
        lca = bw.LCA({rand_act: 1}, method)
        lca.load_lci_data()
        lca.load_lcia_data()
        matrices[str(method)] = lca.characterization_matrix
    technosphere = lca.technosphere_matrix
    biosphere_matrix = lca.biosphere_matrix
    # Create activity map key --> ID | ID --> description
    activity_map = lca.product_dict

    for act in eidb:
        activity_map[activity_map[act.key]] = str(act['name']) + ' | ' + str(act['reference product']) + ' | ' + str(
            act['location'])

    if biosphere in bw.databases:
        eidb_bio = bw.Database(biosphere)
        elem_map = lca.biosphere_dict
        for act in eidb_bio:
            if act.key in elem_map:
                elem_map[elem_map[act.key]] = act['name'] + ' | ' + str(act['categories'])
    else:
        print("The name of the biosphere is not '" + biosphere + "'. Please specify the correct biosphere.")
        return {}

    lci_data = {
        'matrices': matrices,
        'biosphere': biosphere_matrix,
        'technosphere': technosphere,
        'activity_map': activity_map,
        'elem_map': elem_map,
    }

    return lci_data


def retrieve_activities(project, database, keys=None, activities=None, reference_products=None, locations=None):
    """
    Retrieve activities from a database based on specified keys, activities, reference products, and locations.
    """

    bw.projects.set_current(project)
    eidb = bw.Database(database)

    if keys is not None:
        if isinstance(keys, str):
            keys = [keys]
        keys = [eval(key) for key in keys]
        return [act for act in eidb if act.key in keys]

    matching_activities = []

    for act in eidb:
        if (activities is None or act['name'] in activities) and \
                (reference_products is None or act['reference product'] in reference_products) and \
                (locations is None or act['location'] in locations):
            matching_activities.append(act)

    if not matching_activities:
        print("No activities match the given specifications or the input format is incorrect.")
    else:
        return matching_activities

def retrieve_envflows(project='', biosphere='biosphere3', keys=None, activities=None, categories=None):
    """
    Retrieve environmental flows from the biosphere database based on specified keys, activities, and categories.
    """

    bw.projects.set_current(project)
    eidb = bw.Database(biosphere)

    if keys is not None:
        if isinstance(keys, str):
            keys = [keys]
        keys = [eval(key) for key in keys]
        return [flow for flow in eidb if flow.key in keys]

    matching_flows = []

    for flow in eidb:
        if (activities is None or flow['name'] in activities) and \
                (categories is None or str(flow['categories']) in categories):
            matching_flows.append(flow)

    if not matching_flows:
        print("No flows match the given specifications or the input format is incorrect.")
    else:
        return matching_flows


def retrieve_methods(project: str, sub_string: List[str]):
    ''' This function retrieves all the methods that contain the specified list of substrings'''
    bw.projects.set_current(project)
    return [method for method in bw.methods if any([x.lower() in str(method).lower() for x in sub_string])]
