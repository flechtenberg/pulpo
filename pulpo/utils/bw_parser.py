from typing import List, Union, Dict, Any
from pathlib import Path
import pickle
import brightway2 as bw
from scipy import sparse

def import_data(project: str, database: str, method: Union[str, List[str]], directory: str, force=False) -> Dict[str, Union[dict, Any]]:
    """ TODO can we find a faster way to import the data without having to save it locally? """
    """
    Main function to import LCI data for a project from a database.

    :param project: The name of the Ecoinvent project.
    :param database: The name of the Ecoinvent database.
    :param method: The method(s) to evaluate. Can be a single method name or a list of method names.
    :param directory: The working directory where the matrices are saved to or read from.
    :param force: If True, re-load the LCI data regardless of the existing data in the directory.
    :return: A dictionary with the relevant LCI data.

    :rtype: dict[str, Union[dict, Any]]
    """
    bw.projects.set_current(project)
    eidb = bw.Database(database)

    if isinstance(method, str):
        method = [method]  # Convert single string to list of strings
    method = sorted(method)

    paths = get_paths(project, database, method, directory)

    if should_reload_data(paths, force):
        methods = retrieve_methods(project, method)
        random_act = eidb.random()
        myMultiLCA, lca = perform_generic_lca(random_act, methods)
        save_lci_data(directory, project, method, database, myMultiLCA, lca, eidb)

    lci_data = read_lci_data(directory, project, method, database)

    return lci_data


def get_paths(project: str, database: str, method: List[str], directory: str) -> List[str]:
    """
    Get a list of file paths for the LCI data files.
    """
    directory = directory.replace('\\', '/')

    paths = [
        f"{directory}/processed/{project}_{meth.replace(':',';').replace('/','!')}.npz" for meth in method
    ]
    paths.append(f"{directory}/processed/{project}_{database}_biosphere.npz")
    paths.append(f"{directory}/processed/{project}_{database}_technosphere.npz")
    paths.append(f"{directory}/processed/{project}_{database}_names.p")

    return paths


def should_reload_data(paths: List[str], force: bool) -> bool:
    """
    Check if the LCI data should be reloaded.
    """
    return any(not Path(path).exists() for path in paths) or force


def perform_generic_lca(random_act: bw.Database, methods: List[str]) -> tuple:
    """
    Perform a generic LCA calculation.
    """
    bw.calculation_setups['multiLCA'] = {'inv': [{random_act: 1}], 'ia': methods}
    myMultiLCA = bw.MultiLCA('multiLCA')
    lca = myMultiLCA.lca

    return myMultiLCA, lca


def save_lci_data(directory: str, project: str, method: List[str], database: str,
                  myMultiLCA: bw.MultiLCA, lca: bw.LCA, eidb: bw.Database):
    """
    Save the LCI data to files.
    """
    Path(f"{directory}/processed").mkdir(parents=True, exist_ok=True)

    for i in range(0,len(myMultiLCA.methods)):
        sparse.save_npz(f"{directory}/processed/{project}_{str(myMultiLCA.methods[i]).replace(':',';').replace('/','!')}", myMultiLCA.method_matrices[i])

    sparse.save_npz(f"{directory}/processed/{project}_{database}_biosphere", lca.biosphere_matrix)
    sparse.save_npz(f"{directory}/processed/{project}_{database}_technosphere", lca.technosphere_matrix)

    # activity_map[key] --> ID | activity_map[ID] --> description (name | reference product | location)
    activity_map = lca.product_dict
    for act in eidb:
        activity_map[activity_map[act.key]] = str(act['name']) + ' | ' + str(act['reference product']) + ' | ' + str(act['location'])
    pickle.dump(activity_map, open(f"{directory}/processed/{project}_{database}_names.p", 'wb'))


def read_lci_data(directory: str, project: str, method: List[str], database: str) -> Dict[str, Union[dict, Any]]:
    """
    Read the LCI data from files.
    """
    matrices = {meth: sparse.load_npz(f"{directory}/processed/{project}_{meth.replace(':',';').replace('/','!')}.npz") for meth in method}
    biosphere = sparse.load_npz(f"{directory}/processed/{project}_{database}_biosphere.npz")
    technosphere = sparse.load_npz(f"{directory}/processed/{project}_{database}_technosphere.npz")
    activity_map = pickle.load(open(f"{directory}/processed/{project}_{database}_names.p", 'rb'))

    lci_data = {
        'matrices': matrices,
        'biosphere': biosphere,
        'technosphere': technosphere,
        'activity_map': activity_map
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

def retrieve_envflows(project, keys=None, activities=None, categories=None):
    """
    Retrieve environmental flows from the biosphere database based on specified keys, activities, and categories.
    """

    bw.projects.set_current(project)
    eidb = bw.Database('biosphere3')

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


def retrieve_methods(project: str, sub_string: str):
    ''' This function retrieves all the methods that contain the specified list of substrings'''
    bw.projects.set_current(project)
    return [method for method in bw.methods if any([x.lower() in str(method).lower() for x in sub_string])]
