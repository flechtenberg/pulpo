import ast
from typing import List, Union, Dict, Any, TypedDict
import warnings
import bw2calc as bc
import bw2data as bd
from pulpo.utils.utils import get_bw_version, build_bw25_params
from stats_arrays.random import MCRandomNumberGenerator
import numpy as np

class LCIDataDict(TypedDict):
    matrices: Dict[str, np.ndarray]
    intervention_matrix: np.ndarray
    technology_matrix: np.ndarray
    process_map: Dict[Any, Any]
    intervention_params: Any
    characterization_params: Dict[str, Any]
    intervention_map: Dict[Any, Any]
    intervention_map_metadata: Dict[Any, str]
    process_map_metadata: Dict[Any, str]

def set_project(project: str):
    # Set project and check if it exists
    if project not in bd.projects:
        raise ValueError(f"Project '{project}' does not exist. Please check the project name.")
    _ensure_project_current(project)


def _ensure_project_current(project: str):
    """Activate ``project`` only if it is not already the active bw2data project.

    ``bd.projects.set_current()`` always tears down and rebuilds bw2data's sqlite3
    connections (including a full ``gc.collect()``) even when re-selecting the active
    project. Skipping that redundant reset matters because this runs once per
    ``import_data`` call, i.e. once per Monte Carlo iteration.
    """
    if bd.projects.current != project:
        bd.projects.set_current(project)

def import_data(project: str, databases: Union[str, List[str]], method: Union[str, List[str], Dict[str, int]],
                intervention_matrix_name: str, seed: Union[None, int] = None, resample: Union[str, List[str]] = ("A", "B", "Q"),
                compute_uncertainty_params: bool = True) -> LCIDataDict:
    """
    Main function to import LCI data for a project from one or more databases.

    Args:
        project (str): Name of the project.
        databases (Union[str, List[str]]): One or more databases (e.g. foreground and
            background); order does not matter.
        method (Union[str, List[str], Dict[str, int]]): Method(s) for data retrieval.
        intervention_matrix_name (str): Name of the intervention matrix.
        seed (Union[None, int], optional): Seed for RNG. If None, the default A, B, and Q matrices are used.
        compute_uncertainty_params (bool, optional): Whether to assemble the bw25
            'intervention_params' / 'characterization_params' structured arrays. These are
            only needed by the uncertainty sub-package (chance-constraints, GSA), not by
            A/B/Q resampling, so skipping them speeds up repeated Monte Carlo calls.
            Default True to preserve the full LCIDataDict.

    Returns:
        Dict[str, Union[dict, Any]]: Dictionary containing imported LCI data.
    """
    set_project(project)

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
    for database in databases:
        eidbs.append(bd.Database(database))

    bw_version = get_bw_version()
    dist = seed is not None

    if isinstance(resample, str):
        resample = [resample.upper()]
    resample = [r.upper() for r in resample]

    match bw_version:
        case 'bw25':
            lca, characterization_matrices, characterization_params, process_map, bio_params = \
                _load_lci_bw25(eidbs, methods, seed, dist, resample, compute_uncertainty_params)
        case 'bw2':
            lca, characterization_matrices, characterization_params, process_map, bio_params = \
                _load_lci_bw2(eidbs, methods, seed, dist, resample)

    # final A & B matrices
    technology_matrix   = lca.technosphere_matrix
    intervention_matrix = lca.biosphere_matrix


    # Add descriptive strings to the process map for both primary and secondary databases
    process_map_metadata = {}
    for eidb in eidbs:
        for act in eidb:
            process_map_metadata[process_map[act.key]] = (
                f"{act['name']} | {act.get('reference product', '')} | {act.get('location', '')}"
            )

    # ATTN: could probbly ask BW what the biosphere matrix is and then move this code into the cases further up
    if intervention_matrix_name in bd.databases:
        eidb_bio = bd.Database(intervention_matrix_name)
        match bw_version:
            case 'bw25':
                intervention_map = {act.key: lca.dicts.biosphere[act.id] for act in eidb_bio if act.id in lca.dicts.biosphere}  # ATTN: This is adherring to old ways of storring data with keys ... how to work with IDs instead?
            case 'bw2':
                intervention_map = lca.biosphere_dict
        intervention_map_metadata = {}
        for act in eidb_bio:
            if act.key in intervention_map:
                intervention_map_metadata[intervention_map[act.key]] = act['name'] + ' | ' + str(act['categories'])
    else:
        raise Exception(
            "The name of the biosphere is not '" + intervention_matrix_name + "'. Please specify the correct biosphere.")

    lci_data:LCIDataDict = {
        'matrices': characterization_matrices,
        'intervention_matrix': intervention_matrix,
        'technology_matrix': technology_matrix,
        'process_map': process_map,
        'intervention_params': bio_params,
        'characterization_params': characterization_params,
        'intervention_map': intervention_map,
        'intervention_map_metadata':intervention_map_metadata,
        'process_map_metadata':process_map_metadata,
    }

    return lci_data


def _load_lci_bw25(eidbs, methods, seed, dist, resample, compute_uncertainty_params):
    """Build the (optionally resampled) A/B/Q matrices and maps for a bw25 project.

    One LCA with a combined functional unit (one activity per listed database) loads
    the union of all databases and their dependencies into a single index space, so
    database order does not matter.

    Returns ``(lca, characterization_matrices, characterization_params, process_map,
    intervention_params)``; ``intervention_params`` is ``None`` when not requested
    or unavailable.
    """
    characterization_matrices = {}
    characterization_params = {}

    # Build technosphere/biosphere matrices ONCE for all databases (heavy step)
    demand = {eidb.random(): 1 for eidb in eidbs}
    fu, data_objs, _ = bd.prepare_lca_inputs(demand, method=methods[0])
    lca = bc.LCA(demand=fu, data_objs=data_objs, use_distributions=dist, seed_override=seed)
    lca.load_lci_data()

    for method in methods:
        lca.switch_method(method)  # cheap: swaps only the characterization datapackage/matrix
        m = str(method)

        if compute_uncertainty_params:
            cf_params, cf_incomplete = build_bw25_params(
                lca.packages, 'characterization_matrix', lca.dicts.biosphere
            )
            if cf_params is None:
                characterization_params[m] = None
                warnings.warn(
                    f"No{' complete' if cf_incomplete else ''} characterization factor "
                    f"uncertainty information found for method '{m}'. "
                    f"Storing 'characterization_params' as None.",
                    UserWarning, stacklevel=2,
                )
            else:
                characterization_params[m] = cf_params

        if dist and "Q" in resample:
            next(lca.characterization_mm)
            lca.characterization_matrix = lca.characterization_mm.matrix
        characterization_matrices[m] = lca.characterization_matrix

    # Method-independent biosphere uncertainty params: data_objs spans all listed
    # databases (and is unaffected by switch_method), so one call covers everything.
    if compute_uncertainty_params:
        intervention_params, _ = build_bw25_params(
            data_objs, 'biosphere_matrix', lca.dicts.biosphere, lca.dicts.product
        )
        if intervention_params is None:
            warnings.warn(
                "No complete intervention flow uncertainty information found for the "
                "provided databases. Storing 'intervention_params' as None.",
                UserWarning, stacklevel=2,
            )
    else:
        intervention_params = None

    process_map = {act.key: lca.dicts.product[act.id] for eidb in eidbs for act in eidb}
    if dist and "A" in resample:
        next(lca.technosphere_mm)
        lca.technosphere_matrix = lca.technosphere_mm.matrix
    if dist and "B" in resample:
        next(lca.biosphere_mm)
        lca.biosphere_matrix = lca.biosphere_mm.matrix

    return lca, characterization_matrices, characterization_params, process_map, intervention_params


def _load_lci_bw2(eidbs, methods, seed, dist, resample):
    """Build the (optionally resampled) A/B/Q matrices and maps for a legacy bw2 project.

    One LCA with a combined functional unit (one activity per listed database) loads
    the union of all databases and their dependencies into a single index space, so
    database order does not matter.

    A, B and each method's Q are resampled from independent child seeds spawned from
    ``seed``: a shared seed would give every MCRandomNumberGenerator the identical
    percentile stream and inject spurious cross-matrix correlation (bw25 likewise
    draws each matrix independently).

    Returns ``(lca, characterization_matrices, characterization_params, process_map,
    intervention_params)``.
    """
    if dist:
        child_seeds = np.random.SeedSequence(seed).spawn(2 + len(methods))
        tech_seed = int(child_seeds[0].generate_state(1)[0])
        bio_seed = int(child_seeds[1].generate_state(1)[0])
        method_cf_seeds = {
            str(mth): int(child_seeds[2 + i].generate_state(1)[0])
            for i, mth in enumerate(methods)
        }

    characterization_matrices = {}
    characterization_params = {}

    # Build technosphere/biosphere matrices ONCE for all databases (heavy step)
    demand = {eidb.random(): 1 for eidb in eidbs}
    lca = bc.LCA(demand, methods[0])
    lca.load_lci_data()

    for method in methods:
        lca.switch_method(method)  # cheap: swaps only the characterization factors
        m = str(method)
        characterization_params[m] = lca.cf_params
        if dist and "Q" in resample:
            rng = MCRandomNumberGenerator(lca.cf_params, seed=method_cf_seeds[m])
            lca.rebuild_characterization_matrix(rng.next())
        characterization_matrices[m] = lca.characterization_matrix

    process_map = dict(lca.product_dict)
    tech_params, bio_params = lca.tech_params, lca.bio_params

    if dist and "A" in resample:
        lca.rebuild_technosphere_matrix(MCRandomNumberGenerator(tech_params, seed=tech_seed).next())
    if dist and "B" in resample:
        lca.rebuild_biosphere_matrix(MCRandomNumberGenerator(bio_params, seed=bio_seed).next())

    return lca, characterization_matrices, characterization_params, process_map, bio_params


def update_lci_data(lci_data: LCIDataDict, seed: int) -> LCIDataDict:
    """
    Update the LCI data dictionary with new data. For that, c

    Args:
        lci_data (LCIDataDict): Original LCI data dictionary.
        new_data (Dict[str, Any]): New data to be added to the LCI data dictionary.

    Returns:
        LCIDataDict: Updated LCI data dictionary.
    """

    return lci_data


def _activity_orm():
    """Return (ActivityDataset, Activity) ORM handles for the installed bw2data version.

    Both bw2 and bw25 store activities in a SQLite ``activitydataset`` table whose
    ``name``, ``product`` (reference product) and ``location`` are real columns, so
    exact-match filters can run server-side instead of loading every activity proxy.
    Returns ``(None, None)`` if the ORM is unavailable (unexpected backend/version).
    """
    try:  # bw25
        from bw2data.backends import ActivityDataset, Activity
        return ActivityDataset, Activity
    except ImportError:
        pass
    try:  # bw2 (peewee backend)
        from bw2data.backends.peewee.schema import ActivityDataset
        from bw2data.backends.peewee.proxies import Activity
        return ActivityDataset, Activity
    except ImportError:
        return None, None


# Backends whose activities live in the SQLite activitydataset table and can be
# queried server-side. Anything else falls back to iterating the database.
_SQL_BACKENDS = {'sqlite', 'iotable'}


def _query_processes_sql(ActivityDataset, Activity, db_name, keys, activities,
                         reference_products, locations):
    """Fetch matching activities of one database with a single SQL query.

    Only the matching rows are materialized into ``Activity`` proxies, instead of
    unpickling the whole database as the iteration fallback does.
    """
    query = ActivityDataset.select().where(ActivityDataset.database == db_name)
    if keys is not None:
        codes = [code for key_db, code in keys if key_db == db_name]
        if not codes:
            return []
        query = query.where(ActivityDataset.code.in_(codes))
    else:
        if activities is not None:
            query = query.where(ActivityDataset.name.in_(activities))
        if reference_products is not None:
            query = query.where(ActivityDataset.product.in_(reference_products))
        if locations is not None:
            query = query.where(ActivityDataset.location.in_(locations))
    return [Activity(document) for document in query]


def _iterate_processes(db_name, keys, activities, reference_products, locations):
    """Legacy fallback: load every activity of the database and filter in Python."""
    eidb = bd.Database(db_name)
    if keys is not None:
        keys_set = set(keys)
        return [proc for proc in eidb if proc.key in keys_set]
    activity_set = set(activities) if activities is not None else None
    reference_product_set = set(reference_products) if reference_products is not None else None
    location_set = set(locations) if locations is not None else None
    return [
        proc for proc in eidb
        if (activity_set is None or proc['name'] in activity_set) and
           (reference_product_set is None or proc.get('reference product') in reference_product_set) and
           (location_set is None or proc['location'] in location_set)
    ]


def retrieve_processes(project: str, databases: Union[str, List[str]], keys=None, activities=None,
                       reference_products=None, locations=None):
    """
    Retrieve activities from one or more databases based on specified keys, activities, reference products, and locations.

    Filters are matched exactly and combined with AND; ``keys`` takes precedence over
    the other filters. Uses a server-side SQL query where possible (SQLite-backed
    databases) and falls back to iterating the database otherwise.

    Args:
        project (str): Name of the project.
        databases (Union[str, List[str]]): Name of the primary database or a list of databases (foreground, background).
        keys (list, optional): List of keys to filter activities, as "('db', 'code')" strings or (db, code) tuples.
        activities (list, optional): List of activity names to filter.
        reference_products (list, optional): List of reference products to filter.
        locations (list, optional): List of locations to filter.

    Returns:
        list: List of matching activities from the specified databases.
    """
    _ensure_project_current(project)

    # Normalize inputs to lists
    if isinstance(databases, str):
        databases = [databases]
    if activities is not None and not isinstance(activities, list):
        activities = [activities]
    if reference_products is not None and not isinstance(reference_products, list):
        reference_products = [reference_products]
    if locations is not None and not isinstance(locations, list):
        locations = [locations]
    if keys is not None:
        if isinstance(keys, str):
            keys = [keys]
        keys = [key if isinstance(key, tuple) else ast.literal_eval(key) for key in keys]

    ActivityDataset, Activity = _activity_orm()

    matching_processes = []
    for db_name in databases:
        use_sql = (
            ActivityDataset is not None
            and bd.databases.get(db_name, {}).get('backend', 'sqlite') in _SQL_BACKENDS
        )
        if use_sql:
            matching_processes.extend(_query_processes_sql(
                ActivityDataset, Activity, db_name, keys, activities, reference_products, locations
            ))
        else:
            matching_processes.extend(_iterate_processes(
                db_name, keys, activities, reference_products, locations
            ))

    if not matching_processes:
        print("No activities match the given specifications or the input format is incorrect.")
    return matching_processes


def retrieve_env_interventions(project: str = '', intervention_matrix: str = 'biosphere3', keys=None, activities=None,
                               categories=None):
    """
    Retrieve environmental interventions from the biosphere database based on specified keys, activities, and categories.

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
    _ensure_project_current(project)
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
    _ensure_project_current(project)
    return [method for method in bd.methods if any([x.lower() in str(method).lower() for x in sub_string])]