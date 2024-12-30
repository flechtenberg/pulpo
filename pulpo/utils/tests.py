import bw2data as bd
import bw2calc as bc
import copy
from packaging import version

def is_bw25():
    """Check if the installed Brightway packages adhere to bw25 versions."""
    # Define version thresholds
    THRESHOLDS = {
        "bw2calc": "2.0.dev5",
        "bw2data": "4.0.dev11",
    }

    try:
        for pkg, threshold in {"bw2calc": bc, "bw2data": bd}.items():
            pkg_version = ".".join(map(str, threshold.__version__)) if isinstance(threshold.__version__,
                                                                                  tuple) else str(
                threshold.__version__)
            if version.parse(pkg_version) < version.parse(THRESHOLDS[pkg]):
                return False
        return True
    except Exception as e:
        raise RuntimeError(f"Error checking Brightway versions: {e}")

import pulpo as pulpo
from pulpo.utils.bw_parser import import_data, retrieve_methods, retrieve_env_interventions, retrieve_processes

def setup_test_db():
    # Set the current project to "sample_project"
    bd.projects.set_current("sample_project")

    # Keys
    # Biosphere
    co2_key = ('biosphere', 'CO2')
    ch4_key = ('biosphere', 'CH4')
    pm_key = ('biosphere', 'PM')
    h2o_irrigation_key = ('biosphere', 'H2O_irrigation')
    # Technosphere
    wind_turbine_key = ('technosphere', 'wind turbine')
    steam_cycle_key = ('technosphere', 'steam cycle')
    lignite_extraction_key = ('technosphere', 'lignite extraction')
    oil_extraction_key = ('technosphere', 'oil extraction')
    e_car_key = ('technosphere', 'e-Car')

    # Define biosphere flows and create the "biosphere" database if it doesn't exist
    if "biosphere" not in bd.databases:
        biosphere_db = bd.Database("biosphere")
        biosphere_data = {
            ("biosphere", "CO2"): {
                "name": "Carbon dioxide, fossil",
                "categories": ("climate change", "GWP 100a"),
                "type": "emission",
                "unit": "kg",
            },
            ("biosphere", "CH4"): {
                "name": "Methane, agricultural",
                "categories": ("climate change", "GWP 100a"),
                "type": "emission",
                "unit": "kg",
            },
            ("biosphere", "PM"): {
                "name": "Particulate matter, industrial",
                "categories": ("air quality", "particulate matter"),
                "type": "emission",
                "unit": "g",
            },
            ("biosphere", "H2O_irrigation"): {
                "name": "Water, irrigation",
                "categories": ("water use", "irrigation"),
                "type": "resource",
                "unit": "m3",
            },
        }
        biosphere_db.write(biosphere_data)

    # Create the "technosphere" database if it doesn't exist
    if "technosphere" not in bd.databases:
        technosphere_db = bd.Database("technosphere")
        technosphere_db.write({})
        process_data = [
            ("oil extraction", "kg", "GLO", "oil"),
            ("lignite extraction", "kg", "GLO", "lignite"),
            ("steam cycle", "kWh", "GLO", "electricity"),
            ("wind turbine", "kWh", "GLO", "electricity"),
            ("e-Car", "tkm", "GLO", "transport"),
        ]

        for name, unit, location, ref_product in process_data:
            act = technosphere_db.new_activity(name)
            act["unit"] = unit
            act["location"] = location
            act["name"] = name
            act["reference product"] = ref_product
            act.new_exchange(amount=1.0, input=act.key, type="production").save()
            act.save()

        exchange_data = [
            [e_car_key, oil_extraction_key, 0.03, 'technosphere'],
            [e_car_key, lignite_extraction_key, 0.03, 'technosphere'],
            [oil_extraction_key, steam_cycle_key, 0.5, 'technosphere'],
            [lignite_extraction_key, steam_cycle_key, 0.5, 'technosphere'],
            [steam_cycle_key, e_car_key, 0.5, 'technosphere'],
            [wind_turbine_key, e_car_key, 0.5, 'technosphere'],
            [e_car_key, wind_turbine_key, 0.03, 'technosphere'],
            [co2_key, oil_extraction_key, 0.2, 'biosphere'],
            [co2_key, lignite_extraction_key, 0.3, 'biosphere'],
            [co2_key, steam_cycle_key, 1, 'biosphere'],
            [co2_key, wind_turbine_key, 0.1, 'biosphere'],
            [ch4_key, oil_extraction_key, 0.01, 'biosphere'],
            [ch4_key, lignite_extraction_key, 0.02, 'biosphere'],
            [pm_key, steam_cycle_key, 2.0, 'biosphere'],
            [pm_key, wind_turbine_key, 1.5, 'biosphere'],
            [h2o_irrigation_key, steam_cycle_key, 2.0, 'biosphere'],
            [h2o_irrigation_key, wind_turbine_key, 5.0, 'biosphere'],
            [pm_key, e_car_key, 0.1, 'biosphere'],
            [h2o_irrigation_key, e_car_key, 0.1, 'biosphere'],
        ]

        for input, target, amount, type in exchange_data:
            act = [act for act in technosphere_db if act.key==target][0]
            act.new_exchange(amount=amount, input=input, type=type).save()
            act.save()

    # Loop through the list of methods and deregister each one
    # Create a copy of the methods list
    methods_copy = copy.deepcopy(bd.methods)
    # Loop through the copy of methods and deregister each one
    for method in methods_copy:
        bd.Method(method).deregister()

    # Define LCIA methods and CFs
    methods_data = [
        ("climate change", "kg CO2eq", 2, "cc", "climate change CFs", "climate_change", "CO2", [(co2_key, 1), (ch4_key, 29.7)]),
        ("air quality", "ppm", 1, "aq", "air quality CFs", "air_quality", "PM", [(ch4_key, 29.7)]),
        ("resources", "m3", 1, "rc", "resource CFs", "resources", "H2O_irrigation", [(h2o_irrigation_key,1)]),
    ]

    for method_name, unit, num_cfs, abbreviation, description, filename, flow_code, flow_list in methods_data:
        method = bd.Method(("my project", method_name))
        method.register(**{
            "unit": unit,
            "num_cfs": num_cfs,
            "abbreviation": abbreviation,
            "description": description,
            "filename": filename,
        })
        method.write(flow_list)

def sample_lcia():
    # Load the technosphere database
    bd.projects.set_current('sample_project')
    technosphere_db = bd.Database("technosphere")
    # Perform LCA with FU 1 for all activities
    act = {act.key: 1 for act in technosphere_db}
    # Perform Multi-LCA
    bd.calculation_setups['multiLCA'] = {'inv': [act], 'ia': list(bd.methods)}
    myMultiLCA = bd.MultiLCA('multiLCA')
    results = [round(x, 5) for x in myMultiLCA.results[0]]
    return results

def test_database_import():
    result = sample_lcia()
    assert result == [4.22308, 1.5937, 11.1567]  # Example assertion

def test_import_data():
    # Test if the import works and has the expected structure
    methods = {
        "('my project', 'climate change')": 1,
        "('my project', 'air quality')": 1,
        "('my project', 'resources')": 1,
    }

    result = import_data('sample_project', 'technosphere', methods, 'biosphere')

    assert [idx for idx in result] == ['matrices', 'intervention_matrix', 'technology_matrix', 'process_map', 'intervention_map']
    assert result['technology_matrix'].shape == (5, 5)
    assert result['intervention_matrix'].shape == (4, 5)
    assert result['process_map'] == {('technosphere', 'oil extraction'): 0, ('technosphere', 'lignite extraction'): 1, ('technosphere', 'steam cycle'): 2, ('technosphere', 'wind turbine'): 3, ('technosphere', 'e-Car'): 4, 2: 'steam cycle | electricity | GLO', 1: 'lignite extraction | lignite | GLO', 0: 'oil extraction | oil | GLO', 3: 'wind turbine | electricity | GLO', 4: 'e-Car | transport | GLO'}

def test_retrieve_activities():
    key = retrieve_processes('sample_project', 'technosphere', keys=["('technosphere', 'wind turbine')"])
    name = retrieve_processes('sample_project', 'technosphere', activities=['e-Car'])
    location = retrieve_processes('sample_project', 'technosphere', locations=['GLO'])
    assert key[0]['name'] == "wind turbine"
    assert name[0]['name'] == "e-Car"
    assert len(location) == 5

def test_retrieve_methods():
    single_result = retrieve_methods('sample_project', ['climate'])
    multi_result = retrieve_methods('sample_project', ['project'])
    assert single_result == [('my project', 'climate change')]
    assert multi_result == [('my project', 'climate change'), ('my project', 'air quality'), ('my project', 'resources')]

def test_retrieve_envflows():
    result = retrieve_env_interventions('sample_project', intervention_matrix='biosphere', keys="('biosphere', 'PM')")
    assert result[0]['name'] == 'Particulate matter, industrial'

def test_pulpo():
    project = 'sample_project'
    database = 'technosphere'
    methods = {"('my project', 'climate change')": 1,
               "('my project', 'air quality')": 1,
               "('my project', 'resources')": 0}

    worker = pulpo.PulpoOptimizer(project, database, methods, '')
    worker.intervention_matrix = 'biosphere'
    worker.get_lci_data()
    eCar = worker.retrieve_activities(reference_products='transport')
    demand = {eCar[0]: 1}
    elec = worker.retrieve_activities(reference_products='electricity')
    choices = {'electricity': {elec[0]: 100, elec[1]: 100}}
    worker.instantiate(choices=choices, demand=demand)
    worker.solve()
    result_obj = round(worker.instance.OBJ(), 6)
    result_aux = round(worker.instance.impacts_calculated["('my project', 'resources')"].value, 5)
    assert result_obj == 0.103093
    assert result_aux == 5.25773

    upper_limit = {eCar[0]: 1}
    lower_limit = {eCar[0]: 1}
    worker.instantiate(choices=choices, upper_limit=upper_limit, lower_limit=lower_limit)
    worker.solve()
    result_obj = round(worker.instance.OBJ(), 6)
    result_aux = round(worker.instance.impacts_calculated["('my project', 'resources')"].value, 5)
    assert result_obj == 0.1
    assert result_aux == 5.1

def setup():
    test_import_data()
    test_database_import()
    test_retrieve_activities()
    test_retrieve_envflows()
    test_retrieve_methods()
    test_pulpo()
    print('\nAll tests passed successfully. You are good to go!')