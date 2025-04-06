import bw2data as bd
import bw2calc as bc
import copy
from pulpo.utils.utils import is_bw25

from stats_arrays import NormalUncertainty
import numpy as np

def setup_test_db():
    # Set the current project to "sample_project"
    bd.projects.set_current("sample_project_bw25" if is_bw25() else "sample_project")

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
        
        # Add naive uncertainty to all exchanges in the technosphere and biosphere database
        for act in technosphere_db:
            for exc in act.exchanges():
                    exc['uncertainty type'] = NormalUncertainty.id
                    exc['loc'] = np.log(exc['amount'])
                    exc['scale'] = 0.1 * exc['amount']
                    exc.save()

    # @TODO: Add another "foreground" database, to test the import of two dbs.

    # Loop through the list of methods and deregister each one
    # Create a copy of the methods list
    methods_copy = copy.deepcopy(bd.methods)
    # Loop through the copy of methods and deregister each one
    for method in methods_copy:
        bd.Method(method).deregister()

    # Define LCIA methods and CFs
    methods_data = [
        # Climate Change Method
        ("climate change", "kg CO2eq", 2, "cc", "climate change CFs", "climate_change", "CO2", [
            (co2_key, {'uncertainty type': 3, 'loc': np.log(1), 'scale': 0.1, 'shape': np.nan, 'minimum': np.nan, 'maximum': np.nan, 'negative': False, 'amount': 1}),
            (ch4_key, {'uncertainty type': 3, 'loc': np.log(29.7), 'scale': 0.2, 'shape': np.nan, 'minimum': np.nan, 'maximum': np.nan, 'negative': False, 'amount': 29.7}),
        ]),
    
        # Air Quality Method
        ("air quality", "ppm", 1, "aq", "air quality CFs", "air_quality", "PM", [
            (('biosphere', 'CH4'), {'uncertainty type': 3, 'loc': np.log(29.7), 'scale': 0.2, 'shape': np.nan, 'minimum': np.nan, 'maximum': np.nan, 'negative': False, 'amount': 29.7}),
        ]),
    
        # Resources Method
        ("resources", "m3", 1, "rc", "resource CFs", "resources", "H2O_irrigation", [
            (h2o_irrigation_key, {'uncertainty type': 3, 'loc': np.log(1), 'scale': 0.1, 'shape': np.nan, 'minimum': np.nan, 'maximum': np.nan, 'negative': False, 'amount': 1}),
        ]),
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
    bd.projects.set_current("sample_project_bw25" if is_bw25() else "sample_project")
    technosphere_db = bd.Database("technosphere")

    if is_bw25():
        functional_units = {"act": {act.id: 1 for act in technosphere_db}}
        config = {"impact_categories": list(bd.methods)}
        data_objs = bd.get_multilca_data_objs(functional_units=functional_units, method_config=config)
        myMultiLCA = bc.MultiLCA(demands=functional_units, method_config=config, data_objs=data_objs)
        myMultiLCA.lci()
        myMultiLCA.lcia()
        results = [round(myMultiLCA.scores[x], 5) for x in myMultiLCA.scores]
    else:
        act = {act.key: 1 for act in technosphere_db}
        bd.calculation_setups['multiLCA'] = {'inv': [act], 'ia': list(bd.methods)}
        myMultiLCA = bc.MultiLCA('multiLCA')
        results = [round(x, 5) for x in myMultiLCA.results[0]]
    return results

def main():
    setup_test_db()
    print(sample_lcia())

if __name__ == "__main__":
    main()


