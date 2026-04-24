import bw2data as bd
import bw2calc as bc
import copy
from pulpo.utils.utils import is_bw25
from stats_arrays import NormalUncertainty
import numpy as np


# ---------------------------------------------------------------------------
# BIOSPHERE DATABASE
# ---------------------------------------------------------------------------

def setup_biosphere_db():
    """Create a minimal biosphere database with four flows."""
    bd.projects.set_current("sample_project_bw25" if is_bw25() else "sample_project")

    db_name = "biosphere3"

    # Create fresh biosphere3 (same as before)
    if db_name in bd.databases:
        del bd.databases[db_name]

    biosphere_db = bd.Database(db_name)

    biosphere_data = {
        ("biosphere3", "CO2"): {
            "name": "Carbon dioxide, fossil",
            "categories": ("climate change", "GWP 100a"),
            "type": "emission",
            "unit": "kg",
        },
        ("biosphere3", "CH4"): {
            "name": "Methane, agricultural",
            "categories": ("climate change", "GWP 100a"),
            "type": "emission",
            "unit": "kg",
        },
        ("biosphere3", "PM"): {
            "name": "Particulate matter, industrial",
            "categories": ("air quality", "particulate matter"),
            "type": "emission",
            "unit": "g",
        },
        ("biosphere3", "H2O_irrigation"): {
            "name": "Water, irrigation",
            "categories": ("water use", "irrigation"),
            "type": "resource",
            "unit": "m3",
        },
    }

    biosphere_db.write(biosphere_data)
    print(f"{db_name} created with {len(biosphere_data)} flows.")


# ---------------------------------------------------------------------------
# TECHNOSPHERE DATABASE
# ---------------------------------------------------------------------------

def setup_test_db():
    """Create a minimal technosphere database and connect to biosphere3."""
    bd.projects.set_current("sample_project_bw25" if is_bw25() else "sample_project")

    # Ensure biosphere exists
    if "biosphere3" not in bd.databases:
        setup_biosphere_db()

    # Keys
    co2_key = ('biosphere3', 'CO2')
    ch4_key = ('biosphere3', 'CH4')
    pm_key = ('biosphere3', 'PM')
    h2o_irrigation_key = ('biosphere3', 'H2O_irrigation')
    wind_turbine_key = ('technosphere', 'wind turbine')
    steam_cycle_key = ('technosphere', 'steam cycle')
    lignite_extraction_key = ('technosphere', 'lignite extraction')
    oil_extraction_key = ('technosphere', 'oil extraction')
    e_car_key = ('technosphere', 'e-Car')

    # Create technosphere database
    if "technosphere" in bd.databases:
        del bd.databases["technosphere"]

    technosphere_db = bd.Database("technosphere")
    technosphere_db.write({})

    # Define activities
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

    # Exchanges (identical numeric structure)
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

    for input_, target, amount, ex_type in exchange_data:
        act = [a for a in technosphere_db if a.key == target][0]
        act.new_exchange(amount=amount, input=input_, type=ex_type).save()
        act.save()

    # Add uncertainties (unchanged logic)
    for act in technosphere_db:
        for exc in act.exchanges():
            if str(exc["input"]) != str(exc["output"]):
                exc["uncertainty type"] = NormalUncertainty.id
                exc["loc"] = exc["amount"]
                exc["scale"] = 0.1 * exc["amount"]
                exc.save()

    print(f"technosphere database created with {len(technosphere_db)} activities.")


# ---------------------------------------------------------------------------
# BACKGROUND DATABASE
# ---------------------------------------------------------------------------

def setup_background_db():
    """Create background_db with natural gas, electricity, water, SMR hydrogen, and ASU oxygen.
    Adds naive uncertainty to all non-production exchanges."""
    bd.projects.set_current("sample_project_bw25" if is_bw25() else "sample_project")

    # Ensure biosphere is available
    if "biosphere3" not in bd.databases:
        setup_biosphere_db()

    co2_key = ('biosphere3', 'CO2')
    ch4_key = ('biosphere3', 'CH4')
    pm_key = ('biosphere3', 'PM')
    h2o_key = ('biosphere3', 'H2O_irrigation')

    db_name = "background_db"

    if db_name in bd.databases:
        del bd.databases[db_name]
    db = bd.Database(db_name)

    data = {
        # 1. Natural gas extraction
        (db_name, "natural gas extraction"): {
            "name": "natural gas extraction",
            "unit": "kg",
            "location": "GLO",
            "reference product": "natural gas",
            "exchanges": [
                {"input": (db_name, "natural gas extraction"), "amount": 1.0, "type": "production"},
                {"input": ch4_key, "amount": 0.0074, "type": "biosphere"},
            ],
        },

        # 2. Natural gas electricity (NGCC)
        (db_name, "natural gas electricity"): {
            "name": "natural gas electricity",
            "unit": "kWh",
            "location": "GLO",
            "reference product": "electricity",
            "exchanges": [
                {"input": (db_name, "natural gas electricity"), "amount": 1.0, "type": "production"},
                {"input": (db_name, "natural gas extraction"), "amount": 0.19, "type": "technosphere"},
                {"input": co2_key, "amount": 0.44, "type": "biosphere"},
                {"input": ch4_key, "amount": 0.0000028, "type": "biosphere"},
                {"input": pm_key, "amount": 0.015, "type": "biosphere"},
                {"input": h2o_key, "amount": 0.0018, "type": "biosphere"},
            ],
        },

        # 3. Wind electricity
        (db_name, "wind electricity"): {
            "name": "wind electricity",
            "unit": "kWh",
            "location": "GLO",
            "reference product": "electricity",
            "exchanges": [
                {"input": (db_name, "wind electricity"), "amount": 1.0, "type": "production"},
                {"input": co2_key, "amount": 0.012, "type": "biosphere"},
            ],
        },

        # 4. Water supply
        (db_name, "water supply"): {
            "name": "water supply",
            "unit": "m3",
            "location": "GLO",
            "reference product": "water",
            "exchanges": [
                {"input": (db_name, "water supply"), "amount": 1.0, "type": "production"},
                {"input": (db_name, "natural gas electricity"), "amount": 0.49, "type": "technosphere"},
                {"input": h2o_key, "amount": 1.0, "type": "biosphere"},
            ],
        },

        # 5. Hydrogen SMR
        (db_name, "hydrogen SMR"): {
            "name": "hydrogen SMR",
            "unit": "kg",
            "location": "GLO",
            "reference product": "hydrogen",
            "exchanges": [
                {"input": (db_name, "hydrogen SMR"), "amount": 1.0, "type": "production"},
                {"input": (db_name, "natural gas extraction"), "amount": 3.2, "type": "technosphere"},
                {"input": (db_name, "natural gas electricity"), "amount": 0.1, "type": "technosphere"},
                {"input": (db_name, "water supply"), "amount": 0.003, "type": "technosphere"},
                {"input": co2_key, "amount": 9.0, "type": "biosphere"},
            ],
        },

        # 6. O2 ASU
        (db_name, "O2 ASU"): {
            "name": "O2 ASU",
            "unit": "kg",
            "location": "GLO",
            "reference product": "oxygen",
            "exchanges": [
                {"input": (db_name, "O2 ASU"), "amount": 1.0, "type": "production"},
                {"input": (db_name, "natural gas electricity"), "amount": 0.22, "type": "technosphere"},
            ],
        },
    }

    db.write(data)

    # --- Add uncertainties to all non-production exchanges ---
    for act in db:
        for exc in act.exchanges():
            if exc["type"] != "production":  # skip reference flows
                exc["uncertainty type"] = NormalUncertainty.id
                exc["loc"] = exc["amount"]
                exc["scale"] = abs(0.1 * exc["amount"])  # 10% relative stddev
                exc.save()

    print(f"{db_name} created with {len(data)} activities and uncertainty added to exchanges.")

# ---------------------------------------------------------------------------
# FOREGROUND DATABASE
# ---------------------------------------------------------------------------

def setup_foreground_db():
    """Create foreground_db with electrolysis, O2-byproduct, O2-market, DAC, methanol synthesis, and ozone production."""
    bd.projects.set_current("sample_project_bw25" if is_bw25() else "sample_project")

    # Ensure background and biosphere exist
    if "biosphere3" not in bd.databases:
        setup_biosphere_db()
    if "background_db" not in bd.databases:
        setup_background_db()

    co2_key = ('biosphere3', 'CO2')
    pm_key = ('biosphere3', 'PM')

    bg = "background_db"
    db_name = "foreground_db"

    if db_name in bd.databases:
        del bd.databases[db_name]
    db = bd.Database(db_name)

    data = {
        # 1. Hydrogen electrolysis (main product: H2, byproduct handled separately)
        (db_name, "hydrogen electrolysis"): {
            "name": "hydrogen electrolysis",
            "unit": "kg",
            "location": "GLO",
            "reference product": "hydrogen",
            "exchanges": [
                {"input": (db_name, "hydrogen electrolysis"), "amount": 1.0, "type": "production"},
                {"input": (bg, "wind electricity"), "amount": 50.0, "type": "technosphere"},  # 50 kWh/kg H2
                {"input": (bg, "water supply"), "amount": 0.009, "type": "technosphere"},
                {"input": (db_name, "O2-byproduct"), "amount": -8.0, "type": "technosphere"},  # 8 kg O2/kg H2
                {"input": co2_key, "amount": 0.1, "type": "biosphere"},  # small indirect emissions
            ],
        },

        # 2. O2-byproduct (acts as output carrier for O2)
        (db_name, "O2-byproduct"): {
            "name": "O2-byproduct",
            "unit": "kg",
            "location": "GLO",
            "reference product": "oxygen",
            "exchanges": [
                {"input": (db_name, "O2-byproduct"), "amount": -1.0, "type": "production"},
            ],
        },

        # 3. O2-market (aggregates O2 supply from ASU and byproduct)
        (db_name, "O2-market"): {
            "name": "O2-market",
            "unit": "kg",
            "location": "GLO",
            "reference product": "oxygen",
            "exchanges": [
                {"input": (db_name, "O2-market"), "amount": 1.0, "type": "production"},
                {"input": (db_name, "O2-byproduct"), "amount": 1.0, "type": "technosphere"},
            ],
        },

        # 4. direct air capture
        (db_name, "direct air capture"): {
            "name": "direct air capture",
            "unit": "kg",
            "location": "GLO",
            "reference product": "CO2, captured",
            "exchanges": [
                {"input": (db_name, "direct air capture"), "amount": 1.0, "type": "production"},
                {"input": (bg, "wind electricity"), "amount": 1.5, "type": "technosphere"},
                {"input": co2_key, "amount": 1.0, "type": "biosphere"},  # CO2 uptake
            ],
        },

        # 5. Methanol synthesis
        (db_name, "methanol synthesis"): {
            "name": "methanol synthesis",
            "unit": "kg",
            "location": "GLO",
            "reference product": "methanol",
            "exchanges": [
                {"input": (db_name, "methanol synthesis"), "amount": 1.0, "type": "production"},
                {"input": (db_name, "direct air capture"), "amount": 1.375, "type": "technosphere"},  # CO2 feed (~0.375 C loss)
                {"input": (db_name, "hydrogen electrolysis"), "amount": 0.19, "type": "technosphere"},  # ~0.19 kg H2/kg MeOH
                {"input": (bg, "natural gas electricity"), "amount": 1.2, "type": "technosphere"},
                {"input": co2_key, "amount": 0.2, "type": "biosphere"},  # vented CO2
                {"input": pm_key, "amount": 0.001, "type": "biosphere"},
            ],
        },

        # 6. ozone production (uses O2 from market)
        (db_name, "ozone production"): {
            "name": "ozone production",
            "unit": "kg",
            "location": "GLO",
            "reference product": "ozone",
            "exchanges": [
                {"input": (db_name, "ozone production"), "amount": 1.0, "type": "production"},
                {"input": (bg, "O2 ASU"), "amount": 1.0, "type": "technosphere"},
                {"input": (bg, "wind electricity"), "amount": 0.5, "type": "technosphere"},
                {"input": pm_key, "amount": 0.002, "type": "biosphere"},
            ],
        },
    }

    db.write(data)

    # Add uncertainties (same pattern)
    for act in db:
        for exc in act.exchanges():
            if exc["type"] != "production":
                exc["uncertainty type"] = NormalUncertainty.id
                exc["loc"] = exc["amount"]
                exc["scale"] = abs(0.1 * exc["amount"])
                exc.save()


    print(f"{db_name} created with {len(data)} activities and uncertainty added to exchanges.")

# ---------------------------------------------------------------------------
# LCIA METHODS 
# ---------------------------------------------------------------------------

def setup_lcia_methods():
    """Create LCIA methods (identical numeric values)."""
    bd.projects.set_current("sample_project_bw25" if is_bw25() else "sample_project")

    co2_key = ('biosphere3', 'CO2')
    ch4_key = ('biosphere3', 'CH4')
    pm_key = ('biosphere3', 'PM')
    h2o_irrigation_key = ('biosphere3', 'H2O_irrigation')

    # Deregister existing methods
    for method in list(bd.methods):
        bd.Method(method).deregister()

    methods_data = [
        # Climate Change Method
        ("climate change", "kg CO2eq", 2, "cc", "climate change CFs", "climate_change", [
            (co2_key, {'uncertainty type': 3, 'loc': 1, 'scale': 0.1, 'shape': np.nan, 'minimum': np.nan,
                       'maximum': np.nan, 'negative': False, 'amount': 1}),
            (ch4_key, {'uncertainty type': 3, 'loc': 29.7, 'scale': 0.2, 'shape': np.nan, 'minimum': np.nan,
                       'maximum': np.nan, 'negative': False, 'amount': 29.7}),
        ]),
        # Air Quality Method
        ("air quality", "ppm", 1, "aq", "air quality CFs", "air_quality", [
            (('biosphere3', 'CH4'), {'uncertainty type': 3, 'loc': 29.7, 'scale': 0.2, 'shape': np.nan,
                                     'minimum': np.nan, 'maximum': np.nan, 'negative': False, 'amount': 29.7}),
        ]),
        # Resources Method
        ("resources", "m3", 1, "rc", "resource CFs", "resources", [
            (h2o_irrigation_key, {'uncertainty type': 3, 'loc': 1, 'scale': 0.1, 'shape': np.nan, 'minimum': np.nan,
                                  'maximum': np.nan, 'negative': False, 'amount': 1}),
        ]),
    ]

    for name, unit, num_cfs, abbr, desc, filename, flow_list in methods_data:
        method = bd.Method(("my project", name))
        method.register(unit=unit, num_cfs=num_cfs, abbreviation=abbr, description=desc, filename=filename)
        method.write(flow_list)

    print(f"Registered {len(methods_data)} LCIA methods.")


# ---------------------------------------------------------------------------
# SETUP ALL
# ---------------------------------------------------------------------------

def setup_sample_db():
    setup_biosphere_db()
    setup_test_db()
    setup_background_db()
    setup_foreground_db()
    setup_lcia_methods()

# ---------------------------------------------------------------------------
# SAMPLE LCA
# ---------------------------------------------------------------------------

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
        bd.calculation_setups["multiLCA"] = {"inv": [act], "ia": list(bd.methods)}
        myMultiLCA = bc.MultiLCA("multiLCA")
        results = [round(x, 5) for x in myMultiLCA.results[0]]
    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    setup_biosphere_db()
    setup_test_db()
    setup_background_db()
    setup_lcia_methods()
    print(sample_lcia())


if __name__ == "__main__":
    main()
