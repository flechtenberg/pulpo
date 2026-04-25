"""Minimal toy database for the time-dependent PULPO formulation.

Two-timestep electricity dispatch with a battery:

    Activities (each is its own product)
        solar              -- produces 1 kWh; no emissions
        coal               -- produces 1 kWh; 1 kg CO2/kWh
        battery_discharge  -- produces 1 kWh; consumes 1 unit "battery state"
        battery_charge     -- produces 1 unit "battery state"; consumes 1 kWh
                              electricity (modelled as solar input; the choice
                              group rewires this to the electricity pool)

    Choices group {solar, coal, battery_discharge} into a virtual product
    "electricity". Demand is placed on this group at every timestep.

    Storage triple: ``(battery_charge, battery_charge, K)`` carries
    ``K * s[t-1, battery_charge]`` over to the battery-state balance at t.

A toy run with K=0.9, demand 50 kWh per step, t=0 with abundant solar and
t=1 with no solar should charge the battery at t=0 and discharge most of it
at t=1, beating the no-battery (coal only at t=1) baseline.
"""

from __future__ import annotations

import bw2data as bd
import numpy as np

from pulpo.utils.utils import is_bw25

PROJECT_NAME = "elec_time_toy_bw25" if is_bw25() else "elec_time_toy"
DB_NAME = "elec_time_toy_db"
BIOSPHERE_NAME = "biosphere3"

# Public keys (importable so the notebook can build choices/storage by key).
CO2_KEY = (BIOSPHERE_NAME, "CO2")
SOLAR_KEY = (DB_NAME, "solar")
COAL_KEY = (DB_NAME, "coal")
BATTERY_CHARGE_KEY = (DB_NAME, "battery_charge")
BATTERY_DISCHARGE_KEY = (DB_NAME, "battery_discharge")


def _setup_biosphere():
    if BIOSPHERE_NAME in bd.databases:
        return
    biosphere_db = bd.Database(BIOSPHERE_NAME)
    biosphere_db.write({
        CO2_KEY: {
            "name": "Carbon dioxide, fossil",
            "categories": ("climate change", "GWP 100a"),
            "type": "emission",
            "unit": "kg",
        },
    })


def _setup_technosphere():
    if DB_NAME in bd.databases:
        del bd.databases[DB_NAME]
    db = bd.Database(DB_NAME)
    db.write({})

    process_data = [
        ("solar",             "kWh", "GLO", "electricity, solar"),
        ("coal",              "kWh", "GLO", "electricity, coal"),
        ("battery_charge",    "kWh", "GLO", "battery state"),
        ("battery_discharge", "kWh", "GLO", "electricity, battery"),
    ]
    for name, unit, location, ref_product in process_data:
        act = db.new_activity(name)
        act["unit"] = unit
        act["location"] = location
        act["name"] = name
        act["reference product"] = ref_product
        act.new_exchange(amount=1.0, input=act.key, type="production").save()
        act.save()

    # Technosphere/biosphere exchanges.
    # Conventions: "input -> target" means target consumes 1 unit of input.
    exchange_data = [
        # battery_charge consumes 1 kWh of solar electricity. Once the choice
        # group rewires {solar, coal, battery_discharge} -> "electricity",
        # this becomes a generic draw from the electricity pool.
        [SOLAR_KEY, BATTERY_CHARGE_KEY, 1.0, "technosphere"],
        # battery_discharge consumes 1 unit of battery state.
        [BATTERY_CHARGE_KEY, BATTERY_DISCHARGE_KEY, 1.0, "technosphere"],
        # Emissions: only coal emits.
        [CO2_KEY, COAL_KEY, 1.0, "biosphere"],
    ]
    for input_, target, amount, ex_type in exchange_data:
        target_act = next(a for a in db if a.key == target)
        target_act.new_exchange(amount=amount, input=input_, type=ex_type).save()
        target_act.save()


def _setup_lcia_method():
    method_key = ("GWP", "100a")
    if method_key in bd.methods:
        bd.Method(method_key).deregister()
    method = bd.Method(method_key)
    method.register(unit="kg CO2-eq", num_cfs=1, abbreviation="gwp",
                    description="GWP100", filename="gwp100")
    method.write([(CO2_KEY, {"amount": 1.0})])


def setup_elec_time_db():
    """Set up the project, biosphere, technosphere, and a single LCIA method."""
    bd.projects.set_current(PROJECT_NAME)
    _setup_biosphere()
    _setup_technosphere()
    _setup_lcia_method()
    print(
        f"[elec_time_toy] project={PROJECT_NAME!r}, db={DB_NAME!r}, "
        f"activities={len(bd.Database(DB_NAME))}, methods={list(bd.methods)}"
    )


if __name__ == "__main__":
    setup_elec_time_db()
