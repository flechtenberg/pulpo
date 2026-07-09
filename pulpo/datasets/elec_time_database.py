"""Minimal toy database for the time-dependent PULPO formulation.

Five-step electricity dispatch with a battery, modelled in the four-activity
*ESM-style* pattern (CHARGE / HOLD / HOLD t-1 / DISCHARGE) so that the
storage logic is energy-conserving under PULPO's product×product carry-over
matrix ``K`` (see :mod:`pulpo.utils.time_extension`).

    Activities (each is its own product)
        solar              -- produces 1 kWh; no emissions
        coal               -- produces 1 kWh; 1 kg CO2/kWh
        battery_charge     -- CHARGE: consumes 1 kWh electricity, produces
                              1 unit of "charge_product" (its self-product)
        battery_hold       -- HOLD: produces 1 unit of "charge_product" at t,
                              consuming 1 unit of "holdtm1_product" at t (the
                              carried storage state; see K below)
        battery_discharge  -- DISCHARGE: consumes 1 unit of "charge_product",
                              produces 1 kWh of electricity
        battery_holdtm1    -- HOLD t-1 phantom: ref product is
                              "holdtm1_product"; locked at scaling 0 by
                              the upper-bound. Exists only so that the
                              technosphere matrix has a producer for the
                              storage-state product; the real injection
                              comes from the K matrix.

    Choices group {solar, coal, battery_discharge} into a virtual
    "electricity" product (where the demand is placed), and group
    {battery_charge, battery_hold} into a virtual "charge_product" so that
    DISCHARGE can draw from either fresh charge or held-over carry.

    Storage triple: ``("holdtm1_product", "charge_product", K)`` sets
    ``K[holdtm1_product, charge_product] = K``. The carry term in the
    Pyomo demand_constraint becomes

        carry_t = K * net_production_of_charge_product_at_{t-1}
                = K * (CHARGE[t-1] + HOLD[t-1] - DISCHARGE[t-1])

    which is exactly the running storage level after one step of K decay.
    The constraint on holdtm1_product (PRODUCT_STOR, >=) then bounds
    ``HOLD[t] <= K * net_production_of_charge_product[t-1]``, while the
    constraint on charge_product (also >=) bounds
    ``DISCHARGE[t] <= CHARGE[t] + HOLD[t]``. Together this is an
    energy-conserving battery with round-trip efficiency K per held step.
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
BATTERY_HOLD_KEY = (DB_NAME, "battery_hold")
BATTERY_DISCHARGE_KEY = (DB_NAME, "battery_discharge")
BATTERY_HOLDTM1_KEY = (DB_NAME, "battery_holdtm1")

# PULPO choice labels exposed for the storage spec (the K matrix references
# the 'charge_product' choice, not any individual activity).
CHARGE_PRODUCT_CHOICE = "charge_product"
ELECTRICITY_CHOICE = "electricity"


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
        ("battery_charge",    "kWh", "GLO", "charge_product"),
        ("battery_hold",      "kWh", "GLO", "charge_product"),
        ("battery_discharge", "kWh", "GLO", "electricity, battery"),
        ("battery_holdtm1",   "kWh", "GLO", "holdtm1_product"),
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
    # Convention: (input, target, amount, type) means the *target* activity
    # has an exchange of `amount` units of `input`. With type='technosphere'
    # this is a consumption (the matrix entry is -amount).
    exchange_data = [
        # battery_charge consumes 1 kWh of solar electricity. After the
        # 'electricity' choice rewires {solar, coal, discharge} -> 'electricity',
        # this becomes a generic draw from the electricity pool.
        [SOLAR_KEY,           BATTERY_CHARGE_KEY,    1.0, "technosphere"],
        # battery_hold consumes 1 unit of holdtm1_product (the carried state).
        # holdtm1_product is supplied at runtime via the K-matrix injection,
        # not by the static phantom activity (which is locked at 0).
        [BATTERY_HOLDTM1_KEY, BATTERY_HOLD_KEY,      1.0, "technosphere"],
        # battery_discharge consumes 1 unit of charge_product. Pre-rewiring
        # this is wired to battery_charge's product; the 'charge_product'
        # choice then redirects it to the {charge, hold} virtual product.
        [BATTERY_CHARGE_KEY,  BATTERY_DISCHARGE_KEY, 1.0, "technosphere"],
        # Emissions: only coal emits.
        [CO2_KEY,             COAL_KEY,              1.0, "biosphere"],
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
