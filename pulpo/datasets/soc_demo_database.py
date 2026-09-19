"""
soc_demo_database.py

A purpose-built toy database for exercising the SOC (second-order-cone) chance-
constrained formulation and its two "future work" axes documented in
SOC-MECHANICS.md: whether uncertain parameters need to be resampled and refit
to a Normal before ``soc.py`` sees them (they don't - only mean/variance are
needed), and whether every surviving parameter needs to be gap-filled with a
distribution that shifts its mean (it doesn't - a degenerate Normal works too).

Unlike ``sample_database.py`` (which gives every non-production exchange a
blanket ``NormalUncertainty``, leaving nothing to gap-fill or to tell apart
from a closed-form moment computation), every uncertain exchange here is
deliberately assigned one of Normal / Lognormal / Triangular / Uniform, or is
left with no uncertainty declared at all. See the "Distribution gallery"
section of ``notebooks/soc_configuration_sweep.ipynb`` for the full table and
the reasoning behind each choice.

All three biosphere flows (CO2, CH4, N2O) and all three characterization
factors are genuinely climate-change-relevant - unlike a pre-2026-08 revision,
which reused particulate matter, ammonia slip and process water (real but
non-climate indicators) just to get a fifth flow into the distribution
gallery. That made ``METHOD_KEY`` a "toy, mixed indicators" method with CFs
that didn't belong together; folding N2O in as the third GHG instead keeps
the system a single honest climate-change indicator while still exercising
every distribution family. One exception: CO2's own CF carries no
uncertainty at all, because CO2 is the flow GWP100 is defined *against* -
its CF of 1 is exact, not an estimate with error bars, so it is written as a
degenerate Normal (scale=0) rather than assigned a family from the gallery.

Mirrors the ammonia case study's central mechanism (uncertainty treatment
changes which hydrogen route gets chosen) at toy scale: one product (ammonia),
one choice axis (hydrogen route: SMR vs. electrolysis), everything else
reduced to simple utility inputs.

Important scope note for anyone extending this module: PULPO's uncertainty
machinery (``pulpo_unc``'s ``'If'``/``'Cf'`` pipeline) only tracks uncertainty
on *biosphere* (intervention) flows and characterization factors - not on
technosphere-to-technosphere exchange amounts (``bw_parser.import_data`` reads
uncertainty parameters off the biosphere matrix only). So every uncertain
parameter below is a biosphere flow (an emission or a resource draw), even
where the table names it after the physical input it stands in for (e.g.
hydrogen electrolysis's dominant driver is its large electricity draw, but
that draw itself is a deterministic technosphere quantity; the uncertainty
lives on the *shared* ``electricity supply`` process's own CO2 emission,
which electrolysis's disproportionately large consumption then amplifies).
"""

import numpy as np
import bw2data as bd
from stats_arrays import NormalUncertainty, LognormalUncertainty, TriangularUncertainty, UniformUncertainty

from pulpo.utils.utils import is_bw25

METHOD_KEY = ("soc demo", "climate change")


def _project():
    bd.projects.set_current("soc_demo_project_bw25" if is_bw25() else "soc_demo_project")


# ---------------------------------------------------------------------------
# BIOSPHERE DATABASE
# ---------------------------------------------------------------------------

def setup_biosphere_db():
    """Create the biosphere flows used across the SOC demo system.

    Three elementary flows, all genuinely climate-change-relevant (CO2, CH4,
    N2O - the three GHGs that actually carry a GWP100 characterization
    factor), chosen so all four native distribution families (Normal /
    Lognormal / Triangular / Uniform) and the "undefined" gap-fill case each
    appear at least once (see the module docstring's scope note for why they
    are biosphere flows rather than technosphere exchange amounts). Earlier
    revisions also carried particulate matter, ammonia slip and process water
    to diversify the distribution gallery - but those are air-quality/water-use
    indicators, not climate change, so folding their characterization factors
    into ``METHOD_KEY`` was never more than a mechanical trick to get around
    ``import_and_filter_uncertainty_data``'s single-active-method limit. N2O
    replaces them as a third *bona fide* climate-change flow instead.
    """
    _project()
    db_name = "biosphere3"

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
            "name": "Methane, fossil",
            "categories": ("climate change", "GWP 100a"),
            "type": "emission",
            "unit": "kg",
        },
        ("biosphere3", "N2O"): {
            "name": "Dinitrogen monoxide, fossil",
            "categories": ("climate change", "GWP 100a"),
            "type": "emission",
            "unit": "kg",
        },
    }

    biosphere_db.write(biosphere_data)
    print(f"{db_name} created with {len(biosphere_data)} flows.")


# ---------------------------------------------------------------------------
# BACKGROUND DATABASE
# ---------------------------------------------------------------------------

def setup_background_db():
    """Natural gas extraction, generic electricity, SMR hydrogen (the
    conventional route), and N2 air separation."""
    _project()
    if "biosphere3" not in bd.databases:
        setup_biosphere_db()

    co2 = ("biosphere3", "CO2")
    ch4 = ("biosphere3", "CH4")
    n2o = ("biosphere3", "N2O")

    db_name = "soc_demo_background_db"
    if db_name in bd.databases:
        del bd.databases[db_name]
    db = bd.Database(db_name)

    ng_key = (db_name, "natural gas extraction")
    elec_key = (db_name, "electricity supply")
    smr_key = (db_name, "hydrogen SMR")
    n2_key = (db_name, "N2 air separation")

    data = {
        # 1. Natural gas extraction -> "natural gas" (kg)
        ng_key: {
            "name": "natural gas extraction",
            "unit": "kg",
            "location": "GLO",
            "reference product": "natural gas",
            "exchanges": [
                {"input": ng_key, "amount": 1.0, "type": "production"},
                # Fugitive CH4 during extraction - ecoinvent-style right-skewed
                # multiplicative uncertainty.
                {"input": ch4, "amount": 0.006, "type": "biosphere",
                 "uncertainty type": LognormalUncertainty.id,
                 "loc": float(np.log(0.006)), "scale": 0.3, "negative": False,
                 "minimum": np.nan, "maximum": np.nan, "shape": np.nan},
                # Venting/compression CO2.
                {"input": co2, "amount": 0.05, "type": "biosphere",
                 "uncertainty type": NormalUncertainty.id,
                 "loc": 0.05, "scale": 0.005,
                 "minimum": np.nan, "maximum": np.nan, "shape": np.nan},
            ],
        },

        # 2. Electricity supply -> "electricity" (kWh); one generic mix, not a choice.
        elec_key: {
            "name": "electricity supply",
            "unit": "kWh",
            "location": "GLO",
            "reference product": "electricity",
            "exchanges": [
                {"input": elec_key, "amount": 1.0, "type": "production"},
                # Shared across every consuming process below - this is what
                # makes electrolysis's much larger electricity draw amplify
                # the same underlying uncertainty far more than SMR's does.
                #
                # Lognormal, which is both the ecoinvent convention for an
                # emission factor and a deliberate choice here. This parameter
                # carries most of the output variance, so it is what the
                # aggregate's distribution looks like; declaring it Normal would
                # make the Gaussian assumption behind Phi^-1(lambda) true almost
                # by construction, and the calibration check in the sweep
                # notebook's Appendix C would be close to a tautology. A
                # lognormal dominant term makes that check a real test. It also
                # removes a negative tail that a symmetric Normal on a positive
                # quantity necessarily has.
                #
                # Following the stats_arrays/ecoinvent convention, `amount` is
                # the *median* and `loc` is its log; the mean is larger by
                # exp(sigma^2/2). So the stored 0.15 kg CO2/kWh has an expected
                # value of 0.170, which is the median-vs-mean gap a stochastic
                # model has to use and a deterministic LCA does not see.
                #
                # scale=0.5 (GSD 1.65, CV ~53%) is sized, not decorative: it
                # matches the spread of the Normal it replaces, and it is what
                # makes the electrolysis route's risk penalty overtake its
                # mean-cost advantage at a reachable confidence level. At CV
                # ~20% the chance-constrained optimum never leaves the
                # electrolysis capacity cap, even as lambda -> 1.
                {"input": co2, "amount": 0.15, "type": "biosphere",
                 "uncertainty type": LognormalUncertainty.id,
                 "loc": float(np.log(0.15)), "scale": 0.5, "negative": False,
                 "minimum": np.nan, "maximum": np.nan, "shape": np.nan},
            ],
        },

        # 3. Hydrogen SMR -> "hydrogen" (kg); the conventional route.
        smr_key: {
            "name": "hydrogen SMR",
            "unit": "kg",
            "location": "GLO",
            "reference product": "hydrogen",
            "exchanges": [
                {"input": smr_key, "amount": 1.0, "type": "production"},
                {"input": ng_key, "amount": 3.4, "type": "technosphere"},
                {"input": elec_key, "amount": 0.4, "type": "technosphere"},
                # Unreacted methane slip.
                {"input": ch4, "amount": 0.002, "type": "biosphere",
                 "uncertainty type": TriangularUncertainty.id,
                 "loc": 0.002, "minimum": 0.001, "maximum": 0.004,
                 "scale": np.nan, "shape": np.nan},
                # Reformer flue-gas N2O - a small but measured combustion trace.
                {"input": n2o, "amount": 0.0003, "type": "biosphere",
                 "uncertainty type": NormalUncertainty.id,
                 "loc": 0.0003, "scale": 0.000036,
                 "minimum": np.nan, "maximum": np.nan, "shape": np.nan},
                # Reforming-reaction CO2 - the dominant driver of SMR's footprint.
                {"input": co2, "amount": 9.5, "type": "biosphere",
                 "uncertainty type": LognormalUncertainty.id,
                 "loc": float(np.log(9.5)), "scale": 0.15, "negative": False,
                 "minimum": np.nan, "maximum": np.nan, "shape": np.nan},
            ],
        },

        # 4. N2 air separation -> "nitrogen" (kg); simple utility input, not a
        # choice, and - being a purely physical separation with no combustion
        # or reaction step - with no climate-relevant emissions of its own.
        n2_key: {
            "name": "N2 air separation",
            "unit": "kg",
            "location": "GLO",
            "reference product": "nitrogen",
            "exchanges": [
                {"input": n2_key, "amount": 1.0, "type": "production"},
                {"input": elec_key, "amount": 0.12, "type": "technosphere"},
            ],
        },
    }

    db.write(data)
    print(f"{db_name} created with {len(data)} activities.")


# ---------------------------------------------------------------------------
# FOREGROUND DATABASE
# ---------------------------------------------------------------------------

def setup_foreground_db():
    """Hydrogen electrolysis (the alternative route) and ammonia synthesis
    (final product, demand target)."""
    _project()
    if "biosphere3" not in bd.databases:
        setup_biosphere_db()
    if "soc_demo_background_db" not in bd.databases:
        setup_background_db()

    co2 = ("biosphere3", "CO2")
    n2o = ("biosphere3", "N2O")

    bg = "soc_demo_background_db"
    db_name = "soc_demo_foreground_db"
    if db_name in bd.databases:
        del bd.databases[db_name]
    db = bd.Database(db_name)

    elyz_key = (db_name, "hydrogen electrolysis")
    nh3_synth_key = (db_name, "ammonia synthesis")

    data = {
        # 5. Hydrogen electrolysis -> "hydrogen" (kg); the alternative route.
        # No biosphere flow of its own: water electrolysis has no combustion
        # or reaction step that emits CO2/CH4/N2O, so unlike the pre-2026-08
        # revision (which gave it a decorative process-water flow) it carries
        # *only* its large deterministic electricity draw (52 kWh/kg H2 vs
        # SMR's 0.4 kWh/kg H2) - this asymmetry is what carries electrolysis's
        # dominant uncertainty, by amplifying `electricity supply`'s own
        # shared CO2 emission (see module docstring's scope note).
        elyz_key: {
            "name": "hydrogen electrolysis",
            "unit": "kg",
            "location": "GLO",
            "reference product": "hydrogen",
            "exchanges": [
                {"input": elyz_key, "amount": 1.0, "type": "production"},
                {"input": (bg, "electricity supply"), "amount": 52.0, "type": "technosphere"},
            ],
        },

        # 6. Ammonia synthesis -> "ammonia" (kg); final product, demand target.
        nh3_synth_key: {
            "name": "ammonia synthesis",
            "unit": "kg",
            "location": "GLO",
            "reference product": "ammonia",
            "exchanges": [
                {"input": nh3_synth_key, "amount": 1.0, "type": "production"},
                {"input": (bg, "N2 air separation"), "amount": 0.82, "type": "technosphere"},
                # The choice-enabling link: PULPO's `choices` mechanism
                # reroutes this (and hydrogen SMR's own production) onto a
                # shared virtual "hydrogen" product once both are listed as
                # alternatives - see notebooks/uncertainty_toy.ipynb sec. 4.
                {"input": elyz_key, "amount": 0.178, "type": "technosphere"},
                {"input": (bg, "electricity supply"), "amount": 0.06, "type": "technosphere"},
                # Left undefined - vented process CO2.
                {"input": co2, "amount": 0.02, "type": "biosphere"},
                # Trace N2O co-emission - poorly characterized, so given an
                # IPCC-style bounded default range rather than a fitted mean.
                {"input": n2o, "amount": 0.0008, "type": "biosphere",
                 "uncertainty type": UniformUncertainty.id,
                 "minimum": 0.0003, "maximum": 0.0015,
                 "loc": np.nan, "scale": np.nan, "shape": np.nan},
            ],
        },
    }

    db.write(data)
    print(f"{db_name} created with {len(data)} activities.")


# ---------------------------------------------------------------------------
# LCIA METHOD
# ---------------------------------------------------------------------------

def setup_lcia_methods():
    """Register a single, genuinely single-indicator LCIA method: GWP100 over
    the three climate-change flows declared in ``setup_biosphere_db``.

    Unlike the pre-2026-08 revision - which folded air-quality (PM, NH3) and
    water-use (H2O) characterization factors into this method purely because
    ``import_and_filter_uncertainty_data`` only supports one active method at
    a time - every CF registered here is an actual GWP100 value (AR6, rounded):
    CO2 = 1, fossil CH4 = 29.7, N2O = 273. CO2's CF carries no uncertainty:
    it is the *reference* flow GWP100 is defined against, so 1 is exact by
    construction, not an empirically uncertain measurement like the other two.
    """
    _project()

    co2 = ("biosphere3", "CO2")
    ch4 = ("biosphere3", "CH4")
    n2o = ("biosphere3", "N2O")

    for method in list(bd.methods):
        if method == METHOD_KEY:
            bd.Method(method).deregister()

    method = bd.Method(METHOD_KEY)
    method.register(unit="kg CO2eq", num_cfs=3,
                    abbreviation="soc-demo-cc",
                    description="SOC demo toy impact method (see module docstring)",
                    filename="soc_demo_climate_change")
    method.write([
        # CO2 is the reference flow: CF = 1 by definition, not an uncertain
        # quantity, so this is a degenerate Normal (scale=0) rather than a
        # gap to be left undefined - it must never pick up noise from a
        # gap-fill strategy the way a genuinely missing CF would.
        (co2, {"uncertainty type": NormalUncertainty.id, "loc": 1.0, "scale": 0.0,
               "shape": np.nan, "minimum": np.nan, "maximum": np.nan,
               "negative": False, "amount": 1.0}),
        (ch4, {"uncertainty type": LognormalUncertainty.id, "loc": float(np.log(29.7)),
               "scale": 0.1, "shape": np.nan, "minimum": np.nan, "maximum": np.nan,
               "negative": False, "amount": 29.7}),
        # Left undefined - a CF-level gap, not just an 'If'-level one.
        (n2o, 273.0),
    ])
    print(f"Registered LCIA method {METHOD_KEY} with 3 characterization factors.")


# ---------------------------------------------------------------------------
# SETUP ALL
# ---------------------------------------------------------------------------

def setup_soc_demo_db():
    setup_biosphere_db()
    setup_background_db()
    setup_foreground_db()
    setup_lcia_methods()


def main():
    setup_soc_demo_db()


if __name__ == "__main__":
    main()
