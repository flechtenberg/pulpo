"""
Uncertainty Analysis Utilities for PULPO

This module contains utility functions for setting up databases, configuring PULPO workers,
and analyzing Monte Carlo results for the ammonia production case study.
"""

import os
import sys
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Brightway imports
import bw2data as bd
import bw2io as bi

# PULPO imports
from pulpo import pulpo
from pulpo.utils.uncertainty import preparer, processor, plots, gsa
from pulpo.utils import optimizer, saver


def read_credentials(path: Path):
    """Read ecoinvent credentials from a text file."""
    if not path.is_file():
        raise FileNotFoundError(f"Couldn't find credentials file at: {path.resolve()}")
    creds = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # allow "key=value" or "key: value" or "key value"
        for sep in ("=", ":", " "):
            if sep in line:
                k, v = line.split(sep, 1)
                creds[k.strip().lower()] = v.strip()
                break
    if "username" not in creds or "password" not in creds:
        raise ValueError("credentials.txt must contain 'username' and 'password'.")
    return creds["username"], creds["password"]


def setup_ecoinvent_database(project_name, db_name, cred_path):
    """Setup Ecoinvent database in Brightway project."""
    bd.projects.set_current(project_name)
    
    if db_name in bd.databases:
        print(f"Database '{db_name}' already exists in project '{bd.projects.current}'.")
        return
    
    print(f"Importing '{db_name}' database...")
    username, password = read_credentials(cred_path)
    bi.import_ecoinvent_release(
        version="3.10",
        system_model="cutoff",
        username=username,
        password=password,
    )
    print(f"Database '{db_name}' installed successfully.")


def setup_foreground_database(excel_path, fg_db_name, bg_db_name):
    """Setup foreground database from Excel file."""
    if fg_db_name in bd.databases:
        print(f"Database '{fg_db_name}' already exists in project '{bd.projects.current}'.")
        return
    
    print(f"Importing foreground database from {excel_path}...")
    fg_db = bi.ExcelImporter(excel_path)
    fg_db.apply_strategies()
    fg_db.match_database(fields=["name", "unit", "reference product", "location"])
    fg_db.match_database(bg_db_name, fields=["name", "unit", "location", "reference product"])
    
    biosphere_db = [db for db in bd.databases if "biosphere" in db and "3.10" in db][0]
    fg_db.match_database(biosphere_db, fields=["name", "categories", "location"])
    
    fg_db.statistics()
    fg_db.write_database()
    print(f"Database '{fg_db_name}' installed successfully.")


def setup_impact_methods():
    """Setup required impact assessment methods."""
    # Check IPCC 2021 method
    target_method_2021 = ('IPCC 2021', 'climate change', 'GWP 100a, incl. H and bio CO2')
    if target_method_2021 not in bd.methods:
        print(f"Adding premise GWP characterization factors...")
        from premise_gwp import add_premise_gwp
        add_premise_gwp()
        print("Premise GWP added successfully.")
    else:
        print(f"Method '{target_method_2021}' already exists.")
    
    # Check IPCC 2013 uncertain method
    target_method_2013 = ('IPCC 2013', 'climate change', 'global warming potential (GWP100)', 'uncertain')
    if target_method_2013 not in bd.methods:
        print(f"Importing IPCC 2013 GWP method with uncertainty...")
        from bw2io.package import BW2Package
        BW2Package.import_file("data/ipcc_uncertain.bw2package")
        print("IPCC 2013 GWP with uncertainty added successfully.")
    else:
        print(f"Method '{target_method_2013}' already exists.")


def setup_ammonia_case_study():
    """Set up the ammonia production case study with PULPO configuration."""
    project = "ammonia_final"
    database = ["ecoinvent-3.10-cutoff", "ammonia"]
    method = "('IPCC 2013', 'climate change', 'global warming potential (GWP100)', 'uncertain')"
    directory = "develop_tests"
    
    return project, database, method, directory


def create_pulpo_worker(project, database, method, directory):
    """Create and initialize a PULPO optimizer instance."""
    pulpo_worker = pulpo.PulpoOptimizer(project, database, method, directory)
    pulpo_worker.intervention_matrix = "ecoinvent-3.10-biosphere"
    pulpo_worker.get_lci_data()
    
    return pulpo_worker


def get_single_process(worker, query, prefer_locations=("RER", "Europe", "GLO")):
    """Deterministic process retrieval to avoid order issues."""
    matches = worker.retrieve_processes(processes=query)
    if not matches:
        raise ValueError(f"No process found for query: {query}")
    for loc in prefer_locations:
        for p in matches:
            if getattr(p, "location", None) == loc or loc in str(p):
                return p
    return sorted(matches, key=lambda x: str(x))[0]


def define_ammonia_problem(pulpo_worker):
    """Define the ammonia production optimization problem."""
    # Choice definitions with capacities
    choice_config = {
        "biogas": {
            "processes": [
                "anaerobic digestion of agricultural residues",
                "anaerobic digestion of sequential crop",
            ],
            "capacities": [1.60e10, 1.40e10],
        },
        "biomethane": {
            "processes": [
                "upgrading water scrubbing (CCS)",
                "upgrading water scrubbing",
                "upgrading chemical scrubbing",
                "upgrading chemical scrubbing (CCS)",
            ],
            "capacities": [1e20, 1e20, 1e20, 1e20],
        },
        "methane": {
            "processes": ["market for methane fg", "market for biomethane"],
            "capacities": [1e20, 1e20],
        },
        "heat": {
            "processes": ["heat from methane", "heat from methane (CCS)", "heat from hydrogen"],
            "capacities": [1e20, 1e20, 1e20],
        },
        "hydrogen": {
            "processes": [
                "methane pyrolysis",
                "steam methane reforming",
                "steam methane reforming (CCS)",
                "plastics gasification",
                "plastics gasification (CCS)",
                "alkaline electrolysis",
                "PEM electrolysis",
            ],
            "capacities": [1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20],
        },
        "electricity": {
            "processes": ["grid electricity"],
            "capacities": [1e20],
        },
        "ammonia": {
            "processes": [
                "steam reforming, integrated",
                "steam reforming, integrated (CCS)",
                "nitrogen + hydrogen",
            ],
            "capacities": [1e20, 1e20, 1e20],
        },
    }

    # Build choices with deterministic mapping
    choices = {}
    for category, cfg in choice_config.items():
        labels, caps = cfg["processes"], cfg["capacities"]
        if len(labels) != len(caps):
            raise ValueError(f"Length mismatch in '{category}': {len(labels)} labels vs {len(caps)} capacities")
        choices[category] = {get_single_process(pulpo_worker, lbl): cap for lbl, cap in zip(labels, caps)}

    # Demand (EU ammonia, kg/yr)
    demand_process = get_single_process(pulpo_worker, "market for ammonia")
    demand = {demand_process: 17.1e9}

    # Additional upper bounds
    waste_pp = get_single_process(pulpo_worker, "treatment of waste PP")
    waste_ps = get_single_process(pulpo_worker, "treatment of waste PS")
    ccs_process = get_single_process(pulpo_worker, "CCS 200km pipeline 1000m deep")

    upper_bounds = {
        waste_pp: 1e20,
        waste_ps: 1e20,
        ccs_process: 1e20,
    }
    
    return choices, demand, upper_bounds


def get_uncertainty_strategies():
    """Define uncertainty strategies for the ammonia case study."""
    return [
        processor.TriangularBoundInterpolationStrategy(
            uncertain_param_type='If',
            uncertain_param_subgroup='ecoinvent-3.10-cutoff',
            noise_interval={'min': 0.1, 'max': 0.1}
        ),
        processor.UniformBaseStrategy(
            uncertain_param_type='If',
            uncertain_param_subgroup='ammonia',
            upper_scaling_factor=0.5,
            lower_scaling_factor=0.5,
            noise_interval={'min': 0.2, 'max': 0.2}
        ),
        processor.ExpertKnowledgeStrategy(
            uncertain_param_type='If',
            uncertain_param_subgroup='ammonia',
            prob_metadata={
                (715, 23523): {'minimum': .001, 'maximum': .8, 'uncertainty_type': 4},
                (80, 23561): {'minimum': .0002, 'maximum': .002, 'uncertainty_type': 4},
                (81, 23537): {'loc': .002, 'minimum': .00057, 'maximum': .006, 'uncertainty_type': 5},
                (82, 23537): {'loc': .003, 'minimum': .00093, 'maximum': .009, 'uncertainty_type': 5},
                (716, 23537): {'loc': .023, 'minimum': .00663, 'maximum': .069, 'uncertainty_type': 5},
            }
        )
    ]


def analyze_MC_results(results_MC, impact_method=None, show_plot=True):
    """
    Analyze Monte Carlo results from PulpoOptimizer.
    
    Args:
        results_MC (dict): Monte Carlo results dictionary
        impact_method (str or tuple): Impact method to analyze
        show_plot (bool): Whether to display histogram
        
    Returns:
        dict: Analysis results including statistics and choices
    """
    impacts = {}
    choices_data = []

    for i, result in results_MC.items():
        if not result or "Impacts" not in result:
            continue

        imp_df = result["Impacts"]
        if imp_df is None or imp_df.empty:
            continue

        # Auto-select the first method if none given
        if impact_method is None:
            impact_method = imp_df.index[0]
        
        if impact_method in imp_df.index:
            impacts[i] = imp_df.loc[impact_method, "Value"]
        else:
            continue

        # Extract choices
        choices_dict = result.get("Choices", {})
        for tech, df in choices_dict.items():
            if isinstance(df, pd.DataFrame):
                chosen = df[df["Value"] > 0]
                for meta, row in chosen.iterrows():
                    choices_data.append({
                        "iteration": i,
                        "technology": tech,
                        "process": meta,
                        "value": row["Value"]
                    })

    # Convert to DataFrames
    impacts_series = pd.Series(impacts, name="Impact Value")
    choices_df = pd.DataFrame(choices_data)

    # Compute statistics
    impact_stats = impacts_series.describe()

    # Aggregate choice frequency
    if not choices_df.empty:
        choices_summary = (
            choices_df.groupby(["technology", "process"])
            .agg(
                times_chosen=("iteration", "count"),
                avg_value=("value", "mean")
            )
            .sort_values("times_chosen", ascending=False)
        )
    else:
        choices_summary = pd.DataFrame()

    # Plot histogram
    if show_plot and not impacts_series.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(impacts_series, bins=25, edgecolor="black", alpha=0.7)
        plt.title("Monte Carlo Impacts Distribution")
        plt.xlabel("Impact Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    print("\n" + "="*60)
    print("MONTE CARLO ANALYSIS RESULTS")
    print("="*60)
    print("\nImpact Statistics:")
    print("-" * 60)
    print(impact_stats.round(3))
    print("\nTop Choices (by frequency):")
    print("-" * 60)
    if not choices_summary.empty:
        print(choices_summary.head(10))
    else:
        print("No active choices found.")
    print("="*60 + "\n")

    return {
        "impact_stats": impact_stats,
        "impact_values": impacts_series,
        "choices_summary": choices_summary,
    }


def save_results(data, filepath):
    """Save results to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"✓ Results saved to: {filepath}")


def load_results(filepath):
    """Load results from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"✓ Results loaded from: {filepath}")
    return data


def check_file_exists(filepath, force_recalculation=False):
    """Check if results file exists."""
    return os.path.exists(filepath) and not force_recalculation


def analyze_choices_in_risk_range(mc_results, analysis_data, risk_min=0.2, risk_max=0.8, label="Analysis"):
    """
    Analyze technology choices for Monte Carlo results within a risk range.
    """
    impacts = np.sort(analysis_data['impact_values'])
    n = len(impacts)
    i_min, i_max = int(risk_min * n), min(int(risk_max * n), n - 1)
    imp_min, imp_max = impacts[i_min], impacts[i_max]

    print(f"\n{label} — Risk {risk_min:.0%}–{risk_max:.0%} | Impact range: [{imp_min:.2e}, {imp_max:.2e}] kg CO₂-eq")

    # Filter iterations within the risk range
    valid_iters = [
        it['Choices'] for it in mc_results.values()
        if isinstance(it, dict)
        and imp_min <= it['Impacts'].iloc[0, 1] <= imp_max
    ]

    if not valid_iters:
        print("No iterations found in this risk range.")
        return None

    print(f"Found {len(valid_iters)} iterations in range.")

    # Count technology choices
    counts = {}
    for choices in valid_iters:
        for cat, tech in choices.items():
            tech_name = tech[0] if isinstance(tech, tuple) else str(tech)
            counts.setdefault(cat, {}).setdefault(tech_name, 0)
            counts[cat][tech_name] += 1

    # Report percentages
    total = len(valid_iters)
    for cat, techs in counts.items():
        print(f"\n{cat.upper()} Technology Choices:")
        for t, c in sorted(techs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {t}: {c}/{total} ({c / total * 100:.1f}%)")

    return dict(
        risk_range=(risk_min, risk_max),
        impact_range=(imp_min, imp_max),
        n_samples=total,
        choice_counts=counts
    )


def calculate_impact_distribution(overlay_samples, scaling_vector, level_name):
    """Calculate impact distribution for given scaling vector across all overlay samples."""
    from scipy.sparse import diags
    
    impacts = []
    
    # Extract scaling values
    scaling_values = []
    for entry in scaling_vector.values:
        if isinstance(entry, (list, tuple)):
            value = entry[-1]
            while isinstance(value, (list, tuple)) and len(value) > 0:
                value = value[-1]
            scaling_values.append(float(value))
        elif isinstance(entry, np.ndarray):
            scaling_values.append(float(entry.item() if entry.size == 1 else entry.flatten()[-1]))
        else:
            scaling_values.append(float(entry))
    
    S = diags(scaling_values, format='csr')
    
    # Calculate impact for each sample
    for i in range(len(overlay_samples)):
        Q_sample = overlay_samples[i]['lci_data']['matrices']["('IPCC 2013', 'climate change', 'global warming potential (GWP100)', 'uncertain')"]
        B_sample = overlay_samples[i]['lci_data']['intervention_matrix']
        
        impact = (Q_sample @ B_sample @ S).sum()
        impacts.append(float(impact))
    
    print(f"✓ {level_name}: Mean = {np.mean(impacts):.2e}, Std = {np.std(impacts):.2e}")
    return np.array(impacts)
