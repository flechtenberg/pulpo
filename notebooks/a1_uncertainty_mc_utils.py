"""
Uncertainty Analysis Utilities for PULPO - MC vs CC Comparison

This module contains utility functions for analyzing Monte Carlo results
and comparing them with Chance-Constrained optimization results.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import diags

# PULPO imports
from pulpo import pulpo
from pulpo.utils.uncertainty import processor


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
    """Define the ammonia production optimization problem with streamlined configuration."""
    # Choice definitions with capacities bound per-label
    choice_config = {
        "biogas": {
            "processes": [
                "anaerobic digestion of agricultural residues",
                "anaerobic digestion of sequential crop",
                "anaerobic digestion of animal manure",
            ],
            # 2030 EU-27 potentials from biomethane shares (38 bcm total; 24% ag, 21% sequential, 32% manure),
            # converted to raw biogas assuming ~57% CH₄ → 16.0 & 14.0 bcm & 21.3 bcm ≈ 1.60e10 & 1.40e10 & 2.13e10 m³/yr.
            "capacities": [1.60e10, 1.40e10, 2.13e10],
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
            # Methane pyrolysis capped to 10,000 t H2/yr (= 1.0e7 kg/yr); others left high for now.
            # "capacities": [3.0e8, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20],
            "capacities": [1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20],
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
    demand = {demand_process: 17.1e9}  # ~17.1 Mt/yr (EU)

    # Additional upper bounds (shared resources / feedstocks)
    waste_pp = get_single_process(pulpo_worker, "treatment of waste PP")
    waste_ps = get_single_process(pulpo_worker, "treatment of waste PS")
    ccs_process = get_single_process(pulpo_worker, "CCS 200km pipeline 1000m deep")

    upper_bounds = {
        waste_pp: 1e20,     #1.875e9,  # 25% of ~7.5 Mt PP post-consumer waste ≈ 1.875 Mt/yr
        waste_ps: 1e20,     #3.25e8,   # 25% of ~1.3 Mt PS waste ≈ 0.325 Mt/yr
        ccs_process: 1e20,  #5.0e9,    # 5 MtCO2/yr (10% of EU-27 2030 NZIA target)
    }
    
    # Instantiate the optimization problem
    pulpo_worker.instantiate(demand=demand, choices=choices, upper_limit=upper_bounds)
    
    return choices, demand


def get_uncertainty_strategies(method_name):
    """Define uncertainty strategies for the ammonia case study."""
    return [
        processor.TriangularBoundInterpolationStrategy(
            uncertain_param_type='If',
            uncertain_param_subgroup='ecoinvent-3.10-cutoff',
            noise_interval={'min': .1, 'max': .1}
        ),
        processor.TriangluarBaseStrategy(  # Assuming that the Foreground uncertainties are similar to the background ones
            uncertain_param_type='If',
            uncertain_param_subgroup='ammonia',
            upper_scaling_factor=0.95,  # The same as from TriangularBoundInterpolationStrategy of EcoInvent
            lower_scaling_factor=.5,    # The same as from TriangularBoundInterpolationStrategy of EcoInvent
            noise_interval={'min': .1, 'max': .1}
        ),
        # This is set to very low scaling factors to reflect that we assume only off-the-shelf uncertainty in CFs based on literature
        processor.TriangluarBaseStrategy(  # Based on Chen et al. 2021 10-20% variation in CFs
            uncertain_param_type='Cf',
            uncertain_param_subgroup=method_name,
            upper_scaling_factor=0.0001,
            lower_scaling_factor=0.0001,
            noise_interval={'min': .00, 'max': .00},
            inverse_bounds_for_negative_values=False  # This means that the skew is always towards zero, independent if value is larger or smaller to zero
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
                for proc, row in df.iterrows():
                    if row["Value"] > 0:
                        choices_data.append({
                            "iteration": i,
                            "technology": tech,
                            "process": proc,
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
    impacts = []
    
    # Extract scaling values
    scaling_values = []
    for entry in scaling_vector.values:
        if isinstance(entry, (list, tuple)):
            value = entry[-1]
            scaling_values.append(value if not isinstance(value, (list, np.ndarray)) else value[0])
        elif isinstance(entry, np.ndarray):
            scaling_values.append(entry[0] if len(entry) > 0 else 0)
        else:
            scaling_values.append(entry)
    
    S = diags(scaling_values, format='csr')
    
    # Calculate impact for each sample
    for i in range(len(overlay_samples)):
        Q_sample = overlay_samples[i]['lci_data']['matrices']["('IPCC 2013', 'climate change', 'global warming potential (GWP100)', 'uncertain')"]
        B_sample = overlay_samples[i]['lci_data']['intervention_matrix']
        
        impact = (Q_sample @ B_sample @ S).sum()
        impacts.append(float(impact))
    
    print(f"✓ {level_name}: Mean = {np.mean(impacts):.2e}, Std = {np.std(impacts):.2e}")
    return np.array(impacts)
