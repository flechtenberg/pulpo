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
    valid_iters = []
    for it in mc_results.values():
        if not isinstance(it, dict) or 'Impacts' not in it or 'Choices' not in it:
            continue
        
        imp_df = it['Impacts']
        if imp_df is None or imp_df.empty:
            continue
        
        # Get the impact value (first method, "Value" column)
        impact_val = imp_df.iloc[0]['Value'] if 'Value' in imp_df.columns else imp_df.iloc[0, 1]
        
        if imp_min <= impact_val <= imp_max:
            valid_iters.append(it['Choices'])

    if not valid_iters:
        print("No iterations found in this risk range.")
        return None

    print(f"Found {len(valid_iters)} iterations in range.")

    # Count dominant technology choices (highest value in each category)
    counts = {}
    for choices in valid_iters:
        for cat, choice_df in choices.items():
            if isinstance(choice_df, pd.DataFrame) and not choice_df.empty:
                # Find the technology with the highest Value
                if 'Value' in choice_df.columns:
                    dominant_tech = choice_df['Value'].idxmax()
                    # Get clean name
                    if isinstance(dominant_tech, tuple):
                        tech_name = dominant_tech[0] if len(dominant_tech) > 0 else str(dominant_tech)
                    else:
                        tech_name = str(dominant_tech)
                    
                    counts.setdefault(cat, {}).setdefault(tech_name, 0)
                    counts[cat][tech_name] += 1

    # Report percentages
    total = len(valid_iters)
    for cat, techs in counts.items():
        print(f"\n{cat.upper()} Dominant Technology:")
        for t, c in sorted(techs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {t}: {c}/{total} ({c / total * 100:.1f}%)")

    return dict(
        risk_range=(risk_min, risk_max),
        impact_range=(imp_min, imp_max),
        n_samples=total,
        choice_counts=counts
    )


def apply_expert_knowledge_co2_uptake(pulpo_worker, lower_scaling=0.5, upper_scaling=0.1):
    """
    Apply expert knowledge strategy for CO2 uptake from biogas processes.
    
    Args:
        pulpo_worker: PULPO optimizer instance with uncertainty data
        lower_scaling: Lower bound scaling factor (default: 0.5)
        upper_scaling: Upper bound scaling factor (default: 0.1)
        
    Returns:
        list: Expert knowledge strategy for intervention flows
    """
    # Get descriptive names of intervention flows for ammonia processes
    If_ammonia_unc = processor.rename_metadata_index(
        pd.DataFrame.from_records(pulpo_worker.uncertainty_data['If']['ammonia']['defined']).T, 
        pulpo_worker.lci_data, 
        'intervention_flow'
    )
    
    # Intervention flows requiring special attention (CO2 from biogas processes)
    If_names = [
        "anaerobic digestion of agricultural residues | biogas | RER --- Carbon dioxide, in air | ('natural resource', 'in air')",
        "anaerobic digestion of animal manure | biogas | RER --- Carbon dioxide, in air | ('natural resource', 'in air')",
        "anaerobic digestion of sequential crop | biogas | RER --- Carbon dioxide, in air | ('natural resource', 'in air')"
    ]
    
    # Extract and refine intervention flow uncertainty
    matched_If_indcs = If_ammonia_unc.loc[If_names, 'index'].values
    matched_If_unc_metadata = {indx: pulpo_worker.uncertainty_data['If']['ammonia']['defined'][indx] for indx in matched_If_indcs}
    processor.fit_normals(matched_If_unc_metadata, plot_distributions=False, lci_data=pulpo_worker.lci_data)
    
    # Apply refined bounds for intervention flows
    for indx, unc_metadata in matched_If_unc_metadata.items():
        matched_If_unc_metadata[indx]['minimum'] = unc_metadata['amount'] - unc_metadata['amount'] * lower_scaling
        matched_If_unc_metadata[indx]['maximum'] = unc_metadata['amount'] + unc_metadata['amount'] * upper_scaling
    processor.fit_normals(matched_If_unc_metadata, plot_distributions=False, lci_data=pulpo_worker.lci_data)
    
    # Create and return expert knowledge strategy
    return [processor.ExpertKnowledgeStrategy(
        uncertain_param_type='If',
        uncertain_param_subgroup='ammonia',
        prob_metadata=matched_If_unc_metadata
    )]


def apply_expert_knowledge_biomass_bounds(pulpo_worker, lower_multiplier=0.1, upper_multiplier=1.1):
    """
    Apply expert knowledge strategy for biomass feedstock availability bounds.
    
    Args:
        pulpo_worker: PULPO optimizer instance with uncertainty data
        lower_multiplier: Lower bound multiplier (default: 0.1)
        upper_multiplier: Upper bound multiplier (default: 1.1)
        
    Returns:
        list: Combined expert knowledge and triangular base strategies for variable bounds
    """
    # Get descriptive names of variable bounds for biomass processes
    var_bounds_unc = processor.rename_metadata_index(
        pd.DataFrame.from_records(pulpo_worker.uncertainty_data['Var_bounds']['upper_limit']['undefined']).T, 
        pulpo_worker.lci_data, 
        'process'
    )
    
    # Biomass processes requiring special attention
    process_name_patterns = [
        "anaerobic digestion of agricultural residues | biogas | RER",
        "anaerobic digestion of animal manure | biogas | RER",
        "anaerobic digestion of sequential crop | biogas | RER"
    ]
    
    # Find matching process names in the dataframe index
    process_names = []
    for pattern in process_name_patterns:
        matches = [idx for idx in var_bounds_unc.index if pattern in idx]
        if matches:
            process_names.append(matches[0])
        else:
            print(f"Warning: Could not find process matching '{pattern}'")
    
    if not process_names:
        print("Warning: No matching biomass processes found in var_bounds_unc")
        return []
    
    # Extract and refine variable bounds uncertainty
    matched_process_indcs = var_bounds_unc.loc[process_names, 'index'].values
    matched_varbound_unc_metadata = {indx: pulpo_worker.uncertainty_data['Var_bounds']['upper_limit']['undefined'][indx] 
                                     for indx in matched_process_indcs}
    
    # Apply refined bounds for variable bounds
    for indx, unc_metadata in matched_varbound_unc_metadata.items():
        matched_varbound_unc_metadata[indx]['minimum'] = unc_metadata['amount'] * lower_multiplier
        matched_varbound_unc_metadata[indx]['maximum'] = unc_metadata['amount'] * upper_multiplier
        matched_varbound_unc_metadata[indx]['loc'] = unc_metadata['amount']
        matched_varbound_unc_metadata[indx]['uncertainty_type'] = 5
    processor.fit_normals(matched_varbound_unc_metadata, plot_distributions=False, lci_data=pulpo_worker.lci_data)
    
    # Create expert knowledge strategy for variable bounds
    expert_strategy = [processor.ExpertKnowledgeStrategy(
        uncertain_param_type='Var_bounds',
        uncertain_param_subgroup='upper_limit',
        prob_metadata=matched_varbound_unc_metadata
    )]
    
    # Add triangular base strategy for remaining variable bounds
    base_strategy = [processor.TriangluarBaseStrategy(
        uncertain_param_type='Var_bounds',
        uncertain_param_subgroup='upper_limit',
        upper_scaling_factor=.001,
        lower_scaling_factor=.001,
        noise_interval={'min': .05, 'max': .05}
    )]
    
    return expert_strategy + base_strategy


def apply_all_expert_strategies(pulpo_worker, base_strategies):
    """
    Apply all expert knowledge refinements in one call.
    
    Args:
        pulpo_worker: PULPO optimizer instance
        base_strategies: List of base uncertainty strategies to apply first
        
    Returns:
        None (modifies pulpo_worker in place)
    """
    # Apply base strategies
    pulpo_worker.apply_uncertainty_strategies(strategies=base_strategies, plot_results=False)
    
    # Apply CO2 uptake expert knowledge
    co2_strategies = apply_expert_knowledge_co2_uptake(pulpo_worker)
    pulpo_worker.apply_uncertainty_strategies(strategies=co2_strategies)
    
    # Apply biomass bounds expert knowledge
    biomass_strategies = apply_expert_knowledge_biomass_bounds(pulpo_worker)
    if biomass_strategies:
        pulpo_worker.apply_uncertainty_strategies(strategies=biomass_strategies)


def calculate_impact_distribution(overlay_samples, scaling_vector, level_name):
    """Calculate impact distribution for given scaling vector across all overlay samples."""
    impacts = []
    
    # Extract scaling values from the DataFrame's 'Value' column
    if isinstance(scaling_vector, pd.DataFrame):
        if 'Value' in scaling_vector.columns:
            scaling_values = scaling_vector['Value'].values
        else:
            # Fallback: try to get the last column
            scaling_values = scaling_vector.iloc[:, -1].values
    elif isinstance(scaling_vector, pd.Series):
        scaling_values = scaling_vector.values
    else:
        # Assume it's already an array-like
        scaling_values = np.array(scaling_vector)
    
    # Convert to float array
    scaling_array = np.array(scaling_values, dtype=float)
    
    # Create diagonal sparse matrix
    S = diags([scaling_array], offsets=[0], format='csr')
    
    # Calculate impact for each sample
    for i in range(len(overlay_samples)):
        Q_sample = overlay_samples[i]['lci_data']['matrices']["('IPCC 2013', 'climate change', 'global warming potential (GWP100)', 'uncertain')"]
        B_sample = overlay_samples[i]['lci_data']['intervention_matrix']
        
        impact = (Q_sample @ B_sample @ S).sum()
        impacts.append(float(impact))
    
    print(f"✓ {level_name}: Mean = {np.mean(impacts):.2e}, Std = {np.std(impacts):.2e}")
    return np.array(impacts)
