"""
Uncertainty Analysis Plotting Functions for PULPO - MC vs CC Comparison

This module contains visualization functions for uncertainty analysis results,
including comparative plots, cumulative risk curves, and choice analysis visualizations.
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import FormatStrFormatter


def format_plot(ax, xlabel, ylabel, xlim, ylim=None):
    """Utility function for consistent plot styling."""
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3, which='major')
    ax.grid(True, alpha=0.15, which='minor')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    ax.set_box_aspect(1)
    plt.tight_layout()


def plot_comparative_mc_analysis(analysis_strategies, analysis_normal, 
                                cc_pareto_results, results_dir='data/results'):
    """
    Create four progressive comparison plots for Monte Carlo analysis approaches.
    """
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Calculate consistent axis ranges for all plots
    all_values = np.concatenate([
        analysis_strategies['impact_values'], 
        analysis_normal['impact_values']
    ])

    # Prepare CC-Pareto data for cumulative risk plotting
    cc_lambda_vals = []
    cc_impact_vals = []
    for lambda_eps in cc_pareto_results['lambda_epsilon_array']:
        if lambda_eps in cc_pareto_results['results_CC']:
            result = cc_pareto_results['results_CC'][lambda_eps]
            if "Impacts" in result and not result["Impacts"].empty:
                cc_lambda_vals.append(lambda_eps)
                cc_impact_vals.append(float(result["Impacts"].iloc[0, 1]))

    # Include CC-Pareto results in the x-axis range calculation
    if cc_impact_vals:
        all_values_including_cc = np.concatenate([all_values, cc_impact_vals])
    else:
        all_values_including_cc = all_values

    x_min, x_max = all_values_including_cc.min(), all_values_including_cc.max()
    x_range = x_max - x_min
    x_min_plot = x_min - 0.05 * x_range
    x_max_plot = x_max + 0.05 * x_range

    # Sort CC data by impact value for proper cumulative plotting
    cc_sorted_data = sorted(zip(cc_impact_vals, cc_lambda_vals))
    cc_impacts_sorted, cc_lambdas_sorted = zip(*cc_sorted_data) if cc_sorted_data else ([], [])

    # Figure 1: Histogram of Custom Strategies Results
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(analysis_strategies['impact_values'], bins=30, alpha=0.7, color='steelblue', 
            label='$B_{eco}$ + Custom Strategies', edgecolor='black', linewidth=0.5)
    format_plot(ax, 'Impact Value (kg CO₂-eq)', 'Frequency', (x_min_plot, x_max_plot))
    ax.legend(loc='upper left', framealpha=0.9, fontsize=12)
    plt.savefig(f'{results_dir}/mc_comparison_1_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Figure 2: Cumulative Risk of Custom Strategies Results
    fig, ax = plt.subplots(figsize=(5, 5))
    sorted_strat = np.sort(analysis_strategies['impact_values'])
    cumulative_prob_strat = np.arange(1, len(sorted_strat) + 1) / len(sorted_strat)
    ax.plot(sorted_strat, cumulative_prob_strat, color='steelblue', linewidth=2, 
            label='$B_{eco}$ + Custom Strategies')
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
    format_plot(ax, 'Impact Value (kg CO₂-eq)', 'Cumulative Probability (Risk)', 
                (x_min_plot, x_max_plot), (0, 1))
    ax.legend(loc='upper left', framealpha=0.9, fontsize=12)
    plt.savefig(f'{results_dir}/mc_comparison_2_cumulative_strategies.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Figure 3: Add Normal Fits to Cumulative Risk
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(sorted_strat, cumulative_prob_strat, color='steelblue', linewidth=2, 
            label='$B_{eco}$ + Custom Strategies')
    sorted_norm = np.sort(analysis_normal['impact_values'])
    cumulative_prob_norm = np.arange(1, len(sorted_norm) + 1) / len(sorted_norm)
    ax.plot(sorted_norm, cumulative_prob_norm, color='forestgreen', linewidth=2, 
            label='All Normal Fits')
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
    format_plot(ax, 'Impact Value (kg CO₂-eq)', 'Cumulative Probability (Risk)', 
                (x_min_plot, x_max_plot), (0, 1))
    ax.legend(loc='upper left', framealpha=0.9, fontsize=12)
    plt.savefig(f'{results_dir}/mc_comparison_3_cumulative_all.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Figure 4: Compare Normal Fits MC with CC-Pareto Results
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(sorted_norm, cumulative_prob_norm, color='forestgreen', linewidth=2, 
            label='All Normal Fits (MC)')
    if cc_impacts_sorted and cc_lambdas_sorted:
        ax.plot(cc_impacts_sorted, cc_lambdas_sorted, color='red', linewidth=2, 
                marker='o', markersize=6, label='CC-PULPO')
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
    format_plot(ax, 'Impact Value (kg CO₂-eq)', 
                'Cumulative Probability (Risk / Confidence Level λ)', 
                (x_min_plot, x_max_plot), (0, 1))
    ax.legend(loc='upper left', framealpha=0.9, fontsize=12)
    plt.savefig(f'{results_dir}/mc_comparison_4_cc_vs_mc.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n✓ Four progressive comparison plots saved:")
    print(f"  1. {results_dir}/mc_comparison_1_histogram.png - Custom strategies histogram")
    print(f"  2. {results_dir}/mc_comparison_2_cumulative_strategies.png - Custom strategies cumulative risk")
    print(f"  3. {results_dir}/mc_comparison_3_cumulative_all.png - + Normal fits")
    print(f"  4. {results_dir}/mc_comparison_4_cc_vs_mc.png - CC-Pareto vs MC comparison")


def plot_cc_pareto_distributions(impact_distributions, results_dir='data/results'):
    """
    Create incremental figures showing CC-Pareto impact distributions.
    Each figure adds the next confidence level in sorted order.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import Rectangle
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Use whatever lambda values are actually in the impact_distributions
    available_lambdas = sorted(list(impact_distributions.keys()))

    if not available_lambdas:
        print("No CC-Pareto impact distributions found.")
        return
    
    print(f"Plotting impact distributions for λ values: {available_lambdas}")

    # Color palette (keeps consistent ordering)
    colors = ["#355C63", "#D99A3C", "#A53D31", "#4E1512"]

    # Iterate and create incremental figures
    for idx in range(len(available_lambdas)):
        current_lambdas = available_lambdas[: idx + 1]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Determine x-limits for this specific figure based on current lambdas
        current_impacts = np.concatenate([impact_distributions[l] for l in current_lambdas])
        x_min, x_max = np.min(current_impacts), np.max(current_impacts)
        x_range = x_max - x_min if x_max > x_min else max(1.0, abs(x_min))
        x_min_plot = x_min - 0.05 * x_range
        x_max_plot = x_max + 0.05 * x_range

        # Left: Histograms with mean lines only (no fits)
        max_count = 0
        for i, lambda_val in enumerate(current_lambdas):
            impacts = np.asarray(impact_distributions[lambda_val])
            color = colors[i % len(colors)]

            # Plot histogram with automatic binning for this distribution
            counts, bins, patches = ax1.hist(impacts, bins=30, alpha=0.5, color=color, 
                                             edgecolor='black', linewidth=0.5, 
                                             label=f'λ = {lambda_val}')
            max_count = max(max_count, counts.max())
            
            # Mean line
            mean_val = np.mean(impacts)
            ax1.axvline(mean_val, color=color, linestyle='--', linewidth=2, alpha=0.8)

        # Increase font sizes
        ax1.set_xlabel('Impact Value (kg CO₂-eq)', fontsize=14)
        ax1.set_ylabel('Frequency', fontsize=14)
        ax1.set_xlim(x_min_plot, x_max_plot)
        ax1.set_ylim(0, max_count * 1.1)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        ax1.tick_params(axis='both', which='major', labelsize=12)

        # Right: Cumulative risk curves for the same lambdas
        for i, lambda_val in enumerate(current_lambdas):
            impacts = np.asarray(impact_distributions[lambda_val])
            color = colors[i % len(colors)]
            sorted_impacts = np.sort(impacts)
            cumulative = np.arange(1, len(sorted_impacts) + 1) / len(sorted_impacts)
            ax2.plot(sorted_impacts, cumulative, color=color, linewidth=2, label=f'λ = {lambda_val}')

        ax2.axhline(y=0.5, color='black', linestyle=':', alpha=0.6, linewidth=1)
        ax2.set_xlabel('Impact Value (kg CO₂-eq)', fontsize=14)
        ax2.set_ylabel('Cumulative Probability (Risk)', fontsize=14)
        ax2.set_xlim(x_min_plot, x_max_plot)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        ax2.tick_params(axis='both', which='major', labelsize=12)

        # Add inset zoom for high confidence region
        axins = inset_axes(ax2, width="30%", height="30%", loc='center left',
                          bbox_to_anchor=(0.10, 0.13, 1, 1), bbox_transform=ax2.transAxes)
        
        # Plot the same data in the inset
        for i, lambda_val in enumerate(current_lambdas):
            impacts = np.asarray(impact_distributions[lambda_val])
            color = colors[i % len(colors)]
            sorted_impacts = np.sort(impacts)
            cumulative = np.arange(1, len(sorted_impacts) + 1) / len(sorted_impacts)
            axins.plot(sorted_impacts, cumulative, color=color, linewidth=1.5)
        
        # Set the zoom region
        axins.set_xlim(1.4e10, 2.2e10)
        axins.set_ylim(0.98, 1.0)
        axins.grid(True, alpha=0.3, linewidth=0.5)
        axins.tick_params(labelsize=8)
        
        # Add box around the zoomed region in the main plot
        from matplotlib.patches import Rectangle
        zoom_box = Rectangle((1.4e10, 0.98), 0.8e10, 0.02, 
                             linewidth=1.5, edgecolor='gray', 
                             facecolor='none', linestyle='--', alpha=0.7)
        ax2.add_patch(zoom_box)

        plt.tight_layout()
        step = idx + 1
        out_path = f"{results_dir}/cc_pareto_impact_distributions_step{step}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.show()

    print(f"\n✓ Saved {len(available_lambdas)} incremental CC-Pareto distribution figures to {results_dir}")


def plot_choice_analysis(results, mc_results, save_path='data/results/choice_analysis_by_risk.png'):
    """Plot technology choice frequencies and mean total values across risk ranges."""
    
    if not results:
        print("No analysis results to plot.")
        return
    
    # Ensure directory exists for save_path
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Setup categories
    categories = [c for c in sorted({c for r in results.values() for c in r['choice_counts']})
                  if c.lower() != "electricity"]
    risk_labels = list(results.keys())

    # Fixed 2×3 grid
    n_cols, n_rows = 3, 2
    fig_width_in, fig_height_in = 3.35 * n_cols, 3.35 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width_in, fig_height_in))
    axes = np.atleast_1d(axes).ravel()

    # Color palette
    palette = ["#355C63", "#FFF8B5", "#D99A3C", "#A53D31", "#4E1512"]

    def clean_name(name):
        if isinstance(name, str) and "Metadata" in name:
            parts = name.split("Metadata:")
            return parts[1].strip() if len(parts) > 1 else name
        return name[0] if isinstance(name, tuple) else str(name)

    # Label shortening map
    label_shortening_map = {
        'nitrogen + hydrogen': 'N₂ + H₂',
        'steam reforming, integrated (CCS)': 'Steam Reform (CCS)',
        'steam reforming, integrated': 'Steam Reform',
        'anaerobic digestion of agricultural residues': 'Agri Residue AD',
        'anaerobic digestion of sequential crop': 'Sequential Crop AD',
        'upgrading chemical scrubbing': 'Chemical Scrub',
        'upgrading chemical scrubbing (CCS)': 'Chemical Scrub (CCS)',
        'upgrading water scrubbing': 'Water Scrub',
        'upgrading water scrubbing (CCS)': 'Water Scrub (CCS)',
        'grid electricity': 'Grid Electricity',
        'heat from hydrogen': 'From H₂',
        'heat from methane': 'From CH₄',
        'heat from methane (CCS)': 'From CH₄ (CCS)',
        'alkaline electrolysis': 'Alkaline Electrolysis',
        'plastics gasification (CCS)': 'Plastic Gasif (CCS)',
        'plastics gasification': 'Plastic Gasif',
        'steam methane reforming': 'Steam Reform',
        'steam methane reforming (CCS)': 'Steam Reform (CCS)',
        'market for biomethane': 'Biomethane Market',
        'market for methane fg': 'Methane Market'
    }

    # Compute mean values for each category and risk range
    mean_values = {cat: {risk_label: np.nan for risk_label in risk_labels} for cat in categories}

    for risk_label, res in results.items():
        (risk_min, risk_max) = res["risk_range"]
        impacts = np.sort(res["impact_range"])
        imp_min, imp_max = impacts[0], impacts[-1]

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
                valid_iters.append(it)
        
        if not valid_iters:
            continue

        for cat in categories:
            cat_values = []
            for it in valid_iters:
                choices = it.get("Choices", {})
                if cat in choices:
                    choice_df = choices[cat]
                    if isinstance(choice_df, pd.DataFrame):
                        cat_values.append(choice_df["Value"].sum())
                mean_values[cat][risk_label] = np.mean(cat_values)

    # Plot each category
    for idx, (ax, cat) in enumerate(zip(axes, categories)):
        # Determine column position (0=left, 1=middle, 2=right)
        col_idx = idx % n_cols
        
        # Collect all unique technologies across all risk ranges
        all_techs = set()
        for res in results.values():
            counts = res["choice_counts"].get(cat, {})
            all_techs.update(counts.keys())
        
        # Build tech_data ensuring each tech has a value for every risk range
        tech_data = {tech: [] for tech in all_techs}
        for risk_name in risk_labels:
            res = results[risk_name]
            counts = res["choice_counts"].get(cat, {})
            total_count = res["n_samples"]
            for tech_name in all_techs:
                count = counts.get(tech_name, 0)
                tech_data[tech_name].append((count / total_count) * 100)

        if not tech_data:
            ax.text(0.5, 0.5, f'No data for {cat}', ha='center', va='center', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        x = np.arange(len(risk_labels))
        bottom = np.zeros(len(risk_labels))

        tech_names = list(tech_data.items())
        
        # Dictionary mapping for technology name abbreviations
        name_mapping = {
            'steam reforming, integrated': 'Steam Reform',
            'steam reforming, integrated (CCS)': 'Steam Reform (CCS)',
            'steam methane reforming': 'Steam Reform',
            'steam methane reforming (CCS)': 'Steam Reform (CCS)',
            'methane pyrolysis | hydrogen | RER': 'Methane Pyrolysis',
            'anaerobic digestion of agricultural residues': 'Agri Residue AD',
            'anaerobic digestion of sequential crop': 'Sequential Crop AD',
            'anaerobic digestion of animal manure': 'Animal Manure AD',
            'upgrading chemical scrubbing': 'Chemical Scrub',
            'upgrading chemical scrubbing (CCS)': 'Chemical Scrub (CCS)',
            'upgrading water scrubbing': 'Water Scrub',
            'upgrading water scrubbing (CCS)': 'Water Scrub (CCS)',
            'alkaline electrolysis': 'Alkaline Electrol.',
            'plastics gasification': 'Plastic Gasif',
            'plastics gasification (CCS)': 'Plastic Gasif (CCS)',
            'market for biomethane': 'Biomethane',
            'market for methane fg': 'Methane FG',
            'nitrogen + hydrogen': 'N₂ + H₂',
            'heat from hydrogen': 'Heat from H₂',
            'heat from methane': 'Heat from CH₄',
            'heat from methane (CCS)': 'Heat from CH₄ (CCS)',
            'grid electricity': 'Grid Elec.'
        }
        
        # Helper function to shorten technology names for legend
        def shorten_name(name):
            """Shorten technology names for better legend readability."""
            if isinstance(name, tuple):
                name = str(name[0]) if len(name) > 0 else str(name)
            name = str(name).strip()
            
            # Check dictionary mapping first (case-insensitive)
            name_lower = name.lower()
            for key, short in name_mapping.items():
                if key.lower() in name_lower:
                    return short
            
            # Fallback to generic replacements
            name = name.replace('production', 'prod.')
            name = name.replace('electricity', 'elec.')
            name = name.replace('generation', 'gen.')
            # Truncate if still too long
            if len(name) > 40:
                name = name[:37] + '...'
            return name

        # Bars: technology selection frequencies with labels for legend
        for i, (name, vals) in enumerate(tech_names):
            color = palette[i % len(palette)]
            short_name = shorten_name(name)
            ax.bar(x, vals, 0.8, bottom=bottom, color=color, alpha=0.85, 
                   edgecolor='white', linewidth=0.5, label=short_name)
            bottom += vals

        # Formatting with vertical y-labels to save space for legend
        ax.set_xticks(x)
        ax.set_xticklabels(risk_labels, rotation=45, ha='right', fontsize=9)
        
        # Primary y-axis label: only show for first (left) column
        if col_idx == 0:
            ax.set_ylabel("Selection\nFrequency (%)", rotation=90, labelpad=10, va='center', fontsize=9)
        
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.25, linestyle=':')
        ax.set_box_aspect(1)
        
        # Add category title at top
        ax.set_title(cat, fontsize=10, fontweight='bold', pad=5)
        
        # Add legend with reversed order (bottom stacks first) inside plot at lower left
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='lower left', fontsize=8, 
                  frameon=True, framealpha=0.9, edgecolor='gray')

        # Secondary axis: mean "Value" (vertical label)
        ax2 = ax.twinx()
        y_vals = [mean_values[cat][r] for r in risk_labels]
        if any(np.isfinite(y_vals)):
            ax2.plot(x, y_vals, 'o-', color='black', linewidth=2, markersize=5, alpha=0.7)
        
        # Secondary y-axis label: only show for last (right) column
        if col_idx == n_cols - 1:
            ax2.set_ylabel("Mean Total\nValue", rotation=90, labelpad=10, va='center', fontsize=8, color="black")
        
        ax2.tick_params(axis='y', labelsize=8)
        ax2.set_ylim(0, max(y_vals) * 1.1 if any(np.isfinite(y_vals)) else 1)

    # Hide unused axes
    for ax in axes[len(categories):]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Choice analysis plot saved to: {save_path}")
    plt.show()


def create_summary_table(analysis_strategies, analysis_normal):
    """Create comparative summary table for Monte Carlo approaches."""
    
    comparison_data = {
        'Approach': [
            'Custom Strategies',
            'Fitted Normal'
        ],
        'Mean Impact': [
            analysis_strategies['impact_stats']['mean'],
            analysis_normal['impact_stats']['mean']
        ],
        'Std Dev': [
            analysis_strategies['impact_stats']['std'],
            analysis_normal['impact_stats']['std']
        ],
        'Min Impact': [
            analysis_strategies['impact_stats']['min'],
            analysis_normal['impact_stats']['min']
        ],
        'Max Impact': [
            analysis_strategies['impact_stats']['max'],
            analysis_normal['impact_stats']['max']
        ],
        'CV (%)': [
            (analysis_strategies['impact_stats']['std'] / analysis_strategies['impact_stats']['mean']) * 100,
            (analysis_normal['impact_stats']['std'] / analysis_normal['impact_stats']['mean']) * 100
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS - ALL MONTE CARLO APPROACHES")
    print("="*80)
    print("\n", comparison_df.to_string(index=False))
    print("\n" + "="*80)
    
    return comparison_df


def plot_analytical_distributions(analytical_distributions, save_path=None, figsize=(14, 6),
                                   thresholds=None):
    """
    Plot the analytical impact distributions (PDFs and CDFs) for each confidence level.
    Uses consistent color scheme with other plots and includes zoom inset for CDFs.
    
    For a weighted sum of independent normal distributions:
    - If X_i ~ N(μ_i, σ_i²) with scaling factors s_i
    - Then Σ s_i X_i ~ N(Σ s_i μ_i, Σ s_i² σ_i²)
    
    Args:
        analytical_distributions: Dict from compute_analytical_distributions_for_all_lambdas
        save_path: Optional path to save the figure
        figsize: Figure size tuple
        thresholds: Optional dict of {label: value} for threshold lines (in kg CO2-eq).
                    E.g., {"PB": 3.70144e9, "20 Mt": 20e9}
                    Calculates P(impact < threshold) for each distribution.
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    n_levels = len(analytical_distributions)
    
    # Consistent color palette with other plots in the module
    # Extended from the base palette ["#355C63", "#D99A3C", "#A53D31", "#4E1512"]
    colors = [
        "#355C63",  # Teal (low confidence)
        "#5C8A8E",  # Light teal
        "#D99A3C",  # Gold/amber
        "#E6B861",  # Light gold
        "#A53D31",  # Brick red
        "#4E1512",  # Dark red (high confidence)
        "#2E0C0B",  # Very dark red
    ][:n_levels]
    
    # Threshold line styles for multiple thresholds
    threshold_styles = [
        {'color': 'black', 'linestyle': '--', 'linewidth': 2},
        {'color': 'darkred', 'linestyle': '-.', 'linewidth': 2},
        {'color': 'darkblue', 'linestyle': ':', 'linewidth': 2},
    ]
    
    # Create figure with 2 subplots (PDFs and CDFs)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Collect all distributions for combined plot range
    all_x_min = float('inf')
    all_x_max = float('-inf')
    
    # Pre-calculate ranges
    for lambda_val, params in analytical_distributions.items():
        mean, std = params['mean'], params['std']
        all_x_min = min(all_x_min, mean - 4*std)
        all_x_max = max(all_x_max, mean + 4*std)
    
    # ========== Plot 1: PDFs with threshold probabilities in legend ==========
    ax1 = axes[0]
    for i, (lambda_val, params) in enumerate(analytical_distributions.items()):
        mean, std = params['mean'], params['std']
        x = np.linspace(all_x_min, all_x_max, 500)
        pdf = scipy.stats.norm.pdf(x, mean, std)
        
        # Format lambda label with threshold probabilities if provided
        if thresholds is not None and len(thresholds) > 0:
            prob_parts = []
            for j, (th_label, th_value) in enumerate(thresholds.items()):
                prob_below = scipy.stats.norm.cdf(th_value, mean, std)
                # Use more digits for second threshold onwards
                if j == 0:
                    prob_parts.append(f"P<{th_label}: {prob_below:.1%}")
                else:
                    prob_parts.append(f"P<{th_label}: {prob_below:.3%}")
            label = f'λ = {lambda_val:.2f} ({", ".join(prob_parts)})'
        else:
            label = f'λ = {lambda_val:.2f}'
        
        ax1.plot(x, pdf, color=colors[i], linewidth=2.5, label=label)
        ax1.fill_between(x, pdf, alpha=0.15, color=colors[i])
    
    # Add vertical threshold lines if provided
    if thresholds is not None:
        y_max = ax1.get_ylim()[1]
        for j, (th_label, th_value) in enumerate(thresholds.items()):
            style = threshold_styles[j % len(threshold_styles)]
            ax1.axvline(th_value, **style, alpha=0.7)
            # Add label for threshold
            ax1.text(th_value, y_max * (0.95 - j * 0.08), f' {th_label}', fontsize=11, 
                     fontweight='bold', va='top', ha='left', color=style['color'])
    
    ax1.set_xlabel('Total environmental impact [kg CO₂-eq]', fontsize=14)
    ax1.set_ylabel('Probability density', fontsize=14)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_ylim(bottom=0)
    
    # ========== Plot 2: CDFs with zoom inset ==========
    ax2 = axes[1]
    
    x = np.linspace(all_x_min, all_x_max, 500)
    
    # Store data for zoom
    cdf_data = []
    
    for i, (lambda_val, params) in enumerate(analytical_distributions.items()):
        mean, std = params['mean'], params['std']
        cdf = scipy.stats.norm.cdf(x, mean, std)
        
        label = f'λ = {lambda_val:.2f}'
        ax2.plot(x, cdf, color=colors[i], linewidth=2.5, label=label)
        cdf_data.append((lambda_val, mean, std, colors[i]))
    
    ax2.set_xlabel('Total environmental impact [kg CO₂-eq]', fontsize=14)
    ax2.legend(loc='lower left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.set_ylim(0, 1)
    
    # Move y-axis to the right side
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Cumulative probability', fontsize=14)
    
    # ========== Add zoom inset for high confidence region ==========
    zoom_x_min = 1e10
    zoom_x_max = 2.5e10
    zoom_y_min = 0.997
    zoom_y_max = 1.0
    
    # Create inset axes
    axins = inset_axes(ax2, width="40%", height="40%", loc='center left',
                       bbox_to_anchor=(0.08, 0.25, 1, 1), bbox_transform=ax2.transAxes)
    
    # Plot CDFs in inset
    x_zoom = np.linspace(zoom_x_min, zoom_x_max, 300)
    for lambda_val, mean, std, color in cdf_data:
        cdf_zoom = scipy.stats.norm.cdf(x_zoom, mean, std)
        axins.plot(x_zoom, cdf_zoom, color=color, linewidth=2)
    
    # Set zoom limits
    axins.set_xlim(zoom_x_min, zoom_x_max)
    axins.set_ylim(zoom_y_min, zoom_y_max)
    axins.grid(True, alpha=0.3)
    axins.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    axins.tick_params(axis='both', labelsize=10)
    axins.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    # Draw rectangle and connecting lines to indicate zoom region
    mark_inset(ax2, axins, loc1=1, loc2=4, fc="none", ec="0.5", linestyle='--', linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    plt.show()
    
    return fig