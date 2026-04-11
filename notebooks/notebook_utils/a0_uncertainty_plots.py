import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def plot_pareto_from_results(results_CC, results_dir='data/results', lambda_range:tuple=None, legend_abbreviations:dict=None, vlines:list=None, suffix:str=None, hline_y:float=None, previous_results:dict=None, previous_label:str='Previous iteration', previous_alpha:float=0.35, previous_color:str=None, base_case_result:dict=None, base_case_label:str='Base'):
    """
    Streamlined plotting: treat Pareto as the top subplot showing impact cumsum,
    followed by choice bar charts showing technology cumsum and contribution analysis.
    
    Parameters:
    -----------
    legend_abbreviations : dict, optional
        Dictionary to abbreviate legend entries. Keys are original names, values are abbreviations.
        Example: {"steam methane reforming": "SMR", "alkaline electrolysis": "AE"}
    vlines : list, optional
        List of lambda values where vertical lines should be drawn. Default is [0.9, 0.96].
        Example: [0.8, 0.9, 0.95] or [] for no vertical lines.
    suffix : str, optional
        Suffix to append to the filename (e.g., "_iteration_0", "_iteration_1").
        If None, no suffix is added.
    hline_y : float, optional
        Y-value for a significant red horizontal line with shadow on the Pareto plot.
        If None, no horizontal line is drawn.
    previous_results : dict, optional
        Optional secondary results mapping (same structure as `results_CC`) whose
        Pareto impacts will be drawn translucent in the Pareto subplot to show a
        previous iteration or baseline. Keys are lambda values.
    previous_label : str, optional
        Legend label for the previous_results overlay (default: 'Previous iteration').
    previous_alpha : float, optional
        Alpha transparency for the overlay (default: 0.35).
    previous_color : str, optional
        Color for the overlay line/area. If None a neutral gray is used.
    base_case_result : dict, optional
        Optional base case result (from deterministic optimization without uncertainty).
        Will be plotted as leftmost entry. Should have same structure as a single
        results_CC entry (keys: 'Impacts', 'ENV_COST_MATRIX', 'Scaling Vector', 'Choices').
    base_case_label : str, optional
        Label for the base case entry on x-axis (default: 'Base').
    """

    # Filter results by lambda_range
    selected = {}
    for k, v in results_CC.items():
        if lambda_range is None:
            selected[k] = v
        else:
            lam_min, lam_max = lambda_range
            if lam_min <= float(k) <= lam_max:
                selected[k] = v

    # Sort keys numerically
    ordered_keys = sorted(selected.keys(), key=lambda x: float(x))
    
    # Prepend base case if provided
    include_base_case = base_case_result is not None
    if include_base_case:
        # Insert base case at the beginning
        ordered_keys = [base_case_label] + ordered_keys
        selected = {base_case_label: base_case_result, **selected}
    
    # Set default vertical lines if not specified
    if vlines is None:
        vlines = [0.94]
    
    # Extract impacts for Pareto
    impacts = []
    for k in ordered_keys:
        res = selected[k]
        if 'Impacts' in res and not res['Impacts'].empty:
            impacts.append(float(res['Impacts'].iloc[0, 1]))
        else:
            impacts.append(0)

    # If previous_results provided, build matching impacts for overlay plot
    previous_impacts = None
    if previous_results is not None:
        # Filter previous results by same lambda_range (if given)
        prev_selected = {}
        for pk, pv in previous_results.items():
            if lambda_range is None:
                prev_selected[pk] = pv
            else:
                lam_min, lam_max = lambda_range
                if lam_min <= float(pk) <= lam_max:
                    prev_selected[pk] = pv

        # Prepare sorted keys and numeric values for nearest-key lookup
        if prev_selected:
            prev_keys_sorted = sorted(prev_selected.keys(), key=lambda x: float(x))
            prev_lambda_vals = [float(x) for x in prev_keys_sorted]
            previous_impacts = []
            for k in ordered_keys:
                # find closest previous lambda
                closest_idx = min(range(len(prev_lambda_vals)), key=lambda i: abs(prev_lambda_vals[i] - float(k)))
                if abs(prev_lambda_vals[closest_idx] - float(k)) < 0.02:
                    pk = prev_keys_sorted[closest_idx]
                    pres = prev_selected[pk]
                    if 'Impacts' in pres and not pres['Impacts'].empty:
                        previous_impacts.append(float(pres['Impacts'].iloc[0, 1]))
                    else:
                        previous_impacts.append(0)
                else:
                    previous_impacts.append(0)
        else:
            previous_impacts = [0] * len(ordered_keys)
    
    # Calculate environmental impact contributions for each lambda
    def calculate_contributions(result):
        """Calculate environmental impact contributions for a single result."""
        env_cost_matrix = result['ENV_COST_MATRIX']
        scaling_vector = result['Scaling Vector']
        
        # Create process ID -> env_cost mapping
        env_costs_by_process = {
            idx[0] if isinstance(idx, tuple) else idx: env_cost_matrix.iloc[i, 0] 
            for i, idx in enumerate(env_cost_matrix.index)
        }
        
        # Calculate impacts for each process
        impacts = []
        for process_id in scaling_vector.index:
            env_cost = env_costs_by_process.get(process_id, 0)
            supply = scaling_vector.loc[process_id, 'Value']
            impact = env_cost * supply
            if abs(impact) > 1e-6:  # Only store significant impacts
                impacts.append({
                    'process_id': process_id,
                    'metadata': scaling_vector.loc[process_id, 'Metadata'],
                    'impact': impact
                })
        
        # Sort by impact magnitude and categorize
        impact_df = pd.DataFrame(impacts).sort_values('impact', key=abs, ascending=False)
        positive_impacts = impact_df[impact_df['impact'] > 0].sort_values('impact', ascending=False)
        negative_impacts = impact_df[impact_df['impact'] < 0].sort_values('impact', ascending=True)
        
        return positive_impacts, negative_impacts
    
    # Calculate contributions for all lambda values
    all_positive_contributions = {}
    all_negative_contributions = {}
    
    for k in ordered_keys:
        pos, neg = calculate_contributions(selected[k])
        all_positive_contributions[k] = pos
        all_negative_contributions[k] = neg
    
    # Create choices_data - exclude electricity
    choices_data = {}
    for k in ordered_keys:
        res = selected[k]
        choices = res.get('Choices', {})
        for choice_name, choice_df in choices.items():
            # Skip electricity choices
            if 'electricity' in choice_name.lower():
                continue
            if choice_name not in choices_data:
                choices_data[choice_name] = pd.DataFrame(index=choice_df.index)
            choices_data[choice_name][k] = choice_df['Value']
    
    # Remove zero rows
    for choice in list(choices_data.keys()):
        choices_data[choice] = choices_data[choice].loc[~(choices_data[choice] == 0).all(axis=1)]
        if choices_data[choice].empty:
            del choices_data[choice]

    # Create single figure with subplots - Pareto with contributions + choices
    n_choices = len(choices_data)
    n_total = 1 + n_choices  # Pareto (with contributions) + choices
    
    # Height ratios: Pareto gets more space for contributions, choices compact
    height_ratios = [3.0] + [0.50] * n_choices  # Larger Pareto for contributions
    
    fig, axs = plt.subplots(n_total, 1, figsize=(8, sum(height_ratios)),  # Increased width for legends
                           sharex=True, gridspec_kw={'height_ratios': height_ratios})
    if n_total == 1:
        axs = [axs]
    
    # Common x-axis setup
    x_pos = np.arange(len(ordered_keys))
    
    # Build lambda_vals list (handle base case as special non-numeric entry)
    lambda_vals = []
    for k in ordered_keys:
        if k == base_case_label:
            lambda_vals.append(-1)  # Placeholder for base case (not used for tick matching)
        else:
            lambda_vals.append(float(k))
    
    # Define tick positions for specific lambda values: 0.5-0.94 in 0.04 steps, then 0.95-0.99 in 0.01 steps
    desired_ticks = list(np.arange(0.5, 0.95, 0.04)) + list(np.arange(0.95, 1.0, 0.01))
    xtick_positions = []
    xtick_labels = []
    
    # Always include base case if present
    if include_base_case:
        xtick_positions.append(0)
        xtick_labels.append(base_case_label)
    
    for desired_tick in desired_ticks:
        # Find the closest lambda value to the desired tick (skip base case entry)
        start_idx = 1 if include_base_case else 0
        closest_idx = min(range(start_idx, len(lambda_vals)), key=lambda i: abs(lambda_vals[i] - desired_tick))
        if abs(lambda_vals[closest_idx] - desired_tick) < 0.005:  # tolerance for matching
            xtick_positions.append(closest_idx)
            xtick_labels.append(f"{desired_tick:.2f}")
    
    # Fallback if no ticks found (beyond base case)
    if len(xtick_positions) == (1 if include_base_case else 0):
        if include_base_case:
            xtick_positions.append(len(x_pos)-1)
            xtick_labels.append(f"{lambda_vals[-1]:.2f}")
        else:
            xtick_positions = [0, len(x_pos)-1]
            xtick_labels = [f"{lambda_vals[0]:.2f}", f"{lambda_vals[-1]:.2f}"]
    
    # Show every second tick (but always keep base case if present)
    if include_base_case:
        # Keep base case (index 0) and every second tick after that
        xtick_positions = [xtick_positions[0]] + xtick_positions[1::2]
        xtick_labels = [xtick_labels[0]] + xtick_labels[1::2]
    else:
        xtick_positions = xtick_positions[::2]
        xtick_labels = xtick_labels[::2]
    
    # Plot 1: Pareto with integrated contribution analysis
    pareto_ax = axs[0]
    
    # Plot main Pareto line (skip base case if present)
    start_idx = 1 if include_base_case else 0
    pareto_ax.plot(x_pos[start_idx:], impacts[start_idx:], marker='o', linestyle='-', color='gray', linewidth=2, 
                   markersize=6, markeredgecolor='black', markeredgewidth=0.5, label='Total impact', zorder=10)
    
    # Plot base case as separate standalone point with legend entry
    if include_base_case:
        pareto_ax.plot(x_pos[0], impacts[0], marker='D', markersize=8, color='darkgray', 
                       markeredgecolor='black', markeredgewidth=0.5, linestyle='', 
                       label='Deterministic', zorder=11)
    # Overlay previous iteration impacts (if any) with transparency
    if previous_impacts is not None:
        overlay_color = previous_color if previous_color is not None else '#A0A0A0'
        pareto_ax.plot(x_pos, previous_impacts, marker='s', linestyle='--', color=overlay_color, markeredgecolor='black', 
                       linewidth=1.5, alpha=previous_alpha, label=previous_label, zorder=6, markeredgewidth=0.5,)
        # Fill under the overlay to emphasize previous curve (subtle)
        #pareto_ax.fill_between(x_pos, previous_impacts, 0, color=overlay_color, alpha=max(0.05, previous_alpha * 0.4), zorder=4)
    pareto_ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.6)
    
    # Prepare contribution data for stacked bars
    # Get all unique contributors across all lambda values for consistent coloring
    all_contributors = set()
    for k in ordered_keys:
        pos_contrib = all_positive_contributions[k]
        neg_contrib = all_negative_contributions[k]
        if len(pos_contrib) >= 1:
            all_contributors.add(pos_contrib.iloc[0]['metadata'])
        if len(pos_contrib) >= 2:
            all_contributors.add(pos_contrib.iloc[1]['metadata'])
        if len(neg_contrib) >= 1:
            all_contributors.add(neg_contrib.iloc[0]['metadata'])
        if len(neg_contrib) >= 2:
            all_contributors.add(neg_contrib.iloc[1]['metadata'])
    
    # Create color mapping for contributors
    contributor_colors = {
        'pos_1': '#D99A3C',  # Orange for top positive
        'pos_2': '#E8B668',  # Lighter orange for second positive
        'pos_rest': '#FAEAC7',  # Light orange for rest positive
        'neg_1': '#355C63',  # Teal for top negative
        'neg_2': '#5C8A91',  # Lighter teal for second negative
        'neg_rest': '#C4E0E5'  # Light teal for rest negative
    }
    
    # Prepare data arrays for stacked bars
    pos_1_data = []
    pos_2_data = []
    pos_rest_data = []
    neg_1_data = []
    neg_2_data = []
    neg_rest_data = []
    
    # Labels for legend
    pos_1_label = "Other burden"
    pos_2_label = "Other burden"
    neg_1_label = "Other benefit"
    neg_2_label = "Other benefit"
    
    for i, k in enumerate(ordered_keys):
        pos_contrib = all_positive_contributions[k]
        neg_contrib = all_negative_contributions[k]
        
        # Positive contributions
        if len(pos_contrib) >= 1:
            val1 = pos_contrib.iloc[0]['impact']
            pos_1_data.append(val1)
            if i == 0:  # Set label from first lambda
                label = pos_contrib.iloc[0]['metadata']
                pos_1_label = legend_abbreviations.get(label, label[:30] + "..." if len(label) > 30 else label)
        else:
            pos_1_data.append(0)
            
        if len(pos_contrib) >= 2:
            val2 = pos_contrib.iloc[1]['impact']
            pos_2_data.append(val2)
            if i == 0:  # Set label from first lambda
                label = pos_contrib.iloc[1]['metadata']
                pos_2_label = legend_abbreviations.get(label, label[:30] + "..." if len(label) > 30 else label)
        else:
            pos_2_data.append(0)
            
        pos_rest = pos_contrib.iloc[2:]['impact'].sum() if len(pos_contrib) > 2 else 0
        pos_rest_data.append(pos_rest)
        
        # Negative contributions
        if len(neg_contrib) >= 1:
            val1 = neg_contrib.iloc[0]['impact']
            neg_1_data.append(val1)
            if i == 0:  # Set label from first lambda
                label = neg_contrib.iloc[0]['metadata']
                neg_1_label = legend_abbreviations.get(label, label[:30] + "..." if len(label) > 30 else label)
        else:
            neg_1_data.append(0)
            
        if len(neg_contrib) >= 2:
            val2 = neg_contrib.iloc[1]['impact']
            neg_2_data.append(val2)
            if i == 0:  # Set label from first lambda
                label = neg_contrib.iloc[1]['metadata']
                neg_2_label = legend_abbreviations.get(label, label[:30] + "..." if len(label) > 30 else label)
        else:
            neg_2_data.append(0)
            
        neg_rest = neg_contrib.iloc[2:]['impact'].sum() if len(neg_contrib) > 2 else 0
        neg_rest_data.append(neg_rest)
    
    # Convert to numpy arrays
    pos_1_data = np.array(pos_1_data)
    pos_2_data = np.array(pos_2_data)
    pos_rest_data = np.array(pos_rest_data)
    neg_1_data = np.array(neg_1_data)
    neg_2_data = np.array(neg_2_data)
    neg_rest_data = np.array(neg_rest_data)
    
    # Plot stacked bars for contributions
    width = 0.6
    
    # Positive contributions (stack upwards from 0) - plot in reverse order for legend
    pareto_ax.bar(x_pos, pos_rest_data, width, bottom=pos_1_data + pos_2_data, 
                  color=contributor_colors['pos_rest'], alpha=0.7, label='Other burdens', 
                  edgecolor='white', linewidth=0.5)
    pareto_ax.bar(x_pos, pos_2_data, width, bottom=pos_1_data, color=contributor_colors['pos_2'], 
                  alpha=0.7, label=pos_2_label, edgecolor='white', linewidth=0.5, hatch='///')
    pareto_ax.bar(x_pos, pos_1_data, width, bottom=0, color=contributor_colors['pos_1'], 
                  alpha=0.7, label=pos_1_label, edgecolor='white', linewidth=0.5, hatch='\\\\\\')
    
    # Negative contributions (stack downwards from 0)
    pareto_ax.bar(x_pos, neg_1_data, width, bottom=0, color=contributor_colors['neg_1'], 
                  alpha=0.7, label=neg_1_label, edgecolor='white', linewidth=0.5, hatch='\\\\\\')
    pareto_ax.bar(x_pos, neg_2_data, width, bottom=neg_1_data, color=contributor_colors['neg_2'], 
                  alpha=0.7, label=neg_2_label, edgecolor='white', linewidth=0.5, hatch='///')
    pareto_ax.bar(x_pos, neg_rest_data, width, bottom=neg_1_data + neg_2_data, 
                  color=contributor_colors['neg_rest'], alpha=0.7, label='Other benefits', 
                  edgecolor='white', linewidth=0.5)
    
    # Formatting
    pareto_ax.set_ylabel('Impact\n(kg CO₂-eq)', fontsize=10)
    pareto_ax.grid(True, alpha=0.25)
    pareto_ax.tick_params(axis='y', labelsize=8)
    
    # Add vertical line to separate base case from lambda values
    if include_base_case:
        pareto_ax.axvline(x=0.5, color='darkgray', linestyle='--', alpha=0.6, linewidth=1.5, zorder=5)
    
    # Add vertical lines to distinguish lambda ranges
    for vline_lambda in vlines:
        start_idx = 1 if include_base_case else 0
        closest_idx = min(range(start_idx, len(lambda_vals)), key=lambda i: abs(lambda_vals[i] - vline_lambda))
        if abs(lambda_vals[closest_idx] - vline_lambda) < 0.005:
            pareto_ax.axvline(x=closest_idx, color='black', linestyle='-', alpha=0.7, linewidth=1)
    
    # Add optional significant red horizontal line
    if hline_y is not None:
        # Main red line (also add to legend as 'Planetary boundary')
        pareto_ax.axhline(y=hline_y, color='darkred', linestyle='-', alpha=0.8, linewidth=2.5, zorder=6, label='Climate PB')
        # Add "Planetary boundary" label in dark red above the line
        pareto_ax.text(len(x_pos)*0.15, hline_y + 0.45*abs(hline_y) if hline_y != 0 else hline_y + 2, 
                      'Climate PB emission limit\n(EU ammonia supply)', color='darkred', fontsize=10, fontweight='normal', 
                      verticalalignment='bottom', horizontalalignment='center', zorder=7)
    
    # Legend for Pareto plot
    pareto_legend = pareto_ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                                     fontsize=8, frameon=True, fancybox=False, shadow=False)
    pareto_legend.get_frame().set_facecolor('white')
    pareto_legend.get_frame().set_alpha(0.9)
    
    # Plot 2+: Choice bar charts using your color palette
    color_schemes = {
        # 0: Teal Gradient (Base: #355C63)
        0: ["#5C8A91", "#1F3B40", "#C4E0E5"], 
        # 1: Orange Gradient (Base: #D99A3C)
        1: ["#E8B668", "#8C5E1A", "#FAEAC7"], 
        # 2: Red Gradient (Base: #A53D31)
        2: ["#C75E52", "#6B221A", "#F2BDB6"], 
        # 3: Burgundy Gradient (Base: #4E1512)
        3: ["#7A2E2A", "#2E0C0A", "#D18580"], 
        # 4: Purple Gradient (Distinct New Color)
        4: ["#83538C", "#361F40", "#D6B0DE"], 
        # 5: Green Gradient (Distinct New Color)
        5: ["#65915C", "#23401F", "#C7E5C4"], 
        # 6: Slate/Grey Gradient (Neutral Distinct Color)
        6: ["#6E7A8A", "#2B2F36", "#CED4DE"], 
    }
    
    for i, (choice_name, choice_data) in enumerate(choices_data.items()):
        ax = axs[i + 1]
        
        # Select color scheme for this choice (cycle through available schemes)
        colors = color_schemes.get(i % len(color_schemes), color_schemes[0])
        
        # Plot stacked bars with choice-specific colors
        bottom = np.zeros(len(x_pos))
        for j, (tech_name, row) in enumerate(choice_data.iterrows()):
            color = colors[j % len(colors)]
            values = row.values
            
            # Get abbreviated label if available
            label = str(tech_name)
            if legend_abbreviations and label in legend_abbreviations:
                label = legend_abbreviations[label]
            
            # Add hatching for second and third entries
            hatch = None
            if j == 1:  # Second entry
                hatch = '///'
            elif j == 2:  # Third entry
                hatch = '...'
            
            # Plot bars with absolute values
            ax.bar(x_pos, values, 0.8, bottom=bottom, 
                   color=color, label=label, alpha=0.85, 
                   edgecolor='white', linewidth=0.5, hatch=hatch)
            bottom += values
        
        # Plot total line
        totals = choice_data.sum().values
        if len(totals) > 0:
            ax.plot(x_pos, totals, "o-", markersize=3, linewidth=1.5, alpha=0.7, 
                   color='gray', markeredgecolor='black', markeredgewidth=0.5)
        
        # Labels and formatting
        ax.set_ylabel(choice_name, rotation=0, labelpad=30, va='center', fontsize=9)
        ax.set_yticks([])
        
        # Add vertical line to separate base case from lambda values
        if include_base_case:
            ax.axvline(x=0.5, color='darkgray', linestyle='--', alpha=0.6, linewidth=1.5, zorder=5)
        
        # Add vertical lines to distinguish lambda ranges
        for vline_lambda in vlines:
            start_idx = 1 if include_base_case else 0
            closest_idx = min(range(start_idx, len(lambda_vals)), key=lambda i: abs(lambda_vals[i] - vline_lambda))
            if abs(lambda_vals[closest_idx] - vline_lambda) < 0.005:
                ax.axvline(x=closest_idx, color='black', linestyle='-', alpha=0.7, linewidth=1)
        
        # Add individual legend for this choice
        legend = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                          fontsize=8, frameon=True, fancybox=False, shadow=False)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
    
    # Set x-axis on all subplots
    for ax in axs:
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)
        # Shift base case label to the left if present
        if include_base_case and len(ax.get_xticklabels()) > 0:
            ax.get_xticklabels()[0].set_horizontalalignment('right')
    axs[-1].set_xlabel('Reliability level (λ)', fontsize=10)
    
    plt.subplots_adjust(hspace=0.1)
    plt.tight_layout()
    
    # Save with bbox_inches='tight' to include legends
    os.makedirs(results_dir, exist_ok=True)
    
    # Create filename with suffix if provided
    base_filename = 'pareto_and_choices_combined'
    if suffix:
        filename = f'{base_filename}{suffix}'
    else:
        filename = base_filename
    
    # Save as PNG
    png_path = os.path.join(results_dir, f'{filename}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    # Save as SVG
    svg_path = os.path.join(results_dir, f'{filename}.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    plt.show()
    
    return {'png': png_path, 'svg': svg_path}


def plot_gsa_pie_chart(gsa_df, results_dir='data/results', figsize=(6, 6), save_plot=True, show_plot=True, lambda_value=None):
    """
    Create a pie chart visualization of Global Sensitivity Analysis results.
    
    Parameters:
    -----------
    gsa_df : pd.DataFrame
        Processed GSA DataFrame from process_gsa_results function
    results_dir : str
        Directory to save the plot (default: current directory)
    figsize : tuple
        Figure size as (width, height) (default: (6, 6))
    save_plot : bool
        Whether to save the plot to file (default: True)
    show_plot : bool
        Whether to display the plot (default: True)
    lambda_value : float, optional
        Lambda value to include in filename (default: None)
        
    Returns:
    --------
    dict
        Dictionary with paths to saved plot files
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data for pie chart
    labels = gsa_df['Short_Label'].tolist()
    sizes = gsa_df['ST'].tolist()
    
    # Use consistent color scheme from existing plots
    consistent_colors = [
        "#5C8A91",   # Teal (main)
        "#E8B668",   # Orange (main) 
        "#C75E52",   # Red (main)
        "#7A2E2A",   # Burgundy (main)
        "#83538C",   # Purple (main)
        "#65915C"    # Green (main)
    ]

    # Extend colors if needed
    colors = consistent_colors * ((len(labels) // len(consistent_colors)) + 1)
    colors = colors[:len(labels)]
    
    # Create smaller pie chart with black borders and slight explosion
    explode = [0.02] * len(sizes)  # Slight explosion for all slices
    wedges, texts = ax.pie(sizes, colors=colors, startangle=90, radius=0.35, 
                           center=(0, -0.05), explode=explode,
                           wedgeprops={'edgecolor': 'black', 'linewidth': 1.0})
    
    # Add percentage labels only for top 2 values
    # Sort to get the indices of the top 2 values
    sorted_indices = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)
    top_2_indices = sorted_indices[:2]
    
    for i, (wedge, size) in enumerate(zip(wedges, sizes)):
        if i in top_2_indices:  # Only show percentages for top 2
            angle = (wedge.theta2 + wedge.theta1) / 2
            # Adjust radius for smaller pie chart
            x = 0.20 * np.cos(np.radians(angle))
            y = -0.05 + 0.25 * np.sin(np.radians(angle))  # Account for center offset
            percentage = size / sum(sizes) * 100
            ax.text(x, y, f'{percentage:.1f}%', ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    
    # Create legend above the pie chart with 3 columns
    legend = ax.legend(wedges, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
                      ncol=3, fontsize=8, frameon=True, columnspacing=1.2)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    plt.tight_layout()
    
    # Save plot if requested
    save_paths = {}
    if save_plot:
        os.makedirs(results_dir, exist_ok=True)
        
        # Create filename with lambda suffix if provided
        if lambda_value is not None:
            lambda_suffix = f"_lambda_{lambda_value:.2f}"
        else:
            lambda_suffix = ""
        
        # Save as PNG
        png_path = os.path.join(results_dir, f'gsa_sensitivity_pie_chart{lambda_suffix}.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        save_paths['png'] = png_path
        
        # Save as SVG
        svg_path = os.path.join(results_dir, f'gsa_sensitivity_pie_chart{lambda_suffix}.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        save_paths['svg'] = svg_path
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return save_paths


def plot_gsa_bar_chart(total_Si, inv_map, proc_map, results_dir='data/results', figsize=(12, 6), save_plot=True, show_plot=True, lambda_value=None, top_n=9):
    """
    Create a bar chart visualization of GSA results with top contributions and variance whiskers.
    
    Parameters:
    -----------
    total_Si : pd.DataFrame
        GSA results dataframe with sensitivity indices (ST, ST_conf columns)
    inv_map : dict
        Intervention mapping dictionary
    proc_map : dict
        Process mapping dictionary
    results_dir : str
        Directory to save the plot (default: current directory)
    figsize : tuple
        Figure size as (width, height) (default: (12, 6))
    save_plot : bool
        Whether to save the plot to file (default: True)
    show_plot : bool
        Whether to display the plot (default: True)
    lambda_value : float, optional
        Lambda value to include in filename (default: None)
    top_n : int
        Number of top parameters to display (default: 9, will add "Others" as 10th)
        
    Returns:
    --------
    dict
        Dictionary with paths to saved plot files
    """
    # Sort by total sensitivity (ST) in descending order
    total_Si_sorted = total_Si.sort_values('ST', ascending=False)
    top_entries = total_Si_sorted.head(top_n)
    
    # Create labels and data lists
    labels = []
    st_values = []
    st_conf_values = []
    
    # Process top N entries
    for idx, row in top_entries.iterrows():
        if isinstance(idx, tuple) and len(idx) == 2:
            first_mapped = inv_map.get(idx[0], f"Unknown_inv_{idx[0]}")
            second_mapped = proc_map.get(idx[1], f"Unknown_proc_{idx[1]}")
            
            # Create shortened labels
            if "Carbon dioxide, in air" in first_mapped and "animal manure" in second_mapped:
                short_label = "CO₂ uptake\n(AD-Manure)"
            elif "Carbon dioxide, in air" in first_mapped and "agricultural residues" in second_mapped:
                short_label = "CO₂ uptake\n(AD-Agri)"
            elif "Carbon dioxide, non-fossil" in first_mapped and "animal manure" in second_mapped:
                short_label = "CO₂ emission\n(AD-Manure)"
            elif "Methane, non-fossil" in first_mapped and "animal manure" in second_mapped:
                short_label = "CH₄ emission\n(AD-Manure)"
            elif "Carbon dioxide, non-fossil" in first_mapped and "CCS 200km pipeline" in second_mapped:
                short_label = "CO₂\n(CCS pipeline)"
            elif "Methane, fossil" in first_mapped and "natural gas venting" in second_mapped:
                short_label = "CH₄\n(NG venting)"
            elif "Carbon dioxide, fossil" in first_mapped and "waste plastic" in second_mapped:
                short_label = "CO₂\n(Plastics incin.)"
            elif "Carbon dioxide, fossil" in first_mapped and "electricity production, hard coal" in second_mapped:
                short_label = "CO₂\n(coal elec.)"
            elif "Carbon dioxide, non-fossil" in first_mapped and "agricultural residues" in second_mapped:
                short_label = "CO₂\n(AD-Agri Res)"
            elif "Carbon dioxide" in first_mapped and ("sequential crop" in second_mapped.lower() or "sequential" in second_mapped.lower()):
                short_label = "CO₂\n(AD-Seq Crop)"
            elif "Methane, non-fossil" in first_mapped and "agricultural residues" in second_mapped:
                short_label = "CH₄\n(AD-Agri Res)"
            elif "Carbon dioxide" in first_mapped and "steam methane reforming" in second_mapped.lower() and "ccs" in second_mapped.lower():
                short_label = "CO₂\n(SMR-CCS)"
            elif "Carbon dioxide, fossil" in first_mapped and "electricity production" in second_mapped and "lignite" in second_mapped:
                short_label = "CO₂\n(lignite elec.)"
            elif "Carbon dioxide, fossil" in first_mapped and "transport" in second_mapped and "lorry" in second_mapped:
                short_label = "CO₂\n(transport)"
            elif "Carbon dioxide, fossil" in first_mapped and "natural gas" in second_mapped and "gas turbine" in second_mapped:
                short_label = "CO₂\n(NG turbine)"
            elif "Carbon dioxide" in first_mapped and "sawing" in second_mapped.lower():
                short_label = "CO₂\n(sawing)"
            else:
                intervention_short = first_mapped.split(",")[0]
                process_short = second_mapped.split("|")[0].strip()
                short_label = f"{intervention_short}\n({process_short})"
        else:
            mapped_index = inv_map.get(idx, f"Unknown_inv_{idx}")
            if "Methane, non-fossil" in mapped_index:
                short_label = "CH₄ emissions\n(CF)"
            elif "Methane, fossil" in mapped_index:
                short_label = "CH₄ emissions\n(CF)"
            else:
                short_label = mapped_index.split(",")[0]
        
        labels.append(short_label)
        st_values.append(row['ST'])
        st_conf_values.append(row.get('ST_conf', 0) if row.get('ST_conf') is not None else 0)
    
    # Add "Others" as the remaining contribution
    sum_top_st = sum(st_values)
    remaining_contribution = 1.0 - sum_top_st
    labels.append("Others")
    st_values.append(remaining_contribution)
    st_conf_values.append(0)  # No confidence interval for "Others"
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color scheme: use consistent colors with slight variation
    consistent_colors = [
        "#5C8A91",   # Teal (main)
        "#E8B668",   # Orange (main)
        "#C75E52",   # Red (main)
        "#7A2E2A",   # Burgundy (main)
        "#83538C",   # Purple (main)
        "#65915C",   # Green (main)
        "#6E7A8A",   # Slate (main)
        "#D4A574",   # Tan (main)
        "#8B6F47",   # Brown (main)
        "#B0B0B0",   # Gray (for Others)
    ]
    colors = consistent_colors[:len(labels)]
    
    # Create bar positions
    x_pos = np.arange(len(labels))
    
    # Plot bars with error bars (whiskers for variance)
    bars = ax.bar(x_pos, st_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
    
    # Add error bars (confidence intervals as whiskers)
    ax.errorbar(x_pos, st_values, yerr=st_conf_values, fmt='none', ecolor='black', 
                capsize=5, capthick=1.5, linewidth=1.5, zorder=3)
    
    # Formatting
    ax.set_ylabel('Total Sensitivity Index (ST)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Parameters', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(st_values) * 1.15)  # Add 15% padding for labels
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, st_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot if requested
    save_paths = {}
    if save_plot:
        os.makedirs(results_dir, exist_ok=True)
        
        # Create filename with lambda suffix if provided
        if lambda_value is not None:
            lambda_suffix = f"_lambda_{lambda_value:.2f}"
        else:
            lambda_suffix = ""
        
        # Save as PNG
        png_path = os.path.join(results_dir, f'gsa_sensitivity_bar_chart{lambda_suffix}.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        save_paths['png'] = png_path
        
        # Save as SVG
        svg_path = os.path.join(results_dir, f'gsa_sensitivity_bar_chart{lambda_suffix}.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        save_paths['svg'] = svg_path
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return save_paths


def plot_gsa_bar_chart_comparison(results_dict, inv_map, proc_map, results_dir='data/results', figsize=(14, 18), save_plot=True, show_plot=True, top_n=9):
    """
    Create a 4x1 subplot figure comparing GSA results across four lambda values with consistent colors and y-axis scaling.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with lambda values as keys and (total_Si, gsa_df) tuples as values
        Example: {0.50: (total_Si_05, gsa_df_05), 0.74: (total_Si_074, gsa_df_074), ...}
    inv_map : dict
        Intervention mapping dictionary
    proc_map : dict
        Process mapping dictionary
    results_dir : str
        Directory to save the plot (default: current directory)
    figsize : tuple
        Figure size as (width, height) (default: (16, 18))
    save_plot : bool
        Whether to save the plot to file (default: True)
    show_plot : bool
        Whether to display the plot (default: True)
    top_n : int
        Number of top parameters to display per subplot (default: 9)
        
    Returns:
    --------
    dict
        Dictionary with paths to saved plot files
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Extended color palette with harmonious additions
    consistent_colors = [
        "#5C8A91",   # Teal
        "#E8B668",   # Orange
        "#C75E52",   # Red
        "#7A2E2A",   # Burgundy
        "#83538C",   # Purple
        "#65915C",   # Green
        "#6E7A8A",   # Slate
        "#D4A574",   # Tan
        "#8B6F47",   # Brown
        "#4A7C8A",   # Deep Teal
        "#D68A4A",   # Burnt Orange
        "#A84E44",   # Dark Red
        "#9B6B98",   # Mauve
        "#88A67E",   # Sage Green
        "#B0B0B0",   # Gray (for Others)
    ]
    
    # Hatch patterns for when colors repeat (cycle through these)
    hatch_patterns = ['', '///', '\\\\\\', '...', 'xxx', '+++', '|||']
    
    # Helper function to get abbreviated label
    def get_label(idx, inv_map, proc_map):
        if isinstance(idx, tuple) and len(idx) == 2:
            first_mapped = inv_map.get(idx[0], f"Unknown_inv_{idx[0]}")
            second_mapped = proc_map.get(idx[1], f"Unknown_proc_{idx[1]}")
            
            # Create shortened labels (same logic as in plot_gsa_bar_chart)
            if "Carbon dioxide, in air" in first_mapped and "animal manure" in second_mapped:
                return "CO₂ uptake\n(AD-Manure)"
            elif "Carbon dioxide, in air" in first_mapped and "agricultural residues" in second_mapped:
                return "CO₂ uptake\n(AD-Agri)"
            elif "Carbon dioxide, non-fossil" in first_mapped and "animal manure" in second_mapped:
                return "CO₂ emission\n(AD-Manure)"
            elif "Methane, non-fossil" in first_mapped and "animal manure" in second_mapped:
                return "CH₄ emission\n(AD-Manure)"
            elif "Carbon dioxide, non-fossil" in first_mapped and "CCS 200km pipeline" in second_mapped:
                return "CO₂\n(CCS pipeline)"
            elif "Methane, fossil" in first_mapped and "natural gas venting" in second_mapped:
                return "CH₄\n(NG venting)"
            elif "Carbon dioxide, fossil" in first_mapped and "waste plastic" in second_mapped:
                return "CO₂\n(Plastics incin.)"
            elif "Carbon dioxide, fossil" in first_mapped and "electricity production, hard coal" in second_mapped:
                return "CO₂\n(coal elec.)"
            elif "Carbon dioxide, non-fossil" in first_mapped and "agricultural residues" in second_mapped and "sequential" not in second_mapped:
                return "CO₂\n(AD-Agri Res)"
            elif "Carbon dioxide" in first_mapped and ("sequential crop" in second_mapped.lower() or "sequential" in second_mapped.lower()):
                return "CO₂\n(AD-Seq Crop)"
            elif "Methane, non-fossil" in first_mapped and "agricultural residues" in second_mapped:
                return "CH₄\n(AD-Agri Res)"
            elif "Carbon dioxide" in first_mapped and "steam methane reforming" in second_mapped.lower() and "ccs" in second_mapped.lower():
                return "CO₂\n(SMR-CCS)"
            elif "Carbon dioxide, fossil" in first_mapped and "electricity production" in second_mapped and "lignite" in second_mapped:
                return "CO₂\n(lignite elec.)"
            elif "Carbon dioxide, fossil" in first_mapped and "transport" in second_mapped and "lorry" in second_mapped:
                return "CO₂\n(transport)"
            elif "Carbon dioxide, fossil" in first_mapped and "natural gas" in second_mapped and "gas turbine" in second_mapped:
                return "CO₂\n(NG turbine)"
            elif "Carbon dioxide" in first_mapped and "sawing" in second_mapped.lower():
                return "CO₂\n(sawing)"
            else:
                intervention_short = first_mapped.split(",")[0]
                process_short = second_mapped.split("|")[0].strip()
                return f"{intervention_short}\n({process_short})"
        else:
            # Handle non-tuple indices (single intervention/process)
            mapped_index = inv_map.get(idx, f"Unknown_inv_{idx}")
            if "Methane, non-fossil" in mapped_index:
                return "CH₄ emissions\n(CF)"
            elif "Methane, fossil" in mapped_index:
                return "CH₄ emissions\n(CF)"
            else:
                return mapped_index.split(",")[0]
    
    # Build a unified label set across all lambdas for consistent color mapping
    all_unique_labels = set()
    all_data = {}
    
    for lambda_val, (total_Si, _) in results_dict.items():
        total_Si_sorted = total_Si.sort_values('ST', ascending=False)
        top_entries = total_Si_sorted.head(top_n)
        
        labels = []
        st_values = []
        st_conf_values = []
        
        for idx, row in top_entries.iterrows():
            label = get_label(idx, inv_map, proc_map)
            labels.append(label)
            st_values.append(row['ST'])
            st_conf_values.append(row['ST_conf'] if 'ST_conf' in row.index else 0)
            all_unique_labels.add(label)
        
        # Add "Others" row
        remaining_st = total_Si_sorted.iloc[top_n:]['ST'].sum()
        labels.append("Others")
        st_values.append(remaining_st)
        st_conf_values.append(0)
        all_unique_labels.add("Others")
        
        all_data[lambda_val] = {
            'labels': labels,
            'st_values': st_values,
            'st_conf_values': st_conf_values
        }
    
    # Create a mapping from labels to colors (consistent across all subplots)
    sorted_unique_labels = sorted(list(all_unique_labels))
    label_to_color = {label: consistent_colors[i % len(consistent_colors)] 
                      for i, label in enumerate(sorted_unique_labels)}
    
    # Calculate global max for y-axis
    global_max = 0
    for lambda_val in all_data:
        st_values = all_data[lambda_val]['st_values']
        st_conf_values = all_data[lambda_val]['st_conf_values']
        max_with_whisker = max([st + conf for st, conf in zip(st_values, st_conf_values)])
        global_max = max(global_max, max_with_whisker)
    
    # Add 10% padding to global max
    global_max = global_max * 1.1
    
    # Create 4x1 subplot figure
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    
    # Sort lambda values for consistent subplot ordering
    lambda_vals = sorted(results_dict.keys())
    
    for plot_idx, lambda_val in enumerate(lambda_vals):
        ax = axes[plot_idx]
        data = all_data[lambda_val]
        labels = data['labels']
        st_values = data['st_values']
        st_conf_values = data['st_conf_values']
        
        # Assign colors based on consistent mapping
        colors = [label_to_color[label] for label in labels]
        
        # Assign hatch patterns based on label index to add visual distinction
        hatches = [hatch_patterns[sorted_unique_labels.index(label) // len(consistent_colors)] for label in labels]
        
        # Create bar positions
        x_pos = np.arange(len(labels))
        
        # Plot bars with error bars (whiskers for variance) and hatch patterns
        bars = ax.bar(x_pos, st_values, color=colors, alpha=0.8, edgecolor='black', 
                      linewidth=0.8, hatch=[hatches[i] for i in range(len(labels))])
        
        # Add error bars (confidence intervals as whiskers)
        ax.errorbar(x_pos, st_values, yerr=st_conf_values, fmt='none', ecolor='black', capsize=5, capthick=1.5, elinewidth=1)
        
        # Set uniform y-axis limits
        ax.set_ylim(0, global_max)
        
        # Customize subplot
        ax.set_ylabel('Sensitivity', fontsize=13)
        # Place lambda label in a boxed textbox at the upper-right of the subplot
        title_text = f"λ = {lambda_val:.2f}"
        ax.text(0.985, 0.9, title_text, transform=ax.transAxes, fontsize=14,
            ha='right', va='top', zorder=20,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.8))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', labelsize=11)
    
    plt.tight_layout()
    
    # Save plot if requested
    save_paths = {}
    if save_plot:
        os.makedirs(results_dir, exist_ok=True)
        
        # Save as PNG
        png_path = os.path.join(results_dir, 'gsa_sensitivity_bar_chart_comparison.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        save_paths['png'] = png_path
        
        # Save as SVG
        svg_path = os.path.join(results_dir, 'gsa_sensitivity_bar_chart_comparison.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        save_paths['svg'] = svg_path
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return save_paths


def print_gsa_summary(gsa_df, print_mapping_reference=True):
    """
    Print a formatted summary of GSA results.
    
    Parameters:
    -----------
    gsa_df : pd.DataFrame
        Processed GSA DataFrame from process_gsa_results function
    print_mapping_reference : bool
        Whether to print the detailed mapping reference (default: True)
    """
    # Calculate summary statistics
    sum_top_st = gsa_df[gsa_df['Original_Index'] != 'Others']['ST'].sum()
    remaining_contribution = gsa_df[gsa_df['Original_Index'] == 'Others']['ST'].iloc[0]
    
    print("Global Sensitivity Analysis Results - Top Parameters:")
    print("=" * 80)
    print(gsa_df[['Short_Label', 'ST', 'S1']].to_string(index=False))
    print("\n" + "=" * 80)
    print(f"Sum of top parameters ST values: {sum_top_st:.4f}")
    print(f"Remaining contribution: {remaining_contribution:.4f}")
    
    if print_mapping_reference:
        print("\nLabel Mapping Reference:")
        print("=" * 60)
        for _, row in gsa_df.iterrows():
            if row['Original_Index'] != 'Others':
                print(f"Short: {row['Short_Label']}")
                print(f"Full:  {row['Mapped_Index']}")
                print(f"ST:    {row['ST']:.3f}")
                print("-" * 60)