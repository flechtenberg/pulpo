import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_pareto_from_results(results_CC, process_map_metadata=None, cutoff_value=0.01, results_dir='.', lambda_range:tuple=None, show_choices:bool=True):
    """
    Streamlined plotting: treat Pareto as the top subplot showing impact cumsum,
    followed by choice bar charts showing technology cumsum. No PIL needed.
    """
    import os
    import matplotlib.pyplot as plt

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
    
    # Extract impacts for Pareto
    impacts = []
    for k in ordered_keys:
        res = selected[k]
        if 'Impacts' in res and not res['Impacts'].empty:
            impacts.append(float(res['Impacts'].iloc[0, 1]))
        else:
            impacts.append(0)
    
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

    # Create single figure with all subplots - increased Pareto height
    n_choices = len(choices_data)
    n_total = 1 + n_choices  # Pareto + choices
    
    # Height ratios: give more space to Pareto, compact for choices
    height_ratios = [1.5] + [0.4] * n_choices  # Pareto gets 2.5x height, choices get 0.6x
    
    fig, axs = plt.subplots(n_total, 1, figsize=(4, sum(height_ratios)), 
                           sharex=True, gridspec_kw={'height_ratios': height_ratios})
    if n_total == 1:
        axs = [axs]
    
    # Common x-axis setup
    x_pos = np.arange(len(ordered_keys))
    lambda_vals = [float(k) for k in ordered_keys]
    xtick_positions = [i for i, lam in enumerate(lambda_vals) if abs(round(lam * 10) - lam * 10) < 1e-6]
    if not xtick_positions:
        xtick_positions = [0, len(x_pos)-1]
    xtick_labels = [f"{lambda_vals[i]:.1f}" for i in xtick_positions]
    
    # Plot 1: Pareto (impact line)
    axs[0].plot(x_pos, impacts, marker='o', linestyle='-', color='#355C63', linewidth=2, markersize=6)
    axs[0].set_ylabel('Impact\n(kg CO₂-eq)', fontsize=10)
    axs[0].grid(True, alpha=0.25)
    axs[0].tick_params(axis='y', labelsize=8)
    
    # Plot 2+: Choice bar charts using your color palette mixed up
    # Using your original colors: ["#355C63", "#D99A3C", "#A53D31", "#4E1512"]
    base_colors = ["#355C63", "#D99A3C", "#A53D31", "#4E1512"]
    
    # Different combinations and arrangements of your colors for each choice category
    color_schemes = {
        0: ["#355C63", "#D99A3C", "#A53D31", "#4E1512"],  # Original order
        1: ["#D99A3C", "#A53D31", "#355C63", "#4E1512"],  # Start with orange
        2: ["#A53D31", "#4E1512", "#D99A3C", "#355C63"],  # Start with red
        3: ["#4E1512", "#355C63", "#D99A3C", "#A53D31"],  # Start with dark red
        4: ["#355C63", "#A53D31", "#D99A3C", "#4E1512"],  # Teal-red-orange-dark
        5: ["#D99A3C", "#4E1512", "#355C63", "#A53D31"],  # Orange-dark-teal-red
        6: ["#A53D31", "#355C63", "#4E1512", "#D99A3C"],  # Red-teal-dark-orange
        7: ["#4E1512", "#D99A3C", "#A53D31", "#355C63"],  # Dark-orange-red-teal
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
            
            # Plot bars with absolute values
            ax.bar(x_pos, values, 0.8, bottom=bottom, 
                   color=color, label=str(tech_name), alpha=0.85, 
                   edgecolor='white', linewidth=0.5)
            bottom += values
        
        # Plot total line
        totals = choice_data.sum().values
        if len(totals) > 0:
            ax.plot(x_pos, totals, "k-", markersize=2, linewidth=1, alpha=0.7)
        
        # Labels and formatting
        ax.set_ylabel(choice_name, rotation=0, labelpad=30, va='center', fontsize=9)
        ax.set_yticks([])
    
    # Set x-axis only on bottom subplot
    axs[-1].set_xticks(xtick_positions)
    axs[-1].set_xticklabels(xtick_labels)
    axs[-1].set_xlabel('Risk-aversion level (λ)', fontsize=10)
    
    plt.subplots_adjust(hspace=0.1)
    plt.tight_layout()
    
    # Save
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'pareto_and_choices_combined.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return {'combined': save_path}
