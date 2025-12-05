import os
import pandas as pd

def get_ecoinvent_credentials():
    """Get ecoinvent credentials from environment variables or manual specification."""
    # Try environment variables first
    username = os.getenv("ECOINVENT_USERNAME")
    password = os.getenv("ECOINVENT_PASSWORD")
    
    # Fallback to manual specification (uncomment lines above if needed)
    if not username or not password:
        try:
            return globals()['username'], globals()['password']
        except KeyError:
            print("⚠️  Please uncomment and set username/password variables above")
            return None, None
    
    return username, password

def process_gsa_results(total_Si, inv_map, proc_map, top_n=5):
    """
    Process Global Sensitivity Analysis results with shortened labels for visualization.
    
    Parameters:
    -----------
    total_Si : pd.DataFrame
        GSA results dataframe with sensitivity indices
    inv_map : dict
        Intervention mapping dictionary
    proc_map : dict
        Process mapping dictionary
    top_n : int
        Number of top parameters to include (default: 5)
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with original indices, full mapped names, short labels, and sensitivity values
    """
    # Sort by total sensitivity (ST) in descending order
    total_Si_sorted = total_Si.sort_values('ST', ascending=False)
    top_entries = total_Si_sorted.head(top_n)
    
    # Create new DataFrame with mapped indices and short labels
    gsa_data = []
    for idx, row in top_entries.iterrows():
        # Check if index is a tuple (two entries) or single entry
        if isinstance(idx, tuple) and len(idx) == 2:
            # Apply inv_map to first entry, proc_map to second entry
            first_mapped = inv_map.get(idx[0], f"Unknown_inv_{idx[0]}")
            second_mapped = proc_map.get(idx[1], f"Unknown_proc_{idx[1]}")
            mapped_index = f"{first_mapped} | {second_mapped}"
            
            # Create shortened labels for pie charts
            if "Carbon dioxide, in air" in first_mapped and "animal manure" in second_mapped:
                short_label = "CO₂ uptake (AD-Manure)"
            elif "Carbon dioxide, in air" in first_mapped and "agricultural residues" in second_mapped:
                short_label = "CO₂ uptake (AD-Agri)"
            elif "Carbon dioxide, non-fossil" in first_mapped and "animal manure" in second_mapped:
                short_label = "CO₂ emission (AD-Manure)"
            elif "Methane, non-fossil" in first_mapped and "animal manure" in second_mapped:
                short_label = "CH₄ emission (AD-Manure)"
            elif "Carbon dioxide, non-fossil" in first_mapped and "CCS 200km pipeline" in second_mapped:
                short_label = "CO₂ (CCS pipeline)"
            elif "Methane, fossil" in first_mapped and "natural gas venting" in second_mapped:
                short_label = "CH₄ (NG venting)"
            elif "Carbon dioxide, fossil" in first_mapped and "waste plastic" in second_mapped:
                short_label = "CO₂ (Plastics incineration)"
            elif "Carbon dioxide, fossil" in first_mapped and "electricity production, hard coal" in second_mapped:
                short_label = "CO₂ (coal electricity)"
            else:
                # Generic shortening logic
                intervention_short = first_mapped.split(",")[0]  # Take first part before comma
                process_short = second_mapped.split("|")[0].strip()  # Take process name only
                short_label = f"{intervention_short} ({process_short})"
        else:
            # Single entry - apply only inv_map
            mapped_index = inv_map.get(idx, f"Unknown_inv_{idx}")
            if "Methane, non-fossil" in mapped_index:
                short_label = "CH₄ emissions (CF)"
            elif "Methane, fossil" in mapped_index:
                short_label = "CH₄ emissions (CF)"
            else:
                short_label = mapped_index.split(",")[0]  # Take first part before comma
        
        gsa_data.append({
            'Original_Index': idx,
            'Mapped_Index': mapped_index,
            'Short_Label': short_label,
            'ST': row['ST'],
            'S1': row.get('S1', None),
            'S1_conf': row.get('S1_conf', None),
            'ST_conf': row.get('ST_conf', None)
        })
    
    gsa_df = pd.DataFrame(gsa_data)
    
    # Add remaining contribution (1 - sum of top N ST values)
    sum_top_st = gsa_df['ST'].sum()
    remaining_contribution = 1.0 - sum_top_st
    
    # Add a row for the remaining contribution
    remaining_row = {
        'Original_Index': 'Others',
        'Mapped_Index': 'All other parameters combined',
        'Short_Label': 'Others',
        'ST': remaining_contribution,
        'S1': None,
        'S1_conf': None,
        'ST_conf': None
    }
    gsa_df = pd.concat([gsa_df, pd.DataFrame([remaining_row])], ignore_index=True)
    
    return gsa_df, sum_top_st, remaining_contribution