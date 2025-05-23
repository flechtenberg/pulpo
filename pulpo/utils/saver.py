import pandas as pd
import os
from typing import Dict, Any
from pyomo.environ import ConcreteModel


def extract_flows(instance: ConcreteModel, mapping: Dict[str, str], metadata: Dict[str, str], flow_type: str) -> pd.DataFrame:
    """
    Extracts scaling factors or inventory flows from a Pyomo model instance.
    """

    inverse_map = {v: k for k, v in mapping.items()}  # Reverse lookup
    
    # Select correct flow variable ('scaling' or 'intervention')
    flows = instance.scaling_vector if flow_type == 'scaling' else instance.inv_flows if flow_type == 'intervention' else None
    if flows is None:
        raise ValueError("Invalid flow_type. Use 'scaling' or 'intervention'.")

    # Retrieve data from flows
    data = {'ID': [], 'Key': [], 'Metadata': [], 'Value': []}
    for flow in flows:
        data['ID'].append(flow)
        data['Key'].append(inverse_map.get(flow, 'Unknown'))
        data['Metadata'].append(metadata.get(flow, 'No Metadata'))
        data['Value'].append(flows[flow].value)

    return pd.DataFrame(data).set_index('ID').sort_values('Value', ascending=False)


def extract_slack(instance: ConcreteModel) -> pd.DataFrame:
    """
    Extracts and sorts slack values from a Pyomo model.
    """
    ...
    
    return pd.DataFrame(
    {'Value': [v.value for v in instance.slack.values()]},  # Extract .value from each Pyomo variable
    index=instance.slack.keys()
    ).sort_values('Value', ascending=False)


def extract_impacts(instance: ConcreteModel) -> pd.DataFrame:
    """
    Extracts impact values and corresponding weights from the Pyomo instance.
    """

    data:dict = {'Method': [], 'Weight': [], 'Value': []}

    for i in instance.impacts.keys():
        data['Method'].append(i)
        data['Weight'].append(instance.WEIGHTS[i].value if i in instance.WEIGHTS and instance.WEIGHTS[i].value is not None else 0)
        data['Value'].append(instance.impacts[i].value if instance.impacts[i] is not None else 0)

    # Create the DataFrame
    df = pd.DataFrame(data).set_index('Method')

    # Sort by 'Weight' (descending) and then by 'Method' (alphabetically)
    return df.sort_values(by=['Weight', 'Method'], ascending=[False, True])

def extract_choices(instance: ConcreteModel, choices: Dict[str, Dict[Any, float]], process_map: Dict[str, str], process_map_metadata: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Extracts choice results from a Pyomo model and structures them into DataFrames.
    """

    results = {}
    for choice, processes in choices.items():
        data:dict = {
            "Value": [],
            "Capacity": [],
            "Metadata": []
        }
        for process, capacity in processes.items():
            proc_id = process_map.get(process.key)
            if proc_id is None:
                continue
            data["Metadata"].append(process_map_metadata.get(proc_id, "No Metadata"))
            data["Value"].append(instance.scaling_vector[proc_id].value)
            data["Capacity"].append(capacity)

        results[choice] = pd.DataFrame(data).set_index("Metadata").sort_values("Value", ascending=False)
    
    return results


def extract_demand(demand: Dict[Any, float]) -> pd.DataFrame:
    """
    Converts demand data into a structured DataFrame.
    Supports both Brightway Activities and string keys.
    """
    data = [
        {
            "Reference Product": e.get("reference product", "Unknown") if hasattr(e, "get") else str(e),
            "Activity Name": e.get("name", "Unknown") if hasattr(e, "get") else str(e),
            "Location": e.get("location", "Unknown") if hasattr(e, "get") else "Unknown",
            "Value": v
        }
        for e, v in demand.items()
    ]

    return pd.DataFrame(data).set_index(["Reference Product", "Activity Name", "Location"])


def extract_constraints(instance: ConcreteModel, constraints: Dict[Any, float], mapping: Dict[str, str], metadata: Dict[str, str], constraint_type: str) -> pd.DataFrame:
    """
    Extracts scaling factors or inventory flows associated with constraints from a Pyomo model instance.
    """

    inverse_map = {v: k for k, v in mapping.items()}  # Reverse lookup
    
    # Select correct flow variable ('scaling' or 'intervention')
    flows = instance.scaling_vector if constraint_type == 'scaling' else instance.inv_flows if constraint_type == 'intervention' else None
    if flows is None:
        raise ValueError("Invalid flow_type. Use 'scaling' or 'intervention'.")

    # Retrieve data from flows
    data:dict = {'ID': [], 'Key': [], 'Metadata': [], 'Value': [], 'Limit': []}
    for constraint in constraints:
        flow = mapping.get(constraint.key)
        data['ID'].append(flow)
        data['Key'].append(inverse_map.get(flow, 'Unknown'))
        data['Metadata'].append(metadata.get(flow, 'No Metadata'))
        data['Value'].append(flows[flow].value)
        data['Limit'].append(constraints.get(constraint, 'No Limit'))

    return pd.DataFrame(data).set_index('ID').sort_values('Value', ascending=False)


def extract_results(worker: Any) -> Dict[str, Any]:
    """
    Extracts results from the Pyomo model instance and organizes them into a structured format. Calls all other extract functions.
    """
    # Extract common data
    instance = worker.instance
    lci = worker.lci_data
    proc_map, proc_map_meta = lci['process_map'], lci['process_map_metadata']
    interv_map, interv_map_meta = lci['intervention_map'], lci['intervention_map_metadata']

    # Extract choices once for reuse
    choices_dict = extract_choices(instance, worker.choices, proc_map, proc_map_meta)
    
    # Build result data dictionary
    result_data = {
        "Scaling Vector": extract_flows(instance, proc_map, proc_map_meta, 'scaling'),
        "Intervention Vector": extract_flows(instance, interv_map, interv_map_meta, 'intervention'),
        "Slack": extract_slack(instance),
        "Impacts": extract_impacts(instance),
        "Demand": extract_demand(worker.demand),
        "Choices": choices_dict,
        "Constraints Upper": extract_constraints(instance, worker.upper_limit, proc_map, proc_map_meta, 'scaling'),
        "Constraints Lower": extract_constraints(instance, worker.lower_limit, proc_map, proc_map_meta, 'scaling'),
        "Constraints Upper Elem": extract_constraints(instance, worker.upper_elem_limit, interv_map, interv_map_meta, 'intervention')
    }

    return result_data

def save_results(worker: Any, file_name: str) -> None:
    """
    Saves worker/result data to an Excel file with multiple sheets.
    """
    result_data = extract_results(worker)
    choices_dict = result_data.pop("Choices")  # Extract choices separately

    # Prepare output path
    output_dir = os.path.dirname(file_name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save data to Excel
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        # Write the aggregated "Choices" sheet
        combined_choices = []
        for choice_name, df in choices_dict.items():
            divider = pd.DataFrame([[choice_name] + [None] * (len(df.columns) - 1)], columns=df.columns)
            combined_choices.append(divider)
            df_with_index = df.reset_index()
            df_with_index.insert(0, "Original Index", df.index)
            combined_choices.append(df_with_index)
        combined_choices_df = pd.concat(combined_choices, ignore_index=True)
        combined_choices_df.to_excel(writer, sheet_name="Choices", index=False)

        # Write other sheets
        for sheet_name, df in result_data.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name)

    print(f"Results saved to {file_name}")


def summarize_results(worker: Any, zeroes: bool = False) -> None:
    """
    Displays a summary of worker data in the console/Jupyter.
    Only the total impacts, the choices made, and the constraints (if any) are shown.
    """

    try:
        from IPython.display import display, Markdown
    except ImportError:
        display = print
        Markdown = lambda x: x

    # Extract the data
    result_data = extract_results(worker)

    # Helper to filter a dataframe if it has a 'Value' column
    def filter_nonzero(df):
        return df[df["Value"] != 0] if zeroes and "Value" in df.columns else df

    # 1. Display Total Impact(s)
    impacts = result_data.get("Impacts")
    if impacts is not None:
        impacts = filter_nonzero(impacts)
        display(Markdown("## Total Impact(s)"))
        display(impacts)
    else:
        display(Markdown("## Total Impact(s): No data found"))

    # 2. Display Choices Made
    choices_dict = result_data.get("Choices", {})
    display(Markdown("## Choices Made"))
    if choices_dict:
        for choice_name, df in choices_dict.items():
            df = filter_nonzero(df)
            display(Markdown(f"### {choice_name}"))
            display(df)
    else:
        display("No choices data available.")

    # 3. Display Constraints (if any)
    constraint_keys = ["Constraints Upper", "Constraints Lower", "Constraints Upper Elem"]
    display(Markdown("## Constraints"))
    constraints_found = False
    for key in constraint_keys:
        df = result_data.get(key)
        if df is not None:
            df = filter_nonzero(df)
            if not df.empty:
                display(Markdown(f"### {key}"))
                display(df)
                constraints_found = True

    if not constraints_found:
        display("No constraint data to display.")