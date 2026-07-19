import pandas as pd
import os
from pyomo.environ import ConcreteModel, Param
from typing import TypedDict, Dict, Any, Optional, List
import pandas as pd
from pulpo.utils.bw_parser import LCIDataDict
from pulpo.utils.utils import broadcast_over_time

class ResultDataDict(TypedDict, total=False):
    Scaling_Vector: pd.DataFrame
    Intervention_Vector: pd.DataFrame
    Slack: pd.DataFrame
    Impacts: pd.DataFrame
    Demand: pd.DataFrame
    Choices: Dict[str, pd.DataFrame]
    Constraints_Upper: pd.DataFrame
    Constraints_Lower: pd.DataFrame
    Constraints_Upper_Elem: pd.DataFrame
    # Optionally, add parameter DataFrames if extractparams=True
    # e.g. param_name: pd.DataFrame

def extract_flows(instance: ConcreteModel, mapping: Dict[str, str], metadata: Dict[str, str], flow_type: str) -> pd.DataFrame:
    """
    Extracts scaling factors or inventory flows from a Pyomo model instance.

    Transparently handles time-indexed instances: when the underlying Var is
    indexed by (t, id) tuples instead of a plain id, the result gains a 'Time'
    column/index level instead of the static single-'ID' shape.
    """

    inverse_map = {v: k for k, v in mapping.items()}  # Reverse lookup

    # Select correct flow variable ('scaling' or 'intervention')
    flows = instance.scaling_vector if flow_type == 'scaling' else instance.inv_flows if flow_type == 'intervention' else None
    if flows is None:
        raise ValueError("Invalid flow_type. Use 'scaling' or 'intervention'.")

    # Retrieve data from flows
    data = {'ID': [], 'Key': [], 'Metadata': [], 'Value': [], 'Time': []}
    time_indexed = False
    for flow in flows:
        if isinstance(flow, tuple):
            t, key = flow
            time_indexed = True
        else:
            t, key = None, flow
        data['ID'].append(key)
        data['Time'].append(t)
        data['Key'].append(inverse_map.get(key, 'Unknown'))
        data['Metadata'].append(metadata.get(key, 'No Metadata'))
        data['Value'].append(flows[flow].value)

    df = pd.DataFrame(data)
    if time_indexed:
        return df.set_index(['ID', 'Time']).sort_values('Value', ascending=False, kind='stable')
    return df.drop(columns='Time').set_index('ID').sort_values('Value', ascending=False, kind='stable')


def extract_slack(instance: ConcreteModel) -> pd.DataFrame:
    """
    Extracts and sorts slack values from a Pyomo model.

    Slack variables only exist for products with a specified supply
    (identical lower and upper limit), so the result contains one row per
    supply product and is empty when no supply is specified.
    """
    return pd.DataFrame(
    {'Value': [v.value for v in instance.slack.values()]},  # Extract .value from each Pyomo variable
    index=instance.slack.keys()
    ).sort_values('Value', ascending=False, kind='stable')


def extract_impacts(instance: ConcreteModel) -> pd.DataFrame:
    """
    Extracts impact values and corresponding weights from the Pyomo instance.

    Transparently handles time-indexed instances: WEIGHTS is never time-indexed,
    so the indicator component is looked up on its own even when instance.impacts
    is keyed by (t, indicator) tuples; the result gains a 'Time' column/index
    level in that case instead of the static single-'Method' shape.
    """

    data: dict = {'Method': [], 'Weight': [], 'Value': [], 'Time': []}
    time_indexed = False

    for i in instance.impacts.keys():
        if isinstance(i, tuple):
            t, h = i
            time_indexed = True
        else:
            t, h = None, i
        data['Method'].append(h)
        data['Time'].append(t)
        data['Weight'].append(instance.WEIGHTS[h].value if h in instance.WEIGHTS and instance.WEIGHTS[h].value is not None else 0)
        data['Value'].append(instance.impacts[i].value if instance.impacts[i] is not None else 0)

    df = pd.DataFrame(data)
    if time_indexed:
        # Create the DataFrame, sorted by 'Weight' (descending) and then by 'Method' (alphabetically)
        return df.set_index(['Method', 'Time']).sort_values(by=['Weight', 'Method'], ascending=[False, True])
    return df.drop(columns='Time').set_index('Method').sort_values(by=['Weight', 'Method'], ascending=[False, True])

def extract_transgressions(instance: ConcreteModel) -> pd.DataFrame:
    """
    Extracts the goal-programming results: per goal category the impact, the goal
    (soft limit), the transgression level TL = Impact/Goal, and the transgression
    slack max(0, TL - 1). Empty DataFrame when the instance has no goal categories
    (e.g. weighted-sum objective or time-indexed instances).
    """
    columns = ['Impact', 'Goal', 'TL', 'Transgression']
    if not hasattr(instance, 'GOAL_INDICATOR') or len(instance.GOAL_INDICATOR) == 0:
        return pd.DataFrame(columns=columns)

    data: dict = {'Method': [], 'Impact': [], 'Goal': [], 'TL': [], 'Transgression': []}
    for h in instance.GOAL_INDICATOR:
        impact = instance.impacts[h].value
        goal = instance.IMP_GOALS[h].value
        data['Method'].append(h)
        data['Impact'].append(impact)
        data['Goal'].append(goal)
        data['TL'].append(impact / goal if impact is not None else None)
        data['Transgression'].append(instance.transgression[h].value)
    return pd.DataFrame(data).set_index('Method').sort_values('Transgression', ascending=False, kind='stable')


def extract_choices(instance: ConcreteModel, choices: Dict[str, Dict[Any, float]], process_map: Dict[str, str], process_map_metadata: Dict[str, str], time_steps: Optional[List] = None) -> Dict[str, pd.DataFrame]:
    """
    Extracts choice results from a Pyomo model and structures them into DataFrames.

    When `time_steps` is given, `choices` may be static (broadcast across all
    timesteps) or already in `{t: {...}}` form; the result gains a 'Time' index level.
    """

    if time_steps is not None:
        choices_t = broadcast_over_time(choices, time_steps)
        choice_labels = {label for c in choices_t.values() for label in c}
        results = {}
        for choice in choice_labels:
            data: dict = {"Value": [], "Capacity": [], "Metadata": [], "Time": []}
            for t in time_steps:
                for process, capacity in choices_t[t].get(choice, {}).items():
                    proc_id = process_map.get(process.key)
                    if proc_id is None:
                        continue
                    data["Metadata"].append(process_map_metadata.get(proc_id, "No Metadata"))
                    data["Value"].append(instance.scaling_vector[t, proc_id].value)
                    data["Capacity"].append(capacity)
                    data["Time"].append(t)
            results[choice] = pd.DataFrame(data).set_index(["Metadata", "Time"]).sort_values("Value", ascending=False, kind='stable')
        return results

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

        results[choice] = pd.DataFrame(data).set_index("Metadata").sort_values("Value", ascending=False, kind='stable')

    return results


def extract_demand(demand: Dict[Any, float], time_steps: Optional[List] = None) -> pd.DataFrame:
    """
    Converts demand data into a structured DataFrame.
    Supports both Brightway Activities and string keys.

    When `time_steps` is given, `demand` may be static (broadcast across all
    timesteps) or already in `{t: {...}}` form; the result gains a 'Time' index level.
    """
    if time_steps is not None:
        demand_t = broadcast_over_time(demand, time_steps)
        data = [
            {
                "Reference Product": e.get("reference product", "Unknown") if hasattr(e, "get") else str(e),
                "Activity Name": e.get("name", "Unknown") if hasattr(e, "get") else str(e),
                "Location": e.get("location", "Unknown") if hasattr(e, "get") else "Unknown",
                "Time": t,
                "Value": v,
            }
            for t in time_steps
            for e, v in demand_t[t].items()
        ]
        return pd.DataFrame(data, columns=["Reference Product", "Activity Name", "Location", "Time", "Value"]).set_index(["Reference Product", "Activity Name", "Location", "Time"])

    data = [
        {
            "Reference Product": e.get("reference product", "Unknown") if hasattr(e, "get") else str(e),
            "Activity Name": e.get("name", "Unknown") if hasattr(e, "get") else str(e),
            "Location": e.get("location", "Unknown") if hasattr(e, "get") else "Unknown",
            "Value": v
        }
        for e, v in demand.items()
    ]

    # Explicit columns keep the frame well-formed when no demand is specified
    # (supply-driven runs), where data is empty.
    return pd.DataFrame(data, columns=["Reference Product", "Activity Name", "Location", "Value"]).set_index(["Reference Product", "Activity Name", "Location"])


def extract_constraints(instance: ConcreteModel, constraints: Dict[Any, float], mapping: Dict[str, str], metadata: Dict[str, str], constraint_type: str, time_steps: Optional[List] = None) -> pd.DataFrame:
    """
    Extracts scaling factors or inventory flows associated with constraints from a Pyomo model instance.

    When `time_steps` is given, `constraints` may be static (broadcast across all
    timesteps) or already in `{t: {...}}` form; the result gains a 'Time' index level.
    """

    inverse_map = {v: k for k, v in mapping.items()}  # Reverse lookup

    # Select correct flow variable ('scaling' or 'intervention')
    flows = instance.scaling_vector if constraint_type == 'scaling' else instance.inv_flows if constraint_type == 'intervention' else None
    if flows is None:
        raise ValueError("Invalid flow_type. Use 'scaling' or 'intervention'.")

    if time_steps is not None:
        constraints_t = broadcast_over_time(constraints, time_steps)
        data: dict = {'ID': [], 'Key': [], 'Metadata': [], 'Value': [], 'Limit': [], 'Time': []}
        for t in time_steps:
            for constraint, limit in constraints_t[t].items():
                flow = mapping.get(constraint.key)
                data['ID'].append(flow)
                data['Key'].append(inverse_map.get(flow, 'Unknown'))
                data['Metadata'].append(metadata.get(flow, 'No Metadata'))
                data['Value'].append(flows[t, flow].value)
                data['Limit'].append(limit)
                data['Time'].append(t)
        return pd.DataFrame(data).set_index(['ID', 'Time']).sort_values('Value', ascending=False, kind='stable')

    # Retrieve data from flows
    data:dict = {'ID': [], 'Key': [], 'Metadata': [], 'Value': [], 'Limit': []}
    for constraint in constraints:
        flow = mapping.get(constraint.key)
        data['ID'].append(flow)
        data['Key'].append(inverse_map.get(flow, 'Unknown'))
        data['Metadata'].append(metadata.get(flow, 'No Metadata'))
        data['Value'].append(flows[flow].value)
        data['Limit'].append(constraints.get(constraint, 'No Limit'))

    return pd.DataFrame(data).set_index('ID').sort_values('Value', ascending=False, kind='stable')

def extract_params(instance: ConcreteModel) -> Dict[str,pd.DataFrame]:
    """
    Extracts the Parameter values from the optimization model
    """
    data_all:dict = {}
    for param in instance.component_objects(ctype=Param):
        data = {}
        extracted_values = param.extract_values()
        data['ID'] = list(extracted_values.keys())
        data['Value'] = list(extracted_values.values())
        data_all[param.name] = pd.DataFrame(data).set_index('ID').sort_values('Value', ascending=False, kind='stable')
    # The environmental cost coefficients are embedded in the impact
    # constraints rather than stored as a Param; report them from the dense
    # dictionary kept on the instance so the result schema stays unchanged
    # (the CC Pareto plots read result_data['ENV_COST_MATRIX']).
    if hasattr(instance, '_env_cost'):
        data_all['ENV_COST_MATRIX'] = pd.DataFrame({
            'ID': list(instance._env_cost.keys()),
            'Value': list(instance._env_cost.values()),
        }).set_index('ID').sort_values('Value', ascending=False, kind='stable')
    return data_all

def extract_results(worker: Any, extractparams:bool=False) -> ResultDataDict:
    """
    Extracts results from the Pyomo model instance and organizes them into a structured format. Calls all other extract functions.
    """
    # Extract common data
    instance = worker.instance
    lci = worker.lci_data
    proc_map, proc_map_meta = lci['process_map'], lci['process_map_metadata']
    interv_map, interv_map_meta = lci['intervention_map'], lci['intervention_map_metadata']
    time_steps = getattr(worker, 'time_steps', None)

    # Extract choices once for reuse
    choices_dict = extract_choices(instance, worker.choices, proc_map, proc_map_meta, time_steps=time_steps)

    # Build result data dictionary
    result_data = {
        "Scaling Vector": extract_flows(instance, proc_map, proc_map_meta, 'scaling'),
        "Intervention Vector": extract_flows(instance, interv_map, interv_map_meta, 'intervention'),
        "Slack": extract_slack(instance),
        "Impacts": extract_impacts(instance),
        "Transgressions": extract_transgressions(instance),
        "Demand": extract_demand(worker.demand, time_steps=time_steps),
        "Choices": choices_dict,
        "Constraints Upper": extract_constraints(instance, worker.upper_limit, proc_map, proc_map_meta, 'scaling', time_steps=time_steps),
        "Constraints Lower": extract_constraints(instance, worker.lower_limit, proc_map, proc_map_meta, 'scaling', time_steps=time_steps),
        "Constraints Upper Elem": extract_constraints(instance, worker.upper_elem_limit, interv_map, interv_map_meta, 'intervention', time_steps=time_steps)
    }

    # Append the parameter values to it
    if extractparams:
        param_data = extract_params(instance)
        result_data.update(param_data)
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

    # 1b. Display Goal Transgressions (only when a goal-programming run)
    transgressions = result_data.get("Transgressions")
    if transgressions is not None and not transgressions.empty:
        display(Markdown("## Goal Transgressions"))
        display(transgressions)

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


# ATTN: This function is to be deleted later or integrated into another function
def compare_subsequent_paretosolutions(result_data_CC:Dict[float,LCIDataDict], choices:dict, method:str):
    """
    TO BE DELETED LATER OR INTEGRATED INTO ANOTHER FUNCTION 

    Compare impacts and decision choices across multiple Pareto solutions.

    Args:
        result_data_CC (dict of float to dict): Mapping from each lambda level
            to its corresponding solver result dictionary.
    """
    try:
        from IPython.display import display
    except ImportError:
        display = globals()['print']
    impacts = {}
    print(method)
    for lambda_QB, result_data in result_data_CC.items():
        impacts[lambda_QB] = result_data['Impacts'].loc[method,'Value']
        print('{}: {}'.format(lambda_QB, impacts[lambda_QB]))
    # The changs in the choices of the optimizer
    choices_results = {}
    for i_CC, (lambda_QB, result_data) in enumerate(result_data_CC.items()):
        for choice in choices.keys():
            if i_CC == 0:
                choices_results[choice] = result_data['Choices'][choice][['Capacity']]
            choices_results[choice] = choices_results[choice].join(result_data['Choices'][choice]['Value'].rename(lambda_QB), how='left')
    for choice, choice_result in choices_results.items():
        display(choice)
        display(choice_result)

    # # Changes in the scaling vector and the characterized and scaled inventories
    # lambda_array = list(result_data_CC.keys())
    # for lambda_1, lambda_2 in zip(lambda_array[:len(lambda_array)-1], lambda_array[1:len(lambda_array)]):
    #     print(f'lambda_1: {lambda_1}\nlambda_2: {lambda_2}\n')
    #     scaling_vector_diff = ((result_data_CC[lambda_1]['Scaling Vector']['Value'] - result_data_CC[lambda_2]['Scaling Vector']['Value']))
    #     scaling_vector_ratio = (scaling_vector_diff / result_data_CC[lambda_1]['Scaling Vector']['Value']).abs().sort_values(ascending=False)
    #     environmental_cost_mean = {env_cost_index[0]: env_cost['Value'] for env_cost_index, env_cost in result_data_CC[lambda_1]['ENV_COST_MATRIX'].iterrows()}
    #     characterized_scaling_vector_diff = (scaling_vector_diff * pd.Series(environmental_cost_mean).reindex(scaling_vector_diff.index)).abs()
    #     characterized_scaling_vector_diff_relative = (characterized_scaling_vector_diff / result_data_CC[lambda_1]['Impacts'].loc[method, 'Value']).abs().sort_values(ascending=False)

    #     print('Amount of process scaling variables that changed:\n{}: >1% \n{}: >10%\n{}: >100%\n{}: >1000%\n'.format((scaling_vector_ratio > 0.01).sum(), (scaling_vector_ratio > 0.1).sum(), (scaling_vector_ratio > 1).sum(), (scaling_vector_ratio > 10).sum()))
    #     print('Amount of process characterized scaling variables (impacts per process) that changed:\n{}: >1% \n{}: >10%\n{}: >100%\n{}: >1000%\n'.format((characterized_scaling_vector_diff_relative > 0.01).sum(), (characterized_scaling_vector_diff_relative > 0.1).sum(), (characterized_scaling_vector_diff_relative > 1).sum(), (characterized_scaling_vector_diff_relative > 10).sum()))
    #     print('{:.5e}: is the maximum impact change in one process\n{:.5e}: is the total impact change\n'.format(characterized_scaling_vector_diff_relative.max(), characterized_scaling_vector_diff_relative.sum()))

    #     amount_of_rows_for_visiualization = 10
    #     # print('The relative change of the scaling vector (s_lambda_1 - s_lambda_2)/s_lambda_1:\n')
    #     # display(scaling_vector_ratio.iloc[:amount_of_rows_for_visiualization].rename(result_data_CC[lambda_2]['Scaling Vector']['Metadata']).sort_values(ascending=False))
    #     # print('\n---\n')
    #     print('The relative change of the characterized scaling vector (s_lambda_1 - s_lambda_2)*QB_s / QBs:\n')
    #     display(characterized_scaling_vector_diff_relative.iloc[:amount_of_rows_for_visiualization].rename(result_data_CC[lambda_2]['Scaling Vector']['Metadata']))
    #     print('\n---\n')