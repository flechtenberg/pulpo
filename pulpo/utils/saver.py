import pyomo.environ as pyo
from pyomo.repn.plugins.baron_writer import *
import pandas as pd
from pathlib import Path
from IPython.display import display

def extract_results(instance, project, database, choices, constraints, demand, process_map, process_map_metadata, itervention_map, itervention_map_metadata):
    """
    Args:
        instance: The Pyomo model instance.
        project (str): Name of the project.
        database (str): Name of the database.
        choices (dict): Choices for the model.
        constraints (dict): Constraints applied during optimization.
        demand (dict): Demand data used in optimization.
        process_map (dict): Mapping of process IDs to descriptions.
        process_map_metadata (dict): Metadata to the process_map
        itervention_map (dict): Mapping of intervention IDs to descriptions.
        itervention_map_metadata (dict): Metadata of the itervention_map.
    """
    result_data = {}
    process_map_df = pd.DataFrame.from_dict(process_map, orient='index', columns=['process_id']).reset_index(names='Process name').set_index('process_id')
    process_map_df['Process metadata'] = process_map_df.index.map(process_map_metadata)
    itervention_map_df = pd.DataFrame.from_dict(itervention_map, orient='index', columns=['intervention_id']).reset_index(names='Intervention name').set_index('intervention_id')
    itervention_map_df['Invervention metadata'] = itervention_map_df.index.map(itervention_map_metadata)
    # Raw results
    for v in instance.component_objects(ctype=pyo.Var, active=True, descend_into=True):
        df = pd.DataFrame.from_dict(v.get_values(), orient='index', columns=['Value']).sort_values('Value', ascending=False)
        match v.name:
            case 'inv_flows' | 'inv_vector':
                df = df.join(itervention_map_df, how='left').reset_index(names='ID')          
            case 'scaling_vector':
                df = df.join(process_map_df, how='left').reset_index(names='ID')
            case 'impacts' | 'slack' | 'impacts_calculated':
                df = df.reset_index(names='Key')
        result_data[v.name] = df

    # Store the emvironmental costs
    result_data[instance.ENV_COST_MATRIX.name] = pd.DataFrame.from_dict(
        instance.ENV_COST_MATRIX.extract_values(), orient='index', 
        columns=[str(instance.ENV_COST_MATRIX)]
        )


    # Normalize database to a list if it is a string
    if isinstance(database, str):
        database = [database]

    # Store the metadata
    result_data["project and db"] = pd.DataFrame([f"{project}__{db}" for db in database])

    # ATTN: BHL: This needs to be rewritten, it is very convoluted and can be more clear
    choices_data = {}
    for choice, alternatives in choices.items():
        temp_dict = []
        for i_alt, alt in enumerate(alternatives):
            temp_dict.append((alt, i_alt, instance.scaling_vector.get_values()[process_map[alt.key]]))
        choices_data[(choice, 'Process')] = {f'Process {i}': process_map_metadata[process_map[alt.key]] for alt, i, _ in temp_dict}
        choices_data[(choice, 'Capacity')] = {f'Process {i}': alternatives[alt] for alt, i, val in temp_dict}
        choices_data[(choice, 'Value')] = {f'Process {i}': x for _, i, x in temp_dict}
    result_data["choices"] = pd.DataFrame(choices_data)

    result_data["demand"] = pd.DataFrame({"demand":{
        process_map_metadata[process_map[key]] if key in process_map else key: demand[key]
        for key in demand
    }})
    result_data["constraints"] = pd.DataFrame({"Demand": {process_map_metadata[process_map[key]]: constraints[key] for key in constraints}})

    return result_data


def save_results(result_data, directory, file_name):
    with pd.ExcelWriter(f"{directory}/results/{file_name}.xlsx") as writer:
        for sheet_name, dataframe in result_data.items():
            dataframe.to_excel(writer, sheet_name=sheet_name)


def _display_dataframe(df):
    """
    Helper function to display a DataFrame if it is not empty.

    Args:
        df (pd.DataFrame): The DataFrame to display.
    """
    if not df.empty:
        display(df)
    else:
        print('No data available.')


def summarize_results(results_data, zeroes):
    """
    Summarizes the results of the optimization and prints them to the console.

    Args:
        results_data (dict): Dictionary containing extracted results, assumed to be precomputed.
        zeroes (bool): Whether to include zero values in the summary.
    """
    # Display demand
    print('\nThe following demand / functional unit has been specified:')
    _display_dataframe(results_data.get("demand", pd.DataFrame()))

    # Display constraints (if any)
    if "constraints" in results_data and not results_data["constraints"].empty:
        print('\nThe following constraints were implemented and obliged:')
        _display_dataframe(results_data["constraints"])
    else:
        print('No additional constraints have been passed.')

    # Display impact results (if they exist)
    for impact_name in ['impacts', 'impacts_calculated']:
        if impact_name in results_data and not results_data[impact_name].empty:
            print(f'\n{impact_name.replace("_", " ").title()} contained in the objective:')
            _display_dataframe(results_data[impact_name])

    # Display choices
    if "choices" in results_data and not results_data["choices"].empty:
        print('\nThe following choices were made:')
        display(results_data["choices"])
    else:
        print('\nNo choices were recorded.')

