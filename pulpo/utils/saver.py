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
        directory (str): Directory to save the results file.
        name (str): Name of the results file.
    """
    # Recover dictionary values
    list_of_vars = []
    for v in instance.component_objects(ctype=pyo.Var, active=True, descend_into=True):
        for e in v._data:
            v._data[e] = value(v[e])
        list_of_vars.append(v)

    result_data = {}
    inverse_process_map = dict((v, k) for k, v in process_map.items())
    inverse_itervention_map = dict((v, k) for k, v in itervention_map.items())
    # Raw results
    for v in list_of_vars:
        try:
            if str(v) == 'inv_flows' or str(v) == 'inv_vector':
                data = [(k, inverse_itervention_map[k], itervention_map_metadata[k], v) for k, v in v._data.items()]
            else:
                data = [(k, inverse_process_map[k], process_map_metadata[k], v) for k, v in v._data.items()]
            df = pd.DataFrame(data, columns=['ID', 'Process name', "Process metadata", 'Value'])
        except:
            data = [(k, v) for k, v in v._data.items()]
            df = pd.DataFrame(data, columns=['Key', 'Value'])
        df.sort_values(by=['Value'], inplace=True, ascending=False)
        result_data[v.name] = df

    # Normalize database to a list if it is a string
    if isinstance(database, str):
        database = [database]

    # Store the metadata
    result_data["project and db"] = pd.DataFrame([f"{project}__{db}" for db in database])

    choices_data = {}
    for choice in choices:
        i = 0
        temp_dict = []
        for alt in choices[choice]:
            temp_dict.append((alt, i, instance.scaling_vector[process_map[alt.key]]))
            i+=1
        choices_data[(choice, 'Process')] = {'Process ' + str(i): process_map_metadata[process_map[alt.key]] for alt, i, val in temp_dict}
        choices_data[(choice, 'Capacity')] = {'Process ' + str(i): choices[choice][alt] for alt, i, val in temp_dict}
        choices_data[(choice, 'Value')] = {'Process ' + str(i): x for alt, i, x in temp_dict}
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

