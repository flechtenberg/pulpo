import pyomo.environ as pyo
from pyomo.repn.plugins.baron_writer import *
import pandas as pd
from pathlib import Path
from IPython.display import display

def save_results(instance, project, database, choices, constraints, demand, map, elem_map, directory, name):
    """ TODO Imporve readability and structure ...
    There must be a better way to save the outputs of a pyomo model than this code I developed in 2020 """
    # Check if data/results folder exists, if not create it
    Path(directory + '/results').mkdir(parents=True, exist_ok=True)

    # Recover dictionary values
    list_of_vars = []
    for v in instance.component_objects(ctype=pyo.Var, active=True, descend_into=True):
        for e in v._data:
            v._data[e] = value(v[e])
        list_of_vars.append(v)

    # Open xlsx writer
    writer = pd.ExcelWriter(directory + '/results/' + name, engine='xlsxwriter')

    # Raw results
    for v in list_of_vars:
        try:
            data = [(k, map[k], v) for k, v in v._data.items()]
            df = pd.DataFrame(data, columns=['ID', 'Activity', 'Value'])
        except:
            data = [(k, v) for k, v in v._data.items()]
            df = pd.DataFrame(data, columns=['Key', 'Value'])
        df.sort_values(by=['Value'], inplace=True, ascending=False)
        df.to_excel(writer, sheet_name=v.name, index=False)

    # Store the metadata of the TCM
    pd.DataFrame([project + '__' + database]).to_excel(writer, sheet_name='project and db')

    metadata = {}
    for choice in choices:
        i = 0
        temp_dict = []
        for alt in choices[choice]:
            temp_dict.append((alt, i, instance.scaling_vector[map[alt.key]]))
            i+=1
        metadata[(choice, 'Activity')] = {'Activity ' + str(i): map[map[alt.key]] for alt, i, val in temp_dict}
        metadata[(choice, 'Capacity')] = {'Activity ' + str(i): choices[choice][alt] for alt, i, val in temp_dict}
        metadata[(choice, 'Value')] = {'Activity ' + str(i): x for alt, i, x in temp_dict}

    pd.DataFrame(metadata).to_excel(writer, sheet_name='choices')

    metadata = {}
    metadata['Demand'] = {map[map[key]]: demand[key] for key in demand}
    pd.DataFrame(metadata).to_excel(writer, sheet_name='demand')

    metadata = {}
    metadata['Demand'] = {map[map[key]]: constraints[key] for key in constraints}
    pd.DataFrame(metadata).to_excel(writer, sheet_name='constraints')

    # Save xlsx file
    writer.close()
    return

def summarize_results(instance, project, database, choices, constraints, demand, map, zeroes):
    metadata = {}
    metadata['Demand'] = {map[map[key]]: demand[key] for key in demand}
    print('The following demand / functional unit has been specified: ')
    display(pd.DataFrame(metadata))

    # Recover dictionary values
    list_of_vars = []
    for v in instance.component_objects(ctype=pyo.Var, active=True, descend_into=True):
        for e in v._data:
            v._data[e] = value(v[e])
        list_of_vars.append(v)

    # Raw results
    for v in list_of_vars:
        if v.name == 'impacts':
            try:
                data = [(k, map[k], v) for k, v in v._data.items()]
                df = pd.DataFrame(data, columns=['ID', 'Activity', 'Value'])
            except:
                data = [(k, v) for k, v in v._data.items()]
                df = pd.DataFrame(data, columns=['Key', 'Value'])
            df.sort_values(by=['Value'], inplace=True, ascending=False)
            print('\nThese are the impacts contained in the objective:')
            display(df)

        if v.name == 'impacts_calculated':
            try:
                data = [(k, map[k], v) for k, v in v._data.items()]
                df = pd.DataFrame(data, columns=['ID', 'Activity', 'Value'])
            except:
                data = [(k, v) for k, v in v._data.items()]
                df = pd.DataFrame(data, columns=['Key', 'Value'])
            df.sort_values(by=['Value'], inplace=True, ascending=False)
            print('\nThe following impacts were calculated: ')
            display(df)

    print('\nThe following choices were made: ')
    for choice in choices:
        metadata = {}
        i = 0
        temp_dict = []
        for alt in choices[choice]:
            if zeroes == False or instance.scaling_vector[map[alt.key]] != 0:
                temp_dict.append((alt, i, instance.scaling_vector[map[alt.key]]))
                i += 1
        metadata[(choice, 'Activity')] = {'Activity ' + str(i): map[map[alt.key]] for alt, i, val in temp_dict}
        metadata[(choice, 'Capacity')] = {'Activity ' + str(i): choices[choice][alt] for alt, i, val in temp_dict}
        metadata[(choice, 'Value')] = {'Activity ' + str(i): x for alt, i, x in temp_dict}
        print(choice)
        display(pd.DataFrame(metadata))

    if constraints == {}:
        print('No additional constraints have been passed.')
    else:
        metadata = {}
        metadata['Constraints'] = {map[map[key]]: constraints[key] for key in constraints}
        print('\nThe following constraints were implemented and oblieged: ')
        display(pd.DataFrame(metadata))

