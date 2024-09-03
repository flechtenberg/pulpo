import brightway2 as bw
import copy
import numpy as np

np.NaN = np.nan  # Ensures compatibility with the latest NumPy versions

def setup_rice_husk_db():
    # Set the current project to "rice_husk_example"
    bw.projects.set_current("rice_husk_example")

    # Biosphere keys
    co2_key = ('biosphere3', 'CO2')
    ch4_key = ('biosphere3', 'CH4')
    ef_key = ('biosphere3', 'EF')

    # Define biosphere flows and create the 'biosphere3' database if it doesn't exist
    if 'biosphere3' not in bw.databases:
        biosphere_db = bw.Database('biosphere3')
        biosphere_data = {
            ('biosphere3', 'CO2'): {
                'name': 'Carbon dioxide, fossil',
                'categories': ('climate change', 'GWP 100a'),
                'type': 'emission',
                'unit': 'Mt',
            },
            ('biosphere3', 'CH4'): {
                'name': 'Methane, agricultural',
                'categories': ('climate change', 'GWP 100a'),
                'type': 'emission',
                'unit': 'Mt',
            },
            ('biosphere3', 'EF'): {
                'name': 'Economic Flow',
                'categories': ('economic flow', 'dollar'),
                'type': 'emission',
                'unit': 'million dollar',
            },
        }
        biosphere_db.write(biosphere_data)

        print('Elementary flow database created')

    # Technosphere keys
    rice_factory_key = ('rice_husk_example_db', 'Rice factory')
    rice_farming_key = ('rice_husk_example_db', 'Rice farming')
    rice_husk_boiler_key = ('rice_husk_example_db', 'Rice husk boiler')
    natural_gas_boiler_key = ('rice_husk_example_db', 'Natural gas boiler')
    wood_pellet_boiler_key = ('rice_husk_example_db', 'Wood pellet boiler')
    rice_husk_market_key = ('rice_husk_example_db', 'Rice husk market')
    rice_husk_collection1_key = ('rice_husk_example_db', 'Rice husk collection 1')
    rice_husk_collection2_key = ('rice_husk_example_db', 'Rice husk collection 2')
    rice_husk_collection3_key = ('rice_husk_example_db', 'Rice husk collection 3')
    rice_husk_collection4_key = ('rice_husk_example_db', 'Rice husk collection 4')
    rice_husk_collection5_key = ('rice_husk_example_db', 'Rice husk collection 5')
    natural_gas_supply_key = ('rice_husk_example_db', 'Natural gas supply')
    wood_pellet_supply_key = ('rice_husk_example_db', 'Wood pellet supply')
    burning_rice_husk_key = ('rice_husk_example_db', 'Burning of rice husk')
    power_plant_key = ('rice_husk_example_db', 'Power plant')
    transportation_key = ('rice_husk_example_db', 'Transportation by truck')

    # Create the "rice_husk_example_db" database if it doesn't exist
    if 'rice_husk_example_db' not in bw.databases:
        rice_husk_db = bw.Database('rice_husk_example_db')
        rice_husk_db.write({})

        process_data = [
            ('Rice factory', 'Mt', 'GLO', 'Processed rice (in Mt)'),
            ('Rice farming', 'Mt', 'GLO', 'Unprocessed rice (in Mt)'),
            ('Rice husk boiler', 'TWh', 'GLO', 'Thermal energy from husk boiler (in TWh)'),
            ('Natural gas boiler', 'TWh', 'GLO', 'Thermal energy from natural gas boiler (in TWh)'),
            ('Wood pellet boiler', 'TWh', 'GLO', 'Thermal energy from wood pellet boiler (in TWh)'),
            ('Rice husk market', 'Mt', 'GLO', 'Rice husk at farm (Mt)'),
            ('Rice husk collection 1', 'Mt', 'GLO', 'Rice husk from region 1 (in Mt)'),
            ('Rice husk collection 2', 'Mt', 'GLO', 'Rice husk from region 2 (in Mt)'),
            ('Rice husk collection 3', 'Mt', 'GLO', 'Rice husk from region 3 (in Mt)'),
            ('Rice husk collection 4', 'Mt', 'GLO', 'Rice husk from region 4 (in Mt)'),
            ('Rice husk collection 5', 'Mt', 'GLO', 'Rice husk from region 5 (in Mt)'),
            ('Natural gas supply', 'TWh', 'GLO', 'Natural gas (in TWh)'),
            ('Wood pellet supply', 'Mt', 'GLO', 'Wood pellets (in Mt)'),
            ('Burning of rice husk', 'Mt', 'GLO', 'Burned rice husk (in Mt)'),
            ('Power plant', 'TWh', 'GLO', 'Electricity (in TWh)'),
            ('Transportation by truck', 'Gt*km', 'GLO', 'Transportation (in Gt*km)'),
        ]

        # Register each process
        for name, unit, location, ref_product in process_data:
            act = rice_husk_db.new_activity(name)
            act['unit'] = unit
            act['location'] = location
            act['name'] = name
            act['reference product'] = ref_product
            act.new_exchange(amount=1.0, input=act.key, type='production').save()
            act.save()

        # Define the technosphere exchanges
        exchange_data = [
            # Format: [input_key, target_key, amount, exchange_type]
            [rice_farming_key, rice_factory_key, 1.15, 'technosphere'],
            [natural_gas_boiler_key, rice_factory_key, 2.2, 'technosphere'],
            [power_plant_key, rice_factory_key, 0.08, 'technosphere'],
            [transportation_key, rice_factory_key, 0.35, 'technosphere'],
            [rice_husk_market_key, rice_farming_key, -0.6, 'technosphere'],
            [rice_husk_collection1_key, rice_husk_boiler_key, 0.23, 'technosphere'],
            [natural_gas_supply_key, natural_gas_boiler_key, 1.1, 'technosphere'],
            [wood_pellet_supply_key, wood_pellet_boiler_key, 0.25, 'technosphere'],
            [rice_husk_market_key, rice_husk_collection1_key, 1, 'technosphere'],
            [rice_husk_market_key, rice_husk_collection2_key, 1, 'technosphere'],
            [rice_husk_market_key, rice_husk_collection3_key, 1, 'technosphere'],
            [rice_husk_market_key, rice_husk_collection4_key, 1, 'technosphere'],
            [rice_husk_market_key, rice_husk_collection5_key, 1, 'technosphere'],
            [transportation_key, rice_husk_collection1_key, 0.12, 'technosphere'],
            [transportation_key, rice_husk_collection2_key, 0.24, 'technosphere'],
            [transportation_key, rice_husk_collection3_key, 0.36, 'technosphere'],
            [transportation_key, rice_husk_collection4_key, 0.48, 'technosphere'],
            [transportation_key, rice_husk_collection5_key, 0.60, 'technosphere'],
            [rice_husk_market_key, burning_rice_husk_key, 2, 'technosphere'],
            [burning_rice_husk_key, rice_husk_market_key, 1, 'technosphere'],
            [co2_key, rice_farming_key, 6.14e-1, 'biosphere'],
            [ch4_key, rice_farming_key, 1.33e-3, 'biosphere'],
            [co2_key, natural_gas_boiler_key, 2.27e-1, 'biosphere'],
            [ch4_key, natural_gas_boiler_key, 1.47e-3, 'biosphere'],
            [co2_key, natural_gas_supply_key, 3.21e-2, 'biosphere'],
            [ch4_key, natural_gas_supply_key, 1.50e-3, 'biosphere'],
            [co2_key, wood_pellet_supply_key, 1.50e-1, 'biosphere'],
            [ch4_key, wood_pellet_supply_key, 2.56e-4, 'biosphere'],
            [co2_key, power_plant_key, 1.10e-0, 'biosphere'],
            [ch4_key, power_plant_key, 9.15e-4, 'biosphere'],
            [co2_key, transportation_key, 5.76e-2, 'biosphere'],
            [ch4_key, transportation_key, 6.97e-5, 'biosphere'],
            [ef_key, rice_factory_key, 5.00e1, 'biosphere'],
            [ef_key, rice_farming_key, 3.60e2, 'biosphere'],
            [ef_key, natural_gas_supply_key, 1.3e1, 'biosphere'],
            [ef_key, wood_pellet_supply_key, 7.20e1, 'biosphere'],
            [ef_key, rice_husk_collection1_key, 4.50e1, 'biosphere'],
            [ef_key, rice_husk_collection2_key, 3.60e1, 'biosphere'],
            [ef_key, rice_husk_collection3_key, 2.90e1, 'biosphere'],
            [ef_key, rice_husk_collection4_key, 2.30e1, 'biosphere'],
            [ef_key, rice_husk_collection5_key, 1.80e1, 'biosphere'],
            [ef_key, power_plant_key, 6.50e1, 'biosphere'],
            [ef_key, transportation_key, 1.70e2, 'biosphere'],
        ]

        # Add the exchanges to the activities
        for input_key, target_key, amount, exchange_type in exchange_data:
            act = [act for act in rice_husk_db if act.key == target_key][0]
            act.new_exchange(amount=amount, input=input_key, type=exchange_type).save()
            act.save()

        print('Process database created')

        # Loop through the list of methods and deregister each one
        # Create a copy of the methods list
        methods_copy = copy.deepcopy(bw.methods)
        # Loop through the copy of methods and deregister each one
        for method in methods_copy:
            bw.Method(method).deregister()

        # Define LCIA methods and CFs
        methods_data = [
            ('climate change', 'Mt CO2eq', 2, 'cc', 'climate change CFs', 'climate_change', 'CO2',
             [(co2_key, 1), (ch4_key, 25)]),
            ('air quality', 'ppm', 1, 'aq', 'air quality CFs', 'air_quality', 'PM', [(ch4_key, 25)]),
            ('economic flow', 'million dollar', 1, 'ef', 'economic flow CFs', 'economic_flow', 'million dollar', [(ef_key, 1)]),
        ]

        for method_name, unit, num_cfs, abbreviation, description, filename, flow_code, flow_list in methods_data:
            method = bw.Method(('my project', method_name))
            method.register(**{
                'unit': unit,
                'num_cfs': num_cfs,
                'abbreviation': abbreviation,
                'description': description,
                'filename': filename,
            })
            method.write(flow_list)

        print('LCIA methods and CFs defined')

def main():
    setup_rice_husk_db()

if __name__ == '__main__':
    main()
