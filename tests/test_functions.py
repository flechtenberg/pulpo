from pulpo import pulpo

test_project = 'Gasification'
test_database = 'cutoff38'
#test_method1 = "('IPCC 2013', 'climate change', 'GWP 100a')"
test_method2 = "('CML 2001 (superseded)', 'terrestrial ecotoxicity', 'TAETP infinite')"
directory = r'C:\Users\Usuario\Documents\PhD Documents\Python\pulpo\data'
GAMS_PATH = r'C:\GAMS\37\gams.exe'

activities = ["market for hydrogen, liquid"]
reference_products = ["hydrogen, liquid"]
locations = ["RER"]

pulpo_worker = pulpo.PulpoOptimizer(test_project, test_database, test_method2, directory)

# demand
hydrogen_market = pulpo_worker.retrieve_activities(activities=activities, reference_products=reference_products, locations=locations)
demand = {hydrogen_market[0]: 100}

# choices
activities = ["chlor-alkali electrolysis, diaphragm cell",
             "chlor-alkali electrolysis, membrane cell",
             "chlor-alkali electrolysis, mercury cell",
             "hydrogen cracking, APME"]
reference_products = ["hydrogen, liquid"]
locations = ["RER"]

hydrogen_activities = pulpo_worker.retrieve_activities(activities=activities, reference_products=reference_products, locations=locations)
choices = {'hydrogen': {hydrogen_activities[0]: 10000,
                                hydrogen_activities[1]: 10000,
                                hydrogen_activities[2]: 10000,
                                hydrogen_activities[3]: 10000}}

# supply
supply = {}

# constraints
constraints = {}

# objective weighting
#weights = {test_method1: 1, test_method2: 0}

# Get LCI data
lci_data = pulpo_worker.get_lci_data()
# Instantiate
instance = pulpo_worker.instantiate(choices=choices, demand=demand)
# Solve
results = pulpo_worker.solve(GAMS_PATH)
