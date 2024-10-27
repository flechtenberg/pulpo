import copy
import numpy as np
np.NaN = np.nan
import bw2data as bd

# Helper function to convert integers to Latin letters
def int_to_latin(num):
    """Converts an integer to a Latin letter (e.g., 1 -> 'A', 2 -> 'B')."""
    return chr(64 + num)  # Converts 1 to 'A', 2 to 'B', etc.

# Helper function to convert integers to Greek letters
def int_to_greek(num):
    """Converts an integer to a Greek letter (e.g., 1 -> 'α', 2 -> 'β')."""
    greek_letters = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ']
    return greek_letters[num - 1] if num <= len(greek_letters) else str(num)

class Process:
    """Represents a production process for a specific product in a specific region."""

    def __init__(self, product_id, technology_id, region_id):
        """Initializes a process with product, technology, and region identifiers."""
        self.product_id = product_id
        self.technology_id = int_to_latin(technology_id)
        self.region_id = int_to_greek(region_id)
        self.process_id = f"{product_id}_{self.technology_id}_{self.region_id}"
        self.inputs = []  # List of tuples [(input_market_id, quantity), ...]
        self.environmental_flows = []  # List of tuples [(flow_id, quantity), ...]

    def add_input(self, input_market, quantity):
        """Adds an input market with a specified quantity to the process."""
        self.inputs.append((input_market.market_id, quantity))

    def add_environmental_flow(self, flow_id, quantity):
        """Adds an environmental flow with a specified quantity to the process."""
        self.environmental_flows.append((flow_id, quantity))

    def generate_inputs(self, potential_inputs, n_inputs):
        """Randomly generates and normalizes inputs for the process from potential markets."""
        if len(potential_inputs) < n_inputs:
            num_inputs = len(potential_inputs)
        else:
            num_inputs = np.random.randint(0, n_inputs + 1)

        if num_inputs > 0:
            selected_inputs = np.random.choice(potential_inputs, num_inputs, replace=False)

            # Generate random quantities for each selected input
            quantities = np.random.rand(num_inputs)

            # Normalize to ensure the sum is <= 1 (in order to have an M-matrix --- diagonal dominance)
            normalized_quantities = quantities / quantities.sum()

            # Scale down to ensure the sum is strictly smaller than 1
            scaling_factor = np.random.rand()  # A random factor between 0 and 1
            normalized_quantities *= scaling_factor

            for inp, quantity in zip(selected_inputs, normalized_quantities):
                self.add_input(inp, quantity)

    def generate_environmental_flows(self, flow_ids, n_flows):
        """Randomly generates environmental flows for the process."""
        num_flows = np.random.randint(1, n_flows + 1)
        selected_flows = np.random.choice(flow_ids, num_flows, replace=False)
        for flow_id in selected_flows:
            quantity = np.random.rand()
            self.add_environmental_flow(str(flow_id), quantity)

    def __repr__(self):
        return (f"Process(product_id={self.product_id}, "
                f"technology_id={self.technology_id}, "
                f"region_id={self.region_id}, "
                f"inputs={self.inputs}, "
                f"environmental_flows={self.environmental_flows})")

class Market:
    """Represents a market for a specific product in a specific region."""

    def __init__(self, product_id, region_id):
        """Initializes a market with product and region identifiers."""
        self.product_id = product_id
        self.region_id = int_to_greek(region_id)
        self.market_id = f"{product_id}_{self.region_id}"
        self.composition = []  # List of tuples [(process_id, share), ...]

    def add_process(self, process, share):
        """Adds a process with a specified share to the market composition."""
        self.composition.append((process.process_id, share))

    def generate_market_composition(self, candidate_processes):
        """Randomly generates a market composition from candidate processes."""
        included_processes = []
        for process in candidate_processes:
            if np.random.rand() < 0.75:
                share = np.random.rand()
                included_processes.append((process, share))

        # Ensure at least one process is included
        if not included_processes:
            process = np.random.choice(candidate_processes)
            share = np.random.rand()
            included_processes.append((process, share))

        for process, share in included_processes:
            self.add_process(process, share)

        # Normalize shares to sum to 1
        total_share = sum(share for _, share in self.composition)
        if total_share > 0:
            self.composition = [(pid, share / total_share) for pid, share in self.composition]

    def __repr__(self):
        return (f"Market(product_id={self.product_id}, "
                f"region_id={self.region_id}, "
                f"composition={self.composition})")

def create_system(n_prod, n_proc, n_reg, n_inputs, n_flows, seed=None):
    """
    Creates the technosphere and biosphere components for the system.

    Args:
        n_prod (int): Number of products.
        n_proc (int): Maximum number of processes per product.
        n_reg (int): Number of regions.
        n_inputs (int): Maximum number of inputs per process.
        n_flows (int): Number of environmental flows.
        seed (int, optional): Seed for reproducibility.

    Returns:
        processes (list): List of Process instances.
        markets (list): List of Market instances.
        flow_ids (list): List of environmental flow IDs.
    """
    if seed is not None:
        np.random.seed(seed)

    processes = []
    markets = []
    flow_ids = [f"e{i}" for i in range(1, n_flows + 1)]  # Environmental flow IDs: e1, e2, e3, ...

    # Step 1: Initialize Processes
    for product_id in range(1, n_prod + 1):
        regions = list(range(1, n_reg + 1))
        np.random.shuffle(regions)
        active_regions = regions[:np.random.randint(1, n_reg + 1)]

        for region_id in active_regions:
            technologies = list(range(1, n_proc + 1))
            np.random.shuffle(technologies)
            active_technologies = technologies[:np.random.randint(1, n_proc + 1)]

            for technology_id in active_technologies:
                proc = Process(product_id, technology_id, region_id)
                proc.generate_environmental_flows(flow_ids, n_flows)
                processes.append(proc)

    # Step 2: Generate Markets
    for product_id in range(1, n_prod + 1):
        regions = list(range(1, n_reg + 1))
        np.random.shuffle(regions)

        for region_id in regions:
            candidate_processes = [proc for proc in processes if proc.product_id == product_id and proc.region_id == int_to_greek(region_id)]
            if candidate_processes:
                market = Market(product_id, region_id)
                market.generate_market_composition(candidate_processes)
                markets.append(market)

    # Step 3: Assign Inputs to Processes
    for proc in processes:
        potential_inputs = [market for market in markets if market.region_id == proc.region_id]
        if potential_inputs:
            proc.generate_inputs(potential_inputs, n_inputs)

    return processes, markets, flow_ids

def assemble_technosphere_matrix(processes, markets):
    """
    Assembles the technosphere matrix from processes and markets.

    Args:
        processes (list): List of Process instances.
        markets (list): List of Market instances.

    Returns:
        np.ndarray: Technosphere matrix.
    """
    n = len(processes) + len(markets)
    A = np.zeros((n, n))

    process_index = {proc.process_id: idx for idx, proc in enumerate(processes)}
    market_index = {market.market_id: idx + len(processes) for idx, market in enumerate(markets)}

    # Fill in the technosphere matrix
    for proc in processes:
        i = process_index[proc.process_id]
        A[i, i] = 1  # Diagonal element for the process

        for market_id, qty in proc.inputs:
            j = market_index[market_id]
            A[j, i] = -qty  # Negative value for input

    for market in markets:
        i = market_index[market.market_id]
        A[i, i] = 1  # Diagonal element for the market

        for process_id, share in market.composition:
            j = process_index[process_id]
            A[j, i] = -share  # Market composition values

    return A

def assemble_biosphere_matrix(processes, flow_ids):
    """
    Assembles the biosphere matrix from processes and environmental flows.

    Args:
        processes (list): List of Process instances.
        flow_ids (list): List of environmental flow IDs.

    Returns:
        np.ndarray: Biosphere matrix.
    """
    n = len(processes) + len(flow_ids)
    B = np.zeros((len(flow_ids), len(processes)))

    process_index = {proc.process_id: idx for idx, proc in enumerate(processes)}
    flow_index = {flow_id: idx for idx, flow_id in enumerate(flow_ids)}

    # Fill in the biosphere matrix
    for proc in processes:
        j = process_index[proc.process_id]
        for flow_id, qty in proc.environmental_flows:
            i = flow_index[flow_id]
            B[i, j] = qty  # Positive value for environmental flow

    return B

class CharacterizationFactor:
    """Represents a characterization factor for an environmental flow under a specific method."""

    def __init__(self, method_id, flow_id, factor):
        """Initializes a characterization factor with a method, flow, and factor value."""
        self.method_id = method_id  # ID of the method (e.g., m1, m2, etc.)
        self.flow_id = flow_id      # ID of the environmental flow (e.g., e1, e2, etc.)
        self.factor = factor        # Characterization factor value

    def __repr__(self):
        return (f"CharacterizationFactor(method_id={self.method_id}, "
                f"flow_id={self.flow_id}, "
                f"factor={self.factor})")

def create_characterization_factors(n_methods, n_flows, seed=None):
    """
    Creates characterization factors for each method and environmental flow.

    Args:
        n_methods (int): Number of impact assessment methods.
        n_flows (int): Number of environmental flows.
        seed (int, optional): Seed for reproducibility.

    Returns:
        methods (list): List of method IDs.
        characterization_factors (list): List of CharacterizationFactor instances.
    """
    if seed is not None:
        np.random.seed(seed)

    methods = [f"m{i}" for i in range(1, n_methods + 1)]  # Method IDs: m1, m2, m3, ...
    characterization_factors = []

    for method_id in methods:
        num_factors = np.random.randint(1, n_flows + 1)  # Randomly determine the number of factors for this method
        selected_flows = np.random.choice([f"e{i}" for i in range(1, n_flows + 1)], num_factors, replace=False)
        for flow_id in selected_flows:
            factor = np.random.rand()  # Generate a random factor value
            characterization_factors.append(CharacterizationFactor(method_id, flow_id, factor))

    return methods, characterization_factors

def assemble_characterization_matrices(characterization_factors, flow_ids, methods):
    """
    Assembles characterization factor matrices for each method.

    Args:
        characterization_factors (list): List of CharacterizationFactor instances.
        flow_ids (list): List of environmental flow IDs.
        methods (list): List of method IDs.

    Returns:
        dict: Characterization matrices keyed by method ID.
    """
    n_methods = len(methods)
    n_flows = len(flow_ids)
    C_matrices = {method: np.zeros((n_flows, 1)) for method in methods}  # Initialize matrices

    flow_index = {flow_id: idx for idx, flow_id in enumerate(flow_ids)}

    for cf in characterization_factors:
        method_id = cf.method_id
        flow_id = cf.flow_id
        factor = cf.factor
        i = flow_index[flow_id]
        C_matrices[method_id][i, 0] = factor

    return C_matrices

def setup_biosphere_db(flow_ids):
    """Setup or update the biosphere database with environmental flows."""

    # Create biosphere database if it doesn't exist
    if 'biosphere3' not in bd.databases:
        biosphere_db = bd.Database('biosphere3')
        biosphere_data = {}

        # Define environmental flows
        for flow_id in flow_ids:
            biosphere_data[('biosphere3', flow_id)] = {
                'name': f'Environmental flow {flow_id}',
                'categories': ('environmental impact', ''),
                'type': 'emission',
                'unit': 'kg',
            }

        biosphere_db.write(biosphere_data)
        print('Biosphere database created')
    else:
        print('Biosphere database already exists')

def setup_technosphere_db(database, processes, markets):
    """Setup the technosphere database with generated processes and markets."""
    if 'generated_technosphere_db' not in bd.databases:
        technosphere_db = bd.Database(database)
        technosphere_db.write({})

        # Register each process
        for proc in processes:
            act = technosphere_db.new_activity(proc.process_id)
            act['unit'] = 'unit'  # Define the unit appropriately
            act['location'] = proc.region_id
            act['name'] = f'Process {proc.process_id}'
            act['reference product'] = f'Product {proc.product_id}'
            act.new_exchange(amount=1.0, input=act.key, type='production').save()
            act.save()

        # Register each market
        for market in markets:
            act = technosphere_db.new_activity(market.market_id)
            act['unit'] = 'unit'  # Define the unit appropriately
            act['location'] = market.region_id
            act['name'] = f'Market {market.market_id}'
            act['reference product'] = f'Product {market.product_id}'
            act.new_exchange(amount=1.0, input=act.key, type='production').save()
            act.save()

        print('Technosphere database created')
    else:
        print('Technosphere database already exists')

def add_exchanges_to_db(database, processes, markets):
    """Add exchanges to the technosphere and market activities in the database."""
    technosphere_db = bd.Database(database)
    biosphere_db = bd.Database('biosphere3')

    # Add exchanges for processes
    for proc in processes:
        act = technosphere_db.get(proc.process_id)

        # Add market inputs
        for market_id, qty in proc.inputs:
            market_act = technosphere_db.get(market_id)
            act.new_exchange(amount=qty, input=market_act.key, type='technosphere').save()

        # Add environmental flows (biosphere exchanges)
        for flow_id, qty in proc.environmental_flows:
            flow_act = biosphere_db.get(flow_id)
            act.new_exchange(amount=qty, input=flow_act.key, type='biosphere').save()

        act.save()

    # Add exchanges for markets
    for market in markets:
        act = technosphere_db.get(market.market_id)

        # Add process contributions to the market
        for process_id, share in market.composition:
            process_act = technosphere_db.get(process_id)
            act.new_exchange(amount=share, input=process_act.key, type='technosphere').save()

        act.save()

    print('Exchanges added to the technosphere and market activities in the database')

def setup_lcia_methods(characterization_factors, methods):
    """Setup LCIA methods based on generated characterization factors."""
    # Deregister existing methods
    methods_copy = copy.deepcopy(bd.methods)
    for method in methods_copy:
        bd.Method(method).deregister()

    # Register new LCIA methods
    for method_id in methods:
        method = bd.Method(('generated_system_example', method_id))
        method.register()

        cfs = []
        for cf in characterization_factors:
            if cf.method_id == method_id:
                flow_key = ('biosphere3', cf.flow_id)
                cfs.append((flow_key, cf.factor))

        method.write(cfs)

    print('LCIA methods and CFs defined')

def setup_generated_system(project, database, processes, markets, flow_ids, characterization_factors, methods):
    """Main function to set up the entire Brightway2 database."""
    bd.projects.set_current(project)
    setup_biosphere_db(flow_ids)
    setup_technosphere_db(database, processes, markets)
    add_exchanges_to_db(database, processes, markets)
    setup_lcia_methods(characterization_factors, methods)


def setup_generic_db(project, database, n_prod, n_proc, n_reg, n_inputs, n_flows, n_methods, seed=None,
                     return_data=False):
    """
    Sets up a generic LCI database in Brightway2 with specified parameters.

    Args:
        project (str): Name of the Brightway2 project to create or use.
        database (str): Name of the Brightway2 database to create or use.
        n_prod (int): Number of products to generate.
        n_proc (int): Maximum number of processes per product.
        n_reg (int): Number of regions where processes can be active.
        n_inputs (int): Maximum number of inputs per process.
        n_flows (int): Number of environmental flows to generate.
        n_methods (int): Number of impact assessment methods to create.
        seed (int, optional): Seed for reproducibility of random data generation.
        return_data (bool): If True, returns the generated matrices (technosphere, biosphere, and
            characterization).

    Returns:
        tuple: If `return_data` is True, returns a tuple containing:
            - technosphere_matrix (np.ndarray): The technosphere matrix.
            - biosphere_matrix (np.ndarray): The biosphere matrix.
            - characterization_matrices (dict): A dictionary of characterization factor matrices.
    """

    # Set the project, deleting existing databases if the project exists
    if project in bd.projects:
        bd.projects.set_current(project)
        print(f"Project '{project}' already exists. Deleting existing databases.")
        if database in bd.databases:
            del bd.databases[database]
        if 'biosphere3' in bd.databases:
            del bd.databases['biosphere3']
    else:
        bd.projects.set_current(project)
        print(f"Project '{project}' does not exist. Creating new project.")

    print("Proceeding with creating both technosphere and biosphere databases.")

    # Step 1: Create the technosphere and biosphere components
    processes, markets, flow_ids = create_system(n_prod, n_proc, n_reg, n_inputs, n_flows, seed=seed)

    # Step 2: Assemble the technosphere matrix
    technosphere_matrix = assemble_technosphere_matrix(processes, markets)

    # Step 3: Assemble the biosphere matrix
    biosphere_matrix = assemble_biosphere_matrix(processes, flow_ids)

    # Step 4: Create the characterization factors for impact assessment methods
    methods, characterization_factors = create_characterization_factors(n_methods, n_flows, seed=seed)

    # Step 5: Assemble the characterization factor matrices
    characterization_matrices = assemble_characterization_matrices(characterization_factors, flow_ids, methods)

    # Set up the Brightway2 database using the generated data
    setup_generated_system(project, database, processes, markets, flow_ids, characterization_factors, methods)

    if return_data:
        return technosphere_matrix, biosphere_matrix, characterization_matrices


def main():
    setup_generic_db(project="generic_db_project", database="generic_db", n_prod=5, n_proc=3, n_reg=3, n_inputs=4, n_flows=4, n_methods=2, seed=None, return_data=False)

if __name__ == '__main__':
    main()
