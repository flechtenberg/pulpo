import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm, trange
from pulpo.utils import bw_parser

def pre_sample_lci_matrices(
    project,
    databases,
    method,
    intervention_matrix_name,
    n_samples=100,
    resample=("A", "B", "Q"),
    seed=None,
):
    """
    Sequentially sample randomized LCI data for Monte Carlo.
    Runs Brightway only once (thread-safe).
    Returns list of dicts with randomized A, B, Q matrices.
    """
    np.random.seed(seed)
    samples = []

    for i in trange(n_samples, desc="Sampling LCI matrices"):
        seed_i = np.random.randint(0, 1_000_000)
        lci_data_i = bw_parser.import_data(
            project=project,
            databases=databases,
            method=method,
            intervention_matrix_name=intervention_matrix_name,
            seed=seed_i,
            resample=resample,
        )
        # Keep only the matrices needed to solve the model
        samples.append({
            "technology_matrix": lci_data_i["technology_matrix"],
            "intervention_matrix": lci_data_i["intervention_matrix"],
            "matrices": lci_data_i["matrices"],
        })
    return samples


def solve_model_MC_pre_sampled(
    pulpo_optimizer,
    samples,
    GAMS_PATH=False,
    solver_name=None,
    options=None,
    n_jobs=-1,
):
    """
    Parallel Monte Carlo solve using pre-sampled LCI data (no Brightway).
    """
    print(f"Running {len(samples)} Monte Carlo optimizations in parallel (n_jobs={n_jobs})...")

    def _solve_single(i, sample):
        try:
            # Inject pre-sampled matrices
            lci_data = pulpo_optimizer.lci_data.copy()
            lci_data.update(sample)
            pulpo_optimizer.lci_data = lci_data

            pulpo_optimizer.instantiate(
                choices=pulpo_optimizer.choices,
                demand=pulpo_optimizer.demand,
                upper_limit=pulpo_optimizer.upper_limit,
                lower_limit=pulpo_optimizer.lower_limit,
                upper_elem_limit=pulpo_optimizer.upper_elem_limit,
                upper_imp_limit=pulpo_optimizer.upper_imp_limit,
            )
            pulpo_optimizer.solve(
                GAMS_PATH=GAMS_PATH,
                solver_name=solver_name,
                options=options,
            )
            return pulpo_optimizer.extract_results()
        except Exception as e:
            return {"error": str(e)}

    results = Parallel(n_jobs=n_jobs)(
        delayed(_solve_single)(i, sample)
        for i, sample in enumerate(tqdm(samples, desc="Monte Carlo solve"))
    )

    return {i: res for i, res in enumerate(results)}
