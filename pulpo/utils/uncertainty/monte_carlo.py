"""
monte_carlo.py (uncertainty)

Monte Carlo variant driven by prepared uncertainty distributions (no Brightway
resampling). Depends on the uncertainty sub-package, so only loaded when
uncertainty features are used.
"""

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from pulpo.utils.uncertainty import processor


def _apply_draw_to_lci(lci_data: dict, method: str, draw: dict) -> dict:
    """
    Return a shallow copy of lci_data with B and Q updated by 'draw'.
    Only touch the entries that changed.
    """
    lci2 = lci_data.copy()
    # --- Update B (intervention_matrix)
    if draw["If"]:
        B = lci2["intervention_matrix"].tolil(copy=True)
        for (g, j), val in draw["If"].items():
            B[g, j] = val
        lci2["intervention_matrix"] = B.tocsr()

    # --- Update Q for the selected method
    if draw["Cf"]:
        Q = lci2["matrices"][method].tolil(copy=True)
        for g, val in draw["Cf"].items():
            Q[g, g] = val
        # keep other methods intact
        matrices2 = dict(lci2["matrices"])
        matrices2[method] = Q.tocsr()
        lci2["matrices"] = matrices2

    return lci2


def pre_sample_from_uncertainty(
    pulpo_optimizer,
    n_samples: int,
    seed: int | None = None,
):
    """
    Build a list of 'overlays' (lci + var_bounds) directly from prepared uncertainty distributions.
    """
    if pulpo_optimizer.uncertainty_data is None:
        raise ValueError("No uncertainty_data found on the optimizer.")
    method = next(iter(pulpo_optimizer.method))

    # Reproducible per-iteration seeds
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1_000_000, size=n_samples, endpoint=False)

    overlays = []
    for s in tqdm(seeds, desc="Sampling uncertainty draws"):
        draw = processor.draw_uncertainty_sample(
            pulpo_optimizer.uncertainty_data, method=method, seed=int(s)
        )
        lci_overlay = _apply_draw_to_lci(pulpo_optimizer.lci_data, method, draw)
        overlays.append({
            "lci_data": lci_overlay,
            "var_bounds": draw["Var_bounds"],  # applied later on the instance
        })
    return overlays


def solve_model_MC_pre_sampled_uncertainty(
    pulpo_optimizer,
    overlays,
    GAMS_PATH=False,
    solver_name=None,
    options=None,
    n_jobs=-1,
):
    """
    Parallel Monte Carlo solve using overlays created from uncertainty draws (no Brightway).
    """
    print(f"Running {len(overlays)} Monte Carlo optimizations in parallel (n_jobs={n_jobs}).")

    def _solve_single(i, overlay):
        try:
            # Inject overlayed LCI
            lci_data = pulpo_optimizer.lci_data.copy()
            lci_data.update(overlay["lci_data"])
            pulpo_optimizer.lci_data = lci_data

            # Fresh instantiate with the usual user-specified dicts
            pulpo_optimizer.instantiate(
                choices=pulpo_optimizer.choices,
                demand=pulpo_optimizer.demand,
                upper_limit=pulpo_optimizer.upper_limit,
                lower_limit=pulpo_optimizer.lower_limit,
                upper_elem_limit=pulpo_optimizer.upper_elem_limit,
                upper_imp_limit=pulpo_optimizer.upper_imp_limit,
            )

            # Apply sampled variable bounds directly to the mutable Pyomo Params
            vb = overlay["var_bounds"]
            inst = pulpo_optimizer.instance
            # PROCESS-level bounds
            for j, val in vb.get("upper_limit", {}).items():
                inst.UPPER_LIMIT[j] = float(val)
            for j, val in vb.get("lower_limit", {}).items():
                inst.LOWER_LIMIT[j] = float(val)
            # Flow- and impact-level bounds
            for g, val in vb.get("upper_elem_limit", {}).items():
                inst.UPPER_INV_LIMIT[g] = float(val)
            for h, val in vb.get("upper_imp_limit", {}).items():
                inst.UPPER_IMP_LIMIT[h] = float(val)

            pulpo_optimizer.solve(
                GAMS_PATH=GAMS_PATH,
                solver_name=solver_name,
                options=options,
            )
            return pulpo_optimizer.extract_results()
        except Exception as e:
            return {"error": str(e)}

    results = Parallel(n_jobs=n_jobs)(
        delayed(_solve_single)(i, overlay)
        for i, overlay in enumerate(tqdm(overlays, desc="Monte Carlo solve"))
    )
    return {i: res for i, res in enumerate(results)}
