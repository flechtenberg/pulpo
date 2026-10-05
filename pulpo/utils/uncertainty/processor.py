"""
processor.py

The sampler that draws one realization of every declared parameter.
"""

import numpy as np
import stats_arrays
from typing import Union, Dict, Tuple


def _merge_defined_blocks(unc_data: dict, top_key: str) -> Dict[Union[Tuple[int,int],int], dict]:
    """Collect & merge all 'defined' blocks under unc_data[top_key]."""
    out = {}
    for block in unc_data.get(top_key, {}).values():
        out.update(block.get('defined', {}))
    return out


def _sample_one_spec(spec: dict, rng: np.random.Generator) -> float:
    """
    Sample a single uncertainty spec. An exact value (type 1) returns its
    amount, a normal is drawn with numpy, every other family by stats_arrays.

    ``rng`` must reach *both* branches. stats_arrays' ``random_variables``
    falls back to the legacy global ``np.random`` when ``seeded_random`` is
    omitted, so leaving it out made ``draw_uncertainty_sample(seed=...)``
    reproducible for Normal parameters only -- every lognormal, triangular and
    uniform parameter silently ignored the seed and consumed the global stream
    instead. Each of the four families stats_arrays dispatches to calls only
    ``normal``/``lognormal``/``triangular``/``uniform``, all of which exist on
    a ``Generator``, so the same object serves both branches.
    """
    utype = spec.get("uncertainty_type", None)
    if utype == stats_arrays.NoUncertainty.id:
        return float(spec['amount'])
    if utype == stats_arrays.NormalUncertainty.id:
        loc = float(spec.get("loc", 0.0) or 0.0)
        scale = float(spec.get("scale", 0.0) or 0.0)
        # scale may be 0 for degenerate normals -> returns loc deterministically
        return float(rng.normal(loc, scale)) if scale > 0 else loc
    # generic fallback for triangular/lognormal/etc.
    ua = stats_arrays.UncertaintyBase.from_dicts(spec)
    choice = stats_arrays.uncertainty_choices[utype]
    return float(np.asarray(choice.random_variables(ua, 1, seeded_random=rng)).ravel()[0])

def draw_uncertainty_sample(
    uncertainty_data: dict,
    method: str,
    seed: int | None = None,
) -> dict:
    """
    One draw of every declared parameter: ``{'If': {(e, j): value}, 'Cf': {e: value}}``.
    Undeclared parameters keep their amounts and are not part of the draw.
    """
    rng = np.random.default_rng(seed)
    if_draw = {k: _sample_one_spec(v, rng) for k, v in _merge_defined_blocks(uncertainty_data, 'If').items()}
    cf_defined = uncertainty_data.get('Cf', {}).get(method, {}).get('defined', {})
    cf_draw = {k: _sample_one_spec(v, rng) for k, v in cf_defined.items()}
    return {'If': if_draw, 'Cf': cf_draw}
