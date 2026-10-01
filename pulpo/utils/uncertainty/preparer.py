"""Import of the uncertainty data an impact ``X(s) = sum_e q_e sum_j b_ej s_j`` depends on.

Which parameters
----------------
Every entry ``b_ej`` of the intervention matrix on a flow ``e`` that the LCIA
method characterizes, and every characterization factor ``q_e`` of the method
on a flow that occurs in the intervention matrix. Nothing else can move ``X``:
an entry on an uncharacterized flow is multiplied by a CF of exactly zero, and
an entry that is zero in ``B`` is no coefficient at all. No contribution
filter is applied, so the parameter set does not depend on a reference
solution.

Declared and undeclared
-----------------------
A parameter is *declared* when the database or the method gives it a
distribution (``uncertainty_type > 0``) and *undeclared* otherwise. Undeclared
parameters are deterministic in every result (mean = amount, variance 0); they
are kept, under ``'undefined'``, so that the screening of undeclared parameters
can rank them. ``uncertainty_type = 1`` ("no uncertainty") is a declaration of
an exact value and is kept under ``'defined'`` with variance 0.

The container (``UncertaintyData``) is a nested dict::

    {'If': {database: {'defined': {(e, j): spec}, 'undefined': {(e, j): spec}}},
     'Cf': {method:   {'defined': {e: spec},      'undefined': {e: spec}}}}

with ``spec`` a stats_arrays-style dict (``uncertainty_type``, ``amount``,
``loc``, ``scale``, ``shape``, ``minimum``, ``maximum``, ``negative``) and
``(e, j)`` the (flow row, process column) of the entry in ``B``.
"""

import math
from typing import Dict, List, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import stats_arrays


class UncertaintySpec(TypedDict, total=False):
    """One parameter's declared distribution, in stats_arrays' field names."""
    uncertainty_type: int
    amount: float
    loc: float
    scale: float
    shape: float
    minimum: float
    maximum: float
    negative: bool


ParamIndex = Union[Tuple[int, int], int]


class DefUndefBlock(TypedDict, total=False):
    defined: Dict[ParamIndex, UncertaintySpec]
    undefined: Dict[ParamIndex, UncertaintySpec]


class UncertaintyData(TypedDict, total=False):
    If: Dict[str, DefUndefBlock]     # one block per database
    Cf: Dict[str, DefUndefBlock]     # one block, for the LCIA method


#: Families with closed-form moments and quantiles, and the fields each needs.
SUPPORTED_FAMILIES = {
    stats_arrays.NoUncertainty.id: (),
    stats_arrays.LognormalUncertainty.id: ('loc', 'scale'),
    stats_arrays.NormalUncertainty.id: ('loc', 'scale'),
    stats_arrays.UniformUncertainty.id: ('minimum', 'maximum'),
    stats_arrays.TriangularUncertainty.id: ('minimum', 'loc', 'maximum'),
}

_SPEC_FIELDS = ('uncertainty_type', 'amount', 'loc', 'scale', 'shape', 'minimum', 'maximum', 'negative')


def _single_method(worker):
    methods = list(worker.method)
    if len(methods) != 1:
        raise ValueError("The uncertainty of an impact is imported for one LCIA method; the worker "
                         f"has {len(methods)}. Create a worker with that method alone.")
    return methods[0]


def _record(row) -> UncertaintySpec:
    spec = {field: row[field] for field in _SPEC_FIELDS if field in row}
    spec['uncertainty_type'] = int(spec.get('uncertainty_type', 0))
    spec['amount'] = float(spec['amount'])
    if 'negative' in spec:
        spec['negative'] = bool(spec['negative'])
    return spec


def _split(records):
    defined, undefined = {}, {}
    for index, spec in records:
        (defined if spec['uncertainty_type'] > 0 else undefined)[index] = spec
    return defined, undefined


def import_declared(worker, method=None) -> UncertaintyData:
    """The parameters of the worker's impact and their declared distributions.

    Args:
        worker (PulpoOptimizer): a worker after ``get_lci_data()``.
        method (str, optional): the LCIA method; defaults to the worker's only one.

    Returns:
        UncertaintyData: see the module docstring.
    """
    if worker.lci_data is None:
        raise ValueError("Call get_lci_data() before importing the uncertainty data.")
    method = method or _single_method(worker)
    lci = worker.lci_data
    databases = worker.database if isinstance(worker.database, list) else [worker.database]
    if lci.get('intervention_params') is None or lci.get('characterization_params', {}).get(method) is None:
        raise ValueError("The LCI data carries no uncertainty parameters (bw2data stores none for "
                         "this database or method); see bw_parser.import_data.")

    cf_params = pd.DataFrame(lci['characterization_params'][method])
    if cf_params['row'].duplicated().any():
        raise NotImplementedError(f"The method {method!r} characterizes a flow twice.")
    B = sp.csr_matrix(lci['intervention_matrix'])
    used_flows = set(np.flatnonzero(np.diff(B.indptr)).tolist())
    cf_params = cf_params[cf_params['row'].isin(used_flows)]
    flows = set(cf_params['row'].astype(int).tolist())

    params = pd.DataFrame(lci['intervention_params'])
    params = params[params['row'].isin(flows)]
    duplicated = params.duplicated(['row', 'col'], keep=False)
    if duplicated.any():
        first = params[duplicated].iloc[0]
        raise NotImplementedError(
            f"{int(duplicated.sum())} exchanges share a (flow, process) entry of B with another "
            f"exchange, e.g. flow {int(first['row'])} in process {int(first['col'])}; their sum "
            "has no single declared distribution.")

    process_db = {j: key[0] for key, j in lci['process_map'].items()}
    data: UncertaintyData = {'If': {db: {'defined': {}, 'undefined': {}} for db in databases},
                             'Cf': {}}
    records = {db: [] for db in databases}
    for row in params.to_dict('records'):
        index = (int(row['row']), int(row['col']))
        db = process_db.get(index[1])
        if db not in records:
            raise ValueError(f"Process {index[1]} belongs to {db!r}, which is not among the "
                             f"worker's databases {databases}.")
        records[db].append((index, _record(row)))
    for db, recs in records.items():
        data['If'][db]['defined'], data['If'][db]['undefined'] = _split(recs)
    data['Cf'][method] = dict(zip(('defined', 'undefined'),
                                  _split((int(row['row']), _record(row))
                                         for row in cf_params.to_dict('records'))))
    for spec in _iter_defined(data):
        _validate(spec)
    return data


def _iter_defined(data):
    for group in data.values():
        for block in group.values():
            yield from block['defined'].values()


def _validate(spec: UncertaintySpec, index=None):
    utype = int(spec['uncertainty_type'])
    if utype not in SUPPORTED_FAMILIES:
        raise NotImplementedError(
            f"Parameter {index if index is not None else ''} declares uncertainty_type={utype}, "
            "which has no closed-form moments here; supported are 1 (no uncertainty), "
            "2 (lognormal), 3 (normal), 4 (uniform) and 5 (triangular).")
    missing = [f for f in SUPPORTED_FAMILIES[utype]
               if spec.get(f) is None or (isinstance(spec.get(f), float) and math.isnan(spec[f]))]
    if missing:
        raise ValueError(f"Parameter {index if index is not None else ''} of uncertainty_type={utype} "
                         f"lacks {missing}.")
    if utype == stats_arrays.TriangularUncertainty.id and not (
            spec['minimum'] <= spec['loc'] <= spec['maximum']):
        raise ValueError(f"Parameter {index if index is not None else ''}: a triangular distribution "
                         "needs minimum <= loc (the mode) <= maximum.")
    if utype == stats_arrays.UniformUncertainty.id and not spec['minimum'] <= spec['maximum']:
        raise ValueError(f"Parameter {index if index is not None else ''}: a uniform distribution "
                         "needs minimum <= maximum.")
    if utype in (stats_arrays.NormalUncertainty.id, stats_arrays.LognormalUncertainty.id) and spec['scale'] < 0:
        raise ValueError(f"Parameter {index if index is not None else ''}: scale must be non-negative.")


def override(uncertainty_data: UncertaintyData, group: str, subgroup: str,
             specs: Dict[ParamIndex, UncertaintySpec]) -> UncertaintyData:
    """Declare or replace distributions, e.g. from expert elicitation.

    Each spec replaces the parameter's distribution; ``amount`` (the value the
    deterministic model uses) is kept unless the spec gives one. A parameter
    that is not in the data raises ``KeyError``: it is not a coefficient of
    the impact (a zero entry of ``B``, or an uncharacterized flow).

    Args:
        uncertainty_data: from :func:`import_declared`; modified in place and returned.
        group: ``'If'`` or ``'Cf'``.
        subgroup: the database (``'If'``) or the method (``'Cf'``).
        specs: ``{index: spec}`` with ``index`` = ``(flow row, process column)``
            for ``'If'`` and the flow row for ``'Cf'``.
    """
    block = uncertainty_data[group][subgroup]
    replaced = {}
    for index, spec in specs.items():
        current = block['defined'].get(index, block['undefined'].get(index))
        if current is None:
            raise KeyError(f"{index!r} is not a parameter of {group}/{subgroup}: it is not a nonzero, "
                           "characterized entry of the impact.")
        new = {field: np.nan for field in _SPEC_FIELDS if field not in ('negative', 'amount')}
        new['negative'] = bool(current.get('negative', False))
        new['amount'] = current['amount']
        new.update(spec)
        new['uncertainty_type'] = int(new['uncertainty_type'])
        if new['uncertainty_type'] > 0:
            _validate(new, index)
        replaced[index] = new
    # Applied only once every spec has been validated, so a rejected call
    # leaves the data unchanged.
    for index, new in replaced.items():
        block['defined'].pop(index, None)
        block['undefined'].pop(index, None)
        (block['defined'] if new['uncertainty_type'] > 0 else block['undefined'])[index] = new
    return uncertainty_data


def undeclared(uncertainty_data: UncertaintyData) -> Dict[str, Dict[str, Dict[ParamIndex, UncertaintySpec]]]:
    """``{group: {subgroup: {index: spec}}}`` of every undeclared parameter."""
    return {group: {sub: dict(block['undefined']) for sub, block in blocks.items()}
            for group, blocks in uncertainty_data.items()}


def counts(uncertainty_data: UncertaintyData) -> List[dict]:
    """One row per group, subgroup and family: how many parameters each holds."""
    rows = []
    for group, blocks in uncertainty_data.items():
        for sub, block in blocks.items():
            rows.append({'group': group, 'subgroup': sub, 'status': 'undeclared',
                         'uncertainty_type': 0, 'n': len(block['undefined'])})
            types = pd.Series([int(s['uncertainty_type']) for s in block['defined'].values()], dtype=int)
            for utype, n in types.value_counts().sort_index().items():
                rows.append({'group': group, 'subgroup': sub, 'status': 'declared',
                             'uncertainty_type': int(utype), 'n': int(n)})
    return rows
