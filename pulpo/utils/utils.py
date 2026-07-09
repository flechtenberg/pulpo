import bw2data as bd
import bw2calc as bc
from packaging import version
import numpy as np

def is_bw25():
    """Check if the installed Brightway packages adhere to bw25 versions."""
    # Define version thresholds
    THRESHOLDS = {
        "bw2calc": "2.0.dev5",
        "bw2data": "4.0.dev11",
    }

    try:
        for pkg, threshold in {"bw2calc": bc, "bw2data": bd}.items():
            pkg_version = ".".join(map(str, threshold.__version__)) if isinstance(threshold.__version__,
                                                                                  tuple) else str(
                threshold.__version__)
            if version.parse(pkg_version) < version.parse(THRESHOLDS[pkg]):
                return False
        return True
    except Exception as e:
        raise RuntimeError(f"Error checking Brightway versions: {e}")
    
def get_bw_version():
    if is_bw25():
        return "bw25"
    else:
        return "bw2"


# ---------------------------------------------------------------------------
# Time-indexed input helpers
# ---------------------------------------------------------------------------
# Shared by pulpo.utils.time_extension (data prep) and pulpo.utils.saver
# (result extraction) so that both agree on what "time-indexed" means for a
# user-supplied dict without duplicating the detection/broadcast logic.

def is_time_indexed(d, time_steps):
    """Return True if dict ``d`` is keyed by the supplied timestep labels."""
    if not isinstance(d, dict) or not d:
        return False
    return set(d.keys()) == set(time_steps)


def broadcast_over_time(d, time_steps):
    """Convert a (possibly static) input dict into ``{t: dict}`` form."""
    if d is None:
        d = {}
    if not isinstance(d, dict):
        raise TypeError(f"Expected a dict, got {type(d).__name__}")
    if is_time_indexed(d, time_steps):
        for t, sub in d.items():
            if not isinstance(sub, dict):
                raise TypeError(
                    f"Time-indexed input must map each timestep to a dict; "
                    f"got {type(sub).__name__} for t={t!r}"
                )
        return {t: dict(d[t]) for t in time_steps}
    return {t: dict(d) for t in time_steps}


# ---------------------------------------------------------------------------
# BW25 Uncertainty Parameter Handling
# ---------------------------------------------------------------------------

# Combined uncertainty-parameter dtype for the bw25 path. Mirrors the fields of
# the bw2 ``bio_params`` / ``cf_params`` structured arrays that the uncertainty
# preparer consumes (it indexes on ``row``/``col`` and reads ``amount`` plus the
# stats_arrays distribution fields).
BW25_PARAM_DTYPE = np.dtype([
    ('row', np.int64),
    ('col', np.int64),
    ('amount', np.float64),
    ('uncertainty_type', np.uint8),
    ('loc', np.float64),
    ('scale', np.float64),
    ('shape', np.float64),
    ('minimum', np.float64),
    ('maximum', np.float64),
    ('negative', np.bool_),
])

BW25_DISTRIBUTION_FIELDS = (
    'uncertainty_type', 'loc', 'scale', 'shape', 'minimum', 'maximum', 'negative',
)


def build_bw25_params(data_objs, matrix_name, row_mapping, col_mapping=None):
    """Assemble a combined uncertainty-parameter array for a bw25 matrix.

    Rather than relying on hard-coded resource positions, this scans the
    ``datapackage`` objects for the resources belonging to ``matrix_name`` and
    combines the ``indices``, ``data`` and ``distributions`` resources into a
    single structured array. The original ids stored in the ``indices`` resource
    are mapped to matrix positions via ``row_mapping`` (and ``col_mapping`` when
    provided) so the result is indexed identically to the optimisation problem.

    Args:
        data_objs: The datapackages returned by ``bd.prepare_lca_inputs``.
        matrix_name (str): Target matrix, e.g. ``'biosphere_matrix'`` or
            ``'characterization_matrix'``.
        row_mapping (dict): Maps original row ids to matrix row positions.
        col_mapping (dict, optional): Maps original column ids to matrix column
            positions. If ``None`` the raw column ids from the datapackage are
            kept (used for the characterization matrix where the column is a
            dummy value).

    Returns:
        Tuple[Optional[np.ndarray], bool]: The combined parameter array (or
        ``None`` when uncertainty information is missing or incomplete) and a
        flag indicating that data existed but distributions were absent.
    """
    parts = []
    incomplete = False
    for obj in data_objs:
        idx = dat = dist = None
        for res, arr in zip(obj.resources, obj.data):
            if res.get('matrix') != matrix_name:
                continue
            kind = res.get('kind')
            if kind == 'indices':
                idx = arr
            elif kind == 'data':
                dat = arr
            elif kind == 'distributions':
                dist = arr
        # Skip datapackages that do not contribute entries to this matrix.
        if dat is None or len(dat) == 0:
            continue
        # Data present but no (matching) distributions => uncertainty missing.
        if dist is None or len(dist) != len(dat) or idx is None:
            incomplete = True
            continue
        parts.append((idx, dat, dist))

    if incomplete or not parts:
        return None, incomplete

    total = sum(len(dat) for _, dat, _ in parts)
    combined = np.empty(total, dtype=BW25_PARAM_DTYPE)
    pos = 0
    for idx, dat, dist in parts:
        n = len(dat)
        sl = slice(pos, pos + n)
        combined['row'][sl] = [row_mapping.get(int(r), int(r)) for r in idx['row']]
        if col_mapping is not None:
            combined['col'][sl] = [col_mapping.get(int(c), int(c)) for c in idx['col']]
        else:
            combined['col'][sl] = idx['col']
        combined['amount'][sl] = dat
        for field in BW25_DISTRIBUTION_FIELDS:
            combined[field][sl] = dist[field]
        pos += n

    return combined, False
