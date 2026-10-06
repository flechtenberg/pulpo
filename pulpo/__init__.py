import importlib.util
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('pulpo-dev')
except PackageNotFoundError:  # a source tree that is not installed
    __version__ = 'unknown'

# Brightway comes with the bw2 or bw25 extra. Without one, the first import of
# bw2calc would fail with no hint at the fix.
_missing = [name for name in ('bw2calc', 'bw2data') if importlib.util.find_spec(name) is None]
if _missing:
    raise ModuleNotFoundError(
        f"PULPO needs Brightway, but {' and '.join(_missing)} {'is' if len(_missing) == 1 else 'are'} not "
        'installed. Install PULPO with a Brightway extra: pip install "pulpo-dev[bw25]", or '
        '"pulpo-dev[bw2]" for legacy Brightway2 (Python 3.10-3.12).', name=_missing[0])
