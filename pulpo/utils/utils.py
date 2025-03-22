import bw2data as bd
import bw2calc as bc
from packaging import version

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