"""Pytest configuration for pulpo tests.

Redirects bw2data to a temporary directory before any test module is
imported, so that no data is written to the user's Brightway project
directory during the test session.  The temporary directory is removed
automatically when the session ends.

This approach works for both the legacy Brightway 2 stack
(bw2data < 4.0) and the modern Brightway 2.5 stack (bw2data >= 4.0)
because ``projects.change_base_directories`` is available in both.
"""

import glob
import os
import shutil
import site
import tempfile
from ctypes.util import find_library
from pathlib import Path


def _locate_mkl_rt():
    """Return the path to ``libmkl_rt`` if it can be found without a
    slow recursive filesystem search, or ``None`` otherwise."""
    for name in ("mkl_rt", "mkl_rt.1"):
        path = find_library(name)
        if path:
            return path
    # Search only in the Python user-base lib dir – this is a fast,
    # bounded path.  Avoid sys.prefix (/usr) because the recursive glob
    # over /usr/lib can hang on systems with many files.
    for found in glob.glob(f"{site.USER_BASE}/lib*/*mkl_rt*"):
        if os.path.isfile(found):
            return found
    return None


def pytest_configure(config):
    """Set up the test session before any test modules are collected.

    Two things happen here, in order:

    1. ``PYPARDISO_MKL_RT`` is set when the Intel MKL library can be
       located without triggering the slow recursive glob that
       ``pypardiso`` performs over ``/usr/lib*``.  This prevents the
       import of ``pypardiso`` (and therefore ``bw2calc``) from hanging
       in sandboxed CI environments.

    2. bw2data is redirected to a fresh temporary directory so that no
       data is written to the user's real Brightway project folder.
       ``pytest_configure`` runs before test modules are collected (and
       therefore before their module-level ``setup_*`` calls execute),
       so every ``bd.projects.set_current(...)`` call lands in the
       temporary directory.
    """
    # --- 1. Resolve MKL path before pypardiso / bw2calc are imported ---
    if "PYPARDISO_MKL_RT" not in os.environ:
        mkl_path = _locate_mkl_rt()
        if mkl_path:
            os.environ["PYPARDISO_MKL_RT"] = mkl_path

    # --- 2. Redirect bw2data to a temporary directory ---
    import bw2data as bd

    tmpdir = Path(tempfile.mkdtemp())
    config._bw_tmpdir = tmpdir
    bd.projects.change_base_directories(
        base_dir=tmpdir,
        base_logs_dir=tmpdir,
        project_name="test_default",
        update=False,
    )
    # Signal to bw2data that this is a temporary directory (suppresses
    # the "deleting project in temp dir" warning in bw2data 4.x).
    bd.projects._is_temp_dir = True


def pytest_unconfigure(config):
    """Remove the temporary bw2data directory after the test session."""
    tmpdir = getattr(config, "_bw_tmpdir", None)
    if tmpdir is not None:
        shutil.rmtree(tmpdir, ignore_errors=True)
