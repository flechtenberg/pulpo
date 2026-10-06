"""PULPO's warnings, reported at the user's code rather than at PULPO's internals."""
import os
import sys
import warnings

_PACKAGE = os.path.normcase(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + os.sep


def warn(message, category=UserWarning):
    """``warnings.warn``, reported at the first frame outside the ``pulpo`` package.

    A fixed ``stacklevel`` is right for one call path only, and PULPO reaches
    most of its warnings through several (the worker, the time-dependent model,
    the uncertainty module), so the level is found by walking the stack.
    Python 3.12's ``skip_file_prefixes`` would do the same, but PULPO supports 3.10.
    """
    frame, level = sys._getframe(1), 2
    while frame is not None and os.path.normcase(os.path.abspath(frame.f_code.co_filename)).startswith(_PACKAGE):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, category, stacklevel=level)
