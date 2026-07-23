"""
Deprecated. Elettra data loading now goes through the pluggable reader mechanism in
uspy.leem.base/uspy.leem.readers -- use ``LEEMImg(source, reader="elettra")`` or
``LEEMStack(source, reader="elettra")`` (from uspy.leem.base), or drop a ``uspy.toml``
config file with ``reader = "elettra"`` into the data folder, instead of importing this
module's classes directly. See uspy.leem.readers.elettra.ElettraReader for the parsing
logic itself.

fermi_edge/fit_fermi_edge (curve-fitting helpers unrelated to data loading) moved to
uspy.leem.processing, but stay importable from here (lazily, see __getattr__ below) for
backward compatibility.
"""

import warnings

from uspy.leem import base

_DEPRECATION_MSG = (
    "uspy.leem.elettra.{cls} is deprecated; use uspy.leem.base.{cls}(source, "
    'reader="elettra") instead (or a uspy.toml config file, see uspy.leem.readers).'
)

# Functions that moved to uspy.leem.processing but must stay importable from here.
_MOVED_TO_PROCESSING = ("fermi_edge", "fit_fermi_edge")


def LEEMImg(*args, **kwargs) -> base.LEEMImg:  # pylint: disable=invalid-name
    warnings.warn(_DEPRECATION_MSG.format(cls="LEEMImg"), DeprecationWarning, stacklevel=2)
    kwargs.setdefault("reader", "elettra")
    return base.LEEMImg(*args, **kwargs)


def LEEMStack(*args, **kwargs) -> base.LEEMStack:  # pylint: disable=invalid-name
    warnings.warn(_DEPRECATION_MSG.format(cls="LEEMStack"), DeprecationWarning, stacklevel=2)
    kwargs.setdefault("reader", "elettra")
    return base.LEEMStack(*args, **kwargs)


def __getattr__(name: str):
    # PEP 562 module __getattr__: only pulls in uspy.leem.processing (and its heavier
    # dependencies, e.g. symmetrize/numba) if fermi_edge/fit_fermi_edge are actually used,
    # rather than unconditionally on `import uspy.leem.elettra`.
    if name in _MOVED_TO_PROCESSING:
        from uspy.leem import processing

        return getattr(processing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
