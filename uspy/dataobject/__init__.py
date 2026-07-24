"""
Basic data containers.

This package holds the baseline datamodel (``base``: Loadable, DataObject, DataObjectStack)
plus the two dependency-free concrete data types built on top of it (``image``: Image,
ImageStack; ``line``: Line). It has no dependency on ``uspy.roi``.

The four composite types that DO depend on ``uspy.roi`` -- IntensityLine, StitchedLine,
ProfileLine, Waterfall -- have moved to the top-level ``uspy.derived`` module, which breaks
what used to be a ``dataobject <-> roi`` circular import. For backward compatibility they
remain accessible as ``uspy.dataobject.X`` (both ``from uspy.dataobject import X`` and
``do.X`` attribute access, and old pickles that reference ``uspy.dataobject.X`` as the
class's module path), but access now emits a DeprecationWarning pointing at the new
``uspy.derived`` location.
"""

from uspy.dataobject.base import Loadable, DataObject, DataObjectStack
from uspy.dataobject.image import Image, ImageStack
from uspy.dataobject.line import Line

__all__ = [
    "Loadable",
    "DataObject",
    "DataObjectStack",
    "Image",
    "ImageStack",
    "Line",
    "IntensityLine",
    "StitchedLine",
    "ProfileLine",
    "Waterfall",
]

_MOVED_TO_DERIVED = {"IntensityLine", "StitchedLine", "ProfileLine", "Waterfall"}


def __getattr__(name: str):
    """PEP 562 module-level attribute access, for the names that moved to uspy.derived.

    Keeps ``from uspy.dataobject import IntensityLine`` and ``do.IntensityLine`` (and
    unpickling of objects whose class module string is still ``uspy.dataobject``) working,
    while nudging callers towards the new location.
    """
    if name in _MOVED_TO_DERIVED:
        import warnings
        import uspy.derived as _derived

        warnings.warn(
            f"uspy.dataobject.{name} has moved to uspy.derived.{name}; "
            f"update the import, this re-export will eventually be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_derived, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
