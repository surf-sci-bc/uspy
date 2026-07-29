"""
Basic data containers: the baseline datamodel (``base``: Loadable, DataObject,
DataObjectStack) plus the two dependency-free concrete types (``image``, ``line``). Has no
dependency on ``uspy.roi``.

The four ``uspy.roi``-dependent composite types (IntensityLine, StitchedLine, ProfileLine,
Waterfall) moved to ``uspy.derived`` to break a ``dataobject <-> roi`` circular import, but
stay accessible as ``uspy.dataobject.X`` for backward compatibility (incl. old pickles),
with a DeprecationWarning pointing at the new location.
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
    """PEP 562 module-level attribute access for the names that moved to uspy.derived --
    keeps old imports and unpickling working, with a DeprecationWarning."""
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
