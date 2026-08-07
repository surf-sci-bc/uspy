"""Interactive, notebook-only region picking for `uspy.roi.ROI`/`Profile`.

Optional: `pip install 'uspy[interactive]'`, then `%matplotlib widget` in a notebook
cell before calling `pick_rois`. Nothing in core `uspy` imports this package, and the
picker never mutates the stack or writes into the notebook namespace -- its output is
a paste-able code literal (`ROIPicker.code`) that the user copies into a permanent
cell. That literal is the record of the analysis; the picker itself is throwaway.

`uspy.interactive.spec` (geometry only, no matplotlib/ipywidgets) is always importable
without the extra -- only `ROIPicker`/`pick_rois` need it.
"""
from uspy.interactive.spec import (
    Spec,
    specs_to_code,
    spec_from_bbox,
    spec_from_points,
    translate,
    resize_from_handle,
    rotation_from_handle,
    with_rotation,
    handle_position,
    corner_positions,
    vertex_positions,
    move_vertex,
    insert_vertex,
    remove_vertex,
)

__all__ = [
    "Spec",
    "specs_to_code",
    "spec_from_bbox",
    "spec_from_points",
    "translate",
    "resize_from_handle",
    "rotation_from_handle",
    "with_rotation",
    "handle_position",
    "corner_positions",
    "vertex_positions",
    "move_vertex",
    "insert_vertex",
    "remove_vertex",
    "ROIPicker",
    "pick_rois",
]

_MISSING = (
    "uspy.interactive's picker needs the optional 'interactive' extra:\n"
    "    pip install 'uspy[interactive]'\n"
    "then run `%matplotlib widget` in a notebook cell before calling pick_rois()."
)


def __getattr__(name):
    """PEP 562: keep this package importable with zero optional deps (spec.py needs
    none), while ROIPicker/pick_rois raise an actionable error instead of a bare
    ModuleNotFoundError if ipywidgets/ipympl aren't installed."""
    if name in ("ROIPicker", "pick_rois"):
        try:
            from uspy.interactive import picker
        except ImportError as exc:
            raise ImportError(_MISSING) from exc
        return getattr(picker, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
