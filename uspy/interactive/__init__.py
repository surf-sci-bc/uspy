"""Interactive, notebook-only region picking for `uspy.roi.ROI`/`Profile`, plus a
plain stack viewer.

Optional: `pip install 'uspy[interactive]'`, then `%matplotlib widget` in a notebook
cell before calling `pick_rois`/`pick_profiles`/`view_stack`. Nothing in core `uspy`
imports this package, and none of the three ever mutate the source or write into the
notebook namespace -- the two pickers' output is a paste-able code literal (`.code`)
that the user copies into a permanent cell; that literal is the record of the
analysis, the picker itself is throwaway. `pick_rois` drives a live I(V) panel over an
`ROI`; `pick_profiles` drives a 1D intensity line (single image) or waterfall (stack)
over a `Profile`; `view_stack` is just an image and a frame slider/play button, for
looking at a stack rather than picking anything from it.

`uspy.interactive.spec` (geometry only, no matplotlib/ipywidgets) is always importable
without the extra -- only the pickers and the viewer need it.
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
    with_width,
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
    "with_width",
    "handle_position",
    "corner_positions",
    "vertex_positions",
    "move_vertex",
    "insert_vertex",
    "remove_vertex",
    "ROIPicker",
    "pick_rois",
    "ProfilePicker",
    "pick_profiles",
    "StackViewer",
    "view_stack",
]

_MISSING = (
    "uspy.interactive's pickers need the optional 'interactive' extra:\n"
    "    pip install 'uspy[interactive]'\n"
    "then run `%matplotlib widget` in a notebook cell before calling "
    "pick_rois()/pick_profiles()."
)


def __getattr__(name):
    """PEP 562: keep this package importable with zero optional deps (spec.py needs
    none), while ROIPicker/pick_rois/ProfilePicker/pick_profiles raise an actionable
    error instead of a bare ModuleNotFoundError if ipywidgets/ipympl aren't
    installed."""
    if name in ("ROIPicker", "pick_rois"):
        try:
            from uspy.interactive import picker
        except ImportError as exc:
            raise ImportError(_MISSING) from exc
        return getattr(picker, name)
    if name in ("ProfilePicker", "pick_profiles"):
        try:
            from uspy.interactive import profile
        except ImportError as exc:
            raise ImportError(_MISSING) from exc
        return getattr(profile, name)
    if name in ("StackViewer", "view_stack"):
        try:
            from uspy.interactive import viewer
        except ImportError as exc:
            raise ImportError(_MISSING) from exc
        return getattr(viewer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
