"""ROI-specific half of the interactive picker: `ROIPicker` subclasses `uspy.
interactive.shell.PickerBase` (the shared ipywidgets shell -- frame/contrast, region
list, code box, label/color editing) and adds the shape button row, rotation editing,
and the live I(V) preview panel.

ROI-only: the I(V) panel here cannot plot a `Profile` (that's a `Waterfall`, a
different tool over a different axis -- see `uspy.interactive.profile.pick_profiles`),
so `Canvas` is constructed with `shapes=ROI_SHAPES` and a seeded `Profile` is rejected
outright. `uspy.interactive.canvas`/`.spec` themselves stay shape-generic.

Needs the 'interactive' extra (`pip install 'uspy[interactive]'`) and `%matplotlib
widget` run in an earlier notebook cell. `uspy.interactive.spec`/`.canvas` do not need
either -- only this module does.
"""
from __future__ import annotations

import urllib.parse
from typing import Optional, Sequence

import matplotlib as mpl
import numpy as np
import ipywidgets as widgets
from IPython.display import display

import uspy.derived as derived
import uspy.plotting as plot
from uspy.roi import ROI, Profile
from uspy.interactive.spec import Spec, ROI_SHAPES, _SIZABLE
from uspy.interactive.shell import PickerBase, _check_backend, _check_mpl_version, _compact

# -- Fiji-style shape button row ------------------------------------------------
# w_shape stays a single ipywidgets.ToggleButtons (not one Button per shape) so
# .options/.value and the one-line `w_shape.observe(...)` wiring to canvas.new_shape
# are unchanged -- only its rendering is replaced, via injected CSS, with square
# buttons whose glyph depicts the shape, Fiji-style, instead of dropdown text.

#: one small hand-drawn glyph per shape in `ROI_SHAPES`, 24x24 viewBox, stroked in a
#: fixed dark grey so they read on ipywidgets' light button background regardless of
#: notebook theme.
_SHAPE_ICON_SVGS: dict = {
    "point": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        '<circle cx="12" cy="12" r="2.5" fill="#333"/>'
        '<path d="M12 2v6M12 16v6M2 12h6M16 12h6" stroke="#333" stroke-width="2"/>'
        "</svg>"
    ),
    "circle": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        '<circle cx="12" cy="12" r="9" fill="none" stroke="#333" stroke-width="2"/>'
        "</svg>"
    ),
    "ellipse": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        '<ellipse cx="12" cy="12" rx="10" ry="6" fill="none" stroke="#333" stroke-width="2"/>'
        "</svg>"
    ),
    "rectangle": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        '<rect x="3" y="5" width="18" height="14" fill="none" stroke="#333" stroke-width="2"/>'
        "</svg>"
    ),
    "polygon": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        '<polygon points="12,2 22,9 18,21 6,21 2,9" fill="none" stroke="#333" '
        'stroke-width="2" stroke-linejoin="round"/>'
        "</svg>"
    ),
}


def _svg_data_uri(svg: str) -> str:
    """URL-encode `svg` for use as a CSS `background-image: url(...)`."""
    return "data:image/svg+xml," + urllib.parse.quote(svg)


#: CSS, scoped to the `uspy-shape-buttons` class (added to `w_shape` itself, whose
#: top-level DOM node is the button row -- ToggleButtons has no extra wrapper), sizing
#: the buttons square with the option text hidden (`font-size: 0`) and one background-
#: image rule per shape. Targets `button[title="<shape>"]` rather than e.g.
#: `:nth-child` so the rules stay correct regardless of `ROI_SHAPES`' order --
#: ipywidgets renders each `ToggleButtons` option's `tooltips` entry into that
#: button's `title` attribute (native browser tooltip), see `w_shape`'s construction.
#: The `nowrap`/`width: auto` rule keeps the five icons on one line: ToggleButtons'
#: default control-box width (~300px) otherwise wraps the group onto a second row
#: once the buttons shrink to 34px; `> div` targets ipywidgets 8's inner button-group
#: div, which is what actually flexes, since the class sits on the outer node.
_SHAPE_BUTTON_CSS = (
    ".uspy-shape-buttons, .uspy-shape-buttons > div {\n"
    "    flex-wrap: nowrap; width: auto;\n"
    "}\n"
    ".uspy-shape-buttons button {\n"
    "    width: 34px; height: 34px; min-width: 34px; padding: 0;\n"
    "    font-size: 0; background-repeat: no-repeat; background-position: center;\n"
    "    background-size: 20px 20px;\n"
    "}\n"
) + "\n".join(
    f'.uspy-shape-buttons button[title="{shape}"] '
    f'{{ background-image: url("{_svg_data_uri(svg)}"); }}'
    for shape, svg in _SHAPE_ICON_SVGS.items()
)


class ROIPicker(PickerBase):
    """Interactive picker for `ROI` regions on an `ImageStack`, with a live I(V) panel.

    Draw/drag/rotate regions on the image; the code box always shows the exact,
    paste-able literal for the current regions (`repr()` of the real `ROI` objects --
    see `uspy.roi.ROI.__repr__`). That pasted literal is the permanent record of the
    analysis; this picker and its live state are throwaway. `.rois` gives the same
    objects for same-session convenience, but always as *fresh* objects (never the
    picker's own, to avoid `ROI.pad_to`'s stale-mask-cache hazard from a
    mutated-and-reused ROI). Profiles are not offered here -- a profile over a stack is
    a `Waterfall`, not an I(V) curve; see `uspy.interactive.pick_profiles` for that.

    The label/color/rotation row is contextual: with a region selected (click it, or
    pick it from the region list), the row edits *that* region -- nothing about a
    region is fixed once drawn. With nothing selected, the same row sets the defaults
    for the *next* region instead, and color defaults to the next `uspy.roi.
    palette_color` not already in use, cycling the same way implicit `ROI` colors do
    outside the picker. Shape is the one exception: it only ever applies to the next
    region (a circle can't retroactively become a polygon).

    Drawing a new region drags out from the click point as its centre (`ROI`'s own
    centre+extents convention) -- hold Ctrl for the old corner-to-corner bounding-box
    gesture, or Shift to force width == height. Zooming/panning with the toolbar never
    also creates or moves a region. A selected region (or, for a polygon, a selected
    vertex) can be nudged 1px at a time with shift+arrow -- plain arrow keys are left
    alone, so matplotlib's own view-history navigation still works. `d` or `Delete`
    removes the selected region (or just its selected vertex); `Backspace` is
    intentionally not bound to this, since matplotlib itself already uses it to step
    back through that view history. See `uspy.interactive.canvas.Canvas` for details.

    Constructed by `pick_rois()`, which also `display()`s it -- build one directly only
    to skip that (e.g. to `display(picker.widget)` later, in a different cell; the
    widget model is shared, so both views of it stay in sync).
    """

    _shapes = ROI_SHAPES
    _STATUS_TEXT = (
        "drag to draw (centred on the click) -- hold Ctrl for corner-to-corner, "
        "Shift for equal width/height (also while resizing a corner handle); "
        "select a region, Shift+arrow to nudge it, d to delete it"
    )
    _LIVE_LABEL = "live I(V)"

    def __init__(
        self, stack, xaxis: str = "energy", rois: Optional[Sequence[ROI]] = None,
        live: Optional[bool] = None, frame: Optional[int] = None, contrast: str = "auto",
        figsize: tuple = (11, 4.5),
    ):
        super().__init__(stack, xaxis, regions=rois, live=live, frame=frame,
                         contrast=contrast, figsize=figsize)

    # -- PickerBase hooks -------------------------------------------------------

    def _reject_seed(self, region) -> None:
        if isinstance(region, Profile):
            raise ValueError(
                "the ROI picker does not support Profile regions -- a profile over a "
                "stack is a waterfall, not an I(V) curve; use uspy.interactive.pick_profiles"
            )

    def _setup(self) -> None:
        self.xaxis = self._axis_field
        # one metadata read per frame (cheap, cached by DataObjectStack -- see
        # CLAUDE.md's virtual-stack notes), reused by both the live and cached paths
        self._x = np.array([getattr(img, self.xaxis) for img in self.stack])
        self._x_unit = self.stack[0].unit.get(self.xaxis, "")
        self._cache: dict[tuple, derived.IntensityLine] = {}

    def _extra_controls(self) -> list:
        w = widgets
        self.w_shape = w.ToggleButtons(
            options=list(ROI_SHAPES), value="circle", tooltips=list(ROI_SHAPES),
            description="", layout=w.Layout(width="190px"),
        )
        self.w_shape.add_class("uspy-shape-buttons")
        self._shape_css = w.HTML(f"<style>{_SHAPE_BUTTON_CSS}</style>")
        self.w_rotation = w.BoundedIntText(min=0, max=359, value=0, description="rotation",
                                           **_compact("130px", "60px"))
        return [self.w_shape, self.w_rotation]

    def _prefix_children(self) -> list:
        return [self._shape_css]

    def _wire_extra(self) -> None:
        self.w_shape.observe(lambda ch: setattr(self.canvas, "new_shape", ch["new"]), "value")
        self.w_rotation.observe(self._on_rotation_widget_change, "value")

    def _sync_extra(self, spec: Optional[Spec]) -> None:
        if spec is not None and spec.shape in _SIZABLE:
            self.w_rotation.value = dict(spec.params).get("rotation", 0)

    def _on_rotation_widget_change(self, change: dict) -> None:
        if self._syncing:
            return
        index = self.canvas.selected
        if index is None or self.canvas.specs[index].shape not in _SIZABLE:
            return
        self.canvas.set_rotation(index, change["new"])

    # -- I(V) preview -----------------------------------------------------------

    def _live_intensity(self, spec: Spec, stride: Optional[int] = None) -> np.ndarray:
        """Drag-time I(V): numerically the same reduction as `IntensityLine` (mean
        over mask==True) but skipping `np.ma.masked_array`'s construction cost and
        cropping to the region's bounding box first -- see the picker's design notes
        for the measured ~25x speedup this buys on a resident stack. Preview only: on
        release `_on_commit` recomputes the canonical `IntensityLine`, which replaces
        this in the plot, so nothing downstream ever sees this fast path's output.

        `stride`, from `PickerBase._live_frame_stride`, restricts this to a bounded
        subset of frames (`self.stack[::stride]`) on a very large stack -- `None`
        (the common case) samples every frame, unchanged from before."""
        mask = spec.to_object().pad_to(self.height, self.width)
        ys, xs = np.nonzero(mask)
        frames = self.stack if stride is None else self.stack[::stride]
        if ys.size == 0:
            return np.zeros(len(frames))
        box = (slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1))
        sub = mask[box]
        return np.array([img.image[box][sub].mean() for img in frames])

    def _update_iv_preview(self, drag: Optional[tuple] = None) -> None:
        """Recompute the I(V) panel. `drag=(index, draft_spec)` during an active live
        drag draws just that one region via the fast path (dashed); every other
        region uses its cached canonical `IntensityLine` -- one `IntensityLine` per
        *new* (shape, params, step) triple, so deleting/re-adding an identical region,
        changing frame, or changing contrast never re-reads the stack (see the cache
        below). Keyed on geometry alone, not the whole `Spec` -- color/label are
        editable after the fact (see `_sync_controls`), and re-reading the stack just
        because a region was renamed or recolored would defeat the point of caching;
        `line.color`/`line.label` are instead refreshed from the current spec on every
        use, cache hit or miss.
        """
        step = max(1, int(self.w_step.value))
        stack = self.stack[::step] if step > 1 else self.stack
        self.ax_iv.clear()
        self.ax_iv.set_xlabel(f"{self.xaxis} in {self._x_unit}" if self._x_unit else self.xaxis)
        self.ax_iv.set_ylabel("intensity")

        drag_index, drag_spec = drag if drag is not None else (None, None)
        # only bother computing the cap when actually dragging -- the committed lines
        # below never take this branch, so it would otherwise be dead work on every
        # non-drag preview refresh (_on_commit, the Update button, construction).
        live_stride = self._live_frame_stride(len(self.stack)) if drag_spec is not None else None

        lines = []
        for i, spec in enumerate(self.canvas.specs):
            if i == drag_index:
                # call with just `spec` when uncapped (the common case) -- keeps the
                # original one-argument calling convention intact wherever nothing
                # needs to know about the cap at all (e.g. a test double built
                # against `_live_intensity`'s original signature).
                y = (self._live_intensity(drag_spec) if live_stride is None
                    else self._live_intensity(drag_spec, live_stride))
                x = self._x if live_stride is None else self._x[::live_stride]
                self.ax_iv.plot(x, y, color=drag_spec.color, linestyle="--",
                                label=drag_spec.label or "_")
                continue
            key = (spec.shape, spec.params, step)
            line = self._cache.get(key)
            if line is None:
                line = derived.IntensityLine(stack, spec.to_object(), self.xaxis)
                self._cache[key] = line
            line.color = spec.color
            line.label = spec.label
            lines.append(line)
        if lines:
            plot.plot_line(lines, ax=self.ax_iv)
        notes = []
        if step > 1:
            notes.append(f"every {step}. frame")
        if live_stride is not None:
            # matches len(self.stack[::live_stride]) exactly (ceil division), without
            # slicing the stack a second time just to count it
            n_live = -(-len(self.stack) // live_stride)
            notes.append(f"{n_live}/{len(self.stack)} frames while dragging")
        if notes:
            self.ax_iv.set_title("preview: " + ", ".join(notes))
        self.fig.canvas.draw_idle()

    _update_preview = _update_iv_preview  # PickerBase's hook name; kept callable under
                                          # both names since ROIPicker's own tests call
                                          # _update_iv_preview directly by that name.

    # -- public, read-only views ---------------------------------------------------

    @property
    def rois(self) -> list[ROI]:
        """Fresh `ROI` objects for every region (same-session convenience only -- the
        pasted `.code` is the permanent record)."""
        return [s.to_object() for s in self.canvas.specs]


def pick_rois(
    stack, xaxis: str = "energy", rois: Optional[Sequence[ROI]] = None,
    live: Optional[bool] = None, frame: Optional[int] = None, contrast: str = "auto",
    figsize: tuple = (11, 4.5),
) -> ROIPicker:
    """Open the interactive ROI/I(V) picker on `stack` and display it.

    Displays exactly one widget, whether or not the call's return value is used --
    left unassigned in a notebook cell, the trailing `Out[]` echo of the returned
    `ROIPicker` is a one-line `repr()`, not a second copy of the widget.

    Requires `%matplotlib widget` (the ipympl backend) to have been run in an earlier
    cell, and the 'interactive' extra installed (`pip install 'uspy[interactive]'`).

    Parameters
    ----------
    stack : ImageStack
    xaxis : str, optional
        Metadata field for the I(V) panel's x-axis, by default "energy".
    rois : Sequence[ROI], optional
        Existing regions to seed the picker with, for re-opening/adjusting them.
        `Profile`s are not accepted -- see `uspy.interactive.pick_profiles` for that.
    live : bool, optional
        Recompute I(V) while dragging (Fiji-style), not just on release. Defaults to
        True for a resident stack, False for a virtual one (re-reads the whole stack
        from disk on every drag motion event -- see `ROIPicker`'s warning).
    frame : int, optional
        Initial displayed frame index. Defaults to the middle frame (no I/O cost,
        unlike `uspy.plotting.get_max_variance_idx`).
    contrast, figsize
        As in `uspy.plotting.plot_img`/`plot_intensity_img`.

    Returns
    -------
    ROIPicker
        `.code` is the paste-able record of the session; `.rois` gives the same
        regions as fresh objects, for same-session convenience.
    """
    picker = ROIPicker(stack, xaxis=xaxis, rois=rois, live=live, frame=frame,
                       contrast=contrast, figsize=figsize)
    display(picker.widget)
    return picker
