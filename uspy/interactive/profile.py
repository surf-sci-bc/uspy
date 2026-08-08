"""Profile-specific half of the interactive picker: `ProfilePicker` subclasses `uspy.
interactive.shell.PickerBase` (the shared ipywidgets shell -- frame/contrast, region
list, code box, label/color editing) and adds a linewidth control plus a preview panel
whose *kind* depends on what it's picking over:

- a single `Image` -> a 1D intensity profile (`uspy.derived.ProfileLine`), one line per
  drawn `Profile`, overlaid.
- an `ImageStack` -> a waterfall / cut (`uspy.derived.Waterfall`) of the *selected*
  profile -- e.g. an ARPES or LEED cut along a chosen direction across the energy
  axis. Only one waterfall fits in the panel, so it always tracks the current
  selection; other profiles stay drawn on the image and in the code box.

Needs the 'interactive' extra (`pip install 'uspy[interactive]'`) and `%matplotlib
widget` run in an earlier notebook cell.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import ipywidgets as widgets
from IPython.display import display

import uspy.derived as derived
import uspy.plotting as plot
from uspy.roi import ROI, Profile
from uspy.interactive.spec import Spec
from uspy.interactive.shell import PickerBase, _compact


class ProfilePicker(PickerBase):
    """Interactive picker for `Profile` regions on an `Image`/`ImageStack`.

    On a single image, the preview panel is the profiles' 1D intensity plotted
    directly (`uspy.derived.ProfileLine`), one overlaid line per drawn profile. On a
    stack, it is a waterfall (`uspy.derived.Waterfall`) of whichever profile is
    currently *selected* (click it, or pick it from the region list) -- the others
    stay drawn on the image and in the code box, just not in the panel, since only one
    waterfall image fits there at a time.

    The code box always shows the exact, paste-able literal for the current profiles
    (`repr()` of the real `Profile` objects -- see `uspy.roi.Profile.__repr__`). That
    pasted literal is the permanent record of the analysis; this picker and its live
    state are throwaway. `.profiles` gives the same objects for same-session
    convenience, but always as *fresh* objects. ROIs are not offered here -- see
    `uspy.interactive.pick_rois` for I(V) curves over an ROI instead.

    A profile is drawn Fiji-style, like a polygon: click each point, double-click (or
    click the last point twice) to finish. The label/color/width row is contextual:
    with a region selected, the row edits *that* region; with nothing selected, it
    sets the defaults for the *next* one. A selected region (or, mid-draw, a selected
    vertex) can be nudged 1px at a time with shift+arrow; `d` or `Delete` removes the
    selected region (or just its selected vertex). See
    `uspy.interactive.canvas.Canvas` for the full drag/edit gesture set.

    Constructed by `pick_profiles()`, which also `display()`s it.
    """

    _shapes = ("profile",)
    _STATUS_TEXT = (
        "click to place points, double-click to finish (Fiji-style, like a polygon); "
        "select a region, Shift+arrow to nudge a vertex, d to delete it"
    )
    _LIVE_LABEL = "live"

    def __init__(
        self, source, yaxis: str = "energy", profiles: Optional[Sequence[Profile]] = None,
        live: Optional[bool] = None, frame: Optional[int] = None, contrast: str = "auto",
        width: int = 1, cmap: str = "inferno", figsize: tuple = (11, 4.5),
    ):
        self._initial_width = width
        self._cmap = cmap
        super().__init__(source, yaxis, regions=profiles, live=live, frame=frame,
                         contrast=contrast, figsize=figsize)

    # -- PickerBase hooks -------------------------------------------------------

    def _reject_seed(self, region) -> None:
        if isinstance(region, ROI):
            raise ValueError(
                "the profile picker does not support ROI regions -- draw a Profile "
                "instead; use uspy.interactive.pick_rois for I(V) curves over an ROI"
            )

    def _setup(self) -> None:
        self.yaxis = self._axis_field
        self._cache: dict = {}

    def _extra_controls(self) -> list:
        w = widgets
        self.w_width = w.BoundedIntText(min=1, max=999, value=self._initial_width,
                                        description="width", **_compact("110px", "50px"))
        controls = [self.w_width]
        if self.is_stack:
            self.w_cmap = w.Dropdown(options=["inferno", "gray", "viridis", "magma"],
                                     value=self._cmap, description="cmap",
                                     **_compact("140px", "55px"))
            controls.append(self.w_cmap)
        return controls

    def _wire_extra(self) -> None:
        self.canvas.new_width = self._initial_width
        self.w_width.observe(self._on_width_change, "value")
        if self.is_stack:
            self.w_cmap.observe(self._on_cmap_change, "value")

    def _sync_extra(self, spec: Optional[Spec]) -> None:
        self.w_width.value = dict(spec.params)["width"] if spec is not None else \
            (self.canvas.new_width or 1)

    def _on_width_change(self, change: dict) -> None:
        if self._syncing:
            return
        value = change["new"]
        index = self.canvas.selected
        if index is None:
            self.canvas.new_width = value
        else:
            self.canvas.set_width(index, value)

    def _on_cmap_change(self, change: dict) -> None:
        self._cmap = change["new"]
        self._update_preview()

    def _on_select_changed(self, index: Optional[int]) -> None:
        super()._on_select_changed(index)
        if self.is_stack:
            # the waterfall panel always shows the *selected* profile -- unlike
            # ROIPicker, where every region is always plotted (all fit as I(V) lines),
            # a selection change here must repaint the panel, not just sync widgets.
            self._update_preview()

    # -- preview ------------------------------------------------------------------

    def _update_preview(self, drag: Optional[tuple] = None) -> None:
        specs = self.canvas.specs
        self.ax_preview.clear()
        if not specs:
            self.fig.canvas.draw_idle()
            return
        drag_index, drag_spec = drag if drag is not None else (None, None)
        if self.is_stack:
            self._update_waterfall_preview(specs, drag_index, drag_spec)
        else:
            self._update_line_preview(specs, drag_index, drag_spec)
        self.fig.canvas.draw_idle()

    def _update_line_preview(self, specs: list, drag_index, drag_spec) -> None:
        """Single image: every profile overlaid as a 1D intensity line."""
        img = self.frames[0]
        self.ax_preview.set_xlabel("pixel")
        self.ax_preview.set_ylabel("intensity")
        lines = []
        for i, spec in enumerate(specs):
            if i == drag_index:
                y = drag_spec.to_object().apply(img)
                self.ax_preview.plot(np.arange(len(y), dtype=float), y, color=drag_spec.color,
                                     linestyle="--", label=drag_spec.label or "_")
                continue
            key = spec.params
            line = self._cache.get(key)
            if line is None:
                line = derived.ProfileLine(img, spec.to_object())
                self._cache[key] = line
            line.color = spec.color
            line.label = spec.label
            lines.append(line)
        if lines:
            plot.plot_line(lines, ax=self.ax_preview)

    def _update_waterfall_preview(self, specs: list, drag_index, drag_spec) -> None:
        """Stack: a waterfall of the *selected* profile only -- falls back to the last
        one drawn when nothing is explicitly selected, so the panel is never blank
        right after a fresh draw (Canvas auto-selects a region on creation)."""
        index = self.canvas.selected
        if index is None or index >= len(specs):
            index = len(specs) - 1
        live_drag = index == drag_index and drag_spec is not None
        spec = drag_spec if live_drag else specs[index]

        step = max(1, int(self.w_step.value))
        stack = self.stack[::step] if step > 1 else self.stack

        live_stride = None
        if live_drag:
            # bounded worst case for a very large stack -- see
            # PickerBase._live_frame_stride; preview-only, replaced by the exact,
            # full-resolution, cached Waterfall the instant the drag ends.
            live_stride = self._live_frame_stride(len(stack))
            frames = stack if live_stride is None else stack[::live_stride]
            wf = derived.Waterfall(frames, spec.to_object(), self.yaxis)
        else:
            key = (spec.params, step, self.yaxis)
            wf = self._cache.get(key)
            if wf is None:
                wf = derived.Waterfall(stack, spec.to_object(), self.yaxis)
                self._cache[key] = wf

        plot._draw_waterfall(wf, ax=self.ax_preview, cmap=self._cmap, aspect="auto")
        label = f" ({spec.label})" if spec.label else ""
        title = f"profile {index}{label}"
        if live_stride is not None:
            title += f" (preview, {len(frames)}/{len(stack)} frames)"
        self.ax_preview.set_title(title)

    # -- public, read-only views ---------------------------------------------------

    @property
    def profiles(self) -> list[Profile]:
        """Fresh `Profile` objects for every region (same-session convenience only --
        the pasted `.code` is the permanent record)."""
        return [s.to_object() for s in self.canvas.specs]


def pick_profiles(
    source, yaxis: str = "energy", profiles: Optional[Sequence[Profile]] = None,
    live: Optional[bool] = None, frame: Optional[int] = None, contrast: str = "auto",
    width: int = 1, cmap: str = "inferno", figsize: tuple = (11, 4.5),
) -> ProfilePicker:
    """Open the interactive profile picker on `source` and display it.

    On a single image, the right-hand panel is the profiles' 1D intensity; on a stack,
    it's a waterfall of the selected profile over `yaxis` (e.g. an ARPES/LEED cut).

    Displays exactly one widget, whether or not the call's return value is used --
    left unassigned in a notebook cell, the trailing `Out[]` echo of the returned
    `ProfilePicker` is a one-line `repr()`, not a second copy of the widget.

    Requires `%matplotlib widget` (the ipympl backend) to have been run in an earlier
    cell, and the 'interactive' extra installed (`pip install 'uspy[interactive]'`).

    Parameters
    ----------
    source : Image or ImageStack
    yaxis : str, optional
        Stack metadata field for the waterfall's y-axis, by default "energy". Unused
        for a single image.
    profiles : Sequence[Profile], optional
        Existing regions to seed the picker with, for re-opening/adjusting them.
        `ROI`s are not accepted -- see `uspy.interactive.pick_rois` for that.
    live : bool, optional
        Recompute the preview while dragging, not just on release. Defaults to True
        for a resident stack or a single image, False for a virtual stack (re-reads
        the whole stack from disk on every drag motion event).
    frame : int, optional
        Initial displayed frame index (stack only). Defaults to the middle frame.
    contrast, figsize
        As in `uspy.plotting.plot_img`.
    width : int, optional
        Initial profile linewidth (px), by default 1.
    cmap : str, optional
        Waterfall colormap (stack only), by default "inferno".

    Returns
    -------
    ProfilePicker
        `.code` is the paste-able record of the session; `.profiles` gives the same
        regions as fresh objects, for same-session convenience.
    """
    picker = ProfilePicker(source, yaxis=yaxis, profiles=profiles, live=live, frame=frame,
                           contrast=contrast, width=width, cmap=cmap, figsize=figsize)
    display(picker.widget)
    return picker
