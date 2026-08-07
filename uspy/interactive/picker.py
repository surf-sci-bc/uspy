"""ipywidgets shell for the interactive ROI picker: wires
`uspy.interactive.canvas.Canvas` (the draggable-artist layer, pure matplotlib) to
widget controls -- shape/label/color/rotation, frame/contrast, live/auto I(V) preview
-- and to the I(V) preview panel itself.

ROI-only: the I(V) panel here cannot plot a `Profile` (that's a `Waterfall`, a
different tool over a different axis -- see `uspy.plotting.waterfall`), so `Canvas` is
constructed with `shapes=ROI_SHAPES` and a seeded `Profile` is rejected outright.
`uspy.interactive.canvas`/`.spec` themselves stay shape-generic.

Needs the 'interactive' extra (`pip install 'uspy[interactive]'`) and `%matplotlib
widget` run in an earlier notebook cell. `uspy.interactive.spec`/`.canvas` do not need
either -- only this module does.
"""
from __future__ import annotations

import time
import urllib.parse
import warnings
from typing import Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import ipywidgets as widgets
from IPython.display import display

import uspy.derived as derived
import uspy.plotting as plot
from uspy.roi import ROI, Profile
from uspy.interactive.canvas import Canvas
from uspy.interactive.spec import Spec, ROI_SHAPES, _SIZABLE, specs_to_code

_MIN_MPL_VERSION = (3, 6)
#: minimum interval between live-I(V) recomputes while dragging, seconds. A drag
#: fires far more motion events than this; without throttling, a slow stack (or even
#: just a slow browser/comm round trip) would pile up a growing backlog of recomputes
#: that can never catch up with the mouse. See ROIPicker._on_drag.
_LIVE_MIN_INTERVAL = 0.05


def _check_backend(fig) -> None:
    """ipympl's canvas is the only one with `header_visible`; anything else means
    `%matplotlib widget` was not run, and the figure would just be a static image with
    no working mouse/key events."""
    if not hasattr(fig.canvas, "header_visible"):
        plt.close(fig)
        raise RuntimeError(
            "The ROI picker needs the ipympl backend. Run `%matplotlib widget` in a "
            f"cell before pick_rois() (current backend: {mpl.get_backend()})."
        )


def _check_mpl_version() -> None:
    major, minor = mpl.__version__.split(".")[:2]
    if (int(major), int(minor)) < _MIN_MPL_VERSION:
        raise RuntimeError(
            f"The ROI picker needs matplotlib>={'.'.join(map(str, _MIN_MPL_VERSION))} "
            f"(found {mpl.__version__}) for rotatable selection handles."
        )


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


def _compact(width: str, description_width: str) -> dict:
    """`layout`/`style` kwargs sizing a captioned widget to its content, instead of
    ipywidgets' default ~300px control box with an ~80px description column."""
    return {"layout": widgets.Layout(width=width),
            "style": {"description_width": description_width}}


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


class ROIPicker:
    """Interactive picker for `ROI` regions on an `ImageStack`, with a live I(V) panel.

    Draw/drag/rotate regions on the image; the code box always shows the exact,
    paste-able literal for the current regions (`repr()` of the real `ROI` objects --
    see `uspy.roi.ROI.__repr__`). That pasted literal is the permanent record of the
    analysis; this picker and its live state are throwaway. `.rois` gives the same
    objects for same-session convenience, but always as *fresh* objects (never the
    picker's own, to avoid `ROI.pad_to`'s stale-mask-cache hazard from a
    mutated-and-reused ROI). Profiles are not offered here -- a profile over a stack is
    a `Waterfall`, not an I(V) curve; see `uspy.plotting.waterfall` for that instead.

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

    def __init__(
        self, stack, xaxis: str = "energy", rois: Optional[Sequence[ROI]] = None,
        live: Optional[bool] = None, frame: Optional[int] = None, contrast: str = "auto",
        figsize: tuple = (11, 4.5),
    ):
        _check_mpl_version()
        if any(isinstance(r, Profile) for r in rois or ()):
            raise ValueError(
                "the ROI picker does not support Profile regions -- a profile over a "
                "stack is a waterfall, not an I(V) curve; use uspy.plotting.waterfall"
            )
        self.stack = stack
        self.xaxis = xaxis
        self.height, self.width = stack[0].shape
        # one metadata read per frame (cheap, cached by DataObjectStack -- see
        # CLAUDE.md's virtual-stack notes), reused by both the live and cached paths
        self._x = np.array([getattr(img, xaxis) for img in stack])
        self._x_unit = stack[0].unit.get(xaxis, "")

        is_virtual = getattr(stack, "virtual", False)
        if live is None:
            live = not is_virtual
        elif live and is_virtual:
            warnings.warn(
                "live I(V) on a virtual stack re-reads every frame from disk on each "
                "drag motion event -- consider stack.materialize() first, or leave "
                "live off and use the Update button instead.",
                UserWarning, stacklevel=2,
            )
        self.live = live

        self._cache: dict[tuple, derived.IntensityLine] = {}
        self._clock = time.monotonic
        self._last_live_update = 0.0

        with plt.ioff():
            self.fig, (self.ax_img, self.ax_iv) = plt.subplots(1, 2, figsize=figsize)
        _check_backend(self.fig)
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False

        self._frame_idx = frame if frame is not None else len(stack) // 2
        self._contrast = contrast
        self._base_artists: list = []
        self._draw_frame()

        self._syncing = False  # guards widget<->canvas write-back loops, see _sync_controls
        self.canvas = Canvas(self.ax_img, self.height, self.width,
                             on_commit=self._on_commit, on_drag=self._on_drag,
                             on_select=self._on_select_changed, shapes=ROI_SHAPES)
        if rois:
            self.canvas.specs = [Spec.from_roi(r) for r in rois]
            self.canvas.redraw_overlays()

        self._build_widgets()
        self._wire_widgets()
        self._refresh_code()
        self._refresh_list()
        self._sync_controls()
        self._update_iv_preview()
        # Everything above (_draw_frame, Canvas's initial redraw_overlays,
        # _update_iv_preview) draws before self.fig.canvas is ever display()-ed inside
        # _build_widgets(), i.e. before the ipympl comm channel to the browser exists
        # -- those early draw_idle() calls are effectively lost, leaving the canvas
        # blank until some later event (e.g. a click) happens to trigger another one.
        # One explicit, synchronous draw() here, after display(), guarantees a real
        # frame is sent once the frontend can actually receive it.
        self.fig.canvas.draw()

    # -- base frame -----------------------------------------------------------

    def _draw_frame(self) -> None:
        """(Re)draw the grayscale frame. Tracks exactly the artists `plot_img` adds
        (same `_track`-style diffing `Canvas` uses for its own layers) so repeated
        calls -- frame slider, contrast dropdown -- don't accumulate images/text on
        top of each other. Never touches `Canvas`'s own overlay/outline artists,
        which live on the same axes at a higher zorder."""
        for artist in self._base_artists:
            artist.remove()
        before = {id(a) for a in self.ax_img.get_children()}
        plot.plot_img(self.stack[self._frame_idx], ax=self.ax_img, ticks=True,
                      contrast=self._contrast, title=False)
        self._base_artists = [a for a in self.ax_img.get_children() if id(a) not in before]
        self.fig.canvas.draw_idle()

    # -- I(V) preview -----------------------------------------------------------

    def _live_intensity(self, spec: Spec) -> np.ndarray:
        """Drag-time I(V): numerically the same reduction as `IntensityLine` (mean
        over mask==True) but skipping `np.ma.masked_array`'s construction cost and
        cropping to the region's bounding box first -- see the picker's design notes
        for the measured ~25x speedup this buys on a resident stack. Preview only: on
        release `_on_commit` recomputes the canonical `IntensityLine`, which replaces
        this in the plot, so nothing downstream ever sees this fast path's output."""
        mask = spec.to_object().pad_to(self.height, self.width)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return np.zeros(len(self.stack))
        box = (slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1))
        sub = mask[box]
        return np.array([img.image[box][sub].mean() for img in self.stack])

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
        use, cache hit or not.
        """
        step = max(1, int(self.w_step.value))
        stack = self.stack[::step] if step > 1 else self.stack
        self.ax_iv.clear()
        self.ax_iv.set_xlabel(f"{self.xaxis} in {self._x_unit}" if self._x_unit else self.xaxis)
        self.ax_iv.set_ylabel("intensity")

        drag_index, drag_spec = drag if drag is not None else (None, None)
        lines = []
        for i, spec in enumerate(self.canvas.specs):
            if i == drag_index:
                y = self._live_intensity(drag_spec)
                self.ax_iv.plot(self._x, y, color=drag_spec.color, linestyle="--",
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
        if step > 1:
            self.ax_iv.set_title(f"preview: every {step}. frame")
        self.fig.canvas.draw_idle()

    # -- Canvas callbacks -------------------------------------------------------

    def _on_commit(self) -> None:
        if self.w_auto.value:
            self._update_iv_preview()
        self._refresh_code()
        self._refresh_list()
        self._sync_controls()
        if self.canvas.status:
            self.w_status.value = self.canvas.status
            self.canvas.status = ""

    def _on_drag(self, index: int, spec: Spec) -> None:
        if not self.live:
            return
        now = self._clock()
        if now - self._last_live_update < _LIVE_MIN_INTERVAL:
            return
        self._last_live_update = now
        self._update_iv_preview(drag=(index, spec))

    def _on_select_changed(self, index: Optional[int]) -> None:
        """`Canvas.on_select` -- fires when a click on the canvas itself (as opposed
        to picker.py's own list/buttons) changes the selection, so the label/color/
        rotation controls follow it either way."""
        self._sync_controls()

    # -- widgets ---------------------------------------------------------------

    def _build_widgets(self) -> None:
        w = widgets
        self.w_shape = w.ToggleButtons(
            options=list(ROI_SHAPES), value="circle", tooltips=list(ROI_SHAPES),
            description="", layout=w.Layout(width="190px"),
        )
        self.w_shape.add_class("uspy-shape-buttons")
        self._shape_css = w.HTML(f"<style>{_SHAPE_BUTTON_CSS}</style>")
        self.w_label = w.Text(description="label", **_compact("200px", "40px"))
        self.w_color = w.ColorPicker(concise=True, value=to_hex(self.canvas.color_for_new()),
                                     description="color", **_compact("90px", "40px"))
        self.w_rotation = w.BoundedIntText(min=0, max=359, value=0, description="rotation",
                                           **_compact("130px", "60px"))

        # continuous_update=True: unlike a drag on the canvas (many motion events per
        # second, throttled by Canvas._REDRAW_MIN_INTERVAL), a slider only fires one
        # event per integer frame step, and each just re-reads one frame and re-plots
        # it -- cheap enough to update live, so don't add throttling here.
        self.w_frame = w.IntSlider(
            min=0, max=max(0, len(self.stack) - 1), value=self._frame_idx, description="frame",
            continuous_update=True, layout=w.Layout(width="99%"),
            style={"description_width": "initial"},
        )
        self.w_contrast = w.Dropdown(options=["auto", "inner"], value="auto", description="contrast",
                                     **_compact("140px", "60px"))
        self.w_step = w.BoundedIntText(min=1, max=max(1, len(self.stack)), value=1,
                                       description="every n-th", **_compact("150px", "75px"))
        _check_layout = w.Layout(width="auto", margin="0 12px 0 0")
        self.w_live = w.Checkbox(value=self.live, description="live I(V)", indent=False,
                                 layout=_check_layout)
        self.w_auto = w.Checkbox(value=True, description="auto preview", indent=False,
                                 layout=_check_layout)
        self.w_update = w.Button(description="Update", layout=w.Layout(width="90px"))

        self.w_delete = w.Button(description="Delete selected", layout=w.Layout(width="130px"))
        self.w_clear = w.Button(description="Clear all", layout=w.Layout(width="90px"))
        self.w_list = w.Select(options=[], description="regions", rows=6)

        self.w_status = w.Label(
            value="drag to draw (centred on the click) -- hold Ctrl for corner-to-corner, "
                 "Shift for equal width/height (also while resizing a corner handle); "
                 "select a region, Shift+arrow to nudge it, d to delete it"
        )
        self.w_code = w.Textarea(value="", layout=w.Layout(width="100%", height="140px"))
        self.w_copy = w.Button(description="Copy code")
        self._copy_out = w.Output(layout=w.Layout(height="0px", overflow="hidden"))

        self.canvas_out = w.Output()
        with self.canvas_out:
            display(self.fig.canvas)

        # flex_flow="row wrap": lets a narrow notebook wrap the row instead of
        # overflowing, now that each control is sized to its content rather than
        # padding out a fixed-width row; align_items="center" lines the 34px shape
        # icons up with the text/number fields next to them.
        row = w.Layout(flex_flow="row wrap", align_items="center")
        self.widget = w.VBox([
            self._shape_css,
            w.HBox([self.w_shape, self.w_label, self.w_color, self.w_rotation,
                    self.w_delete, self.w_clear], layout=row),
            w.HBox([self.w_frame], layout=w.Layout(width="100%")),
            w.HBox([self.w_contrast, self.w_step, self.w_live, self.w_auto, self.w_update],
                   layout=row),
            w.HBox([self.canvas_out, self.w_list]),
            self.w_status,
            w.HBox([self.w_code, self.w_copy]),
            self._copy_out,
        ])

    def _wire_widgets(self) -> None:
        self.w_shape.observe(lambda ch: setattr(self.canvas, "new_shape", ch["new"]), "value")
        self.w_label.observe(self._on_label_change, "value")
        self.w_color.observe(self._on_color_change, "value")
        self.w_rotation.observe(self._on_rotation_widget_change, "value")

        self.w_frame.observe(self._on_frame_change, "value")
        self.w_contrast.observe(self._on_contrast_change, "value")
        self.w_live.observe(lambda ch: setattr(self, "live", ch["new"]), "value")
        self.w_update.on_click(lambda _btn: self._update_iv_preview())

        self.w_delete.on_click(lambda _btn: self._delete_selected())
        self.w_clear.on_click(lambda _btn: self._clear_all())
        self.w_list.observe(self._on_list_select, "value")
        self.w_copy.on_click(self._on_copy)

    def _on_label_change(self, change: dict) -> None:
        if self._syncing:
            return
        value = change["new"] or None
        index = self.canvas.selected
        if index is None:
            self.canvas.new_label = value
        else:
            self.canvas.set_label(index, value)

    def _on_color_change(self, change: dict) -> None:
        if self._syncing:
            return
        value = change["new"]
        index = self.canvas.selected
        if index is None:
            self.canvas.new_color = value
        else:
            self.canvas.set_color(index, value)

    def _on_rotation_widget_change(self, change: dict) -> None:
        if self._syncing:
            return
        index = self.canvas.selected
        if index is None or self.canvas.specs[index].shape not in _SIZABLE:
            return
        self.canvas.set_rotation(index, change["new"])

    def _on_frame_change(self, change: dict) -> None:
        self._frame_idx = change["new"]
        self._draw_frame()

    def _on_contrast_change(self, change: dict) -> None:
        self._contrast = change["new"]
        self._draw_frame()

    def _on_list_select(self, change: dict) -> None:
        if self._syncing:
            return
        index = change["new"]
        if index is None:
            return
        self.canvas.select(index)  # on_select -> _sync_controls loads label/color/rotation

    def _delete_selected(self) -> None:
        self.canvas.delete_selected()

    def _clear_all(self) -> None:
        self.canvas.clear()

    def _on_copy(self, _btn) -> None:
        from IPython.display import Javascript  # local: optional, browser-only feature
        text = self.w_code.value.replace("\\", "\\\\").replace("`", "\\`")
        with self._copy_out:
            self._copy_out.clear_output()
            display(Javascript(f"navigator.clipboard.writeText(`{text}`)"))
        self.w_status.value = ("copied -- if your browser blocked it, select the "
                               "code box and press Ctrl+C")

    # -- code / list / control sync ------------------------------------------------

    def _refresh_code(self) -> None:
        self.w_code.value = specs_to_code(self.canvas.specs)

    def _refresh_list(self) -> None:
        def entry(i, spec):
            label = f" ({spec.label})" if spec.label else ""
            return f"{i}: {spec.shape}{label}"
        selected = self.canvas.selected
        # rebuilding .options resets .value to None -- restore the selection so the
        # list doesn't visibly deselect on every commit (add/edit/delete all rebuild).
        self.w_list.options = [(entry(i, s), i) for i, s in enumerate(self.canvas.specs)]
        if selected is not None and selected < len(self.w_list.options):
            self.w_list.value = selected

    def _sync_controls(self) -> None:
        """Point the shape/label/color/rotation row at whatever is currently
        selected, so a region's data can be edited after it's drawn instead of only
        at creation time. With a selection, load that region's label/color/rotation
        into the widgets (editing them then calls `Canvas.set_label`/`set_color`/
        `set_rotation` -- see `_on_label_change` etc.); with no selection, show and
        reset the *next region's* defaults instead, so an explicit color picked for
        one region doesn't linger and silently override the color cycle for every
        region after it. Guarded by `_syncing` -- setting a widget's `.value` here
        would otherwise re-enter the very `_on_*_change` handlers this is meant to
        drive, each of which calls a `Canvas` setter that fires `on_commit`, which
        calls back into this method.
        """
        self._syncing = True
        try:
            index = self.canvas.selected
            if index is None:
                self.w_label.value = self.canvas.new_label or ""
                self.w_color.value = to_hex(self.canvas.auto_color())
                self.canvas.new_color = None
                if self.w_list.value is not None:
                    self.w_list.value = None
            else:
                spec = self.canvas.specs[index]
                self.w_label.value = spec.label or ""
                # spec.color is whatever a caller/notebook wrote (e.g. ROI(...,
                # color="b")) -- ipywidgets.ColorPicker only accepts HTML colors
                # (#rrggbb or a CSS name) and raises TraitError on anything else
                # matplotlib itself would happily accept (named shorthands like "b",
                # "C0"/"tab:blue", float-RGB tuples). to_hex() normalizes for the
                # widget only; spec.color itself, and the emitted code, are untouched.
                self.w_color.value = to_hex(spec.color)
                if spec.shape in _SIZABLE:
                    self.w_rotation.value = dict(spec.params).get("rotation", 0)
                # `on_select` (Canvas._set_selected) fires *before* the commit that
                # triggered it has reached `_refresh_list()` (see _append_spec), so
                # `index` can briefly be out of range for the not-yet-rebuilt
                # `w_list.options` -- guard the same way `_refresh_list` does; the
                # second `_sync_controls()` call from `_on_commit`, after
                # `_refresh_list()` has run, sets it correctly.
                if self.w_list.value != index and index < len(self.w_list.options):
                    self.w_list.value = index
        finally:
            self._syncing = False

    # -- public, read-only views ---------------------------------------------------

    @property
    def specs(self) -> list[Spec]:
        return list(self.canvas.specs)

    @property
    def code(self) -> str:
        """The paste-able code block -- the canonical record of this session."""
        return specs_to_code(self.canvas.specs)

    @property
    def rois(self) -> list[ROI]:
        """Fresh `ROI` objects for every region (same-session convenience only -- the
        pasted `.code` is the permanent record)."""
        return [s.to_object() for s in self.canvas.specs]

    def __repr__(self) -> str:
        return (f"{type(self).__name__}({len(self.stack)} frames, "
               f"{len(self.canvas.specs)} regions, frame {self._frame_idx})")


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
        `Profile`s are not accepted -- see `uspy.plotting.waterfall` instead.
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
