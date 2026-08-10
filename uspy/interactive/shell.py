"""Shared ipywidgets shell for the interactive pickers and the stack viewer.

`FrameView` owns everything about displaying one frame of a `DataObject`/
`DataObjectStack` source: source resolution (`is_stack`/`stack`/`frames`/`height`/
`width`), figure/backend setup, the frame-index/contrast/fields state and redraw
(`_draw_frame`), and the frame-slider/contrast-dropdown widgets. `PickerBase` builds on
it, adding `uspy.interactive.canvas.Canvas` (the draggable-artist layer, pure
matplotlib), label/color/live-preview controls, and a preview panel a subclass owns the
content of. `uspy.interactive.viewer.StackViewer` builds on `FrameView` directly --
just the frame and nothing else.

`uspy.interactive.picker.ROIPicker` (I(V) over an ROI) and `uspy.interactive.profile.
ProfilePicker` (a 1D profile / waterfall over a `Profile`) are the two concrete
pickers; both subclass `PickerBase` here and differ only in: which shapes `Canvas`
accepts, what extra widgets sit in the tool row, and what the preview panel draws. See
each hook's docstring below for the exact contract a subclass must satisfy.

Needs the 'interactive' extra (`pip install 'uspy[interactive]'`) and `%matplotlib
widget` run in an earlier notebook cell. `uspy.interactive.spec`/`.canvas` do not need
either -- only this module (and its subclasses) do.
"""
from __future__ import annotations

import time
import warnings
from typing import Optional, Sequence

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import ipywidgets as widgets
from IPython.display import display

import uspy.dataobject as do
import uspy.plotting as plot
from uspy.interactive.canvas import Canvas
from uspy.interactive.spec import Spec, specs_to_code

_MIN_MPL_VERSION = (3, 6)


def _check_backend(fig) -> None:
    """ipympl's canvas is the only one with `header_visible`; anything else means
    `%matplotlib widget` was not run, and the figure would just be a static image with
    no working mouse/key events."""
    if not hasattr(fig.canvas, "header_visible"):
        plt.close(fig)
        raise RuntimeError(
            "The interactive picker needs the ipympl backend. Run `%matplotlib "
            f"widget` in a cell before calling it (current backend: {mpl.get_backend()})."
        )


def _check_mpl_version() -> None:
    major, minor = mpl.__version__.split(".")[:2]
    if (int(major), int(minor)) < _MIN_MPL_VERSION:
        raise RuntimeError(
            f"The interactive picker needs matplotlib>={'.'.join(map(str, _MIN_MPL_VERSION))} "
            f"(found {mpl.__version__}) for rotatable selection handles."
        )


def _compact(width: str, description_width: str) -> dict:
    """`layout`/`style` kwargs sizing a captioned widget to its content, instead of
    ipywidgets' default ~300px control box with an ~80px description column."""
    return {"layout": widgets.Layout(width=width),
            "style": {"description_width": description_width}}


class FrameView:
    """Owns everything about displaying one frame of a `DataObject`/`DataObjectStack`
    source on a matplotlib axes: source resolution (`is_stack`/`stack`/`frames`/
    `height`/`width`), figure/backend setup, the frame-index/contrast/fields state and
    redraw (`_draw_frame`), and the frame-slider/contrast-dropdown widgets
    (construction/wiring only -- a subclass places them in its own layout).

    `is_stack` is `isinstance(source, uspy.dataobject.DataObjectStack)`; for a single
    `DataObject`, `self.stack` is None and `self.frames` is a length-1 sequence.

    A subclass provides:

    - `_setup()`: called once `self.stack`/`self.frames`/`self.height`/`self.width`
      are set, before any figure exists -- the place to precompute/cache whatever else
      the subclass needs (e.g. `ROIPicker`'s per-frame x-axis array).
    - `_make_figure(figsize)`: create `self.fig` and its axes; must set `self.ax_img`.
      Defaults to a single axes.
    """

    def __init__(
        self, source, frame: Optional[int] = None, contrast: str = "auto",
        fields: Optional[Sequence[str]] = None, figsize: tuple = (11, 4.5),
    ):
        _check_mpl_version()

        self.is_stack = isinstance(source, do.DataObjectStack)
        self.stack = source if self.is_stack else None
        self.frames = source if self.is_stack else [source]
        self.height, self.width = self.frames[0].shape
        self._setup()

        with plt.ioff():
            self._make_figure(figsize)
        _check_backend(self.fig)
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False

        self._frame_idx = frame if frame is not None else (len(self.stack) // 2 if self.is_stack else 0)
        self._contrast = contrast
        self._fields = fields
        self._base_artists: list = []
        self._draw_frame()

    # -- subclass hooks -----------------------------------------------------------

    def _setup(self) -> None:
        pass

    def _make_figure(self, figsize: tuple) -> None:
        self.fig, self.ax_img = plt.subplots(figsize=figsize)

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
        img = self.stack[self._frame_idx] if self.is_stack else self.frames[0]
        plot.plot_img(img, ax=self.ax_img, ticks=True, contrast=self._contrast,
                      fields=self._fields, title=False)
        self._base_artists = [a for a in self.ax_img.get_children() if id(a) not in before]
        self.fig.canvas.draw_idle()

    # -- frame/contrast widgets -----------------------------------------------------

    def _build_frame_controls(self) -> None:
        """Construct (but do not place) the frame slider and contrast dropdown --
        stack only; a subclass calls this behind its own `if self.is_stack` and places
        the widgets in its own layout."""
        w = widgets
        # continuous_update=True: unlike a drag on the canvas (many motion events per
        # second, throttled by Canvas._REDRAW_MIN_INTERVAL), a slider only fires one
        # event per integer frame step, and each just re-reads one frame and re-plots
        # it -- cheap enough to update live.
        self.w_frame = w.IntSlider(
            min=0, max=max(0, len(self.stack) - 1), value=self._frame_idx, description="frame",
            continuous_update=True, layout=w.Layout(width="99%"),
            style={"description_width": "initial"},
        )
        self.w_contrast = w.Dropdown(options=["auto", "inner", "fixed"], value="auto",
                                     description="contrast", **_compact("140px", "60px"))

    def _wire_frame_controls(self) -> None:
        self.w_frame.observe(self._on_frame_change, "value")
        self.w_contrast.observe(self._on_contrast_change, "value")

    def _on_frame_change(self, change: dict) -> None:
        self._frame_idx = change["new"]
        self._draw_frame()

    def _on_contrast_change(self, change: dict) -> None:
        value = change["new"]
        if value == "fixed":
            # freeze *this* frame's range as a plain (lo, hi) plot_img already
            # accepts -- computed from the one displayed frame, never the whole
            # stack, so this stays cheap on a virtual stack.
            img = self.stack[self._frame_idx] if self.is_stack else self.frames[0]
            data = np.nan_to_num(img.image)
            value = (float(data.min()), float(data.max()))
        self._contrast = value
        self._draw_frame()

    def __repr__(self) -> str:
        if self.is_stack:
            return f"{type(self).__name__}({len(self.stack)} frames, frame {self._frame_idx})"
        return f"{type(self).__name__}()"


class PickerBase(FrameView):
    """Owns everything about the picker shell that is not specific to what shape is
    being drawn or what the preview panel shows: the second (preview) axes, the
    live/auto/step controls (only relevant, and only built, for a stack -- see
    `is_stack` on `FrameView`), the region list, the code box + copy button, and the
    label/color editing that applies to any region regardless of shape. Frame display
    itself (the image axes, the frame slider, the contrast dropdown) is `FrameView`'s.

    A subclass provides:

    - `_shapes` (class attribute): the shapes `Canvas` accepts (e.g. `ROI_SHAPES` or
      `("profile",)`).
    - `_reject_seed(region)`: raise `ValueError` for a seed region of the wrong kind
      (an `ROI` handed to the profile picker, or vice versa). No-op by default.
    - `_setup()`: see `FrameView`.
    - `_update_preview(drag=None)`: (re)draw the preview panel (`self.ax_preview`).
      `drag=(index, draft_spec)` during an active live drag.
    - `_extra_controls() -> list[Widget]`: widgets appended to the shared label/color
      row (e.g. the shape button row, a rotation box, a profile-width box).
    - `_prefix_children() -> list[Widget]`: widgets prepended above that row (e.g. the
      shape-button CSS injection).
    - `_wire_extra()` / `_sync_extra(spec_or_None)`: wiring/loading for whatever
      `_extra_controls` built.

    For a single `DataObject`, every stack-only widget (frame slider, contrast, step,
    live, auto, update) is simply never built -- code that reads them must be behind
    an `if self.is_stack` (as `_on_commit`/`_wire_widgets` here already are).
    """

    _shapes: tuple = ()
    _STATUS_TEXT: str = ""
    _LIVE_LABEL: str = "live"
    _LIVE_MIN_INTERVAL = 0.05  # seconds; see _on_drag
    #: cap on how many frames a live-drag preview samples, regardless of stack size --
    #: see _live_frame_indices. The exact, full-resolution result is always rebuilt on
    #: release (_on_commit), so this only ever affects what's shown mid-drag.
    _LIVE_MAX_FRAMES = 150

    def _make_figure(self, figsize: tuple) -> None:
        self.fig, (self.ax_img, self.ax_preview) = plt.subplots(1, 2, figsize=figsize)
        # legacy alias: ROIPicker's own test suite reaches into `.ax_iv` by name --
        # both names point at the same Axes.
        self.ax_iv = self.ax_preview

    def __init__(
        self, source, xaxis: str, regions: Optional[Sequence] = None,
        live: Optional[bool] = None, frame: Optional[int] = None, contrast: str = "auto",
        figsize: tuple = (11, 4.5),
    ):
        for region in regions or ():
            self._reject_seed(region)
        self._axis_field = xaxis  # read by _setup(), called from FrameView.__init__

        super().__init__(source, frame=frame, contrast=contrast, figsize=figsize)

        is_virtual = bool(self.is_stack and getattr(source, "virtual", False))
        if live is None:
            live = not is_virtual
        elif live and is_virtual:
            warnings.warn(
                "live preview on a virtual stack re-reads every frame from disk on "
                "each drag motion event -- consider stack.materialize() first, or "
                "leave live off and use the Update button instead.",
                UserWarning, stacklevel=3,
            )
        self.live = live

        self._clock = time.monotonic
        self._last_live_update = 0.0

        self._syncing = False  # guards widget<->canvas write-back loops, see _sync_controls
        self.canvas = Canvas(self.ax_img, self.height, self.width,
                             on_commit=self._on_commit, on_drag=self._on_drag,
                             on_select=self._on_select_changed, shapes=self._shapes)
        if regions:
            self.canvas.specs = [Spec.from_roi(r) for r in regions]
            self.canvas.redraw_overlays()

        self._build_widgets()
        self._wire_widgets()
        self._refresh_code()
        self._refresh_list()
        self._sync_controls()
        self._update_preview()
        # Everything above (_draw_frame, Canvas's initial redraw_overlays,
        # _update_preview) draws before self.fig.canvas is ever display()-ed inside
        # _build_widgets(), i.e. before the ipympl comm channel to the browser exists
        # -- those early draw_idle() calls are effectively lost, leaving the canvas
        # blank until some later event (e.g. a click) happens to trigger another one.
        # One explicit, synchronous draw() here, after display(), guarantees a real
        # frame is sent once the frontend can actually receive it.
        self.fig.canvas.draw()

    # -- subclass hooks -----------------------------------------------------------

    def _reject_seed(self, region) -> None:
        pass

    def _update_preview(self, drag: Optional[tuple] = None) -> None:
        raise NotImplementedError

    def _extra_controls(self) -> list:
        return []

    def _prefix_children(self) -> list:
        return []

    def _wire_extra(self) -> None:
        pass

    def _sync_extra(self, spec: Optional[Spec]) -> None:
        pass

    # -- Canvas callbacks -------------------------------------------------------

    def _on_commit(self) -> None:
        if not self.is_stack or self.w_auto.value:
            self._update_preview()
        self._refresh_code()
        self._refresh_list()
        self._sync_controls()
        if self.canvas.status:
            self.w_status.value = self.canvas.status
            self.canvas.status = ""

    def _on_drag(self, index: int, spec: Spec) -> None:
        """Recompute the preview mid-drag, throttled to `_LIVE_MIN_INTERVAL`. The
        throttle is measured from when the *previous* recompute finished, not from
        when this event arrived -- if a single `_update_preview` call takes longer
        than the interval (a large/slow stack), motion events queue up in the kernel
        while it's busy; stamping the clock only after the work completes means the
        next allowed call is still a full interval away, instead of immediately
        passing the check and re-triggering another slow rebuild back-to-back for as
        long as the mouse keeps moving."""
        if not self.live:
            return
        now = self._clock()
        if now - self._last_live_update < self._LIVE_MIN_INTERVAL:
            return
        self._update_preview(drag=(index, spec))
        self._last_live_update = self._clock()

    def _live_frame_stride(self, n_frames: int) -> Optional[int]:
        """Integer stride that brings a mid-drag preview's frame count down to (at
        most) `_LIVE_MAX_FRAMES` -- `None` if `n_frames` is already at or under the
        cap (the common case: no subsampling, identical to today).

        A stride, not an arbitrary index list: the result is meant to be used as a
        plain slice, `stack[::stride]` -- `DataObjectStack.__getitem__` already
        builds a correctly-typed sub-stack from a slice (virtual or resident, no
        pairwise-compatibility surprises), so there's no need to reconstruct
        anything here, and both a stack and a plain array of the same original
        length can be sliced identically (`self._x[::stride]` alongside
        `self.stack[::stride]`) to keep an x-axis in sync -- see `ROIPicker.
        _update_iv_preview`.

        Bounded worst case for a very large stack; the exact, full-resolution
        result is always rebuilt on release via the unchanged non-drag path
        (`_on_commit` -> `_update_preview()`), so nothing pasted, saved, or
        returned by a subclass's `.rois`/`.profiles`/`.code` is ever affected --
        preview-only, same contract as `ROIPicker._live_intensity`."""
        if n_frames <= self._LIVE_MAX_FRAMES:
            return None
        return -(-n_frames // self._LIVE_MAX_FRAMES)  # ceil division

    def _on_select_changed(self, index: Optional[int]) -> None:
        """`Canvas.on_select` -- fires when a click on the canvas itself (as opposed
        to picker.py's own list/buttons) changes the selection, so the label/color
        controls (and, for a subclass whose preview follows the selection, the
        preview itself) follow it either way."""
        self._sync_controls()

    # -- widgets ---------------------------------------------------------------

    def _build_widgets(self) -> None:
        w = widgets
        self.w_label = w.Text(description="label", **_compact("200px", "40px"))
        self.w_color = w.ColorPicker(concise=True, value=to_hex(self.canvas.color_for_new()),
                                     description="color", **_compact("90px", "40px"))
        extra = self._extra_controls()

        if self.is_stack:
            self._build_frame_controls()
            self.w_step = w.BoundedIntText(min=1, max=max(1, len(self.stack)), value=1,
                                           description="every n-th", **_compact("150px", "75px"))
            _check_layout = w.Layout(width="auto", margin="0 12px 0 0")
            self.w_live = w.Checkbox(value=self.live, description=self._LIVE_LABEL, indent=False,
                                     layout=_check_layout)
            self.w_auto = w.Checkbox(value=True, description="auto preview", indent=False,
                                     layout=_check_layout)
            self.w_update = w.Button(description="Update", layout=w.Layout(width="90px"))

        self.w_delete = w.Button(description="Delete selected", layout=w.Layout(width="130px"))
        self.w_clear = w.Button(description="Clear all", layout=w.Layout(width="90px"))
        self.w_list = w.Select(options=[], description="regions", rows=6)

        self.w_status = w.Label(value=self._STATUS_TEXT)
        self.w_code = w.Textarea(value="", layout=w.Layout(width="100%", height="140px"))
        self.w_copy = w.Button(description="Copy code")
        self._copy_out = w.Output(layout=w.Layout(height="0px", overflow="hidden"))

        self.canvas_out = w.Output()
        with self.canvas_out:
            display(self.fig.canvas)

        # flex_flow="row wrap": lets a narrow notebook wrap the row instead of
        # overflowing, now that each control is sized to its content rather than
        # padding out a fixed-width row.
        row = w.Layout(flex_flow="row wrap", align_items="center")
        children = list(self._prefix_children())
        children.append(w.HBox([self.w_label, self.w_color, *extra, self.w_delete, self.w_clear],
                               layout=row))
        if self.is_stack:
            children.append(w.HBox([self.w_frame], layout=w.Layout(width="100%")))
            children.append(w.HBox([self.w_contrast, self.w_step, self.w_live, self.w_auto,
                                    self.w_update], layout=row))
        children.append(w.HBox([self.canvas_out, self.w_list]))
        children.append(self.w_status)
        children.append(w.HBox([self.w_code, self.w_copy]))
        children.append(self._copy_out)
        self.widget = w.VBox(children)

    def _wire_widgets(self) -> None:
        self.w_label.observe(self._on_label_change, "value")
        self.w_color.observe(self._on_color_change, "value")

        if self.is_stack:
            self._wire_frame_controls()
            self.w_live.observe(lambda ch: setattr(self, "live", ch["new"]), "value")
            self.w_update.on_click(lambda _btn: self._update_preview())

        self.w_delete.on_click(lambda _btn: self._delete_selected())
        self.w_clear.on_click(lambda _btn: self._clear_all())
        self.w_list.observe(self._on_list_select, "value")
        self.w_copy.on_click(self._on_copy)

        self._wire_extra()

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

    def _on_list_select(self, change: dict) -> None:
        if self._syncing:
            return
        index = change["new"]
        if index is None:
            return
        self.canvas.select(index)  # on_select -> _sync_controls loads label/color/extras

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
        """Point the label/color row (and, via `_sync_extra`, whatever a subclass
        added) at whatever is currently selected, so a region's data can be edited
        after it's drawn instead of only at creation time. With a selection, load that
        region's label/color into the widgets (editing them then calls `Canvas.
        set_label`/`set_color` -- see `_on_label_change` etc.); with no selection, show
        and reset the *next region's* defaults instead, so an explicit color picked
        for one region doesn't linger and silently override the color cycle for every
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
                self._sync_extra(None)
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
                self._sync_extra(spec)
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
    def specs(self) -> list:
        return list(self.canvas.specs)

    @property
    def code(self) -> str:
        """The paste-able code block -- the canonical record of this session."""
        return specs_to_code(self.canvas.specs)

    def __repr__(self) -> str:
        if self.is_stack:
            return (f"{type(self).__name__}({len(self.stack)} frames, "
                   f"{len(self.canvas.specs)} regions, frame {self._frame_idx})")
        return f"{type(self).__name__}({len(self.canvas.specs)} regions)"
