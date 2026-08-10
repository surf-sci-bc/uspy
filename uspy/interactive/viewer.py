"""A minimal notebook stack viewer: an image and a frame slider, nothing else -- no
region drawing, no preview panel, no code box. Subclasses `uspy.interactive.shell.
FrameView`, which already implements frame display and the frame/contrast widgets for
the two region pickers; this module adds only a play/pause control, a metadata-overlay
toggle, and the layout around them.

Needs the 'interactive' extra (`pip install 'uspy[interactive]'`) and `%matplotlib
widget` run in an earlier notebook cell.
"""
from __future__ import annotations

from typing import Optional, Sequence

import ipywidgets as widgets
from IPython.display import display

import uspy.dataobject as do
from uspy.interactive.shell import FrameView


class StackViewer(FrameView):
    """Interactive viewer for a `DataObjectStack`: the current frame plus a slider,
    a play/pause control, a contrast dropdown, and a metadata-overlay toggle.

    Read-only -- it draws frames and nothing else. For drawing ROIs/profiles while
    scrubbing, see `uspy.interactive.pick_rois`/`pick_profiles`, which include the same
    frame/contrast controls plus a live preview panel.

    Constructed by `view_stack()`, which also `display()`s it.
    """

    def __init__(
        self, stack, contrast: str = "auto", fields: Optional[Sequence[str]] = None,
        frame: Optional[int] = None, interval: int = 200, figsize: tuple = (6, 6),
    ):
        if not isinstance(stack, do.DataObjectStack):
            raise ValueError(
                "StackViewer needs a DataObjectStack -- for a single image, plot it "
                "directly with uspy.plotting.plot_img instead."
            )
        self._initial_fields = fields
        self._interval = interval
        super().__init__(stack, frame=frame, contrast=contrast, fields=fields, figsize=figsize)

        self._build_widgets()
        self._wire_widgets()
        # see PickerBase.__init__: draw_idle() calls before display() are lost, since
        # the ipympl comm channel doesn't exist yet -- one explicit, synchronous draw
        # after display() guarantees a real frame reaches the frontend.
        self.fig.canvas.draw()

    # -- widgets ---------------------------------------------------------------

    def _build_widgets(self) -> None:
        w = widgets
        self._build_frame_controls()
        self.w_play = w.Play(min=0, max=max(0, len(self.stack) - 1), value=self._frame_idx,
                             interval=self._interval)
        self.w_fields = w.Checkbox(value=True, description="metadata", indent=False,
                                   layout=w.Layout(width="auto"))

        self.canvas_out = w.Output()
        with self.canvas_out:
            display(self.fig.canvas)

        row = w.Layout(flex_flow="row wrap", align_items="center")
        self.widget = w.VBox([
            w.HBox([self.w_play, self.w_frame], layout=w.Layout(width="100%")),
            w.HBox([self.w_contrast, self.w_fields], layout=row),
            self.canvas_out,
        ])

    def _wire_widgets(self) -> None:
        self._wire_frame_controls()
        # widgets.link (not jslink): each step already needs a kernel round trip to
        # re-plot the frame and push the new image to the browser, so a purely
        # browser-side link buys no responsiveness here -- link keeps the two
        # widgets' values in the kernel in sync too, which is what makes
        # w_frame.observe(_on_frame_change) fire for a step driven by the play button.
        widgets.link((self.w_play, "value"), (self.w_frame, "value"))
        self.w_fields.observe(self._on_fields_change, "value")

    def _on_fields_change(self, change: dict) -> None:
        self._fields = self._initial_fields if change["new"] else ()
        self._draw_frame()


def view_stack(
    stack, contrast: str = "auto", fields: Optional[Sequence[str]] = None,
    frame: Optional[int] = None, interval: int = 200, figsize: tuple = (6, 6),
) -> StackViewer:
    """Open the interactive stack viewer on `stack` and display it.

    Displays exactly one widget, whether or not the call's return value is used --
    left unassigned in a notebook cell, the trailing `Out[]` echo of the returned
    `StackViewer` is a one-line `repr()`, not a second copy of the widget.

    Requires `%matplotlib widget` (the ipympl backend) to have been run in an earlier
    cell, and the 'interactive' extra installed (`pip install 'uspy[interactive]'`).

    Parameters
    ----------
    stack : DataObjectStack
    contrast, fields
        As in `uspy.plotting.plot_img`, plus `contrast="fixed"`: freeze the *currently
        displayed* frame's intensity range instead of renormalizing on every frame
        change -- useful to see genuine intensity changes across the stack. `fields`
        defaults to `stack[0].default_fields`, same as `plot_img`.
    frame : int, optional
        Initial displayed frame index. Defaults to the middle frame.
    interval : int, optional
        Milliseconds between frames while playing, by default 200.
    figsize : tuple, optional
        As in `uspy.plotting.plot_img`.

    Returns
    -------
    StackViewer

    Notes
    -----
    On a virtual stack, each displayed frame is read from disk on first access and
    never cached (see `uspy.dataobject.deferred.DeferredArray`) -- scrubbing re-reads
    one frame per step, which is fine; playback re-reads one frame per `interval`.
    """
    viewer = StackViewer(stack, contrast=contrast, fields=fields, frame=frame,
                         interval=interval, figsize=figsize)
    display(viewer.widget)
    return viewer
