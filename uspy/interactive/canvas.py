"""Draggable matplotlib artists for the interactive ROI/Profile picker: every picked
region rendered live and independently editable at once (move, resize, rotate,
per-vertex polygon/profile editing) -- unlike `matplotlib.widgets`' selectors, which
only ever keep one region live at a time (see the picker's design notes for why that
ruled them out).

Driven entirely by raw matplotlib mouse/key events, not ipywidgets, so the whole drag
state machine is importable and testable headlessly under the Agg backend with
synthetic `MouseEvent`/`KeyEvent`s -- no notebook, no ipympl, no optional extra needed
for this module specifically (only `picker.py`, which wraps a `Canvas` in the
ipywidgets shell, needs those).
"""
from __future__ import annotations

import math
import time
from typing import Callable, Optional

import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle, Polygon, PathPatch

from uspy.interactive.spec import (
    SHAPES,
    Spec,
    _SIZABLE,
    spec_from_drag,
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
    with_width,
)
from uspy.roi import palette_color

#: distance (data units) the rotate handle sits out from a shape's centre
HANDLE_DISTANCE = 30
#: hit-test tolerance for handles/vertices, in *display* pixels (zoom-independent)
GRAB_RANGE = 8
#: hit-test tolerance for a polygon edge (double-click-to-insert-vertex), data units
EDGE_TOLERANCE = 6
#: handle/vertex marker size, points (mpl markersize) -- adjust here to resize every
#: handle at once. Independent of GRAB_RANGE, which is the hit-test radius in display
#: pixels, not the marker's drawn size.
HANDLE_SIZE = 4
#: marker size for a "point" region, points
POINT_MARKER_SIZE = 10
#: default proximity tolerance for `_point_near_segment`, data units
_SEGMENT_TOLERANCE = 4
#: shift+arrow -> (dx, dy) nudge, in data units. Shift-modified so matplotlib's own
#: left/right view-history bindings (rcParams["keymap.back"]/["keymap.forward"],
#: matched against the exact event.key string) still fire on a *plain* arrow and never
#: double up with a nudge -- no keymap surgery needed. dy is negated for "up": data-y
#: is the image row, increasing downward (imshow origin upper, same as everywhere else
#: in this codebase).
_NUDGE = {
    "shift+left": (-1, 0), "shift+right": (1, 0),
    "shift+up": (0, -1), "shift+down": (0, 1),
}
#: keys that delete the selected region (or its selected vertex). "backspace" is
#: deliberately NOT here: it is in rcParams["keymap.back"], and matplotlib's own
#: key_press_handler is connected to the same key_press_event by FigureManagerBase on
#: every managed figure -- so binding it here made deleting a region also rewind the
#: plot to its previous view (the same class of clash "shift+" avoids for _NUDGE
#: above). "d" is unbound in every default keymap. "delete" (forward-delete) is kept:
#: it collides with nothing, and is the natural key on a keyboard that has one -- a
#: Mac's Delete key reports as "backspace", which is why "d" exists as an alternative.
_DELETE_KEYS = ("d", "delete")
#: minimum interval between canvas redraws while dragging, seconds. A drag fires far
#: more motion events than this (a browser/ipympl can emit dozens per second while a
#: button is held); matplotlib's ipympl/webagg backend sends a full-figure re-render
#: over the websocket on *every* draw_idle() call with no debouncing of its own
#: (confirmed by reading backend_webagg_core.FigureCanvasWebAggCore.draw_idle), so an
#: unthrottled redraw-per-event queues a backlog of full re-renders that can only
#: drain sequentially -- this is what made the picker unusable (drags took 20-30s to
#: settle). The Spec draft itself is still recomputed on every raw event (cheap, pure
#: Python), so the final position on release is always exact regardless of how many
#: intermediate redraws this throttle skipped.
_REDRAW_MIN_INTERVAL = 0.07


def _modifiers(event) -> set:
    """Keyboard modifiers held during a mouse event, normalized to a set containing
    zero or more of {"ctrl", "shift"}. ipympl/webagg's `MouseEvent.modifiers` (added
    in mpl 3.8, forwarded from the browser's own event.modifiers) is the primary
    source; `event.key`, which the Agg-backed tests set instead (mirroring how
    matplotlib's own interactive backends report a plain keypress), is also checked so
    this works identically in both. `"control"` (mpl's own spelling) is normalized to
    `"ctrl"`."""
    mods = {m.replace("control", "ctrl") for m in (getattr(event, "modifiers", None) or ())}
    mods.update(p.replace("control", "ctrl")
               for p in (getattr(event, "key", None) or "").split("+"))
    return mods


def _safe_hex(color) -> Optional[str]:
    """`matplotlib.colors.to_hex(color)`, or `None` if `color` isn't something
    matplotlib can parse as a color. Used only to de-duplicate already-used colors in
    `Canvas.auto_color()` -- `Spec.color` is whatever a caller/notebook wrote (see
    `Spec.from_roi`), so it must never be assumed parseable; an unparseable color
    should just occupy no palette slot, not break auto-cycling."""
    try:
        return mcolors.to_hex(color)
    except (ValueError, TypeError):
        return None


class Canvas:
    """Owns the draggable region layer on one matplotlib Axes.

    `specs` is the canonical, committed list of regions (mirrors `ROIPicker.specs`).
    `new_shape`/`new_color`/`new_label`/`new_width` are plain attributes the caller
    (picker.py's widgets, or a test) sets to control what a click/drag on empty canvas
    creates next -- kept as plain attributes rather than constructor-only arguments so
    picker.py can bind them straight to widget `.value`s with no extra glue.
    `new_color=None` (the default) means auto-cycle: `color_for_new()` then hands out
    the lowest `uspy.roi.palette_color` index not already used by a committed region --
    the same palette non-interactive uspy uses for implicit `ROI`/`Profile` colors, so
    the picker's regions come out C0/C1/C2/... the same way a notebook's would.

    Does not draw the base image frame -- that is `plot_img`'s job (called by
    picker.py), reused rather than reimplemented (CLAUDE.md: keep code focused, reuse
    the codebase).

    Drawing a new region drags out from the click point as its centre, matching
    `ROI`'s own centre+extents convention (see `spec.spec_from_drag`) -- hold Ctrl for
    the alternate corner-to-corner bounding-box gesture, or Shift to force width ==
    height (always on for "circle", which is inherently square). The same Shift ==
    square constraint applies to dragging an existing region's corner resize handle
    (see `spec.resize_from_handle`'s `square` argument) -- a circle resized this way
    stays a circle instead of degrading to an ellipse. A toolbar zoom/pan drag never
    also creates or moves a region (`_nav_active`).
    """

    def __init__(
        self, ax: Axes, height: int, width: int,
        on_commit: Optional[Callable[[], None]] = None,
        on_drag: Optional[Callable[[int, Spec], None]] = None,
        on_select: Optional[Callable[[Optional[int]], None]] = None,
        shapes: tuple = SHAPES,
    ):
        self.ax = ax
        self.height, self.width = height, width
        self.specs: list[Spec] = []
        #: shapes `new_shape` may be set to -- e.g. the I(V) picker restricts this to
        #: `uspy.interactive.spec.ROI_SHAPES` so a Profile (no mask, no I(V) curve)
        #: can never be drawn there; a future waterfall-style picker would pass the
        #: full `SHAPES` (or just `("profile",)`) instead.
        self.shapes = tuple(shapes)
        self._new_shape = "circle" if "circle" in self.shapes else self.shapes[0]
        self.new_color: Optional[str] = None  # None -> auto-cycle, see color_for_new()
        self.new_label: Optional[str] = None
        self.new_width = 1  # profile linewidth for newly drawn profiles

        #: called with no args once `self.specs` has settled (add/edit/delete commit)
        self.on_commit = on_commit
        #: called with (index, draft_spec) on every motion event of a drag that is
        #: editing an *existing* region (not while drawing a brand new one) -- this is
        #: what drives the picker's live I(V) preview
        self.on_drag = on_drag
        #: called with the new selection (an index, or None) whenever it *changes* --
        #: lets picker.py's widgets follow clicks on the canvas, not just its own
        #: list/buttons
        self.on_select = on_select

        self.status: str = ""  # last rejection/info message, for picker.py to surface

        self._clock = time.monotonic  # overridable in tests
        self._last_redraw = 0.0

        self._overlay_artists: list = []
        self._outline_artists: dict = {}
        self._drag: Optional[dict] = None
        self._pending_points: list[tuple[float, float]] = []
        self._selected: Optional[int] = None
        self._selected_vertex: Optional[int] = None

        self._cids = [
            ax.figure.canvas.mpl_connect("button_press_event", self._on_press),
            ax.figure.canvas.mpl_connect("motion_notify_event", self._on_move),
            ax.figure.canvas.mpl_connect("button_release_event", self._on_release),
            ax.figure.canvas.mpl_connect("key_press_event", self._on_key),
        ]
        self.redraw_overlays()

    def close(self) -> None:
        """Disconnect all event handlers. Call when the picker widget is torn down."""
        for cid in self._cids:
            self.ax.figure.canvas.mpl_disconnect(cid)

    @property
    def new_shape(self) -> str:
        """Shape a click/drag on empty canvas creates next."""
        return self._new_shape

    @new_shape.setter
    def new_shape(self, value: str) -> None:
        if value not in self.shapes:
            raise ValueError(f"{value!r} is not one of this canvas's allowed shapes {self.shapes}")
        self._new_shape = value

    # -- programmatic editing, for picker.py's widgets (as opposed to mouse/key
    # -- driven editing above) -----------------------------------------------

    @property
    def selected(self) -> Optional[int]:
        """Index of the currently selected region, or None."""
        return self._selected

    def select(self, index: Optional[int]) -> None:
        """Select a region by index (e.g. from a region-list widget), revealing its
        handles the same way clicking it would."""
        self._set_selected(index)
        self._draw_outlines()

    def delete_selected(self) -> None:
        """Remove the whole selected region. A no-op if nothing is selected."""
        if self._selected is None:
            return
        del self.specs[self._selected]
        self._set_selected(None)
        self.redraw_overlays()
        self._fire_commit()

    def clear(self) -> None:
        """Remove every region."""
        self.specs = []
        self._set_selected(None)
        self.redraw_overlays()
        self._fire_commit()

    def auto_color(self) -> str:
        """The next auto-cycled color, always as `#rrggbb`: the lowest-index
        `palette_color` not currently used by any committed region, so region 0 = C0,
        region 1 = C1, and deleting a region frees its color back up for the next one
        drawn -- the same palette `uspy.roi.StyledObject` hands out implicit colors
        from, but derived from the current regions rather than a session-global
        counter, so it stays reproducible across re-runs of the picker. Falls back to
        continuing the cycle by position once every palette color is already in use
        (matching how the palette wraps everywhere else in uspy) rather than search
        forever.

        Normalized through `matplotlib.colors.to_hex` (a picker-*generated* color, so
        hex is the right canonical form) because `palette_color` can return anything
        matplotlib accepts as a color -- not just hex, e.g. a custom `axes.prop_cycle`
        can hold float-RGB tuples or CSS names -- while picker.py hands this straight
        to an `ipywidgets.ColorPicker`, which only accepts HTML colors (`#rrggbb` or a
        CSS name) and raises `TraitError` on anything else (see the regression this
        fixes: a tuple-valued `prop_cycle` crashed `pick_rois()` outright). `to_hex`
        rounds each channel to 1/255, so for an exotic float palette the auto-assigned
        color can differ from the raw cycler value by less than one 8-bit step --
        imperceptible, and this is the only place uspy ever re-derives a color from
        `palette_color`, so it never touches a color the user actually typed in.
        """
        used = {_safe_hex(spec.color) for spec in self.specs} - {None}
        n = len(self.specs) + 1
        for i in range(n):
            color = mcolors.to_hex(palette_color(i))
            if color not in used:
                return color
        return mcolors.to_hex(palette_color(len(self.specs)))

    def color_for_new(self) -> str:
        """The color the next newly-drawn region will get: `new_color` if the caller
        set one explicitly, otherwise the next auto-cycled color."""
        return self.new_color or self.auto_color()

    def _set_selected(self, index: Optional[int], vertex: Optional[int] = None) -> None:
        """Set `_selected`/`_selected_vertex` and fire `on_select` iff the *selected
        region* actually changed (not just the vertex within it) -- the single
        chokepoint every selection mutation below goes through, so picker.py's widgets
        can stay in sync with clicks on the canvas itself, not just its own list/
        buttons."""
        changed = index != self._selected
        self._selected = index
        self._selected_vertex = vertex
        if changed and self.on_select is not None:
            self.on_select(index)

    def set_rotation(self, index: int, degrees: float) -> None:
        """Set a region's rotation to an exact value (e.g. from a typed-in degrees
        box), independent of the rotate handle drag."""
        self.specs[index] = with_rotation(self.specs[index], degrees)
        self.redraw_overlays()
        self._fire_commit()

    def set_width(self, index: int, width: int) -> None:
        """Set an existing profile's linewidth to an exact value (e.g. from a typed-in
        width box) -- the counterpart to `new_width`, which only affects a profile not
        yet drawn."""
        self.specs[index] = with_width(self.specs[index], width)
        self.redraw_overlays()
        self._fire_commit()

    def set_label(self, index: int, label: Optional[str]) -> None:
        spec = self.specs[index]
        self.specs[index] = Spec(spec.shape, spec.params, spec.color, label or None)
        self.redraw_overlays()
        self._fire_commit()

    def set_color(self, index: int, color: str) -> None:
        spec = self.specs[index]
        self.specs[index] = Spec(spec.shape, spec.params, color, spec.label)
        self.redraw_overlays()
        self._fire_commit()

    def nudge_selected(self, dx: int, dy: int) -> None:
        """Move the selected region by `(dx, dy)` -- or, if one of its vertices is
        selected (polygon/profile), just that vertex, leaving the others fixed. A
        no-op with nothing selected. Driven by shift+arrow (`_on_key`), but public so
        picker.py or a test can call it directly."""
        if self._selected is None:
            return
        spec = self.specs[self._selected]
        if self._selected_vertex is not None:
            points = vertex_positions(spec)
            x, y = points[self._selected_vertex]
            self.specs[self._selected] = move_vertex(
                spec, self._selected_vertex, (x + dx, y + dy))
        else:
            self.specs[self._selected] = translate(spec, dx, dy)
        self.redraw_overlays()
        self._fire_commit()

    # -- rendering --------------------------------------------------------------

    def redraw_overlays(self) -> None:
        """Re-rasterize every committed region's true mask from its Spec. Call
        whenever `self.specs` changes (add/delete/commit) -- NOT on every drag motion,
        which only needs the cheap outline artists redrawn instead (`_draw_outlines`).
        Profiles have no raster mask; `Profile.plot`'s own PathPatch outline stands in
        for both roles.
        """
        for artist in self._overlay_artists:
            artist.remove()
        self._overlay_artists = []
        for spec in self.specs:
            if spec.shape == "profile":
                continue
            obj = spec.to_object()
            mask = obj.pad_to(self.height, self.width)
            image = self.ax.imshow(
                mask, cmap=obj.cmap, alpha=obj.style.get("alpha", 0.5),
                interpolation="none", zorder=1,
            )
            self._overlay_artists.append(image)
        self._draw_outlines()

    def _draw_outlines(self, draft_only: bool = False) -> None:
        """(Re)draw region outlines (+ handles on the selected one), plus any
        in-progress draft/pending-chain preview. Artists are tracked per-key in
        `self._outline_artists` (`"_draft"`/`"_pending"` for the in-progress preview,
        a spec's list index for its committed outline) so a targeted redraw can
        remove and rebuild *only* the key that actually changed, leaving every other
        region's `Line2D`/`Patch` objects completely untouched.

        `draft_only=True` -- used for every motion event of a drag -- rebuilds only
        `"_draft"`/`"_pending"`. This, plus the caller-side throttle in `_on_move`, is
        what makes a drag cheap: previously this rebuilt *every* region's artists and
        called `canvas.draw_idle()` (a full-figure re-render + websocket push, with no
        debouncing at the matplotlib/ipympl layer -- see `_REDRAW_MIN_INTERVAL`) on
        every single motion event, regardless of how many regions existed or whether
        they were even involved in the drag.

        `draft_only=False` (the default) does a full rebuild -- for structural
        changes: the specs list itself changing, or the selection moving to a
        different region (which needs both the previously- and newly-selected
        region's handles redrawn). Not high-frequency, so the extra work is fine.
        """
        if draft_only:
            keys = {"_draft", "_pending"}
        else:
            # the union with the *previous* keys ensures a shrunk specs list (e.g.
            # after a delete) also clears the now-orphaned higher-index entries,
            # instead of leaving their artists stranded on the axes.
            keys = set(self._outline_artists) | set(range(len(self.specs))) | {"_draft", "_pending"}
        for key in keys:
            for artist in self._outline_artists.pop(key, ()):
                artist.remove()
            built = self._build_outline_for(key)
            if built:
                self._outline_artists[key] = built
        self.ax.figure.canvas.draw_idle()

    def _build_outline_for(self, key) -> list:
        if key == "_draft":
            if self._drag is not None and self._drag.get("draft") is not None:
                # skip the width band while dragging: it's not needed for the
                # gesture and recomputing/re-filling a PathPatch on every throttled
                # redraw is unnecessary work during the highest-frequency artist
                # rebuild in the picker.
                return self._build_outline(self._drag["draft"], selected=True, band=False)
            return []
        if key == "_pending":
            return self._build_pending_outline() if self._pending_points else []
        if key >= len(self.specs):
            return []
        # while this region is being actively edit-dragged, its live position is
        # shown via the "_draft" key instead -- skip the stale committed-position
        # outline so a drag doesn't show two overlapping outlines/handle sets.
        if (self._drag is not None and self._drag.get("kind") != "new-bbox"
                and self._drag.get("index") == key):
            return []
        return self._build_outline(self.specs[key], selected=(key == self._selected))

    def _build_outline(self, spec: Spec, selected: bool, band: bool = True) -> list:
        artists: list = []
        color = spec.color
        geom = dict(spec.params)
        if spec.shape == "point":
            artists.append(self._marker(geom["x0"], geom["y0"], color, marker="+",
                                        size=POINT_MARKER_SIZE))
        elif spec.shape in _SIZABLE:
            x0, y0 = geom["x0"], geom["y0"]
            rotation = geom.get("rotation", 0)
            # matplotlib's Patch `angle=` and ROI's `rotation=` are opposite-signed --
            # see uspy.interactive.spec._local_to_screen's docstring for the ROI-side
            # calibration; matplotlib was checked directly against Rectangle.get_corners
            # / Ellipse's patch transform, which rotate a data-space point by +angle.
            patch_angle = -rotation
            width = geom["radius"] * 2 if spec.shape == "circle" else geom["width"]
            height = geom["radius"] * 2 if spec.shape == "circle" else geom["height"]
            if spec.shape == "rectangle":
                patch = Rectangle(
                    (x0 - width / 2, y0 - height / 2), width=width, height=height,
                    angle=patch_angle, rotation_point="center",
                    fill=False, edgecolor=color, linewidth=1.5, zorder=2,
                )
            else:
                patch = Ellipse((x0, y0), width=width, height=height, angle=patch_angle,
                                fill=False, edgecolor=color, linewidth=1.5, zorder=2)
            self.ax.add_patch(patch)
            artists.append(patch)
            if selected:
                for cx, cy in corner_positions(spec):
                    artists.append(self._marker(cx, cy, color))
                hx, hy = handle_position(spec, HANDLE_DISTANCE)
                artists.append(self._marker(hx, hy, color, marker="o"))
                handle_line = Line2D([x0, hx], [y0, hy], color=color, linewidth=1, zorder=2)
                self.ax.add_line(handle_line)
                artists.append(handle_line)
        else:  # polygon / profile
            points = vertex_positions(spec)
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            if spec.shape == "polygon":
                patch = Polygon(list(zip(xs, ys)), closed=True, fill=False,
                                edgecolor=color, linewidth=1.5, zorder=2)
                self.ax.add_patch(patch)
                artists.append(patch)
            else:
                if band:
                    # true-to-size preview of the averaging band `Profile.apply`
                    # actually samples, in data coordinates -- reuses the same
                    # geometry `uspy.roi.Profile.plot` draws outside the picker.
                    band_path = spec.to_object().band_path()
                    if band_path is not None:
                        patch = PathPatch(band_path, facecolor=color, edgecolor="none",
                                          alpha=0.5, zorder=1)
                        self.ax.add_patch(patch)
                        artists.append(patch)
                line = Line2D(xs, ys, color=color, linewidth=1.5, zorder=2)
                self.ax.add_line(line)
                artists.append(line)
            if selected:
                artists.extend(self._marker(x, y, color) for x, y in points)
        return artists

    def _build_pending_outline(self) -> list:
        xs = [p[0] for p in self._pending_points]
        ys = [p[1] for p in self._pending_points]
        color = self.color_for_new()
        line = Line2D(xs, ys, color=color, linewidth=1.5, linestyle="--", zorder=3)
        self.ax.add_line(line)
        return [line] + [self._marker(x, y, color) for x, y in self._pending_points]

    def _marker(self, x: float, y: float, color: str, marker: str = "s",
               size: float = HANDLE_SIZE) -> Line2D:
        line = Line2D([x], [y], marker=marker, color=color, markerfacecolor="white",
                      markersize=size, markeredgewidth=1.5, zorder=4)
        self.ax.add_line(line)
        return line

    # -- hit-testing --------------------------------------------------------------

    def _display_distance(self, xy1: tuple, xy2: tuple) -> float:
        p1 = self.ax.transData.transform(xy1)
        p2 = self.ax.transData.transform(xy2)
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _hit_test(self, xy: tuple) -> Optional[tuple]:
        """Return `(index, kind, extra)` for whichever handle/vertex/edge/body is
        under `xy`, or None. Handles and vertices are only checked for the currently
        *selected* region -- keeps hit-testing unambiguous when regions overlap, at
        the cost of needing one click to select a region before its handles appear
        (a click that itself already starts a body-move, so it's a single gesture,
        not an extra step). Bodies of *every* region are always hit-testable, so
        every region stays independently movable regardless of selection.
        """
        if self._selected is not None and self._selected < len(self.specs):
            index = self._selected
            spec = self.specs[index]
            if spec.shape in _SIZABLE:
                if self._display_distance(xy, handle_position(spec, HANDLE_DISTANCE)) <= GRAB_RANGE:
                    return index, "rotate", None
                for corner in corner_positions(spec):
                    if self._display_distance(xy, corner) <= GRAB_RANGE:
                        return index, "resize", None
            elif spec.shape in ("polygon", "profile"):
                for vi, vertex in enumerate(vertex_positions(spec)):
                    if self._display_distance(xy, vertex) <= GRAB_RANGE:
                        return index, "vertex", vi
                if spec.shape == "polygon":
                    edge = self._nearest_edge(spec, xy)
                    if edge is not None:
                        return index, "edge", edge
        for index in reversed(range(len(self.specs))):
            if self._hit_body(self.specs[index], xy):
                return index, "move", None
        return None

    def _hit_body(self, spec: Spec, xy: tuple) -> bool:
        geom = dict(spec.params)
        if spec.shape == "point":
            return self._display_distance(xy, (geom["x0"], geom["y0"])) <= GRAB_RANGE
        if spec.shape == "profile":
            # proximity to the polyline itself, widened to the visible band's
            # half-width so a click anywhere inside the drawn band grabs it -- not
            # just a precise hit on the centerline.
            points = vertex_positions(spec)
            width = dict(spec.params).get("width", 1)
            tol = max(_SEGMENT_TOLERANCE, width / 2)
            return any(self._point_near_segment(xy, a, b, tol) for a, b in zip(points, points[1:]))
        # circle/ellipse/rectangle/polygon all have a rasterizable interior: reuse the
        # same mask the overlay is drawn from, so a click *anywhere inside* the visible
        # highlighted region grabs it -- not just a precise hit on its outline.
        row, col = int(round(xy[1])), int(round(xy[0]))
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        return bool(spec.to_object().pad_to(self.height, self.width)[row, col])

    def _nearest_edge(self, spec: Spec, xy: tuple) -> Optional[int]:
        points = vertex_positions(spec)
        n = len(points)
        for i in range(n):
            if self._point_near_segment(xy, points[i], points[(i + 1) % n], EDGE_TOLERANCE):
                return i
        return None

    @staticmethod
    def _point_near_segment(p: tuple, a: tuple, b: tuple, tol: float = _SEGMENT_TOLERANCE) -> bool:
        px, py = p
        ax_, ay = a
        bx, by = b
        abx, aby = bx - ax_, by - ay
        denom = abx * abx + aby * aby
        t = 0.0 if denom == 0 else max(0.0, min(1.0, ((px - ax_) * abx + (py - ay) * aby) / denom))
        cx, cy = ax_ + t * abx, ay + t * aby
        return math.hypot(px - cx, py - cy) <= tol

    # -- event handlers -------------------------------------------------------

    def _nav_active(self) -> bool:
        """True while the matplotlib toolbar's zoom/pan tool (or any other widget)
        owns the mouse, so a rubber-band zoom/pan drag over the image does not also
        draw or move a region. `toolbar.mode` is the empty string/`_Mode.NONE` (falsy)
        when idle and `"zoom rect"`/`"pan/zoom"` while one of those tools is active;
        `widgetlock` additionally covers `pan` and any other tool that grabs the mouse.
        Under Agg (no ipympl, e.g. this module's own headless tests) `canvas.toolbar`
        is None, so this is always False."""
        canvas = self.ax.figure.canvas
        return bool(canvas.widgetlock.locked()
                    or getattr(getattr(canvas, "toolbar", None), "mode", ""))

    def _on_press(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if self._nav_active():
            return
        xy = (event.xdata, event.ydata)

        if self._pending_points:
            self._pending_points.append(xy)
            if event.dblclick:
                self._commit_pending()
            else:
                self._draw_outlines()
            return

        hit = self._hit_test(xy)
        if hit is not None:
            index, kind, extra = hit
            self._set_selected(index, extra if kind == "vertex" else None)
            if kind == "edge":
                if event.dblclick:
                    self.specs[index] = insert_vertex(self.specs[index], extra, xy)
                    self.redraw_overlays()
                    self._fire_commit()
                else:
                    self._draw_outlines()  # just selecting -- reveal handles
                return
            self._drag = {"kind": kind, "index": index, "vertex": extra,
                          "start": xy, "origin": self.specs[index],
                          "draft": self.specs[index]}
            self._draw_outlines()
            return

        # empty canvas: start (or place) a new region of self.new_shape
        self._set_selected(None)
        if self.new_shape == "point":
            self._append_spec(Spec(
                "point", (("x0", int(round(xy[0]))), ("y0", int(round(xy[1])))),
                self.color_for_new(), self.new_label,
            ))
        elif self.new_shape in ("polygon", "profile"):
            self._pending_points = [xy]
            self._draw_outlines()
        else:
            self._drag = {"kind": "new-bbox", "start": xy, "draft": None}
            self._draw_outlines()

    def _on_move(self, event) -> None:
        if self._drag is None or event.inaxes is not self.ax or event.xdata is None:
            return
        xy = (event.xdata, event.ydata)
        drag = self._drag
        kind = drag["kind"]

        if kind == "new-bbox":
            drag["draft"] = self._draft(drag["start"], xy, _modifiers(event))
        elif kind == "move":
            # from the drag-start spec by the *total* displacement: rounding each
            # event's delta would drop slow (sub-pixel-per-event) motion entirely.
            drag["draft"] = translate(drag["origin"], xy[0] - drag["start"][0],
                                      xy[1] - drag["start"][1])
        elif kind == "resize":
            drag["draft"] = resize_from_handle(drag["draft"], xy,
                                               square="shift" in _modifiers(event))
        elif kind == "rotate":
            drag["draft"] = rotation_from_handle(drag["draft"], xy)
        elif kind == "vertex":
            drag["draft"] = move_vertex(drag["draft"], drag["vertex"], xy)

        # Redraw only at a bounded rate (see _REDRAW_MIN_INTERVAL) -- the draft above
        # is always recomputed on every event regardless, so the final, on-release
        # position is exact no matter how many intermediate redraws this skips.
        now = self._clock()
        if now - self._last_redraw >= _REDRAW_MIN_INTERVAL:
            self._last_redraw = now
            self._draw_outlines(draft_only=True)

        if kind != "new-bbox" and self.on_drag is not None:
            self.on_drag(drag["index"], drag["draft"])

    def _on_release(self, event) -> None:
        if self._drag is None:
            return
        drag, self._drag = self._drag, None
        draft = drag["draft"]
        if draft is None:
            return  # a "new-bbox" drag that never moved (a plain click, not a drag)

        if drag["kind"] == "new-bbox":
            if not draft.to_object().pad_to(self.height, self.width).any():
                self.status = "region is entirely outside the frame -- discarded"
                self._draw_outlines()
                return
            self._append_spec(draft)
        else:
            self.specs[drag["index"]] = draft
            self.redraw_overlays()
            self._fire_commit()

    def _on_key(self, event) -> None:
        if event.key == "escape":
            editing = self._drag is not None and self._drag["kind"] != "new-bbox"
            self._pending_points = []
            self._drag = None
            self._draw_outlines()
            if editing:  # on_drag may have shown the draft: resync to the restored spec
                self._fire_commit()
            return
        if self._drag is not None:
            return  # edit keys mid-drag would act on specs the release then overwrites
        if event.key in _NUDGE:
            if self._selected is not None:
                self.nudge_selected(*_NUDGE[event.key])
            return
        if event.key not in _DELETE_KEYS or self._selected is None:
            return
        if self._selected_vertex is not None:
            try:
                self.specs[self._selected] = remove_vertex(self.specs[self._selected],
                                                            self._selected_vertex)
            except ValueError as exc:
                self.status = str(exc)
                return
            self._selected_vertex = None
            self.redraw_overlays()
            self._fire_commit()
        else:
            self.delete_selected()

    # -- shared helpers -------------------------------------------------------

    def _draft(self, start: tuple, end: tuple, mods: set) -> Spec:
        """The in-progress Spec for a "new-bbox" creation drag from `start` to `end`,
        Fiji-style: centred on the press point by default (`ROI`'s own centre+extents
        convention), Ctrl switches to the old corner-to-corner bounding-box gesture,
        Shift forces width == height (always on for "circle", which is inherently
        square). Read fresh from `mods` on every motion event (not latched at
        drag-start), so holding/releasing either key mid-drag changes the draft live.

        color_for_new() is stable across a whole drag: the region being drawn isn't in
        self.specs yet (only _append_spec adds it, on release), so every motion
        event's auto_color() call sees the same "used" set and returns the same color
        -- it doesn't need to be pinned once at drag-start.
        """
        return spec_from_drag(
            self.new_shape, start, end, self.color_for_new(), self.new_label,
            centered="ctrl" not in mods, square="shift" in mods,
        )

    def _commit_pending(self) -> None:
        points, self._pending_points = self._pending_points, []
        try:
            spec = spec_from_points(self.new_shape, points, self.color_for_new(),
                                    self.new_label, width=self.new_width)
        except ValueError as exc:
            self.status = str(exc)
            self._draw_outlines()
            return
        self._append_spec(spec)

    def _append_spec(self, spec: Spec) -> None:
        self.specs.append(spec)
        self._set_selected(len(self.specs) - 1)
        self.redraw_overlays()
        self._fire_commit()

    def _fire_commit(self) -> None:
        if self.on_commit is not None:
            self.on_commit()
