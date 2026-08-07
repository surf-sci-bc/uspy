"""Pure geometry for the interactive ROI/Profile picker: `Spec`, conversions to and
from `uspy.roi` objects, code emission, and the screen-coordinate <-> shape-local-frame
math the draggable canvas layer (canvas.py) needs for move/resize/rotate/vertex-edit.

No pyplot, no ipywidgets -- only `uspy.roi` and the standard library, so this is
importable and unit-testable in any environment that has uspy itself, no optional
extra required.

Coordinate convention throughout: images are drawn with `imshow` (origin upper), so
data-x is the column and data-y is the row, increasing downward -- the same convention
`uspy.roi.ROI` itself uses. There is no y-flip anywhere in this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np

from uspy.roi import ROI, Profile

#: Shapes that produce a `uspy.roi.ROI` -- i.e. everything with a rasterizable mask.
#: "square" is deliberately absent: a square is just a rectangle with width == height,
#: so the picker only ever emits "rectangle" for it (see `Spec.from_roi`); ROI itself
#: still accepts "square" as a `source` string.
ROI_SHAPES = ("point", "circle", "ellipse", "rectangle", "polygon")

#: Every shape this geometry layer can express. "profile" makes a `uspy.roi.Profile`,
#: which has no mask and no I(V) curve -- `uspy.interactive.canvas.Canvas`'s `shapes=`
#: allow-list is what keeps it out of the ROI-only I(V) picker while still letting a
#: future waterfall-style picker (or a test) construct one.
SHAPES = ROI_SHAPES + ("profile",)

#: Shapes with a centre, a size, and a rotation -- i.e. the ones the resize/rotate
#: handles in canvas.py operate on.
_SIZABLE = ("circle", "ellipse", "rectangle")


@dataclass(frozen=True)
class Spec:
    """One picked region: shape name, ordered integer geometry, and an *explicit*
    color. Frozen and hashable, so it can key a cache (see picker.py's I(V) cache) and
    two identical picks compare equal.

    Color is mandatory and never left implicit: `StyledObject` (uspy/roi.py) hands out
    default colors from a class-level counter that depends on global construction
    order, so an implicit color would not reproduce what was actually picked.
    """

    shape: str
    params: tuple[tuple[str, Any], ...]
    color: Union[str, tuple]  # matplotlib accepts far more than HTML strings, e.g. an
                              # (r, g, b[, a]) float tuple -- see canvas.py's auto_color()
                              # and picker.py's _sync_controls() for where that has to be
                              # normalized down to an HTML color for ipywidgets.ColorPicker
    label: Optional[str] = None

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise ValueError(f"Unknown shape {self.shape!r}, expected one of {SHAPES}")

    # -- conversion to/from uspy.roi objects -------------------------------------

    def to_dict(self) -> dict:
        """A `ROI.from_spec`-compatible dict. Not meaningful for shape == "profile"."""
        return {"shape": self.shape, **dict(self.params)}

    def to_object(self) -> Union[ROI, Profile]:
        """Build a *fresh* ROI/Profile every call -- never cache-and-mutate one:
        `ROI.pad_to` keys its `_mask_buffer` on the requested extent alone, so a
        mutated-then-reused ROI can silently return a stale mask.
        """
        if self.shape == "profile":
            geom = dict(self.params)
            points = [tuple(p) for p in geom["points"]]
            return Profile(points, width=geom["width"], color=self.color, label=self.label)
        return ROI.from_spec(self.to_dict(), color=self.color, label=self.label)

    def to_code(self) -> str:
        """The paste-able expression for this region. Reuses `ROI`/`Profile`'s own
        `__repr__`, so the emitted literal and what a notebook prints can never drift
        apart into two separate implementations of "what does this ROI look like"."""
        return repr(self.to_object())

    @classmethod
    def from_roi(cls, region: Union[ROI, Profile], label: Optional[str] = None) -> Spec:
        """Seed a Spec from an ROI/Profile already defined in the notebook (used by
        `pick_rois(stack, rois=[...])` to re-open existing regions for editing).

        Raises
        ------
        ValueError
            For an array-sourced ROI (`ROI.from_array`, or the result of `__add__`),
            which has no parametric shape to seed a draggable region from.
        """
        if isinstance(region, Profile):
            points = tuple((int(p[0]), int(p[1])) for p in region.points)
            return cls("profile", (("points", points), ("width", int(region.width))),
                       region.color, region.label if label is None else label)

        if region.shape_name == "array":
            raise ValueError(
                "cannot seed the picker from an array-sourced ROI (from_array, or the "
                "result of __add__) -- it has no parametric shape to edit"
            )
        x0, y0 = int(region.position[0]), int(region.position[1])
        roi_label = region.label if label is None else label
        if region.shape_name == "polygon":
            corners = np.asarray(region.params["corners"]) + region.position
            params = (("corners", tuple((int(x), int(y)) for x, y in corners)),)
            return cls("polygon", params, region.color, roi_label)
        if region.shape_name == "point":
            return cls("point", (("x0", x0), ("y0", y0)), region.color, roi_label)

        shape = "rectangle" if region.shape_name == "square" else region.shape_name
        params = (("x0", x0), ("y0", y0)) + tuple(
            (k, v if isinstance(v, str) else int(v)) for k, v in region.params.items()
        )
        return cls(shape, params, region.color, roi_label)


def specs_to_code(specs: Sequence[Spec], imports: bool = True) -> str:
    """Render a list of Specs as one self-contained, paste-able cell: a `rois = [...]`
    list and/or a `profiles = [...]` list (whichever groups are non-empty), each entry
    via `Spec.to_code()` (i.e. `ROI`/`Profile`'s own `__repr__`), so the block can never
    drift from what printing the objects themselves would show.
    """
    rois = [s for s in specs if s.shape != "profile"]
    profiles = [s for s in specs if s.shape == "profile"]
    if not rois and not profiles:
        return "# nothing picked yet"

    lines = []
    if imports:
        names = [n for n, group in (("ROI", rois), ("Profile", profiles)) if group]
        lines.append(f"from uspy.roi import {', '.join(names)}")
        lines.append("")
    if rois:
        lines.append("rois = [")
        lines.extend(f"    {s.to_code()}," for s in rois)
        lines.append("]")
        if profiles:
            lines.append("")
    if profiles:
        lines.append("profiles = [")
        lines.extend(f"    {s.to_code()}," for s in profiles)
        lines.append("]")
    return "\n".join(lines)


# -- creation from picked screen geometry ----------------------------------------


def spec_from_bbox(
    shape: str, x0: float, y0: float, width: float, height: float, rotation: float,
    color: str, label: Optional[str] = None,
) -> Spec:
    """Map a bounding box (centre `x0, y0`, full `width`/`height`, `rotation` in
    degrees) to a Spec. `ROI`'s `(x0, y0)` is the *centre* for circle/ellipse/
    rectangle (`_ref_point = [width/2, height/2]`, uspy/roi.py), so this is a direct
    round-to-int pass-through -- no top-left/centre conversion needed once the caller
    (canvas.py) has already computed the centre from a selector or dragged patch.
    """
    x0, y0 = int(round(x0)), int(round(y0))
    width, height = int(round(width)), int(round(height))
    rotation = int(round(rotation)) % 360
    if shape == "ellipse" and width == height and width % 2 == 0:
        # a circle is rotationally symmetric, so canonicalize regardless of the
        # dragged rotation -- and only when the diameter is even, since a circle's
        # radius must be an integer to rasterize byte-identically to the ellipse form.
        return Spec("circle", (("x0", x0), ("y0", y0), ("radius", width // 2)), color, label)
    params = [("x0", x0), ("y0", y0), ("width", width), ("height", height)]
    if rotation:
        params.append(("rotation", rotation))
    return Spec(shape, tuple(params), color, label)


def spec_from_drag(
    shape: str, start: tuple[float, float], end: tuple[float, float], color: str,
    label: Optional[str] = None, *, centered: bool = True, square: bool = False,
) -> Spec:
    """Map a mouse drag from `start` to `end` to a Spec, Fiji-style.

    centered : `start` (the press point) is the region's centre and the drag sets the
        half-extents -- the default, matching `ROI`'s own centre+extents convention
        (canvas.py's Ctrl modifier switches this off for the old corner-to-corner
        bounding-box gesture instead).
    square : force width == height, using the larger dragged extent (canvas.py's Shift
        modifier). Always on for "circle", which is inherently square -- the radius is
        then simply the distance from `start` to `end`.

    Delegates the actual rounding/canonicalization to `spec_from_bbox`, which already
    owns that logic (including ellipse -> circle canonicalization).
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    is_circle = shape == "circle"
    square = square or is_circle
    if centered:
        x0, y0 = start
        if is_circle:
            # the natural "radius" for a centred drag is the distance dragged, not
            # the larger axis-aligned extent (that's the corner-to-corner measure
            # below) -- this is what makes the circle grow out from the click point.
            width = height = 2 * math.hypot(dx, dy)
        elif square:
            width = height = 2 * max(abs(dx), abs(dy))
        else:
            width, height = 2 * abs(dx), 2 * abs(dy)
    else:
        x0, y0 = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        if square:
            width = height = max(abs(dx), abs(dy))
        else:
            width, height = abs(dx), abs(dy)

    if shape == "circle":
        # force an even diameter so spec_from_bbox's ellipse->circle canonicalization
        # (which requires width == height and even) actually fires -- see there.
        diameter = max(2, int(round(width / 2)) * 2)
        return spec_from_bbox("ellipse", x0, y0, diameter, diameter, 0, color, label)
    return spec_from_bbox(shape, x0, y0, max(width, 2), max(height, 2), 0, color, label)


def spec_from_points(
    shape: str, points: Sequence[tuple[float, float]], color: str,
    label: Optional[str] = None, width: int = 1,
) -> Spec:
    """Map polygon/profile vertices (absolute image coordinates) to a Spec.
    `ROI.polygon` derives its own top-left `position` from the corners, so they pass
    straight through with no x0/y0 -- adding one would make `ROI.from_spec`'s polygon
    branch raise `TypeError` (it splats leftover spec keys onto
    `ROI.polygon(corners, **spec)`, whose first positional argument is already
    `corners`).
    """
    pts = _dedupe(tuple((int(round(x)), int(round(y))) for x, y in points))
    if shape == "profile":
        if len(pts) < 2:
            raise ValueError("a profile needs at least 2 points")
        return Spec("profile", (("points", pts), ("width", int(width))), color, label)
    if len(pts) < 3:
        raise ValueError("a polygon needs at least 3 corners")
    return Spec("polygon", (("corners", pts),), color, label)


def _dedupe(points: tuple) -> tuple:
    out = [points[0]]
    for point in points[1:]:
        if point != out[-1]:
            out.append(point)
    return tuple(out)


# -- editing an existing Spec ------------------------------------------------------


def translate(spec: Spec, dx: float, dy: float) -> Spec:
    """Shift the whole region by `(dx, dy)`: the centre for circle/ellipse/rectangle/
    point, or every vertex for polygon/profile."""
    dx, dy = int(round(dx)), int(round(dy))
    geom = dict(spec.params)
    if "x0" in geom:
        geom["x0"] += dx
        geom["y0"] += dy
    else:
        key = _vertex_key(spec)
        geom[key] = tuple((x + dx, y + dy) for x, y in geom[key])
    return _with_geom(spec, geom)


def resize_from_handle(spec: Spec, xy: tuple[float, float], *, square: bool = False) -> Spec:
    """Resize a circle/ellipse/rectangle spec, centre fixed, via a corner handle
    dragged to `xy` (screen/data coordinates). The drag point is mapped into the
    shape's own unrotated local frame (`_screen_to_local`) first, so resizing behaves
    the same regardless of the shape's current rotation.

    square : force width == height, using the larger local half-extent (canvas.py's
        Shift modifier, mirroring `spec_from_drag`'s own `square` -- same meaning
        during a resize as during the initial draw)."""
    _require_sizable(spec, "resize")
    x0, y0 = _center(spec)
    rotation = _rotation(spec)
    lx, ly = _screen_to_local(x0, y0, xy[0], xy[1], rotation)
    base_shape = "ellipse" if spec.shape in ("circle", "ellipse") else "rectangle"
    if square:
        size = max(abs(lx), abs(ly)) * 2
        if base_shape == "ellipse":
            # force an even diameter so spec_from_bbox's ellipse->circle
            # canonicalization (width == height and even) actually fires -- see there.
            size = max(2, int(round(size / 2)) * 2)
        else:
            size = max(2, int(round(size)))
        width = height = size
    else:
        width = max(2, int(round(abs(lx) * 2)))
        height = max(2, int(round(abs(ly) * 2)))
    return spec_from_bbox(base_shape, x0, y0, width, height, rotation, spec.color, spec.label)


def rotation_from_handle(spec: Spec, xy: tuple[float, float]) -> Spec:
    """Set rotation from a rotate handle dragged to `xy`, centre fixed. See
    `_screen_to_local`'s docstring for the sign convention (empirically calibrated
    against `ROI`'s actual rotated masks, not assumed)."""
    _require_sizable(spec, "rotate")
    x0, y0 = _center(spec)
    screen_angle = math.degrees(math.atan2(xy[1] - y0, xy[0] - x0))
    return with_rotation(spec, -screen_angle)


def with_rotation(spec: Spec, degrees: float) -> Spec:
    """Set rotation to an exact value (e.g. from a typed-in degrees box), centre and
    size unchanged. The counterpart to `rotation_from_handle`, which derives the angle
    from a dragged handle position instead of taking one directly."""
    _require_sizable(spec, "rotate")
    x0, y0 = _center(spec)
    geom = dict(spec.params)
    if spec.shape == "circle":
        width = height = geom["radius"] * 2
    else:
        width, height = geom["width"], geom["height"]
    base_shape = "ellipse" if spec.shape == "circle" else spec.shape
    return spec_from_bbox(base_shape, x0, y0, width, height, degrees, spec.color, spec.label)


def handle_position(spec: Spec, distance: float) -> tuple[float, float]:
    """Where to draw the rotate handle: `distance` px out from centre along the
    shape's local +x axis. The inverse of `rotation_from_handle`."""
    _require_sizable(spec, "place a rotate handle on")
    x0, y0 = _center(spec)
    return _local_to_screen(x0, y0, distance, 0, _rotation(spec))


def corner_positions(spec: Spec) -> list[tuple[float, float]]:
    """The four resize-handle screen positions for a circle/ellipse/rectangle spec."""
    _require_sizable(spec, "place resize handles on")
    x0, y0 = _center(spec)
    geom = dict(spec.params)
    if spec.shape == "circle":
        width = height = geom["radius"] * 2
    else:
        width, height = geom["width"], geom["height"]
    rotation = _rotation(spec)
    local_corners = [(width / 2, height / 2), (-width / 2, height / 2),
                     (-width / 2, -height / 2), (width / 2, -height / 2)]
    return [_local_to_screen(x0, y0, lx, ly, rotation) for lx, ly in local_corners]


def vertex_positions(spec: Spec) -> tuple[tuple[int, int], ...]:
    """The polygon corners / profile points, in absolute image coordinates."""
    return dict(spec.params)[_vertex_key(spec)]


def move_vertex(spec: Spec, index: int, xy: tuple[float, float]) -> Spec:
    """Move a single polygon corner / profile point to `xy` (Fiji-style: the other
    vertices are untouched)."""
    key = _vertex_key(spec)
    geom = dict(spec.params)
    points = list(geom[key])
    points[index] = (int(round(xy[0])), int(round(xy[1])))
    geom[key] = tuple(points)
    return _with_geom(spec, geom)


def insert_vertex(spec: Spec, edge_index: int, xy: tuple[float, float]) -> Spec:
    """Insert a new vertex at `xy`, on the edge after `edge_index`."""
    key = _vertex_key(spec)
    geom = dict(spec.params)
    points = list(geom[key])
    points.insert(edge_index + 1, (int(round(xy[0])), int(round(xy[1]))))
    geom[key] = tuple(points)
    return _with_geom(spec, geom)


def remove_vertex(spec: Spec, index: int) -> Spec:
    """Remove a vertex. Refuses to go below 3 polygon corners / 2 profile points --
    below that there's no well-defined shape left to edit."""
    key = _vertex_key(spec)
    geom = dict(spec.params)
    points = list(geom[key])
    minimum = 3 if spec.shape == "polygon" else 2
    if len(points) <= minimum:
        raise ValueError(f"a {spec.shape} needs at least {minimum} points")
    del points[index]
    geom[key] = tuple(points)
    return _with_geom(spec, geom)


# -- internal helpers ----------------------------------------------------------


def _vertex_key(spec: Spec) -> str:
    if spec.shape == "polygon":
        return "corners"
    if spec.shape == "profile":
        return "points"
    raise ValueError(f"{spec.shape!r} has no vertices")


def _require_sizable(spec: Spec, action: str) -> None:
    if spec.shape not in _SIZABLE:
        raise ValueError(f"cannot {action} a {spec.shape!r} region")


def _center(spec: Spec) -> tuple[int, int]:
    geom = dict(spec.params)
    return geom["x0"], geom["y0"]


def _rotation(spec: Spec) -> int:
    return dict(spec.params).get("rotation", 0)


def _with_geom(spec: Spec, geom: dict) -> Spec:
    """Rebuild a Spec from a geom dict that changed only *values*, not the key set --
    key order is preserved from `spec.params` (creation/rotation/resize helpers that
    can add or drop keys, e.g. an omitted `rotation`, go through `spec_from_bbox`
    instead, which owns that logic)."""
    params = tuple((k, geom[k]) for k, _ in spec.params)
    return Spec(spec.shape, params, spec.color, spec.label)


def _local_to_screen(x0: float, y0: float, lx: float, ly: float, rotation_deg: float
                     ) -> tuple[float, float]:
    """A point at local-frame offset `(lx, ly)` (i.e. before any rotation) from a
    shape's centre `(x0, y0)`, mapped to screen/data coordinates at the shape's
    current `rotation_deg`. Inverse of `_screen_to_local`.

    The sign convention is empirically calibrated, not assumed: `ROI`'s rectangle/
    square path rotates via `scipy.ndimage.rotate(mask, rotation_deg, reshape=False)`,
    and a directed calibration (an offset marker blob rotated by known angles) showed
    that a point at local angle 0 lands at screen angle `-rotation_deg` -- i.e.
    increasing `rotation` turns the shape *clockwise* in image/data coordinates (data-y
    increasing downward, same as everywhere else in this codebase). `draw_ellipse`'s
    own `rotation=` parameter was cross-checked against the same calibration and
    agrees mod 180 (an ellipse's axis has no directed sense to check further).
    """
    distance = math.hypot(lx, ly)
    if distance == 0:
        return x0, y0
    local_angle = math.degrees(math.atan2(ly, lx))
    screen_angle = math.radians(local_angle - rotation_deg)
    return x0 + distance * math.cos(screen_angle), y0 + distance * math.sin(screen_angle)


def _screen_to_local(x0: float, y0: float, x: float, y: float, rotation_deg: float
                     ) -> tuple[float, float]:
    """Inverse of `_local_to_screen`: a screen/data point `(x, y)` relative to a
    shape's centre `(x0, y0)` and current `rotation_deg`, mapped into the shape's own
    unrotated local frame."""
    dx, dy = x - x0, y - y0
    distance = math.hypot(dx, dy)
    if distance == 0:
        return 0.0, 0.0
    screen_angle = math.degrees(math.atan2(dy, dx))
    local_angle = math.radians(screen_angle + rotation_deg)
    return distance * math.cos(local_angle), distance * math.sin(local_angle)
