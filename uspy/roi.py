"""
Classes for defining shapes and regions on images.
"""
# pylint: disable=abstract-method
from __future__ import annotations
from typing import Any, Sequence, Union, Optional
from collections.abc import Iterable
from functools import partial

import numpy as np
import matplotlib as mpl
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from scipy import ndimage
from skimage.draw import ellipse as draw_ellipse, polygon as draw_polygon, rectangle as draw_rectangle
from skimage.measure import profile_line

import uspy.dataobject as do
from uspy.dataobject.deferred import is_deferred


def _apply_mask(full_mask: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """`np.ma.masked_array(arr, mask=~full_mask)`, as a module-level function bound to
    `full_mask` via `functools.partial` (picklable, unlike a closure/lambda) so it can be
    used with `DeferredArray.map` -- see `ROI.apply`."""
    return np.ma.masked_array(arr, mask=~full_mask)


def palette_color(index: int) -> str:
    """The `index`-th color of matplotlib's `axes.prop_cycle`, wrapping around.

    The single source of truth for "what color comes next" -- used both by
    `StyledObject`'s implicit-color counter below and by `uspy.interactive.canvas.Canvas`'s
    auto-cycling (which needs to compute a color for a given position without mutating any
    shared counter).
    """
    colors = mpl.rc_params()["axes.prop_cycle"].by_key()["color"]
    return colors[index % len(colors)]


class StyledObject:  # do.Loadable):
    """Contains a style dictionary that can have class-wise defaults. The dictionary
    is intended for matplotlib keyword arguments."""

    _idx = 0
    _default_style = {}

    def __init__(
        self, style: dict[str, Any] = None, color: Optional[str] = None
    ) -> None:
        style_ = self._default_style.copy()
        if style is not None:
            style_.update(style)
        if color is not None:
            style_["color"] = color
        if "color" not in style_:
            n_colors = len(mpl.rc_params()["axes.prop_cycle"])
            style_["color"] = palette_color(StyledObject._idx)
            StyledObject._idx = (StyledObject._idx + 1) % n_colors
        self.style = style_

    @property
    def color(self) -> str:
        """Convenience access to the color."""
        return self.style["color"]

    @property
    def cmap(self) -> mpl.colors.Colormap:
        """A linear colormap that matches self.color for values above 1 and
        is transparent for 0."""
        color = mpl.colors.to_rgba(self.color)
        # cmap_mat = np.array([np.linspace(c, 0, 1) for c in color]).T
        cmap_mat = np.array([[0, 0, 0, 0], color])
        cmap = mpl.colors.ListedColormap(cmap_mat)
        # cmap.set_bad(alpha=0)
        return cmap

    def plot(self, ax: mpl.axes.Axes) -> None:
        """Plot this object onto matplotlib axes."""
        raise NotImplementedError


class ROI(StyledObject):
    """An image region represented as a boolean 2D array."""

    # take care: OpenCV-points are given as (x, y),
    # but numpy 2d image arrays are in (y, x)-coordinates
    shapes = ("circle", "ellipse", "square", "rectangle", "point", "polygon")

    def __init__(
        self,
        x0: int,
        y0: int,
        source: Union[np.ndarray, str, Iterable],
        style: dict[str, Any] = None,
        color: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.position = np.array([x0, y0])
        self._ref_point = np.array([0, 0])
        self.params = kwargs
        self.label = label
        # the shape name, kept for __repr__: a plain "array" for from_array/__add__
        # results, which have no parametric form to reconstruct.
        self.shape_name = source if isinstance(source, str) else "array"

        if isinstance(source, str):
            self._apply_params(shape=source, **self.params)
        elif isinstance(source, Iterable):
            # pass  # polygon
            self.array = np.array(source, dtype=bool)
        assert self.array.ndim == 2

        super().__init__(style=style, color=color)
        self._mask_buffer = np.zeros((0, 0), dtype=bool)
        self._center_buffer = True

    def _apply_params(self, shape: Optional[str] = "circle", **kwargs):
        """Convert a geometric rectangular or elliptic description to a boolean mask.
        Arguments belonging to the shapes:
            - circle: radius
            - ellipse: width, height, rotation
            - square: width, rotation
            - rectangle: width, height, rotation
            - point: (none)
            - polygon: corners
        where radius, width and height are in pixels; rotation is in degrees (anti-clockwise).
        Coordinates follow the array/position convention used throughout this class: width/x
        maps to columns, height/y maps to rows.
        """
        if shape == "point":
            self._ref_point = np.array([0, 0])
            self.array = np.array([[True]])
            return self.array
        elif shape != "polygon":
            if "radius" in kwargs:
                if "width" in kwargs:
                    raise ValueError("Both radius and width were given for a shape-ROI")
                width = kwargs.pop("radius") * 2
            else:
                width = kwargs.pop("width")
            height = kwargs.pop("height", width)
            # square canvas so shapes don't get clipped when rotated below
            max_extent = int(np.sqrt(width**2 + height**2))
            mask = np.zeros((max_extent, max_extent), dtype=bool)
        elif shape == "polygon":
            corners = kwargs.pop("corners")
            width = np.max(corners[:, 0]) - np.min(corners[:, 0])
            height = np.max(corners[:, 1]) - np.min(corners[:, 1])
            # polygons never rotate, so the canvas can match their actual
            # (height, width) extent exactly instead of being padded square --
            # padding it square let draw_polygon's boundary-inclusive
            # rasterization leak an extra row/col on the smaller dimension
            # whenever width != height (that extra edge only got clipped away
            # by the array bound when width == height happened to match the
            # square canvas side).
            mask = np.zeros((int(np.ceil(height)), int(np.ceil(width))), dtype=bool)

        rotation = kwargs.pop("rotation", 0)
        if shape in ("circle", "ellipse", "square", "rectangle"):
            # Unrotated shapes are drawn flush against the canvas origin (r0=c0=0), as
            # before -- that placement is already correct, so rotation=0 stays byte-for-
            # -byte identical to the pre-fix behavior. A rotated shape instead needs to
            # sit at the canvas's own center: skimage's draw_ellipse rotates about the
            # center point it's given, so an off-center ellipse simply runs out of margin
            # on whichever side is closer to the edge and gets clipped there; scipy's
            # ndimage.rotate(reshape=False) always rotates about the *array's* geometric
            # center regardless of where the shape sits in it, so an off-center rectangle
            # doesn't just clip -- it orbits around the array center as it rotates.
            if rotation:
                r0, c0 = (max_extent - height) // 2, (max_extent - width) // 2
            else:
                r0 = c0 = 0
            self._ref_point = np.array([c0 + width / 2, r0 + height / 2])
        if shape in ("circle", "ellipse"):
            rows, cols = draw_ellipse(
                r0 + height / 2,
                c0 + width / 2,
                height / 2,
                width / 2,
                shape=mask.shape,
                rotation=np.deg2rad(rotation),
            )
            mask[rows, cols] = True
        elif shape in ("square", "rectangle"):
            rows, cols = draw_rectangle((r0, c0), extent=(height, width), shape=mask.shape)
            mask[rows, cols] = True
            if rotation:
                mask = ndimage.rotate(mask.astype(float), rotation, reshape=False) > 0.5
        elif shape == "polygon":
            rows, cols = draw_polygon(corners[:, 1], corners[:, 0], shape=mask.shape)
            mask[rows, cols] = True
        else:
            raise ValueError(f"Unknown shape {shape}")

        self.array = mask
        return self.array

    @property
    def array(self) -> np.ndarray:
        n = self._array_shape[0] * self._array_shape[1]
        return np.unpackbits(self._packed_array)[:n].reshape(self._array_shape).astype(bool)

    @array.setter
    def array(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=bool)
        self._array_shape = arr.shape
        self._packed_array = np.packbits(arr.ravel())

    @classmethod
    def point(cls, x0: int, y0: int, **kwargs) -> ROI:
        """Construct a point ROI."""
        return cls(x0, y0, source="point", **kwargs)

    @classmethod
    def circle(cls, x0: int, y0: int, **kwargs) -> ROI:
        """Construct a circular ROI."""
        return cls(x0, y0, source="circle", **kwargs)

    @classmethod
    def ellipse(cls, x0: int, y0: int, **kwargs) -> ROI:
        """Construct an elliptic ROI."""
        return cls(x0, y0, source="ellipse", **kwargs)

    @classmethod
    def polygon(cls, corners, **kwargs) -> ROI:
        """Construct a polygon ROI."""
        corners = np.array(corners)
        x0 = np.min(corners[:, 0])
        y0 = np.min(corners[:, 1])
        corners[:, 0] -= x0
        corners[:, 1] -= y0
        return cls(x0, y0, source="polygon", corners=corners, **kwargs)

    @classmethod
    def from_array(cls, array: Iterable, x0: int = 0, y0: int = 0, **kwargs) -> ROI:
        """Construct a ROI directly from a boolean mask array."""
        return cls(x0, y0, source=array, **kwargs)

    @classmethod
    def from_spec(cls, spec: dict[str, Any], shape: Optional[tuple[int, int]] = None, **kwargs) -> ROI:
        """Construct a ROI from a plain dict spec, e.g. a `uspy.toml` `[uspy.mask]` table
        (see `uspy.leem.readers`). Every geometry value may be an absolute pixel number, or
        a percent string (`"45%"`) scaled against `shape=(height, width)`: `x0`/`width`/
        `radius`/`corners`' x-components scale with width, `y0`/`height`/`corners`'
        y-components scale with height; `rotation` is never scaled. `shape` is only needed
        when `spec` actually contains a percent string.

        Percent scaling floors to the nearest pixel (`int()`) -- e.g. `"45%"` of a
        1024px-wide frame is 460px, matching the historical hardcoded channelplate circle
        (`radius = width // 2 * 9 // 10`).

        Raises
        ------
        ValueError
            If a percent string is given without `shape`, is not parseable, or `spec`'s
            `shape` names an unknown ROI shape (see `ROI.shapes`).
        """
        spec = dict(spec, **kwargs)  # copy: never mutate the caller's dict
        shape_name = spec.pop("shape", "circle")
        height, width = shape if shape is not None else (None, None)

        def resolve(value, extent, key):
            if not isinstance(value, str):
                return value
            if not value.endswith("%"):
                raise ValueError(
                    f"Invalid value {value!r} for {key!r}: expected a number or a "
                    "percent string like '45%'"
                )
            try:
                pct = float(value[:-1])
            except ValueError as exc:
                raise ValueError(f"Invalid percent value {value!r} for {key!r}") from exc
            if extent is None:
                raise ValueError(
                    f"Percent value {value!r} for {key!r} requires "
                    "from_spec(..., shape=(height, width))"
                )
            return int(pct / 100 * extent)

        if shape_name == "polygon":
            corners = [
                [resolve(x, width, "corners"), resolve(y, height, "corners")]
                for x, y in spec.pop("corners")
            ]
            return cls.polygon(corners, **spec)

        x0 = resolve(spec.pop("x0"), width, "x0")
        y0 = resolve(spec.pop("y0"), height, "y0")
        for key in ("width", "radius"):
            if key in spec:
                spec[key] = resolve(spec[key], width, key)
        if "height" in spec:
            spec["height"] = resolve(spec["height"], height, "height")
        return cls(x0, y0, source=shape_name, **spec)

    def apply(
        self, obj: do.Image, return_array: bool = False
    ) -> Union[do.Image, np.ndarray]:
        """Apply the MaskLike to a Dataobject and either return the masked
        DataObject or the raw masked data."""
        full_mask = self.pad_to(*obj.shape).astype(bool)
        if return_array:
            return np.ma.masked_array(obj.image, mask=~full_mask)
        result = obj.copy()
        current = result._data["image"]  # pylint: disable=protected-access
        fn = partial(_apply_mask, full_mask)
        if is_deferred(current):
            # Compose into a new DeferredArray instead of loading -- keeps a virtual
            # element's mask application out-of-core (see `_apply_mask`'s docstring, and
            # `LEEMStack.align()`'s per-element mask, which routes through here).
            image = current.map(fn, shape=current._shape, dtype=current._dtype)  # pylint: disable=protected-access
        else:
            image = fn(current)
        result._data["image"] = image  # pylint: disable=protected-access
        return result

    def pad_to(self, width: int, height: int) -> np.ndarray:
        """Return the the mask padded to a given extent."""
        if self._mask_buffer.shape == (width, height):
            return self._mask_buffer
        # find sizes and corners. The low corner is the difference between the mask's
        # reference point and the position:
        mask_size = self.array.shape
        pad_size = np.array((width, height)).astype(int)
        low_corner = self.origin[::-1].astype(int)
        high_corner = pad_size - mask_size - low_corner

        result = np.zeros(pad_size, dtype=bool)
        # crop the mask, if need be (i.e., if corners have negative components):
        crop_low = np.clip(-low_corner, 0, None)
        crop_high = mask_size - np.clip(-high_corner, 0, None)
        if any(crop_low > mask_size) or any(crop_high < 0):
            return result
        mask = self.array[crop_low[0] : crop_high[0], crop_low[1] : crop_high[1]]

        # pad the mask until it matches the pad_size:
        pad_low = np.clip(low_corner, 0, None)
        pad_high = pad_size - np.clip(high_corner, 0, None)
        result[pad_low[0] : pad_high[0], pad_low[1] : pad_high[1]] = mask
        # put results in the buffer variables
        self._mask_buffer = result

        return result

    @property
    def area(self) -> int:
        """Total area in pixels. Is only valid when the mask is not cut off at the sides!"""
        return self.array.astype(bool).sum()

    @property
    def ref_point(self) -> np.ndarray:
        """Reference point for the mask."""
        return self._ref_point

    @property
    def origin(self) -> np.ndarray:
        """Lower left origin of the mask considering its position and ref_point."""
        return self.position - self.ref_point

    @property
    def center_of_mass(self) -> np.ndarray:
        """The center of mass position of the mask array."""
        y_i, x_i = np.nonzero(self.array)
        return np.array([x_i.mean(), y_i.mean()])

    def plot(self, ax: mpl.axes.Axes):
        """Plot the styled object onto a matplotlib axes."""
        img_height = int(abs(ax.get_xlim()[1] - ax.get_xlim()[0]))
        img_width = int(abs(ax.get_ylim()[1] - ax.get_ylim()[0]))
        array = self.pad_to(img_width, img_height)
        ax.imshow(array, cmap=self.cmap, alpha=self.style.get("alpha", 0.5))

    def __add__(self, other: ROI) -> ROI:
        """Return the union of two ROIs as a new ROI, placed correctly according
        to each ROI's position (works for differently sized/positioned ROIs)."""
        origin1, origin2 = self.origin, other.origin
        # extents in (x, y) order to match position/origin:
        extent1 = np.array(self.array.shape)[::-1]
        extent2 = np.array(other.array.shape)[::-1]

        low = np.minimum(origin1, origin2)
        high = np.maximum(origin1 + extent1, origin2 + extent2)
        size_xy = np.ceil(high - low).astype(int)

        canvas = np.zeros(size_xy[::-1], dtype=bool)
        for roi, origin in ((self, origin1), (other, origin2)):
            off_x, off_y = np.round(origin - low).astype(int)
            arr = roi.array
            canvas[off_y : off_y + arr.shape[0], off_x : off_x + arr.shape[1]] |= arr

        x0, y0 = low
        return ROI.from_array(canvas, x0=x0, y0=y0)

    def __repr__(self) -> str:
        """A call that reconstructs this ROI, e.g. `ROI.circle(300, 350, radius=40,
        color="#1f77b4", label="Cu")`. `self.params` keeps exactly the shape kwargs the
        caller originally gave (see __init__), so this reproduces them verbatim -- a
        `radius=` stays `radius=`, an omitted `rotation` stays omitted, etc.

        `shape_name == "array"` (from_array, or a `__add__` union) has no parametric
        form, so the repr says so explicitly instead of pretending to be eval-able.
        """
        x0, y0 = int(self.position[0]), int(self.position[1])
        color_kw = f"color={self.color!r}"
        label_kw = f", label={self.label!r}" if self.label is not None else ""

        if self.shape_name == "array":
            h, w = self.array.shape
            return (
                f"ROI.from_array(<bool array {h}x{w}>, x0={x0}, y0={y0}, "
                f"{color_kw}{label_kw})"
            )
        if self.shape_name == "point":
            return f"ROI.point({x0}, {y0}, {color_kw}{label_kw})"
        if self.shape_name == "polygon":
            corners = np.asarray(self.params["corners"]) + self.position
            corners_repr = "[" + ", ".join(f"({int(x)}, {int(y)})" for x, y in corners) + "]"
            return f"ROI.polygon({corners_repr}, {color_kw}{label_kw})"

        params_repr = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        params_kw = f", {params_repr}" if params_repr else ""
        if self.shape_name in ("circle", "ellipse"):
            return f"ROI.{self.shape_name}({x0}, {y0}{params_kw}, {color_kw}{label_kw})"
        # square/rectangle have no ROI.<shape> factory -- use the raw constructor.
        return f"ROI({x0}, {y0}, source={self.shape_name!r}{params_kw}, {color_kw}{label_kw})"


class Profile(StyledObject):
    """Extract profiles from images."""

    def __init__(
        self,
        points: Sequence,
        width: int = 1,
        style: dict[str, Any] = None,
        color: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.points = points
        self.width = width
        self.label = label

        super().__init__(style=style, color=color)

    # def apply(self, obj: do.DataObject):
    #     ret_val = np.array([])
    #     for src, dst in zip(self.points, self.points[1:]):
    #         ret_val = np.append(
    #             ret_val,
    #             profile_line(obj.image, src[::-1], dst[::-1], self.width),
    #         )

    #     return ret_val

    def apply(self, obj):
        profiles = []

        for src, dst in zip(self.points, self.points[1:]):
            profiles.append(
                profile_line(
                    obj.image,
                    src[::-1],
                    dst[::-1],
                    self.width,
                )
            )

        return np.concatenate(profiles)

    @staticmethod
    def _add_ring(all_verts: list, all_codes: list, ring_points: np.ndarray) -> None:
        """Append a closed polygon ring to the given vertex/code lists, forcing
        counter-clockwise winding so that a nonzero-winding fill never cancels an
        overlap between rings into a hole."""
        ring_points = np.asarray(ring_points, dtype=float)
        signed_area = np.sum(
            ring_points[:, 0] * np.roll(ring_points[:, 1], -1)
            - np.roll(ring_points[:, 0], -1) * ring_points[:, 1]
        )
        if signed_area < 0:
            ring_points = ring_points[::-1]
        all_verts.extend(ring_points)
        all_verts.append(ring_points[0])
        all_codes.append(MplPath.MOVETO)
        all_codes.extend([MplPath.LINETO] * (len(ring_points) - 1))
        all_codes.append(MplPath.CLOSEPOLY)

    def band_path(self) -> Optional[MplPath]:
        """Build a single compound Path for the whole profile band: one rectangle per
        segment (width `self.width`) plus a disk at each interior vertex to round the
        join. One Path avoids gaps at corners and double-blended alpha seams that one
        polygon per segment would produce. Returns None if there's no segment to draw."""
        points = [np.asarray(p, dtype=float) for p in self.points]
        # drop consecutive duplicates: zero-length segments have no direction
        deduped = points[:1]
        for point in points[1:]:
            if np.linalg.norm(point - deduped[-1]) > 0:
                deduped.append(point)
        points = deduped
        if len(points) < 2:
            return None

        half_width = self.width / 2
        all_verts: list = []
        all_codes: list = []

        for p0, p1 in zip(points, points[1:]):
            direction = p1 - p0
            normal = np.array([-direction[1], direction[0]])
            normal = normal / np.linalg.norm(normal) * half_width
            self._add_ring(all_verts, all_codes, [p0 + normal, p1 + normal, p1 - normal, p0 - normal])

        for vertex in points[1:-1]:  # interior joints only; endpoints get flat caps
            theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
            disk = vertex + half_width * np.column_stack([np.cos(theta), np.sin(theta)])
            self._add_ring(all_verts, all_codes, disk)

        return MplPath(all_verts, all_codes)

    def plot(self, ax):
        """Plot the profile true-to-size in data coordinates: a filled band `self.width`
        pixels wide with rounded joins, plus a thin centerline. Unlike a plain
        `ax.plot(..., linewidth=self.width)`, thickness is in data/pixel units, so it
        stays true-to-size across zoom levels and figure sizes."""
        alpha = self.style.get("alpha", 0.5)
        band_path = self.band_path()
        if band_path is not None:
            ax.add_patch(PathPatch(band_path, facecolor=self.color, edgecolor="none", alpha=alpha))
        xvalues = [point[0] for point in self.points]
        yvalues = [point[1] for point in self.points]
        ax.plot(xvalues, yvalues, color=self.color, linewidth=1, alpha=alpha)

    def __repr__(self) -> str:
        """A call that reconstructs this Profile, e.g. `Profile([(100, 100), (400, 300)],
        width=10, color="#9467bd", label="step edge")`."""
        points_repr = "[" + ", ".join(f"({int(x)}, {int(y)})" for x, y in self.points) + "]"
        color_kw = f"color={self.color!r}"
        label_kw = f", label={self.label!r}" if self.label is not None else ""
        return f"Profile({points_repr}, width={self.width!r}, {color_kw}{label_kw})"