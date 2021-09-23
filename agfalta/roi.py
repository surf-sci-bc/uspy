"""
Classes for defining shapes and regions on images.
"""
# pylint: disable=abstract-method
from __future__ import annotations
from numbers import Number
from typing import Any, Union, Optional
from collections.abc import Iterable

import cv2
import numpy as np
import matplotlib as mpl

from agfalta.dataobject import Loadable, Image


class StyledObject(Loadable):
    """Contains a style dictionary that can have class-wise defaults. The dictionary
    is intended for matplotlib keyword arguments."""
    _idx = 0
    _default_style = {}

    """Contains information on how to include this object into a matplotlib plot."""
    def __init__(self, style: dict[str, Any] = None, color: Optional[str] = None) -> None:
        style_ = self._default_style.copy()
        if style is not None:
            style_.update(style)
        if color is not None:
            style_["color"] = color
        if "color" not in style_:
            cycler = mpl.rc_params()["axes.prop_cycle"]
            color = cycler.by_key()["color"][StyledObject._idx]
            StyledObject._idx += 1
            if StyledObject._idx + 1 > len(cycler):
                StyledObject._idx = 0
            style_["color"] = color
        self.style = style_

    @property
    def color(self) -> str:
        """Convenience access to the color."""
        return self.style["color"]

    @property
    def artist(self) -> mpl.artist.Artist:
        """ROI Representation in matplotlib."""
        raise NotImplementedError


class Contour(StyledObject):
    """Define a contour in two dimensions thorugh a list of 2D coordinates."""
    _default_style = {"fill": False, "linewidth": 3}

    def __init__(self, corners: Optional[Iterable[Iterable]] = None,
                 ref_point: Optional[tuple] = None,
                 style: dict[str, Any] = None, color: Optional[str] = None) -> None:
        super().__init__(style=style, color=color)
        self.corners = np.array(corners)
        self.corners -= self.corners.min(axis=0)
        if not self.corners.shape[1] == 2:
            raise ValueError(f"Contour points are not 2-dimensional (shape: {self.corners.shape})")
        if ref_point is None:
            self._ref_point = self.center_of_mass
        else:
            self._ref_point = np.array([ref_point]).squeeze()
            self._ref_point -= self.corners.min(axis=0)
        self._shape = "polygon"
        self.params = dict(width=None, height=None, rotation=None)

    @classmethod
    def from_params(cls, *args, shape: Optional[str] = "circle", **kwargs):
        """Convert a geometric rectangular or elliptic description to polygon contour.
        Arguments belonging to the shapes:
            - circle: radius
            - ellipse: width, height, rotation
            - square: width, rotation
            - rectangle: width, height, rotation
        where radius, width and height are in pixels; rotation is in degrees (anti-clockwise).
        """
        if "radius" in kwargs:
            if "width" in kwargs:
                raise ValueError("Both radius and width were given for a shape-ROI")
            width = kwargs.pop("radius") * 2
        else:
            width = kwargs.pop("width")
        height = kwargs.pop("height", width)
        rotation = kwargs.pop("rotation", 0)

        if shape in ("circle", "ellipse"):
            corners = cv2.ellipse2Poly(
                center=(0, 0), axes=(width // 2, height // 2),
                angle=int(rotation), arcStart=0, arcEnd=360, delta=1
            )
        elif shape in ("square", "rectangle"):
            rot_c, rot_s = np.cos(-rotation * np.pi / 180), np.sin(-rotation * np.pi / 180)
            rot_matrix = np.array([[rot_c, -rot_s], [rot_s, rot_c]])
            corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])
            corners = np.rint(np.dot(corners, rot_matrix)).astype(np.int32)
            corners = corners - corners.min(axis=0)
        else:
            raise ValueError(f"Unknown shape {shape}")

        obj = cls(*args, corners=corners, **kwargs)
        obj.params = dict(width=width, height=height, rotation=rotation)
        obj._shape = shape
        return obj

    @property
    def shape(self) -> str:
        """Shape should be immutable."""
        return self._shape

    @property
    def ref_point(self) -> np.ndarray:
        """Reference point for the contour."""
        return self._ref_point
    @ref_point.setter
    def ref_point(self, value: np.ndarray):
        if not value.shape[1] == 2:
            raise ValueError(f"Given reference point is not 2-dimensional (shape: {value.shape})")
        self._ref_point = value

    @property
    def center_of_mass(self) -> np.ndarray:
        """The center of mass position of the mask array."""
        x, y = self.corners[:, 0].mean(), self.corners[:, 1].mean()
        return np.array([x, y])

    @property
    def artist(self) -> mpl.artist.Artist:
        art = mpl.patches.Polygon(self.corners, **self.style)
        return art

    def get_artist(self, position: Optional[Iterable]) -> mpl.artist.Artist:
        """Create artist at a given position (x, y)."""
        if position is None:
            position = self.ref_point
        if self._shape == "polygon":
            art = mpl.patches.Polygon(self.corners + position, **self.style)
        elif self._shape in ("circle", "ellipse"):
            art = mpl.patches.Ellipse(
                self.center_of_mass - self.ref_point + position,
                self.params["width"], self.params["height"], angle=self.params["rotation"],
                **self.style,
            )
        elif self._shape in ("square", "rectangle"):
            # pylint: disable=invalid-unary-operand-type
            rot_c = np.cos(-self.params["rotation"] * np.pi / 180)
            rot_s = np.sin(-self.params["rotation"] * np.pi / 180)
            rot_matrix = np.array([[rot_c, -rot_s], [rot_s, rot_c]])
            lower_left = np.dot([-self.params["width"] / 2, -self.params["height"] / 2], rot_matrix)
            art = mpl.patches.Rectangle(
                lower_left + self.center_of_mass - self.ref_point + position,
                self.params["width"], self.params["height"], angle=self.params["rotation"],
                **self.style,
            )
        else:
            raise ValueError(f"Unknown shape {self._shape}")
        return art


# class Mask(StyledObject):
#     """Filled contour."""
#     def __init__(self, contour_or_array: Union[Contour,Iterable],
#                  style: dict[str, Any] = None, color: Optional[str] = None) -> None:
#         if isinstance(contour_or_array, Iterable):
#             self.contour = None
#             self.array = np.array(contour_or_array)
#             if len(self.array.shape) != 2:
#                 raise ValueError("Array is not two-dimensional.")
#         else:
#             self.contour = contour_or_array
#             self.array = self._make_polygon(self.contour)
#             if style is None:
#                 style = self.contour.style
#         super().__init__(style=style, color=color)

#     @staticmethod
#     def _make_polygon(contour: Contour) -> np.ndarray:
#         """Create a polygon mask."""
#         corners = contour.corners - contour.corners.min(axis=0)
#         array = np.zeros(corners.max(axis=0)[::-1] + 1)
#         return cv2.fillConvexPoly(array, corners, color=1).astype(np.bool)

#     @property
#     def area(self) -> int:
#         """Total area in pixels. Is only valid when the mask is not cut off at the sides!"""
#         return self.array.astype(np.bool).sum()

#     @property
#     def ref_point(self) -> np.ndarray:
#         """Reference point for the mask."""
#         return self.contour.ref_point

#     @property
#     def center_of_mass(self) -> np.ndarray:
#         """The center of mass position of the mask array."""
#         x_i, y_i = np.nonzero(self.array)
#         return np.array([x_i.mean(), y_i.mean()])


class ROI(StyledObject):
    """An image region represented as a boolean 2D array."""
    # take care: OpenCV-points are given as (x, y),
    # but numpy 2d image arrays are in (y, x)-coordinates
    def __init__(self, x0: int, y0: int, source: Union[Contour,Iterable],
                 style: dict[str, Any] = None, color: Optional[str] = None) -> None:
        if isinstance(source, Iterable):
            self.contour = None
            self.array = np.array(source)
            if len(self.array.shape) != 2:
                raise ValueError("Array is not two-dimensional.")
        else:
            self.contour = source
            self.array = self.contour2array(self.contour)
            if style is None:
                style = self.contour.style

        super().__init__(style=style, color=color)
        self.position = np.array([x0, y0])
        self._mask_buffer = np.array([[]])
        self._center_buffer = True

    @staticmethod
    def contour2array(contour: Contour) -> np.ndarray:
        """Create a polygon mask."""
        corners = contour.corners - contour.corners.min(axis=0)
        array = np.zeros(corners.max(axis=0)[::-1] + 1)
        return cv2.fillConvexPoly(array, corners, color=1).astype(np.bool)

    @classmethod
    def circle(cls, x0: int, y0: int, radius: Number, rotation: Number = 0,
               style: dict[str, Any] = None, color: Optional[str] = None, **kwargs) -> ROI:
        """Construct a circular ROI."""
        contour = Contour.from_params(
            shape="circle", radius=radius, rotation=rotation,
            style=style, color=color
        )
        return cls(x0, y0, source=contour, **kwargs)

    @classmethod
    def ellipse(cls, x0: int, y0: int, width: int, height: int, rotation: Number = 0,
                style: dict[str, Any] = None, color: Optional[str] = None, **kwargs) -> ROI:
        """Construct an elliptic ROI."""
        contour = Contour.from_params(
            shape="ellipse", width=width, height=height, rotation=rotation,
            style=style, color=color
        )
        return cls(x0, y0, source=contour, **kwargs)

    @classmethod
    def rectangle(cls, x0: int, y0: int, width: int, height: int, rotation: Number = 0,
                  style: dict[str, Any] = None, color: Optional[str] = None, **kwargs) -> ROI:
        """Construct a rectangular ROI."""
        contour = Contour.from_params(
            shape="rectangle", width=width, height=height, rotation=rotation,
            style=style, color=color
        )
        return cls(x0, y0, source=contour, **kwargs)

    @classmethod
    def polygon(cls, x0: int, y0: int, corners: Iterable,
                style: dict[str, Any] = None, color: Optional[str] = None, **kwargs) -> ROI:
        """Construct a polygonic ROI from a list of corners."""
        contour = Contour(corners=np.array(corners), style=style, color=color)
        return cls(x0, y0, source=contour, **kwargs)

    @classmethod
    def point(cls, x0: int, y0: int, **kwargs) -> ROI:
        """Construct a point ROI."""
        array = np.array([[True]])
        return cls(x0, y0, source=array, **kwargs)

    @classmethod
    def from_array(cls, x0: int, y0: int, array: Iterable, **kwargs) -> ROI:
        """Construct a polygonic ROI from a list of corners. Seems Superfluous?????"""
        return cls(x0, y0, source=array, **kwargs)

    def apply(self, obj: Image, return_array: bool = False) -> Union[Image,np.ndarray]:
        """Apply the MaskLike to a Dataobject and either return the masked
        DataObject or the raw masked data."""
        full_mask = self.pad_to(*obj.image.shape).astype(np.bool)
        if return_array:
            return np.ma.masked_array(obj.image, mask=~full_mask)
        result = obj.copy()
        result.image = np.ma.masked_array(result.image, mask=~full_mask)
        return result

    def pad_to(self, width: int, height: int, center: bool = False) -> np.ndarray:
        """Return the the mask padded to a given extent."""
        if self._mask_buffer.shape == (height, width) and center == self._center_buffer:
            return self._mask_buffer
        # find sizes and corners. The low corner is the difference between the mask's
        # center of mass and the position:
        if center:
            mask_ref = self.center_of_mass
        else:
            mask_ref = self.ref_point
        mask_size = self.array.shape
        pad_size = np.array((width, height))
        low_corner = (self.position[::-1] - mask_ref[::-1]).astype(int)
        high_corner = (pad_size - mask_size - low_corner).astype(int)

        result = np.zeros(pad_size)
        # crop the mask, if need be (i.e., if corners have negative components):
        crop_low = np.clip(-low_corner, 0, None)
        crop_high = mask_size - np.clip(-high_corner, 0, None)
        if any(crop_low > mask_size) or any(crop_high < 0):
            return result
        mask = self.array[crop_low[0]:crop_high[0], crop_low[1]:crop_high[1]]

        # pad the mask until it matches the pad_size:
        pad_low = np.clip(low_corner, 0, None)
        pad_high = pad_size - np.clip(high_corner, 0, None)
        result[pad_low[0]:pad_high[0], pad_low[1]:pad_high[1]] = mask
        self._mask_buffer = result
        self._center_buffer = center

        return result

    @property
    def area(self) -> int:
        """Total area in pixels. Is only valid when the mask is not cut off at the sides!"""
        return self.array.astype(np.bool).sum()

    @property
    def ref_point(self) -> np.ndarray:
        """Reference point for the mask."""
        if self.contour is None:
            return np.array([0, 0])
        return self.contour.ref_point

    @property
    def center_of_mass(self) -> np.ndarray:
        """The center of mass position of the mask array."""
        x_i, y_i = np.nonzero(self.array)
        return np.array([x_i.mean(), y_i.mean()])

    @property
    def artist(self) -> mpl.artist.Artist:
        if isinstance(self.contour, Contour):
            return self.contour.get_artist(self.position)
        if self.array.shape == (1, 1):
            self.style["markersize"] = self.style.get("markersize", 10)
            self.style["marker"] = self.style.get("marker", "x")        # "X" is also good
            art = mpl.lines.Line2D([self.position[0]], [self.position[1]], **self.style)
            return art
        # something with art = mpl.image.AxesImage()??
        raise NotImplementedError

    # def __add__(self, other: ROI) -> ROI:
    #     return ROI.from_array(self.mask.array + other.mask.array)



class Profile:
    """Extract profiles from images."""



# class ROIold:
#     """An image region represented as a boolean 2D array."""
#     # take care: OpenCV-points are given as (x, y),
#     # but numpy 2d image arrays are in (y, x)-coordinates
#     _shapes = ("circle", "ellipse", "square", "rectangle", "polygon", "custom")
#     _param_keys = ("radius", "width", "height", "rotation")

#     def __init__(self, x0: int, y0: int,
#                  shape: Optional[str] = None,
#                  corners: Optional[Iterable[Iterable]] = None,
#                  array: Optional[np.ndarray] = None,
#                  style: Optional[dict] = None, **params):
#         self.position = np.array([y0, x0])
#         self.params = params
#         self._style = style

#         self.shape = shape
#         self.corners = corners
#         self.array = array
#         try:
#             if self.corners is not None:
#                 assert self.shape is None and self.array is None
#                 self.corners = np.array(self.corners)
#                 assert self.corners.shape[1] == 2
#                 self.shape = "polygon"
#                 self._make_polygon()
#             elif self.array is not None:
#                 assert self.corners is None and self.shape is None
#                 assert isinstance(self.array, np.ndarray) and len(self.array.shape) == 2
#                 self.shape = "custom"
#                 self.corners = np.array([self.array.shape[::-1]]) // 2
#             else:
#                 assert self.corners is None and self.array is None
#                 if self.shape is None:
#                     self.shape = "circle"
#                 assert self.shape in self._shapes
#                 self._make_shape()
#             assert isinstance(self.array, np.ndarray) and len(self.array.shape) == 2
#         except AssertionError as exc:
#             raise ValueError(f"Either more than one of {shape=}|{corners=}|{array=} "
#                               "were given or their type is not compatible") from exc

#     def _make_polygon(self) -> None:
#         """Create a polygon mask."""
#         self.corners -= self.corners.min(axis=0)
#         self.array = np.zeros(self.corners.max(axis=0)[::-1] + 1)
#         self.array = cv2.fillConvexPoly(self.array, self.corners, color=1).astype(np.bool)

#     def _make_shape(self) -> None:
#         """Create a rectangular or elliptic mask. Arguments belonging to the shapes:
#             - circle: radius
#             - ellipse: width, height, rotation
#             - square: width, rotation
#             - rectangle: width, height, rotation
#         where radius, width and height are in pixels; rotation is in degrees (anti-clockwise).
#         """
#         if "radius" in self.params:
#             if "width" in self.params:
#                 raise ValueError("Both radius and width were given for a shape-ROI")
#             self.params["width"] = self.params.pop("radius") * 2
#         self.params["height"] = self.params.get("height", self.params["width"])
#         self.params["rotation"] = self.params.get("rotation", 0)

#         width, height = self.params["width"], self.params["height"]
#         rotation = -self.params["rotation"]

#         if self.shape in ("circle", "ellipse"):
#             self.corners = cv2.ellipse2Poly(
#                 center=(0, 0), axes=(width // 2, height // 2),
#                 angle=int(rotation), arcStart=0, arcEnd=360, delta=1
#             )
#         elif self.shape in ("square", "rectangle"):
#             rot_c, rot_s = np.cos(-rotation * np.pi / 180), np.sin(-rotation * np.pi / 180)
#             rot_matrix = np.array([[rot_c, -rot_s], [rot_s, rot_c]])
#             corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])
#             corners = np.rint(np.dot(corners, rot_matrix)).astype(np.int32)
#             self.corners = corners - corners.min(axis=0)
#         else:
#             raise ValueError(f"Unknown shape {self.shape}")
#         self._make_polygon()

#     @property
#     def style(self) -> dict[str, Any]:
#         """Return a style dictionary."""
#         return self._style

#     @property
#     def mask(self) -> np.ndarray:
#         """Return a bool array."""
#         return self.array.astype(np.bool)

#     def apply(self, obj: Image, return_array: bool = False) -> Union[Image,np.ndarray]:
#         """Apply the MaskLike to a Dataobject and either return the masked
#         DataObject or the raw masked data."""
#         full_mask = self.pad_to(*obj.image.shape).astype(np.bool)
#         if return_array:
#             return full_mask * obj.image
#         result = obj.copy()
#         result.image = np.ma.masked_array(result.image, mask=full_mask)
#         return result

#     @property
#     def area(self) -> int:
#         """Total area in pixels. Is only valid when the mask is not cut off at the sides!"""
#         return self.mask.sum()

#     def pad_to(self, width: int, height: int):
#         """Return the the mask padded to a given extent."""
#         # find sizes and corners. The low corner is the difference between the mask's
#         # center of mass and the position:
#         mask_size = self.mask.shape
#         pad_size = np.array((height, width))
#         low_corner = (self.position - self.corners.mean(axis=0)).astype(int)
#         high_corner = (pad_size - mask_size - low_corner).astype(int)

#         result = np.zeros(pad_size)
#         # crop the mask, if need be (i.e., if corners have negative components):
#         crop_low = np.clip(-low_corner, 0, None)
#         crop_high = mask_size - np.clip(-high_corner, 0, None)
#         if any(crop_low > mask_size) or any(crop_high < 0):
#             return result
#         mask = self.mask[crop_low[0]:crop_high[0], crop_low[1]:crop_high[1]]

#         # pad the mask until it matches the pad_size:
#         pad_low = np.clip(low_corner, 0, None)
#         pad_high = pad_size - np.clip(high_corner, 0, None)
#         result[pad_low[0]:pad_high[0], pad_low[1]:pad_high[1]] = mask
#         return result

#     @property
#     def artist(self) -> mpl.artist.Artist:
#         """ROI Representation in matplotlib."""
