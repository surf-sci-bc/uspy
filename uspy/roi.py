"""
Classes for defining shapes and regions on images.
"""
# pylint: disable=abstract-method
from __future__ import annotations
from typing import Any, Union, Optional
from collections.abc import Iterable

import cv2
import numpy as np
import matplotlib as mpl
from scipy import ndimage

import uspy.dataobject as do


class StyledObject():#do.Loadable):
    """Contains a style dictionary that can have class-wise defaults. The dictionary
    is intended for matplotlib keyword arguments."""
    _idx = 0
    _default_style = {}

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

    def __init__(self, x0: int, y0: int,
                 source: Union[np.ndarray,str,Iterable],
                 style: dict[str, Any] = None, color: Optional[str] = None,
                 **kwargs) -> None:
        self.position = np.array([x0, y0])
        self._ref_point = np.array([0, 0])
        self.params = kwargs

        if isinstance(source, str):
            self._apply_params(shape=source, **self.params)
        elif isinstance(source, Iterable):
            pass            # polygon
        else:
            self.array = np.array(source)
        assert self.array.ndim == 2

        super().__init__(style=style, color=color)
        self._mask_buffer = np.array([[]])
        self._center_buffer = True

    def _apply_params(self, shape: Optional[str] = "circle", **kwargs):
        """Convert a geometric rectangular or elliptic description to polygon contour.
        Arguments belonging to the shapes:
            - circle: radius
            - ellipse: width, height, rotation
            - square: width, rotation
            - rectangle: width, height, rotation
        where radius, width and height are in pixels; rotation is in degrees (anti-clockwise).
        """
        if shape == "point":
            self._ref_point = np.array([0, 0])
            array = np.array([[1]])
            return

        if "radius" in kwargs:
            if "width" in kwargs:
                raise ValueError("Both radius and width were given for a shape-ROI")
            width = kwargs.pop("radius") * 2
        else:
            width = kwargs.pop("width")
        height = kwargs.pop("height", width)
        rotation = kwargs.pop("rotation", 0)

        max_extent = int(np.sqrt(width**2 + height**2))
        mask = np.zeros((max_extent, max_extent))

        if shape in ("circle", "ellipse"):
            self._ref_point = np.array([width / 2, height / 2])
            array = cv2.ellipse(
                mask,
                center=tuple(self._ref_point.astype(int)),
                axes=(width // 2, height // 2),
                angle=rotation,
                startAngle=0,
                endAngle=360,
                color=1,
                thickness=-1
            )
        elif shape in ("square", "rectangle"):
            self._ref_point = np.array([width / 2, height / 2])
            array = cv2.rectangle(
                mask,
                pt1=tuple(self._ref_point.astype(int)),
                pt2=(width, height),
                color=1,
                thickness=-1
            )
            array = ndimage.rotate(array, rotation)
        elif shape == "polygon":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown shape {shape}")

        self.array = array
        return array

    @classmethod
    def point(cls, x0: int, y0: int, **kwargs) -> ROI:
        """Construct a point ROI."""
        array = np.array([[True]])
        return cls(x0, y0, source=array, **kwargs)

    @classmethod
    def from_array(cls, x0: int, y0: int, array: Iterable, **kwargs) -> ROI:
        """Construct a polygonic ROI from a list of corners. Seems Superfluous?????"""
        return cls(x0, y0, source=array, **kwargs)

    def apply(self, obj: do.Image, return_array: bool = False) -> Union[do.Image,np.ndarray]:
        """Apply the MaskLike to a Dataobject and either return the masked
        DataObject or the raw masked data."""
        full_mask = self.pad_to(*obj.image.shape).astype(bool)
        if return_array:
            return np.ma.masked_array(obj.image, mask=~full_mask)
        result = obj.copy()
        result.image = np.ma.masked_array(result.image, mask=~full_mask)
        return result

    def pad_to(self, width: int, height: int) -> np.ndarray:
        """Return the the mask padded to a given extent."""
        if self._mask_buffer.shape == (height, width):
            return self._mask_buffer
        # find sizes and corners. The low corner is the difference between the mask's
        # reference point and the position:
        mask_size = self.array.shape
        pad_size = np.array((width, height)).astype(int)
        low_corner = self.origin[::-1].astype(int)
        high_corner = pad_size - mask_size - low_corner

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

    # def __add__(self, other: ROI) -> ROI:
    #     return ROI.from_array(self.mask.array + other.mask.array)



class Profile:
    """Extract profiles from images."""
