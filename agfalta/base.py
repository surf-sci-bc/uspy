"""
Basic data containers.
"""
# pylint: disable=abstract-method
from __future__ import annotations
from typing import Any, Union, Optional
from collections.abc import Iterable
from numbers import Number
import copy
import pickle
from pathlib import Path
from skimage.io import imread


from deepdiff import DeepDiff
import cv2
import numpy as np
import matplotlib as mpl


class Loadable:
    """
    Base class for loadable objects. Contains methods for serializing
    and deserializing.
    """
    class ThinObject:
        """
        Inner class which is a thin version of the full Loadable. Meant for
        lightweight data saving.
        """
        def __init__(self, obj: Loadable) -> None:
            self.reduce(obj)

        def reduce(self, obj: Loadable) -> None:
            """Make the ThinObject a light version of the full Loadable
            that can easily be serialized. E.g.: Retain only filenames instead
            of the raw data."""
            state = obj.__dict__
            self.__dict__.update(state)

        def reconstruct(self) -> Loadable:
            """Rebuild the full Loadable from the ThinObject."""
            obj = Loadable()
            state = self.__dict__
            obj.__dict__.update(state)
            return obj

    def dump(self, fname: str, thin: bool = False) -> None:
        """Dumps the object into a JSON/pickle file."""
        if thin:
            obj = Loadable.ThinObject(self)
        else:
            obj = self
        with Path(fname).open("wb") as pfile:
            try:
                pickle.dump(obj, pfile, protocol=4)
            except RecursionError:
                print("Did not save due to recursion error.")
                raise

    @classmethod
    def load(cls, fname: str) -> Loadable:
        """Returns an object retrieved from a JSON/pickle file."""
        with Path(fname).open("wb") as pfile:
            obj = pickle.load(pfile)
        if isinstance(obj, Loadable.ThinObject):
            obj = obj.reconstruct()
        return obj


class DataObject:
    """
    Base class for data objects like images, lines, points, ...
    """
    _data_keys = ()
    _unit_defaults = {}
    _meta_defaults = {}

    def __init__(self, source) -> None:
        self._data = {}
        self._meta = self._meta_defaults.copy()
        self._units = self._unit_defaults.copy()
        self._source = None
        parsed = self.parse(source)
        for k in self._data_keys:
            if k not in parsed:
                raise ValueError(f"Parse function did not give '{k}' data.")
        for k, value in parsed.items():
            if k.startswith("_"):
                continue
            if k in self._data_keys:
                self._data[k] = value
            elif "_unit" in k:
                self._units[k.replace("_unit", "")] = value
            else:
                self._meta[k] = value
        for k in self._meta:
            if k not in self._unit_defaults:
                self._unit_defaults[k] = ""

    def parse(self, source: str) -> dict[str, Any]:
        """Read in data. Should set the _source attribute."""
        raise NotImplementedError

    def copy(self) -> DataObject:
        """Return a deep copy."""
        return copy.deepcopy(self)

    @property
    def source(self) -> Optional[Any]:
        """The original object from which this instance was created (e.g., filename)."""
        return self._source

    @property
    def data(self) -> dict[str, Any]:
        """Mutable data container dictionary."""
        return self._data

    @property
    def meta(self) -> dict[str, Any]:
        """Mutable metadata dictionary."""
        return self._meta

    @property
    def unit(self) -> dict[str, str]:
        """Dictionary with unit strings for meta values."""
        return self._units

    def get_field_string(self, field: str, fmt: str = ".5g") -> str:
        """Get a string that contains the value and its unit."""
        value = getattr(self, field)
        if value == np.nan:
            return "NaN"
        if isinstance(value, Number):
            value = f"{value:{fmt}}"
        unit = self._units.get(field, "")
        return f"{value} {unit}".strip()

    def __getattr__(self, attr: str) -> Any:
        if attr in ("_meta", "_data"):
            raise AttributeError("Premature access to _meta or _data")
        if attr in self._data:
            return self._data[attr]
        if attr in self._meta:
            return self._meta[attr]
        raise AttributeError(f"No attribute named {attr}")

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr.startswith("_") or hasattr(self, attr):
            super().__setattr__(attr, value)
        elif attr in self._data:
            self._data[attr] = value
        else:
            self._meta[attr] = value

    def __eq__(self, other: DataObject):
        try:
            assert not DeepDiff(
                self.meta, other.meta, ignore_order=True,
                number_format_notation="e", significant_digits=5,
                ignore_numeric_type_changes=True
            )
            assert not DeepDiff(
                self.data, other.data, ignore_order=True,
                number_format_notation="e", significant_digits=5,
                ignore_numeric_type_changes=True
            )
        except AssertionError:
            return False
        return True

    def is_compatible(self, other: DataObject) -> bool:
        """Check if another DataObject has data of the same dimensions."""
        return isinstance(other, type(self))


class DataObjectStack(Loadable):
    """Contains multiple DataObjects. E.g., for an image stack."""
    _unique_attrs = ()
    _type = DataObject

    def __init__(self, source: Union[str,Iterable], virtual: bool = False) -> None:
        self._virtual = virtual
        if isinstance(source, Iterable) and isinstance(source[0], DataObject):
            if virtual:
                print("WARNING: Stack won't be virtual (data objects were directly given)")
                self._virtual = False
            # if stack is created from objects, all objects have to be the
            for obj in source[1:]:
                if not source[0].is_compatible(obj):
                    raise TypeError(f"Not all initialization objects are of type {self._type}")

            self._elements = source
        else:
            self._elements = self._split_source(source)
            if not self.virtual:
                self._construct()

    def _split_source(self, source: Union[str, Iterable]) -> list:
        """Split the source parameter of the constructor into source arguments
        that can be passed to single DataObjects."""
        return source

    def _construct(self) -> None:
        """Build the stack from a list of sources."""
        self._virtual = False
        sources = self._elements
        if not sources:
            raise ValueError("Empty source for DataObjectStack (wrong file path?)")
        self._elements = [self._single_construct(sources[0])]
        for source in sources[1:]:
            element = self._single_construct(source)
            assert self._elements[0].is_compatible(element)
            self._elements.append(element)

    def _deconstruct(self) -> None:
        """Retain the source objects instead of the direct elements."""
        self._virtual = True
        sources = [element.source for element in self._elements]
        if None in sources:
            raise ValueError("Can't virtualize stack without source information.")
        self._elements = sources

    def _single_construct(self, source: Any) -> DataObject:
        """Construct a single DataObject."""
        return self._type(source)

    @property
    def elements(self) -> list:
        """Contents."""
        return self._elements
    @property
    def sources(self) -> list:
        """Sources."""
        if self.virtual:
            return self._elements
        return [element.source for element in self._elements]

    @property
    def virtual(self) -> bool:
        """If a data stack is virtual, it does only load the data on demand."""
        return self._virtual
    @virtual.setter
    def virtual(self, value: bool) -> None:
        if self._virtual == value:
            return
        if value:
            self._deconstruct()
        else:
            self._construct()
        self._virtual = value

    def copy(self, virtual: Optional[bool] = None):
        """Returns a deepcopy."""
        other = copy.deepcopy(self)
        if virtual is not None:
            other.virtual = virtual
        return other

    def __getitem__(self, index: Union[int,slice]) -> Union[DataObject,DataObjectStack]:
        elements = self._elements[index]
        if isinstance(index, int):
            if self.virtual: # if virtual elements contains just sources not DataObjects
                return self._single_construct(elements)
            return elements
        return type(self)(elements, virtual=self.virtual)

    def __setitem__(self, index: Union[int,slice], other: Union[DataObject,Iterable]) -> None:
        # check compatibility -- implement in dataobject? (like img size)
        if isinstance(index, int) and isinstance(other, DataObject):
            if self.virtual:
                if other.source is None:
                    raise ValueError("Can't put DataObjects without source into virtual stack")
                insert = other.source
            else:
                insert = other
                if not self._elements[0].is_compatible(other):
                    raise ValueError("Incompatible element assignment")

        elif isinstance(index, slice):
            raise ValueError("Slices are not supported for item assignment.")

        self._elements.__setitem__(index, insert)

    def extend(self, other: Union[DataObject,Iterable]) -> None:
        """Add new DataObjects to the end of the stack."""
        if isinstance(other, DataObject):
            elements = [other]
        if isinstance(other, type(self)):
            if not self.virtual:
                elements = other[:]
            elif other.virtual:
                elements = other.sources

        if self.virtual:
            if None in elements:
                raise ValueError("Can't put DataObjects without source into virtual stack")
        else:
            for element in elements:
                if not self._elements[0].is_compatible(element):
                    raise ValueError("Incompatible element assignment")
        self._elements.extend(elements)

    def __getattr__(self, attr: str):
        if attr.startswith("_") or attr in self.__dict__:
            raise AttributeError
        return np.array([getattr(obj, attr) for obj in self])

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr.startswith("_") or attr in self.__dict__:
            return super().__setattr__(attr, value)
        if isinstance(value, Iterable) and len(value) == len(self):
            if self.virtual:
                raise ValueError(f"Can't set attribute '{attr}' for virtual stack")
            for obj, single_value in zip(self, value):
                setattr(obj, attr, single_value)
        else:
            if isinstance(value, Iterable) and len(value) != len(self):
                print(
                    f"Attribute {attr} with length {len(value)} cannot be assigned elementwise to "
                    f"stack with length {len(self)}. It will be assigned to the stack itself."
                )
            return super().__setattr__(attr, value)

    def __delitem__(self, index: Union[int,slice]) -> None:
        self._elements.__delitem__(index)

    def __len__(self) -> int:
        return len(self._elements)

    def __iadd__(self, other: Union[DataObjectStack,DataObject,Number]) -> DataObjectStack:
        if isinstance(other, type(self)):
            self.extend(other)
        else:
            if self.virtual:
                raise ValueError("Can't do this '+' operation on virtual stacks.")
            for element in self._elements:
                element += other
        return self
    def __add__(self, other: Union[DataObjectStack,DataObject,Number]) -> DataObjectStack:
        result = self.copy(virtual=False)
        result += other
        return result
    def __radd__(self, other: Union[DataObjectStack,DataObject,Number]) -> DataObjectStack:
        return self.__add__(other)

    def __isub__(self, other: Union[DataObject,DataObject,Number]) -> DataObjectStack:
        if self.virtual:
            raise ValueError("Can't do '-' on virtual stacks.")

        for element in self._elements:
            element -= other

        return self
    def __sub__(self, other: Union[DataObjectStack,DataObject,Number]) -> DataObjectStack:
        result = self.copy(virtual=False)
        result -= other
        return result

    def __imul__(self, other: Union[DataObject,Number]) -> DataObjectStack:
        if self.virtual:
            raise ValueError("Can't do '*' on virtual stacks.")
        for element in self._elements:
            element *= other
        return self
    def __mul__(self, other: Union[DataObject,Number]) -> DataObjectStack:
        result = self.copy(virtual=False)
        result *= other
        return result
    def __rmul__(self, other: Union[DataObject,Number]) -> DataObjectStack:
        return self.__mul__(other)

    def __itruediv__(self, other: Union[DataObject,Number]) -> DataObjectStack:
        if self.virtual:
            raise ValueError("Can't do '/' on virtual stacks.")
        for element in self._elements:
            element /= other
        return self

    def __truediv__(self, other: Union[DataObject,Number]) -> DataObjectStack:
        result = self.copy(virtual=False)
        result /= other
        return result


class Image(DataObject):
    """
    Base class for all spatial 2D data.
    """
    # pylint: disable=no-member
    _data_keys = ("image",)

    def __init__(self, *args, **kwargs) -> None:
        self._mask = None
        super().__init__(*args, **kwargs)

    def parse(self, source: str) -> dict[str, Any]:
        return {"image": np.float32(imread(source))}

    @property
    def mask(self) -> np.ndarray:
        """Set a mask onto the image. TODO"""
        return self._mask
    @mask.setter
    def mask(self, value: np.ndarray) -> None:
        pass

    def __iadd__(self, other: Union[Image,Number,np.ndarray]) -> Image:
        if isinstance(other, Image):
            self.image += other.image
        elif isinstance(other, (Number, np.ndarray)):
            self.image += other
        else:
            raise TypeError(f"Unsupported Operation '+' for types {type(self)} and {type(other)}")
        return self
    def __add__(self, other: Union[Image,Number,np.ndarray]) -> Image:
        result = self.copy()
        result += other
        return result
    def __radd__(self, other: Union[Image,Number,np.ndarray]) -> Image:
        if other == 0:
            return self
        return self.__add__(other)

    def __isub__(self, other: Union[Image,Number,np.ndarray]) -> Image:
        if isinstance(other, Image):
            self.image -= other.image
        elif isinstance(other, (Number, np.ndarray)):
            self.image -= other
        else:
            raise TypeError(f"Unsupported Operation '-' for types {type(self)} and {type(other)}")
        return self
    def __sub__(self, other: Union[Image,Number,np.ndarray]) -> Image:
        result = self.copy()
        result -= other
        return result

    def __imul__(self, other: Union[Image,Number,np.ndarray]) -> Image:
        if isinstance(other, Image):
            self.image *= other.image
        elif isinstance(other, (Number, np.ndarray)):
            self.image *= other
        else:
            raise TypeError(f"Unsupported Operation '*' for types {type(self)} and {type(other)}")
        return self
    def __mul__(self, other: Union[Image,Number,np.ndarray]) -> Image:
        result = self.copy()
        result *= other
        return result
    def __rmul__(self, other: Union[Image,Number,np.ndarray]) -> Image:
        return self.__mul__(other)

    def __itruediv__(self, other: Union[Image,Number,np.ndarray]) -> Image:
        if isinstance(other, (Number, np.ndarray)):
            self.image /= other
        elif isinstance(other, Image):
            self.image /= other.image
        else:
            raise TypeError(f"Unsupported Operation '/' for types {type(self)} and {type(other)}")
        return self
    def __truediv__(self, other: Union[Image,Number,np.ndarray]) -> Image:
        result = self.copy()
        result /= other
        return result

    def is_compatible(self, other: DataObject) -> bool:
        if not super().is_compatible(other):
            return False
        try:
            return self.image.shape == other.image.shape
        except AttributeError:
            return False


class ROI(Loadable):
    """An image region represented as a boolean 2D array."""
    # take care: OpenCV-points are given as (x, y),
    # but numpy 2d image arrays are in (y, x)-coordinates
    _shapes = ("circle", "ellipse", "square", "rectangle", "polygon", "custom")
    _param_keys = ("radius", "width", "height", "rotation")

    def __init__(self, x0: int, y0: int,
                 shape: Optional[str] = None,
                 corners: Optional[Iterable[Iterable]] = None,
                 array: Optional[np.ndarray] = None,
                 style: Optional[dict] = None, **params):
        self.position = np.array([y0, x0])
        self.params = params
        self.style = style

        self.shape = shape
        self.corners = corners
        self.array = array
        try:
            if self.corners is not None:
                assert self.shape is None and self.array is None
                self.corners = np.array(self.corners)
                assert self.corners.shape[1] == 2
                self.shape = "polygon"
                self._make_polygon()
            elif self.array is not None:
                assert self.corners is None and self.shape is None
                assert isinstance(self.array, np.ndarray) and len(self.array.shape) == 2
                self.shape = "custom"
                self.corners = np.array([self.array.shape[::-1]]) // 2
            else:
                assert self.corners is None and self.array is None
                if self.shape is None:
                    self.shape = "circle"
                assert self.shape in self._shapes
                self._make_shape()
            assert isinstance(self.array, np.ndarray) and len(self.array.shape) == 2
        except AssertionError as exc:
            raise ValueError(f"Either more than one of {shape=}|{corners=}|{array=} "
                              "were given or their type is not compatible") from exc

    def _make_polygon(self) -> None:
        """Create a polygon mask."""
        self.corners -= self.corners.min(axis=0)
        self.array = np.zeros(self.corners.max(axis=0)[::-1] + 1)
        self.array = cv2.fillConvexPoly(self.array, self.corners, color=1).astype(np.bool)

    def _make_shape(self) -> None:
        """Create a rectangular or elliptic mask. Arguments belonging to the shapes:
            - circle: radius
            - ellipse: width, height, rotation
            - square: width, rotation
            - rectangle: width, height, rotation
        where radius, width and height are in pixels; rotation is in degrees (anti-clockwise).
        """
        if "radius" in self.params:
            if "width" in self.params:
                raise ValueError("Both radius and width were given for a shape-ROI")
            self.params["width"] = self.params.pop("radius") * 2
        self.params["height"] = self.params.get("height", self.params["width"])
        self.params["rotation"] = self.params.get("rotation", 0)

        width, height = self.params["width"], self.params["height"]
        rotation = -self.params["rotation"]

        if self.shape in ("circle", "ellipse"):
            self.corners = cv2.ellipse2Poly(
                center=(0, 0), axes=(width // 2, height // 2),
                angle=int(rotation), arcStart=0, arcEnd=360, delta=1
            )
        elif self.shape in ("square", "rectangle"):
            rot_c, rot_s = np.cos(-rotation * np.pi / 180), np.sin(-rotation * np.pi / 180)
            rot_matrix = np.array([[rot_c, -rot_s], [rot_s, rot_c]])
            corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])
            corners = np.rint(np.dot(corners, rot_matrix)).astype(np.int32)
            self.corners = corners - corners.min(axis=0)
        else:
            raise ValueError(f"Unknown shape {self.shape}")
        self._make_polygon()

    @property
    def mask(self) -> np.ndarray:
        """Returns smallest possible numpy array that encompasses the ROI."""
        return self.array.astype(np.bool)

    @property
    def area(self) -> int:
        """Total area in pixels."""
        return self.mask.sum()

    def pad_to(self, width: int, height: int):
        """Return the the mask padded to a given extent."""
        # find sizes and corners. The low corner is the difference between the mask's
        # center of mass and the position:
        mask_size = self.mask.shape
        pad_size = np.array((height, width))
        low_corner = (self.position - self.corners.mean(axis=0)).astype(int)
        high_corner = (pad_size - mask_size - low_corner).astype(int)

        result = np.zeros(pad_size)
        # crop the mask, if need be (i.e., if corners have negative components):
        crop_low = np.clip(-low_corner, 0, None)
        crop_high = mask_size - np.clip(-high_corner, 0, None)
        if any(crop_low > mask_size) or any(crop_high < 0):
            return result
        mask = self.mask[crop_low[0]:crop_high[0], crop_low[1]:crop_high[1]]

        # pad the mask until it matches the pad_size:
        pad_low = np.clip(low_corner, 0, None)
        pad_high = pad_size - np.clip(high_corner, 0, None)
        result[pad_low[0]:pad_high[0], pad_low[1]:pad_high[1]] = mask
        return result

    def __mul__(self, other: Union[Image,np.ndarray]) -> np.ndarray:
        """Apply the ROI by doing: result = roi * img"""

    @property
    def artist(self) -> mpl.artist.Artist:
        """ROI Representation in matplotlib."""

    @property
    def color(self) -> str:
        """Color used in any representations."""
        return self.style["color"]
