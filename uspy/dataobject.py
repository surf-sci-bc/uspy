"""
Basic data containers.
"""
# pylint: disable=abstract-method
from __future__ import annotations
import glob
from typing import Any, Callable, Sequence, Union, Optional
from collections.abc import Iterable
from numbers import Number
from pathlib import Path
import bz2
import copy
import pickle
import warnings

from deepdiff import DeepDiff
import numpy as np
import scipy.interpolate
import scipy.integrate
import pandas as pd
import imageio
import json_tricks
import tifffile
from tifffile.tifffile import TiffFileError
import cv2 as cv

import uspy.roi as roi


class Loadable:
    """
    Base class for loadable objects. Contains methods for serializing
    and deserializing.
    """

    _pickle_extension = ".pickle"

    def dump(self, fname: str, thin: bool = False, use_pickle: bool = True) -> None:
        """Dumps the object into a JSON/pickle file."""
        with Path(fname).open("wb") as pfile:
            pfile.write(self.dumps(thin, use_pickle))

    def dumps(self, _thin: bool = False, use_pickle: bool = True) -> None:
        """Dumps the object with pickle and returns it as a bytestring"""
        # if thin:
        #     obj = ThinObject(self)
        # else:
        #     obj = self
        # try:
        if use_pickle:
            try:
                return pickle.dumps(self, protocol=4)
            except RecursionError:
                print("Did not pickle due to recursion error.")
                raise
        else:
            return json_tricks.dumps(self, allow_nan=True).encode("utf-8")

    def save(self, fname: str) -> DataObject:
        """Save the DataObject to a file depending on the filename extension."""
        _fname = fname
        compress = thin = pickl = False

        if fname.endswith(self._pickle_extension + ".bz2"):
            compress = True
            pickl = True
        elif fname.endswith(self._pickle_extension):
            pickl = True
        elif not fname.endswith(".json"):
            raise ValueError("No meaningful fileextension recognized.")

        # if _fname.endswith(".bz2"):
        #     _fname = _fname.removesuffix(".bz2")
        #     compress = True
        # if _fname.endswith((".thin" + self._pickle_extension, ".thin.json")):
        #     thin = True
        # if _fname.endswith(self._pickle_extension):
        #     pickl = True
        # elif not _fname.endswith(".json"):
        #     raise ValueError("No meaningful fileextension recognized.")
        # if not _fname.endswith(self._pickle_extension):
        #     _fname += self._pickle_extension
        #     if compress:
        #         _fname += ".bz2"
        #     fname = _fname

        if compress:
            print("Compressing data...")
            with bz2.BZ2File(fname, "w") as bzfile:
                bzfile.write(self.dumps(thin, use_pickle=pickl))
            print("Done.")
        else:
            with Path(fname).open("wb") as pfile:
                pfile.write(self.dumps(thin, use_pickle=pickl))

        return self

    # def _reduce(self) -> dict:
    #     raise NotImplementedError

    # @classmethod
    # def _reconstruct(cls, state) -> Loadable:
    #     raise NotImplementedError

    @classmethod
    def load(cls, fname: str) -> Loadable:
        """Returns an object retrieved from a JSON/pickle file."""

        if fname.endswith(cls._pickle_extension + ".bz2"):
            print("Uncompressing data...")
            file = bz2.BZ2File(fname, "rb")
            obj = pickle.load(file)
            print("Done.")
        elif fname.endswith(cls._pickle_extension):
            with Path(fname).open("rb") as pfile:
                obj = pickle.load(pfile)
        elif fname.endswith(".json"):
            obj = json_tricks.load(fname)

        # if isinstance(obj, Loadable.ThinObject):
        #    obj = obj.reconstruct()
        return obj


# class ThinObject:
#     """
#     Inner class which is a thin version of the full Loadable. Meant for
#     lightweight data saving.
#     """

#     def __init__(self, obj: Loadable) -> None:
#         self._obj = obj
#         self._type = type(obj)

#     def __getstate__(self) -> dict:
#         state = self._obj._reduce()
#         for _, val in state.items():
#             # Check if something in state can be casted to Thinobject
#             # if isinstance(val, Loadable):
#             try:
#                 val = ThinObject(val)
#             # Is Loadable but has not _reduce function or is not Loadable
#             except (NotImplementedError, AttributeError):
#                 # See if it is a list of something that can be made thin
#                 try:
#                     for item in val:
#                         item = ThinObject(item)
#                 except (TypeError, AttributeError, NotImplementedError):
#                     pass  # do val stays val...

#             state["_type"] = self._type
#             # state.update(self._obj._reduce())
#             return state

#     def __setstate__(self, state: dict) -> None:
#         obj = state["_type"]._reconstruct(state)
#         # cast ThinObject into the Object that it was originally
#         self.__class__ = obj.__class__
#         self.__dict__ = obj.__dict__


class DataObject(Loadable):
    """
    Base class for data objects like images, lines, points, ...

    Parameters
    ----------
    source: Any
        The source from which the object is created
    """

    _data_keys = ()
    _unit_defaults = {}
    _meta_defaults = {}
    default_fields = ()

    def __new__(cls, *_args, **_kwargs):
        obj = super().__new__(cls)  # really not forward args and kwargs?
        obj._data = {}
        obj._meta = obj._meta_defaults.copy()
        obj._units = obj._unit_defaults.copy()
        obj._source = None
        return obj

    def __init__(self, source: Any) -> None:
        """__init__(self, source: Any)"""
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

    def get_field_string(self, field: str, fmt: Optional[str] = None) -> str:
        """Get a string that contains the value and its unit."""
        if fmt is None:
            fmt = ".5g"
        value = getattr(self, field)
        if value == np.nan:
            return "NaN"
        if isinstance(value, Number):
            value = f"{value:{fmt}}"
        unit = self._units.get(field, "")
        return f"{value} {unit}".strip()

    def __getattr__(self, attr: str) -> Any:
        # if attr in ("_meta", "_data"):
        #    raise AttributeError("Premature access to _meta or _data")
        if attr in self._data:
            return self._data[attr]
        if attr in self._meta:
            return self._meta[attr]
        raise AttributeError(f"No attribute named {attr}")

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr in ("_meta", "_data") and attr in self.__dict__:
            return
        if attr.startswith("_") or attr in self.__dict__:
            super().__setattr__(attr, value)
        elif attr in self._data:
            self._data[attr] = value
        else:
            self._meta[attr] = value

    def __eq__(self, other: DataObject):
        try:
            assert not DeepDiff(
                self.meta,
                other.meta,
                ignore_order=True,
                number_format_notation="e",
                significant_digits=5,
                ignore_numeric_type_changes=True,
            )
            assert not DeepDiff(
                self.data,
                other.data,
                ignore_order=True,
                number_format_notation="e",
                significant_digits=5,
                ignore_numeric_type_changes=True,
            )
        except AssertionError:
            return False
        return True

    def is_compatible(self, other: DataObject) -> bool:
        """[summary]

        Parameters
        ----------
        other : DataObject
            [description]

        Returns
        -------
        bool
            [description]
        """
        # """Check if another DataObject has data of the same dimensions."""
        return isinstance(other, type(self))

    # def _reduce(self) -> dict:
    #     return {"source": self._source}

    # @classmethod
    # def _reconstruct(cls, state: dict) -> DataObject:
    #     return cls(state["source"])

    def __json_encode__(self) -> dict:
        """Returns dict with all information to reconstruct object from a .json file"""
        return {"source": self._source}

    def __json_decode__(self, **attrs) -> None:
        """Accepts attributes from .json file and initilizes new object."""
        self.__init__(**attrs)

    def __str__(self) -> str:
        prettystring = ""
        prettystring += f"Source: {self.source}\n"
        prettystring += "Metadata:\n"
        for k, v in self.meta.items():
            unit = self.get(f"{k}_unit", "")
            prettystring += f"\t{k}:\t{v} (unit: {unit})\n"
        prettystring += "Data:\n"
        for k, v in self.data.items():
            prettystring += f"\t{k}:\t{v}\n"
        return prettystring


class DataObjectStack(Loadable):
    """Contains multiple DataObjects. E.g., for an image stack."""

    _unique_attrs = ()
    _type = DataObject

    def __init__(self, source: Union[str, Iterable], virtual: bool = False) -> None:
        self._virtual = virtual
        # If the Stack is initialized from a list of Objects
        if isinstance(source, Iterable) and isinstance(source[0], self._type):
            if virtual:
                warnings.warn(
                    UserWarning(
                        "Stack won't be virtual (data objects were directly given)"
                    )
                )
                self._virtual = False
            # if stack is created from objects, all objects have to be the same
            for obj in source[1:]:
                if not source[0].is_compatible(obj):
                    raise TypeError(
                        f"Not all initialization objects are of type {self._type}"
                    )

            self._elements = self._split_source(source)
            # self._elements = source
        else:
            self._elements = self._split_source(source)
            if not self.virtual:
                self._construct()

    def _split_source(self, source: Union[str, Iterable]) -> list:
        """Split the source parameter of the constructor into source arguments
        that can be passed to single DataObjects."""
        # return_val = source
        if not isinstance(source, list):
            raise ValueError(f"Cannot create Objects from {type(source)}")
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

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[DataObject, DataObjectStack]:
        elements = self._elements[index]
        if isinstance(index, int):
            # if virtual elements contains just sources not DataObjects
            if self.virtual:
                return self._single_construct(elements)
            return elements
        return type(self)(elements, virtual=self.virtual)

    def __setitem__(
        self, index: Union[int, slice], other: Union[DataObject, Iterable]
    ) -> None:
        # check compatibility -- implement in dataobject? (like img size)
        if not isinstance(other, (self._type, type(self))):
            raise ValueError(f"Unsupported type {type(other)} for {type(self)}")
        if isinstance(index, int) and isinstance(other, type(self)):
            raise ValueError(f"Cant set single Elements to {type(self)}")

        if self.virtual:
            if isinstance(other, self._type):
                if other.source is None:
                    raise ValueError(
                        "Can't put DataObjects without source into virtual stack"
                    )
                insert = other.source
            else:
                if None in other.sources:
                    raise ValueError(
                        "Can't put DataObjects without source into virtual stack"
                    )
                insert = other.sources

        else:
            insert = other
            if isinstance(other, type(self)):
                check = other[0]
            else:
                check = other

            if not self[0].is_compatible(check):
                raise ValueError("Incompatible element assignment")

        self._elements.__setitem__(index, insert)

    def extend(self, other: Union[DataObject, Iterable]) -> None:
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
                raise ValueError(
                    "Can't put DataObjects without source into virtual stack"
                )
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
        if attr in ("_meta", "_data") and attr in self.__dict__:
            return
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

    def __delitem__(self, index: Union[int, slice]) -> None:
        self._elements.__delitem__(index)

    def __len__(self) -> int:
        return len(self._elements)

    def __iadd__(
        self, other: Union[DataObjectStack, DataObject, Number]
    ) -> DataObjectStack:
        if isinstance(other, type(self)):
            self.extend(other)
        else:
            if self.virtual:
                raise ValueError("Can't do this '+' operation on virtual stacks.")
            for element in self._elements:
                element += other
        return self

    def __add__(
        self, other: Union[DataObjectStack, DataObject, Number]
    ) -> DataObjectStack:
        result = self.copy(virtual=False)
        result += other
        return result

    def __radd__(
        self, other: Union[DataObjectStack, DataObject, Number]
    ) -> DataObjectStack:
        return self.__add__(other)

    def __isub__(self, other: Union[DataObject, DataObject, Number]) -> DataObjectStack:
        if self.virtual:
            raise ValueError("Can't do '-' on virtual stacks.")

        for element in self._elements:
            element -= other

        return self

    def __sub__(
        self, other: Union[DataObjectStack, DataObject, Number]
    ) -> DataObjectStack:
        result = self.copy(virtual=False)
        result -= other
        return result

    def __imul__(self, other: Union[DataObject, Number]) -> DataObjectStack:
        if self.virtual:
            raise ValueError("Can't do '*' on virtual stacks.")
        for element in self._elements:
            element *= other
        return self

    def __mul__(self, other: Union[DataObject, Number]) -> DataObjectStack:
        result = self.copy(virtual=False)
        result *= other
        return result

    def __rmul__(self, other: Union[DataObject, Number]) -> DataObjectStack:
        return self.__mul__(other)

    def __itruediv__(self, other: Union[DataObject, Number]) -> DataObjectStack:
        if self.virtual:
            raise ValueError("Can't do '/' on virtual stacks.")
        for element in self._elements:
            element /= other
        return self

    def __truediv__(self, other: Union[DataObject, Number]) -> DataObjectStack:
        result = self.copy(virtual=False)
        result /= other
        return result

    # def _reduce(self) -> dict:
    #     return {"sources": self.sources, "virtual": self.virtual}

    # @classmethod
    # def _reconstruct(cls, state: dict) -> DataObjectStack:
    #     return cls(state["sources"], state["virtual"])

    def __json_encode__(self) -> dict:
        return {"source": self.elements, "virtual": self.virtual}

    def __json_decode__(self, **attrs) -> None:
        self.__init__(**attrs)


class Image(DataObject):
    """
    Base class for all spatial 2D data.
    """

    # pylint: disable=no-member
    _data_keys = ("image",)
    _meta_keys = ("width", "height")
    # TODO make a sensible default
    # default_mask = None

    def __init__(self, *args, **kwargs) -> None:
        self._mask = None
        super().__init__(*args, **kwargs)

    def parse(self, source: Union(str, np.ndarray)) -> dict[str, Any]:
        # self._source = source
        # return {"image": np.float32(imread(source))}

        if isinstance(source, np.ndarray):
            # if the object given already is a numpy array:
            image = source
            self._source = None
        elif source.lower().endswith((".tiff", ".tif")):
            image = tifffile.imread(source)
            self._source = source
        else:
            image = np.float32(imageio.imread(source))
            self._source = source

        if image.ndim != 2:
            raise ValueError(f"{source} is not a single image")

        return_val = {"image": image, "width": image.shape[1], "height": image.shape[0]}
        return return_val

    @property
    def area(self) -> Number:
        if isinstance(self._mask, np.ma.MaskedArray):
            return self._mask.count()
        else:
            return self._mask.size

    @property
    def mask(self) -> np.ndarray:
        """Set a mask onto the image. TODO"""
        return self._mask

    @mask.setter
    def mask(self, value: np.ndarray) -> None:
        pass

    def save(self, fname: str, **kwargs) -> None:
        if fname.lower().endswith((".tiff", ".tif")):
            tifffile.imwrite(fname, self.image, **kwargs)
        else:
            try:
                imageio.imwrite(fname, np.uint16(self.image))
            except ValueError:
                super().save(fname)

    def filter(self, method="gaussian", **kwargs):
        """Apply 2D Filter to image

        Applies a 2D Filter to the image, specified by the *method* argument. Valid methods are
        "gaussian", "blur", "median" and "kernel". Depending on the selected method different kwargs
        can be passed.


        Parameters
        ----------
        method : str, optional
            Applied filter type. Either "gaussian", "blur", "median" or "kernel", by default "gaussian".
        kwargs : Depending on the passed method

        Methods: possible kwargs are specified below

        - gaussian: Applies a gaussian filter to the image

            - size : int or tuple, by default (3, 3)
                size of the applied kernel
            - sigma: int, by default -1
                width of the applied gaussian. When negative, sigma is calculated from kernel

        - blur: Applies a normlized box filter with size of kernel

            - size: int or tuple, by default (3,3)
                size of applied kernel

        - media: Applies median filter

            - size: int, by default 3
                size of kernel

        - kernel: Convolutes image with kernel

            - kernel: np.ndarray, by default None
                kernel to be applied to the image

        Returns
        -------
        Image
            Returns image with applied filter.

        Raises
        ------
        ValueError
            If specified method is not valid.
        """

        img = self.copy()

        if method == "gaussian":
            sigma = kwargs.pop("sigma", -1)
            size = kwargs.pop("size", (3, 3))
            if isinstance(size, int):
                size = (size, size)
            img.image = cv.GaussianBlur(self.image, size, sigma, **kwargs)
        elif method == "blur":
            size = kwargs.pop("size", (3, 3))
            if isinstance(size, int):
                size = (size, size)
            img.image = cv.blur(self.image, size, *kwargs)
        elif method == "median":
            size = kwargs.pop("size", 3)
            img.image = cv.medianBlur(self.image, size)
        elif method == "kernel":
            kernel = kwargs.pop("kernel", None)
            img.image = cv.filter2D(self.image, ddepth=-1, kernel=kernel, *kwargs)
        else:
            raise ValueError("Unkown Filter")

        return img

    def contrast(self, method="equal"):
        img = self.copy()
        if method == "auto":
            pass

        if method == "equal":
            img.image = cv.equalizeHist(img.image.astype(np.uint8))
        elif method == "clahe":
            clahe = cv.createCLAHE()
            img.image = clahe.apply(img.image)

        return img

    def __iadd__(self, other: Union[Image, Number, np.ndarray]) -> Image:
        if isinstance(other, Image):
            self.image += other.image
        elif isinstance(other, (Number, np.ndarray)):
            self.image += other
        else:
            raise TypeError(
                f"Unsupported Operation '+' for types {type(self)} and {type(other)}"
            )
        return self

    def __add__(self, other: Union[Image, Number, np.ndarray]) -> Image:
        result = self.copy()
        result += other
        return result

    def __radd__(self, other: Union[Image, Number, np.ndarray]) -> Image:
        if other == 0:
            return self
        return self.__add__(other)

    def __isub__(self, other: Union[Image, Number, np.ndarray]) -> Image:
        if isinstance(other, Image):
            self.image -= other.image
        elif isinstance(other, (Number, np.ndarray)):
            self.image -= other
        else:
            raise TypeError(
                f"Unsupported Operation '-' for types {type(self)} and {type(other)}"
            )
        return self

    def __sub__(self, other: Union[Image, Number, np.ndarray]) -> Image:
        result = self.copy()
        result -= other
        return result

    def __imul__(self, other: Union[Image, Number, np.ndarray]) -> Image:
        if isinstance(other, Image):
            self.image *= other.image
        elif isinstance(other, (Number, np.ndarray)):
            self.image *= other
        else:
            raise TypeError(
                f"Unsupported Operation '*' for types {type(self)} and {type(other)}"
            )
        return self

    def __mul__(self, other: Union[Image, Number, np.ndarray]) -> Image:
        result = self.copy()
        result *= other
        return result

    def __rmul__(self, other: Union[Image, Number, np.ndarray]) -> Image:
        return self.__mul__(other)

    def __itruediv__(self, other: Union[Image, Number, np.ndarray]) -> Image:
        if isinstance(other, (Number, np.ndarray)):
            self.image /= other
        elif isinstance(other, Image):
            self.image /= other.image
        else:
            raise TypeError(
                f"Unsupported Operation '/' for types {type(self)} and {type(other)}"
            )
        return self

    def __truediv__(self, other: Union[Image, Number, np.ndarray]) -> Image:
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


class ImageStack(DataObjectStack):
    """DataObjectStack for Images which provides a default parsing mechanism for
    common image formats like png or tiff."""

    _type = Image

    def _split_source(self, source: Union[str, Iterable]) -> list:

        if isinstance(source, list):
            return super()._split_source(source)

        if isinstance(source, np.ndarray):
            # if the object given already is a numpy array:
            images = source
        else:
            try:
                images = tifffile.imread(source)
            except (
                FileNotFoundError,
                TypeError,
                TiffFileError,
            ):  # if not check if list of images
                return super()._split_source(source)

        if len(images.shape) != 3:
            raise ValueError(f"{source} is not a stack")

        return [images[i, :, :] for i in range(images.shape[0])]

    def average(self, avg: int) -> ImageStack:
        """Averages over consecutive images in a stack.

        Parameters
        ----------
        avg : int
            Number of consecutive image that will be averaged. The total number of images must be
            dividable by *avg*.

        Returns
        -------
        ImageStack
            Stack with averaged images. The metadata of the first element in every averaged slice of
            images is preserved.

        Raises
        ------
        ValueError
            If number of images in stack is not dividable by *avg*

        Examples
        --------
        >>> stack = ImageStack("testdata/test_stack_IV_RuO2_normed_aligned_80-140.tif")[0:60]
        >>> len(stack)
        60
        >>> avg_stack = stack.average(4)
        >>> len(avg_stack)
        15

        """
        if len(self) % avg != 0:
            raise ValueError(
                f"Number of Images in Stack ({len(self)}) is not dividable by {avg}"
            )

        imgs = []

        for i in range(0, len(self) // avg):
            img = np.mean(self[i * avg : i * avg + avg])
            imgs.append(img)

        return type(self)(imgs)

    def save(self, fname: str) -> None:
        if fname.lower().endswith((".tiff", ".tif")):
            array = np.stack([image.image for image in self])
            tifffile.imwrite(fname, array)
        else:
            super().save(fname)


class Line(DataObject):
    """
    Base class for all 1D data.

    Parameters
    ----------
    source : Union[str, np.ndarray]
        When a str: path to .csv file with x y data as columns. Delimiter is guessed from the file
        When np.ndarray: 2d numpy array containing the x and y values either als rows or cols

    Attributes
    ----------
    x, y : np.ndarray
        Numpy arrays containing the x and y data of the line
    ydim, xdmin : str
        Dimension of the x and y values, e.g. time, energy ...

    Examples
    --------
    Initialization from .csv file

    .. code-block:: none

       example.csv:
       U in V,  I in A
       0.0,     0.0
       1.0,     2.0

    >>> line = Line("example.csv")
    >>> line.x, line.y
    array([0.0, 1.0]), array([0.0, 2.0])

    When reading from a .csv file the column header is interpreted as *xdim* and *ydim*. If the
    header has the form *A in B*, *A* is the dimension and *B* is the corresponding unit.

    >>> line.xdim, line.ydim
    U, I

    The x and y values are also accessible over their corresponding dimension:

    >>> line.U
    array([0.0, 1.0])

    Initialization from numpy array

    >>> line = Line(np.array([[0.0, 1.0], [0.0, 2.0]]))
    >>> line.x
    array([0.0, 1.0])
    >>> line.xdim, line.ydim
    x, y

    """

    # pylint: disable=no-member
    _data_keys = ("x", "y")
    _meta_defaults = {
        "ydim": "y",
        "xdim": "x",
        "color": "k",
    }
    _unit_defaults = {"x": "a.u.", "y": "a.u."}

    def __init__(self, source: Union[str, np.ndarray]) -> None:
        super().__init__(source)

    def __getattr__(self, attr: str) -> Any:
        # _data and _meta have to be handled by super() to avoid recursion
        if attr in ("_data", "_meta"):
            return super().__getattr__(attr)
        if attr == self._meta["xdim"]:
            return self._data["x"]
        if attr == self._meta["ydim"]:
            return self._data["y"]

        return super().__getattr__(attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        # _data and _meta have to be handled by super() to avoid recursion
        if attr in ("_data", "_meta"):
            return super().__setattr__(attr, value)

        if attr == self._meta["xdim"]:
            self._data["x"] = value
        elif attr == self._meta["ydim"]:
            self._data["y"] = value
        else:
            return super().__setattr__(attr, value)

    def parse(self, source: Union[str, np.ndarray]) -> dict[str, Any]:
        if isinstance(source, np.ndarray):
            if source.ndim == 2:
                if source.shape[0] == 2:
                    source = source.T
                df = pd.DataFrame(source)
                self._source = None

            else:
                raise ValueError("Not a Line")
        elif isinstance(source, str):
            df = pd.read_csv(source)

            if len(df.columns) != 2:
                raise ValueError("Not a Line")

            self._source = source

        x = df.to_numpy()[:, 0]
        y = df.to_numpy()[:, 1]

        xdim, ydim = list(df.columns)
        x_unit, y_unit = self._unit_defaults["x"], self._unit_defaults["y"]
        if (xdim, ydim) == (0, 1) or (xdim, ydim) == ("0", "1"):
            xdim = self._meta_defaults["xdim"]
            ydim = self._meta_defaults["ydim"]
        else:
            try:
                xdim, x_unit = xdim.split(" in ")
                ydim, y_unit = ydim.split(" in ")
            except ValueError:
                pass

        return {
            "x": x,
            "y": y,
            "xdim": xdim,
            "ydim": ydim,
            "x_unit": x_unit,
            "y_unit": y_unit,
        }

    @property
    def length(self) -> int:
        """Length of the x and y data."""
        return len(self.x)

    @property
    def area(self) -> float:
        "Area under the line integrated by simpson rule"
        return scipy.integrate.simpson(self.y, self.x)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Pandas Dataframe of the lines x,y data"""
        return pd.DataFrame(
            data=np.stack([self.x, self.y], axis=1),
            columns=[
                f"{self.xdim} in {self._units['x']}",
                f"{self.ydim} in {self._units['y']}",
            ],
        )

    def interpolate(
        self,
        x_data: Union[Number, list[Number], np.ndarray, None] = None,
        order: Union[str, int] = "cubic",
        **kwargs,
    ) -> Union[np.ndarray, Callable]:

        """
        Interpolates the x,y data of the Line with a spline of order 'order'

        Parameters
        ----------
        x_data: Number, list of numbers or 1d numpy array or None
            x_values at which the interpolation will be evaluated
        order: str or int
            When a string must be either "linear", "quad" or "cubic".
            When an int must be 1 <= order <= 5
        kwargs:
            Additional keyword arguments are passed through to
            scipy.interpolate.InterpolatedUnivariateSpline

        Returns
        -------
        if x_data was not None:
            numpy array containing interpolated Numbers
        if x_data was None:
            the interpolation function itself
        """

        k = {"linear": 1, "quad": 2, "cubic": 3}
        if order not in k.keys():
            k[order] = order

        func = scipy.interpolate.InterpolatedUnivariateSpline(
            self.x, self.y, k=k[order], **kwargs
        )
        if x_data is None:
            return func
        return func(x_data)

    def integral(self) -> Line:
        """Returns the integrated values of a cubic spline evaluated at the x values of the line"""
        func = self.interpolate()
        y = np.array([func.integral(self.x[0], x) for x in self.x])
        return Line(np.array([self.x, y]))

    def derivative(self) -> Line:
        """Returns the derivative of a cubic spline evaluated at the x values of the line"""
        func = self.interpolate()
        y = np.array([func.derivative()(x) for x in self.x[1:]])
        return Line(np.array([self.x[1:], y]))

    def smooth(self, kernel: Union[int, np.ndarray]) -> Line:
        """
        Recieves a kernel and convolves the values of the line

        Parameters
        ----------
        kernel : np.ndarray, int
            Kernel that will be convolved with the line data.
            If ``int`` an equaly distributed kernel of size ``kernel`` will be applied.

        Returns
        -------
        Line
            smoothed line

        Examples
        --------
        Rolling average over three data points of line. Note that sum(kernel) should be 1.

        >>> kernel = np.array([1,1,1])/3
        >>> smoothed = line.smooth(kernel)

        Above code is equivalent to:

        >>> smoothed = line.smooth(3)

        The weights do not have to be the same. To put more weigh on the current point than on
        adjecent points:

        >>> kernel = np.array([1,2,1])/4
        >>> smoothed = line.smooth(kernel)
        """

        if isinstance(kernel, int):
            kernel = np.ones(kernel) / kernel

        line = self.copy()
        # pylint: disable=attribute-defined-outside-init
        line.x, line.y = self.x, np.convolve(self.y, kernel, mode="same")
        return line

    def save(self, fname: str) -> Line:
        """
        Saves the line as .csv

        Parameters
        ----------
        fname: str
            Filename of the file to save.
            If it ends with .csv a csv file of the dataframe representation of the line is saved
            Otherwise the parent(dataobject) save method is called.

        Returns
        -------
            None
        """
        if fname.lower().endswith(".csv") or fname.lower().endswith(".txt"):
            self.dataframe.to_csv(fname, index=False, header=True)
        else:
            super().save(fname)

        return self

    def is_compatible(self, other: Line) -> bool:
        try:
            assert super().is_compatible(other)
            assert self.length == other.length
            return True
        except AssertionError:
            return False

    def __iadd__(self, other: Union[Line, Number, np.ndarray]) -> Line:
        if isinstance(other, Line):
            self.y += other.y
        elif isinstance(other, (Number, np.ndarray)):
            self.y += other
        else:
            raise TypeError(
                f"Unsupported Operation '+' for types {type(self)} and {type(other)}"
            )

        return self

    def __isub__(self, other: Union[Line, Number, np.ndarray]) -> Line:
        if isinstance(other, Line):
            self.y -= other.y
        elif isinstance(other, (Number, np.ndarray)):
            self.y -= other
        else:
            raise TypeError(
                f"Unsupported Operation '-' for types {type(self)} and {type(other)}"
            )

        return self

    def __imul__(self, other: Union[Line, Number, np.ndarray]) -> Line:
        if isinstance(other, Line):
            self.y *= other.y
        elif isinstance(other, (Number, np.ndarray)):
            self.y *= other
        else:
            raise TypeError(
                f"Unsupported Operation '*' for types {type(self)} and {type(other)}"
            )

        return self

    def __itruediv__(self, other: Union[Line, Number, np.ndarray]) -> Line:
        if isinstance(other, Line):
            self.y /= other.y
        elif isinstance(other, (Number, np.ndarray)):
            self.y /= other
        else:
            raise TypeError(
                f"Unsupported Operation '/' for types {type(self)} and {type(other)}"
            )

        return self

    def __add__(self, other: Union[Line, Number, np.ndarray]) -> Line:
        result = self.copy()
        result += other
        return result

    def __radd__(self, other: Union[Line, Number, np.ndarray]) -> Line:
        if other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other: Union[Line, Number, np.ndarray]) -> Line:
        result = self.copy()
        result -= other
        return result

    def __mul__(self, other: Union[Line, Number, np.ndarray]) -> Line:
        result = self.copy()
        result *= other
        return result

    def __rmul__(self, other: Union[Line, Number, np.ndarray]) -> Line:
        return self.__mul__(other)

    def __truediv__(self, other: Union[Line, Number, np.ndarray]) -> Line:
        result = self.copy()
        result /= other
        return result


class IntensityLine(Line):
    """
    A Class to extract Intensities from an ImageStack

    Unlike Line from which it inherits it takes an ImagseStack, a ROI and a String representing the
    dimension along which the itensities in the stack should be extracted, e.g. time, energy ...
    For each image the the mean value of intensites inside the ROI is extracted as y-values together
    with the dimensional value of the image (time, energy, ...) as the lines x-axes.
    """

    def __init__(self, stack: ImageStack, roi_: roi.ROI, xaxis: str) -> None:
        """
        Parameters
        ----------
        stack : ImageStack
            ImageStack from which the intensities will be extracted
        roi_ : roi.ROI
            ROI from within the intensities will be extracted in each image,
        xaxis : str
            a string indicates the axis of the dimension along the intensites shall be extracted,
            e.g. time, energy ...
        """
        super().__init__((stack, roi_, xaxis))

    def parse(self, source: tuple[ImageStack, roi.ROI, str]) -> dict[str, Any]:
        """Applies a ROI to an Imagestack and extracts mean of ROI from each image

        Parameters
        ----------
        source : tuple[ImageStack, roi.ROI, str]
            Must be a Tuple of

            1. an ImageStack from which the intensities will be extracted,
            2. a ROI from within the intensities will be extracted in each image,
            3. a string indicates the axis of the dimension along the intensites shall be \
                extracted, e.g. time, energy ...

        Returns
        -------
        dict
            a dict containing the x,y-values of the line, the stack, the ROI, the x- and
            y-dimensions and the unit of the x axis extracted from the images. The y-dimension is
            "intensity" by default with unit "a.u."
        """

        stack, roi_, xaxis = source
        self._source = source
        y = []
        x = []
        for img in stack:
            masked_img = roi_.apply(img)
            y.append(np.mean(masked_img.image))
            x.append(getattr(img, xaxis))

        return {
            "x": np.array(x),
            "y": np.array(y),
            "stack": stack,
            "roi": roi_,
            "xdim": xaxis,
            "ydim": "intensity",
            "x_unit": stack[0].unit[xaxis],
        }


class StitchedLine(Line):
    """
    Class for combining IntensityLines from multiple ImageStacks.

    Works like IntensityLine, but accepts multiple ImageStacks and the same amount of ROIs. Every
    stack is mapped to a ROI (in the order that they are given, e.g. first stack to first ROI) and
    the intensity profile is extracted. If the x-values of the extracted Lines are overlapping, the
    Lines are scaled to matched each other. The stacks may be sorted by their first xaxis value
    before extraction.

    Attributes
    ----------
    x, y : np.ndarray
        x and y values of stitched lines
    lines : list[IntensityLine]
        list of single IntensityLines that are used for stitching
    xdim, ydim : str
        string representing the dimension of the x and y values, default for y is "intensity"
    """

    def __init__(
        self,
        stacks: Union[Sequence[ImageStack], ImageStack],
        rois: Union[Sequence[roi.ROI], roi.ROI],
        xaxis: str,
    ) -> None:
        """
        Parameters
        ----------
        stacks : Sequence[ImageStack] or ImageStack
            ImageStacks, which will be used for Line extraction
        rois : Sequence[roi.ROI] or roi.ROI
            ROIs used for Line extraction. Must be of the same length as ``stacks``. If stacks is a
            single stack, ROI has to be a single ROI
        xaxis:
            a string indicates the axis of the dimension along the intensites shall be extracted,
            e.g. time, energy ...

        Raises
        ------
        ValueError
            When len(stacks) != len(rois)
        """
        if isinstance(stacks, ImageStack):
            stacks = [stacks]
            rois = [rois]

        super().__init__((stacks, rois, xaxis))

    def parse(self, source: Sequence[Sequence, Sequence, str]) -> dict[str, Any]:
        """Extract IntensityLines from Stacks and handle their stitching.

        Parameters
        ----------
        source : Sequence[Sequence, Sequence, str]
            List or tuple containing

            1. A list or tuple of ImageStacks
            2. A list or tuple of ROIs with the same length as the tuple of ImageStacks
            3. A str representing the xaxis along which the intensity is extracted

        Returns
        -------
        dict[str, Any]
            dict containing all metadata and data, including rois and stacks

        Raises
        ------
        ValueError
            When len(stacks) != len(rois)
        """

        stacks, rois, xaxis = source

        if len(stacks) != len(rois):
            raise ValueError(f"Cannot map {len(rois)} to {len(stacks)}")

        lines = [
            IntensityLine(stack, roi, xaxis)
            for stack, roi in sorted(
                zip(stacks, rois), key=lambda x: getattr(x[0][0], xaxis)
            )
        ]

        if len(lines) == 1:
            x, y = lines[0].x, lines[0].y
        else:
            x, y = self._stitch_curves(lines)

        self._source = source

        return {
            "x": x,
            "y": y,
            "lines": lines,
            "xdim": xaxis,
            "ydim": "intensity",
            "x_unit": stacks[0][0].unit[xaxis],
        }

    def _stitch_curves(
        self, lines: Sequence[IntensityLine]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine x,y values to mulitple lines to one pair of x,y values

        Parameters
        ----------
        lines : Sequence[IntensityLine]
            List or tuple of Lines to be stitched together

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            tuple containing the x and y values of the stitched curve
        """
        coeffs = [1]

        for curr_line, next_line in zip(lines, lines[1:]):
            # calculate overlap in x values between lines, assuming ordered values:

            start_x = next_line.x[0]
            end_x = curr_line.x[-1]

            if start_x > end_x:
                print("No Overlap detected.")
                coeffs.append(1)
            elif start_x == end_x:
                print("One Point overlap")
                coeffs.append(next_line.y[0] / curr_line.y[-1])
            else:
                curr_line_points = np.array(
                    [
                        (x, y)
                        for x, y in zip(curr_line.x, curr_line.y)
                        if start_x <= x <= end_x
                    ]
                )
                next_spline = next_line.interpolate(order="cubic")

                # pylint: disable=cell-var-from-loop
                func = lambda x, a: a * next_spline(x)
                # pylint: disable=unbalanced-tuple-unpacking
                popt, _ = scipy.optimize.curve_fit(
                    func, curr_line_points[:, 0], curr_line_points[:, 1]
                )

                coeffs.append(popt)

        x_data = np.array([])
        y_data = np.array([])

        for coeff, line in zip(coeffs, lines):
            x_data = np.append(x_data, line.x)
            y_data = np.append(y_data, coeff * line.y)

        # sort values and average over same x values
        data = np.array([x_data, y_data]).T
        df = pd.DataFrame(data).groupby(0).mean()

        x = df.index.values
        y = df.values.flatten()

        return x, y
