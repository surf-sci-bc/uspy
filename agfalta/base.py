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

from deepdiff import DeepDiff
import numpy as np



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
        """Immutable data container."""
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
        if attr.startswith("_"):
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
        raise NotImplementedError


class DataObjectContainer(Loadable):
    """Contains multiple DataObjects. E.g., for an image stack."""
    _unique_attrs = ()
    _type = DataObject

    def __init__(self, source: Union[str,Iterable], virtual: bool = False) -> None:
        if virtual and isinstance(source, Iterable) and isinstance(source[0], DataObject):
            print("WARNING: Container won't be virtual (data objects were directly given)")
            virtual = False
        self._virtual = virtual
        self._elements = self._split_source(source)
        if not self.virtual:
            self._construct()

    def _split_source(self, source: Union[str, Iterable]) -> list:
        """Split the source parameter of the constructor into source arguments
        that can be passed to single DataObjects."""
        raise NotImplementedError

    def _construct(self) -> None:
        """Build the container from a list of sources."""
        self._virtual = False
        sources = self._elements
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
            raise ValueError("Can't virtualize container without source information.")

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
        """If a data container is virtual, it does only load the data on demand."""
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

    def __getitem__(self, index: Union[int,slice]) -> Union[DataObject,Iterable]:
        elements = self._elements.__getitem__(index)
        if isinstance(elements, Iterable):
            return type(self)(elements, virtual=False)
        if isinstance(elements, DataObject):
            return elements
        return self._single_construct(elements)

    def __setitem__(self, index: Union[int,slice], other: Union[DataObject,Iterable]) -> None:
        # check compatibility -- implement in dataobject? (like img size)
        if isinstance(other, DataObject):
            elements = [other]
        elif isinstance(other, type(self)):
            if not self.virtual:
                elements = other[:]
            elif other.virtual:
                elements = other.sources

        if self.virtual:
            if None in elements:
                raise ValueError("Can't put DataObjects without source into virtual container")
        else:
            for element in elements:
                if not self._elements[0].is_compatible(element):
                    raise ValueError("Incompatible element assignment")
        self._elements.__setitem__(index, elements)

    def extend(self, other: Union[DataObject,Iterable]) -> None:
        """Add new DataObjects to the end of the container."""
        if isinstance(other, DataObject):
            elements = [other]
        if isinstance(elements, type(self)):
            if not self.virtual:
                elements = other[:]
            elif other.virtual:
                elements = other.sources

        if self.virtual:
            if None in elements:
                raise ValueError("Can't put DataObjects without source into virtual container")
        else:
            for element in elements:
                if not self._elements[0].is_compatible(element):
                    raise ValueError("Incompatible element assignment")
        self._elements.extend(elements)

    def __delitem__(self, index: Union[int,slice]) -> None:
        self._elements.__delitem__(index)

    def __len__(self) -> int:
        return len(self._elements)

    def __iadd__(self, other: Union[DataObjectContainer,DataObject,Number]) -> DataObjectContainer:
        if isinstance(other, type(self)):
            self.extend(other)
        else:
            if self.virtual:
                raise ValueError("Can't do this '+' operation on virtual stacks.")
            for element in self._elements:
                element += other
        return self
    def __add__(self, other: Union[DataObjectContainer,DataObject,Number]) -> DataObjectContainer:
        result = self.copy(virtual=False)
        result += other
        return result
    def __radd__(self, other: Union[DataObjectContainer,DataObject,Number]) -> DataObjectContainer:
        return self.__add__(other)

    def __isub__(self, other: Union[DataObject,DataObject,Number]) -> DataObjectContainer:
        if self.virtual:
            raise ValueError("Can't do '-' on virtual stacks.")
        if isinstance(other, type(self)):
            self._elements -= other.elements
        else:
            for element in self._elements:
                element -= other
        return self
    def __sub__(self, other: Union[DataObjectContainer,DataObject,Number]) -> DataObjectContainer:
        result = self.copy(virtual=False)
        result -= other
        return result

    def __imul__(self, other: Union[DataObject,Number]) -> DataObjectContainer:
        if self.virtual:
            raise ValueError("Can't do '*' on virtual stacks.")
        for element in self._elements:
            element *= other
        return self
    def __mul__(self, other: Union[DataObject,Number]) -> DataObjectContainer:
        result = self.copy(virtual=False)
        result *= other
        return result
    def __rmul__(self, other: Union[DataObject,Number]) -> DataObjectContainer:
        return self.__mul__(other)

    def __idiv__(self, other: Union[DataObject,Number]) -> DataObjectContainer:
        if self.virtual:
            raise ValueError("Can't do '/' on virtual stacks.")
        for element in self._elements:
            element /= other
        return self
    def __div__(self, other: Union[DataObject,Number]) -> DataObjectContainer:
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
        try:
            return self.image.shape == other.image.shape
        except AttributeError:
            return False
