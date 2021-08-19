"""
Basic data containers.
"""
# pylint: disable=abstract-method
from __future__ import annotations
import bz2
from typing import Any, Union, Optional
from collections.abc import Iterable
from numbers import Number
import copy
import pickle
from pathlib import Path
import tifffile
import imageio
import pandas as pd


from deepdiff import DeepDiff
import numpy as np
from tifffile.tifffile import TiffFileError


class Loadable:
    """
    Base class for loadable objects. Contains methods for serializing
    and deserializing.
    """
    _pickle_extension = ".pickle"

    def dump(self, fname: str, thin: bool = False) -> None:
        """Dumps the object into a JSON/pickle file."""
        with Path(fname).open("wb") as pfile:
            pfile.write(self.dumps(thin))

    def dumps(self, thin: bool = False) -> None:
        """Dumps the object with pickle and returns it as a bytestring"""
        if thin:
            obj = ThinObject(self)
        else:
            obj = self
        try:
            return pickle.dumps(obj, protocol=4)
        except RecursionError:
            print("Did not pickle due to recursion error.")
            raise

    def save(self, fname: str) -> None:
        """Save the DataObject to a file depending on the filename extension."""
        _fname = fname
        compress = thin = False
        if _fname.endswith(".bz2"):
            _fname = _fname.removesuffix(".bz2")
            compress = True
        if _fname.endswith((".thin" + self._pickle_extension, ".thin")):
            thin = True
        if not _fname.endswith(self._pickle_extension):
            _fname+=self._pickle_extension
            if compress:
                _fname += ".bz2"
            fname = _fname

        if compress:
            print("Compressing data...")
            with bz2.BZ2File(fname, 'w') as bzfile:
                bzfile.write(self.dumps(thin))
            return

        self.dump(fname, thin)

    def _reduce(self) -> dict:
        raise NotImplementedError

    @classmethod
    def _reconstruct(cls, state) -> Loadable:
        raise NotImplementedError

    @classmethod
    def load(cls, fname: str) -> Loadable:
        """Returns an object retrieved from a JSON/pickle file."""

        if fname.endswith(".bz2"):
            print("Uncompressing data...")
            file = bz2.BZ2File(fname, 'rb')
            obj = pickle.load(file)
        else:
            with Path(fname).open("rb") as pfile:
                obj = pickle.load(pfile)

        #if isinstance(obj, Loadable.ThinObject):
        #    obj = obj.reconstruct()
        return obj

class ThinObject:
    """
    Inner class which is a thin version of the full Loadable. Meant for
    lightweight data saving.
    """
    def __init__(self, obj: Loadable) -> None:
        self._obj = obj
        self._type = type(obj)

    def __getstate__(self) -> dict:
        state = self._obj._reduce()
        for _, val in state.items():
            # Check if something in state can be casted to Thinobject
            #if isinstance(val, Loadable):
            try:
                val = ThinObject(val)
            # Is Loadable but has not _reduce function or is not Loadable
            except (NotImplementedError, AttributeError):
                # See if it is a list of something that can be made thin
                try:
                    for item in val:
                        item = ThinObject(item)
                except (TypeError, AttributeError, NotImplementedError):
                    pass # do val stays val...

            state["_type"] = self._type
            #state.update(self._obj._reduce())
            return state

    def __setstate__(self, state: dict) -> None:
        obj = state["_type"]._reconstruct(state)
        # cast ThinObject into the Object that it was originally
        self.__class__ = obj.__class__
        self.__dict__ = obj.__dict__

class DataObject(Loadable):
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

    def _reduce(self) -> dict:
        return {"source": self._source}

    @classmethod
    def _reconstruct(cls, state: dict) -> DataObject:
        return cls(state["source"])


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

            self._elements = self._split_source(source)
        else:
            self._elements = self._split_source(source)
            if not self.virtual:
                self._construct()

    def _split_source(self, source: Union[str, Iterable]) -> list:
        """Split the source parameter of the constructor into source arguments
        that can be passed to single DataObjects."""
        return_val = source
        if not isinstance(return_val, list):
            raise ValueError(f"Cannot create Objects from {type(self)}")
        return source

    def _construct(self) -> None:
        """Build the stack from a list of sources."""
        self._virtual = False
        sources = self._elements
        if len(sources) == 0: # sources might be an arbitrary iterable, so length is tested
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
        if not isinstance(other, (self._type, type(self))):
            raise ValueError(f"Unsupported type {type(other)} for {type(self)}")
        if isinstance(index, int) and isinstance(other, type(self)):
            raise ValueError(f"Cant set single Elements to {type(self)}")

        if self.virtual:
            if isinstance(other, self._type):
                if other.source is None:
                    raise ValueError("Can't put DataObjects without source into virtual stack")
                insert = other.source
            else:
                if None in other.sources:
                    raise ValueError("Can't put DataObjects without source into virtual stack")
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

    def _reduce(self) -> dict:
        return {"sources": self.sources,
                "virtual": self.virtual}

    @classmethod
    def _reconstruct(cls, state: dict) -> DataObjectStack:
        return cls(state["sources"], state["virtual"])


class Image(DataObject):
    """
    Base class for all spatial 2D data.
    """
    # pylint: disable=no-member
    _data_keys = ("image",)
    _meta_keys = ("width", "height")

    def __init__(self, *args, **kwargs) -> None:
        self._mask = None
        super().__init__(*args, **kwargs)

    def parse(self, source: Union(str, np.ndarray)) -> dict[str, Any]:
        #self._source = source
        #return {"image": np.float32(imread(source))}

        if isinstance(source, np.ndarray):
            # if the object given already is a numpy array:
            image = source
            self._source = None
        elif source.lower().endswith(('.tiff', '.tif')):
            image = tifffile.imread(source)
            self._source = source
        else:
            image = np.float32(imageio.imread(source))
            self._source = source

        if image.ndim != 2:
            raise ValueError(f"{source} is not a single image")

        return_val = {
            "image": image,
            "width": image.shape[0],
            "height": image.shape[1]
        }
        return return_val

    @property
    def mask(self) -> np.ndarray:
        """Set a mask onto the image. TODO"""
        return self._mask
    @mask.setter
    def mask(self, value: np.ndarray) -> None:
        pass

    def save(self, fname: str, **kwargs) -> None:
        if fname.lower().endswith((".tiff",".tif")):
            tifffile.imwrite(fname, self.image, **kwargs)
        else:
            try:
                imageio.imwrite(fname, np.uint16(self.image))
            except ValueError:
                super().save(fname)

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

class ImageStack(DataObjectStack):
    """DataObjectStack for Images which provides a default parsing mechanism for
    common image formats like png or tiff."""
    _type = Image

    def _split_source(self, source: Union[str, Iterable]) -> list:

        if isinstance(source, np.ndarray):
            # if the object given already is a numpy array:
            images = source
        else:
            try:
                images = tifffile.imread(source)
            except (FileNotFoundError, TypeError, TiffFileError): # if not check if list of images
                return super()._split_source(source)

        if len(images.shape) != 3:
            raise ValueError(f"{source} is not a stack")

        return [images[i, :, :] for i in range(images.shape[0])]

    def save(self, fname: str) -> None:
        if fname.lower().endswith((".tiff",".tif")):
            array = np.stack([image.image for image in self])
            tifffile.imwrite(fname, array)
        else:
            super().save(fname)

class Line(DataObject):
    """
    Base class for all 1D data.
    """
    # pylint: disable=no-member
    _data_keys = ("table",)
    # _meta_keys = ("npoints","xdim","ydim","xunit","yunit")
    _meta_defaults = {
        "ydim": "y",
        "xdim": "x",
    }
    _unit_defaults = {
        "x": "a.u.",
        "y": "a.u."
    }

    #def __init__(self, *args, **kwargs) -> None:
    #    super().__init__(*args, **kwargs)

    def parse(self, source: Union(str, np.ndarray)) -> dict[str, Any]:

        if isinstance(source, np.ndarray):
            if source.ndim == 2:
                if source.shape[0] == 2:
                    #x = source[0,:]
                    #y = source[1,:]
                    #df = pd.DataFrame(source.T)
                    source = source.T
                #else:
                    #x = source[:,0]
                    #y = source[:,1]
                df = pd.DataFrame(
                    source,
                    columns=[self._meta_defaults["xdim"], self._meta_defaults["ydim"]]
                )
            else:
                raise ValueError("Not a Line")

        elif isinstance(source, str):
            #data = np.genfromtxt(source)
            df = pd.read_csv(source)

            if len(df.columns) != 2:
                raise ValueError("Not a Line")

            #if "0" in df and "1" in df:

            df = df.rename(
                columns={"0":self._meta_defaults["xdim"], "1":self._meta_defaults["ydim"]}
            )


            #x = data[:, 0]
            #y = data[:, 1]

        #assert len(x) == len(y)

        return {"table": df, "npoints": len(df)}

    @property
    def x(self) -> np.ndarray:
        """Raw x vector."""
        return self.table.iloc[:,0].to_numpy()

    @property
    def y(self) -> np.ndarray:
        """Raw y vector."""
        return self.table.iloc[:,1].to_numpy()

    @property
    def length(self) -> int:
        """Length of the x and y data."""
        return len(self.x)

    def save(self, fname: str) -> None:
        if fname.lower().endswith(".csv"):
            #c = np.stack([self.x,self.y],axis=1)
            #np.savetxt(fname, c)
            self.table.to_csv(fname, index=False, header=True)

        else:
            super().save(fname)
