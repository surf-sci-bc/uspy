"""
Basic data containers.
"""

# pylint: disable=abstract-method
from __future__ import annotations
from typing import Any, Callable, Union, Optional
from collections.abc import Iterable
from numbers import Number
import copy
import operator
import warnings

from deepdiff import DeepDiff
import numpy as np
from tqdm import tqdm

from uspy.serialize import Loadable

__all__ = ["Loadable", "DataObject", "DataObjectStack"]


class _ArithmeticMixin:
    """Mixin providing elementwise +, -, *, / for DataObject subclasses that hold a single
    numeric array. Subclasses set ``_op_attr`` to that array's attribute name (e.g. "image"
    for Image, "y" for Line). Accepts a same-shaped instance duck-typed on ``_op_attr``, a
    plain Number, or an ndarray; in-place variants mutate the array in place."""

    _op_attr: Optional[str] = None

    def _operand(self, other: Any) -> Any:
        if isinstance(other, (Number, np.ndarray)):
            return other
        operand = getattr(other, self._op_attr, None)
        if operand is None:
            raise TypeError(
                f"Unsupported Operation for types {type(self)} and {type(other)}"
            )
        return operand

    def _iop(self, other: Any, op: Callable[[Any, Any], Any]) -> _ArithmeticMixin:
        arr = getattr(self, self._op_attr)
        arr = op(arr, self._operand(other))  # in-place (numpy __iadd__ etc.), same object
        setattr(self, self._op_attr, arr)
        return self

    def __iadd__(self, other: Any) -> _ArithmeticMixin:
        return self._iop(other, operator.iadd)

    def __add__(self, other: Any) -> _ArithmeticMixin:
        result = self.copy()
        result += other
        return result

    def __radd__(self, other: Any) -> _ArithmeticMixin:
        if other == 0:
            return self
        return self.__add__(other)

    def __isub__(self, other: Any) -> _ArithmeticMixin:
        return self._iop(other, operator.isub)

    def __sub__(self, other: Any) -> _ArithmeticMixin:
        result = self.copy()
        result -= other
        return result

    def __imul__(self, other: Any) -> _ArithmeticMixin:
        return self._iop(other, operator.imul)

    def __mul__(self, other: Any) -> _ArithmeticMixin:
        result = self.copy()
        result *= other
        return result

    def __rmul__(self, other: Any) -> _ArithmeticMixin:
        return self.__mul__(other)

    def __itruediv__(self, other: Any) -> _ArithmeticMixin:
        return self._iop(other, operator.itruediv)

    def __truediv__(self, other: Any) -> _ArithmeticMixin:
        result = self.copy()
        result /= other
        return result


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
    _setters = ()  # list of attributes that have setters, so __setattr__ can handle the correct

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
            # Gap-fill this instance's _units, not the shared class-level _unit_defaults
            # dict, which would pollute every sibling subclass that inherits it.
            self._units.setdefault(k, "")

    def parse(self, source: str) -> dict[str, Any]:
        """Read in data. Should set the _source attribute."""
        raise NotImplementedError

    def copy(self) -> DataObject:
        """Return a deep copy."""
        return copy.deepcopy(self)

    def _with_data(self, **data: Any) -> DataObject:
        """Return a new instance of the same type with the given data array(s) substituted
        for the current ones, without deep-copying any array (unlike `copy()`). Meant for
        methods computing a brand-new array from scratch (e.g. `Image.filter`) where a plain
        `self.copy()` would deep-copy the original array only to immediately discard it.

        Weaker independence contract than `copy()`: `_data`/`_meta`/`_units` are independent
        of `self` afterwards, but everything else (`_source`, `LEEMImg._reader`/
        `_time_origin`/`default_mask`, ...) is shared by reference, not deep-copied.
        """
        new = copy.copy(self)  # shares _data/_meta/_units/other attrs by reference
        # object.__setattr__ bypasses __setattr__'s no-op-on-existing-dict guard, needed
        # here to give `new` its own independent dicts instead of sharing `self`'s.
        object.__setattr__(new, "_data", dict(self._data, **data))
        object.__setattr__(new, "_meta", copy.deepcopy(self._meta))
        object.__setattr__(new, "_units", copy.deepcopy(self._units))
        return new

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
        if isinstance(value, Number) and np.isnan(value):
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
        elif isinstance(getattr(type(self), attr, None), property):
            # A class-level property (e.g. LEEMImg.time_origin) must run its own setter (or
            # raise, if read-only) instead of being silently swallowed into _meta below.
            super().__setattr__(attr, value)
        elif attr in self._data:
            self._data[attr] = value
        else:
            self._meta[attr] = value
            # Arbitrary metadata must always have a unit entry, or plotting/derivation code
            # indexing `_units` by field name would KeyError; setdefault never clobbers a
            # unit set earlier via set_unit() or a reader.
            self._units.setdefault(attr, "")

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

    def incompatibility_reason(self, other: DataObject) -> str:
        """Human-readable reason `other` failed `is_compatible(other)`, used in error
        messages when a stack can't be built from mismatched elements. Subclasses with a
        more specific `is_compatible` (e.g. `Image` checking pixel shape) should override
        this to explain what differs."""
        if not isinstance(other, type(self)):
            return f"{type(other).__name__} is not a {type(self).__name__}"
        return "not compatible"

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
            unit = self._units.get(k, "")
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
        # An empty list is a deliberate, unsupported request to build a stack out of
        # nothing; checked before the isinstance/source[0] probe below, which would
        # otherwise raise a bare IndexError for `source == []`.
        if isinstance(source, list) and not source:
            raise ValueError(
                f"Cannot create an empty {type(self).__name__}: a stack is a container for "
                "one or more objects. (wrong slice or filter?)"
            )
        # str/bytes are excluded here so `source[0]` is never evaluated on a string. Both empty and
        # non-empty string sources still reach _split_source below for the real "no files
        # found" error.
        if (
            not isinstance(source, (str, bytes))
            and isinstance(source, Iterable)
            and isinstance(source[0], self._type)
        ):
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
        for source in tqdm(sources[1:]):
            element = self._single_construct(source)
            if not self._elements[0].is_compatible(element):
                reason = self._elements[0].incompatibility_reason(element)
                raise ValueError(
                    f"Cannot build {type(self).__name__}: {element.source!r} is not "
                    f"compatible with {self._elements[0].source!r} ({reason})."
                )
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

    def set_unit(self, attr: str, unit: str) -> None:
        """Set the unit string for a per-element metadata field on every element, without
        touching the values themselves. Used e.g. by `LEEMReader.post_process_stack`
        implementations that distribute a shared per-frame value (and its unit) across a
        stack -- overridden by `LazyImageStack` to stay pixel-I/O-free."""
        if self.virtual:
            # Iterating a virtual stack yields freshly-constructed, throwaway elements (see
            # __getitem__) -- writing to their _units would silently vanish instead of
            # persisting anywhere, same failure mode __setattr__ already guards against.
            raise ValueError(
                f"Can't set unit for '{attr}' for virtual stack: elements are re-parsed "
                "from their source on every access, so a per-element write would be lost "
                "immediately. Set `stack.virtual = False` first."
            )
        for element in self:
            element._units[attr] = unit  # pylint: disable=protected-access

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
            else:
                elements = [element.source for element in other]

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
        if isinstance(getattr(type(self), attr, None), property):
            # A stack-level property (e.g. `virtual`, `LEEMStack.time_origin`) must run its
            # own setter (or raise, if read-only) rather than being distributed to elements.
            return super().__setattr__(attr, value)
        # A stack has no metadata of its own: a scalar (incl. plain string) is set on every
        # element; a sequence matching the stack's length is distributed elementwise; any
        # other length attaches to the stack itself, unless `attr` names a real per-element
        # field -- that would silently shadow this class's own __getattr__ gather, so it's
        # refused instead.
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            value = list(value)
            is_sequence = True
        else:
            is_sequence = False

        if is_sequence and len(value) == len(self):
            if self.virtual:
                raise ValueError(f"Can't set attribute '{attr}' for virtual stack")
            for obj, single_value in zip(self, value):
                setattr(obj, attr, single_value)
        elif is_sequence:
            if not len(self):
                # No element to check `attr` against for a per-element-field collision, so
                # it can't be safely attached to the stack either.
                raise ValueError(
                    f"Can't assign a length-{len(value)} iterable to attribute '{attr}' on an "
                    "empty stack: there are no elements to distribute it over or check for a "
                    "collision against. (wrong slice or filter?)"
                )
            collides = hasattr(self[0], attr)
            if collides:
                raise ValueError(
                    f"'{attr}' is a per-element attribute; can't overwrite it with a "
                    "mismatched-length iterable."
                )
            print(
                f"Attribute {attr} with length {len(value)} cannot be assigned "
                f"elementwise to stack with length {len(self)}. It will be assigned "
                "to the stack itself."
            )
            return super().__setattr__(attr, value)
        else:
            # Scalar: always metadata on every element.
            if self.virtual:
                raise ValueError(
                    f"Can't set attribute '{attr}' for virtual stack: elements are "
                    "re-parsed from their source on every access, so a per-element write "
                    "would be lost immediately. Set `stack.virtual = False` first."
                )
            if not len(self):
                raise ValueError(
                    f"Can't set attribute '{attr}' on an empty stack: there are no "
                    "elements to set it on."
                )
            for obj in self:
                setattr(obj, attr, value)

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

    def __json_encode__(self) -> dict:
        return {"source": self.elements, "virtual": self.virtual}

    def __json_decode__(self, **attrs) -> None:
        self.__init__(**attrs)
