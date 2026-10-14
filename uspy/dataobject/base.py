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
from uspy.dataobject.deferred import (
    DeferredArray,
    is_deferred,
    materialize,
    ReparsePixels,
    ReparseMeta,
)

__all__ = ["Loadable", "DataObject", "DataObjectStack"]


class _ArithmeticMixin:
    """Mixin providing elementwise +, -, *, / for DataObject subclasses that hold a single
    numeric array. Subclasses set ``_op_attr`` to that array's attribute name (e.g. "image"
    for Image, "y" for Line). Accepts a same-shaped instance duck-typed on ``_op_attr``, a
    plain Number, or an ndarray; in-place variants rebind the result, never mutating the array."""

    _op_attr: Optional[str] = None

    def _operand(self, other: Any) -> Any:
        if isinstance(other, (Number, np.ndarray, DeferredArray)):
            return other
        # Raw _data access (not getattr/materialize): if `other`'s op-attr value is still a
        # DeferredArray, this lets the operation COMPOSE into a new DeferredArray instead of
        # forcing a load -- the whole point of doing arithmetic on a virtual DataObject/stack.
        other_data = getattr(other, "_data", None)
        if isinstance(other_data, dict) and self._op_attr in other_data:
            return other_data[self._op_attr]
        # Fall back to plain getattr duck-typing for anything else that merely exposes the
        # right attribute (not necessarily a DataObject).
        operand = getattr(other, self._op_attr, None)
        if operand is None:
            raise TypeError(
                f"Unsupported Operation for types {type(self)} and {type(other)}"
            )
        return operand

    def _iop(self, other: Any, op: Callable[[Any, Any], Any]) -> _ArithmeticMixin:
        # `op` is the *out-of-place* operator, rebound onto `_data`: the stored array may alias
        # the caller's own (`Image(arr)`, `ImageStack(arr)` frames, `Line(arr)` views), so it
        # must never be mutated. Raw _data access (not getattr): reading would materialize a
        # deferred value just to add to it; operator.add on a DeferredArray composes a new one.
        current, operand = self._data[self._op_attr], self._operand(other)
        if not is_deferred(current):
            # A resident object stays resident: load a deferred operand (one frame) instead of
            # composing the result into a DeferredArray with no source behind it.
            operand = materialize(operand)
        self._data[self._op_attr] = op(current, operand)
        return self

    def __iadd__(self, other: Any) -> _ArithmeticMixin:
        return self._iop(other, operator.add)

    def __add__(self, other: Any) -> _ArithmeticMixin:
        result = self.copy()
        result += other
        return result

    def __radd__(self, other: Any) -> _ArithmeticMixin:
        if other == 0:
            return self
        return self.__add__(other)

    def __isub__(self, other: Any) -> _ArithmeticMixin:
        return self._iop(other, operator.sub)

    def __sub__(self, other: Any) -> _ArithmeticMixin:
        result = self.copy()
        result -= other
        return result

    def __imul__(self, other: Any) -> _ArithmeticMixin:
        return self._iop(other, operator.mul)

    def __mul__(self, other: Any) -> _ArithmeticMixin:
        result = self.copy()
        result *= other
        return result

    def __rmul__(self, other: Any) -> _ArithmeticMixin:
        return self.__mul__(other)

    def __itruediv__(self, other: Any) -> _ArithmeticMixin:
        return self._iop(other, operator.truediv)

    def __truediv__(self, other: Any) -> _ArithmeticMixin:
        result = self.copy()
        result /= other
        return result


def _require_source_for_json(obj: Any, fname: str = ".json") -> None:
    """Raise ValueError before a JSON target is touched if `obj` (a DataObject, or every
    element of a stack) has no `source`: JSON stores only the source, so the file would be
    unloadable."""
    objs = obj.elements if isinstance(obj, DataObjectStack) else [obj]
    if fname.endswith(".json") and any(o.source is None for o in objs):
        raise ValueError(
            f"Can't save a {type(obj).__name__} without a source to JSON (it stores only the "
            "source, so the file could not be loaded). Save to .h5 or .pickle instead."
        )


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
        # deepcopy, not .copy(): a shallow copy would leave every instance sharing the same
        # mutable default value (e.g. LEEMImg._meta_defaults["warp_matrix"] = np.eye(3) --
        # every LEEMImg would share one np.eye(3) object until it set its own).
        obj._meta_store = copy.deepcopy(obj._meta_defaults)
        obj._units_store = copy.deepcopy(obj._unit_defaults)
        # Set *before* anything else can touch _meta/_units (see the _meta/_units properties
        # below) -- a lazily-populated DataObject (virtual=True) installs a loader here after
        # construction; _meta_loader stays None for a normal eager object forever.
        obj._meta_loader = None
        obj._source = None
        return obj

    def _resolve_meta(self) -> None:
        """Run `_meta_loader` (if one is installed) exactly once, merging its result into
        `_meta_store`/`_units_store` the same way `__init__` routes a freshly-parsed dict, then
        clear the loader permanently. Called by the `_meta`/`_units` properties on first touch
        of either container -- a read OR a write -- never on a per-key miss: a per-field
        DataObject (e.g. LEEMImg) already ships every field in `_meta_defaults`, so a
        miss-triggered loader would never fire and would silently return the wrong (default)
        value instead of the parsed one."""
        loader = self.__dict__.get("_meta_loader")
        if loader is None:
            return
        self.__dict__["_meta_loader"] = None
        parsed = loader()
        meta_store = self.__dict__["_meta_store"]
        units_store = self.__dict__["_units_store"]
        for key, value in parsed.items():
            if key.startswith("_"):
                continue
            if "_unit" in key:
                units_store[key.replace("_unit", "")] = value
            else:
                meta_store[key] = value
        for key in meta_store:
            units_store.setdefault(key, "")

    @property
    def _meta(self) -> dict:
        if self.__dict__.get("_meta_loader") is not None:
            self._resolve_meta()
        return self.__dict__["_meta_store"]

    @property
    def _units(self) -> dict:
        if self.__dict__.get("_meta_loader") is not None:
            self._resolve_meta()
        return self.__dict__["_units_store"]

    def __init__(self, source: Any, defer: bool = False) -> None:
        """__init__(self, source: Any, defer: bool = False)

        `defer=True` builds the object without reading any data: `_data` entries are
        `DeferredArray` handles (see `uspy.dataobject.deferred`) and metadata is resolved from
        a loader on first touch (see `_resolve_meta`) instead of being parsed up front. This is
        what `DataObjectStack(..., virtual=True)` builds its elements with.
        """
        if defer:
            deferred = self.parse_deferred(source)
            for k in self._data_keys:
                if k not in deferred:
                    raise ValueError(f"parse_deferred function did not give '{k}' data.")
                self._data[k] = deferred[k]
            self._meta_loader = deferred.get("_meta_loader")
            return
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

    def parse_deferred(self, source: Any) -> dict[str, Any]:
        """Like `parse`, but for `defer=True`: must set `_source` and return a dict mapping
        every key in `_data_keys` to a `DeferredArray`, plus (optionally) a `"_meta_loader"`
        entry -- a picklable zero-arg callable returning the flat metadata dict `_resolve_meta`
        expects (same shape as `parse()`'s return value, minus the data keys).

        Default implementation: a generic re-parse fallback (`ReparsePixels`/`ReparseMeta`,
        see `uspy.dataobject.deferred`) that re-parses `source` from scratch on every access --
        today's `virtual=True` semantics, minus the throwaway stack element. This gives every
        `DataObject` subclass `defer=True` support for free; override for a cheaper per-format
        path (e.g. `Image` reads a TIFF page/header instead of the whole file, `LEEMImg` uses
        its reader's `parse_pixels`/`parse_metadata`)."""
        self._source = source
        cls = type(self)
        result = {key: DeferredArray(ReparsePixels(cls, source, key)) for key in self._data_keys}
        result["_meta_loader"] = ReparseMeta(cls, source)
        return result

    @property
    def is_virtual(self) -> bool:
        """True if any `_data` value is still a `DeferredArray` (not yet loaded)."""
        return any(isinstance(v, DeferredArray) for v in self._data.values())

    def materialize(self) -> DataObject:
        """Materialize every deferred `_data` value in place. Returns self."""

        for key, value in list(self._data.items()):
            if isinstance(value, DeferredArray):
                self._data[key] = value.load()
        return self

    def defer(self) -> DataObject:
        """Drop resident `_data` values back to deferred handles, re-derived from `source` on
        next access (via `parse_deferred`). Note: any in-memory pixel edit not yet reflected
        at `source` is discarded by this, exactly like re-virtualizing already discarded a
        stack's resident elements before this refactor -- not a new loss, just a new spelling
        of the same one."""
        if self._source is None:
            raise ValueError(f"Can't defer a {type(self).__name__} with no source to reload from.")
        deferred = self.parse_deferred(self._source)
        for key in self._data_keys:
            self._data[key] = deferred[key]
        return self

    def materialized(self) -> DataObject:
        """Non-mutating counterpart of `materialize()`: returns `self` unchanged if nothing is
        deferred, otherwise a *new* object with every deferred `_data` value loaded, built via
        `_with_data` (no array deep-copy). Used by `DataObjectStack.materialize()` so a
        virtual->eager transition never mutates an element that another stack (e.g. a slice)
        might still be holding a reference to."""
        if not self.is_virtual:
            return self
        return self._with_data(**{k: materialize(v) for k, v in self._data.items()})

    def deferred(self) -> DataObject:
        """Non-mutating counterpart of `defer()`: returns `self` unchanged if already fully
        deferred, otherwise a *new* object whose `_data` values are deferred handles re-derived
        from `source` (via `parse_deferred`), same discard-on-reload caveat as `defer()`. Used by
        `DataObjectStack.defer()` for the same reason as `materialized()`."""
        if self.is_virtual:
            return self
        if self._source is None:
            raise ValueError(f"Can't defer a {type(self).__name__} with no source to reload from.")
        handles = self.parse_deferred(self._source)
        return self._with_data(**{k: handles[k] for k in self._data_keys})

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
        `_time_origin`/`_mask_spec`, ...) is shared by reference, not deep-copied.
        """
        new = copy.copy(self)  # shares _data/_meta/_units/other attrs by reference
        # object.__setattr__ bypasses __setattr__'s no-op-on-existing-dict guard, needed
        # here to give `new` its own independent dicts instead of sharing `self`'s. `_meta`/
        # `_units` are read-only properties now, so the stores (not the properties) are the
        # target; the loader is carried over unresolved rather than forced to resolve here --
        # `self`/`new` each resolve it independently, at most once, whenever first touched.
        object.__setattr__(new, "_data", dict(self._data, **data))
        object.__setattr__(new, "_meta_store", copy.deepcopy(self.__dict__["_meta_store"]))
        object.__setattr__(new, "_units_store", copy.deepcopy(self.__dict__["_units_store"]))
        object.__setattr__(new, "_meta_loader", self.__dict__.get("_meta_loader"))
        return new

    @property
    def source(self) -> Optional[Any]:
        """The original object from which this instance was created (e.g., filename)."""
        return self._source

    @property
    def data(self) -> dict[str, Any]:
        """Mutable data container dictionary. Returns the live `_data` dict when every value
        is already resident (the common case, and what `_with_data`'s `result.data is not
        original.data` independence check relies on); returns a materialized copy (deferred
        values loaded, nothing cached) when some value is still a `DeferredArray`, since a
        materialized copy can't alias `self`'s deferred storage."""
        if any(isinstance(v, DeferredArray) for v in self._data.values()):
            return {k: materialize(v) for k, v in self._data.items()}
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
        # A private-name miss (e.g. `_meta`/`_units`/`_meta_store` failing their own property
        # lookup during the __new__/unpickling window, before _meta_store exists yet) must
        # not fall through to `self._data`/`self._meta` below -- both are properties that
        # would re-trigger the very lookup that got us here, and infinitely recurse.
        if attr.startswith("_"):
            raise AttributeError(f"No attribute named {attr}")
        if attr in self._data:
            # The single materialization point: a deferred _data value (a DeferredArray, see
            # uspy.dataobject.deferred) is loaded here so callers always see a real ndarray.
            return materialize(self._data[attr])
        if attr in self._meta:
            return self._meta[attr]
        if attr in type(self)._meta_defaults:
            # Field added to `_meta_defaults` after this object was built (e.g. unpickled from
            # an older version): defaults are only copied in at construction. Deepcopy, like
            # `__new__`, so a mutable default is never shared; not written back on read.
            return copy.deepcopy(type(self)._meta_defaults[attr])
        raise AttributeError(f"No attribute named {attr}")

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr in ("_meta", "_units", "_data") and "_meta_store" in self.__dict__:
            return
        if attr.startswith("_") or attr in self.__dict__:
            super().__setattr__(attr, value)
        elif isinstance(getattr(type(self), attr, None), property):
            # A class-level property (e.g. LEEMImg.time_origin) must run its own setter (or
            # raise, if read-only) instead of being silently swallowed into _meta below.
            super().__setattr__(attr, value)
        elif attr in self._data:
            if is_deferred(self._data[attr]) and not is_deferred(value):
                # A virtual element's whole memory-boundedness rests on `_data[attr]` staying
                # a DeferredArray -- replacing it here with a resident array pins this one
                # element's data in memory for good. A containing stack's `virtual` property
                # is derived from its elements, so it *will* notice (flip to False) the moment
                # this element is checked -- but that's still one element's memory pinned for
                # good, and (if the element is shared, e.g. via a stack slice) may force an
                # unrelated stack to report non-virtual too. Compose onto the deferred value
                # instead (`DeferredArray.map`/arithmetic; see `LEEMImg.warp`/`LEEMImg.normalize`)
                # so it survives as a new DeferredArray.
                warnings.warn(
                    UserWarning(
                        f"Assigning a resident array to '{attr}' replaces this element's "
                        "deferred data with a resident one, permanently pinning it in memory "
                        "-- any stack containing this element (directly or via a slice) will "
                        "report virtual=False from now on. Compose onto the deferred value "
                        "instead (DeferredArray.map(fn) or arithmetic) to keep it out-of-core "
                        "-- see LEEMImg.warp/LEEMImg.normalize."
                    )
                )
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

    def save(self, fname: str, **kwargs) -> DataObject:
        _require_source_for_json(self, fname)
        return super().save(fname, **kwargs)

    def __json_encode__(self) -> dict:
        """Returns dict with all information to reconstruct object from a .json file"""
        _require_source_for_json(self)
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
        # Raw _data, not the materializing `data` property: printing/repr'ing an object must
        # never trigger a pixel read (or a bulk metadata parse) as a side effect. A deferred
        # entry shows its DeferredArray repr (shape/dtype/source) instead of its values.
        for k, v in self._data.items():
            prettystring += f"\t{k}:\t{v}\n"
        return prettystring


class DataObjectStack(Loadable):
    """Contains multiple DataObjects. E.g., for an image stack.

    `virtual=True` means the stack's elements are built with deferred (`defer=True`) data:
    real, stable `DataObject`s whose `_data` values are `DeferredArray` handles rather than
    resident arrays (see `uspy.dataobject.deferred`) -- metadata is still resolved and cached
    normally, only the (potentially large) data arrays are read on demand. This differs from
    older versions of this class, where a virtual stack's elements were bare sources
    (filenames/arrays) re-parsed into a throwaway `DataObject` on every single access: that
    meant no write could ever persist, `stack[i] is stack[i]` never held, and metadata had to
    be fully re-read on every gather. None of that applies any more."""

    _unique_attrs = ()
    _type = DataObject

    def __init__(self, source: Union[str, Iterable], virtual: bool = False) -> None:
        # `virtual` is derived from the elements (see the `virtual` property below), not
        # stored -- there is no `self._virtual` to set here.
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
            objects_are_virtual = all(obj.is_virtual for obj in source)
            if virtual and not objects_are_virtual:
                warnings.warn(
                    UserWarning(
                        "Stack won't be virtual (data objects were directly given)"
                    )
                )
                # Materialize (into new objects, never mutating the caller's own) any element
                # that's still deferred -- a no-op for one that's already resident. Covers a
                # mix of virtual/resident objects too, which used to leave a silent mix behind.
                source = [obj.materialized() for obj in source]
            elif not virtual:
                # Keep the invariant that a non-virtual stack's elements are all resident:
                # materialize rather than silently carry a deferred (or mixed) set of objects
                # under a `virtual` flag that says otherwise. Same materialized() no-op logic
                # as above -- this is the only thing enforcing the invariant now that `virtual`
                # is derived from the elements instead of a stored flag.
                source = [obj.materialized() for obj in source]
            # else: virtual requested and every object already virtual -- nothing to do.

            # if stack is created from objects, all objects have to be the same. For a virtual
            # stack only the type is checked: `is_compatible` may compare pixel shapes, which
            # re-parses every deferred element (see check_compatible for the full check).
            for obj in source[1:]:
                if not (
                    isinstance(obj, type(source[0]))
                    if objects_are_virtual and virtual
                    else source[0].is_compatible(obj)
                ):
                    raise TypeError(
                        f"Not all initialization objects are of type {self._type}"
                    )

            self._elements = list(source)
        else:
            sources = self._split_source(source)
            if virtual:
                self._elements = self._construct_virtual(sources)
            else:
                self._elements = self._construct(sources)

    def _split_source(self, source: Union[str, Iterable]) -> list:
        """Split the source parameter of the constructor into source arguments
        that can be passed to single DataObjects."""
        # return_val = source
        if not isinstance(source, list):
            raise ValueError(f"Cannot create Objects from {type(source)}")
        return source

    def _construct(self, sources: list) -> list:
        """Build a list of resident elements from a list of sources, checking pairwise
        compatibility as each one is built (mismatched elements are caught immediately,
        since every element is loaded anyway)."""
        if not sources:
            raise ValueError("Empty source for DataObjectStack (wrong file path?)")
        elements = [self._single_construct(sources[0])]
        for source in tqdm(sources[1:]):
            element = self._single_construct(source)
            if not elements[0].is_compatible(element):
                reason = elements[0].incompatibility_reason(element)
                raise ValueError(
                    f"Cannot build {type(self).__name__}: {element.source!r} is not "
                    f"compatible with {elements[0].source!r} ({reason})."
                )
            elements.append(element)
        return elements

    def _construct_virtual(self, sources: list) -> list:
        """Build a list of deferred elements from a list of sources -- no data is read, and
        (unlike `_construct`) no pairwise compatibility check runs, since that would cost one
        metadata resolve per element just to build a virtual stack. Call `check_compatible()`
        explicitly to check early instead of at first bulk use."""
        if not sources:
            raise ValueError("Empty source for DataObjectStack (wrong file path?)")
        return [self._single_construct(source, defer=True) for source in sources]

    def materialize(self) -> DataObjectStack:
        """Load every element's deferred data in place. Returns self, so calls can be chained.

        Rebinds `self._elements` to freshly materialized copies (`DataObject.materialized()`),
        after checking they're actually compatible -- a virtual stack never checked this at
        construction (see `_construct_virtual`). Rebinding rather than mutating each element in
        place matters because elements may be shared with another stack (e.g. a slice, see
        `__getitem__`): mutating them here would silently pin that other stack's data in memory
        too, behind its back. No-op (returns immediately) if already resident."""
        if not self.virtual:
            return self
        self.check_compatible()
        self._elements = [element.materialized() for element in self._elements]
        return self

    def defer(self) -> DataObjectStack:
        """Drop every element's data back to deferred handles, in place. Returns self, so calls
        can be chained.

        Rebinds `self._elements` to freshly deferred copies (`DataObject.deferred()`), so data
        is re-derived from `source` on next access instead of staying resident. Note: any
        in-memory edit to an element's data that isn't reflected at `source` is discarded by
        this -- the same thing that happened when a virtual stack's elements were discarded
        outright and replaced by bare sources. As in `materialize()`, rebinding (not mutating in
        place) keeps this from reaching into another stack that happens to share these elements.
        No-op (returns immediately) if already virtual."""
        if self.virtual:
            return self
        if None in self.sources:
            raise ValueError("Can't defer stack without source information.")
        self._elements = [element.deferred() for element in self._elements]
        return self

    def materialized(self) -> DataObjectStack:
        """Non-mutating counterpart of `materialize()`: returns `self` unchanged if already
        resident, otherwise a *new* stack (built via `_respun`) with every element materialized
        -- `self` and its elements are left untouched."""
        if not self.virtual:
            return self
        self.check_compatible()
        return self._respun([element.materialized() for element in self._elements])

    def deferred(self) -> DataObjectStack:
        """Non-mutating counterpart of `defer()`: returns `self` unchanged if already virtual,
        otherwise a *new* stack (built via `_respun`) with every element deferred -- `self` and
        its elements are left untouched."""
        if self.virtual:
            return self
        if None in self.sources:
            raise ValueError("Can't defer stack without source information.")
        return self._respun([element.deferred() for element in self._elements])

    def _respun(self, elements: list) -> DataObjectStack:
        """A new stack of this type, same unique attrs, with `elements` swapped in. Built via
        the subclass-aware slice path (`self[:]`, which subclasses like `LEEMStack` already
        override to carry `time_origin`/`reader` through) so `materialized()`/`deferred()` don't
        need their own copy of that per-subclass logic. Zero-I/O: `self[:]` rebuilds a stack of
        the same virtual-ness as `self`, then the elements list is swapped for the already
        materialized/deferred one."""
        new = self[:]
        new._elements = elements
        return new

    def _single_construct(self, source: Any, defer: bool = False) -> DataObject:
        """Construct a single DataObject, resident (`defer=False`) or deferred (`defer=True`)."""
        return self._type(source, defer=defer)

    def check_compatible(self) -> None:
        """Check every element is pairwise-compatible with the first, the same check
        `_construct` already makes as it builds an eager stack. A virtual stack skips this at
        construction (see `_construct_virtual`) to stay zero-I/O; call this explicitly to fail
        early instead of at first bulk use (`virtual = False`, `average()`, `save()`, a data-key
        gather like `stack.image`, ...), which call it automatically."""
        if not self._elements:
            return
        first = self._elements[0]
        for element in self._elements[1:]:
            if not first.is_compatible(element):
                reason = first.incompatibility_reason(element)
                raise ValueError(
                    f"Cannot use this {type(self).__name__}: {element.source!r} is not "
                    f"compatible with {first.source!r} ({reason}). This is detected now "
                    "rather than at construction because the stack is virtual; call "
                    "stack.check_compatible() right after building it to fail early instead."
                )

    @property
    def elements(self) -> list:
        """Contents: the real DataObjects, whether virtual (deferred data) or not."""
        return self._elements

    @property
    def sources(self) -> list:
        """Sources."""
        return [element.source for element in self._elements]

    def set_unit(self, attr: str, unit: str) -> None:
        """Set the unit string for a per-element metadata field on every element, without
        touching the values themselves. Used e.g. by `LEEMReader.post_process_stack`
        implementations that distribute a shared per-frame value (and its unit) across a
        stack."""
        for element in self:
            element._units[attr] = unit  # pylint: disable=protected-access

    @property
    def virtual(self) -> bool:
        """If a data stack is virtual, it only loads data on demand. Derived from the elements
        (`all(element.is_virtual for element in self._elements)`) rather than a stored flag, so
        it can never misreport: any element pinned resident behind the stack's back (e.g. by
        sharing elements with a slice that got materialized, or by `img.image = some_array`)
        immediately shows up here instead of being silently out of sync."""
        return all(element.is_virtual for element in self.__dict__.get("_elements", ()))

    @virtual.setter
    def virtual(self, value: bool) -> None:
        warnings.warn(
            "Setting `.virtual` is deprecated; use stack.materialize()/stack.defer() to "
            "convert in place, or stack.materialized()/stack.deferred() for a new stack.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.defer() if value else self.materialize()

    def copy(self):
        """Returns a deepcopy."""
        return copy.deepcopy(self)

    def __getitem__(
        self, index: Union[int, slice, Iterable]
    ) -> Union[DataObject, DataObjectStack]:
        if isinstance(index, slice):
            elements = self._elements[index]
        else:
            try:
                return self._elements[operator.index(index)]  # also numpy integers
            except TypeError:
                pass
            index = np.asarray(index)  # sequence of ints, or boolean mask
            if index.dtype == bool:
                index = np.flatnonzero(index)
            elements = [self._elements[i] for i in index]
        return type(self)(elements, virtual=self.virtual)

    @staticmethod
    def _require_deferred(obj: DataObject) -> DataObject:
        """`obj` itself if deferred; ValueError for a resident one, which a virtual stack can't
        hold without silently dropping its in-memory edits (re-deriving it from `source`)."""
        if not obj.is_virtual:
            raise ValueError(
                "Can't put a resident DataObject into a virtual stack: its in-memory edits "
                "would be lost. Pass `obj.deferred()` (explicitly discards in-memory edits and "
                "re-reads from source) or save it first and load the saved file."
            )
        return obj

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
                insert = self._require_deferred(other)
            else:
                insert = [self._require_deferred(element) for element in other]

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
            elements = list(other)

        if self.virtual:
            elements = [self._require_deferred(element) for element in elements]
        else:
            for element in elements:
                if not self._elements[0].is_compatible(element):
                    raise ValueError("Incompatible element assignment")
        self._elements.extend(elements)

    def __getattr__(self, attr: str):
        if attr.startswith("_") or attr in self.__dict__:
            raise AttributeError
        if self.virtual and attr in self._type._data_keys:  # pylint: disable=protected-access
            warnings.warn(
                UserWarning(
                    f"Gathering '{attr}' across a virtual {type(self).__name__} materializes "
                    "every element's data into memory all at once -- for a large/data-heavy "
                    "stack, prefer iterating elements one at a time instead."
                )
            )
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
        # An ndarray with ndim >= 2 (e.g. a 3x3 matrix) is one value, never a sequence of rows.
        is_array = isinstance(value, np.ndarray) and value.ndim >= 2
        if is_array and len(self) and value.shape[0] == len(self):
            warnings.warn(
                UserWarning(
                    f"'{attr}': array of shape {value.shape} was broadcast as one value to "
                    f"every element, not split into rows. Pass list(array) to assign one row "
                    "per element."
                )
            )
        if not is_array and isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            value = list(value)
            is_sequence = True
        else:
            is_sequence = False

        if is_sequence and len(value) == len(self):
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
            if not len(self):
                raise ValueError(
                    f"Can't set attribute '{attr}' on an empty stack: there are no "
                    "elements to set it on."
                )
            for obj in self:
                setattr(obj, attr, value.copy() if is_array else value)

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
            # Composes on a virtual stack's elements (see _ArithmeticMixin._iop) instead of
            # materializing them -- stays out-of-core, same as the eager case stays resident.
            for element in self._elements:
                element += other
        return self

    def __add__(
        self, other: Union[DataObjectStack, DataObject, Number]
    ) -> DataObjectStack:
        result = self.copy()  # preserves virtual -- no forced materialization
        result += other
        return result

    def __radd__(
        self, other: Union[DataObjectStack, DataObject, Number]
    ) -> DataObjectStack:
        return self.__add__(other)

    def __isub__(self, other: Union[DataObject, DataObject, Number]) -> DataObjectStack:
        for element in self._elements:
            element -= other
        return self

    def __sub__(
        self, other: Union[DataObjectStack, DataObject, Number]
    ) -> DataObjectStack:
        result = self.copy()
        result -= other
        return result

    def __imul__(self, other: Union[DataObject, Number]) -> DataObjectStack:
        for element in self._elements:
            element *= other
        return self

    def __mul__(self, other: Union[DataObject, Number]) -> DataObjectStack:
        result = self.copy()
        result *= other
        return result

    def __rmul__(self, other: Union[DataObject, Number]) -> DataObjectStack:
        return self.__mul__(other)

    def __itruediv__(self, other: Union[DataObject, Number]) -> DataObjectStack:
        for element in self._elements:
            element /= other
        return self

    def __truediv__(self, other: Union[DataObject, Number]) -> DataObjectStack:
        result = self.copy()
        result /= other
        return result

    def save(self, fname: str, **kwargs) -> DataObjectStack:
        _require_source_for_json(self, fname)
        return super().save(fname, **kwargs)

    def __json_encode__(self) -> dict:
        _require_source_for_json(self)
        return {"source": self.elements, "virtual": self.virtual}

    def __json_decode__(self, **attrs) -> None:
        self.__init__(**attrs)
