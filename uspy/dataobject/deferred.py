"""
Deferred (out-of-core) array data for `DataObject`/`DataObjectStack`.

A `_data` entry (e.g. `Image._data["image"]`) may hold a `DeferredArray` instead of a real
`np.ndarray`. It behaves like a handle: it knows its `shape`/`dtype` (cheaply, without reading
pixel data, whenever the loader can tell it), and it materializes into a real array only via
`.load()` / `np.asarray(...)`.

Caching contract (deliberate, matches `uspy.leem.base.LEEMStack(..., virtual=True)`):
    - shape/dtype are cached once resolved (they're metadata, not data)
    - pixel data is NEVER cached -- `.load()` re-runs the loader every single call. Reading
      `img.image` twice reads the file twice. This is what bounds memory for a stack that
      doesn't fit in RAM: nothing holds a full frame alive longer than the caller does.

Composition: elementwise `+ - * /` (both operand orders) and `.map(fn)` build a *new*
`DeferredArray` wrapping the operation instead of evaluating it -- so `deferred_stack * 2` or
`(mcp_array - deferred_image) / (mcp_array - dark)` (see `uspy.leem.base.LEEMImg.normalize`)
stay out-of-core: nothing is read until something finally calls `.load()`.

Loaders (and the values combined into a composed `DeferredArray`) must be picklable -- never a
closure/lambda -- so a virtual/deferred `DataObject`/stack pickles to "sources + metadata",
not to fully materialized pixel data. See `ReaderPixels`/`TiffPageLoader`/`ImageIOLoader`/
`ReparsePixels` for the concrete loaders used elsewhere in `uspy`.
"""
# pylint: disable=protected-access

from __future__ import annotations

import operator
import warnings
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Callable, Optional

import numpy as np

__all__ = [
    "DeferredArray",
    "is_deferred",
    "materialize",
    "ReparsePixels",
    "ReparseMeta",
    "ReaderPixels",
    "ReaderMeta",
]


def is_deferred(value: Any) -> bool:
    """True if `value` is a `DeferredArray` (not yet loaded)."""
    return isinstance(value, DeferredArray)


def materialize(value: Any) -> Any:
    """Return `value.load()` if `value` is a `DeferredArray`, else `value` unchanged."""
    if isinstance(value, DeferredArray):
        return value.load()
    return value


_OPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "truediv": operator.truediv,
}


@dataclass(frozen=True)
class ReparsePixels:
    """Generic fallback pixel loader for any `DataObject` subclass with no cheaper deferred
    path (see `DataObject.parse_deferred`'s default implementation): re-parses `source` from
    scratch and pulls out one data key. This is exactly today's `virtual=True` semantics
    (bounded, uncached, per access) minus the throwaway stack element -- every subclass gets
    `defer=True` support for free; override `parse_deferred` for a cheaper per-format path
    (e.g. `Image`'s TIFF-page/imageio loaders, `LEEMImg`'s reader-backed ones)."""

    cls: type
    source: Any
    key: str

    def __call__(self) -> np.ndarray:
        return self.cls(self.source)._data[self.key]  # pylint: disable=protected-access


@dataclass(frozen=True)
class ReparseMeta:
    """Generic fallback metadata loader pairing `ReparsePixels`: re-parses `source` from
    scratch and returns a flat dict in the same shape `DataObject.__init__`/`_resolve_meta`
    expect (meta keys plus "<key>_unit" keys)."""

    cls: type
    source: Any

    def __call__(self) -> dict:
        obj = self.cls(self.source)
        result = dict(obj._meta)  # pylint: disable=protected-access
        for key, value in obj._units.items():  # pylint: disable=protected-access
            result[f"{key}_unit"] = value
        return result


@dataclass(frozen=True)
class ReaderPixels:
    """Pixel loader backed by a `uspy.leem.readers.LEEMReader` (or anything with the same
    `parse_pixels(source)` method): the cheap per-format path `LEEMImg.parse_deferred` uses
    instead of the generic `ReparsePixels` re-parse-the-whole-object fallback."""

    reader: Any
    source: Any

    def __call__(self) -> np.ndarray:
        return self.reader.parse_pixels(self.source)


@dataclass(frozen=True)
class ReaderMeta:
    """Metadata loader backed by a `LEEMReader`'s `parse_metadata(source)` -- the cheap,
    pixel-free per-format path `LEEMImg.parse_deferred` uses instead of `ReparseMeta`."""

    reader: Any
    source: Any

    def __call__(self) -> dict:
        return dict(self.reader.parse_metadata(self.source))


@dataclass(frozen=True)
class MappedArray:
    """Loader that applies a picklable unary function to another loaded array."""

    inner: Any  # DeferredArray | ndarray-producing loader-free value handled by caller
    fn: Callable[[np.ndarray], np.ndarray]

    def __call__(self) -> np.ndarray:
        return self.fn(materialize(self.inner) if is_deferred(self.inner) else self.inner)


@dataclass(frozen=True)
class BinaryOp:
    """Loader that combines two (possibly deferred) operands with a named elementwise op."""

    op_name: str
    left: Any
    right: Any

    def __call__(self) -> np.ndarray:
        op = _OPS[self.op_name]
        left = materialize(self.left) if is_deferred(self.left) else self.left
        right = materialize(self.right) if is_deferred(self.right) else self.right
        return op(left, right)


class DeferredArray:
    """A not-yet-loaded array. See module docstring for the caching/composition contract."""

    # Tells numpy to hand binary ops involving a DeferredArray back to *our* __radd__ etc.
    # instead of broadcasting/materializing via __array__ -- required for e.g.
    # `ndarray - deferred_array` (LEEMImg.normalize's `mcp_array - self.image`) to compose
    # rather than silently materialize.
    __array_priority__ = 1000
    __array_ufunc__ = None

    def __init__(
        self,
        loader: Callable[[], np.ndarray],
        shape: Optional[tuple] = None,
        dtype: Optional[Any] = None,
        shape_provider: Optional[Callable[[], tuple]] = None,
    ):
        self.loader = loader
        self._shape = tuple(shape) if shape is not None else None
        self._dtype = np.dtype(dtype) if dtype is not None else None
        self.shape_provider = shape_provider

    ### shape/dtype -- resolved lazily, cached once known, pixels never cached

    def _resolve_shape_dtype(self) -> None:
        if self.shape_provider is not None:
            self._shape = tuple(self.shape_provider())
        if self._shape is not None and self._dtype is not None:
            return
        # Last resort: load once to learn shape/dtype. This is the one case a "shape" query
        # can cost a real read, when neither the constructor nor a shape_provider knew it.
        # np.asarray here (not plain .load()) is deliberate and local to this fallback: it's
        # the one place an ndarray's .shape/.dtype are genuinely needed, not a general
        # coercion of whatever the loader returns (see .load()'s docstring).
        arr = np.asarray(self.load())
        if self._shape is None:
            self._shape = arr.shape
        if self._dtype is None:
            self._dtype = arr.dtype

    @property
    def shape(self) -> tuple:
        if self._shape is None:
            self._resolve_shape_dtype()
        return self._shape

    @property
    def dtype(self):
        if self._dtype is None:
            self._resolve_shape_dtype()
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        n = 1
        for dim in self.shape:
            n *= dim
        return n

    @property
    def nbytes(self) -> int:
        return self.size * (self.dtype.itemsize if self.dtype is not None else 4)

    ### materialization -- never cached

    def load(self) -> Any:
        """Run the loader and return its result verbatim -- never caches: call this again and
        the loader runs again. Deliberately does *not* force the result through `np.asarray`:
        most `_data_keys` values are ndarrays already, but the generic re-parse fallback
        (`ReparsePixels`, used by any `DataObject` subclass with no cheaper deferred path) can
        defer a bare Python scalar just as well (e.g. a test helper whose data key holds a
        plain Number) -- coercing that to a 0-d ndarray here would silently change its type
        (and broke `DeepDiff`-based `DataObject.__eq__`, which chokes on a 0-d array). Use
        `np.asarray(deferred)` (below) when a real ndarray is specifically what's wanted."""
        return self.loader()

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        arr = np.asarray(self.load())
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    ### composition

    def map(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        shape: Optional[tuple] = None,
        dtype: Optional[Any] = None,
    ) -> "DeferredArray":
        """Return a new DeferredArray applying `fn` on load; does not load now. `fn` must be
        picklable (a module-level function, not a lambda/closure -- `functools.partial` around
        one is fine) if the result needs to survive a pickle without materializing.

        `shape`/`dtype` default to unknown: a mapped function may change either, so callers
        that know `fn` preserves them (e.g. a same-size warp) should pass the *already-known*
        `self._shape`/`self._dtype` through explicitly -- never the `.shape`/`.dtype`
        properties, which could themselves force a load to answer."""
        return DeferredArray(MappedArray(self, fn), shape=shape, dtype=dtype)

    def _binop(self, op_name: str, other: Any, reflect: bool = False) -> "DeferredArray":
        if not isinstance(other, (Number, np.ndarray, DeferredArray)):
            return NotImplemented
        left, right = (other, self) if reflect else (self, other)
        shape = self._shape if not reflect else (
            other.shape if isinstance(other, DeferredArray) else getattr(other, "shape", self._shape)
        )
        return DeferredArray(BinaryOp(op_name, left, right), shape=shape)

    def __add__(self, other):
        return self._binop("add", other)

    def __radd__(self, other):
        return self._binop("add", other, reflect=True)

    def __sub__(self, other):
        return self._binop("sub", other)

    def __rsub__(self, other):
        return self._binop("sub", other, reflect=True)

    def __mul__(self, other):
        return self._binop("mul", other)

    def __rmul__(self, other):
        return self._binop("mul", other, reflect=True)

    def __truediv__(self, other):
        return self._binop("truediv", other)

    def __rtruediv__(self, other):
        return self._binop("truediv", other, reflect=True)

    ### copying -- immutable value object, so copying is free and never loads

    def __copy__(self) -> "DeferredArray":
        return self

    def __deepcopy__(self, memo) -> "DeferredArray":
        return self

    ### identity-only equality -- an elementwise __eq__ would return an ndarray of bools and
    ### break DeepDiff-based DataObject.__eq__, which compares _data values with `==`/asserts

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def __repr__(self) -> str:
        shape = self._shape if self._shape is not None else "?"
        dtype = self._dtype if self._dtype is not None else "?"
        return f"<DeferredArray {dtype} {shape} via {type(self.loader).__name__}>"

    ### pickling -- try the loader as-is; if it can't be pickled (e.g. built from a lambda),
    ### materialize with a warning rather than let the whole dump fail

    def __reduce__(self):
        try:
            import pickle

            pickle.dumps(self.loader)
        except Exception:  # pylint: disable=broad-except
            warnings.warn(
                f"DeferredArray's loader ({self.loader!r}) is not picklable; materializing "
                "it into a real array for this pickle instead of deferring. Use a picklable "
                "loader (a module-level function/callable, not a lambda/closure) to keep "
                "this deferred through a pickle round-trip.",
                UserWarning,
            )
            return (np.asarray, (self.load(),))
        return (DeferredArray, (self.loader, self._shape, self._dtype, self.shape_provider))
