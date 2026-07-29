"""
Out-of-core image stacks for data too large to hold in memory.

`LazyImageStack` (and its LEEM counterpart, `uspy.leem.base.LazyLEEMStack`) builds one lazy
`dask.array` task graph over all frames' pixel data up front, unlike `DataObjectStack`'s
uncached ``virtual=True`` mode. Slicing, `+ - * /`, and `.normalize()` stay lazy/chunked, so
a stack far larger than RAM can be sliced, aggregated, or streamed with bounded memory.
Per-frame metadata is read once at construction and kept fully resident as a plain list of
dicts, so ``stack.energy`` etc. is instant and pixel-I/O-free.

This is an additional backend alongside the eager and ``virtual=True`` ones; it doesn't
change their behavior. Only indexing/slicing, per-frame metadata, `average()`, elementwise
arithmetic, and `.normalize()` are supported -- anything needing every element's pixels at
once (`.align()`, `.filter()`, item assignment) raises; call `.materialize()` first.

Requires the optional ``dask``/``zarr`` dependencies (``pip install uspy[bigdata]``); only
actually building/using a lazy stack needs them, not importing this module.
"""

# pylint: disable=protected-access

from __future__ import annotations
from typing import Any, Optional, Union
from collections.abc import Iterable
from numbers import Number
import warnings

import numpy as np
import tifffile
from tifffile.tifffile import TiffFileError

from uspy.dataobject.base import DataObject
from uspy.dataobject.image import Image, ImageStack, list_image_files

try:
    import dask
    import dask.array as da

    _HAVE_DASK = True
except ImportError:  # pragma: no cover - exercised via test with dask uninstalled
    da = None
    _HAVE_DASK = False


def _require_dask() -> None:
    if not _HAVE_DASK:
        raise ImportError(
            "This needs the optional 'dask'/'zarr' dependencies for the out-of-core lazy "
            "stack backend. Install them with: pip install uspy[bigdata]"
        )


# Materializing a lazy stack bigger than this warns instead of silently blocking for a long
# time / risking an OOM. Not a hard limit -- we have no reliable way to know free RAM without
# adding a new dependency (e.g. psutil), so this is just a sanity-check threshold.
_MATERIALIZE_WARN_BYTES = 4 * 1024**3  # 4 GiB


class LazyImageStack(ImageStack):
    """Out-of-core `ImageStack` backed by a lazy `dask.array` of pixel data. See the module
    docstring for the overall design. State:

    - ``self._dask``: lazy ``dask.array`` of shape ``(N, H, W)``, chunked along frames.
    - ``self._meta_list``: plain list of N per-frame metadata dicts (fully resident).
    - ``self._sources``: per-frame source paths if built from files, else ``None``.

    Uses dask's default local, in-process, threaded scheduler.
    """

    _type = Image
    _materialized_type = ImageStack

    def __new__(cls, *args, **kwargs):
        # A Lazy* class already IS the lazy backend, so it never re-dispatches on `lazy=`
        # like ImageStack.__new__ does.
        return object.__new__(cls)

    def __init__(self, source: Union[str, list], chunks: int = 1, lazy: bool = True) -> None:
        """
        Parameters
        ----------
        source
            Either a path to a single 3D TIFF stack, or a list of per-frame image file paths.
        chunks
            Number of frames per dask chunk along the stack axis, by default 1 (bounds memory
            most tightly; increase for fewer, larger tasks if that's a better tradeoff).
        lazy
            Accepted and ignored -- a re-entry guard for `ImageStack(source, lazy=True)`,
            which invokes this `__init__` a second time via `isinstance` dispatch.
        """
        if "_dask" in self.__dict__:
            return
        _require_dask()
        self._source = source
        self._sources, self._dask, self._meta_list = self._build(source, chunks)
        self._stack_meta = {}

    # -- construction ------------------------------------------------------------------

    def _build(self, source: Union[str, list], chunks: int) -> tuple:
        if isinstance(source, list) and source and isinstance(source[0], str):
            return self._build_from_files(source, chunks)

        # A glob/directory expands to per-frame files before falling through to a single 3D
        # TIFF path below (mirrors ImageStack._split_source's precedence).
        fnames = list_image_files(source)
        if len(fnames) > 1:
            return self._build_from_files(fnames, chunks)
        if len(fnames) == 1:
            single = fnames[0]
            if single.lower().endswith((".tif", ".tiff")):
                try:
                    probe = tifffile.imread(single)
                except (
                    FileNotFoundError,
                    TypeError,
                    TiffFileError,
                    ValueError,
                    IsADirectoryError,
                ):
                    return self._build_from_files(fnames, chunks)
                if probe.ndim == 3:
                    return self._build_from_tiff_stack(single, chunks)
            return self._build_from_files(fnames, chunks)

        if isinstance(source, str) and source.lower().endswith((".tif", ".tiff")):
            return self._build_from_tiff_stack(source, chunks)
        raise ValueError(
            f"Cannot build a {type(self).__name__} from {source!r}; expected a 3D TIFF path "
            "or a list of per-frame image file paths."
        )

    def _build_from_tiff_stack(self, path: str, chunks: int) -> tuple:
        store = tifffile.imread(path, aszarr=True)
        arr = da.from_zarr(store)
        if arr.ndim != 3:
            raise ValueError(f"{path} is not a 3D image stack (shape {arr.shape})")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        arr = arr.rechunk((chunks, *arr.shape[1:]))
        meta_list = [{} for _ in range(arr.shape[0])]
        return None, arr, meta_list

    def _build_from_files(self, fnames: list, chunks: int) -> tuple:
        if not fnames:
            raise ValueError("Empty source for LazyImageStack (wrong file path?)")
        first_pixels = self._read_pixels(fnames[0])
        shape, dtype = first_pixels.shape, np.float32
        meta_list = [self._read_metadata(fnames[0])]
        frames = [da.from_array(first_pixels, chunks=shape)]
        for fname in fnames[1:]:
            meta = self._read_metadata(fname)
            # Catch a shape mismatch now, at construction, rather than once that frame's
            # chunk is computed (mirrors DataObjectStack._construct's check).
            frame_shape = self._shape_from_meta(meta)
            if frame_shape is not None and frame_shape != shape:
                raise ValueError(
                    f"Cannot build {type(self).__name__}: {fname!r} is not compatible "
                    f"with {fnames[0]!r} (image shape {frame_shape} != {shape})."
                )
            meta_list.append(meta)
            frames.append(
                da.from_delayed(
                    dask.delayed(self._read_pixels)(fname), shape=shape, dtype=dtype
                )
            )
        arr = da.stack(frames).rechunk((chunks, *shape))
        return list(fnames), arr, meta_list

    @staticmethod
    def _shape_from_meta(meta: dict) -> Optional[tuple]:
        """Pixel shape implied by a `_read_metadata` dict's "height"/"width" keys, or `None`
        if either is missing (the mismatch check in `_build_from_files` is then skipped)."""
        height, width = meta.get("height"), meta.get("width")
        if height is None or width is None:
            return None
        return (height, width)

    def _read_pixels(self, fname: str) -> np.ndarray:
        """Read just the pixel array of one frame; a full parse via `self._type` here,
        overridden by subclasses with a cheaper pluggable-reader path (e.g. `LazyLEEMStack`)."""
        return np.asarray(self._type(fname).image, dtype=np.float32)

    def _read_metadata(self, fname: str) -> dict:
        """Read one frame's metadata as a flat dict (meta keys plus '<key>_unit' keys). Base
        implementation does a full parse and discards pixels; `LazyLEEMStack` overrides this
        with a real metadata-only parse (no pixel I/O)."""
        obj = self._type(fname)
        result = dict(obj.meta)
        for key, value in obj.unit.items():
            result[f"{key}_unit"] = value
        return result

    @staticmethod
    def _apply_meta(element: DataObject, meta: dict) -> None:
        """Route a flat metadata dict (as returned by `_read_metadata`, or a reader's
        parse_image/parse_metadata) into an already-constructed element's `_meta`/`_units`,
        mirroring the key routing `DataObject.__init__` does for a freshly-parsed dict."""
        for key, value in meta.items():
            if key.startswith("_"):
                continue
            if "_unit" in key:
                element._units[key.replace("_unit", "")] = value
            else:
                element._meta[key] = value

    def _make_element(self, pixels: np.ndarray, meta: dict, index: int) -> DataObject:
        element = self._type(pixels)
        self._apply_meta(element, meta)
        if self._sources is not None:
            element._source = self._sources[index]
        return element

    def _wrap(self, dask_array, meta_list: list, sources: Optional[list]) -> "LazyImageStack":
        """Build a new stack of the same concrete type/config as `self` (reader, time origin,
        etc. for subclasses that carry extra state) over a given array/metadata/sources,
        without re-parsing/re-reading anything from disk. Used internally by slicing,
        arithmetic, normalize, average, and copy."""
        new = object.__new__(type(self))
        new._source = self._source
        new._sources = sources
        new._dask = dask_array
        new._meta_list = meta_list
        new._stack_meta = dict(self._stack_meta)
        return new

    # -- access --------------------------------------------------------------------------

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            idx = int(index)
            if idx < 0:
                idx += len(self)
            if not 0 <= idx < len(self):
                raise IndexError(f"{type(self).__name__} index out of range")
            pixels = np.asarray(self._dask[idx].compute())
            return self._make_element(pixels, self._meta_list[idx], idx)
        if isinstance(index, slice):
            sources = self._sources[index] if self._sources is not None else None
            return self._wrap(self._dask[index], self._meta_list[index], sources)
        raise TypeError(f"Unsupported index type {type(index)} for {type(self).__name__}")

    def __len__(self) -> int:
        return int(self._dask.shape[0])

    def __getattr__(self, attr: str):
        if attr.startswith("_") or attr in self.__dict__:
            raise AttributeError(attr)
        if isinstance(getattr(self._type, attr, None), property):
            # A computed per-element property (e.g. LEEMImg.rel_time) isn't a plain
            # _meta_list key; resolve it on a lightweight, metadata-only element per frame
            # instead, checked before the dict gather since it must win over a same-named
            # stored meta default. Placeholder pixels keep this pixel-I/O-free.
            placeholder = np.zeros(self._dask.shape[1:], dtype=np.float32)
            return np.array(
                [
                    getattr(self._make_element(placeholder, meta, i), attr)
                    for i, meta in enumerate(self._meta_list)
                ]
            )
        # Per-frame metadata is fully resident -- gathering it never touches pixel data.
        try:
            return np.array([m[attr] for m in self._meta_list])
        except KeyError:
            pass
        if attr in self._stack_meta:
            return self._stack_meta[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no per-frame or stack-level attribute "
            f"{attr!r}"
        )

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr.startswith("_") or attr in self.__dict__:
            return super().__setattr__(attr, value)
        # Real class-level properties (e.g. `virtual`, which explicitly rejects being
        # toggled) need their own setter to run rather than being masked by this override.
        if isinstance(getattr(type(self), attr, None), property):
            return super().__setattr__(attr, value)
        # A read-only property on the *element* class can't be broadcast into per-frame
        # metadata; refuse it rather than silently writing a dict entry the getter ignores.
        element_prop = getattr(self._type, attr, None)
        if isinstance(element_prop, property) and element_prop.fset is None:
            raise AttributeError(f"can't set attribute '{attr}'")
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            value = list(value)
            if len(value) == len(self):
                for meta, single_value in zip(self._meta_list, value):
                    meta[attr] = single_value
                    # Arbitrary metadata must always have a unit entry (see
                    # DataObject.__setattr__) -- units live inline as "<attr>_unit" keys here.
                    meta.setdefault(f"{attr}_unit", "")
                return None
            # Mismatched-length sequence: can't be per-frame metadata, so it would have to
            # attach to the stack itself -- unless `attr` names a real per-frame field, in
            # which case that would silently shadow that field's gather in __getattr__.
            if self._meta_list and attr in self._meta_list[0]:
                raise ValueError(
                    f"'{attr}' is a per-element attribute; can't overwrite it with a "
                    "mismatched-length iterable."
                )
            self._stack_meta[attr] = value
            return None
        # Scalar (incl. plain string, not split character-wise): broadcast to every frame,
        # mirroring DataObjectStack.__setattr__ for eager stacks.
        if not self._meta_list:
            raise ValueError(
                f"Can't set attribute '{attr}' on an empty stack: there are no elements to "
                "set it on."
            )
        for meta in self._meta_list:
            meta[attr] = value
            meta.setdefault(f"{attr}_unit", "")

    def set_unit(self, attr: str, unit: str) -> None:
        """Lazy counterpart to `DataObjectStack.set_unit`: writes directly into the resident
        `_meta_list` dicts instead of iterating constructed elements (which, for a lazy stack,
        would mean decompressing every frame's pixel data just to set a unit string)."""
        for meta in self._meta_list:
            meta[f"{attr}_unit"] = unit

    @property
    def sources(self) -> Optional[list]:
        """Per-frame source paths, or None if this stack wasn't built from per-frame files
        (e.g. a 3D TIFF source, or the result of arithmetic/normalize/average)."""
        return self._sources

    @property
    def virtual(self) -> bool:
        return False

    @virtual.setter
    def virtual(self, value: bool) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} doesn't support the eager 'virtual' toggle. Call "
            ".materialize() to get a regular in-memory stack."
        )

    @property
    def lazy(self) -> bool:
        """Always True; identifies this as a dask-array-backed out-of-core stack."""
        return True

    def copy(self, virtual: Optional[bool] = None) -> "LazyImageStack":
        if virtual is not None:
            raise NotImplementedError(
                f"{type(self).__name__} doesn't support the 'virtual' toggle; see "
                ".materialize()."
            )
        # The dask graph is only ever extended (never mutated in place) by any operation
        # here, so it's safe to share rather than deep-copy.
        sources = list(self._sources) if self._sources is not None else None
        return self._wrap(self._dask, list(self._meta_list), sources)

    # -- unsupported eager-only operations ------------------------------------------------

    def __setitem__(self, index, other) -> None:
        raise TypeError(
            f"{type(self).__name__} doesn't support item assignment (dask arrays are "
            "immutable). Build a new stack instead."
        )

    def __delitem__(self, index) -> None:
        raise TypeError(f"{type(self).__name__} doesn't support item deletion.")

    def extend(self, other) -> None:
        raise TypeError(
            f"{type(self).__name__} doesn't support extend(). Materialize both stacks first, "
            "or build a new lazy stack over the concatenated sources."
        )

    def filter(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} doesn't support .filter(); call .materialize() first."
        )

    # -- arithmetic ------------------------------------------------------------------------

    def _operand_array(self, other: Any):
        if isinstance(other, LazyImageStack):
            return other._dask
        if isinstance(other, DataObject):
            return np.asarray(other.image, dtype=np.float32)
        if isinstance(other, (Number, np.ndarray)):
            return other
        raise TypeError(
            f"Unsupported operand type for {type(self).__name__}: {type(other)}. Expected a "
            "number, an (H,W) array, a single Image, or another lazy stack."
        )

    def __add__(self, other):
        return self._wrap(
            self._dask + self._operand_array(other), list(self._meta_list), self._sources
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._wrap(
            self._dask - self._operand_array(other), list(self._meta_list), self._sources
        )

    def __rsub__(self, other):
        return self._wrap(
            self._operand_array(other) - self._dask, list(self._meta_list), self._sources
        )

    def __mul__(self, other):
        return self._wrap(
            self._dask * self._operand_array(other), list(self._meta_list), self._sources
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._wrap(
            self._dask / self._operand_array(other), list(self._meta_list), self._sources
        )

    def __rtruediv__(self, other):
        return self._wrap(
            self._operand_array(other) / self._dask, list(self._meta_list), self._sources
        )

    def _no_inplace(self, symbol: str):
        raise TypeError(
            f"In-place '{symbol}=' isn't supported on {type(self).__name__} (dask arrays are "
            f"immutable); use 'stack = stack {symbol} other' instead."
        )

    def __iadd__(self, other):
        self._no_inplace("+")

    def __isub__(self, other):
        self._no_inplace("-")

    def __imul__(self, other):
        self._no_inplace("*")

    def __itruediv__(self, other):
        self._no_inplace("/")

    # -- aggregation -------------------------------------------------------------------------

    def average(self, avg: int) -> "LazyImageStack":
        """Lazy counterpart to `ImageStack.average`: averages over consecutive groups of `avg`
        frames. The metadata of the first frame in every averaged group is preserved, matching
        the eager version."""
        if len(self) % avg != 0:
            raise ValueError(
                f"Number of images in stack ({len(self)}) is not dividable by {avg}"
            )
        n = len(self) // avg
        rechunked = self._dask.rechunk((avg, *self._dask.shape[1:]))
        averaged = rechunked.reshape(n, avg, *self._dask.shape[1:]).mean(axis=1)
        averaged = averaged.astype(np.float32)
        meta_list = [dict(self._meta_list[i * avg]) for i in range(n)]
        return self._wrap(averaged, meta_list, None)

    # -- materialization / persistence --------------------------------------------------------

    def materialize(self) -> ImageStack:
        """Eagerly load the whole stack into memory as a regular (non-lazy) stack. Warns (but
        doesn't refuse) if the stack looks too large to comfortably fit in RAM."""
        nbytes = self._dask.nbytes
        if nbytes > _MATERIALIZE_WARN_BYTES:
            warnings.warn(
                f"Materializing a {nbytes / 1024**3:.1f} GiB {type(self).__name__} into "
                "memory -- this may exhaust available RAM.",
                ResourceWarning,
            )
        computed = np.asarray(self._dask.compute())
        elements = [
            self._make_element(computed[i], self._meta_list[i], i) for i in range(len(self))
        ]
        return self._materialized_type(elements)

    def save(self, fname: str, **kwargs) -> "LazyImageStack":
        """Persist pixel data to a *new* file -- never touches the original source data.
        Supports '.h5'/'.hdf5'/'.nxs' (NeXus-flavoured HDF5, see `uspy.hdf5` -- unlike '.zarr'
        below, this keeps per-frame metadata, streamed one chunk at a time via plain iteration
        over `self`), '.zarr' (chunked, itself lazily re-readable, pixels only -- no metadata),
        and '.tif'/'.tiff' (streamed frame-by-frame, bounded memory). Pickle/JSON aren't
        supported -- see `dumps`."""
        from uspy.hdf5 import is_hdf5_path

        if is_hdf5_path(fname):
            from uspy.hdf5 import write

            write(self, fname, **kwargs)
            return self
        if kwargs:
            raise TypeError(
                f"save() got unexpected keyword argument(s) {list(kwargs)!r}: these are only "
                "accepted for an HDF5 target (.h5/.hdf5/.nxs)."
            )
        if fname.lower().endswith(".zarr"):
            self._dask.to_zarr(fname)
            return self
        if fname.lower().endswith((".tif", ".tiff")):
            shape = (len(self),) + tuple(self._dask.shape[1:])

            def frames():
                for i in range(len(self)):
                    yield np.asarray(self._dask[i].compute())

            # Generator + explicit shape/dtype keeps writing streamed (one frame at a time)
            # as a single stack; photometric="minisblack" avoids tifffile misreading a
            # 3-frame stack as RGB.
            tifffile.imwrite(
                fname,
                frames(),
                shape=shape,
                dtype=np.float32,
                bigtiff=True,
                photometric="minisblack",
            )
            return self
        raise NotImplementedError(
            f"{type(self).__name__}.save only supports '.zarr' or '.tif'/'.tiff' targets. "
            "Call .materialize() first for other formats."
        )

    def dumps(self, use_pickle: bool = True) -> bytes:
        raise NotImplementedError(
            f"{type(self).__name__} doesn't support pickle/json serialization of its dask "
            "task graph (this could silently discard lazy transformations like arithmetic/"
            "normalize/average that were applied since it was built from source, or try to "
            "pickle unpicklable store/lock objects). Call .materialize() first, or use "
            ".save('*.zarr'/'*.tif') to persist pixel data directly."
        )
