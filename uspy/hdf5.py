"""
NeXus-flavoured HDF5 export/import -- an archivable, open-format alternative to pickle/JSON.

`.pickle` is Python-version-fragile and unsafe to distribute; `.json` only stores a source
path and re-reads the original file on load; `.tif`/`.zarr` store pixels but drop metadata.
`obj.save("deposit.h5")` (any `DataObject`/`DataObjectStack`) instead writes one self-contained
HDF5 file with pixels, every metadata value (Python type preserved), and every unit -- readable
by bare `h5py`, MATLAB, silx and DAWN, and by `Cls.load("deposit.h5")` losslessly.

One generic writer/reader serves every type built on `_data`/`_meta`/`_units` (see
`uspy.dataobject.base`). Derived types export as their parent (`IntensityLine`/`StitchedLine`/
`ProfileLine`/`IVCurve` -> `Line`, `Waterfall` -> `Image`); anything that can't be represented
that way (an upstream `ImageStack`/`ROI`/
`Profile` reference) is simply not written, with a `DroppedFieldWarning` naming it. A stack
column missing on some frames is filled (nan/''/False) with a `MissingFieldWarning`; a column
whose per-frame unit differs is written under its single most common unit with a
`UnitMismatchWarning`; genuinely mixed-type columns are coerced to a uniform dtype with a plain
warning. The file itself is deliberately uspy-agnostic: two content groups (`/entry/data`,
`/entry/metadata`) plus standard NeXus scaffolding, and exactly three `uspy_*` attributes in
the whole file -- no per-field type tag; a value's Python type is inferred from its HDF5 dtype
on read, and a missing/unrecognized `uspy_class` falls back to structural inference (2-D
signal -> `Image`, 3-D -> `ImageStack`, `y`-over-`x` -> `Line`) with a warning. `write()`/
`save()` only ever creates a new file; the exported object and its source are never touched.

File layout
-----------
::

    /                     @NX_class="NXroot" @default="entry" @creator="uspy"
                          @file_time @HDF5_Version
                          @uspy_version @uspy_format_version
    /entry                @NX_class="NXentry" @default="data"
                          @uspy_class="uspy.leem.base.LEEMStack"   # the only reconstruction attr
      title               e.g. "LEEMStack (4 frames)" or a source basename
      program_name        "uspy"  @version=<uspy.__version__>
      start_time          ISO8601, only if a finite 'timestamp' meta value exists
      instrument          @NX_class="NXinstrument"          (only if a reader name is known)
        name              e.g. "elmitec"
      /entry/data         @NX_class="NXdata" @signal="image" @axes=["energy", ".", "."]
        image   (N,H,W)   chunked, gzip+shuffle+fletcher32 by default
        frame   (N,)      = arange(N), always written, @long_name="frame index"
        energy  -> hard link to /entry/metadata/energy   (whichever key was picked as the axis)
      /entry/metadata     @NX_class="NXcollection"
        energy      (N,) f8          @units="eV"
        temperature (N,) f8          @units="°C"
        fov         (N,) vlen-utf8
        source      (N,) vlen-utf8   # original per-frame acquisition path, ordinary metadata
        warp_matrix (N,3,3) f8
        ...
        <key>_missing (N,) bool      # only written if `key` is absent on SOME (not all) frames
        <key>_unit    (N,) vlen-utf8 # only written if `key`'s unit differs across frames

A single object (`Image`/`LEEMImg`/`Line`) is the same minus the leading `N` axis: `/entry/data`
holds `image` (H,W) or `x`/`y` (M,), and `/entry/metadata` holds plain scalar/array datasets
instead of per-frame columns.

Two escape hatches worth knowing about:

- `HDF5_USE_FILE_LOCKING`: HDF5's file locking can misbehave on some network filesystems
  (NFS/SMB mounts common in lab environments). If `save()`/`load()` hangs or raises an OS-level
  locking error there, set the environment variable ``HDF5_USE_FILE_LOCKING=FALSE`` (a plain
  HDF5 library setting, nothing uspy-specific) before starting Python.
- Compression defaults (gzip level 4, shuffle, fletcher32 checksums) favour archival
  readability over write speed or ratio -- gzip is the only compressor guaranteed to be
  readable by *any* HDF5 install with no optional plugin. Pass ``compression=None`` to
  `save()`/`write()` for large virtual stacks where write time matters more.
"""

from __future__ import annotations

import importlib
import os
import urllib.parse
import warnings
from datetime import datetime
from typing import Any, Optional

import numpy as np
import h5py

# uspy.dataobject/uspy.leem classes are imported locally inside functions, not at module level,
# so `import uspy` still works in an environment missing h5py until an actual .h5 path is hit.

HDF5_EXTENSIONS = (".h5", ".hdf5", ".nxs")
FORMAT_VERSION = "1.0"

# Chunk sizing target for pixel data: chunks smaller than this waste B-tree overhead on tiny
# arrays; chunks larger than this stop paying for themselves under gzip and make a single-frame
# read fetch more than one frame's worth of compressed data.
_CHUNK_TARGET_BYTES = 1 * 1024**2
_CHUNK_MAX_BYTES = 32 * 1024**2


class DroppedFieldWarning(UserWarning):
    """A metadata value (e.g. an upstream ImageStack/ROI/Profile reference) can't be
    represented in HDF5 and was omitted, with no record left in the archive."""


class UnitMismatchWarning(UserWarning):
    """A stack's per-frame metadata field has different units across frames; the most
    common unit is written as the single NeXus 'units' string, values are untouched."""


class MissingFieldWarning(UserWarning):
    """A stack's per-frame metadata field is missing on some frames; the gap is filled
    (nan/''/False) rather than raising, so a round-tripped stack reads a fabricated value
    where the original would have raised AttributeError."""


def is_hdf5_path(fname: Any) -> bool:
    """True if `fname` is a string ending in one of `HDF5_EXTENSIONS` (case-insensitive).
    Deliberately excludes ".hdf" (HDF4, used by `uspy.leem.readers.smart`)."""
    return isinstance(fname, str) and fname.lower().endswith(HDF5_EXTENSIONS)


######### value coercion: Python <-> HDF5, no type tags -- see module docstring ##################


def _to_plain_array(value: Any) -> Optional[np.ndarray]:
    """`np.asarray(value)`, but `None` (not an empty/object array) if that produces an object
    dtype -- i.e. `value` wasn't actually a uniform list/tuple of representable scalars."""
    try:
        arr = np.asarray(value)
    except Exception:  # pylint: disable=broad-except
        return None
    if arr.dtype == object:
        return None
    return arr


def _classify(value: Any) -> str:
    """Coarse kind of a single metadata value, used to decide how a *stack* column (which may
    mix kinds across frames) should be coerced. Order matters: bool before int (bool is an int
    subclass), int/float before the ndarray fallback for 0-d numpy scalars."""
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, (float, np.floating)):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, np.ndarray) and value.dtype != object:
        return "ndarray"
    if isinstance(value, (list, tuple)) and _to_plain_array(value) is not None:
        return "ndarray"
    return "other"  # ROI, Profile, DataObject(Stack), dict, callables, ragged/object arrays...


def _encode_value(group: h5py.Group, name: str, value: Any) -> bool:
    """Write `value` as a dataset named `name` under `group`. Returns False (writes nothing)
    for `None` or anything not representable as a scalar/string/numeric ndarray; the caller
    warns about the latter, not `None`."""
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        group.create_dataset(name, data=np.bool_(value))
        return True
    if isinstance(value, (int, np.integer)):
        group.create_dataset(name, data=np.int64(value))
        return True
    if isinstance(value, (float, np.floating)):
        group.create_dataset(name, data=np.float64(value))
        return True
    if isinstance(value, str):
        group.create_dataset(name, data=value, dtype=h5py.string_dtype(encoding="utf-8"))
        return True
    if isinstance(value, (list, tuple)):
        value = _to_plain_array(value)
        if value is None:
            return False
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "US":
            group.create_dataset(
                name, data=value.astype(object), dtype=h5py.string_dtype(encoding="utf-8")
            )
            return True
        if value.dtype == object:
            flat = value.ravel()
            if flat.size and all(isinstance(item, str) for item in flat):
                group.create_dataset(name, data=value, dtype=h5py.string_dtype(encoding="utf-8"))
                return True
            return False
        group.create_dataset(name, data=value)
        return True
    return False


def _decode_value(dataset: h5py.Dataset) -> Any:
    """Inverse of `_encode_value`. A scalar (0-d) dataset decodes to the matching native
    Python scalar; anything else to an `np.ndarray` with its stored dtype."""
    if h5py.check_string_dtype(dataset.dtype) is not None:
        decoded = dataset.asstr("utf-8")[()]
        if dataset.shape == ():
            return str(decoded)
        return np.asarray(decoded, dtype=object)
    raw = dataset[()]
    if dataset.shape == ():
        if dataset.dtype == np.dtype("bool"):
            return bool(raw)
        if np.issubdtype(dataset.dtype, np.integer):
            return int(raw)
        if np.issubdtype(dataset.dtype, np.floating):
            return float(raw)
    return np.asarray(raw)


def _decode_column(dataset: h5py.Dataset) -> list:
    """Decode a per-frame metadata column dataset into a plain Python list of length N (one
    native value per frame): scalars for a 1-D column, an ndarray per row for a column that is
    itself multi-dimensional per frame (e.g. `warp_matrix` stacked as `(N, 3, 3)`)."""
    if h5py.check_string_dtype(dataset.dtype) is not None:
        return list(dataset.asstr("utf-8")[:])
    raw = dataset[()]
    if raw.ndim == 1:
        if dataset.dtype == np.dtype("bool"):
            return [bool(v) for v in raw]
        if np.issubdtype(dataset.dtype, np.integer):
            return [int(v) for v in raw]
        if np.issubdtype(dataset.dtype, np.floating):
            return [float(v) for v in raw]
        return list(raw)
    return [np.asarray(raw[i]) for i in range(raw.shape[0])]


def _sanitize_key(key: str) -> tuple:
    """Escape `key` for use as an HDF5 name (which can't contain '/') via `urllib.parse.quote`;
    ordinary snake_case keys pass through unchanged. Returns (safe_key, was_changed)."""
    safe = urllib.parse.quote(key, safe="")
    return safe, safe != key


def _unsanitize_key(safe_key: str) -> str:
    return urllib.parse.unquote(safe_key)


######### class resolution: "derived types export as their parent" #############################


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _target_class(obj: Any) -> tuple:
    """(target_class, kind) an object should be archived as; kind is "object" or "stack".
    A stack's element type is read off its own `_type` attribute to decide between
    `LEEMStack`/`ImageStack`."""
    from uspy.dataobject.base import DataObjectStack
    from uspy.dataobject.image import Image, ImageStack
    from uspy.dataobject.line import Line
    from uspy.leem.base import LEEMImg, LEEMStack

    if isinstance(obj, DataObjectStack):
        elem_cls = getattr(obj, "_type", Image)
        if issubclass(elem_cls, LEEMImg):
            return LEEMStack, "stack"
        return ImageStack, "stack"
    if isinstance(obj, LEEMImg):
        return LEEMImg, "object"
    if isinstance(obj, Line):
        return Line, "object"
    if isinstance(obj, Image):
        return Image, "object"
    raise TypeError(
        f"uspy.hdf5 doesn't know how to export a {type(obj).__name__}; the supported base "
        "types are Image, LEEMImg, Line, ImageStack, LEEMStack (every other uspy data type "
        "exports as one of these)."
    )


def _resolve_class(qualname: str) -> Optional[type]:
    """Import and return the class named by a dotted path, or None if that fails for any
    reason. Callers are responsible for the "uspy." prefix restriction (see `_resolve_target`);
    this only refuses to try when the prefix is already missing, as a second line of defence."""
    if not isinstance(qualname, str) or not qualname.startswith("uspy."):
        return None
    module_name, _, cls_name = qualname.rpartition(".")
    if not module_name or not cls_name:
        return None
    try:
        module = importlib.import_module(module_name)
        return getattr(module, cls_name)
    except (ImportError, AttributeError, ValueError):
        return None


def _structural_fallback(entry: h5py.Group) -> tuple:
    """Infer (target_class, kind) purely from the shape of the `/entry/data` signal, for a
    file with no (or an unresolvable) 'uspy_class' attribute -- so a hand-written, uspy-unaware
    HDF5 file following this layout is still loadable."""
    from uspy.dataobject.image import Image, ImageStack
    from uspy.dataobject.line import Line

    data_grp = entry["data"]
    signal_name = data_grp.attrs.get("signal", "image")
    signal = data_grp[signal_name]
    if signal_name == "y":
        return Line, "object"
    if signal.ndim == 2:
        return Image, "object"
    if signal.ndim == 3:
        return ImageStack, "stack"
    raise ValueError(
        f"Cannot infer a uspy type for HDF5 data of shape {signal.shape} at "
        f"'entry/data/{signal_name}'."
    )


def _resolve_target(entry: h5py.Group) -> tuple:
    """(target_class, kind) for a file being read, from its 'uspy_class' attribute. Raises on
    a non-'uspy.'-prefixed class name (would otherwise import arbitrary code); falls back to
    structural inference, with a warning, if the attribute is missing or unrecognized."""
    qualname = entry.attrs.get("uspy_class")
    if qualname is None:
        warnings.warn(
            "HDF5 archive has no 'uspy_class' attribute; falling back to structural inference "
            "(pixel/line shape) to pick a uspy type.",
            UserWarning,
        )
        return _structural_fallback(entry)

    if not isinstance(qualname, str) or not qualname.startswith("uspy."):
        raise ValueError(
            f"Refusing to resolve 'uspy_class' attribute {qualname!r}: only classes under "
            "the 'uspy' package are ever imported to reconstruct an archive."
        )

    cls = _resolve_class(qualname)
    if cls is None:
        warnings.warn(
            f"HDF5 archive names {qualname!r}, which could not be imported/found in this "
            "uspy version; falling back to structural inference (pixel/line shape).",
            UserWarning,
        )
        return _structural_fallback(entry)

    from uspy.dataobject.image import Image, ImageStack
    from uspy.dataobject.line import Line
    from uspy.leem.base import LEEMImg, LEEMStack

    mapping = {
        LEEMStack: "stack",
        ImageStack: "stack",
        LEEMImg: "object",
        Line: "object",
        Image: "object",
    }
    if cls not in mapping:
        warnings.warn(
            f"{qualname!r} is not one of uspy.hdf5's supported base types; falling back to "
            "structural inference.",
            UserWarning,
        )
        return _structural_fallback(entry)
    return cls, mapping[cls]


######### chunking / compression #################################################################


def _compression_kwargs(opts: dict) -> dict:
    kwargs = {}
    if opts.get("compression") is not None:
        kwargs["compression"] = opts["compression"]
        if opts.get("compression_opts") is not None:
            kwargs["compression_opts"] = opts["compression_opts"]
    if opts.get("shuffle"):
        kwargs["shuffle"] = True
    if opts.get("fletcher32"):
        kwargs["fletcher32"] = True
    return kwargs


def _image_chunks(height: int, width: int, itemsize: int, requested=None) -> tuple:
    if requested:
        return tuple(requested)
    frame_bytes = height * width * itemsize
    if frame_bytes == 0 or frame_bytes <= _CHUNK_MAX_BYTES:
        return (max(height, 1), max(width, 1))
    rows = max(1, int(_CHUNK_MAX_BYTES // max(width * itemsize, 1)))
    return (min(rows, height), width)


def _stack_chunks(n: int, height: int, width: int, itemsize: int, requested=None) -> tuple:
    if requested:
        return tuple(requested)
    frame_bytes = height * width * itemsize
    if frame_bytes == 0:
        return (1, max(height, 1), max(width, 1))
    if frame_bytes > _CHUNK_MAX_BYTES:
        rows = max(1, int(_CHUNK_MAX_BYTES // max(width * itemsize, 1)))
        return (1, min(rows, height), width)
    frames_per_chunk = max(1, min(n, int(np.ceil(_CHUNK_TARGET_BYTES / frame_bytes))))
    return (frames_per_chunk, height, width)


######### write ###################################################################################


def write(
    obj: Any,
    fname: str,
    *,
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 4,
    shuffle: bool = True,
    fletcher32: bool = True,
    chunks: Optional[tuple] = None,
    frame_axis: Optional[str] = "auto",
    title: Optional[str] = None,
) -> None:
    """Write any `DataObject`/`DataObjectStack` to a new NeXus-flavoured HDF5 file at `fname`.
    See the module docstring for the file layout and what is/isn't preserved. Never touches
    `obj` or any file it was read from -- `fname` is always a brand-new file."""
    import uspy

    opts = {
        "compression": compression,
        "compression_opts": compression_opts,
        "shuffle": shuffle,
        "fletcher32": fletcher32,
        "chunks": chunks,
        "frame_axis": frame_axis,
        "title": title,
    }
    target_cls, kind = _target_class(obj)
    if kind == "stack" and len(obj) == 0:
        raise ValueError("Cannot export an empty stack to HDF5 (wrong slice/filter?).")

    with h5py.File(fname, "w") as f:
        f.attrs["NX_class"] = "NXroot"
        f.attrs["default"] = "entry"
        f.attrs["creator"] = "uspy"
        f.attrs["file_time"] = datetime.utcnow().isoformat() + "Z"
        f.attrs["HDF5_Version"] = h5py.version.hdf5_version
        f.attrs["uspy_version"] = uspy.__version__
        f.attrs["uspy_format_version"] = FORMAT_VERSION

        if kind == "object":
            _write_object(f, obj, target_cls, opts)
        else:
            _write_stack(f, obj, target_cls, opts)


def _write_program_name(entry: h5py.Group) -> None:
    import uspy

    ds = entry.create_dataset(
        "program_name", data="uspy", dtype=h5py.string_dtype(encoding="utf-8")
    )
    ds.attrs["version"] = uspy.__version__


def _write_instrument(entry: h5py.Group, reader_name: Optional[str]) -> None:
    if not reader_name:
        return
    instr = entry.create_group("instrument")
    instr.attrs["NX_class"] = "NXinstrument"
    instr.create_dataset("name", data=reader_name, dtype=h5py.string_dtype(encoding="utf-8"))


def _reader_name(obj: Any) -> Optional[str]:
    reader = getattr(obj, "_reader", None)
    return getattr(reader, "name", None) if reader is not None else None


def _iso_time(timestamp: Any) -> Optional[str]:
    if isinstance(timestamp, (int, float, np.integer, np.floating)) and np.isfinite(timestamp):
        return datetime.utcfromtimestamp(float(timestamp)).isoformat() + "Z"
    return None


######### write: single object ####################################################################


def _write_object(f: h5py.File, obj: Any, target_cls: type, opts: dict) -> None:
    from uspy.dataobject.line import Line

    entry = f.create_group("entry")
    entry.attrs["NX_class"] = "NXentry"
    entry.attrs["default"] = "data"
    entry.attrs["uspy_class"] = _qualname(target_cls)

    _write_program_name(entry)

    meta_grp = entry.create_group("metadata")
    meta_grp.attrs["NX_class"] = "NXcollection"
    data_keys = {"x", "y"} if target_cls is Line else {"image"}
    dropped = _write_meta_object(meta_grp, obj, data_keys)
    _write_source_object(meta_grp, obj)

    data_grp = entry.create_group("data")
    data_grp.attrs["NX_class"] = "NXdata"
    if target_cls is Line:
        _write_line_data(data_grp, obj)
    else:
        _write_image_data(data_grp, obj, opts)
        _maybe_link_waterfall_axes(data_grp, meta_grp, obj)

    _write_instrument(entry, _reader_name(obj))

    title = opts.get("title")
    if title is None:
        src = getattr(obj, "_source", None)
        title = os.path.basename(src) if isinstance(src, str) else type(obj).__name__
    entry.create_dataset("title", data=str(title), dtype=h5py.string_dtype(encoding="utf-8"))

    iso = _iso_time(obj._meta.get("timestamp"))
    if iso is not None:
        entry.create_dataset("start_time", data=iso, dtype=h5py.string_dtype(encoding="utf-8"))

    if dropped:
        _warn_dropped_object(dropped, obj)


def _write_meta_object(meta_grp: h5py.Group, obj: Any, data_keys: set) -> list:
    dropped = []
    for key, value in obj._meta.items():
        if key in data_keys or value is None:
            continue
        safe_key, renamed = _sanitize_key(key)
        if renamed:
            warnings.warn(
                f"metadata key {key!r} contains characters not valid in an HDF5 name; "
                f"stored as {safe_key!r} (restored to {key!r} on load).",
                UserWarning,
            )
        if _encode_value(meta_grp, safe_key, value):
            unit = obj._units.get(key)
            if unit:
                meta_grp[safe_key].attrs["units"] = unit
        else:
            dropped.append(key)
    return dropped


def _write_source_object(meta_grp: h5py.Group, obj: Any) -> None:
    if "source" in obj._meta:
        return  # already ordinary metadata, e.g. re-exporting a previously-loaded archive
    src = getattr(obj, "_source", None)
    if isinstance(src, str):
        _encode_value(meta_grp, "source", src)


def _warn_dropped_object(dropped: list, obj: Any) -> None:
    types = sorted({type(obj._meta[key]).__name__ for key in dropped})
    warnings.warn(
        DroppedFieldWarning(
            f"{len(dropped)} metadata field(s) could not be represented in the HDF5 export "
            f"and were omitted: {', '.join(sorted(dropped))} (types: {', '.join(types)})."
        )
    )


def _write_line_data(data_grp: h5py.Group, obj: Any) -> None:
    x = np.asarray(obj._data["x"], dtype=np.float64)
    y = np.asarray(obj._data["y"], dtype=np.float64)
    xds = data_grp.create_dataset("x", data=x)
    yds = data_grp.create_dataset("y", data=y)
    data_grp.attrs["signal"] = "y"
    data_grp.attrs["axes"] = ["x"]
    xds.attrs["long_name"] = str(obj._meta.get("xdim", "x"))
    yds.attrs["long_name"] = str(obj._meta.get("ydim", "y"))
    x_unit = obj._units.get("x")
    if x_unit:
        xds.attrs["units"] = x_unit
    y_unit = obj._units.get("y")
    if y_unit:
        yds.attrs["units"] = y_unit


def _write_image_data(data_grp: h5py.Group, obj: Any, opts: dict) -> h5py.Dataset:
    arr = np.asarray(obj._data["image"])
    height, width = arr.shape
    chunks = _image_chunks(height, width, arr.dtype.itemsize, opts.get("chunks"))
    kwargs = _compression_kwargs(opts)
    ds = data_grp.create_dataset("image", data=arr, chunks=chunks, **kwargs)
    data_grp.attrs["signal"] = "image"
    data_grp.attrs["axes"] = [".", "."]
    return ds


def _maybe_link_waterfall_axes(data_grp: h5py.Group, meta_grp: h5py.Group, obj: Any) -> None:
    """A `Waterfall` (exported as a plain `Image`) keeps its x/y axes as ordinary meta;
    hard-link them into `/entry/data` as NeXus axes too, for a nicer third-party plot."""
    if type(obj).__name__ != "Waterfall":
        return
    height, width = obj._data["image"].shape
    axes = [".", "."]
    y = obj._meta.get("y")
    if isinstance(y, np.ndarray) and y.shape == (height,) and "y" in meta_grp:
        data_grp["y"] = meta_grp["y"]
        axes[0] = "y"
    x = obj._meta.get("x")
    if isinstance(x, np.ndarray) and x.shape == (width,) and "x" in meta_grp:
        data_grp["x"] = meta_grp["x"]
        axes[1] = "x"
    data_grp.attrs["axes"] = axes


######### write: stack ############################################################################


def _iter_frame_meta(stack: Any):
    """Yield (meta_dict, units_dict) per frame -- uniformly for eager and virtual stacks,
    since a virtual stack's elements are real DataObjects with their own `_meta`/`_units`
    (resolving metadata, not pixels, on first touch if not already cached)."""
    for element in stack:
        yield dict(element._meta), dict(element._units)


def _write_stack(f: h5py.File, stack: Any, target_cls: type, opts: dict) -> None:
    n = len(stack)
    if n == 0:
        raise ValueError("Cannot export an empty stack to HDF5 (wrong slice/filter?).")

    entry = f.create_group("entry")
    entry.attrs["NX_class"] = "NXentry"
    entry.attrs["default"] = "data"
    entry.attrs["uspy_class"] = _qualname(target_cls)

    _write_program_name(entry)

    frames = list(_iter_frame_meta(stack))

    meta_grp = entry.create_group("metadata")
    meta_grp.attrs["NX_class"] = "NXcollection"
    dropped, key_order = _write_stack_meta(meta_grp, frames, n)

    _write_instrument(entry, _reader_name(stack))

    data_grp = entry.create_group("data")
    data_grp.attrs["NX_class"] = "NXdata"
    _write_stack_images(data_grp, stack, n, opts)

    frame_axis_key = _pick_frame_axis(key_order, frames, opts.get("frame_axis", "auto"))
    axis0 = "."
    if frame_axis_key is not None:
        safe_key, _ = _sanitize_key(frame_axis_key)
        if safe_key in meta_grp:
            data_grp[safe_key] = meta_grp[safe_key]
            axis0 = frame_axis_key
    data_grp.attrs["signal"] = "image"
    data_grp.attrs["axes"] = [axis0, ".", "."]

    title = opts.get("title")
    if title is None:
        title = f"{target_cls.__name__} ({n} frames)"
    entry.create_dataset("title", data=str(title), dtype=h5py.string_dtype(encoding="utf-8"))

    if frames:
        iso = _iso_time(frames[0][0].get("timestamp"))
        if iso is not None:
            entry.create_dataset(
                "start_time", data=iso, dtype=h5py.string_dtype(encoding="utf-8")
            )

    if dropped:
        _warn_dropped_stack(dropped, frames)


def _write_stack_images(data_grp: h5py.Group, stack: Any, n: int, opts: dict) -> None:
    # stack[i].shape/._dtype resolve as cheaply as each element allows (header metadata for a
    # still-deferred frame, no pixel read) -- getting the dtype promoted across all n frames
    # this way costs no more than the write loop below already will, instead of reading every
    # frame's pixels here just to learn its dtype and then again to write it.
    height, width = stack[0].shape
    dtype = stack[0]._dtype  # pylint: disable=protected-access
    for i in range(1, n):
        dtype = np.promote_types(dtype, stack[i]._dtype)  # pylint: disable=protected-access

    chunks = _stack_chunks(n, height, width, np.dtype(dtype).itemsize, opts.get("chunks"))
    kwargs = _compression_kwargs(opts)
    ds = data_grp.create_dataset(
        "image", shape=(n, height, width), dtype=dtype, chunks=chunks, **kwargs
    )
    for i in range(n):
        ds[i] = np.asarray(stack[i].image, dtype=dtype)

    frame_ds = data_grp.create_dataset("frame", data=np.arange(n, dtype=np.int64))
    frame_ds.attrs["long_name"] = "frame index"


def _write_stack_meta(meta_grp: h5py.Group, frames: list, n: int) -> tuple:
    keys, seen = [], set()
    for meta, _ in frames:
        for key in meta:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    dropped = []
    written = []
    missing_report = {}

    for key in keys:
        values = [None] * n
        present = [False] * n
        for i, (meta, _) in enumerate(frames):
            value = meta.get(key)
            if value is not None:
                values[i] = value
                present[i] = True

        if not any(present):
            continue  # entirely None/absent -- matches single-object None semantics, no warn

        kinds = {_classify(v) for v, p in zip(values, present) if p}
        column, warn_mixed = _coerce_column(kinds, values, present)
        if column is None:
            dropped.append(key)
            continue

        safe_key, renamed = _sanitize_key(key)
        if renamed:
            warnings.warn(
                f"metadata key {key!r} contains characters not valid in an HDF5 name; "
                f"stored as {safe_key!r} (restored to {key!r} on load).",
                UserWarning,
            )
        if not _encode_value(meta_grp, safe_key, column):
            dropped.append(key)
            continue
        written.append(key)

        missing_idx = [i for i, p in enumerate(present) if not p]
        if missing_idx:
            missing_report[key] = missing_idx
        if warn_mixed:
            warnings.warn(
                f"stack metadata field {key!r} has mixed types across frames "
                f"({sorted(kinds)}); coerced to a uniform column.",
                UserWarning,
            )

        _write_stack_column_units(meta_grp, safe_key, key, frames, n)

    if missing_report:
        parts = "; ".join(f"{key!r} (frames {idxs})" for key, idxs in missing_report.items())
        warnings.warn(
            MissingFieldWarning(
                f"some stack frames are missing a value for: {parts}. These are filled with "
                "nan/''/False in the export and will read back as a value, not a missing key."
            )
        )

    return dropped, written


def _coerce_column(kinds: set, values: list, present: list):
    """(column_ndarray_or_None, warn_mixed) for a stack metadata column. `kinds` is the set of
    `_classify` results across present frames; `None` means the whole column is dropped because
    at least one value is fundamentally unrepresentable (ROI/DataObject/dict/callable)."""
    if "other" in kinds:
        return None, False
    if kinds == {"bool"}:
        arr = np.array([bool(v) if p else False for v, p in zip(values, present)], dtype=bool)
        return arr, False
    if kinds == {"int"}:
        arr = np.array([int(v) if p else 0 for v, p in zip(values, present)], dtype=np.int64)
        return arr, False
    if kinds == {"float"}:
        arr = np.array(
            [float(v) if p else np.nan for v, p in zip(values, present)], dtype=np.float64
        )
        return arr, False
    if kinds and kinds <= {"int", "float"}:
        arr = np.array(
            [float(v) if p else np.nan for v, p in zip(values, present)], dtype=np.float64
        )
        return arr, True
    if kinds == {"str"}:
        arr = np.array([str(v) if p else "" for v, p in zip(values, present)], dtype=object)
        return arr, False
    if kinds == {"ndarray"}:
        norm = [
            v if isinstance(v, np.ndarray) else _to_plain_array(v)
            for v, p in zip(values, present)
            if p
        ]
        shapes = {a.shape for a in norm}
        if len(shapes) == 1 and next(iter(shapes)) != ():
            shape = next(iter(shapes))
            filler = np.zeros(shape)

            def _as_arr(v):
                return v if isinstance(v, np.ndarray) else np.asarray(v)

            arr = np.stack(
                [_as_arr(v) if p else filler for v, p in zip(values, present)]
            )
            return arr, False
        return _stringify_column(values, present), True
    return _stringify_column(values, present), True


def _stringify_column(values: list, present: list) -> np.ndarray:
    return np.array([str(v) if p else "" for v, p in zip(values, present)], dtype=object)


def _write_stack_column_units(meta_grp: h5py.Group, safe_key: str, key: str, frames: list, n: int) -> None:
    units_seen = {}
    for _, units in frames:
        unit = units.get(key) or ""
        units_seen[unit] = units_seen.get(unit, 0) + 1
    non_empty = {u: c for u, c in units_seen.items() if u}
    if not non_empty:
        return
    if len(non_empty) == 1:
        meta_grp[safe_key].attrs["units"] = next(iter(non_empty))
        return
    most_common = max(non_empty.items(), key=lambda kv: kv[1])[0]
    meta_grp[safe_key].attrs["units"] = most_common
    warnings.warn(
        UnitMismatchWarning(
            f"{key!r} has {len(non_empty)} different units across the {n} frames of this "
            f"stack ({sorted(non_empty)}); the NeXus 'units' attribute is set to the most "
            f"common one ({most_common!r})."
        )
    )


def _warn_dropped_stack(dropped: list, frames: list) -> None:
    types = set()
    for key in dropped:
        for meta, _ in frames:
            value = meta.get(key)
            if value is not None:
                types.add(type(value).__name__)
    warnings.warn(
        DroppedFieldWarning(
            f"{len(dropped)} metadata field(s) could not be represented in the HDF5 export "
            f"and were omitted: {', '.join(sorted(dropped))} (types: {', '.join(sorted(types))})."
        )
    )


def _pick_frame_axis(key_order: list, frames: list, requested) -> Optional[str]:
    if requested is None:
        return None
    if requested != "auto":
        return requested if requested in key_order else None
    for key in ("energy", "rel_time", "timestamp"):
        if key not in key_order:
            continue
        values = []
        ok = True
        for meta, _ in frames:
            value = meta.get(key)
            if not isinstance(value, (int, float, np.integer, np.floating)):
                ok = False
                break
            fvalue = float(value)
            if not np.isfinite(fvalue):
                ok = False
                break
            values.append(fvalue)
        if not ok or len(values) < 2:
            continue
        diffs = np.diff(values)
        if np.all(diffs > 0) or np.all(diffs < 0):
            return key
    return None


######### read #####################################################################################


def read(fname: str) -> Any:
    """Read an HDF5 file written by `write()` (or a compatible one) back into a uspy object.
    Always returns one of the five base types (`Image`, `LEEMImg`, `Line`, `ImageStack`,
    `LEEMStack`); a derived type such as `Waterfall` reloads as its parent."""
    with h5py.File(fname, "r") as f:
        entry = f["entry"]
        target_cls, kind = _resolve_target(entry)
        if kind == "object":
            return _read_object(entry, target_cls)
        return _read_stack(entry, target_cls)


def _read_object(entry: h5py.Group, target_cls: type) -> Any:
    from uspy.dataobject.line import Line

    data_grp = entry["data"]
    meta_grp = entry.get("metadata")

    if target_cls is Line:
        x = np.asarray(data_grp["x"][()], dtype=np.float64)
        y = np.asarray(data_grp["y"][()], dtype=np.float64)
        obj = target_cls(np.stack([x, y]))
        # x/y units live as attrs on the data-group datasets themselves (see
        # _write_line_data), not in /entry/metadata -- restore them the same way.
        x_unit = data_grp["x"].attrs.get("units")
        if x_unit:
            obj._units["x"] = x_unit
        y_unit = data_grp["y"].attrs.get("units")
        if y_unit:
            obj._units["y"] = y_unit
    else:
        image = np.asarray(data_grp["image"][()])
        obj = target_cls(image)

    if meta_grp is not None:
        for safe_key in meta_grp:
            true_key = _unsanitize_key(safe_key)
            dataset = meta_grp[safe_key]
            obj._meta[true_key] = _decode_value(dataset)
            unit = dataset.attrs.get("units")
            if unit:
                obj._units[true_key] = unit

    return obj


def _read_stack(entry: h5py.Group, target_cls: type) -> Any:
    from uspy.leem.base import LEEMImg, LEEMStack

    elem_cls = LEEMImg if target_cls is LEEMStack else _element_class_for(target_cls)

    data_grp = entry["data"]
    meta_grp = entry.get("metadata")
    signal_name = data_grp.attrs.get("signal", "image")
    images = np.asarray(data_grp[signal_name][()])
    n = images.shape[0]
    if n == 0:
        # uspy never writes an empty stack, so this means a foreign or truncated file.
        raise ValueError(
            f"{entry.name}/data/{signal_name} contains 0 frames -- this file holds no stack "
            "data (uspy never writes an empty stack; a foreign or truncated file?)."
        )

    columns, units_map = {}, {}
    if meta_grp is not None:
        for safe_key in meta_grp:
            if safe_key == "frame":
                continue
            true_key = _unsanitize_key(safe_key)
            dataset = meta_grp[safe_key]
            columns[true_key] = _decode_column(dataset)
            unit = dataset.attrs.get("units")
            if unit:
                units_map[true_key] = unit

    elements = []
    for i in range(n):
        element = elem_cls(np.asarray(images[i]))
        for key, values in columns.items():
            element._meta[key] = values[i]
            if key in units_map:
                element._units[key] = units_map[key]
        elements.append(element)

    stack = target_cls(elements)

    # Each element got its own private TimeOrigin from __init__; re-point them at the stack's
    # shared instance (no-op for ImageStack, which has none) so rel_time behaves normally.
    shared_origin = getattr(stack, "_time_origin", None)
    if shared_origin is not None:
        for element in stack:
            element._time_origin = shared_origin

    return stack


def _element_class_for(target_cls: type):
    from uspy.dataobject.image import Image

    return getattr(target_cls, "_type", Image)
