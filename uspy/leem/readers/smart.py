"""
Reader for the SMART photoemission electron microscope at the SMART/BESSY beamline
(Helmholtz-Zentrum Berlin).

SMART data is **HDF4** (not HDF5 -- `h5py` cannot read it). Each file holds one 3-D SD
dataset "Range0" (int16, shape `(n_frames, height, width)`); a single image is a 1-frame
file. All metadata is one free-text "KEY = VALUE;" file-description annotation, no
per-dataset attributes.

Unlike Elmitec/Elettra, a whole stack/scan (e.g. an I(V) scan, "SCAN_MODE = EDC") is one
multi-frame file, with the energy axis given by `MICRO_START_1`/`MICRO_END_1`/`POINTS_1`.
A stack element is addressed as a `(path, frame_index)` tuple, not a bare path.

Long scans are split by SMART into "<base>.hdf", "<base>_1.hdf", "<base>_2.hdf", ... .
`list_files()` does not auto-discover/concatenate siblings; pass the basename ("<base>",
no extension) or a "<base>*.hdf" glob to load a whole split scan as one stack (`natsort`
sorts "." before "_", so parts concatenate in order). The exact "<base>.hdf" path loads
only that one file.

Uses the optional `pyhdf` package (wraps the HDF4 C library) when installed; otherwise
falls back to the pure-Python `uspy.leem.readers._hdf4` reader, which only covers
uncompressed contiguous/linked-block storage (all SMART actually writes) -- a compressed
or chunked file needs `pyhdf` (`conda install -c conda-forge pyhdf`, then
`pip install uspy[smart]`).

SMART records only one wall-clock time per file ("DATE"/"TIME", acquisition start); with
no true per-frame clock, `_metadata_dict` estimates per-frame `timestamp` as
`start + frame_index * exposure` -- an estimate, not a measurement.
"""
# pylint: disable=missing-docstring

from __future__ import annotations
from typing import Any, Union
from collections.abc import Iterable
from datetime import datetime
from functools import lru_cache
import glob
import os
import re

import numpy as np
import natsort

from uspy.leem.readers.base import LEEMReader
from uspy.leem.readers._hdf4 import Hdf4File

# The name of the (only) Scientific Data Set in a SMART HDF4 file.
SDS_NAME = "Range0"

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "mai": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "okt": 10, "nov": 11, "dec": 12, "dez": 12,
}

# "KEY = VALUE;" pairs, possibly several per line (e.g. "IDENT = SMART;    VERSION = 0;").
_FD_ENTRY_RE = re.compile(r"([A-Za-z0-9_]+)\s*=\s*([^;]*);")


@lru_cache(maxsize=1)
def _pyhdf():
    """`(HDF, SD, AN_FILE_LABEL, AN_FILE_DESC)` from pyhdf if importable, else `None`
    (callers then fall back to `uspy.leem.readers._hdf4`). Lazy so `SmartReader` imports
    whether or not pyhdf is installed."""
    # pylint: disable=import-outside-toplevel,invalid-name
    try:
        from pyhdf.HDF import HDF
        from pyhdf.SD import SD
        from pyhdf.AN import AN_FILE_LABEL, AN_FILE_DESC
    except ImportError:
        return None
    return HDF, SD, AN_FILE_LABEL, AN_FILE_DESC


@lru_cache(maxsize=8)
def _hdf4_fallback(path: str, _mtime: float) -> Hdf4File:
    """A parsed pure-Python `Hdf4File` for `path`, cached on `(path, mtime)` so a stack's file
    structure (DD table, linked-block layout) is parsed once, not once per frame."""
    return Hdf4File(path)


def _parse_fd_text(text: str) -> dict:
    """Parse a SMART HDF4 file-description annotation into a flat {key: str_value} dict."""
    return {key: value.strip() for key, value in _FD_ENTRY_RE.findall(text)}


@lru_cache(maxsize=128)
def _read_smart_annotations(path: str, _mtime: float) -> tuple:
    """Return `(file_label, file_description_dict)` for `path`. Cached on `(path, mtime)` so
    a multi-frame stack file's metadata is only read/parsed once, not once per frame."""
    pyhdf = _pyhdf()
    if pyhdf is None:  # pure-Python fallback
        hdf4 = _hdf4_fallback(path, _mtime)
        return hdf4.file_label(), _parse_fd_text(hdf4.file_description())
    hdf_cls, _sd_cls, an_file_label, an_file_desc = pyhdf
    hdf_file = hdf_cls(path)
    try:
        annot = hdf_file.astart()
        try:
            n_file_label, n_file_desc, _n_data_label, _n_data_desc = annot.fileinfo()
            label = ""
            if n_file_label:
                ann = annot.select(0, an_file_label)
                label = ann.annotation()
                ann.endaccess()
            desc = ""
            if n_file_desc:
                ann = annot.select(0, an_file_desc)
                desc = ann.annotation()
                ann.endaccess()
        finally:
            annot.end()
    finally:
        hdf_file.close()
    return label, _parse_fd_text(desc)


def _read_smart_metadata(path: str) -> dict:
    """The file-description ("KEY = VALUE;") dict for `path`."""
    _label, meta = _read_smart_annotations(path, os.path.getmtime(path))
    return meta


@lru_cache(maxsize=128)
def _read_smart_dims(path: str, _mtime: float) -> tuple:
    """``(height, width)`` of the "Range0" dataset's frames, from the file header only --
    no pixel data touched. Cached on `(path, mtime)` since a multi-frame scan file calls
    this once per frame."""
    pyhdf = _pyhdf()
    if pyhdf is None:  # pure-Python fallback -- already parses dims, not pixels, in __init__
        return _hdf4_fallback(path, _mtime).shape[1:]
    _hdf_cls, sd_cls, _an_file_label, _an_file_desc = pyhdf
    sd_file = sd_cls(path)
    try:
        sds = sd_file.select(SDS_NAME)
        try:
            _name, _rank, dims, _type, _nattrs = sds.info()  # no data read
        finally:
            sds.endaccess()
    finally:
        sd_file.end()
    return tuple(dims[1:])  # dims is (n_frames, height, width)


def _n_frames(fd: dict) -> int:
    """Number of frames stored in a file: 1 for a single image, or the scan's point count
    (e.g. 161 for an I(V) energy scan) -- both come from the same "POINTS_1" field."""
    try:
        return int(float(fd["POINTS_1"]))
    except (KeyError, ValueError):
        return 1


def _energy_axis(
    fd: dict, prefix: str, n_frames: int, negative_means_unused: bool = False
) -> np.ndarray:
    """Linear energy axis from "<prefix>_START_1"/"<prefix>_END_1" ("MICRO" for the imaging
    electron energy, "MONO" for the photon/monochromator energy). `negative_means_unused`
    is MONO-only: SMART writes -1/-1 there when no monochromator is used; MICRO can be
    genuinely negative (mirror-electron-microscopy mode)."""
    try:
        start = float(fd[f"{prefix}_START_1"])
        end = float(fd[f"{prefix}_END_1"])
    except (KeyError, ValueError):
        return np.full(n_frames, np.nan)
    if negative_means_unused and start < 0 and end < 0:
        return np.full(n_frames, np.nan)
    if n_frames <= 1:
        return np.array([start], dtype=float)
    return np.linspace(start, end, n_frames)


def _smart_timestamp(date_str: str, time_str: str) -> float:
    """Parse SMART's "9.Apr 2025" + "9:32:01" into a Unix timestamp. The files carry no
    timezone information, so this is naive local time, like Elmitec/Elettra timestamps."""
    try:
        day_str, rest = date_str.split(".", 1)
        month_str, year_str = rest.split()
        month = _MONTHS[month_str.strip().lower()[:3]]
        hour, minute, second = (int(x) for x in time_str.split(":"))
        return datetime(int(year_str), month, int(day_str), hour, minute, second).timestamp()
    except (ValueError, KeyError):
        return np.nan


def _metadata_dict(fd: dict, index: int, height: int, width: int) -> dict[str, Any]:
    """Fields shared by `SmartReader.parse_image`/`parse_metadata`, everything except the
    "image" data key. `height`/`width` come from the real pixel array in the former, a
    pixel-free header read (`_read_smart_dims`) in the latter."""
    n_frames = _n_frames(fd)
    energy = _energy_axis(fd, "MICRO", n_frames)
    photon_energy = _energy_axis(fd, "MONO", n_frames, negative_means_unused=True)

    try:
        # SMART records exposure ("INTERATION_TIME") in ms; uspy stores it in seconds.
        exposure = float(fd["INTERATION_TIME"]) / 1000.0
    except (KeyError, ValueError):
        exposure = np.nan

    idict = {
        "height": height,
        "width": width,
        "energy": energy[index] if index < len(energy) else np.nan,
        "energy_unit": "eV",
        "photon_energy": photon_energy[index] if index < len(photon_energy) else np.nan,
        # SMART records only one DATE/TIME (acquisition start) per file, so per-frame
        # timestamp is estimated as start + index * exposure -- not a real measurement.
        "timestamp": _smart_timestamp(fd.get("DATE", ""), fd.get("TIME", ""))
        + (index * exposure if index and np.isfinite(exposure) else 0.0),
        "scan_mode": fd.get("SCAN_MODE", ""),
        "comment": fd.get("COMMENT", ""),
        "ident": fd.get("IDENT", ""),
    }
    if np.isfinite(exposure):
        idict["exposure"] = exposure
        idict["exposure_unit"] = "s"
    try:
        idict["averaging"] = float(fd["AVERAGES"])
    except (KeyError, ValueError):
        pass
    # Not present in SMART metadata: temperature, pressure, fov, objective, emission --
    # left at LEEMImg's own defaults (nan / "Unknown FoV").
    return idict


def _read_smart_frame(path: str, index: int) -> np.ndarray:
    """Read a single frame (as `(height, width)` float32) from the "Range0" dataset."""
    pyhdf = _pyhdf()
    if pyhdf is None:  # pure-Python fallback
        frame = _hdf4_fallback(path, os.path.getmtime(path)).read_frame(index)
        return np.asarray(frame, dtype=np.float32)
    _hdf_cls, sd_cls, _an_file_label, _an_file_desc = pyhdf
    sd_file = sd_cls(path)
    try:
        sds = sd_file.select(SDS_NAME)
        try:
            frame = sds[index, :, :]
        finally:
            sds.endaccess()
    finally:
        sd_file.end()
    return np.asarray(frame, dtype=np.float32)


class SmartReader(LEEMReader):
    """Reads data from the SMART photoemission microscope at BESSY (HDF4 files). A stack
    element is addressed as a `(path, frame_index)` tuple since a whole scan (e.g. an I(V)
    energy series) is stored in a single multi-frame file rather than one file per frame."""

    name = "smart"
    extensions = (".hdf",)
    # Beamline pattern (see uspy.leem.readers.elettra): binding_energy is only meaningful
    # once photon_energy is known; default to nan (SMART LEEM acquisitions have no
    # monochromator energy -- see MONO_START_1/MONO_END_1 handling in parse_image below).
    meta_defaults = {"photon_energy": np.nan, "energy_offset": 0}
    unit_defaults = {"photon_energy": "eV"}

    def handles(self, source: Any) -> bool:
        if isinstance(source, str):
            return source.endswith(self.extensions)
        return (
            isinstance(source, tuple)
            and len(source) == 2
            and isinstance(source[0], str)
            and source[0].endswith(self.extensions)
            and isinstance(source[1], (int, np.integer))
        )

    def parse_image(self, source: Union[str, tuple]) -> dict[str, Any]:
        path, index = (source, 0) if isinstance(source, str) else source
        fd = _read_smart_metadata(path)
        image = _read_smart_frame(path, index)
        idict = _metadata_dict(fd, index, image.shape[0], image.shape[1])
        idict["image"] = image
        return idict

    def parse_metadata(self, source: Union[str, tuple]) -> dict[str, Any]:
        """Metadata-only counterpart to `parse_image`, for the lazy stack backend
        (`uspy.dataobject.lazy`): sources frame dimensions from the file header
        (`_read_smart_dims`) instead of reading pixel data."""
        path, index = (source, 0) if isinstance(source, str) else source
        fd = _read_smart_metadata(path)
        height, width = _read_smart_dims(path, os.path.getmtime(path))
        return _metadata_dict(fd, index, height, width)

    def list_files(self, source: Union[str, Iterable]) -> list:
        if not isinstance(source, str):
            return []
        fnames = []
        if source.endswith(".hdf"):  # .../path/*.hdf, or an exact single file
            fnames = natsort.natsorted(glob.glob(source))
        if not fnames:  # .../path/
            fnames = natsort.natsorted(glob.glob(f"{source}*.hdf"))
        if not fnames:  # .../path
            fnames = natsort.natsorted(glob.glob(f"{source}/*.hdf"))

        sources = []
        for fname in fnames:
            fd = _read_smart_metadata(fname)
            sources.extend((fname, i) for i in range(_n_frames(fd)))
        return sources
