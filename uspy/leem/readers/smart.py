"""
Reader for the SMART photoemission electron microscope at the SMART/BESSY beamline
(Helmholtz-Zentrum Berlin).

SMART data is saved as **HDF4** files (not HDF5 -- `h5py` cannot read them) via the HDF4
"Scientific Data Set" (SD) API. Each file holds a single 3-D dataset named "Range0"
(int16, shape `(n_frames, height, width)`); a plain single-image acquisition is just a
1-frame file. There are no per-dataset SD attributes -- instead, all acquisition metadata is
written as one free-text HDF4 *file description* annotation ("KEY = VALUE;" pairs,
several per line), alongside a *file label* annotation giving the original acquisition path
(used here only to confirm the file is SMART data).

Unlike Elmitec/Elettra, a stack (e.g. an I(V) energy scan, "SCAN_MODE = EDC") is *not* one
file per frame -- the whole scan is written into a single multi-frame file, with the energy
axis given by `MICRO_START_1`/`MICRO_END_1`/`POINTS_1` in the file description. A stack
element is therefore addressed as a `(path, frame_index)` tuple rather than a bare path.

Very long scans that don't fit in a single file are split by SMART into "<base>.hdf",
"<base>_1.hdf", "<base>_2.hdf", ... -- note the *first* part is the un-suffixed "<base>.hdf";
only the continuations get a "_N" suffix. This reader does *not* auto-discover and concatenate
those siblings -- `list_files()` only expands the frames of the file(s) it is actually given
(a single file, an explicit list, a glob, or a folder of files). To load a whole split scan as
one stack, pass the **basename** ("<base>", no extension) or a "<base>*.hdf" glob: `natsort`
sorts the un-suffixed base file before its numbered siblings (since "." < "_"), so the parts
concatenate in acquisition order. Passing the exact "<base>.hdf" path loads only that one file.

Reading files uses the optional `pyhdf` package (which wraps the HDF4 C library) when it is
installed -- it is the preferred, battle-tested backend. `pyhdf` is awkward to install on
macOS/Windows, though, so it is fully optional: when it is absent, a small built-in pure-Python
HDF4 reader (`uspy.leem.readers._hdf4`) is used as a fallback. The fallback covers the
uncompressed contiguous/linked-block storage that SMART actually writes; a (hypothetical)
compressed or chunked file would still need `pyhdf` (`conda install -c conda-forge pyhdf`, then
`pip install uspy[smart]`).
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
    """Return `(HDF, SD, AN_FILE_LABEL, AN_FILE_DESC)` from pyhdf if it is importable, else
    `None`. pyhdf is the preferred backend but optional: when it is missing, the callers fall
    back to the built-in pure-Python HDF4 reader (`uspy.leem.readers._hdf4`). Done lazily (and
    cached) so `SmartReader` imports/registers whether or not pyhdf is installed."""
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
    electron energy, "MONO" for the photon/monochromator energy). `negative_means_unused` is
    for MONO only: SMART writes -1/-1 there for plain LEEM (no monochromator) acquisitions --
    it must NOT be applied to MICRO, since a genuinely negative start voltage is a valid
    setting in mirror-electron-microscopy mode."""
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
        n_frames = _n_frames(fd)

        energy = _energy_axis(fd, "MICRO", n_frames)
        photon_energy = _energy_axis(fd, "MONO", n_frames, negative_means_unused=True)

        idict = {
            "image": image,
            "height": image.shape[0],
            "width": image.shape[1],
            "energy": energy[index] if index < len(energy) else np.nan,
            "energy_unit": "eV",
            "photon_energy": photon_energy[index] if index < len(photon_energy) else np.nan,
            "timestamp": _smart_timestamp(fd.get("DATE", ""), fd.get("TIME", "")),
            "scan_mode": fd.get("SCAN_MODE", ""),
            "comment": fd.get("COMMENT", ""),
            "ident": fd.get("IDENT", ""),
        }
        try:
            # SMART records exposure ("INTERATION_TIME") in ms; uspy stores it in seconds.
            idict["exposure"] = float(fd["INTERATION_TIME"]) / 1000.0
            idict["exposure_unit"] = "s"
        except (KeyError, ValueError):
            pass
        try:
            idict["averaging"] = float(fd["AVERAGES"])
        except (KeyError, ValueError):
            pass
        # Not present in SMART metadata: temperature, pressure, fov, objective, emission --
        # left at LEEMImg's own defaults (nan / "Unknown FoV").
        return idict

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
