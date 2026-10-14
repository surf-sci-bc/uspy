"""
Reader for the Nanospectroscopy beamline at Elettra.

Elettra saves either ".tif" files (with metadata in sidecar ".txt" files and, for stacks,
a shared "tracefile") or ".dat" files (the Elmitec binary format, with beamline metadata
in a sidecar ".txt" file).
"""
# pylint: disable=missing-docstring

from __future__ import annotations
from typing import Any, Union
from collections.abc import Iterable
import glob
import os
from pathlib import Path

import numpy as np
import natsort
import pandas as pd
import tifffile

from uspy.leem.readers.base import LEEMReader
from uspy.leem.readers.elmitec import parse_dat, ATTR_NAMES

# Per-image metadata that varies across a stack of Elettra ".tif" files is not saved in
# per-image sidecars but in a single shared "*trace.txt" file. Maps a trace column prefix
# to the (attribute name, unit) it should be stored as.
TRACE_ATTRS = {
    "time": ("timestamp", "s"),
    "STV": ("energy", "eV"),
    "MOBJ": ("objective", "mA"),
    "Mesh": ("mesh", "V"),
    "BeamCurrent": ("beam_current", "mA"),
    "temperature": ("temperature", "°C"),
    "emiss": ("emission", "µA"),
    "PMCH": ("pressure1", "mBar"),
}


class ElettraReader(LEEMReader):
    """Reads data from the Nanospectroscopy beamline at Elettra."""

    name = "elettra"
    extensions = (".dat", ".tif", ".tiff")
    meta_defaults = {"photon_energy": np.nan, "energy_offset": 0}
    unit_defaults = {"photon_energy": "eV"}

    def _parse(self, source: str, read_image: bool) -> dict[str, Any]:
        """Shared implementation of `parse_image`/`parse_metadata`. Elettra's per-image
        metadata is stored independently of the pixel block, so `read_image=False` skips
        the actual pixel read while returning the same metadata `parse_image` would."""
        folder = Path(source).parent

        if source.endswith((".tif", ".tiff")):
            if read_image:
                idict = {"image": tifffile.imread(source).astype(np.float32)}
            else:
                idict = {}

            fname = glob.glob(f"{folder}/*camera.txt")
            idict.update(read_camera_metadata(fname[0]))

            fname = glob.glob(f"{folder}/*microscope.txt")
            idict.update(read_microscope_metadata(fname[0]))
            # .tif sidecars use the raw Elmitec-style names (e.g. "Start Voltage"),
            # unlike parse_dat below, which already renames them internally -- so this
            # is the one place ATTR_NAMES still needs to be applied.
            needs_rename = True
        elif source.endswith(".dat"):
            # already renamed via ATTR_NAMES internally
            idict = parse_dat(source, read_image=read_image)
            needs_rename = False
        else:
            raise ValueError(f"{type(self).__name__} cannot parse {source}")

        try:
            fname = glob.glob(f"{folder}/*beamline.txt")
            idict.update(read_beamline_metadata(fname[0]))
        except IndexError:
            pass

        if needs_rename:
            for new, old in ATTR_NAMES.items():
                idict[new] = idict.pop(old, np.nan)
            # .tif files carry no timestamp; prefer a same-basename ".dat" sidecar's real
            # embedded one, else fall back to file mtime (only approximate -- can be wrong
            # if files were copied/extracted after the fact). Overwritten later by a
            # tracefile's "time" column, if present, in post_process_stack().
            dat_source = str(Path(source).with_suffix(".dat"))
            if os.path.exists(dat_source):
                idict["timestamp"] = parse_dat(dat_source, read_image=False)["timestamp"]
            else:
                idict["timestamp"] = os.path.getmtime(source)

        return idict

    def parse_image(self, source: str) -> dict[str, Any]:
        return self._parse(source, read_image=True)

    def parse_metadata(self, source: str) -> dict[str, Any]:
        """Metadata-only counterpart to `parse_image`, used to populate a virtual
        (`defer=True`) `LEEMImg`'s metadata; avoids reading every frame's pixels just for
        metadata."""
        return self._parse(source, read_image=False)

    def list_files(self, source: Union[str, Iterable]) -> list:
        if not isinstance(source, str):
            return []

        # Elettra numbers files with 3 digits, so alphanumeric sorting fails past 999
        # files -- natsort is needed. .dat is preferred over .tif (more robust, e.g. timestamp).
        fnames = []
        if source.endswith(".dat"):  # .../path/*.dat
            fnames = natsort.natsorted(glob.glob(f"{source}"))
        if not fnames:  # .../path/
            fnames = natsort.natsorted(glob.glob(f"{source}*.dat"))
        if not fnames:  # .../path
            fnames = natsort.natsorted(glob.glob(f"{source}/*.dat"))

        if source.endswith((".tif", ".tiff")):  # .../path/*.tif(f)
            fnames = natsort.natsorted(glob.glob(f"{source}"))
        if not fnames:  # .../path/
            fnames = natsort.natsorted(
                glob.glob(f"{source}*.tif") + glob.glob(f"{source}*.tiff")
            )
        if not fnames:  # .../path
            fnames = natsort.natsorted(
                glob.glob(f"{source}/*.tif") + glob.glob(f"{source}/*.tiff")
            )

        return fnames

    def post_process_stack(self, stack) -> None:
        # Only look for a tracefile when the stack was actually built from file sources
        # (not from a list/slice of already-constructed images), and only once a source
        # file can be located at all (e.g. not for in-memory-only stacks).
        sources = stack.sources
        if not sources or not isinstance(sources[0], str):
            return

        fname = glob.glob(f"{Path(sources[0]).parent}/*trace.txt")
        if not fname:  # no tracefile -- e.g. a stack of .dat files
            return
        df = pd.read_csv(fname[0], delimiter=" ")
        if len(df) != len(stack):
            raise ValueError(
                f"Tracefile {fname[0]} has {len(df)} rows but the stack has {len(stack)} "
                "images; load the complete set of images (and slice the stack afterwards)."
            )

        has_time_column = False
        for header, vals in df.items():
            attr = header.split("_")[0]
            name, unit = TRACE_ATTRS.get(attr, (attr, "a.u."))
            setattr(stack, name, vals)
            stack.set_unit(name, unit)
            if attr == "time":
                has_time_column = True

        if has_time_column:
            # the tracefile's own timestamps are logged in ms; per-image timestamps
            # from parse_image() (real epoch seconds) are left untouched otherwise.
            stack.timestamp = [x / 1000.0 for x in stack.timestamp]
        # time_origin is seeded by the caller after post_process_stack runs, unless an
        # explicit one was passed in -- nothing to (re-)set here.


def read_camera_metadata(fname: str) -> dict:
    """Reads the Camera metadata of the Elettra LEEM file format

    Parameters
    ----------
    fname : str
        Filename of camera file

    Returns
    -------
    dict
        Dictionary containing metadata
    """
    with open(fname, "r", encoding="latin-1") as f:
        for line in f:
            if "width" in line:
                width, height = line.split(" ")[1].split("x")
            elif "exposure" in line:
                # Elettra saves exposure in ms, but Elmitec in seconds
                exp = float(line.split(" ")[2]) / 1000
            elif "averaging" in line:
                avg = int(line.split(" ")[1])

    return {
        "width": int(width),
        "height": int(height),
        "exposure": exp,
        "averaging": avg,
    }


def read_microscope_metadata(fname: str) -> dict:
    """Reads the Microscope metadata of the Elettra LEEM file format

    Parameters
    ----------
    fname : str
        Filename of microcope file

    Returns
    -------
    dict
        Dictionary containing metadata
    """

    idict = {}
    with open(fname, "r", encoding="latin-1") as f:

        for line in f:
            key, val = line.strip("\n ").split(", ")
            if key == "disabled" or key == "invalid":
                continue
            if key == "Operation Mode":
                key = "fov"
            elif key == "Emission":
                key = "Emission Cur."

            if key != "fov":
                val = float(val)

            idict[key] = val

    return idict


def read_beamline_metadata(fname: str) -> dict:
    """Reads the beamline metadata of the Elettra LEEM file format

    The beamline file contains a lot of information. Currently only the photon energy
    is read because its the most important.

    Parameters
    ----------
    fname : str
        Filename of beamline file

    Returns
    -------
    dict
        Dictionary containing metadata
    """

    with open(fname, "r", encoding="latin-1") as f:
        idict = {}
        for line in f:
            if "Monochromator energy" in line:
                idict["photon_energy"] = float(line.split(",")[3])
                idict["photon_energy_unit"] = "eV"
            elif "Mesh" in line:
                idict["mesh"] = float(line.split(",")[3])
                idict["mesh_unit"] = line.split(",")[4]

    return idict
