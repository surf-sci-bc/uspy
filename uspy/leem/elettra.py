import glob
import os
from pathlib import Path
from typing import Any, Iterable, Union
import numpy as np

import pandas as pd
import uspy.leem.base as leembase
import tifffile


class LEEMImg(leembase.LEEMImg):
    """_summary_

    Parameters
    ----------
    leembase : _type_
        _description_
    """

    @property
    def binding_energy(self):
        return self.photon_energy - self.energy

    @property
    def kinetic_energy(self):
        return self.energy

    def parse(self, source: str) -> dict[str, Any]:
        """Parses the File Format of the Elettra Nanospectrocopy beamline

        Elettra saves either .tif files or .dat files. If .tif files are read,
        all metadata is extracted from the metadata files in the same directory.
        If a .dat file is used, only the beamline metadata is extracted from the
        metadata files. The rest is extracted directly from the .dat file

        Parameters
        ----------
        source : str
            Filename of the data file

        Returns
        -------
        dict[str, Any]
            dictionary containing the data and metadata
        """

        p = Path(source).parent
        self._source = source
        if source.endswith(".tif"):
            # read image
            image = tifffile.imread(source)
            idict = {"image": image.astype(np.float32)}

            fname = glob.glob(f"{p}/*camera.txt")
            idict.update(read_camera_metadata(fname[0]))

            fname = glob.glob(f"{p}/*microscope.txt")
            idict.update(read_microscope_metadata(fname[0]))

            # when using .tif files no timestamp is saved, so it is assumed the modification date
            # of the file is the timestamp, so take care

            idict["timestamp"] = os.path.getmtime(source)

        elif source.endswith(".dat"):
            idict = leembase.parse_dat(source)

        fname = glob.glob(f"{p}/*beamline.txt")
        idict.update(read_beamline_metadata(fname[0]))

        for new, old in leembase.ATTR_NAMES.items():
            idict[new] = idict.pop(old, np.nan)
        # idict[f"{new}_unit"] = idict.pop(f"{old}_unit", "")

        return idict


class LEEMStack(leembase.LEEMStack):
    """_summary_

    Parameters
    ----------
    leembase : _type_
        _description_
    """

    _type = LEEMImg

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # When using the Elettra .tif fileformat some metadata that is changing between images are
        # not stored as single metadata files, but in a seperate "tracefile", that has to be read
        # after the individual images are constructed

        source = kwargs.pop("source", args[0])

        # the Trancefile should only be read, if the stack is directly created from a list of
        # filenames and not from Objects, because then slices of an existing stack would trigger
        # reading the tracefile, which would result in bad metadata

        if (
            isinstance(source, list)
            and isinstance(source[0], str)
            or isinstance(source, str)
        ):
            fname = glob.glob(f"{Path(self[0]._source).parent}/*trace.txt")
            if fname:  # fname is empty list there is no tracefile
                fname = fname[0]
                df = pd.read_csv(fname, delimiter=" ")

                # pylint: disable=no-member
                self.energy = df.iloc[:, 1].to_list()
                self.objective = df.iloc[:, 2].to_list()
                self.mesh = df.iloc[:, 3].to_list()
                self.beam_current = df.iloc[:, 4].to_list()
                self.temperature = df.iloc[:, 5].to_list()

    def _split_source(self, source: Union[str, Iterable]) -> list:
        """_summary_

        Parameters
        ----------
        source : Union[str, Iterable]
            _description_

        Returns
        -------
        list
            _description_
        """

        if isinstance(source, str):
            fnames = []

            # First check for .dat files. We are explicitly not using super() here, because if no
            # .dat files are present, ImageStack is triggered as it's parents which would then find
            # the .tif files that are always present.
            # .dat Files are prefered because they save the data in a more robust way, e.g timestamp

            if source.endswith(".dat"):  # .../path/*.dat
                fnames = sorted(glob.glob(f"{source}"))
            if not fnames:  # .../path/
                fnames = sorted(glob.glob(f"{source}*.dat"))
            if not fnames:  # .../path
                fnames = sorted(glob.glob(f"{source}/*.dat"))

            if source.endswith(".tif"):  # .../path/*.tif
                fnames = sorted(glob.glob(f"{source}"))
            if not fnames:  # .../path/
                fnames = sorted(glob.glob(f"{source}*.tif"))
            if not fnames:  # .../path
                fnames = sorted(glob.glob(f"{source}/*.tif"))

            if fnames:
                return fnames

        # If source is not a string e.g. a list of filenames it has to be handled by the parents
        return super()._split_source(source)

    def _single_construct(self, source: Any) -> LEEMImg:
        """Is identical to base.LEEMStack._single_construct, but
        calls the elettra.LEEMImg class
        """
        return LEEMImg(source, time_origin=self._time_origin)


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
