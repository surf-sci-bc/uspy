"""Manages database file and has import filters."""
# pylint: disable=logging-format-interpolation
# pylint: disable=global-statement

import os
import re
import sqlite3

import numpy as np


def parse_spectrum_file(fname):
    """Checks file extension and calls appropriate parsing method."""
    specdicts = []
    with open(fname, "r") as sfile:
        firstline = sfile.readline()
    if fname.split(".")[-1] == "txt":
        if "Region" in firstline:
            for specdict in parse_eistxt(fname):
                specdicts.append(specdict)
        if "[Info]" in firstline:
            specdicts.append(parse_arpestxt(fname))
        elif re.fullmatch(r"\d+\.\d+,\d+\n", firstline):
            specdicts.append(parse_simple_xy(fname, delimiter=","))
    elif fname.split(".")[-1] == "xy":
        if re.fullmatch(r"\d+\.\d+,\d+\n", firstline):
            delimiter = ","
        else:
            delimiter = None
        specdicts.append(parse_simple_xy(fname, delimiter=delimiter))
    if not specdicts:
        raise ValueError("Could not parse file '{}'".format(fname))
    return specdicts

def parse_simple_xy(fname, delimiter=None):
    """
    Parses the most simple x, y file with no header.
    """
    energy, intensity = np.genfromtxt(
        fname,
        delimiter=delimiter,
        unpack=True
    )
    specdict = {
        "energy": energy,
        "intensity": intensity,
        "energy_scale": "binding",
    }
    return specdict

def parse_eistxt(fname):
    """Splits Omicron EIS txt file."""
    splitregex = re.compile(r"^Region.*")
    skip_once_regex = re.compile(r"Layer.*")
    skip_regex = re.compile(r"^[0-9]+\s*False.*")
    split_eislines = []
    with open(fname, "br") as eisfile:
        for line in eisfile:
            line = line.decode("utf-8", "backslashreplace")
            if re.match(splitregex, line):
                split_eislines.append([])
                do_skip = False
            elif re.match(skip_regex, line):
                do_skip = True
            elif re.match(skip_once_regex, line):
                continue
            if not do_skip:
                split_eislines[-1].append(line)

    for data in split_eislines:
        energy, intensity = np.genfromtxt(
            data,
            skip_header=4,
            unpack=True
        )
        header = [line.split("\t") for line in data[:4]]
        specdict = {
            "energy": energy,
            "intensity": intensity,
            "energy_scale": "binding",
            "eis_region": int(header[1][0]),
            "sweeps": int(header[1][6]),
            "dwelltime": float(header[1][7]),
            "pass_energy": float(header[1][9]),
            "notes": header[1][12],
        }
        yield specdict


def parse_arpestxt(fname):
    """Reads a txt file obtained from Elettra's VUV beamline."""
    properties = {}
    energy = []
    intensity = []
    is_data = False
    databegin_regex = re.compile(r"^\[Data [0-9]+\]")
    datarow_regex = re.compile(
        r"^\s[0-9]+\.?[0-9]*(E\+)?[0-9]*\s*[0-9]+\.?[0-9]*(E\+)?[0-9]*\s*$"
    )
    with open(fname, "r") as afile:
        for line in afile:
            if "=" in line:
                key, value = line.split("=")[:2]
                properties[key.strip()] = value.strip()
            if re.match(databegin_regex, line):
                is_data = True
                continue
            if not is_data:
                continue
            if not re.match(datarow_regex, line):
                is_data = False
                continue
            estring, istring = line.strip().split()
            energy.append(float(estring))
            intensity.append(float(istring))
    specdict = {
        "energy": energy,
        "intensity": intensity,
        "energy_scale": properties["Energy Scale"].lower(),
        "sweeps": int(properties["Number of Sweeps"]),
        "dwelltime": 0.001 * float(properties["Step Time"]),
        "pass_energy": float(properties["Pass Energy"]),
        "name": properties["Spectrum Name"],
        "notes": properties["Comments"],
        "time": f"{properties['Date']}, {properties['Time']}",
    }
    return specdict


PHOTON_ENERGIES = {
    "Al": 1486.3,
    "Mg": 1253.4
}
def get_element_info(element, source):
    """Yields tuples containing the element name, orbital, its binding energy and rsf.
    For Auger peaks, RSF is zero."""
    try:
        photon_energy = float(source)
    except ValueError:
        photon_energy = PHOTON_ENERGIES.get(source, 0)
    dbfname = os.path.dirname(__file__) + "/rsf.db"
    print(dbfname)
    with sqlite3.connect(dbfname) as database:
        cursor = database.cursor()
        sql = """
            SELECT IsAuger, Orbital, BE, RSF
            FROM Peak
            WHERE Element=? AND (Source=? OR Source="Any")
        """
        cursor.execute(sql, (element.title(), source))
        for isauger, orbital, energy, rsf in cursor.fetchall():
            if isauger == 1.0:
                binding_energy = photon_energy - energy
                orbital = orbital.upper()
            else:
                binding_energy = energy
            yield element.title(), orbital, binding_energy, rsf


def export_txt(fname, spectrum):
    """Export given spectra and everything that belongs to it as txt."""
    column_stack = [
        spectrum.energy,
        spectrum.intensity,
    ]
    name = re.sub(r"\s+", "_", spectrum.name)
    header = "{:_<15}_Energy\t{:_<14}_intensity\t".format(name, name)
    if spectrum.background.any():
        column_stack.append(spectrum.background)
        header += "{:_<13}_background\t".format(name)
    if spectrum.fit.any():
        column_stack.append(spectrum.fit)
        header += "{:_<20}_fit\t".format(name)
    for peak in spectrum.peaks:
        column_stack.append(peak.intensity)
        peak_name = re.sub(r"\s+", "_", peak.name)
        header += "{:_<12}_{:_<3}_peakint\t".format(name, peak_name)
    data = np.column_stack(column_stack)
    np.savetxt(fname, data, delimiter="\t", header=header)


def export_params(fname, spectrum):
    """Export given spectra and everything that belongs to it as txt."""
    params = ("area", "fwhm", "position", "alpha", "beta", "gamma")
    header = "\t".join(params)
    data = []
    for peak in spectrum.peaks:
        row = []
        row.append(peak.label)
        row.append(peak.shape)
        for par in params:
            value = peak.get_constraints(par)["value"]
            row.append(value)
        data.append(row)
    np.savetxt(fname, data, delimiter="\t", header=header, fmt="%s")
