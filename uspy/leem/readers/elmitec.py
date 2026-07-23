"""
Reader for the Elmitec UKSOFT2001 ".dat" binary format (the default LEEM format).
"""
# pylint: disable=missing-docstring

from __future__ import annotations
from typing import Any, Union
from collections.abc import Iterable
import glob
from pathlib import Path

import numpy as np

from uspy.utility import parse_bytes, parse_cp1252_until_null
from uspy.leem.readers.base import LEEMReader

# Format: meta_key: (byte_position, encoding)
HEADER_ONE = {
    "_id": (0, "cp1252"),
    "_size": (20, "short"),
    "_version": (22, "short"),
    "_bitsperpix": (24, "short"),
    "width": (40, "short"),
    "height": (42, "short"),
    "_noimg": (44, "short"),
    "_recipe_size": (46, "short"),
}
HEADER_TWO = {
    "_isize": (0, "short"),
    "_iversion": (2, "short"),
    "_colorscale_low": (4, "short"),
    "_colorscale_high": (6, "short"),
    "time": (8, "time"),
    "_mask_xshift": (16, "short"),
    "_mask_yshift": (18, "short"),
    "_usemask": (20, "bool"),
    "_att_markupsize": (22, "short"),
    "_spin": (24, "short"),
}
ATTR_NAMES = {
    "timestamp": "time",
    "energy": "Start Voltage",
    "temperature": "Sample Temp.",
    "pressure1": "Gauge #1",
    "pressure2": "Gauge #2",
    "objective": "Objective",
    "emission": "Emission Cur.",
}
# Format: byte_position: (block_length, field_dict)
# where field_dict is formatted like the above HEADER_ONE and HEADER_TWO
VARIABLE_HEADER = {
    255: (0, None),  # stop byte
    100: (8, {"x_position": (0, "float"), "y_position": (4, "float")}),
    228: (
        8,
        {"x_position": (0, "float"), "y_position": (4, "float")},
    ),  # Image Version 8 of Elettra
    # Average Images: 0 means no averaging, 255 means sliding average
    104: (6, {"exposure": (0, "float"), "averaging": (4, "short")}),
    105: (0, {"_img_title": (0, "cp1252")}),
    242: (2, {"mirror_state": (0, "bool")}),
    243: (4, {"screen_voltage": (0, "float")}),
    244: (4, {"mcp_voltage": (0, "float")}),
}
UNIT_CODES = {
    "1": "V",
    "2": "mA",
    "3": "A",
    "4": "°C",
    "5": "K",
    "6": "mV",
    "7": "pA",
    "8": "nA",
    "9": "\xb5A",
    "B": "µm",
}


def parse_dat(fname: str, debug: bool = False) -> dict[str, Any]:
    """Parse a UKSOFT2001 file."""
    data = {}

    def parse_block(block, field_dict):
        for key, (pos, encoding) in field_dict.items():
            data[key] = parse_bytes(block, pos, encoding)
            if debug:
                print(f"\t{key} -> {data[key]}")

    with Path(fname).open("rb") as uk_file:
        parse_block(uk_file.read(104), HEADER_ONE)  # first fixed header

        if data["_recipe_size"] > 0:  # optional recipe
            data["recipe"] = parse_bytes(
                uk_file.read(data["_recipe_size"]), 0, "cp1252"
            )
            uk_file.seek(128 - data["_recipe_size"], 1)

        parse_block(uk_file.read(26), HEADER_TWO)  # second fixed header

        leemdata_version = parse_bytes(uk_file.read(2), 0, "short")
        if leemdata_version != 2:
            uk_file.seek(388, 1)
        bit = uk_file.read(1)[0]
        while bit != 255:
            if debug:
                print(bit)
            if bit in VARIABLE_HEADER:  # fixed byte codes
                block_length, field_dict = VARIABLE_HEADER[bit]
                buffer = uk_file.read(block_length)
                parse_block(buffer, field_dict)
                if debug:
                    print("\tknown")
            elif bit in (106, 107, 108, 109, 234, 235, 236, 237):  # varian pressures
                key = parse_cp1252_until_null(uk_file, debug)
                data[f"{key}_unit"] = parse_cp1252_until_null(uk_file, debug)
                data[key] = parse_bytes(uk_file.read(4), 0, "float")
                if debug:
                    print(f"\tknown: pressure {key} -> {data[key]}")
            elif bit in (110, 238):  # field of view
                fov_str = parse_cp1252_until_null(uk_file, debug)
                try:
                    data["fov"], data["preset"] = fov_str.split("\t")
                except ValueError:
                    data["fov"], data["preset"] = fov_str, ""
                data["fov_cal"] = parse_bytes(uk_file.read(4), 0, "float")
                if debug:
                    print(f"\tfov: {fov_str}\n\tfov_cal: {data['fov_cal']}")
            elif bit in (0, 1, 63, 66, 113, 128, 176, 216, 240, 232, 233):
                if debug:
                    print(f"unknown byte {bit}")
            elif bit:  # self-labelled stuff
                keyunit = parse_cp1252_until_null(uk_file, debug)
                # For some b, the string is empty. They should go in the tuple above.
                if not keyunit:
                    bit = uk_file.read(1)[0]
                    continue
                data[f"{keyunit[:-1]}_unit"] = UNIT_CODES.get(keyunit[-1], "")
                data[keyunit[:-1]] = parse_bytes(uk_file.read(4), 0, "float")
                if debug:
                    print(f"\tunknown: {keyunit[:-1]} -> {data[keyunit[:-1]]}")
            bit = uk_file.read(1)[0]

        size = data["width"] * data["height"]
        uk_file.seek(-2 * size, 2)
        image = np.fromfile(uk_file, dtype=np.uint16, sep="", count=size)
        image = np.array(image, dtype=np.float32)
        image = np.flipud(image.reshape((data["height"], data["width"])))
        data["image"] = image.astype(np.float32)

    for new, old in ATTR_NAMES.items():
        data[new] = data.pop(old, np.nan)
        data[f"{new}_unit"] = data.pop(f"{old}_unit", "")
    try:
        if data["averaging"] == 0:
            data["averaging"] = 1
        elif data["averaging"] == 255:
            data["averaging"] = 0
    except KeyError:
        pass
    data["energy_unit"] = "eV"

    return data


class ElmitecReader(LEEMReader):
    """Reads the Elmitec UKSOFT2001 ".dat" binary format used by UView. This is the
    default reader used when no other reader is selected."""

    name = "elmitec"
    extensions = (".dat",)

    def parse_image(self, source: str) -> dict[str, Any]:
        return parse_dat(source)

    def list_files(self, source: Union[str, Iterable]) -> list:
        if not isinstance(source, str):
            return []
        fnames = []
        if source.endswith(".dat"):  # .../path/*.dat
            fnames = sorted(glob.glob(f"{source}"))
        if not fnames:  # .../path/
            fnames = sorted(glob.glob(f"{source}*.dat"))
        if not fnames:  # .../path
            fnames = sorted(glob.glob(f"{source}/*.dat"))
        return fnames
