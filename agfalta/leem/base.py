"""
Basic classes for Elmitec LEEM ".dat"-file parsing and data visualization.
"""
# pylint: disable=missing-docstring

from __future__ import annotations
from typing import Any, Union, Optional
from collections.abc import Iterable
from numbers import Number
from datetime import datetime
import glob
from pathlib import Path

import numpy as np

from agfalta.utility import parse_bytes, parse_cp1252_until_null
from agfalta.base import DataObjectStack, Image


class TimeOrigin:
    def __init__(self, value: Optional[Union[Number,TimeOrigin]] = None):
        if isinstance(value, TimeOrigin):
            value = value.value
        elif value is None:
            value = np.nan
        self.value = float(value)


class LEEMImg(Image):
    """
    LEEM image that exposes metadata as attributes.
    Default attributes are:
    - image: numpy array containing the image
    - energy (in eV), temperature (in °C), fov (in µm), timestamp (in s)
    """
    _meta_defaults = {
        "energy": np.nan,
        "temperature": np.nan,
        "fov": np.nan,
        "timestamp": np.nan,
    }
    _unit_defaults = {
        "energy": "eV",
        "temperature": "°C",
        "pressure": "Torr",
        "objective": "mA",
        "fov": "µm",
        "timestamp": "s",
        "exposure": "s",
        "dose": "L",
        "emission": "µA",
        "resolution": "µm/px",
        "x_position": "µm",
        "y_position": "µm",
    }

    def __init__(self, *args, time_origin: Union[TimeOrigin,Number] = None,
                 **kwargs) -> None:
        if not isinstance(time_origin, TimeOrigin):
            time_origin = TimeOrigin(time_origin)
        super().__init__(*args, **kwargs)
        self._time_origin = time_origin   # is a list so it can be mutable

    def parse(self, source: str) -> dict[str, Any]:
        if isinstance(source, Image):
            self._source = None
            return dict(source.meta, **source.data)
        self._source = source
        return parse_dat(source)

    @property
    def pressure(self) -> Number:
        for k in ("pressure", "pressure1", "pressure2", "MCH", "PCH"):
            pressure_ = self._meta.get(k, np.nan)
            if not np.isnan(pressure_):
                return pressure_
        return np.nan
    @pressure.setter
    def pressure(self, value: Number) -> None:
        self._meta["pressure"] = value

    @property
    def fov(self) -> Union[Number,str]:
        fov_ = self._meta.get("fov", np.nan)
        if fov_ < 0:
            self._units["fov"] = ""
            return "LEED"
        else:
            self._units["fov"] = "µm"
        return fov_
    @fov.setter
    def fov(self, value: Union[Number,str]) -> None:
        if isinstance(value, str):
            assert value == "LEED"
            self._units["fov"] = ""
        else:
            self._units["fov"] = "µm"
        self._meta["FoV"] = value

    @property
    def resolution(self) -> Number:
        fov_ = self._meta.get("fov", np.nan)
        if fov_ < 0:
            fov_ = np.nan
        return fov_ / self._meta.get("fov_cal", np.nan)

    @property
    def time_origin(self) -> Number:
        return self._time_origin.value
    @time_origin.setter
    def time_origin(self, value: Number) -> None:
        self._time_origin.value = value

    @property
    def rel_time(self) -> Number:
        return self.timestamp - self._time_origin.value

    @property
    def isotime(self) -> str:
        if np.isnan(self.timestamp):
            return "??-??-?? ??:??:??"
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")


class LEEMStack(DataObjectStack):
    _type = LEEMImg

    def __init__(self, *args,
                 time_origin: Optional[Union[TimeOrigin,Number]] = None,
                 **kwargs) -> None:
        # initialize with np.nan for the first calls to self._single_construct
        self._time_origin = TimeOrigin(time_origin)
        super().__init__(*args, **kwargs)
        # now we can access the 0th element and set the value
        if time_origin is None:
            self._time_origin.value = self[0].timestamp

    def _split_source(self, source: Union[str,Iterable]) -> list:
        if isinstance(source, str):
            fnames = sorted(glob.glob(f"{source}*.dat"))
            if not fnames:
                fnames = sorted(glob.glob(f"{source}/*.dat"))
            return fnames
        return source

    def _single_construct(self, source: Any) -> LEEMImg:
        """Construct a single DataObject."""
        return LEEMImg(source, time_origin=self._time_origin)

    @property
    def time_origin(self) -> Number:
        return self._time_origin.value
    @time_origin.setter
    def time_origin(self, value: Number) -> None:
        self._time_origin.value = value

    def __getitem__(self, index: Union[int,slice]) -> Union[LEEMImg,LEEMStack]:
        elements = self._elements[index]
        if isinstance(index, int):
            if self.virtual: # if virtual elements contains just sources not DataObjects
                return self._single_construct(elements)
            return elements
        return type(self)(elements, virtual=self.virtual, time_origin=self._time_origin)



# Format: meta_key: (byte_position, encoding)
HEADER_ONE = {
    "_id":              (0, "cp1252"),
    "_size":            (20, "short"),
    "_version":         (22, "short"),
    "_bitsperpix":      (24, "short"),
    "width":            (40, "short"),
    "height":           (42, "short"),
    "_noimg":           (44, "short"),
    "_recipe_size":     (46, "short"),
}
HEADER_TWO = {
    "_isize":           (0, "short"),
    "_iversion":        (2, "short"),
    "_colorscale_low":  (4, "short"),
    "_colorscale_high": (6, "short"),
    "time":             (8, "time"),
    "_mask_xshift":     (16, "short"),
    "_mask_yshift":     (18, "short"),
    "_usemask":         (20, "bool"),
    "_att_markupsize":  (22, "short"),
    "_spin":            (24, "short")
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
    255: (0, None),      # stop byte
    100: (8, {"x_position":         (0, "float"),
              "y_position":         (4, "float")}),
    # Average Images: 0 means no averaging, 255 means sliding average
    104: (6, {"exposure":           (0, "float"),
              "averaging":          (4, "short")}),
    105: (0, {"_img_title":         (0, "cp1252")}),
    242: (2, {"mirror_state":       (0, "bool")}),
    243: (4, {"screen_voltage":     (0, "float")}),
    244: (4, {"mcp_voltage":        (0, "float")})
}
UNIT_CODES = {"1": "V", "2": "mA", "3": "A", "4": "°C",
              "5": "K", "6": "mV", "7": "pA", "8": "nA", "9": "\xb5A"}


def parse_dat(fname: str, debug: bool = False) -> dict[str, Any]:
    """Parse a UKSOFT2001 file."""
    data = {}
    def parse_block(block, field_dict):
        for key, (pos, encoding) in field_dict.items():
            data[key] = parse_bytes(block, pos, encoding)
            data[f"{key}_unit"] = ""
            if debug:
                print(f"\t{key} -> {data[key]}")

    with Path(fname).open("rb") as uk_file:
        parse_block(uk_file.read(104), HEADER_ONE)              # first fixed header

        if data["_recipe_size"] > 0:                            # optional recipe
            data["recipe"] = parse_bytes(uk_file.read(data["_recipe_size"]), 0, "cp1252")
            uk_file.seek(128 - data["_recipe_size"], 1)

        parse_block(uk_file.read(26), HEADER_TWO)               # second fixed header

        leemdata_version = parse_bytes(uk_file.read(2), 0, "short")
        if leemdata_version != 2:
            uk_file.seek(388, 1)
        bit = uk_file.read(1)[0]
        while bit != 255:
            if debug:
                print(bit)
            if bit in VARIABLE_HEADER:                          # fixed byte codes
                block_length, field_dict = VARIABLE_HEADER[bit]
                buffer = uk_file.read(block_length)
                parse_block(buffer, field_dict)
                if debug:
                    print("\tknown")
            elif bit in (106, 107, 108, 109, 235, 236, 237):    # varian pressures
                key = parse_cp1252_until_null(uk_file, debug)
                data[f"{key}_unit"] = parse_cp1252_until_null(uk_file, debug)
                data[key] = parse_bytes(uk_file.read(4), 0, "float")
                if debug:
                    print(f"\tknown: pressure {key} -> {data[key]}")
            elif bit in (110, 238):                             # field of view
                fov_str = parse_cp1252_until_null(uk_file, debug)
                if "LEED" in fov_str:
                    data["fov"] = -1
                else:
                    data["fov"] = int(fov_str.split("\t")[0].replace("µm", "").strip())
                data["fov_cal"] = parse_bytes(uk_file.read(4), 0, "float")
                if debug:
                    print(f"\tfov: {fov_str}")
            elif bit in (0, 1, 63, 66, 113, 128, 176, 216, 240, 232, 233):
                if debug:
                    print(f"unknown byte {bit}")
            elif bit: # self-labelled stuff
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
        image = np.fromfile(uk_file, dtype=np.uint16, sep='', count=size)
        image = np.array(image, dtype=np.float32)
        image = np.flipud(image.reshape((data["height"], data["width"])))
        data["image"] = image

    for new, old in ATTR_NAMES.items():
        data[new] = data.pop(old, np.nan)
        data[f"{new}_unit"] = data.pop(f"{old}_unit", "")
    if data["averaging"] == 0:
        data["averaging"] = 1
    elif data["averaging"] == 255:
        data["averaging"] = 0

    data["energy_unit"] = "eV"

    return data
