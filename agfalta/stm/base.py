"""Base class for STM images."""

# pylint: disable=invalid-name
# pylint: disable=missing-docstring

import netCDF4 as nc
import numpy as np


class STMImage:
    _attrs = {
        "z": "FloatField",
        "time_passed": "time",
        "value": "value",
        "x": "dimx",
        "y": "dimy",
        "time": "reftime",
        "comment": "comment",
        "title": "title",
        "rangex": "rangex",
        "rangey": "rangey",
        "rangez": "rangez",
        "dx": "dx",
        "dy": "dy",
        "dz": "dz",
        "offsetx": "offsetx",
        "offsety": "offsety",
        "rotation": "alpha",
        "timestamp_start": "t_start",
        "timestamp_end": "t_end",
        "original_path": "basename",
        "direction": "spm_scancontrol",
        "hardware_setup": "sranger_info",
        "bias": "sranger_mk2_hwi_bias",
        "speed": "sranger_mk2_hwi_scan_speed_x",
        "CP": "sranger_mk2_hwi_usr_cp",
        "CI": "sranger_mk2_hwi_usr_ci"
    }
    _dim_attrs = {
        "resx": "dimx",
        "resy": "dimy"
    }
    def __init__(self, fname):
        self.ds = nc.Dataset(fname)     # pylint: disable=no-member
        self._variables = None
        self._dimensions = None

    @property
    def metadata(self):
        return self.ds.__dict__

    @property
    def dimensions(self):
        if self._dimensions is None:
            self._dimensions = dict(
                (dim.name, dim) for dim in self.ds.dimensions.values()
            )
        return self._dimensions

    @property
    def variables(self):
        if self._variables is None:
            self._variables = dict(
                (var.name, var) for var in self.ds.variables.values()
            )
        return self._variables

    def get_variable_value(self, key):
        if key in self._dim_attrs:
            dim = self.dimensions.get(self._dim_attrs[key])
            if dim is None:
                return None
            return dim.size
        if key in self._attrs:
            var = self.variables.get(self._attrs[key])
            if var is None:
                return None
            val = var[:]
            if var.dtype.type is np.bytes_:
                bstr = val[~val.mask].tobytes()
                val = bstr.decode("utf-8", errors="ignore").strip()
            return val
        raise ValueError(f"Unknown data field '{key}'")

    def get_field_string(self, attr):
        var = self.variables.get(self._attrs.get(attr))
        if var is None:
            var = self.variables.get(self._dim_attrs.get(attr))
        value = self.get_variable_value(attr)

        try:
            unit = var.unit
        except AttributeError:
            try:
                unit = var.var_unit
            except AttributeError:
                unit = ""
        translate_dict = {
            "AA": "Å",
            "date string": "",
            "Grad": "\b°",
            "A/s": "Å/s",
            "1": ""
        }
        unit = translate_dict.get(unit, unit)

        return f"{value} {unit}".strip()

    def __getattr__(self, attr):
        return self.get_variable_value(attr)

    @property
    def z(self):
        z = self.variables[self._attrs["z"]]
        return np.squeeze(z)
