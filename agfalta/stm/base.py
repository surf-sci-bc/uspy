"""Base class for STM images."""

# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=too-many-arguments

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
        "vx": "sranger_mk2_hwi_scan_speed_x",
        "CP": "sranger_mk2_hwi_usr_cp",
        "CI": "sranger_mk2_hwi_usr_ci"
    }
    _dim_attrs = {
        "resx": "dimx",
        "resy": "dimy"
    }
    def __init__(self, fname):
        self.ds = nc.Dataset(fname)

    @property
    def metadata(self):
        return self.ds.__dict__

    @property
    def dimensions(self):
        return self.ds.dimensions.values()
        # return dict((dim.name, dim.size) for dim in self.ds.dimensions.values())
    
    @property
    def variables(self):
        return self.ds.variables.values()

    def get_variable(self, key):
        if key not in self._attrs:
            raise AttributeError(f"Unknown key '{key}'")
        for var in self.variables:
            if var.name == self._attrs[key]:
                return var
        raise AttributeError(f"'{self._attrs[key]}' not in variables")

    def get_variable_value(self, key):
        try:
            var = self.get_variable(key)
            data = var[:]
            if var.dtype.type is np.bytes_:
                bstr = data[~data.mask].tobytes()
                try:
                    return bstr.decode("utf-8").strip()
                except UnicodeDecodeError:
                    return bstr
        except AttributeError:
            if key not in self._dim_attrs:
                raise AttributeError(f"Unknown key '{key}'")
            for dim in self.dimensions:
                if dim.name == self._dim_attrs[key]:
                    data = dim.size
                    break
            else:
                raise AttributeError(f"'{self._dim_attrs[key]}' not in dimensions")            
        return data

    def get_field_string(self, attr):
        try:
            var = self.get_variable(attr)
        except AttributeError:
            return self.get_variable_value(attr)
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
        z = self.get_variable("z")[:]
        return np.squeeze(z)

