"""
Basic classes for Elmitec LEEM ".dat"-file parsing and data visualization.
"""
# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=attribute-defined-outside-init

import struct
import pickle
from datetime import datetime, timedelta, timezone
import glob

from skimage.io import imread
import numpy as np


LEEMBASE_VERSION = 1.1


def main():
    """Example usage."""
    mcp = LEEMImg("channelplate.dat")
    print(mcp.meta)
    print(mcp.data.shape)
    print(mcp.temperature)
    print(mcp.timestamp)
    print(mcp.get_unit("energy"))
    print(mcp.energy)
    mcp.energy = 9
    print(mcp.get_field_string("energy"))

    stack = LEEMStack("raw_datfiles", virtual=False)
    print(len(stack))
    print(stack.data.shape) # numpy array
    print(stack.energy[:4])
    print(stack.pressure2[::10])
    print(stack[3].energy)
    for img in stack[:2]:
        print(img.exposure)



class Loadable:
    _pickle_extension = ".unknown"

    def __init__(self, path, *args, **kwargs):
        try:
            instance = self.load(path, *args, **kwargs)
            self.__dict__ = instance.__dict__
        except (ValueError, TypeError, AttributeError):
            super().__init__(*args, **kwargs)

    def __setstate__(self, state):
        """Make sure that the path is inserted first."""
        self.path = state.pop("path")
        if "fnames" in state:
            self.fnames = state.pop("fnames")
        self.__dict__.update(state)

    @classmethod
    def load(cls, path, *_args, **kwargs):
        # pylint: disable=protected-access
        if path.endswith(cls._pickle_extension):
            with open(path, "rb") as pfile:
                print(f"Loading stack from '{path}'")
                instance = pickle.load(pfile)
                if "time_origin" in kwargs:
                    instance._time_origin = kwargs["time_origin"]
                return instance
        raise ValueError("File not compatible")

    def save(self, path):
        if not path.endswith(self._pickle_extension):
            path += self._pickle_extension
        with open(path, "wb") as pfile:
            try:
                pickle.dump(self, pfile, protocol=4)
            except RecursionError:
                print("WARING: Did not save due to recursion error.")
                raise



class LEEMImg(Loadable):
    """
    LEEM image that exposes metadata as attributes. Usage:
        >>> img = LEEMImgBase(fname)
        >>> print(img.energy)
        >>> print(img.height)
        >>> data = img.data
    and so on.
    Possible attributes are:
        - data              numpy array containing the image
        - energy            in eV
        - temperature
        - fov               0: LEED, nan: unknown FoV
        - pressure1         in Torr
        - pressure2
        - width             in pixels
        - height
        - time              as string
        - timestamp         as UNIX timestamp
        - rel_time          seconds since start of stack (fallback: timestamp)
        - exposure          camera exposure in s
        - averaging         0: sliding average
        - Objective         lens current in mA
    """
    _attrs_with_unit = {
        "energy": "Start Voltage",
        "temperature": "Sample Temp.",
        "pressure1": "Gauge #1",
        "pressure2": "Gauge #2",
        "objective": "Objective",
    }
    _attrs = {
        "width": "width",
        "height": "height",
        "time_dtobject": "time",
        "exposure": "Camera Exposure",
        "averaging": "Average Images",
    }
    _fallback_units = {
        "energy": "V",
        "temperature": "°C",
        "pressure1": "Torr",
        "pressure2": "Torr",
        "objective": "mA",
        "fov": "µm",
        "timestamp": "s",
        "exposure": "s",
        "rel_time": "s",
        "dose": "L",
    }
    _pickle_extension = ".limg"

    def __init__(self, path, time_origin=datetime.min, nolazy=False):
        self.path = path
        self._time_origin = time_origin
        self._meta = None
        self._data = None
        # this order is crucial for proper loading
        super().__init__(path)

        try:
            self.parse_nondat(path)
        except (ValueError, TypeError, AttributeError):
            if not self.path.endswith(".dat"):
                raise ValueError(f"'{self.path}' does not exist or is not a *.dat file")
            with open(path, "rb") as _:
                pass            # making sure this is a file name
        if nolazy:
            # pylint: disable=pointless-statement
            self.meta
            self.data

    def parse_nondat(self, path):
        """Use this for other formats than pickle (which is already
        implemented by Loadable)."""
        try:
            data = np.float32(imread(path))
        except IOError:
            # if the object given already is a numpy array:
            data = path
            self.path = "NO_PATH"
        if len(data.shape) != 2:
            raise ValueError(f"File '{path}' is not a single image")
        self._data = data
        self._meta = {}

    @property
    def meta(self):
        """Dictionary containing all header attributes."""
        if self._meta is None:
            self._meta = parse_header(self.path)
        return self._meta

    @property
    def data(self):
        """Numpy array containing the image data."""
        if self._data is None:
            self._data = parse_data(self.path)
        return self._data
    @data.setter
    def data(self, value):
        if value.shape != (self.height, self.width):
            raise ValueError("Image has wrong dimensions")
        self._data = value

    def __getattr__(self, attr):
        # if these don't exist, there is a problem:
        if attr in ("path", "_meta", "_data", "_time_origin"):
            raise AttributeError
        try:
            return self.meta.get(self._attrs[attr], np.nan)
        except KeyError:
            try:
                return self.meta.get(self._attrs_with_unit[attr], (np.nan, None))[0]
            except KeyError:
                raise AttributeError

    def __setattr__(self, attr, value):
        if attr in self._attrs:
            self.meta[self._attrs[attr]] = value
        elif attr in self._attrs_with_unit:
            unit = self._fallback_units[attr]
            self.meta[self._attrs_with_unit[attr]] = (value, unit)
        else:
            super().__setattr__(attr, value)

    def get_unit(self, field):
        """Unit string for the specified field."""
        try:
            return self.meta.get(self._attrs_with_unit[field], (np.nan, None))[1]
        except KeyError:
            if hasattr(self, field):
                return self._fallback_units.get(field, "")
            raise ValueError(f"Unknown field {field}")

    def get_field_string(self, field):
        """Returns a string with the field value and unit."""
        value = getattr(self, field)
        if value == np.nan:
            return "NaN"
        return f"{value:.5g} {self.get_unit(field)}".strip()

    @property
    def timestamp(self):
        """Start voltage."""
        return self.time_dtobject.timestamp()

    @property
    def time(self):
        """Start voltage."""
        return self.time_dtobject.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def rel_time(self):
        """Relative time in s (see set_time_origin()). Only makes sense for a stack."""
        return (self.time_dtobject - self._time_origin).seconds

    @property
    def time_origin(self):
        """Expects a timestamp."""
        return self._time_origin
    @time_origin.setter
    def time_origin(self, value):
        """Expects a timestamp."""
        try:
            self._time_origin = datetime.fromtimestamp(value)
        except ValueError:
            self._time_origin = value

    @property
    def fov(self):
        """Either a positive number in µm, 0 for LEED, or NaN for unknown FoV."""
        fov_str = self.meta.get("FoV", "")
        if "LEED" in fov_str:
            return 0
        try:
            return float(fov_str.split("µ")[0])
        except ValueError:      # when the string does not contain a number
            return np.nan

    @property
    def averaging(self):
        """Number of images that are averaged. 0 means sliding average."""
        avg = self.meta.get("Average Images", np.nan)
        if avg == 255:
            return 0
        elif avg == 0:
            return 1
        return avg


class LEEMStack(Loadable):
    # pylint: disable=too-few-public-methods
    """Container object for LEEMImg instances. It has the same attributes as LEEMImgBase
    and returns them as numpy arrays.
    It can be used like this:
        stack = LEEMStack("path/to/imagestack")
        img8 = stack[8]
        for img in stack:
            print(img.energy)
        print(stack.energy)     # same as above
        if img5 in stack:
            print("yes")
    """
    _pickle_extension = ".lstk"

    def __init__(self, path, virtual=False, nolazy=False):
        self.path = path
        self._time_origin = datetime.min
        self._virtual = virtual
        self._images = None
        self._data = None
        try:
            self.parse_nondat(path)
            self._virtual = False
        except (ValueError, TypeError, AttributeError):
            self.fnames = sorted(glob.glob(f"{path}/*.dat"))

        # This order is crucial!
        super().__init__(path)
        if not self.fnames:
            raise ValueError(f"'{self.path}' does not exist or contains no *.dat files")
        if nolazy:
            # pylint: disable=pointless-statement
            self[0]

    def parse_nondat(self, path):
        """Use this for other formats than pickle (which is already
        implemented by Loadable)."""
        try:
            data = np.float32(imread(path))
        except IOError:
            # if the object given already is a numpy array:
            data = path
            self.path = "NO_PATH"
        if len(data.shape) != 3:
            raise ValueError(f"File {path} is not a single image")
        self._images = [LEEMImg(data[i, :, :]) for i in range(data.shape[0])]
        self.fnames = [None] * data.shape[0]

    def delete_frames(self, indexes):
        for index in sorted(indexes, reverse=True):
            del self._images[index]
            del self.fnames[index]
        self._data = None
        if 0 in indexes:
            self._time_origin = datetime.min

    @property
    def time_origin(self):
        if self._time_origin == datetime.min:
            try:
                self._time_origin = self._images[0].time_dtobject
            except (IndexError, TypeError):
                try:
                    self._time_origin = LEEMImg(self.fnames[0]).time_dtobject
                except (IndexError, TypeError):
                    pass
        return self._time_origin

    def __getitem__(self, index):
        if self._virtual:
            return [LEEMImg(fname, self.time_origin) for fname in self.fnames.__getitem__(index)]
        if self._images is None:
            self._images = [LEEMImg(fname, self.time_origin) for fname in self.fnames]
        return self._images[index]

    def __len__(self):
        return len(self.fnames)

    def __getattr__(self, attr):
        # if these don't exist, there is a problem:
        if attr in ("fnames", "path", "_images", "_virtual", "_data", "_time_origin"):
            raise AttributeError
        try:
            return np.array([getattr(img, attr) for img in self])
        except AttributeError:
            raise AttributeError

    def __setattr__(self, attr, value):
        if attr in ("fnames", "path", "_images", "_virtual", "_data", "_time_origin"):
            super().__setattr__(attr, value)
        elif len(self) == len(value):
            for img, single_value in zip(self, value):
                setattr(img, attr, single_value)
        else:
            raise ValueError(f"Value '{value}' for '{attr}' has wrong shape")


    @property
    def data(self):
        print("WARNING: Using stack.data is deprecated!")
        if self._data is None:
            if self._images is None:
                self._images = [LEEMImg(fname, self.time_origin) for fname in self.fnames]
            self._data = np.stack([img.data for img in self._images], axis=0)
        return self._data



def _parse_string_until_null(fd):
    buffer = b""
    while b"\x00" not in buffer:
        buffer += fd.read(1)
    return buffer[:-1].decode("cp1252")

def _parse_bytes(buffer, pos, encoding):
    if encoding == "cp1252":
        return buffer[pos:].split(b"\x00")[0].decode("cp1252")
    elif encoding == "short":
        return struct.unpack("<h", buffer[pos:pos + 2])[0]
    elif encoding == "float":
        return struct.unpack("<f", buffer[pos:pos + 4])[0]
    elif encoding == "time":
        epoch_start = datetime(year=1601, month=1, day=1)  # begin of windows time
        timestamp = struct.unpack("<Q", buffer[pos:pos + 8])[0]
        seconds_since_epoch = timestamp / 10**7  # conversion from 100ns to s
        utc_time = epoch_start + timedelta(seconds=seconds_since_epoch)
        return utc_time.replace(tzinfo=timezone.utc).astimezone(tz=None)
    elif encoding == "bool":
        return struct.unpack("<?", buffer[pos:pos + 1])[0]
    else:
        raise ValueError("Unknown encoding")

# Format: meta_key: (byte_position, encoding)
HEADER_ONE = {
    "id":               (0, "cp1252"),
    "size":             (20, "short"),
    "version":          (22, "short"),
    "bitsperpix":       (24, "short"),
    "width":            (40, "short"),
    "height":           (42, "short"),
    "_noimg":           (44, "short"),
    "_recipe_size":     (46, "short"),
}
HEADER_TWO = {
    "isize":            (0, "short"),
    "iversion":         (2, "short"),
    "colorscale_low":   (4, "short"),
    "colorscale_high":  (6, "short"),
    "time":             (8, "time"),
    "mask_xshift":      (16, "short"),
    "mask_yshift":      (18, "short"),
    "usemask":          (20, "bool"),
    "att_markupsize":   (22, "short"),
    "spin":             (24, "short")
}
# Format: byte_position: (block_length, field_dict)
# where field_dict is formatted like the above HEADER_ONE and HEADER_TWO
VARIABLE_HEADER = {
    255: (0, None),      # stop byte
    100: (8, {"Mitutoyo X":         (0, "float"),
              "Mitutoyo Y":         (4, "float")}),
    # Average Images: 0 means no averaging, 255 means sliding average
    104: (6, {"Camera Exposure":    (0, "float"),
              "Average Images":     (4, "short")}),
    105: (0, {"Image Title":        (0, "cp1252")}),
    242: (2, {"MirrorState":        (0, "bool")}),
    243: (4, {"MCPScreen":          (0, "float")}),
    244: (4, {"MCPChanneplate":     (0, "float")})
}
UNIT_CODES = {"1": "V", "2": "mA", "3": "A", "4": "°C",
              "5": "K", "6": "mV", "7": "pA", "8": "nA", "9": "\xb5A"}

def parse_header(fname):
    meta = {}
    def parse_block(block, field_dict):
        for key, (pos, encoding) in field_dict.items():
            meta[key] = _parse_bytes(block, pos, encoding)

    with open(fname, 'rb') as f:
        parse_block(f.read(104), HEADER_ONE)                    # first fixed header

        if meta["_recipe_size"] > 0:                            # optional recipe
            meta["recipe"] = _parse_bytes(f.read(meta["_recipe_size"]), 0, "cp1252")
            f.seek(128 - meta["_recipe_size"], 1)

        parse_block(f.read(26), HEADER_TWO)                     # second fixed header

        leemdata_version = _parse_bytes(f.read(2), 0, "short")
        if leemdata_version != 2:
            f.seek(388, 1)
        b = f.read(1)[0]
        while b != 255:
            if b in VARIABLE_HEADER:                            # fixed byte codes
                block_length, field_dict = VARIABLE_HEADER[b]
                buffer = f.read(block_length)
                parse_block(buffer, field_dict)
            elif b in (106, 107, 108, 109, 235, 236, 237):      # varian pressures
                key = _parse_string_until_null(f)
                unit = _parse_string_until_null(f)
                value = _parse_bytes(f.read(4), 0, "float")
                meta[key] = (value, unit)
            elif b in (110, 238):                               # field of view
                fov_str = _parse_string_until_null(f)
                meta["LEED"] = "LEED" in fov_str
                meta["FoV"] = fov_str.split("\t")[0].strip()
                meta["FoV cal"] = _parse_bytes(f.read(4), 0, "float")
            else:                                               # self-labelled stuff
                keyunit = _parse_string_until_null(f)
                unit = UNIT_CODES.get(keyunit[-1], "")
                meta[keyunit[:-1]] = (_parse_bytes(f.read(4), 0, "float"), unit)
            b = f.read(1)[0]
    return meta


def parse_data(fname, width=None, height=None):
    with open(fname, "rb") as f:
        if None in (width, height):
            block = f.read(104)
            width = _parse_bytes(block, *HEADER_ONE["width"])
            height = _parse_bytes(block, *HEADER_ONE["height"])
        size = width * height
        f.seek(-2 * size, 2)
        data = np.fromfile(f, dtype=np.uint16, sep='', count=size)
        data = np.array(data, dtype=np.float32)
        data = np.flipud(data.reshape((height, width)))
    return data


if __name__ == "__main__":
    main()
