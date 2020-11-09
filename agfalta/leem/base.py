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
import copy

from skimage.io import imread
import numpy as np

from agfalta.utility import progress_bar


LEEMBASE_VERSION = 1.1


class Loadable:
    _pickle_extension = ".unknown"

    def load_pickle(self, path, *args, **kwargs):
        instance = self.unpickle(path, *args, **kwargs)
        self.__dict__.update(instance.__dict__)

    def __setstate__(self, state):
        """Make sure that the path is inserted first."""
        self.path = state.pop("path")
        if "fnames" in state:
            self.fnames = state.pop("fnames")
        self.__dict__.update(state)

    @classmethod
    def unpickle(cls, path, *_args, **kwargs):
        # pylint: disable=protected-access
        if path.endswith(cls._pickle_extension):
            with open(path, "rb") as pfile:
                # print(f"Loading stack from '{path}'")
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

        try:                        # assume datfile or pickle
            if path.endswith(".dat"):
                if nolazy:
                    _ = self.meta
                    _ = self.data
            elif path.endswith(self._pickle_extension):
                super().load_pickle(path)
            else:
                raise AttributeError
        except AttributeError:      # assume other readable file or numpy array
            try:
                self.parse_nondat(path)
            except (ValueError, TypeError, AttributeError):
                raise ValueError(f"{path} does not exist or has wrong file format.")

    def parse_nondat(self, path):
        """Use this for other formats than pickle (which is already
        implemented by Loadable)."""
        try:
            data = np.float32(imread(path))
        except IOError:
            # if the object given already is a numpy array:
            data = path
            path = "NO_PATH"
        if len(data.shape) != 2:
            raise ValueError(f"File '{path}' is not a single image")
        self.path = path
        self._data = data
        self._meta = {
            "height": data.shape[0],
            "width": data.shape[1],
            "time": datetime.min
        }

    def copy(self):
        return copy.deepcopy(self)

    def __eq__(self, other):
        try:
            assert self.path == other.path
            assert self.meta == other.meta
            assert (self.data == other.data).all()
            return True
        except (AssertionError, AttributeError):
            return False

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

    @property
    def meta(self):
        """Dictionary containing all header attributes."""
        if self._meta is None:
            self._meta = parse_header(self.path)
        return self._meta

    @property
    def additional_meta(self):
        if self._meta is None:
            self._meta = parse_header(self.path)
        add_meta = dict([
            (key, self._meta[key])
            for key in self._meta
            if key not in self._attrs.values()
            and key not in self._attrs_with_unit.values()
            and key not in ("FoV", )
        ])
        return add_meta

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
        if self.time_dtobject == datetime.min:
            return np.nan
        return self.time_dtobject.timestamp()

    @property
    def time(self):
        """Start voltage."""
        if self.time_dtobject == datetime.min:
            return "??-??-?? ??:??:??"
        return self.time_dtobject.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def rel_time(self):
        """Relative time in s (see set_time_origin()). Only makes sense for a stack."""
        if self.time_dtobject == datetime.min:
            return np.nan
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
    _unique_attrs = ("fnames", "path", "_images", "_virtual", "virtual",
                     "_time_origin", "time_origin", "_silent")

    def __init__(self, path, virtual=False, nolazy=False, time_origin=datetime.min,
                 verbose=False):
        # pylint: disable=too-many-branches, too-many-arguments
        self.path = path
        self._virtual = virtual
        self._images = None
        self.fnames = None

        self._time_origin = time_origin
        self._silent = not verbose

        try:                # first, assume a string that yields fnames or a pickle
            if path.endswith(".dat"):
                self.fnames = sorted(glob.glob(f"{path}*"))
            elif path.endswith(self._pickle_extension):
                super().load_pickle(path)
            else:
                self.fnames = sorted(glob.glob(f"{path}/*.dat"))
            if not self.fnames:
                raise AttributeError
        except AttributeError:
            try:            # now, assume a list that yields fnames in some way
                if isinstance(path[0], LEEMImg):
                    self.fnames = [img.path for img in path]
                    if not [img.data.shape == path[0].data.shape for img in path]:
                        raise ValueError("Incompatible image dimensions")
                    self._images = [img for img in path]
                    self._virtual = False
                else:
                    self.fnames = [fname for fname in path if fname.endswith(".dat")]
                if not self.fnames:
                    raise AttributeError
                self.path = "NO_PATH"
            except AttributeError:
                try:        # now, try everything else
                    self.parse_nondat(path)
                    self._virtual = False
                except (AttributeError, ValueError):
                    raise ValueError(f"'{self.path}' does not exist, cannot be read"
                                     " successfully or contains no *.dat files")
        if nolazy:
            for img in self:
                _ = img.meta
                _ = img.data

    def parse_nondat(self, path):
        """Use this for other formats than pickle (which is already
        implemented by Loadable)."""
        try:
            data = np.float32(imread(path))
        except IOError:
            # if the object given already is a numpy array:
            data = path
            path = "NO_PATH"
        if len(data.shape) != 3:
            raise ValueError(f"File {path} is not an image stack")
        self.path = path
        self._images = [LEEMImg(data[i, :, :]) for i in range(data.shape[0])]
        self.fnames = ["NO_PATH"] * data.shape[0]

    def _load_images(self):
        self._virtual = False
        if self._images is None:
            self._images = [LEEMImg(self.fnames[0], self.time_origin)]
            if len(self.fnames) < 2:
                return
            for fname in progress_bar(self.fnames[1:], "Loading images...", silent=self._silent):
                img = LEEMImg(fname, self.time_origin)
                if img.data.shape != self._images[0].data.shape:
                    raise ValueError("Image has the wrong dimensions")
                self._images.append(img)

    def copy(self):
        if not self._silent:
            print("Copying stack...")
        return copy.deepcopy(self)

    def __eq__(self, other):
        try:
            assert self.path == other.path
            assert self.fnames == other.fnames
            for self_img, other_img in zip(self, other):
                assert self_img == other_img
            return True
        except (AssertionError, AttributeError):
            return False

    def __getitem__(self, indexes):
        if isinstance(indexes, int):
            if self.virtual:
                return LEEMImg(self.fnames[indexes], time_origin=self.time_origin)
            self._load_images()
            return self._images[indexes]
        else:
            if self.virtual:
                return LEEMStack(self.fnames.__getitem__(indexes),
                                 time_origin=self.time_origin, virtual=True)
            self._load_images()
            return LEEMStack(self._images.__getitem__(indexes),
                             time_origin=self.time_origin)

    def __setitem__(self, indexes, imges):
        if isinstance(indexes, int) and isinstance(imges, LEEMImg):
            if imges.data.shape != self[0].data.shape:
                raise ValueError("Incompatible image dimensions")
            self.fnames[indexes] = imges.path
            if self.virtual:
                if imges.path != "NO_PATH":
                    return
                self._load_images()
            self._images[indexes] = imges
        else:
            fnames = [img.path for img in imges]
            self.fnames.__setitem__(indexes, fnames)
            if self.virtual:
                if "NO_PATH" not in fnames:
                    return
                self._load_images()
            if isinstance(imges, LEEMStack):
                self._images.__setitem__(indexes, [img for img in imges])
            if all([isinstance(img, LEEMImg) for img in imges]):
                if not [img.data.shape == imges[0].data.shape for img in imges[1:]]:
                    raise ValueError("Incompatible image dimensions")
                self._images.__setitem__(indexes, imges)
            else:
                raise TypeError("LEEMStack only takes LEEMImg elements")

    def __delitem__(self, indexes):
        self.fnames.__delitem__(indexes)
        if not self._virtual:
            self._images.__delitem__(indexes)

    def __len__(self):
        return len(self.fnames)

    def __getattr__(self, attr):
        #TODO add buffering of vectors
        if attr in self._unique_attrs:
            raise AttributeError
        try:
            if self.virtual:
                iterator = progress_bar(self, f"Collecting attr '{attr}'...", silent=self._silent)
            else:
                iterator = self
            return np.array([getattr(img, attr) for img in iterator])
        except AttributeError:
            raise AttributeError(f"Unknown attribute {attr}")

    def __setattr__(self, attr, value):
        if attr in self._unique_attrs:
            super().__setattr__(attr, value)
        elif not self.virtual and hasattr(value, "__len__") and len(self) == len(value):
            for img, single_value in zip(self, value):
                setattr(img, attr, single_value)
        elif not hasattr(self[0], attr):
            super().__setattr__(attr, value)
        else:
            raise ValueError(f"Value '{value}' for '{attr}' has wrong shape")

    @property
    def virtual(self):
        return self._virtual
    @virtual.setter
    def virtual(self, value):
        if not value:
            self._virtual = False
            self._load_images()
        else:
            if "NO_PATH" in self.fnames:
                raise ValueError("Stack can't be virtual retroactively")
            self._virtual = True
            self._images = None

    @property
    def verbose(self):
        return not self._silent
    @verbose.setter
    def verbose(self, value):
        self._silent = not value

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

    @property
    def data(self):
        raise NotImplementedError("stack.data is no longer supported")
    #     print("WARNING: Using stack.data is deprecated! Sane behaviour is not guaranteed")
    #     if self._data is None:
    #         self._load_images()
    #         self._data = np.stack([img.data for img in self._images], axis=0)
    #     return self._data

def calculate_dose(stack, pressurefield="pressure1", approx=1):
    """Maybe get rid of approx?."""
    pressure = getattr(stack, pressurefield)[::approx]
    rel_time = stack.rel_time[::approx]
    approx_dose = np.cumsum(pressure * np.gradient(rel_time))
    # scale up to original length:
    long_dose = np.repeat(approx_dose, approx)
    cutoff = len(long_dose) - len(stack)
    if cutoff < 0:
        raise ValueError("Error calculating dose")
    return long_dose[cutoff // 2 : len(stack) + cutoff // 2]


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
        return utc_time.replace(tzinfo=timezone.utc).astimezone(tz=None).replace(tzinfo=None)
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
            elif b not in (0, 66, 113, 176, 232, 233):          # self-labelled stuff
                keyunit = _parse_string_until_null(f)
                if not keyunit:     # happens for some b?
                    b = f.read(1)[0]
                    continue
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
