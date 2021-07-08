"""Some functions for processing LEEM images and stacks."""
# pylint: disable=missing-docstring
# pylint: disable=invalid-name

import copy
from collections import abc

import numpy as np
import scipy.signal
import scipy.constants as sc
import skimage.measure
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.lines

from agfalta.leem.utility import stackify
from agfalta.utility import progress_bar


def calculate_dose(stack, pressurefield="pressure1", approx=1):
    """Maybe get rid of approx?."""
    stack = stackify(stack)
    pressure = getattr(stack, pressurefield)[::approx]
    rel_time = stack.rel_time[::approx]
    approx_dose = np.cumsum(pressure * np.gradient(rel_time)) * 1e6
    # scale up to original length:
    long_dose = np.repeat(approx_dose, approx)
    cutoff = len(long_dose) - len(stack)
    if cutoff < 0:
        raise ValueError("Error calculating dose")
    return long_dose[cutoff // 2 : len(stack) + cutoff // 2]


class ROI:
    _defaults = {
        "circle": {"radius": 10},
        "rectangle": {"width": 50, "height": 50},
        "ellipse": {"xradius": 10, "yradius": 20}
    }
    kwargs = (
        "x0", "y0", "type_", "color", "radius", "width", "height", "xradius", "yradius"
    )
    def __init__(self, x0, y0, type_=None, color="k", alpha=1, **kwargs):
        self.position = np.array([x0, y0])
        self._img_shape = None
        self.area = None
        self._mask = None
        if type_ is None:
            for t, props in self._defaults.items():
                if any(k in kwargs for k in props):
                    type_ = t
                    break
            else:
                type_ = "circle"
        self.type_ = type_
        assert self.type_ in self._defaults
        self.params = copy.deepcopy(self._defaults[self.type_])
        self.params.update(kwargs)
        self.alpha = alpha
        self.color = color

    def __repr__(self):
        return (f"{self.type_}(position:{self.position},"
                + ",".join(f"{k}:{v}" for k, v in self.params.items()) + ")")

    def apply(self, img_array):
        if img_array.shape != self._img_shape or self._mask is None:
            self._mask = self.create_mask(*img_array.shape)
            self._img_shape = img_array.shape
        mask = self._mask
        return img_array * mask

    def create_mask(self, img_height, img_width):
        y, x = np.arange(0, img_height), np.arange(0, img_width)
        y, x = y[:, np.newaxis], x[np.newaxis, :]
        x0, y0 = self.position
        if self.type_ == "circle":
            r = self.params["radius"]
            mask = (x - x0)**2 + (y - y0)**2 < r**2
        elif self.type_ == "ellipse":
            xr, yr = self.params["xradius"], self.params["yradius"]
            mask = ((x - x0) / xr)**2 + ((y - y0) / yr)**2 < 1
        elif self.type_ == "rectangle":
            w, h = self.params["width"], self.params["height"]
            mask = (x >= x0) & (x < x0 + w) & (y >= y0) & (y < y0 + h)
        else:
            raise ValueError("Unknown ROI type")
        self.area = mask.sum()
        return mask

    @property
    def artist(self):
        if self.type_ == "circle":
            art = plt.Circle(
                self.position, self.params["radius"],
                color=self.color, alpha=self.alpha, fill=False
            )
        elif self.type_ == "ellipse":
            art = matplotlib.patches.Ellipse(
                self.position,
                self.params["xradius"] * 2, self.params["yradius"] * 2,
                color=self.color, alpha=self.alpha, fill=False
            )
        elif self.type_ == "rectangle":
            art = plt.Rectangle(
                self.position, self.params["width"], self.params["height"],
                color=self.color, alpha=self.alpha, fill=False
            )
        else:
            raise ValueError("Unknown ROI type")
        return art



def roify(*args, **kwargs):
    """Takes either a single ROI, an iterable of ROIs or a set of
    arguments for the ROI constructor. Returns a list of ROIs (for the
    first and latter case, this list has length 1)."""
    if "rois" in kwargs:
        args = (*args, kwargs.pop("rois"))
    if args and isinstance(args[0], ROI):
        if kwargs or len(args) > 1:
            print("WARNING: too many arguments for roify()")
        return [args[0]]
    elif args and isinstance(args[0], abc.Iterable):
        if all(isinstance(roi, ROI) for roi in args[0]):
            if kwargs or len(args) > 1:
                print("WARNING: too many arguments for roify()")
            return args[0]
    return [ROI(*args, **kwargs)]

# def roi_intensity(img, *args, **kwargs):
#     img = imgify(img)
#     roi = roify(*args, **kwargs)
#     cutout = roi.apply(img.data)
#     return np.mean(cutout)

# def roi_intensity_stack(stack, *args, **kwargs):
#     stack = stackify(stack)
#     roi = roify(*args, **kwargs)
#     intensity = np.zeros(len(stack))
#     for i, img in enumerate(stack):
#         intensity[i] = np.mean(roi.apply(img.data))
#     return intensity

def get_max_variance_idx(stack):
    stack = stackify(stack)
    max_var = 0
    max_index = 0
    for i, img in enumerate(stack):
        var = np.var(
            (img.data.flatten() - np.amin(img.data))
            / (np.amax(img.data) - np.amin(img.data))
        )
        if var > max_var:
            max_var = var
            max_index = i
    return max_index


class Profile(ROI):
    """
    Describes a profile through a LEEM or LEED image.
    Arguments:
        - x0, y0:   the center of the profile
        - theta:    inclination with respect to the horizontal
        - length:   the profile extends by length/2 from the center in each direction
        - width:    width to average over
        - reduce_func:  how to average along width, either a function or "gaussian" or "rect"
        - alpha, color: used when plotting the line
    Attributes:
        - endpoints
        - length
        - artist    can be added to a matplotlib Axes
    Methods:
        - apply(img_array)  takes a 2D numpy array and returns a 1D profile
    """
    _defaults = {
        "profile": {"theta": 0, "length": 100, "width": 10}
    }
    kwargs = ("x0", "y0", "type_", "color", "width", "length", "theta")

    def __init__(self, *args, alpha=0.3, reduce_func="gaussian", **kwargs):
        super().__init__(*args, type_="profile", alpha=alpha, **kwargs)
        self.params["theta"] *= np.pi / 180
        self.params["length"] = int(self.params["length"])
        self.reduce = reduce_func

    def apply(self, img_array):
        profile = skimage.measure.profile_line(
            img_array, self.endpoints[0, :], self.endpoints[1, :],
            linewidth=self.params["width"], mode="constant", reduce_func=self.reduce
        )
        return profile

    @property
    def length(self):
        return self.params["length"]

    @property
    def endpoints(self):
        theta = self.params["theta"]
        length = self.params["length"] - 1 # profile_line will include endpoint
        dyx = np.array([np.cos(theta) * length, -np.sin(theta) * length])
        return np.array([self.position - dyx * 0.4999, self.position + dyx * 0.4999])
        # dyx * 0.4999 to stay below the length (profile_line will do ceil)

    @property
    def xy(self):
        x = np.linspace(self.endpoints[0, 0], self.endpoints[1, 0], self.length)
        y = np.linspace(self.endpoints[0, 1], self.endpoints[1, 1], self.length)
        return np.stack([x, y]).T

    @property
    def xyC(self):
        """xy "edge" values for pcolormesh"""
        x = np.linspace(self.endpoints[0, 0], self.endpoints[1, 0], self.length + 1)
        y = np.linspace(self.endpoints[0, 1], self.endpoints[1, 1], self.length + 1)
        return np.stack([x, y]).T

    @property
    def reduce(self):
        return self._reduce_func
    @reduce.setter
    def reduce(self, func):
        width = self.params["width"]
        if func == "gaussian":
            window = scipy.signal.windows.gaussian(width, std=width / 2)
            func = lambda x: np.mean(window * x)
        elif func in ("rect", "boxcar"):
            window = scipy.signal.windows.boxcar(width)
            func = lambda x: np.mean(window * x)
            # func = None # also works?
        elif not callable(func):
            raise ValueError(f"Unkown reduce_func {func}")
        self._reduce_func = func

    def create_mask(self, img_height, img_width):
        raise NotImplementedError

    @property
    def artist(self):
        art = matplotlib.lines.Line2D(
            self.endpoints[:, 0], self.endpoints[:, 1],
            lw=self.params["width"], color=self.color, alpha=self.alpha,
            solid_capstyle="butt"
        )
        return art


class RSM:
    def __init__(self, stack, profile, gamma=None, k_res=7.67e7):
        self.stack = stackify(stack)
        self.profile = profile
        if gamma is None:
            gamma = profile.position
        self.gamma = np.array(gamma)
        self.k_res = k_res

        self.data = None

    def __call__(self):
        if self.data is not None:
            return self.data

        kpara = np.linalg.norm(self.profile.xyC - self.gamma, axis=1) * self.k_res
        kxi_center = np.linalg.norm(self.profile.position - self.profile.xy[0, :])
        kxi_x = np.linalg.norm(self.profile.xyC - self.profile.xyC[0, :], axis=1)
        kxi = (kxi_x - kxi_center) * self.k_res
        kx = np.tile(kxi, (len(self.stack) + 1, 1))

        E = self.stack.energy
        dE = np.mean(np.diff(E))
        EC = np.append(E - dE, E[-1] + dE)

        ky = np.zeros((len(EC), self.profile.length + 1))
        z = np.zeros((len(self.stack), self.profile.length))

        for i, img in enumerate(progress_bar(self.stack, "Calculating RSM...")):
            ky[i, :] = self.get_kperp(EC[i], kpara)
            z[i, :] = self.profile.apply(img.data)

        ky[-1, :] = self.get_kperp(EC[-1], kpara)
        self.data = (kx, ky, z)
        return self.data

    def get_kpara(self):
        kx = np.linalg.norm(self.profile.xyC - self.gamma, axis=1) * self.k_res
        return kx

    def get_kperp(self, E, kpara):
        energy = E * sc.e
        k0 = np.sqrt(2 * sc.m_e * energy) / sc.hbar     # total k
        kpara = kpara.clip(max=k0)
        kperp = k0 + np.sqrt(k0**2 - kpara**2)          # kpara^2 + kperp^2 = k^2
        return np.nan_to_num(kperp, 0)
