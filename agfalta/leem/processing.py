"""Some functions for processing LEEM images and stacks."""
# pylint: disable=missing-docstring
# pylint: disable=invalid-name

import copy
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt

from agfalta.leem.utility import stackify, imgify


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
    def __init__(self, x0, y0, type_="circle", **kwargs):
        self.position = x0, y0
        self._img_shape = None
        self._mask = None
        self._type = type_
        assert self._type in self._defaults
        self.params = copy.deepcopy(self._defaults[self._type])
        self.params.update(kwargs)

    def apply(self, img_array):
        if img_array.shape != self._img_shape or self._mask is None:
            self._mask = self.create_mask(*img_array.shape)
        mask = self._mask
        return img_array * mask

    def create_mask(self, img_width=None, img_height=None):
        x, y = np.arange(0, img_width), np.arange(0, img_height)
        x, y = x[:, np.newaxis], y[np.newaxis, :]
        x0, y0 = self.position
        if self._type == "circle":
            r = self.params["radius"]
            mask = (x - x0)**2 + (y - y0)**2 < r
        elif self._type == "ellipse":
            xr, yr = self.params["xradius"], self.params["yradius"]
            mask = ((x - x0) / xr)**2 + ((y - y0) / yr)**2 < 1
        elif self._type == "rectangle":
            w, h = self.params["width"], self.params["height"]
            mask = (x >= x0) & (x < x0 + w) & (y >= y0) & (y < y0 + h)
        else:
            raise ValueError("Unknown ROI type")
        return mask

    def artist(self, color="k"):
        if self._type == "circle":
            art = plt.Circle(
                self.position, self.params["radius"], 
                color=color, fill=False
            )
        elif self._type == "ellipse":
            art = plt.Ellipse(
                self.position, 
                self.params["xradius"] * 2, self.params["yradius"] * 2,
                color=color, fill=False
            )
        elif self._type == "rectangle":
            art = plt.Rectangle(
                self.position, self.params["width"], self.params["height"],
                color=color, fill=False
            )
        else:
            raise ValueError("Unknown ROI type")
        return art
        


def roify(*args, **kwargs):
    """Takes either a single ROI, an iterable of ROIs or a set of
    arguments for the ROI constructor. Returns a list of ROIs (for the 
    first and latter case, this list has length 1)."""
    if isinstance(args[0], ROI):
        if kwargs or len(args) > 1:
            print("WARNING: too many arguments for roify()")
        return [args[0]]
    elif isinstance(args[0], Iterable):
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
