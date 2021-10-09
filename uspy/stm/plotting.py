"""Plotting helper functions."""

# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=too-many-arguments

import matplotlib.pyplot as plt
import numpy as np

from uspy.stm.base import STMImage
from uspy.stm import processing


def plot_img(fname, background="planar"):
    img = STMImage(fname)
    if background is None:
        data = img.z
    elif background == "planar":
        data = processing.subtract_plane(img.z)
    else:
        raise ValueError(f"Unknown background type {background}")
    plt.imshow(data)


def get_strar(fname):
    for var in STMImage(fname).variables:
        if "Reference time" in var.long_name:
            return var


def print_metadata(fname, verbose=False):
    keys = [
        "direction", "bias", "speed", "CP", "CI",
        "rangex", "rangey", "rangez", "dx", "dy", "dz",
        "offsetx", "offsety", "rotation",
        "resy", "resy",
        "time", "time_passed", "timestamp_start", "timestamp_end",
    ]
    if verbose:
        keys.extend([
            "title", "comment", "value", "original_path",
            "hardware_setup"
        ])
    img = STMImage(fname)
    for key in keys:
        print(f"{key:>25}: {img.get_field_string(key)}")
    if verbose:
        for k, v in img.metadata.items():
            print(f"{k:>25}: {v}")



def print_sizes(fname):
    for dim in STMImage(fname).dimensions:
        print(f"{dim.name:>25}: {dim.size}")


def inspect(fname):
    for var in STMImage(fname).variables:
        print(var.name)
        print(var.__dict__)
        print(f"    type: {var.dtype}, dimensions: {var.dimensions}, "
              f"shape: {var.shape}")
        data = var[:]
        if var.dtype.type is np.bytes_:
            bstr = data[~data.mask].tobytes()
            try:
                print(bstr.decode("utf-8"))
            except UnicodeDecodeError:
                print(bstr)
        else:
            print(data)
