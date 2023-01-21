"""Plotting helper functions."""
# pylint: disable=unused-import
# pylint: disable=function-redefined

from uspy.leem.base import LEEMImg
from uspy.leem.utility import imgify, stackify

import uspy.plotting

# from uspy.plotting import plot_img as base_plot_img
# from uspy.plotting import plot_intensity as base_plot_intensity

# from uspy.plotting import (
#     make_video,
#     plot_mov,
#     plot_line,
#     plot_intensity,
#     plot_intensity_img,
# )

from uspy.plotting import *


def plot_img(
    img, *args, fields=("temperature", "pressure1", "energy", "fov"), **kwargs
):
    """See uspy.plotting.plot_img"""
    img = imgify(img)
    return uspy.plotting.plot_img(img, fields=fields, *args, **kwargs)


def make_video(
    stack, *args, fields=("temperature", "pressure1", "energy", "fov"), **kwargs
):
    stack = stackify(stack)
    return uspy.plotting.make_video(stack, fields=fields, *args, **kwargs)


def plot_aligns(stack, xaxis=None):
    stack = stackify(stack)
    if xaxis is None:
        plt.plot([m[0:2, 2] for m in stack.warp_matrix])
        plt.xlabel("Index")
    else:
        plt.plot(getattr(stack, xaxis), [m[0:2, 2] for m in stack.warp_matrix])
        plt.xlabel(xaxis)
    plt.show()
