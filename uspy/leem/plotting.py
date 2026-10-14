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


def plot_img(img, fields=None, *args, **kwargs):
    """See uspy.plotting"""
    img = imgify(img)
    return uspy.plotting.plot_img(img, fields, *args, **kwargs)


_UNSET = object()


def make_video(stack, *args, fields=_UNSET, **kwargs):
    # virtual=True: a path/glob/list source shouldn't be forced eager just to reach
    # uspy.plotting.make_video, which already builds (and streams) a virtual stack itself --
    # an already-constructed LEEMStack (any virtuality) passes through stackify unchanged.
    stack = stackify(stack, virtual=True)
    # distinguish "fields not passed" (-> use the stack's configured defaults) from an
    # explicit fields=None (-> uspy.plotting.make_video treats that as "no overlay")
    if fields is _UNSET:
        fields = stack[0].default_fields
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
