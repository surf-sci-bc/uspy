"""Plotting helper functions."""
# pylint: disable=unused-import

from uspy.leem.base import LEEMImg
from uspy.plotting import plot_img as base_plot_img
from uspy.plotting import (
    make_video,
    plot_mov,
    plot_line,
    plot_intensity,
    plot_intensity_img,
)


def plot_img(img, *args, **kwargs):
    """See uspy.plotting.plot_img"""
    img = LEEMImg(img)
    return base_plot_img(img, *args, **kwargs)
