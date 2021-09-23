"""Plotting helper functions."""
# pylint: disable=unused-import

from agfalta.leem.base import LEEMImg
from agfalta.plotting import plot_img as base_plot_img
from agfalta.plotting import make_video, plot_mov


def plot_img(img, *args, **kwargs):
    """See agfalta.plotting.plot_img"""
    img = LEEMImg(img)
    return base_plot_img(img, *args, **kwargs)
