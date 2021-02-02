"""Plotting helper functions."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=too-many-arguments

from os.path import basename
import itertools

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import ipywidgets as widgets
from IPython.display import display

from agfalta.leem.utility import stackify, imgify
from agfalta.leem.processing import ROI, roify, get_max_variance_idx



def info(obj):
    """Prints info about a LEEM image or stack."""
    try:
        img = imgify(obj)
        print("Image attributes:")
        for attr in img.attrs:
            print(f"\t{attr}: {img.get_field_string(attr)}")
        return
    except FileNotFoundError:
        pass
    try:
        stack = stackify(obj)
        print("Stack attributes:")
        for attr in stack.unique_attrs:
            if attr.startswith("_"):
                continue
            val = str(getattr(stack, attr))
            if len(val) > 100:
                val = val[:80] + "..."
            print(f"\t{attr}: {val}")
        if len(stack) == 0:
            print("\tStack object is empty")
            return
        print("Stack attributes from its images (only value of first "
              "image is given):")
        for attr in stack[0].attrs:
            print(f"\t{attr}, {stack[0].get_field_string(attr)}")
    except FileNotFoundError:
        print("Object is not valid (maybe wrong path?)")


def plot_img(img, ax=None, title=None,
             fields=("temperature", "pressure1", "energy", "fov"),
             figsize=(6, 6), ticks=False, **kwargs):
    """Plots a single LEEM image with some metadata. If ax is given,
    the image is plotted onto that axes object. Takes either
    a file name or a LEEMImg object. Fields given are shown in the
    corners of the image."""
    img = imgify(img)
    if title is None and img.path != "NO_PATH":
        title = img.path
    if title is not None and len(title) > 25:
        title = f"...{title[-25:]}"

    ax = _get_ax(ax, figsize=figsize, ticks=ticks, title=title)
    ax.imshow(
        img.data,
        cmap="gray",
        clim=(np.nanmin(img.data), np.nanmax(img.data)),
        aspect=1,
        **kwargs
    )

    if fields is None:
        return ax
    for i, field in enumerate(fields):
        if i > 3:
            print(f"Ignoring field {field}, not enough space")
            continue
        if field is None:
            continue
        ax.text(
            x=IMG_POS[i][2], y=IMG_POS[i][3], 
            s=img.get_field_string(field),
            va=IMG_POS[i][0], ha=IMG_POS[i][1],
            transform=ax.transAxes, color="yellow", fontsize=14,
        )
    return ax

def plot_image(*args, **kwargs):
    """Alias for agfalta.leem.plotting.plot_img()."""
    return plot_img(*args, **kwargs)


def plot_mov(stack, cols=4, virtual=False, **kwargs):
    """Uses plot_img() to plot LEEMImges on axes objects in a grid.
    Takes either a file name, folder name or LEEMStack object."""
    stack = stackify(stack, virtual=virtual)

    ncols = 4
    nrows = math.ceil(len(stack) / ncols)
    fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=(ncols * 5, nrows * 5)
    )

    # zip_longest() pads the shorter list with None
    for ax, img in itertools.zip_longest(axes.flatten(), stack):
        if img is None:
            fig.delaxes(ax)
        else:
            plot_img(img, ax=ax, **kwargs)

def plot_movie(*args, **kwargs):
    """Alias for agfalta.leem.plotting.plot_mov()."""
    return plot_mov(*args, **kwargs)


def plot_meta(stack, fields="temperature"):
    """Plots some metadata of the stack over time."""
    stack = stackify(stack)
    if isinstance(fields, str):
        fields = [fields]

    fig, axes = plt.subplots(len(fields), figsize=(6, len(fields) * 3))
    fig.subplots_adjust(hspace=0.3)
    # Reshape in case the supplot has only one plot, so it stays iterable
    axes = np.array(axes).reshape(-1)

    time = stack.rel_time
    for i, field in enumerate(fields):
        val = getattr(stack, field)
        if field == "temperature":
            axes[i].plot(time[val < 2000], val[val < 2000])
            if len(time[val < 2000]) < len(time):
                print("Points have been excluded from plot because of unreasonably "
                      "high temperature")
        else:
            axes[i].plot(time, val)
        if field in ("pressure", "pressure1", "pressure2"):
            axes[i].set_yscale('log')
        axes[i].set_ylabel(f"{field} in {stack[0].get_unit(field)}")
        axes[i].set_xlabel("Time in s")


def print_meta(stack, fields=("energy", "temperature", "pressure1",
                              "pressure2", "objective", "fov",
                              "exposure", "averaging", "width", "height")):
    """Prints metadata of a stack as a table."""
    if isinstance(fields, str):
        fields = [fields]
    stack = stackify(stack)

    table = []
    for img in stack:
        row = [basename(img.path)]
        for field in fields:
            row.append(img.get_field_string(field))
        table.append(row)
    pd.set_option('display.expand_frame_repr', False)
    df = pd.DataFrame(table)
    df.columns = ("Name", *fields)
    display(df)


def plot_intensity(stack, *args, xaxis="energy", ax=None, **kwargs):
    """Plots the image intensity in a specified ROI over a specified
    x axis. The x axis can be any attribute of the stack.
    Either you give the ROI object itself or the parameters for a ROI
    The ROI is defined by its center x0, y0 and either of:
    - type_="circle", radius=XX
    - type_="rectangle", width=XX, height=XX
    - type_="ellipse", xradius=XX, yradius=XX
    You can omit type_, then it selects a circle. The radius can then
    also be omitted, defaulting to 10.
    Examples:
        plot_intensity(stack, x0, y0, radius=3)
        plot_intensity(stack, roi)
        plot_intensity(stack, x0, y0, type_="rectangle", width=5, height=4)
    Returns a tuple of the axes object and the ROI
    """
    stack = stackify(stack)
    rois = roify(*args, **kwargs)
    if len(rois) > 1:
        raise ValueError("Cant handle multiple ROIs")
    roi = rois[0]
    #TODO add support for multiple ROIs

    x = getattr(stack, xaxis)

    intensity = np.zeros(len(stack))
    for i, img in enumerate(stack):
        for roi in rois:
            intensity[i] = np.mean(roi.apply(img.data))

    ax = _get_ax(ax, xlabel=xaxis, ylabel="Intensity in a.u.")
    ax.plot(x, intensity)
    return ax, roi

def plot_iv(*args, **kwargs):
    """Alias for agfalta.leem.plotting.plot_intensity() with xaxis set
    to "energy"."""
    return plot_intensity(*args, xaxis="energy", **kwargs)


def plot_intensity_img(stack, *args, xaxis="energy", img_idx=None, **kwargs):
    """Does the same thing as agfalta.leem.plotting.plot_intensity() 
    but also shows an image of the stack and the ROI on it on the right.
    """
    stack = stackify(stack)
    rois = roify(*args, **kwargs)
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if img_idx is None:
        img_idx = get_max_variance_idx(stack)
    plot_img(stack[img_idx], ax=ax2, ticks=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, roi in enumerate(rois):
        plot_intensity(stack, roi, ax=ax1, xaxis=xaxis)
        ax2.add_artist(roi.artist(colors[i]))

    return ax1, ax2

def plot_iv_img(*args, **kwargs):
    """Alias for agfalta.leem.plotting.plot_intensity_img() with xaxis set
    to "energy"."""
    return plot_intensity_img(*args, xaxis="energy", **kwargs)



# Utility:

IMG_POS = {    # (verticalalignment, horizontalalignment, x, y)
    0: ("top", "left", 0.01, 1),
    1: ("top", "right", 1, 1),
    2: ("bottom", "left", 0.01, 0.01),
    3: ("bottom", "right", 1, 0.01),
}


def _get_ax(ax, **kwargs):
    """Helper for preparing an axis object"""
    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.get("figsize", (6.4, 4.8)))
    if not kwargs.get("ticks", True):
        ax.set_xticks([])
        ax.set_yticks([])
    if kwargs.get("title", False):
        ax.set_title(kwargs["title"])
    if kwargs.get("xlabel", False):
        ax.set_xlabel(kwargs["xlabel"])
    if kwargs.get("ylabel", False):
        ax.set_ylabel(kwargs["ylabel"])
    return ax
