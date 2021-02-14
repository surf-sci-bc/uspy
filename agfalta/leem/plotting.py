"""Plotting helper functions."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=too-many-arguments

import os.path
import itertools
from collections import abc

import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import skvideo.io
from IPython.display import display, Video

from agfalta.leem.utility import stackify, imgify
from agfalta.leem.processing import roify, get_max_variance_idx



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
             figsize=None, ticks=False, log=False, **kwargs):
    """Plots a single LEEM image with some metadata. If ax is given,
    the image is plotted onto that axes object. Takes either
    a file name or a LEEMImg object. Fields given are shown in the
    corners of the image."""
    img = imgify(img)
    if title is None and img.path != "NO_PATH":
        title = img.path
    if title and len(title) > 25:
        title = f"...{title[-25:]}"
    if isinstance(fields, str):
        fields = [fields]

    ax = _get_ax(ax, figsize=figsize, ticks=ticks, title=title)
    if log:
        data = np.log(img.data)
    else:
        data = img.data
    ax.imshow(
        data,
        cmap="gray",
        clim=(np.nanmin(data), np.nanmax(data)),
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
            x=_MPL_IMG_POS[i][2], y=_MPL_IMG_POS[i][3],
            s=img.get_field_string(field),
            va=_MPL_IMG_POS[i][0], ha=_MPL_IMG_POS[i][1],
            transform=ax.transAxes, color="yellow", fontsize=14,
        )
    return ax

def plot_image(*args, **kwargs):
    """Alias for agfalta.leem.plotting.plot_img()."""
    print("plot_image() is DEPRECATED, use plot_img() instead")
    return plot_img(*args, **kwargs)



def make_video(stack, ofile, skip=None, fps=24,
               overwrite=True, scale=0.5, contrast="auto",
               fields=None, log=False, **kwargs):
    """
    Save a movie as an mp4 file and display it.
    * ofile sets the output file, which should have an .mp4 extension. It
      can also be in a subfolder, for example like this:
      ofile="subfolder/movie.mp4"
    * fields works the same way as for plot_img
    * contrast can either be "auto" (each image is autoscaled),
      or "maximum" where everything is scaled according to the most extreme
      intensity values (may be slow)
      or an integer, in which case the min/max values are taken from that
      image number in the stack
      or two numbers (as list or tuple) that are the min/max contrast values
    * fps sets the fps of the output video (i.e. speed)
    * scale will make the video larger/smaller (default scale=0.5 will
      produce half-sized videos)
    * skip=n will use only every n-th frame
    * if overwrite is set to False, make_video will be faster and only
      display the video file that is given but not update it
    """
    if not overwrite and os.path.isfile(ofile):
        return Video(ofile)
    # make input sane
    stack = stackify(stack, virtual=True)
    if "increment" in kwargs:
        skip = kwargs.pop("increment")
    if skip:
        stack = stack[::skip]
    if isinstance(fields, str):
        fields = [fields]

    # find the contrast values
    if contrast == "auto":
        print("WARNING: The video is on auto contrast.")
    elif contrast == "maximum":
        c0, c1 = 2e16, 0
        for img in stack:
            c0 = min(c0, np.nanmin(img.data))
            c1 = max(c1, np.nanmax(img.data))
        contrast = sorted((c0, c1))
    elif isinstance(contrast, int):
        c0 = np.nanmin(stack[contrast].data)
        c1 = np.nanmax(stack[contrast].data)
        contrast = sorted((c0, c1))
    elif not isinstance(contrast, abc.Iterable) and not len(contrast) == 2:
        raise ValueError(f"Invalid '{contrast=}'")
    if contrast != "auto" and log:
        c0 = np.log(contrast[0])
        c1 = np.log(contrast[1])
        contrast = sorted((c0, c1))

    # set up ffmpeg
    ffmpeg_indict = {"-r": str(fps)}
    ffmpeg_outdict = {
        "-vf": f"scale=iw*{scale}:-1",
        "-vcodec": "libx264",
        "-pix_fmt": "yuv420p",
        "-profile:v": "high",
    }
    writer = skvideo.io.FFmpegWriter(
        ofile, inputdict=ffmpeg_indict, outputdict=ffmpeg_outdict
    )

    # loop through the images
    height, width = stack[0].data.shape
    for img in stack:
        data = img.data
        if log:
            data = np.log(data)
        # set contrast
        if contrast == "auto":
            data = np.nan_to_num(data)
            data = cv2.normalize(
                data, np.ones_like(data, dtype=np.uint8),
                0, 255, norm_type=cv2.NORM_MINMAX
            )
        else:
            data = (data - contrast[0]) / (contrast[1] - contrast[0]) * 255
            data = np.clip(data, 0, 255).astype(np.uint8)
        # write metadata
        if fields:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
            for i, field in enumerate(fields):
                text = img.get_field_string(field).encode("ascii", errors="ignore").decode()
                for pos, char in enumerate(text):
                    xy = _CV2_IMG_POS(i, height, width, text, pos=pos)
                    data = cv2.putText(
                        data, char,
                        org=xy, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 255, 0), thickness=2,
                        lineType=cv2.LINE_AA
                    )
        writer.writeFrame(data)
    writer.close()

    # return display in jupyter
    return Video(ofile)


def plot_mov(stack, ncols=4, virtual=True, skip=None, **kwargs):
    """Uses plot_img() to plot LEEMImges on axes objects in a grid.
    Takes either a file name, folder name or LEEMStack object."""
    stack = stackify(stack, virtual=virtual)
    if "increment" in kwargs:
        skip = kwargs.pop("increment")
    if skip:
        stack = stack[::skip]

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
    print("plot_movie() is DEPRECATED, use plot_mov() instead")
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
    try:
        stack = stackify(stack)
    except FileNotFoundError:
        stack = stackify([imgify(stack)])

    table = []
    for img in stack:
        row = [os.path.basename(img.path)]
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

    ax = _get_ax(ax, xlabel=xaxis, ylabel="Intensity in a.u.")
    x = getattr(stack, xaxis)
    for roi in rois:
        intensity = np.zeros(len(stack))
        for i, img in enumerate(stack):
            intensity[i] = np.mean(roi.apply(img.data))
        ax.plot(x, intensity)

    return ax, roi

def plot_iv(*args, **kwargs):
    """Alias for agfalta.leem.plotting.plot_intensity() with xaxis set
    to "energy"."""
    return plot_intensity(*args, xaxis="energy", **kwargs)


def plot_intensity_img(stack, *args, xaxis="rel_time", img_idx=None, **kwargs):
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


def plot_rois(img, *args, **kwargs):
    img = imgify(img)
    rois = roify(*args, **kwargs)
    ax = plot_img(img, ticks=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, roi in enumerate(rois):
        ax.add_artist(roi.artist(colors[i]))


# Utility:

_MPL_IMG_POS = {    # (verticalalignment, horizontalalignment, x, y)
    0: ("top", "left", 0.01, 1),
    1: ("top", "right", 1, 1),
    2: ("bottom", "left", 0.01, 0.01),
    3: ("bottom", "right", 1, 0.01),
}


_CV2_CONTRASTS = {
    "auto": cv2.NORM_MINMAX,
}
def _CV2_IMG_POS(idx, height, width, text="", pos=0):
    """Applies only for FONT_HERSHEY_SIMPLEX (~20px wide chars)"""
    margin = 0.01
    charwidth, charheight = 20, 23
    x = width
    y = height
    if idx in (0, 2):
        x *= margin
    if idx in (1, 3):
        x = x * (1 - margin) - len(text) * charwidth
    x += pos * charwidth
    if idx in (0, 1):
        y = y * margin + charheight
    if idx in (2, 3):
        y = y * (1 - margin)
    return int(x), int(y)


def _get_ax(ax, **kwargs):
    """Helper for preparing an axis object"""
    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.get("figsize", None))
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
