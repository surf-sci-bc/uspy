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
import matplotlib.colors
import pandas as pd
import skvideo.io
from IPython.display import display, Video

from agfalta.leem.utility import stackify, imgify
from agfalta.leem.processing import roify, get_max_variance_idx
from agfalta.leem.driftnorm import normalize_image



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


def plot_img(img, fields=("temperature", "pressure", "energy", "fov"), field_color=None,
             mcp=None, dark_counts=100, contrast=None, invert=False, log=False,
             ax=None, title=None, figsize=None, dpi=100, ticks=False,
             cutout_diameter=None, **kwargs):
    """Plots a single LEEM image with some metadata. Takes either a file name
    or a LEEMImg object. Metadata fields given are shown in the corners of the
    image.

    Optional keyword arguments:
    - fields:
      List of metadata fields to show in the image. Set to None to disable.
      Maximum 4 fields show up: (topleft, topright, bottomleft, bottomright)
    - field_color:
      Color of field labels (defaults to yellow normally, defaults to black if
      invert or cutout_diameter is set)
    - mcp:
      Filename of a channelplate background file (will be divided by)
    - dark_counts:
      Works only together with mcp -- subtracts dark_counts from
      the image and from the background file before dividing them (default 100)
    - contrast:
      "auto": Use the maximum dynamic range of the image.
      "inner": Same as "auto", but it will only use the inner 60 % of the image.
      (n, m) -- two numbers: Use those values as lower/upper intensity boundary
    - invert:
      Invert grayscale
    - log:
      Whether to show the logarithmic (useful for LEED)
    - ax:
      If given, use this matplotlib axes to plot on. Else make a new one.
    - title:
      Defaults to the filename, set it to an empty string "" to disable it.
      (only if ax is not given)
    - figsize:
      Size of image in inches (only if ax is not given)
    - dpi:
      Image resolution (only if ax is not given)
    - ticks:
      Whether to show x and y coordinates for the pixels (only if ax is not given)
    - cutout_diameter:
      Cut away the beam tube for publication-ready images:
      A value of 1 means to use the biggest circle that fits in the image,
      lower values mean smaller cutouts. Also sets the "fields" font color
      to black
    All other arguments will be passed to matplotlib's plot() function (e.g. linewidth)
    Returns an axes object.
    """
    img = imgify(img)
    if mcp is not None:
        img = normalize_image(img, mcp=mcp, dark_counts=dark_counts)
    if title is None and img.path != "NO_PATH":
        title = img.path
    if title and len(title) > 25:
        title = f"...{title[-25:]}"
    if isinstance(fields, str):
        fields = [fields]

    ax = _get_ax(
        ax, figsize=figsize, dpi=dpi,
        ticks=ticks, title=title, axis_off=True
    )

    data = np.nan_to_num(img.data)
    if log:
        data = np.log(data)

    if cutout_diameter:
        h, w = data.shape
        radius = min(h, w) / 2 * cutout_diameter
        y, x = np.arange(0, h), np.arange(0, w)
        y, x = y[:, np.newaxis], x[np.newaxis, :]
        mask = (x - w/2)**2 + (y - h/2)**2 > radius**2
        data = np.ma.masked_where(mask, data)


    if contrast in ("auto", "maximum"):
        contrast = data.min(), data.max()
    elif contrast == "inner":
        dy, dx = map(int, 0.2 * np.array(data.shape))
        inner = img.data[dy:-dy, dx:-dx]
        contrast = inner.min(), inner.max()
    if isinstance(contrast, abc.Iterable) and len(contrast) == 2:
        data = np.clip(data, contrast[0], None) - contrast[0]
        data = data / (contrast[1] - contrast[0]) * 255
        data = np.clip(data, 0, 255).astype(np.uint8)
    elif contrast is not None:
        raise ValueError(f"Invalid '{contrast=}'")

    if invert:
        data = -data + data.max()

    ax.imshow(
        data,
        cmap="gray",
        clim=(np.nanmin(data), np.nanmax(data)),
        aspect=1,
        **kwargs
    )

    if fields is None:
        return ax

    if field_color is None:
        field_color = "yellow"
        if cutout_diameter or invert:
            field_color = "black"
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
            transform=ax.transAxes, color=field_color, fontsize=14,
        )
    return ax

def plot_image(*args, **kwargs):
    """Alias for agfalta.leem.plotting.plot_img()."""
    print("plot_image() is DEPRECATED, use plot_img() instead")
    return plot_img(*args, **kwargs)



def make_video(stack, ofile, skip=None,
               fields=None, field_color=None,
               mcp=None, dark_counts=100, contrast="auto", invert=False, log=False,
               fps=24, overwrite=True, scale=0.5,
               **kwargs):
    """
    Save a movie as an mp4 file and display it.
    - stack:
      Either a LEEMStack object, a path to a folder that contains *.dat images,
      the path to an image file that contains all images (e.g. *.tif), or a list of
      LEEMImg objects
    - ofile:
      The output filename, which should have an .mp4 extension. It can also be in a
      subfolder, for example like this: ofile="subfolder/movie.mp4"

    Optional keyword arguments:
    - skip:  (equivalent to "increment")
      If set to an integer n, it will use only every n-th frame
      (faster than increasing fps)
    - fields: (see also plot_img())
      List of metadata fields to show in the image. Set to None to disable.
      Maximum 4 fields show up: (topleft, topright, bottomleft, bottomright)
    - field_color: (see also plot_img())
      Color of field labels (defaults to yellow normally, defaults to black if
      invert or cutout_diameter is set)
    - mcp:
      Filename of a channelplate background file (will be divided by)
    - dark_counts:
      Works only together with mcp -- subtracts dark_counts from
      the image and from the background file before dividing them (default 100)
    - contrast:
      "auto": Use the maximum dynamic range for EACH image (default).
      "inner": Same as "auto", but it will only use the inner 60 % of each image.
      "maximum": Use the maximum dynamic range of the whole stack.
      n -- integer: Use the dynamic range of the n-th image in the stack
      (n, m) -- two numbers: Use those values as lower/upper intensity boundary
    - invert:
      Invert grayscale
    - log:
      Whether to show the logarithmic (useful for LEED)
    - fps:
      An integer, the fps of the output video (i.e. speed)
    - overwrite:
      If it is set to False, make_video() does not recalculate the video but only
      show the one given by "ofile" (all other parameters are ignored if ofile
      exists). Faster if you don't need to change anything.
    - scale:
      Will make the video larger/smaller by changing the resolution (default 0.5)

    Returns an iPython.display Video object
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
    if contrast in ("auto", "inner"):
        contrast_type = contrast
        print("WARNING: The video is on auto contrast.")
    elif contrast == "maximum":
        c0, c1 = 2e16, 0
        for img in stack:
            c0 = min(c0, np.nanmin(img.data))
            c1 = max(c1, np.nanmax(img.data))
        contrast = sorted((c0, c1))
        print(f"Set contrast to {contrast}")
        contrast_type = "static"
    elif isinstance(contrast, int):
        c0 = np.nanmin(stack[contrast].data)
        c1 = np.nanmax(stack[contrast].data)
        contrast = sorted((c0, c1))
        print(f"Set contrast to {contrast}")
        contrast_type = "static"
    elif isinstance(contrast, abc.Iterable) and len(contrast) == 2:
        contrast_type = "static"
    else:
        raise ValueError(f"Invalid '{contrast=}'")

    if contrast_type == "static" and log:
        c0 = min(np.log(contrast[0]), 1)
        c1 = min(np.log(contrast[1]), 1)
        contrast = sorted((c0, c1))
        print(f"Contrast set to {c0} -- {c1}")

    # set up ffmpeg
    ffmpeg_indict = {"-r": str(fps)}
    ffmpeg_outdict = {
        "-vf": f"scale=ceil(iw/2*{scale})*2:-2",
        "-vcodec": "libx264",
        "-pix_fmt": "yuv420p",
        "-profile:v": "high",
    }
    writer = skvideo.io.FFmpegWriter(
        ofile, inputdict=ffmpeg_indict, outputdict=ffmpeg_outdict#, verbosity=1
    )

    # loop through the images
    height, width = stack[0].data.shape
    dy, dx = map(int, 0.2 * np.array(stack[0].data.shape))

    if field_color is None:
        field_color = "yellow"
        if invert:
            field_color = "black"
    color = tuple(int(rgb * 255) for rgb in matplotlib.colors.to_rgb(field_color))

    for img in stack:
        if mcp is not None:
            img = normalize_image(img, mcp=mcp, dark_counts=dark_counts)
        data = np.nan_to_num(img.data)
        if log:
            data = np.nan_to_num(np.log(data))
        # set contrast
        if contrast_type == "inner":
            inner = data[dy:-dy, dx:-dx]
            contrast = inner.min(), inner.max()
        if contrast_type == "auto":
            contrast = data.min(), data.max()
        data = cv2.normalize(
            np.clip(data, contrast[0], contrast[1]),
            np.ones_like(data, dtype=np.uint8),
            0, 255, norm_type=cv2.NORM_MINMAX
        )

        if invert:
            data = -data + data.max()

        if fields:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
            for i, field in enumerate(fields):
                text = img.get_field_string(field).encode("ascii", errors="ignore").decode()
                for pos, char in enumerate(text):
                    xy = _CV2_IMG_POS(i, height, width, text, pos=pos)
                    data = cv2.putText(
                        data, char,
                        org=xy, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=color, thickness=2,
                        lineType=cv2.LINE_AA
                    )
        writer.writeFrame(data)
    writer.close()

    # return display in jupyter
    return Video(ofile)


def plot_mov(stack, skip=None, ncols=4, virtual=True, dpi=100, **kwargs):
    """
    Uses plot_img() to plot LEEMImges on axes objects in a grid.
    - stack:
      Either a LEEMStack object, a path to a folder that contains *.dat images,
      the path to an image file that contains all images (e.g. *.tif), or a list of
      LEEMImg objects

    Optional keyword arguments:
    - skip:  (equivalent to "increment")
      If set to an integer n, it will use only every n-th frame
      (faster than increasing fps)
    - ncols:
      Number of columns of the grid.
    For more arguments, see the help text for plot_img(). All arguments in there
    can also be given.
    Returns an array of the axes objects.
    """
    stack = stackify(stack, virtual=virtual)
    if "increment" in kwargs:
        skip = kwargs.pop("increment")
    if skip:
        stack = stack[::skip]

    nrows = math.ceil(len(stack) / ncols)
    figsize = kwargs.pop("figsize", (ncols * 5, nrows * 5))
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, dpi=dpi)

    # zip_longest() pads the shorter list with None
    for ax, img in itertools.zip_longest(axes.flatten(), stack):
        if img is None:
            fig.delaxes(ax)
            continue
        plot_img(img, ax=ax, **kwargs)

    return axes.flatten()

def plot_movie(*args, **kwargs):
    """Alias for agfalta.leem.plotting.plot_mov()."""
    print("plot_movie() is DEPRECATED, use plot_mov() instead")
    return plot_mov(*args, **kwargs)


def plot_meta(stack, fields="temperature"):
    """Plots some metadata of the stack over time. Returns an array of the
    axes objects."""
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

    return axes


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


def plot_intensity(stack, *args, xaxis="rel_time", ax=None, **kwargs):
    """Plots the image intensity in a specified ROI over a specified
    x axis. The x axis can be any attribute of the stack.
    Either you give:
    - The ROI object itself (from leem.ROI())
    - A list of ROI objects
    - The parameters for a ROI:
        x0=XX, y0=XX and one of these:
        * type_="circle", radius=XX                 (default if omitted)
        * type_="rectangle", width=XX, height=XX
        * type_="ellipse", xradius=XX, yradius=XX
    Examples:
        plot_intensity(stack, x0=300, y0=200, radius=3)
        plot_intensity(stack, x0=300, y0=100, width=15, height=20)  # makes rectangle
        roi = leem.ROI(300, 200, radius=10)
        plot_intensity(stack, roi)
        plot_intensity(stack, x0=100, y0=50, type_="ellipse", xradius=5, xradius=4)
        roi2 = leem.ROI(200, 100, radius=5)
        plot_intensity(stack, (roi, roi2))

    Returns the axes object
    """
    ax = _get_ax(ax, xlabel=xaxis, ylabel="Intensity in a.u.")
    data = get_intensity(stack, *args, rois, **kwargs)
    x = data[0]
    for intensity in data[1:]:
        ax.plot(x, intensity)
    return ax

def get_intensity(stack, *args, xaxis="rel_time", ofile=None, **kwargs):
    """Calculate intensity profile along a stack in a given ROI (or multiple ROIs).
    ROIs are supplied in the same way as for plot_intensity().
    Returns a 2D-numpy array: The first row contains the value of "xaxis",
    every following row contains the intensity along one of the ROIs.
    If ofile is given, the result is saved in csv format under the given file name.
    """
    stack = stackify(stack)
    rois = roify(*args, **kwargs)
    x = getattr(stack, xaxis)
    cols = [x]
    for roi in rois:
        cols.append(np.zeros(len(stack)))

    for i, img in enumerate(stack):
        for j, roi in enumerate(rois):
            cols[j + 1][i] = roi.apply(img.data).sum() / roi.area

    data = np.stack(cols)
    if ofile is not None:
        np.savetxt(
            ofile, data.T,
            header=f"{xaxis} | " + " | ".join(str(roi) for roi in rois)
        )
    return data


def plot_iv(*args, **kwargs):
    """Alias for agfalta.leem.plotting.plot_intensity() with xaxis set
    to "energy"."""
    return plot_intensity(*args, xaxis="energy", **kwargs)


def plot_intensity_img(stack, *args, xaxis="rel_time", img_idx=None, **kwargs):
    """Does the same thing as agfalta.leem.plotting.plot_intensity()
    but also shows an image of the stack and the ROI on it on the right.
    Returns 2 axes objects: The first one contains the plot, the second one the image.
    """
    stack = stackify(stack)
    rois = roify(*args, **kwargs)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if img_idx is None:
        img_idx = get_max_variance_idx(stack)
    plot_img(stack[img_idx], ax=ax2, ticks=True)
    plot_rois(rois, ax=ax2)

    for roi in rois:
        plot_intensity(stack, roi, ax=ax1, xaxis=xaxis)

    return ax1, ax2

def plot_iv_img(*args, **kwargs):
    """Alias for agfalta.leem.plotting.plot_intensity_img() with xaxis set
    to "energy"."""
    return plot_intensity_img(*args, xaxis="energy", **kwargs)


def plot_rois(*args, img=None, ax=None, **kwargs):
    """Plot rois onto a given axes object. The ROI can either be given like in
    plot_intensity()."""
    rois = roify(*args, **kwargs)
    if img is not None:
        img = imgify(img)
        ax = plot_img(img, ticks=True)
    elif ax is None:
        ax = _get_ax(ax)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, roi in enumerate(rois):
        ax.add_artist(roi.artist(colors[i]))
    return ax


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
        _, ax = plt.subplots(
            figsize=kwargs.get("figsize", None),
            dpi=kwargs.get("dpi", 100)
        )
    if not kwargs.get("ticks", True):
        ax.set_xticks([])
        ax.set_yticks([])
        if kwargs.get("axis_off", False):
            ax.set_axis_off()
    if kwargs.get("title", False):
        ax.set_title(kwargs["title"])
    if kwargs.get("xlabel", False):
        ax.set_xlabel(kwargs["xlabel"])
    if kwargs.get("ylabel", False):
        ax.set_ylabel(kwargs["ylabel"])
    return ax
