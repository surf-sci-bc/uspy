"""Plotting helper functions."""
# pylint: disable=too-many-arguments
# pylint: disable=invalid-name


from __future__ import annotations
from typing import Union, Optional
from collections.abc import Iterable
from numbers import Number
import os.path
import itertools
import math

import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.colors
import skvideo.io
from IPython.display import display, Video

import agfalta.dataobject as do
from agfalta.leem.utility import stackify, imgify
from agfalta.leem.processing import roify, get_max_variance_idx, ROI, RSM
from agfalta.leem.driftnorm import normalize_image


def plot_img(
    img: Union[do.Image, str],
    fields: Optional[Union[str, Iterable[str]]] = None,
    field_color: Optional[str] = None,
    contrast: Optional[Union[str, Iterable[Number, Number]]] = None,
    invert: bool = False,
    log: bool = False,
    cmap: str = "gray",
    ax: Optional[mpl.axes.Axes] = None,
    title: Optional[Union[str, bool]] = None,
    figsize: Optional[Iterable[Number, Number]] = None,
    dpi: int = 100,
    ticks: bool = False,
    mask: Union[ROI, bool] = False,
    **kwargs,
) -> mpl.axes.Axes:
    """Plots a single image with some metadata.

    Uses matplotlib's imshow() to display an image. It either takes a dataobject.Image or just
    the path to an image as mandatory argument. Optional arguments change the plot
    appearance.

    Parameters
    ----------
    img : Union[do.Image, str]
        The image that will be displayed.
    fields : Union[str, Iterable[str]], optional
        Iterable of metadata field strings. The corresponding metadata will be displayed in the
        image corners (maximum 4 fields: topleft, topright, bottomleft, bottomright).
        Defaults to sensible values for the type of image that is put in.
    field_color : str, optional
        Font color of the field strings in the corners, by default yellow or black, depending on
        the background.
    contrast : Union[str, Iterable[Number, Number]], optional
        Sets the lower and upper intensity value of the image. None and "auto" will use the
        full dynamical range. "inner" does the same but disregards the outer 20% of the image.
        Alternatively, an iterable with two numbers for the lower and upper boundary is given
        directly. Defaults to None.
    invert : bool, optional
        If True, image intensity is inverted. By default False.
    log : bool, optional
        If True, plots the logarithmic of the image. By default False.
    ax : mpl.axes.Axes, optional
        If given, use this matplotlib axes object to plot on. Otherwise create a new one.
    title : Union[str, bool], optional
        Sets an image title. Defaults to the image's source (usually the path). When Latex is
        enabled for matplotlib, this is set to False.
    figsize : Iterable[Number, Number], optional
        Size of the figure to plot on. Has no effect if a preexisting ax is given (see above).
    dpi : int, optional
        Plot DPI, by default 100. Has no effect if a preexisting ax is given (see above).
    ticks : bool, optional
        If True, show x and y coordinates for the pixels on each axis, by default False.
    mask : Union[ROI, bool], optional
        If given, use a ROI to mask out parts of the image. If set to True, take the default
        mask of the image class. By default False.
    cmap : str, optional
        The colormap that matplotlib should use. Defaults to grayscale.
    **kwargs :
        All other arguments will be passed to matplotlib's plot() function (e.g., linewidth).

    Returns
    -------
    ax : mpl.axes.Axes
        The matplotlib axes object that the plot was drawn on.
    """

    if isinstance(img, str):
        img = do.Image(img)

    if title is None and mpl.rcParams["text.usetex"]:
        title = False
    elif title is True:
        title = str(img.source)
        if len(title) > 25:
            title = f"...{title[-25:]}"

    if isinstance(fields, str):
        fields = [fields]

    ax = _get_ax(ax, figsize=figsize, dpi=dpi, ticks=ticks, title=title, axis_off=True)

    data = np.nan_to_num(img.image)
    if mask is True:
        mask = img.default_mask
    if isinstance(mask, ROI):
        data = mask.apply(data, return_array=True)

    if log:
        data = np.ma.log(data)

    if contrast is None or contrast in ("auto", "maximum"):
        contrast = data.min(), data.max()
    elif contrast == "inner":
        dy, dx = map(int, 0.2 * np.array(data.shape))
        inner = img.image[dy:-dy, dx:-dx]
        contrast = inner.min(), inner.max()
    elif log:
        # contrast = np.where(np.array(contrast) > 0, np.log(contrast), 0)
        contrast = np.array(contrast)
        contrast[contrast > 0] = np.log(contrast[contrast > 0])

    data = np.clip(data, contrast[0], None) - contrast[0]
    data = data / (contrast[1] - contrast[0]) * 255
    data = np.clip(data, 0, 255).astype(np.uint8)

    if invert:
        data = -data + data.max()

    ax.imshow(
        data, cmap=cmap, clim=(np.nanmin(data), np.nanmax(data)), aspect=1, **kwargs
    )

    if fields is None:
        fields = img.default_fields

    if field_color is None:
        field_color = "yellow"
        if mask or invert:
            field_color = "black"

    for i, field in enumerate(fields):
        if i > 3:
            print(f"Ignoring field {field}, not enough space")
            continue
        if not field:
            continue
        ax.text(
            x=_MPL_IMG_POS[i][2],
            y=_MPL_IMG_POS[i][3],
            s=img.get_field_string(field),
            va=_MPL_IMG_POS[i][0],
            ha=_MPL_IMG_POS[i][1],
            transform=ax.transAxes,
            color=field_color,
            fontsize=14,
        )
    return ax


def make_video(
    stack,
    ofile,
    skip=None,
    fields=None,
    field_color=None,
    mcp=None,
    dark_counts=100,
    contrast="auto",
    invert=False,
    log=False,
    fps=24,
    overwrite=True,
    scale=0.5,
    **kwargs,
):
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
            c0 = min(c0, np.nanmin(img.image))
            c1 = max(c1, np.nanmax(img.image))
        contrast = sorted((c0, c1))
        print(f"Set contrast to {contrast}")
        contrast_type = "static"
    elif isinstance(contrast, int):
        c0 = np.nanmin(stack[contrast].image)
        c1 = np.nanmax(stack[contrast].image)
        contrast = sorted((c0, c1))
        print(f"Set contrast to {contrast}")
        contrast_type = "static"
    elif isinstance(contrast, Iterable) and len(contrast) == 2:
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
        ofile, inputdict=ffmpeg_indict, outputdict=ffmpeg_outdict  # , verbosity=1
    )

    # loop through the images
    height, width = stack[0].image.shape
    dy, dx = map(int, 0.2 * np.array(stack[0].image.shape))

    if field_color is None:
        field_color = "yellow"
        if invert:
            field_color = "black"
    color = tuple(int(rgb * 255) for rgb in matplotlib.colors.to_rgb(field_color))

    for img in stack:
        if mcp is not None:
            img = normalize_image(img, mcp=mcp, dark_counts=dark_counts)
        data = np.nan_to_num(img.image)
        if log:
            data = np.nan_to_num(np.log(data))
        # set contrast
        if contrast_type == "inner":
            inner = data[dy:-dy, dx:-dx]
            contrast = inner.min(), inner.max()
        if contrast_type == "auto":
            contrast = data.min(), data.max()
        data = cv.normalize(
            np.clip(data, contrast[0], contrast[1]),
            np.ones_like(data, dtype=np.uint8),
            0,
            255,
            norm_type=cv.NORM_MINMAX,
        )

        if invert:
            data = -data + data.max()

        if fields:
            data = cv.cvtColor(data, cv.COLOR_GRAY2RGB)
            for i, field in enumerate(fields):
                if field is None:
                    continue
                text = img.get_field_string(field).replace("µ", "u")
                text = text.encode("ascii", errors="ignore").decode()

                for pos, char in enumerate(text):
                    xy = _CV2_IMG_POS(i, height, width, text, pos=pos)
                    data = cv.putText(
                        data,
                        char,
                        org=xy,
                        fontFace=cv.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=color,
                        thickness=2,
                        lineType=cv.LINE_AA,
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
                print(
                    "Points have been excluded from plot because of unreasonably "
                    "high temperature"
                )
        else:
            axes[i].plot(time, val)
        if field in ("pressure", "pressure1", "pressure2"):
            axes[i].set_yscale("log")
        axes[i].set_ylabel(f"{field} in {stack[0].get_unit(field)}")
        axes[i].set_xlabel("Time in s")

    return axes


def print_meta(
    stack,
    fields=(
        "energy",
        "temperature",
        "pressure1",
        "pressure2",
        "objective",
        "fov",
        "exposure",
        "averaging",
        "width",
        "height",
    ),
):
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
    pd.set_option("display.expand_frame_repr", False)
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
    rois = roify(*args, **kwargs)
    ax = _get_ax(ax, xlabel=xaxis, ylabel="Intensity in a.u.")
    data = get_intensity(stack, rois, xaxis=xaxis)
    x = data[0]
    for roi, intensity in zip(rois, data[1:]):
        if roi.color is not None:
            ax.plot(x, intensity, color=roi.color)
        else:
            ax.plot(x, intensity)
    return ax


def plot_intensity_curve(curves, ax=None):
    # if isinstance(curves, IntensityCurve):
    #     curves = [curves]
    ax = _get_ax(ax, xlabel=curves[0].xaxis, ylabel="Intensity in a.u.")
    for curve in curves:
        try:
            color = curve.roi.color
        except AttributeError:
            color = curve.roi[0].color
        if color is not None:
            ax.plot(curve.x, curve.y, color=color)
        else:
            ax.plot(curve.x, curve.y)
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
            cols[j + 1][i] = roi.apply(img.image).sum() / roi.area

    data = np.stack(cols)

    if ofile is not None:
        np.savetxt(
            ofile, data.T, header=f"{xaxis} | " + " | ".join(str(roi) for roi in rois)
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
    for kwarg in ROI.kwargs:
        kwargs.pop(kwarg, None)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if img_idx is None:
        img_idx = get_max_variance_idx(stack)
    plot_img(stack[img_idx], ax=ax2, ticks=True, **kwargs)
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
    for kwarg in ROI.kwargs:
        kwargs.pop(kwarg, None)
    if ax is None:
        ax = _get_ax(ax)
    if img is not None:
        img = imgify(img)
        ax = plot_img(img, ticks=True, ax=ax)

    for roi in rois:
        ax.add_artist(roi.artist)
    return ax


def plot_rsm(
    stack=None,
    profile=None,
    rsm=None,
    log=True,
    figsize=(5, 8),
    rasterized=True,
    dpi=150,
    ax=None,
    cmap="gray",
    vmin=None,
    vmax=None,
    **kwargs,
):
    """Plot an RSM along a cut specified by profile.
    Arguments:
        - stack     the LEED stack
        - profile   see help(leem.processing.Profile)
                    can be plotted like a ROI with plot_rois()
        - gamma     position of the specular beam (default: profile center)
        - BZ_pix    Brillouin zone size in pixels
        - d         lattice spacing associated with BZ_pix (in meters!)
        - Vi        inner potential (default: 0)
    """
    if stack is None or profile is None:
        assert isinstance(rsm, RSM)
    elif rsm is None:
        rsm = RSM(stack, profile, **kwargs)
    else:
        assert rsm.stack == stack and rsm.profile == profile
    kx, ky, z = rsm()
    kx = kx * 1e-10
    ky = ky * 1e-10
    if log:
        z = np.log(z)
        if vmin is not None:
            vmin = np.log(vmin)
        if vmax is not None:
            vmax = np.log(vmax)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_xlabel("kx in A^-1")
        ax.set_ylabel("ky in A^-1")
        secax = ax.secondary_xaxis(
            "top",
            functions=(
                lambda x: x * 100 / (rsm.k_BZ * 1e-10),
                lambda x: x / 100 * (rsm.k_BZ * 1e-10),
            ),
        )
        secax.set_xlabel("BZ in %")
    ax.pcolormesh(kx, ky, z, cmap=cmap, rasterized=rasterized, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    return rsm, ax


# Utility:

_MPL_IMG_POS = {  # (verticalalignment, horizontalalignment, x, y)
    0: ("top", "left", 0.01, 1),
    1: ("top", "right", 1, 1),
    2: ("bottom", "left", 0.01, 0.01),
    3: ("bottom", "right", 1, 0.01),
}


_CV2_CONTRASTS = {
    "auto": cv.NORM_MINMAX,
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
            figsize=kwargs.get("figsize", None), dpi=kwargs.get("dpi", 100)
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
