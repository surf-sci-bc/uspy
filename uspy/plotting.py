"""Plotting helper functions."""

# pylint: disable=too-many-arguments
# pylint: disable=invalid-name


from __future__ import annotations
from typing import Sequence, Union, Optional
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

import uspy.dataobject as do
import uspy.derived as derived
import uspy.roi as rois
from uspy.leem.utility import stackify, imgify
from scipy.interpolate import RegularGridInterpolator


def _axis_label(name: str, unit: str) -> str:
    """Axis label for a metadata field: "energy in eV", or just "energy" when the field has
    no unit -- the default for arbitrary user-added metadata (see DataObject.__setattr__)."""
    return f"{name} in {unit}" if unit else name


def _display_data(image, keep=None, log=False):
    """Masked float array of what gets displayed: NaN -> 0, pixels outside the boolean
    array `keep` masked, and with `log` the natural log (non-positive pixels masked)."""
    data = np.ma.array(np.nan_to_num(image).astype(float), mask=None if keep is None else ~keep)
    return np.ma.log(data) if log else data


def _resolve_contrast(data, contrast, log=False):
    """Lower/upper display limits shared by `plot_img`, `make_video` and `waterfall`.

    Parameters
    ----------
    data : np.ma.MaskedArray
        Displayed data as from `_display_data` (log already applied, outside-mask pixels
        masked); used for None/"auto"/"maximum" (its min/max) and "inner" (the same, over
        the central 60 %). May be None for an explicit `contrast`.
    contrast : None, str or sequence of two numbers
        Explicit limits are in data units; with `log` they are converted to the log domain
        (non-positive limits map to 0, i.e. a data value of 1).
    log : bool, optional
        Whether `data` is displayed logarithmically.

    Returns
    -------
    tuple[float, float]
    """
    if contrast is None or (isinstance(contrast, str) and contrast in ("auto", "maximum")):
        return data.min(), data.max()
    if isinstance(contrast, str):
        if contrast != "inner":
            raise ValueError(f"Invalid '{contrast=}'")
        dy, dx = (0.2 * np.array(data.shape)).astype(int)
        inner = data[dy : data.shape[0] - dy, dx : data.shape[1] - dx]
        return inner.min(), inner.max()
    limits = np.asarray(contrast, dtype=float)
    if limits.shape != (2,):
        raise ValueError(f"Invalid '{contrast=}'")
    if log:
        limits = np.log(limits, out=np.zeros(2), where=limits > 0)
    return limits[0], limits[1]


def _to_uint8(data, lo, hi):
    """Map [lo, hi] linearly onto 0..255 (clipping outside); masked pixels stay masked."""
    span = hi - lo if hi > lo else 1.0
    return ((np.clip(data, lo, hi) - lo) / span * 255).astype(np.uint8)


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
        Alternatively, an iterable with two numbers for the lower and upper boundary (in data
        units, also with `log`) is given directly; such a fixed contrast maps a value to the same
        gray level in every image. "inner" is evaluated on the displayed (log/mask-processed)
        data. Defaults to None.
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
        If an ROI, use it to mask out parts of the image. If True, use ``img.default_mask``
        -- the instrument's default mask (e.g. the channelplate aperture for a `LEEMImg`,
        resolved from a `uspy.toml` `[uspy.mask]` table or the reader's built-in default;
        see `uspy.leem.base.LEEMImg.default_mask`); a no-op for a plain `do.Image`, which has
        no default mask of its own. By default False.
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

    if mask is True:
        mask = img.default_mask

    mask_applied = isinstance(mask, rois.ROI)
    keep = mask.pad_to(*img.shape).astype(bool) if mask_applied else None
    data = _display_data(img.image, keep, log)
    data = _to_uint8(data, *_resolve_contrast(data, contrast, log))

    if invert:
        data = 255 - data

    # fixed 0..255 range: a static contrast must map a value to the same gray in every frame
    ax.imshow(data, cmap=cmap, vmin=0, vmax=255, **kwargs)

    if fields is None:
        fields = img.default_fields

    if field_color is None:
        field_color = "yellow"
        if mask_applied or invert:
            field_color = "black"

    for i, field in enumerate(fields):
        if i > 3:
            print(f"Ignoring field {field}, not enough space")
            continue
        if not field:
            continue
        # print(img.get_field_string(field))
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
    crf: int = 23,
    mask: Union[rois.ROI, np.ndarray, bool, None] = None,
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
    - mcp, dark_counts:
      Not implemented here -- normalization was removed when the legacy
      uspy.leem.driftnorm module was quarantined. Passing either raises ValueError;
      call LEEMStack.normalize() on the stack before make_video() instead.
    - contrast:
      "auto": Use the maximum dynamic range for EACH image (default).
      "inner": Same as "auto", but it will only use the inner 60 % of each image.
      "maximum": Use the maximum dynamic range of the whole stack.
      n -- integer: Use the dynamic range of the n-th image in the stack
      (n, m) -- two numbers or an array: Use those values (data units, also with log) as
      lower/upper intensity boundary; a value has the same gray level in every frame
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
    - mask: ROI, np.ndarray, bool or None, default None
      Pixels outside the mask are shown white and excluded from "auto"/"inner"/"maximum"
      contrast (applied at each frame read -- see the comment below). If True, use
      stack[0].default_mask -- the instrument's default mask (e.g. the channelplate
      aperture; see LEEMImg.default_mask). If an ROI, converted via ROI.pad_to. If an
      ndarray, used directly (must match the frame shape; cast to bool). False/None: no mask.

    Returns an iPython.display Video object
    """
    if mcp is not None or dark_counts != 100:
        raise ValueError(
            "make_video() no longer normalizes images itself (mcp/dark_counts are not "
            "applied). Call LEEMStack.normalize() on the stack before make_video() instead."
        )
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

    ## Apply mask if applicable

    if mask is True:
        mask = stack[0].default_mask
    if isinstance(mask, rois.ROI):
        mask_array = mask.pad_to(stack[0].height, stack[0].width)
    elif mask is not None and mask is not False:
        mask_array = np.asarray(mask).astype(bool)
    else:
        mask_array = np.ones(stack[0].shape, dtype=bool)
    inv_mask = ~mask_array

    ## Mask is applied at each read below (`np.ma.array(img.image, mask=inv_mask)`), not
    ## written back into the stack: `img.image = ...` would replace a deferred element's
    ## DeferredArray with a resident masked array, pinning the whole stack in memory for a
    ## virtual stack (see DataObject.__setattr__'s warning) -- and reading `img.image` is
    ## already uncached for a virtual stack, so wrapping at each read costs nothing extra.

    # find the contrast values: "auto"/"inner" are resolved per frame, everything else is a
    # fixed (lower, upper) pair in data units (log is applied by _resolve_contrast).
    if isinstance(contrast, str) and contrast in ("auto", "inner"):
        print("WARNING: The video is on auto contrast.")
    elif isinstance(contrast, str) and contrast == "maximum":
        # Streamed frame-by-frame rather than `np.ma.min/max(stack.image)`, which would
        # gather every frame into one full (N, H, W) array (twice) -- defeating a virtual
        # stack's whole point for a data-heavy stack.
        c0, c1 = 2e16, 0
        for img in stack:
            masked = np.ma.array(img.image, mask=inv_mask)
            c0 = min(c0, np.ma.min(masked))
            c1 = max(c1, np.ma.max(masked))
        contrast = sorted((c0, c1))
        print(f"Set contrast to [{contrast[0]}, {contrast[1]}]")
    elif isinstance(contrast, (int, np.integer)):
        # Single read (not two) of `stack[contrast].image` -- a virtual stack's pixel data is
        # never cached, so a second `.image` access would re-read the file from scratch.
        masked = np.ma.array(stack[contrast].image, mask=inv_mask)
        contrast = sorted((np.ma.min(masked), np.ma.max(masked)))
        print(f"Set contrast to [{contrast[0]}, {contrast[1]}]")
    elif not isinstance(contrast, str):
        _resolve_contrast(None, contrast)  # validates: two numbers
    else:
        raise ValueError(f"Invalid '{contrast=}'")

    if log and not isinstance(contrast, str):
        print("Contrast set to {} -- {}".format(*_resolve_contrast(None, contrast, log)))

    # set up ffmpeg
    ffmpeg_indict = {"-r": str(fps)}
    ffmpeg_outdict = {
        "-vf": f"scale=ceil(iw/2*{scale})*2:-2",
        "-vcodec": "libx264",
        "-pix_fmt": "yuv420p",
        "-profile:v": "high",
        "-level": "4.2",
        "-crf": str(crf),
    }
    writer = skvideo.io.FFmpegWriter(
        ofile, inputdict=ffmpeg_indict, outputdict=ffmpeg_outdict  # , verbosity=1
    )

    # loop through the images
    height, width = stack[0].shape

    if field_color is None:
        field_color = "yellow"
        if invert:
            field_color = "black"
    color = tuple(int(rgb * 255) for rgb in matplotlib.colors.to_rgb(field_color))

    for img in stack:
        data = _display_data(img.image, mask_array, log)
        data = _to_uint8(data, *_resolve_contrast(data, contrast, log)).filled(0)

        data[inv_mask] = 255  # make outside of mask white

        if invert:
            data = 255 - data

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


def plot_meta(stack, fields=None, axes=None):
    """Plots some metadata of the stack over time. Returns an array of the
    axes objects."""
    stack = stackify(stack)
    if fields is None:
        fields = stack[0].meta_fields
    if isinstance(fields, str):
        fields = [fields]
    if axes is None:
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
        axes[i].set_ylabel(_axis_label(field, stack[0]._units.get(field, "")))
        axes[i].set_xlabel("Time in s")

    return axes


def print_meta(stack, fields=None):
    """Prints metadata of a stack as a table."""
    try:
        stack = stackify(stack)
    except FileNotFoundError:
        stack = stackify([imgify(stack)])

    if fields is None:
        fields = stack[0].print_fields
    if isinstance(fields, str):
        fields = [fields]

    table = []
    for img in stack:
        row = [os.path.basename(img.source)]
        for field in fields:
            row.append(img.get_field_string(field))
        table.append(row)
    pd.set_option("display.expand_frame_repr", False)
    df = pd.DataFrame(table)
    df.columns = ("Name", *fields)
    display(df)


def plot_line(
    line: Union[do.Line, Sequence[do.Line]],
    ax: Union[mpl.axes.Axes, None] = None,
    export_to=False,
    **kwargs,
) -> mpl.axes.Axes:
    """Plot a Line Dataobject

    Parameters
    ----------
    line : Union[do.Line, Sequence[do.Line]]
    ax : Union[mpl.axes.Axes, None], optional
        mpl.Axes object to plot on, by default a new Axes is created (=None)
    export_to : str, optional
        If given, save each labelled line to this path, with "$" replaced by its label.
    **kwargs :
        `label` and `color` are either one value for all lines or one per line (a color
        that is itself a tuple, e.g. RGB, counts as one value); they override the lines'
        own `label`/`color` for this plot only. Everything else goes to `Axes.plot`.

    Returns
    -------
    mpl.axes.Axes
        mpl.Axes object where the line is plotted on.
    """

    line = line if isinstance(line, Sequence) else [line]
    n = len(line)

    def per_line(value, scalar, default):
        """`value` for each line: a scalar applies to all, otherwise one entry per line."""
        if value is None:
            return default
        if scalar(value):
            return [value] * n
        value = list(value)
        if len(value) != n:
            raise ValueError(f"Got {len(value)} values for {n} lines.")
        return value

    labels = per_line(kwargs.pop("label", None), lambda v: isinstance(v, str), [li.label for li in line])
    colors = per_line(kwargs.pop("color", None), mpl.colors.is_color_like, [li.color for li in line])

    ax = _get_ax(ax, xlabel=line[0].xdim, ylabel=line[0].ydim)
    for li, label, color in zip(line, labels, colors):
        ax.plot(
            li.x,
            li.y,
            color=color,
            label="_" if label is None else label,  # underscore is ignored by ax.legend
            **kwargs,
        )
        if export_to:
            if label is None or "_" in label[0]:
                print("Line without label will not be exported.")
            elif label:
                li.save(f"{export_to}".replace("$", label))

    for li in ax.lines:  # Check if a label is set, then generate legend
        if "_" not in li.get_label()[0]:
            ax.legend()
            break

    return ax


def plot_intensity(
    stack: Union[do.ImageStack, Sequence[do.ImageStack]],
    roi: Union[rois.ROI, Sequence[rois.ROI]],
    xaxis: str,
    ax: Union[mpl.axes.Axes, None] = None,
    return_line=False,
    **kwargs,
) -> mpl.axes.Axes:
    """Plots the intensity of a ROI for each image in a stack.

    Parameters
    ----------
    stack : do.ImageStack | Sequence of do.ImageStack
    roi : rois.ROI | Sequence of rois.ROI
    xaxis : str
        The name of the attribute that is used for the x
    ax : mpl.axes.Axes | None, optional
        Axes object to plot on. None (the default) means that a new Axes is created.
    return_line : bool, optional
        If True, return a tuple of the Axes and the line object, by default False.

    Returns
    -------
    mpl.axes.Axes
        Axes object that was used for plotting.
    mpl.axes.Axes, mpl.lines.Line2D
        If return_line is True.
    """

    if isinstance(stack, Sequence):
        line = derived.StitchedLine(stack, roi, xaxis)
        line.color = roi[0].color
        line.label = roi[0].label
    elif isinstance(roi, Sequence):
        line = [derived.IntensityLine(stack, roi, xaxis) for roi in roi]
        for li, ro in zip(line, roi):
            li.color = ro.color
            li.label = ro.label
    else:
        line = derived.IntensityLine(stack, roi, xaxis)
        line.color = roi.color
        line.label = roi.label

    ax = plot_line(line, ax, **kwargs)
    if return_line:
        return ax, line
    return ax


def plot_intensity_img(
    stack: Union[do.ImageStack, Sequence[do.ImageStack]],
    roi: Union[rois.ROI, Sequence[rois.ROI]],
    xaxis: str,
    img_idx: int = None,
    stack_idx: int = 0,
    return_line=False,
    **kwargs,
) -> tuple[mpl.axes.Axes]:
    """_summary_

    Parameters
    ----------
    stack : Union[do.ImageStack, Sequence[do.ImageStack]]
        _description_
    roi : Union[rois.ROI, Sequence[rois.ROI]]
        _description_
    xaxis : str
        _description_
    img_idx : int, optional
        _description_, by default None
    stack_idx : int, optional
        _description_, by default 0
    return_line : bool, optional
        _description_, by default False

    Returns
    -------
    tuple[mpl.axes.Axes]
        _description_
    """

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if isinstance(stack, Sequence):
        img_stack = stack[stack_idx]
    else:
        img_stack = stack

    if img_idx is None:
        img_idx = get_max_variance_idx(img_stack)

    plot_img(
        img_stack[img_idx],
        ax=ax2,
        ticks=True,
        contrast=kwargs.pop("contrast", "auto"),
        mask=kwargs.pop("mask", False),
    )

    ax1 = plot_intensity(stack, roi, xaxis, ax1, return_line=return_line, **kwargs)

    roi = [roi] if isinstance(roi, rois.ROI) else roi
    for ro in roi:
        ro.plot(ax2)

    return ax1, ax2


def _draw_waterfall(
    wf: derived.Waterfall,
    ax: mpl.axes.Axes,
    *,
    interpolate: bool = False,
    npoints: Optional[tuple] = None,
    cmap: str = "inferno",
    contrast=None,
    invert: bool = False,
    log: bool = False,
    extent: Optional[Sequence] = None,
    aspect="auto",
) -> mpl.axes.Axes:
    """Render an already-built `Waterfall` onto `ax`, without mutating it.

    Split out of `waterfall()` so a caller holding onto a `Waterfall` (e.g. the
    interactive profile picker's cache, see `uspy.interactive.profile`) can redraw it
    repeatedly -- different contrast/cmap/frame range -- without recomputing
    `profile.apply()` over the whole stack each time. `wf.image`/`wf.x`/`wf.y` are read
    into local variables and never written back; the contrast/log/invert normalization
    below operates entirely on the local copy.

    Parameters
    ----------
    wf : uspy.derived.Waterfall
    ax : matplotlib.axes.Axes
    interpolate : bool, optional
        Regrid onto a `npoints`-shaped regular grid via `RegularGridInterpolator`.
    npoints : tuple[int, int], optional
        Required when `interpolate` is True.
    cmap, invert, log : as in `waterfall`.
    contrast : tuple[float, float], optional
        Clip range (data units, also with `log`) before normalizing to uint8;
        `None`/"auto"/"maximum" uses the data's own min/max, "inner" its central 60 %.
    extent : sequence, optional
        Passed to `imshow`; defaults to `[0, wf.width, wf.y[-1], wf.y[0]]` (y inverted
        so frame 0 is drawn at the top, matching a monotonically increasing `wf.y`).
    aspect : as in `imshow`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    image, x, y = wf.image, wf.x, wf.y

    if interpolate:
        nx, ny = npoints or (None, None)
        if not all((nx, ny)):
            raise ValueError(
                "For interpolation the number of points in the interpolation grid "
                "need to be specified npoints=(nx,ny)."
            )
        XX, YY = np.meshgrid(np.linspace(min(x), max(x), nx), np.linspace(min(y), max(y), ny))
        image = RegularGridInterpolator(
            (x, y), image.T, bounds_error=False, fill_value=None
        )((XX, YY))

    if extent is None:
        extent = [0, wf.width, y[-1], y[0]]

    image = _display_data(image, None, log)
    image = _to_uint8(image, *_resolve_contrast(image, contrast, log))

    if invert:
        image = 255 - image

    ax.imshow(image, extent=extent, cmap=cmap, aspect=aspect, vmin=0, vmax=255)
    ax.set_ylabel(_axis_label(wf.yaxis, wf.stack[0]._units.get(wf.yaxis, "")))
    ax.set_xlabel("Pixel")

    return ax


def waterfall(
    stack,
    profile,
    yaxis="rel_time",
    interpolate=False,
    cmap="inferno",
    contrast=None,
    invert=False,
    ax=None,
    log=False,
    **kwargs,
):
    """Plot a `uspy.derived.Waterfall`: a profile's intensity, sampled per frame and
    stacked along `yaxis` -- e.g. an ARPES/LEED cut across the energy axis.

    Parameters
    ----------
    stack : ImageStack
    profile : uspy.roi.Profile
    yaxis : str, optional
        Stack metadata field for the y axis, by default "rel_time".
    interpolate : bool, optional
        Regrid onto a `npoints`-shaped regular grid; requires `npoints=(nx, ny)`.
    cmap : str, optional
    contrast : tuple[float, float] or "auto", optional
    invert, log : bool, optional
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _get_ax(ax, **kwargs)
    wf = derived.Waterfall(stack, profile, yaxis)
    npoints = kwargs.pop("npoints", None)
    extent = kwargs.pop("extent", None)
    aspect = kwargs.pop("aspect", "auto")
    return _draw_waterfall(
        wf, ax, interpolate=interpolate, npoints=npoints, cmap=cmap, contrast=contrast,
        invert=invert, log=log, extent=extent, aspect=aspect,
    )


# def plot_intensity(stack, *args, xaxis="rel_time", ax=None, **kwargs):
#     """Plots the image intensity in a specified ROI over a specified
#     x axis. The x axis can be any attribute of the stack.
#     Either you give:
#     - The ROI object itself (from leem.ROI())
#     - A list of ROI objects
#     - The parameters for a ROI:
#         x0=XX, y0=XX and one of these:
#         * type_="circle", radius=XX                 (default if omitted)
#         * type_="rectangle", width=XX, height=XX
#         * type_="ellipse", xradius=XX, yradius=XX
#     Examples:
#         plot_intensity(stack, x0=300, y0=200, radius=3)
#         plot_intensity(stack, x0=300, y0=100, width=15, height=20)  # makes rectangle
#         roi = leem.ROI(300, 200, radius=10)
#         plot_intensity(stack, roi)
#         plot_intensity(stack, x0=100, y0=50, type_="ellipse", xradius=5, xradius=4)
#         roi2 = leem.ROI(200, 100, radius=5)
#         plot_intensity(stack, (roi, roi2))
#     Returns the axes object
#     """
#     rois = roify(*args, **kwargs)
#     ax = _get_ax(ax, xlabel=xaxis, ylabel="Intensity in a.u.")
#     data = get_intensity(stack, rois, xaxis=xaxis)
#     x = data[0]
#     for roi, intensity in zip(rois, data[1:]):
#         if roi.color is not None:
#             ax.plot(x, intensity, color=roi.color)
#         else:
#             ax.plot(x, intensity)
#     return ax


# def plot_intensity_curve(curves, ax=None):
#     # if isinstance(curves, IntensityCurve):
#     #     curves = [curves]
#     ax = _get_ax(ax, xlabel=curves[0].xaxis, ylabel="Intensity in a.u.")
#     for curve in curves:
#         try:
#             color = curve.roi.color
#         except AttributeError:
#             color = curve.roi[0].color
#         if color is not None:
#             ax.plot(curve.x, curve.y, color=color)
#         else:
#             ax.plot(curve.x, curve.y)
#     return ax


# def get_intensity(stack, *args, xaxis="rel_time", ofile=None, **kwargs):
#     """Calculate intensity profile along a stack in a given ROI (or multiple ROIs).
#     ROIs are supplied in the same way as for plot_intensity().
#     Returns a 2D-numpy array: The first row contains the value of "xaxis",
#     every following row contains the intensity along one of the ROIs.
#     If ofile is given, the result is saved in csv format under the given file name.
#     """
#     stack = stackify(stack)
#     rois = roify(*args, **kwargs)
#     x = getattr(stack, xaxis)
#     cols = [x]
#     for roi in rois:
#         cols.append(np.zeros(len(stack)))

#     for i, img in enumerate(stack):
#         for j, roi in enumerate(rois):
#             cols[j + 1][i] = roi.apply(img.image).sum() / roi.area

#     data = np.stack(cols)

#     if ofile is not None:
#         np.savetxt(
#             ofile, data.T, header=f"{xaxis} | " + " | ".join(str(roi) for roi in rois)
#         )
#     return data


def plot_iv(*args, **kwargs):
    """Alias for uspy.leem.plotting.plot_intensity() with xaxis set
    to "energy"."""
    return plot_intensity(*args, xaxis="energy", **kwargs)


# def plot_intensity_img(stack, *args, xaxis="rel_time", img_idx=None, **kwargs):
#     """Does the same thing as uspy.leem.plotting.plot_intensity()
#     but also shows an image of the stack and the ROI on it on the right.
#     Returns 2 axes objects: The first one contains the plot, the second one the image.
#     """
#     stack = stackify(stack)
#     rois = roify(*args, **kwargs)
#     for kwarg in ROI.kwargs:
#         kwargs.pop(kwarg, None)

#     _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

#     if img_idx is None:
#         img_idx = get_max_variance_idx(stack)
#     plot_img(stack[img_idx], ax=ax2, ticks=True)
#     plot_rois(rois, ax=ax2)

#     for roi in rois:
#         plot_intensity(stack, roi, ax=ax1, xaxis=xaxis)

#     return ax1, ax2


def plot_iv_img(*args, **kwargs):
    """Alias for uspy.leem.plotting.plot_intensity_img() with xaxis set
    to "energy"."""
    return plot_intensity_img(*args, xaxis="energy", **kwargs)


def plot_rois(rois, img=None, ax=None, **kwargs):
    """Plot rois onto a given axes object. The ROI can either be given like in
    plot_intensity()."""
    if ax is None:
        ax = _get_ax(ax)
    if img is not None:
        img = imgify(img)
        ax = plot_img(img, ticks=True, ax=ax)

    for roi in rois:
        roi.plot(ax=ax)
        # ax.add_artist(roi.artist)
    return ax


# Commented out 2026-08-04 alongside uspy.leem.processing.RSM (not in use, and RSM.__call__
# itself no longer works with the current uspy.roi.Profile -- see RSM's own comment for
# why). See CLAUDE.md and the coverage-improvement plan discussion.
# def plot_rsm(
#     stack=None,
#     profile=None,
#     rsm=None,
#     log=True,
#     figsize=(5, 8),
#     rasterized=True,
#     dpi=150,
#     ax=None,
#     cmap="gray",
#     vmin=None,
#     vmax=None,
#     **kwargs,
# ):
#     """Plot an RSM along a cut specified by profile.
#     Arguments:
#         - stack     the LEED stack
#         - profile   see help(leem.processing.Profile)
#                     can be plotted like a ROI with plot_rois()
#         - gamma     position of the specular beam (default: profile center)
#         - BZ_pix    Brillouin zone size in pixels
#         - d         lattice spacing associated with BZ_pix (in meters!)
#         - Vi        inner potential (default: 0)
#     """
#     if stack is None or profile is None:
#         assert isinstance(rsm, RSM)
#     elif rsm is None:
#         rsm = RSM(stack, profile, **kwargs)
#     else:
#         assert rsm.stack == stack and rsm.profile == profile
#     kx, ky, z = rsm()
#     kx = kx * 1e-10
#     ky = ky * 1e-10
#     if log:
#         z = np.log(z)
#         if vmin is not None:
#             vmin = np.log(vmin)
#         if vmax is not None:
#             vmax = np.log(vmax)
#     if ax is None:
#         _, ax = plt.subplots(figsize=figsize, dpi=dpi)
#         ax.set_xlabel("kx in A^-1")
#         ax.set_ylabel("ky in A^-1")
#         secax = ax.secondary_xaxis(
#             "top",
#             functions=(
#                 lambda x: x * 100 / (rsm.k_BZ * 1e-10),
#                 lambda x: x / 100 * (rsm.k_BZ * 1e-10),
#             ),
#         )
#         secax.set_xlabel("BZ in %")
#     ax.pcolormesh(kx, ky, z, cmap=cmap, rasterized=rasterized, vmin=vmin, vmax=vmax)
#     ax.set_aspect("equal")
#     return rsm, ax


# Utility:

_MPL_IMG_POS = {  # (verticalalignment, horizontalalignment, x, y)
    0: ("top", "left", 0.01, 1),
    1: ("top", "right", 1, 1),
    2: ("bottom", "left", 0.01, 0.01),
    3: ("bottom", "right", 1, 0.01),
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


def get_max_variance_idx(stack):
    # stack = stackify(stack)
    max_var = 0
    max_index = 0
    for i, img in enumerate(stack):
        vmin, vmax = np.amin(img.image), np.amax(img.image)
        if vmax == vmin:
            # A constant frame has zero variance -- guard against the 0/0 division below,
            # which would otherwise produce nan and always lose to max_var's initial 0.
            var = 0.0
        else:
            var = np.var((img.image.flatten() - vmin) / (vmax - vmin))
        if var > max_var:
            max_var = var
            max_index = i
    return max_index
