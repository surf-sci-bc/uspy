"""Plotting helper functions."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=too-many-arguments

import math
from os.path import basename
# import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display#, HTML

from agfalta.leem.base import LEEMBASE_VERSION
from agfalta.leem.utility import try_load_stack, try_load_img

if LEEMBASE_VERSION > 1.1:
    print("WARNING: LEEM_base version is newer than expected.")



def calc_dose(stack):
    print("WARNING: calc_dose() is deprecated")
    stack = try_load_stack(stack)
    # dose = np.zeros(len(stack.pressure1))
    for i, _ in enumerate(stack):
        if i == 0:
            stack[0].dose = 0
            continue
        stack[i].dose = (
            stack[i - 1].dose
            + (stack.pressure1[i] + stack.pressure1[i - 1])
            / (stack.rel_time[i] - stack.rel_time[i - 1])
            / 2
            * 1e6
        )
    return stack


def plot_img(img, *args, ax=None, title=None,
             fields=("temperature", "pressure1", "energy", "fov"),
             figsize=(6, 6), ticks=False, **kwargs):
    img = try_load_img(img)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if title is None:
        title = img.path
        if len(title) > 25:
            title = "..." + title[-25:]
    ax.imshow(
        img.data, *args,
        cmap="gray",
        clim=(np.nanmin(img.data), np.nanmax(img.data)),
        aspect=1,
        **kwargs
    )

    ax.set_title(title)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    pos = {
        0: ("top", "left", 0.01, 1),
        1: ("top", "right", 1, 1),
        2: ("bottom", "left", 0.01, 0.01),
        3: ("bottom", "right", 1, 0.01),
    }
    for i, field in enumerate(fields):
        if field is None:
            continue
        label = img.get_field_string(field)
        ax.text(
            pos[i][2],
            pos[i][3],
            label,
            verticalalignment=pos[i][0],
            horizontalalignment=pos[i][1],
            transform=ax.transAxes,
            color="yellow",
            fontsize=14,
        )
    return ax


def plot_movie(stack, *args, start_index=None, end_index=None, increment=None,
               cols=4, virtual=False, **kwargs):
    stack = try_load_stack(stack, virtual=virtual)
    images = [img for img in stack[start_index:end_index:increment]]

    cols = 4
    rows = math.ceil(len(images) / cols)

    fig, axes = plt.subplots(
        ncols=cols, nrows=rows, figsize=(cols * 5, rows * 5)
    )  # , constrained_layout=True)
    if rows > 1:
        for i, img in enumerate(images):
            ax = axes[i // cols, i % cols]
            plot_img(img, ax=ax, *args, **kwargs)
        for i in range(len(images), rows * cols):
            ax = axes[i // cols, i % cols]
            fig.delaxes(ax)
    else:

        for i, img in enumerate(images):
            ax = axes[i]
            plot_img(img, ax=ax, *args, **kwargs)
        for i in range(len(images), cols):
            ax = axes[i]
            fig.delaxes(ax)


def plot_meta(stack, fields="temperature"):
    stack = try_load_stack(stack)
    if isinstance(fields, str):
        fields = [fields]

    fig, ax = plt.subplots(len(fields), figsize=(6, len(fields) * 3))

    # Reshape in case the supplot has only one plot, so it stays iterable
    ax = np.array(ax).reshape(-1)
    fig.subplots_adjust(hspace=0.3)

    time = stack.rel_time
    for i, field in enumerate(fields):
        ax[i].set_title(field)
        # print(field)
        val = getattr(stack, field)
        if field == "temperature":
            ax[i].plot(time[val < 2000], val[val < 2000])
            if len(time[val < 2000]) < len(time):
                print("Points have been excluded from plot because of unreasonably "
                      "high temperature")
        else:
            ax[i].plot(time, val)
        if field in ('pressure1', 'pressure2'):
            ax[i].set_yscale('log')

def print_meta(stack, fields=("temperature", "pressure1",)):
    # stack = LEEMStack(stack)
    stack = try_load_stack(stack)
    try:
        meta_stack = []
        for img in stack:
            meta_img = [basename(img.path)]
            for field in fields:
                meta_img.append(img.get_field_string(field))
            meta_stack.append(meta_img)
        pd.set_option('display.expand_frame_repr', False)
        table = pd.DataFrame(meta_stack)#, columns=[fields])
        table.columns = ("Name",)+fields
        display(table)#, columns=[fields]))
        #display(HTML(table.to_html()))
    except TypeError:       # if stack is not iterable, it is a single image?
        meta = []
        for field in fields:
            meta.append([field, stack.get_field_string(field)])

        table = pd.DataFrame(meta, columns=["Metadata", "Value"])
        display(table)


class ROI:
    # pylint: disable=too-few-public-methods
    def __init__(self, x0, y0, **kwargs):
        self.x0 = x0
        self.y0 = y0
        if "r" in kwargs:
            self.r = kwargs["r"]
            self._type = "circle"
        else:
            raise ValueError("ROI needs more arguments, unknown ROI type")

    def get_mask(self, width, height):
        x, y = np.arange(0, width), np.arange(0, height)
        if self._type == "circle":
            mask = (
                (x[np.newaxis, :] - self.x0)**2
                + (y[:, np.newaxis] - self.y0)**2
                < self.r
            )
        else:
            raise ValueError("Unknown ROI type")
        return mask

def get_roi(*args, **kwargs):
    if isinstance(args[0], ROI):
        if kwargs or len(args) > 1:
            print("WARNING: too many arguments for get_img_area()")
        return args[0]
    return ROI(*args, **kwargs)


def get_marker_intensity(stack, x0, y0, r=10):
    stack = try_load_stack(stack)
    h, w = stack[0].height, stack[0].width
    x, y = np.arange(0, w), np.arange(0, h)
    mask = (x[np.newaxis, :] - x0)**2 + (y[:, np.newaxis] - y0)**2 < r
    intensity = np.zeros(len(stack))
    for i, img in enumerate(stack):
        intensity[i] = np.mean(img.data * mask)
    return intensity

def plot_intensity(stack, x0, y0, r=10, ax=None, xaxis="energy"):
    x = getattr(stack, xaxis)
    intensity = get_marker_intensity(stack, x0, y0, r=r)
    if ax is None:
        _, ax = plt.subplots()
        ax.set_xlabel("Energy")
    ax.plot(x, intensity)
    return ax, (x0, y0, r)

def plot_iv(stack, x0, y0, r=10, ax=None):
    return plot_intensity(stack, x0, y0, r=r, ax=ax, xaxis="energy")

def draw_marker(ax, markers):
    # colors = ('r','g','b','c','m','y','w')
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    for i, (m0, m1, m2) in enumerate(markers):
        circle = plt.Circle((m0, m1), m2, color=colors[i], fill=False)
        ax.add_artist(circle)

def plot_intensity_img(stack, markers, xaxis="energy", img_idx=None):
    stack = try_load_stack(stack)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for marker in markers:
        plot_intensity(stack, *marker, ax=ax1, xaxis=xaxis)
    if img_idx is not None:
        img = stack[img_idx]
    else:
        img = get_max_variance_img(stack)
    draw_marker(plot_img(img, ax=ax2, ticks=True), markers=markers)
    return (ax1, ax2)

def plot_iv_img(stack, markers, **kwargs):
    return plot_intensity_img(stack, markers, xaxis="energy", **kwargs)


def get_max_variance_img(stack):
    stack = try_load_stack(stack)
    max_var = 0
    max_index = 0
    for i, img in enumerate(stack):
        var = np.var(
            (img.data.flatten() - np.amin(img.data))
            / (np.amax(img.data) - np.amin(img.data))
        )
        if var > max_var:
            max_var = var
            max_index = i
    return stack[max_index]
