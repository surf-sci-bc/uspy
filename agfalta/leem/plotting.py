# import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from os.path import basename
from IPython.display import display, HTML

from .base import LEEMBASE_VERSION, LEEMStack, LEEMImg
from .. import LEEMDIR
from .utility import try_load_stack, try_load_img


if LEEMBASE_VERSION > 1.1:
    print("WARNING: LEEM_base version is newer than expected.")


def draw_marker(ax, markers):
    # color = ('r','g','b','c','m','y','w')
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    for i, marker in enumerate(markers):
        circle = plt.Circle(
            (marker[0], marker[1]), marker[2], color=colors[i], fill=False
        )
        ax.add_artist(circle)


def calc_dose(stack):
    stack = try_load_stack(stack)
    dose = np.zeros(len(stack.pressure1))
    for i, image in enumerate(stack):
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


def plot_img(img, *args, ax=None, title=None, fields=(None, None, "energy", "fov"), figsize=(6,6), ticks=False, **kwargs):
    img = try_load_img(img)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if title is None:
        title = img.path
        if len(title) > 25:
            title = "..." + title[-25:]
    ax.imshow(
        img.data, *args, cmap="gray", clim=(np.amin(img.data), np.amax(img.data)), aspect=1, **kwargs
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
        val = getattr(img, field)
        unit = img.get_unit(field)
        label = img.get_field_string(field)

        if field in (
            "pressure1",
            "pressure2",
            "dose",
        ):
            if val >= 10000 or 0 < val <= 0.1:
                label = f"{np.format_float_scientific(val,precision=2)}{unit}"
            else:
                label = f"{val:.2f}{unit}"
        elif isinstance(val, str):
            label = val
        else:
            label = f"{val:.0f}{unit}"
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


def plot_movie(
    stack, start_index=0, end_index=-1, increment=1, cols=4, virtual=True, *args, **kwargs
):
    stack = try_load_stack(stack, virtual=virtual)
    images = [img for img in stack[start_index:end_index:increment]]

    cols = 4
    rows = math.ceil(len(images) / cols)

    fig, axes = plt.subplots(
        ncols=cols, nrows=rows, figsize=(cols * 5, rows * 5)
    )  # , constrained_layout=True)

    for i, img in enumerate(images):
        ax = axes[i // cols, i % cols]
        plot_img(img, ax=ax, *args, **kwargs)
    for i in range(len(images), rows * cols):
        ax = axes[i // cols, i % cols]
        fig.delaxes(ax)


def plot_meta(stack, fields="temperature"):
    # stack = LEEMStack(stack)
    stack = try_load_stack(stack)

    if isinstance(fields,str):
        fields = [fields]

    fig, ax = plt.subplots(len(fields), figsize=(6, len(fields) * 3))

    #Reshape in case the supplot has only one plot, so it stays iterable
    ax = np.array(ax).reshape(-1)

    fig.subplots_adjust(hspace=0.3)
    for i, field in enumerate(fields):
        ax[i].set_title(field)
        print(field)
        time = stack.rel_time
        val = getattr(stack, field)
        if field == "temperature":
            ax[i].plot(time[val < 2000], val[val < 2000])
            if len(time[val < 2000])<len(time):
                print("Points have been excluded from plot because of unreasonable high temperature")
        else:
            ax[i].plot(time, val)
        if field in ('pressure1','pressure2'):
            ax[i].set_yscale('log')
       # if "temperature" in field:
       #     xdata = ax[i].lines[0].get_xdata()
       #     ydata = ax[i].lines[0].get_ydata()
       #     ydata = ydata[ydata < 2000]
       #     xdata = xdata[ydata < 2000]
       #
       #     for i, temp in enumerate(ydata):
       #         if(temp>2000):
       #             xdata = xdata.delete(i)
       #             ydata = ydata.delete(i)

            #if not all(i <2300 for i in ax[i].lines[0].get_ydata()):
            #    for point in ax[i].lines[0]:
            #        print(point)



   # if isinstance(fields,str):
    #    fig, ax = plt.subplots(figsize=(6, 3))
    #    ax.set_title(fields)
    #    ax.plot(getattr(stack, fields))
    #    if "pressure" in fields:
    #            ax.set_yscale('log')
    #else:
    #    fig, ax = plt.subplots(len(fields), figsize=(6, len(fields) * 3))
    #    fig.subplots_adjust(hspace=0.3)
    #    for i, field in enumerate(fields):
    #        ax[i].set_title(field)
    #        ax[i].plot(stack.rel_time, getattr(stack, field))
    #        if "pressure" in field:
    #            ax[i].set_yscale('log')


def print_meta(stack, fields=("temperature","pressure1",)):
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
    except:
        meta = []
        for field in fields:
            meta.append([field,stack.get_field_string(field)])

        table = pd.DataFrame(meta, columns=["Metadata", "Value"])
        display(table)


def plot_iv(stack, x0, y0, r=10, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.set_xlabel("Energy")

    stack = try_load_stack(stack)

    # coordinate transformation from matplotlib coordinates to stack coordinates

    y1 = stack[0].width - x0
    x1 = stack[0].height - y0

    iv = np.zeros(len(stack.energy))
    data = np.array([img.data for img in stack])
    pixles = 0
    for x in range(-r, r):
        for y in range(-r, r):
            if x ** 2 + y ** 2 < r ** 2:
                # print([iv,stack.data[:,y,x]])
                iv = np.sum([iv, data[:, x - x1, y - y1]], axis=0)
                # stack.data[:,x-x0,y-y0] = np.zeros(len(stack.energy))
                pixles += 1
    ax.plot(stack.energy, iv / pixles)
    # _, ax = plt.subplots()
    # ax.imshow(stack.data[20], cmap='gray', clim=(np.amin(stack.data[20]),np.amax(stack.data[20])), aspect=1)

    return ax, [x0, y0, r]


def calc_var(stack):
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
