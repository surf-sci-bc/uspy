"""
Tools for pretty printing etc.
"""
# pylint: disable=missing-docstring
# pylint: disable=invalid-name

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from agfalta.leem.base import LEEMImg, LEEMStack


def try_load_img(img):
    # pylint: disable=bare-except
    try:
        return LEEMImg(img)
    except:
        return img

def try_load_stack(stack, virtual=False):
    # pylint: disable=bare-except
    try:
        return LEEMStack(stack, virtual=virtual)
    except:
        return stack

def timing_notification(title=""):
    def timer(wrapped):
        def wrapper(*args, **kwargs):
            start = datetime.now()
            print(f"Started {title}")
            ret = wrapped(*args, **kwargs)
            duration = str(datetime.now() - start).split('.')[0]
            print(f"Finished {title} in {duration}")
            return ret
        return wrapper
    return timer


class ProgressBar(object):
    """
    Inspired by https://stackoverflow.com/questions/3173320/
    Call in a loop to create terminal progress bar
    @params:
    iteration   : current iteration (Int)
    total       : total iterations (Int)
    suffix      : suffix string (Str)
    decimals    : number of decimals in percent complete (Int)
    length      : character length of bar (Int)
    fill        : bar fill character (Str)
    printEnd    : end character (e.g. "\r", "\r\n") (Str)
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, total, suffix="", length=25, fill="▇"):
        self.total = total
        self.suffix = suffix
        self.length = length
        self.fill = fill
        self.unfill = "░"
        self.iteration = 0
        self.started = False
        self._start_time = datetime.now()
        self.finished = False

    def show(self, iteration):
        self.iteration = iteration
        if not self.started:
            self.started = True
            self._start_time = datetime.now()
        self.print()

    def increment(self, amount=1):
        self.iteration += amount
        if not self.started:
            self.started = True
            self._start_time = datetime.now()
        self.print()

    def print(self):
        fraction = self.iteration / float(self.total)
        filled_length = int(self.length * self.iteration // self.total)
        prog = self.fill * filled_length + self.unfill * (self.length - filled_length)
        duration = datetime.now() - self._start_time
        eta = duration * (1 / max(fraction, 1e-5) - 1)
        if eta > timedelta(days=1):
            eta = "> 1 day"
        else:
            eta = str(eta).split(".")[0]
        statement = (f"\r▕{prog}▏ {100*fraction:.1f} % "
                     f"{self.suffix} ({str(duration).split('.')[0]} / ETA: {eta})")
        # print("\r" + " " * (len(statement) + 5), end="\r")
        print(statement, end="\r")
        if not self.finished and self.iteration >= self.total:
            self.finish()

    def finish(self):
        if not self.finished:
            self.finished = True
            duration = datetime.now() - self._start_time
            statement = (f"\r▕{self.fill * self.length}▏ {100:.1f} % "
                         f"{self.suffix} ({str(duration).split('.')[0]})")
            print("\r" + " " * (len(statement) + 15), end="\r")
            print(statement, end="\r")
            print("")

SLIDERS = []
def plot_stack(stack, init=0):
    """Not intended for jupyter lab."""
    fig, ax = plt.subplots()
    img = ax.imshow(stack[init].data, cmap="gray")
    ax.set_title(f"slice {init}")
    def callback(val):
        ax.set_title(f"slice {val:.0f}")
        cut = stack[int(val)].data
        img.set_data(cut)
        if not np.isnan(np.nansum(cut)):
            img.set_clim(vmin=np.nanpercentile(cut, 1), vmax=np.nanpercentile(cut, 99))
        ax.get_figure().canvas.draw()
    plt.subplots_adjust(bottom=0.15)
    control_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        control_ax, "",
        0, len(stack) - 1, valinit=init, valstep=1, valfmt="%d")
    slider.on_changed(callback)
    SLIDERS.append(slider)
def plot_img(img):
    """Not intended for jupyter lab."""
    _, ax = plt.subplots()
    img = ax.imshow(img.data[:, :], cmap="gray")
