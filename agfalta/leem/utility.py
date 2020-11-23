"""
Tools for pretty printing etc.
"""
# pylint: disable=missing-docstring
# pylint: disable=invalid-name

import sys
import contextlib
from datetime import datetime, timedelta

from agfalta.leem.base import LEEMImg, LEEMStack


def imgify(img):
    if isinstance(img, LEEMImg):
        return img
    return LEEMImg(img)

def stackify(stack, virtual=False):
    if isinstance(stack, LEEMStack):
        return stack
    return LEEMStack(stack, virtual=virtual)

def timing_notification(title=""):
    print("agfalta.leem.utility.timing_notification moved to agfalta.utility")
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


class DummyFile(object):
    # pylint: disable=too-few-public-methods
    def write(self, x):
        pass
@contextlib.contextmanager
def silence():
    print("agfalta.leem.utility.silence moved to agfalta.utility")
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def progress_bar(it, suffix="", total=None, size=25, fill="▇", empty="░", silent=False):
    # pylint: disable=too-many-arguments
    print("agfalta.leem.utility.progress_bar moved to agfalta.utility")
    if silent:
        return it
    start_time = datetime.now()
    if total is None:
        total = len(it)
    def display(i):
        x = i / total
        prog = fill * int(x * size) + empty * (size - int(x * size))
        duration = datetime.now() - start_time
        statement = f"\r▕{prog}▏ {100*x:.1f} % {suffix} ({str(duration).split('.')[0]}"
        eta = duration * (1 / max(x, 1e-5) - 1)
        if eta > timedelta(days=1):
            statement += " / ETA: > 1 day"
        elif eta > timedelta(seconds=3):
            statement += f" / ETA: {str(eta).split('.')[0]}"
        statement += ")"
        return statement
    for i, item in enumerate(it):
        print("\033[K" + display(i), end="\r")
        yield item
    print("\033[K" + display(total))



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
        print("agfalta.leem.utility.ProgressBar moved to agfalta.utility")
        self.total = total
        self.suffix = suffix
        self.length = length
        self.fill = fill
        self.unfill = "░"
        self.iteration = 0
        self._start_time = datetime.now()
        self.started = False
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

# SLIDERS = []
# def plot_stack(stack, init=0):
#     """Not intended for jupyter lab."""
#     fig, ax = plt.subplots()
#     img = ax.imshow(stack[init].data, cmap="gray")
#     ax.set_title(f"slice {init}")
#     def callback(val):
#         ax.set_title(f"slice {val:.0f}")
#         cut = stack[int(val)].data
#         img.set_data(cut)
#         if not np.isnan(np.nansum(cut)):
#             img.set_clim(vmin=np.nanpercentile(cut, 1), vmax=np.nanpercentile(cut, 99))
#         ax.get_figure().canvas.draw()
#     plt.subplots_adjust(bottom=0.15)
#     control_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
#     slider = Slider(
#         control_ax, "",
#         0, len(stack) - 1, valinit=init, valstep=1, valfmt="%d")
#     slider.on_changed(callback)
#     SLIDERS.append(slider)
# def plot_img(img):
#     """Not intended for jupyter lab."""
#     _, ax = plt.subplots()
#     img = ax.imshow(img.data[:, :], cmap="gray")
