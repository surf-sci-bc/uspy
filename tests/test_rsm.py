"""For now, only explorative testing."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring

import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from agfalta.leem import base
from agfalta.leem import rsm


def main():
    stack = base.LEEMStack("testdata/test_IVLEED_VO2-RuO2")
    stack = stack[::4]
    # stack = stack[::10]
    # rsm_calc = rsm.RSM(
    #     stack,
    #     [503, 519],
    #     profile_start=[105, 303],
    #     profile_end=[919, 710],
    #     # kpara_per_pix=3.84e7
    #     kpara_per_pix=7.67e7
    # )

    # plt.imshow(np.log(stack[10].data))
    # plt.scatter(
    #     [503, 304, 711, 105, 919],
    #     [510, 411, 610, 303, 710],
    #     s=5, color="k"
    # )
    # plt.show()
    # sys.exit()
    # for img in stack[:10]:
    #     plt.plot(np.log(rsm_calc.get_line(img.data)))
    # plt.figure()
    # kpara = rsm_calc.get_kpara_along_line()
    # kperp = rsm_calc.get_kperp_along_line(stack.energy[10], kpara)
    # for energy in stack.energy:
    #     plt.plot(rsm_calc.get_kperp_along_line(energy, kpara))
    # plt.plot(kpara)
    # plt.figure()
    # plt.plot(kperp)

    # plt.figure()
    cut = rsm.RSMCut(start=[105, 303], end=[919, 710])
    kx, ky, z = rsm.get_rsm(
        stack, cut,
        xy0=[503, 519],
        kpara_per_pix=7.67e7
    )
    plt.pcolormesh(
        kx * 1e-10, ky * 1e-10,
        z,
        shading="flat",
        cmap="gray"
    )
    plot_stack(stack)
    plt.show()

    sys.exit()

SLIDERS = []
def plot_stack(stack, init=0):
    """Not intended for jupyter lab."""
    fig, ax = plt.subplots()
    cut = stack[init].data
    cut = np.log(cut)
    img = ax.imshow(cut, cmap="gray")
    ax.set_title(f"slice {init}")
    def callback(val):
        ax.set_title(f"slice {val:.0f}")
        cut = stack[int(val)].data
        cut = np.log(cut)
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


if __name__ == "__main__":
    main()
