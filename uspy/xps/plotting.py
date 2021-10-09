"""Simple functions for plotting XPS spectra"""
# pylint: disable=missing-docstring

import matplotlib.pyplot as plt

from uspy.xps import io


def plot_spectrum(fname, region=1):
    specdicts = io.parse_spectrum_file(fname)
    spec = specdicts[region - 1]
    plt.plot(spec["energy"], spec["intensity"])
