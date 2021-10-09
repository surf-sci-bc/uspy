"""Functions for processing STM images."""

# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=too-many-arguments

import numpy as np


def subtract_plane(z):
    z0 = np.mean(z)
    nx, ny = z.shape
    dx, dy = np.gradient(z)[0].mean(), np.gradient(z)[1].mean()
    XR2, YR2 = dx * nx / 2, dy * ny / 2
    XX, YY = np.mgrid[-XR2:XR2:nx*1j, -YR2:YR2:ny*1j]
    return z - XX - YY - z0