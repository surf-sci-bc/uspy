"""Calculate reciprocal space maps from stacks."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring

import numpy as np
import scipy.constants as sc
from scipy import signal

from agfalta.leem.utility import ProgressBar


def rsm(stack, start, end, xy0, kpara_per_pix=7.67e7):
    cut = RSMCut(start=start, end=end)
    kx, ky, z = get_rsm(stack, cut, xy0=xy0, kpara_per_pix=kpara_per_pix)
    return kx, ky, z


def get_rsm(stack, cut, xy0, kpara_per_pix):
    progbar = ProgressBar(len(stack), suffix="Calculating RSM...")
    res_y, res_x = len(stack), np.rint(cut.length).astype(int)

    z = np.zeros((res_y, res_x))
    kx = np.zeros((res_y + 1, res_x + 1))
    kx[:, :] = kpara_per_pix * cut.length * np.linspace(-0.5, 0.5, res_x + 1)
    ky = np.zeros((res_y + 1, res_x + 1))

    kpara = get_kpara(cut, xy0, kpara_per_pix, length=res_x + 1)
    dE = np.mean(np.diff(stack.energy))

    for i, img in enumerate(stack):
        ky[i, :] = get_kperp(stack.energy[i] - dE / 2, kpara)
        z[i, :] = np.log(cut(img.data, length=res_x))
        progbar.increment()
    ky[-1, :] = get_kperp(stack.energy[-1] + dE / 2, kpara)
    progbar.finish()
    return kx, ky, z


class RSMCut:
    # pylint: disable=too-few-public-methods
    def __init__(self, start=None, end=None, theta=0, l=200, d=0, width=10):
        # pylint: disable=too-many-arguments
        if None in (start, end):
            c, s = np.cos(theta), np.sin(theta)
            rot_matrix = np.array([[c, -s], [s, c]])
            start = np.dot(rot_matrix, [d, -l])
            end = np.dot(rot_matrix, [d, l])
        self.start = np.array(start)
        self.end = np.array(end)
        # self.width = int(width + width % 2)
        self.width = width
        self.length = np.linalg.norm(self.start - self.end)

    def get_xy(self, length=None):
        if length is None:
            length = np.ring(len(self)).astype(int)
        x = np.linspace(self.start[0], self.end[0], length)
        y = np.linspace(self.start[1], self.end[1], length)
        return np.stack([x, y])

    def __call__(self, img_array, length=None):
        """
        See also:
        https://stackoverflow.com/questions/7878398/
        how-to-extract-an-arbitrary-line-of-values-from-a-numpy-array
        """
        if length is None:
            length = self.length

        dx, dy = (self.start - self.end) / self.length
        x, y = self.get_xy(length=length)

        zi = np.zeros((self.width, length))
        for r in range(-self.width // 2, self.width // 2):
            zi[r, :] = img_array[(x + r * dy).astype(int), (y + r * dx).astype(int)]

        gaussian_kernel = signal.windows.gaussian(self.width, std=self.width / 2)
        def reduce(x):
            return np.mean(gaussian_kernel * x)
        z = np.apply_along_axis(reduce, 0, zi)
        return z


def get_kpara(cut, xy0, kpara_per_pix, length=None):
    if length is None:
        length = len(cut) + 1
    xy0 = np.array(xy0)
    x, y = cut.get_xy(length=length) - xy0.reshape(2, 1)
    kpara = np.sqrt(x**2 + y**2) * kpara_per_pix
    return kpara


def get_kperp(energy_eV, kpara):
    energy = energy_eV * sc.e
    k0 = np.sqrt(2 * sc.m_e * energy) / sc.hbar
    kpara = kpara.clip(max=k0)          # prevent sqrt(negative values)
    kperp = k0 + np.sqrt(k0**2 - kpara**2) # pythagoras: kpara^2 + kperp^2 = k0^2
    kperp = np.nan_to_num(kperp, 0)
    return kperp
