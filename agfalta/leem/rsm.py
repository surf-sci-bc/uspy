"""Calculate reciprocal space maps from stacks."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring

import numpy as np
import scipy.constants as sc
import skimage.measure as skm




class RSM:
    def __init__(self, stack, xy_specular, profile_start, profile_end,
                 kpara_per_pix=1, avg=5):
        # pylint: disable=too-many-arguments
        self.stack = stack
        self.xy0 = np.array(xy_specular)
        self.profile = np.array([profile_start, profile_end])
        # self.p = np.array([
        #     [profile_start[0] - xy_specular[0], profile_start[1] - xy_specular[1]],
        #     [profile_end[0] - xy_specular[0], profile_end[1] - xy_specular[1]]
        # ])
        self.p_length_pix = int(np.ceil(np.sqrt(
            (self.profile[1, 0] - self.profile[0, 0])**2 +
            (self.profile[1, 1] - self.profile[1, 0])**2
        )))
        self.avg = avg
        self.kpara_per_pix = kpara_per_pix
        self.p_length = len(self.get_line(stack[0].data))

    def get_rsm(self):
        rsm = np.zeros((len(self.stack), self.p_length))
        k = np.zeros((2, len(self.stack) + 1, self.p_length + 1))
        kpara = self.get_kpara_along_line()
        kpara_directed = self.kpara_per_pix * self.p_length_pix * np.linspace(
            -0.5, 0.5, self.p_length + 1)
        energy_step = np.mean(np.diff(self.stack.energy))
        for i, img in enumerate(self.stack):
            k[0, i, :] = kpara_directed
            k[1, i, :] = self.get_kperp_along_line(self.stack.energy[i] - energy_step, kpara)
            line = np.log(self.get_line(img.data))
            rsm[i, :] = line
        k[0, -1, :] = kpara_directed
        k[1, -1, :] = self.get_kperp_along_line(self.stack.energy[-1] + energy_step, kpara)
        return k, rsm

    def get_line(self, img_array):
        profile = skm.profile_line(
            img_array,
            self.profile[0, :],
            self.profile[1, :],
            linewidth=self.avg,
            order=2,                    # order of spline interpolation
            mode="constant",            # how to treat values outside image
            cval=0,                     # constant value outside image
            # reduce_func=np.mean         # aggregation func perp. to the line
        )
        return profile

    def get_kpara_along_line(self):
        dp = self.profile - self.xy0
        x = dp[0, 0] + (dp[1, 0] - dp[0, 0]) * np.linspace(0, 1, self.p_length + 1)
        y = dp[0, 1] + (dp[1, 1] - dp[0, 1]) * np.linspace(0, 1, self.p_length + 1)
        kpara = np.sqrt(x**2 + y**2) * self.kpara_per_pix
        return kpara

    @staticmethod
    def get_kperp_along_line(energy_eV, kpara):
        energy = energy_eV * sc.e
        k0 = np.sqrt(2 * sc.m_e * energy) / sc.hbar
        kperp = np.sqrt(k0**2 - kpara**2)   # pythagoras: kpara^2 + kperp^2 = k0^2
        kperp = np.nan_to_num(kperp, 0)
        return kperp
