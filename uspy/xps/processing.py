"""Provides functions for processing spectrum data."""
# pylint: disable=invalid-name

import numpy as np


class IgnoreUnderflow:
    """Context manager for suppressing numpy underflow runtime errors."""

    # pylint: disable=too-few-public-methods
    def __init__(self):
        self.old_settings = {}

    def __enter__(self):
        self.old_settings = np.seterr(under="ignore")

    def __exit__(self, exc_type, exc_value, traceback):
        np.seterr(**self.old_settings)


def calculate_background(bg_type, bg_bounds, energy, intensity):
    """Calculates a numpy array representing the background."""
    background = intensity.copy()
    if bg_type is None:
        return np.zeros_like(energy)
    if bg_bounds is None:
        bg_bounds = [energy.min(), energy.max()]
    for lower, upper in zip(bg_bounds[0::2], bg_bounds[1::2]):
        idx1, idx2 = sorted(
            [np.searchsorted(energy, lower), np.searchsorted(energy, upper)]
        )
        if idx1 == idx2:
            continue
        if bg_type == "shirley":
            # try:
            background[idx1:idx2] = shirley(energy[idx1:idx2], intensity[idx1:idx2])
        # except FloatingPointError:
        #    print("shirley: division by zero")
        elif bg_type == "linear":
            background[idx1:idx2] = linear_bg(energy[idx1:idx2], intensity[idx1:idx2])
        elif bg_type == "tougaard":
            background[idx1:idx2] = tougaard_calculate(
                energy[idx1:idx2], intensity[idx1:idx2]
            )[0]
        # raise NotImplementedError("Tougaard not implemented")
        else:
            raise ValueError("Unknown background type '{}'".format(bg_type))
    return background


def intensity_at_energy(energy, intensity, energy_value):
    """Gives back the intensity value at energy_value."""
    idx = np.searchsorted(energy, energy_value)
    return intensity[idx]


def shirley(energy, intensity, tol=1e-5, maxit=50):
    """Calculates shirley background for given x, y values. The alogirthm assumes constant energy spacing but does not enforce it."""
    # if energy[-1] < energy[0]:
    #    raise ValueError("Energy not increasing.")
    # if not is_equidistant(energy):
    #    raise ValueError("Energy not evenly spaced.")

    if energy[0] < energy[-1]:
        is_reversed = True
        energy = energy[::-1]
        intensity = intensity[::-1]
    else:
        is_reversed = False

    # energy = energy[::-1]
    # intensity = intensity[::-1]
    np.seterr(all="raise")

    background = np.ones(energy.shape) * intensity[-1]
    integral = np.zeros(energy.shape)
    spacing = (energy[-1] - energy[0]) / (len(energy) - 1)

    # spacings = [energy1 - energy2 for energy1, energy2 in zip(energy, energy[1:])]

    rest = intensity - background
    ysum = rest.sum() - np.cumsum(rest)
    integral = spacing * (ysum - 0.5 * (rest + rest[-1]))

    for _ in range(maxit):
        rest = intensity - background
        integral = spacing * (rest.sum() - np.cumsum(rest))
        bnew = (intensity[0] - intensity[-1]) * integral / integral[0] + intensity[-1]
        if np.linalg.norm((bnew - background) / intensity[0]) < tol:
            background = bnew
            break
        else:
            background = bnew
    else:
        print("shirley: Max iterations exceeded before convergence.")
    if is_reversed:
        return background[::-1]
    else:
        return background


def tougaard_calculate(x, y, tb=2866, tc=1643, tcd=1, td=1, maxit=100):
    # https://warwick.ac.uk/fac/sci/physics/research/condensedmatt/surface/people/james_mudd/igor/

    # Sanity check: Do we actually have data to process here?
    if not (any(x) and any(y)):
        print("One of the arrays x or y is empty. Returning zero background.")
        return [x * 0, tb]

    # KE in XPS or PE in XAS
    if x[0] < x[-1]:
        is_reversed = True
    # BE in XPS
    else:
        is_reversed = False

    Btou = y * 0

    it = 0
    while it < maxit:
        if is_reversed == False:
            for i in range(len(y) - 1, -1, -1):
                Bint = 0
                for j in range(len(y) - 1, i - 1, -1):
                    Bint += (
                        (y[j] - y[len(y) - 1])
                        * (x[0] - x[1])
                        * (x[i] - x[j])
                        / (
                            (tc + tcd * (x[i] - x[j]) ** 2) ** 2
                            + td * (x[i] - x[j]) ** 2
                        )
                    )
                Btou[i] = Bint * tb

        else:
            for i in range(len(y) - 1, -1, -1):
                Bint = 0
                for j in range(len(y) - 1, i - 1, -1):
                    Bint += (
                        (y[j] - y[len(y) - 1])
                        * (x[1] - x[0])
                        * (x[j] - x[i])
                        / (
                            (tc + tcd * (x[j] - x[i]) ** 2) ** 2
                            + td * (x[j] - x[i]) ** 2
                        )
                    )
                Btou[i] = Bint * tb

        Boffset = Btou[0] - (y[0] - y[len(y) - 1])
        # print(Boffset, Btou[0], y[0], tb)
        if abs(Boffset) < (0.000001 * Btou[0]) or maxit == 1:
            break
        else:
            tb = tb - (Boffset / Btou[0]) * tb * 0.5
        it += 1

    print("Tougaard B:", tb, ", C:", tc, ", C':", tcd, ", D:", td)
    print([y[len(y) - 1] + Btou, tb])
    return [y[len(y) - 1] + Btou, tb]


def linear_bg(energy, intensity):
    """Calculates linear background for given x, y values."""
    samples = len(energy)
    background = np.linspace(intensity[0], intensity[-1], samples)
    return background


def is_equidistant(energy, tol=1e-08):
    """Returns True only when energy is equidistant."""
    spacings = np.unique(np.diff(energy))
    for spacing in spacings[1:]:
        if not np.isclose(spacing, spacings[0], atol=tol):
            return False
    return True


def make_increasing(energy, intensity):
    """Makes energy increasing and sorts intensity accordingly."""
    idxes = energy.argsort()
    return energy[idxes], intensity[idxes]


def make_equidistant(energy, intensity):
    """Makes x, y pair so that x is equidistant."""
    spacings = np.unique(np.diff(energy))
    if all([np.isclose(spacing, spacings[0]) for spacing in spacings[1:]]):
        return energy, intensity
    # eliminate spacings that are too small
    while np.isclose(0, spacings.min()):
        spacings.remove(spacings.min())
    samples = int((energy.max() - energy.min()) / spacings.min())
    spaced_energy = np.linspace(energy.min(), energy.max(), samples)
    spaced_intensity = np.interp(spaced_energy, energy, intensity)
    return spaced_energy, spaced_intensity
