"""Some functions for processing LEEM images and stacks."""
# pylint: disable=missing-docstring
# pylint: disable=invalid-name

import copy
from collections import abc

import cv2
from matplotlib import image
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.constants as sc
import scipy.signal
import scipy.ndimage
import skimage.measure
from uspy import roi as rois
from uspy.dataobject import DataObject, IntensityLine
from uspy.leem.base import LEEMStack, LEEMImg
from symmetrize import pointops as po, tps, sym
from scipy.optimize import curve_fit
from numba import jit
import numba


# import uspy.leem.driftnorm as driftnorm

# from uspy.leem.base import LEEMStack#, Loadable
from uspy.leem.utility import stackify
from uspy.utility import progress_bar


def calculate_dose(stack, pressurefield="pressure1", approx=1):
    """Maybe get rid of approx?."""
    stack = stackify(stack)
    pressure = getattr(stack, pressurefield)[::approx]
    rel_time = stack.rel_time[::approx]
    approx_dose = np.cumsum(pressure * np.gradient(rel_time)) * 1e6
    # scale up to original length:
    long_dose = np.repeat(approx_dose, approx)
    cutoff = len(long_dose) - len(stack)
    if cutoff < 0:
        raise ValueError("Error calculating dose")
    return long_dose[cutoff // 2 : len(stack) + cutoff // 2]


def center_of_mass(img, roi):
    img = roi.apply(img)
    return scipy.ndimage.center_of_mass(img.image.T)


def maximum(img, roi):
    img = roi.apply(img)
    return np.unravel_index(img.image.argmax(), img.image.shape)


def fit_2d_gaussian(img, roi):
    img = roi.apply(img)
    m, n = img.image.shape
    R, C = np.mgrid[:m, :n]
    out = np.ma.column_stack((C.ravel(), R.ravel(), img.image.ravel()))
    return out


def correct_img_distortion(stack, pks: list, symmetry="hexagonal", rot=None):
    """
    Image distortion corrections using thin plate splines.

    After https://github.com/RealPolitiX/symmetrize/tree/master/examples

    Related Publication:
        Rui Patrick Xian, Laurenz Rettig, Ralph Ernstorfer.
        Symmetry-guided nonrigid registration: The case for distortion correction in
        multidimensional photoemission spectroscopy.
        Ultramicroscopy 202 (2019), 133-139
        https://doi.org/10.1016/j.ultramic.2019.04.004
    """

    pcent, psur = po.pointset_center(pks, method="centroidnn")
    psur_ord = po.pointset_order(psur, direction="ccw")

    if symmetry == "hexagonal":
        arot = (
            np.ones(
                5,
            )
            * 60
        )
    else:
        arot = np.array(rot)

    mcvd = po.cvdist(psur_ord, pcent).mean()
    ptargs = sym.rotVertexGenerator(
        pcent,
        fixedvertex=psur_ord[0, :],
        cvd=mcvd,
        arot=arot,
        direction=-1,
        scale=1,
        ret="all",
    )

    if isinstance(stack, LEEMImg):
        stack = LEEMStack([stack])

    warped_stack = stack.copy()

    interp_order = 3  # interpolation order

    for img, warped_img in zip(stack, warped_stack):
        img_warped, spline_warp = tps.tpsWarping(
            psur_ord,
            ptargs,
            img.image.T,  # .T because symmetrize expects a different x,y convention
            None,
            interp_order,
            ret="all",
        )

        warped_img.image = img_warped.T
        warped_img.spline_warp = spline_warp

    if len(warped_stack) == 1:
        return warped_stack[0]

    return warped_stack


def correct_stack_distortion(stack, spline):
    try:
        rdeform, cdeform = spline.spline_warp
    except AttributeError:
        rdeform, cdeform = spline
    stack_warped = stack.copy()
    stack_warped.image = sym.applyWarping(
        stack.image, axis=0, warptype="deform_field", dfield=[rdeform, cdeform]
    )

    return stack_warped


def calc_correlation(stack, line):
    data = stack.image
    corr = np.zeros_like(stack[0].image)
    for ii in range(stack[0].height):
        for jj in range(stack[0].width):
            corr[ii, jj] = scipy.spatial.distance.correlation(data[:, ii, jj], line.y)
    return corr


def _fit_plane(x_points, y_points, z_points):
    """Fits a plane to a set of points.
    The plane is represented by the coefficients a, b, and c of the equation
    z = ax + by + c.
    Args:
      x_points: A numpy array of shape (n,) containing the x coordinates of the points.
      y_points: A numpy array of shape (n,) containing the y coordinates of the points.
      z_points: A numpy array of shape (n,) containing the z coordinates of the points.
    Returns:
      A tuple containing the coefficients a, b, and c of the plane.
    """
    # Create a numpy array with the points.
    points = np.array([x_points, y_points, z_points]).T
    # Compute the normal vector to the plane.
    normal_vector = np.cross(points[1] - points[0], points[2] - points[0])
    # Compute the constant term of the plane.
    constant_term = -np.dot(normal_vector, points[0])
    return normal_vector, constant_term


def _1gaussian(x, amp1, cen1, sigma1, a, b):
    return (
        amp1
        * (1 / (sigma1 * (np.sqrt(2 * np.pi))))
        * (np.exp((-1.0 / 2.0) * (((x - cen1) / sigma1) ** 2)))
        + a
        + b * x
    )


def xps_correct(stack, roi, radius=5, p0=None, plot=True):
    # Fits gauss peaks to points in stack

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    if type(roi[0]) is np.array:  # if roi is list of points
        try:
            roi = [rois.ROI.circle(*r, radius=5) for r in roi]
        except:
            print("roi must either be a list of rois or a list of coordinates")

    # Extract curves

    signal = np.array(
        [
            IntensityLine(
                stack,
                r,
                xaxis="binding_energy",
            ).y
            for r in roi
        ]
    )

    # Calculate shifts by fitting a gaussian to the peaks

    energies = stack.binding_energy

    shifts = []
    for line in signal:
        if not p0:
            p1 = [
                np.max(line) - np.min(line),
                energies[np.argmax(line)],
                1,
                np.min(line),
                0,
            ]
        else:
            p1 = p0

        popt, _ = curve_fit(_1gaussian, energies, line, p0=p1)
        # print(popt)
        shifts.append(popt[1])
        if plot:
            color = next(ax1._get_lines.prop_cycler)["color"]
            ax1.plot(energies, line, "+", color=color)
            ax1.plot(
                np.linspace(energies[0], energies[-1], 100),
                _1gaussian(np.linspace(energies[0], energies[-1], 100), *popt),
                color=color,
            )
            ax1.set_title("Peaks + Gaussian Fit")
    shifts = np.array(shifts)
    shifts -= shifts.mean()
    # print(shifts)

    points = np.array([r.position for r in roi])

    # Fit plane to shifts
    normal_vector, constant_term = _fit_plane(points[:, 0], points[:, 1], shifts)

    xx, yy = np.mgrid[0 : stack[0].width, 0 : stack[0].height]
    zz = (
        -(normal_vector[0] * yy + normal_vector[1] * xx + constant_term)
        / normal_vector[2]
    )

    if plot:
        min, max = np.min(shifts), np.max(shifts)
        ax2.set_aspect("equal")
        p = ax2.pcolormesh(zz, vmin=min, vmax=max)
        ax2.contour(zz, 30, colors=["k"], vmin=min, vmax=max)
        fig.colorbar(p)
        ax2.scatter(*points.T, c=shifts, s=100, ec="k", vmin=min, vmax=max)
        ax2.invert_yaxis()
        ax2.set_title("Energy Correction Plane Fit")
        plt.plot()

    return -zz


@jit(nopython=True, parallel=True)
# shift pixel columns pixelwise in z direction
def _shift_z_cols(stack, z_corr):
    ret_val = np.zeros_like(stack)

    for xx in numba.prange(stack.shape[1]):
        for yy in numba.prange(stack.shape[2]):
            shift = int(np.floor(z_corr[xx, yy]))  # integer shift
            for zz in numba.prange(stack.shape[0] - 1):  # iterate over each column
                if (
                    zz + shift > stack.shape[0] - 1 or zz + shift < 0
                ):  # if pixels are shifted out of range skip them
                    continue
                diff = z_corr[xx, yy] - shift  # fractional shift
                ret_val[zz + shift, xx, yy] = (diff) * stack[zz, xx, yy] + (
                    1 - diff
                ) * stack[
                    zz + 1, xx, yy
                ]  # caluclate shifted pixel by weighting
    return ret_val


def correct_energies(stack, z_corr):
    corr = stack.copy()
    dE = stack[0].energy - stack[1].energy
    z_corr = z_corr / dE
    corr.image = _shift_z_cols(corr.image, z_corr)
    return corr


# class ROI:
#     # pylint: disable=too-many-instance-attributes
#     _defaults = {
#         "circle": {"radius": 10},
#         "rectangle": {"width": 50, "height": 50, "rot": 0},
#         "ellipse": {"xradius": 10, "yradius": 20, "rot": 0},
#     }
#     kwargs = (
#         "x0",
#         "y0",
#         "type_",
#         "color",
#         "radius",
#         "width",
#         "height",
#         "xradius",
#         "yradius",
#     )
#     _color_idx = 0

#     def __init__(self, x0, y0, type_=None, color=None, artist_kw=None, **kwargs):
#         # pylint: disable=too-many-arguments
#         self.position = np.array([x0, y0])
#         self._img_shape = None
#         self.area = None
#         self._mask = None

#         if type_ is None:
#             for t, props in self._defaults.items():
#                 if any(k in kwargs for k in props) and all(k in props for k in kwargs):
#                     type_ = t
#                     break
#             else:
#                 type_ = "circle"
#         if not all(k in self._defaults[type_] for k in kwargs):
#             raise ValueError(f"kwargs {kwargs} don't match a ROI type")
#         self.type_ = type_
#         assert self.type_ in self._defaults
#         self.params = copy.deepcopy(self._defaults[self.type_])
#         self.params.update(kwargs)

#         if artist_kw is None:
#             artist_kw = dict()
#         if color is None and "color" not in artist_kw:
#             color = plt.rcParams["axes.prop_cycle"].by_key()["color"][ROI._color_idx]
#             ROI._color_idx += 1
#         artist_kw["color"] = artist_kw.get("color", color)
#         artist_kw["fill"] = artist_kw.get("fill", False)
#         self.artist_kw = artist_kw

#     def __repr__(self):
#         return (
#             f"{self.type_}(position:{self.position},"
#             + ",".join(f"{k}:{v}" for k, v in self.params.items())
#             + ")"
#         )

#     @property
#     def color(self):
#         return self.artist_kw["color"]

#     def apply(self, img_array):
#         if img_array.shape != self._img_shape or self._mask is None:
#             self._mask = self.create_mask(*img_array.shape)
#             self._img_shape = img_array.shape
#         mask = self._mask
#         return img_array * mask

#     def create_mask(self, img_height, img_width):
#         mask = np.zeros((img_height, img_width))
#         rot = self.params.get("rot", 0)

#         if self.type_ == "circle":
#             mask = cv2.circle(
#                 mask,
#                 center=tuple(self.position),
#                 radius=self.params["radius"],
#                 color=1,
#                 thickness=-1,
#             ).astype(np.bool)
#         elif self.type_ == "ellipse":
#             mask = cv2.ellipse(
#                 mask,
#                 center=tuple(self.position),
#                 axes=(self.params["xradius"], self.params["yradius"]),
#                 angle=self.params["rot"],
#                 startAngle=0,
#                 endAngle=360,
#                 color=1,
#                 thickness=-1,
#             ).astype(np.bool)
#         elif self.type_ == "rectangle":
#             w, h = self.params["width"], self.params["height"]
#             rot = -self.params["rot"] * np.pi / 180
#             R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
#             corners = np.array(
#                 [[-w / 2, -h / 2], [-w / 2, h / 2], [w / 2, h / 2], [w / 2, -h / 2]]
#             )
#             corners = np.rint(np.dot(corners, R) + self.position).astype(np.int32)
#             mask = cv2.fillConvexPoly(mask, corners, color=1).astype(np.bool)
#         else:
#             raise ValueError("Unknown ROI type")

#         self.area = mask.sum()
#         return mask

#     @property
#     def artist(self):
#         if self.type_ == "circle":
#             art = plt.Circle(self.position, self.params["radius"], **self.artist_kw)
#         elif self.type_ == "ellipse":
#             art = matplotlib.patches.Ellipse(
#                 self.position,
#                 self.params["xradius"] * 2,
#                 self.params["yradius"] * 2,
#                 angle=self.params["rot"],
#                 **self.artist_kw,
#             )
#         elif self.type_ == "rectangle":
#             w, h = self.params["width"], self.params["height"]
#             rot = -self.params["rot"] * np.pi / 180
#             R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
#             lower_left = np.rint(np.dot([-w / 2, -h / 2], R) + self.position)
#             art = plt.Rectangle(
#                 lower_left,
#                 self.params["width"],
#                 self.params["height"],
#                 angle=self.params["rot"],
#                 **self.artist_kw,
#             )
#         else:
#             raise ValueError("Unknown ROI type")
#         return art


# def roify(*args, **kwargs):
#     """Takes either a single ROI, an iterable of ROIs or a set of
#     arguments for the ROI constructor. Returns a list of ROIs (for the
#     first and latter case, this list has length 1)."""
#     if "rois" in kwargs:
#         args = (*args, kwargs.pop("rois"))
#     if args and isinstance(args[0], ROI):
#         if kwargs or len(args) > 1:
#             print("WARNING: too many arguments for roify()")
#         return [args[0]]
#     elif args and isinstance(args[0], abc.Iterable):
#         if all(isinstance(roi, ROI) for roi in args[0]):
#             if kwargs or len(args) > 1:
#                 print("WARNING: too many arguments for roify()")
#             return args[0]
#     return [ROI(*args, **kwargs)]


# def roi_intensity(img, *args, **kwargs):
#     img = imgify(img)
#     roi = roify(*args, **kwargs)
#     cutout = roi.apply(img.data)
#     return np.mean(cutout)

# def roi_intensity_stack(stack, *args, **kwargs):
#     stack = stackify(stack)
#     roi = roify(*args, **kwargs)
#     intensity = np.zeros(len(stack))
#     for i, img in enumerate(stack):
#         intensity[i] = np.mean(roi.apply(img.data))
#     return intensity


def get_max_variance_idx(stack):
    stack = stackify(stack)
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
    return max_index


# class Profile(ROI):
#     """
#     Describes a profile through a LEEM or LEED image.
#     Arguments:
#         - x0, y0:   the center of the profile
#         - theta:    inclination with respect to the horizontal
#         - length:   the profile extends by length/2 from the center in each direction
#         - width:    width to average over
#         - reduce_func:  how to average along width, either a function or "gaussian" or "rect"
#         - alpha, color: used when plotting the line
#     Attributes:
#         - endpoints
#         - length
#         - artist    can be added to a matplotlib Axes
#     Methods:
#         - apply(img_array)  takes a 2D numpy array and returns a 1D profile
#     """

#     _defaults = {"profile": {"theta": 0, "length": 100, "width": 10}}
#     kwargs = ("x0", "y0", "type_", "color", "width", "length", "theta")

#     def __init__(self, *args, reduce_func="gaussian", artist_kw=None, **kwargs):
#         if artist_kw is None:
#             artist_kw = dict()
#         artist_kw["alpha"] = artist_kw.get("alpha", 0.3)
#         super().__init__(*args, type_="profile", artist_kw=artist_kw, **kwargs)
#         self.artist_kw.pop("fill")
#         self.params["theta"] *= np.pi / 180
#         self.params["length"] = int(self.params["length"])
#         self.reduce = reduce_func

#     def apply(self, img_array):
#         profile = skimage.measure.profile_line(
#             img_array,
#             self.endpoints[0, ::-1],
#             self.endpoints[1, ::-1],
#             linewidth=self.params["width"],
#             mode="constant",
#             reduce_func=self.reduce,
#         )
#         return profile

#     @property
#     def length(self):
#         return self.params["length"]

#     @property
#     def endpoints(self):
#         theta = self.params["theta"]
#         length = self.params["length"] - 1  # profile_line will include endpoint
#         dyx = np.array([np.cos(theta) * length, -np.sin(theta) * length])
#         return np.array([self.position - dyx * 0.4999, self.position + dyx * 0.4999])
#         # dyx * 0.4999 to stay below the length (profile_line will do ceil)

#     @property
#     def xy(self):
#         x = np.linspace(self.endpoints[0, 0], self.endpoints[1, 0], self.length)
#         y = np.linspace(self.endpoints[0, 1], self.endpoints[1, 1], self.length)
#         return np.stack([x, y]).T

#     @property
#     def xyC(self):
#         """xy "edge" values for pcolormesh"""
#         x = np.linspace(self.endpoints[0, 0], self.endpoints[1, 0], self.length + 1)
#         y = np.linspace(self.endpoints[0, 1], self.endpoints[1, 1], self.length + 1)
#         return np.stack([x, y]).T

#     @property
#     def reduce(self):
#         return self._reduce_func

#     @reduce.setter
#     def reduce(self, func):
#         width = self.params["width"]
#         if func == "gaussian":
#             window = scipy.signal.windows.gaussian(width, std=width / 2)
#             func = lambda x: np.mean(window * x)
#         elif func in ("rect", "boxcar"):
#             window = scipy.signal.windows.boxcar(width)
#             func = lambda x: np.mean(window * x)
#             # func = None # also works?
#         elif not callable(func):
#             raise ValueError(f"Unkown reduce_func {func}")
#         self._reduce_func = func

#     def create_mask(self, img_height, img_width):
#         raise NotImplementedError

#     @property
#     def artist(self):
#         art = matplotlib.lines.Line2D(
#             self.endpoints[:, 0],
#             self.endpoints[:, 1],
#             lw=self.params["width"],
#             solid_capstyle="butt",
#             **self.artist_kw,
#         )
#         return art


class RSM:
    def __init__(self, stack, profile, BZ_pix, d, gamma=None, Vi=0):
        self.stack = stackify(stack)
        self.profile = profile
        if gamma is None:
            gamma = profile.position

        self.gamma = np.array(gamma)
        self.k_BZ = 2 * np.pi / d
        self.k_res = self.k_BZ / BZ_pix
        self.Vi = Vi

        self.data = None

    def __call__(self):
        if self.data is not None:
            return self.data

        kpara = np.linalg.norm(self.profile.xyC - self.gamma, axis=1) * self.k_res
        kxi_center = np.linalg.norm(self.profile.position - self.profile.xy[0, :])
        kxi_x = np.linalg.norm(self.profile.xyC - self.profile.xyC[0, :], axis=1)
        kxi = (kxi_x - kxi_center) * self.k_res
        kx = np.tile(kxi, (len(self.stack) + 1, 1))

        E = self.stack.energy
        dE = np.mean(np.diff(E))
        EC = np.append(E - dE, E[-1] + dE)

        ky = np.zeros((len(EC), self.profile.length + 1))
        z = np.zeros((len(self.stack), self.profile.length))

        for i, img in enumerate(progress_bar(self.stack, "Calculating RSM...")):
            ky[i, :] = self.get_kperp(EC[i], kpara)
            z[i, :] = self.profile.apply(img.data)

        ky[-1, :] = self.get_kperp(EC[-1], kpara)
        self.data = (kx, ky, z)
        return self.data

    def get_kpara(self):
        kx = np.linalg.norm(self.profile.xyC - self.gamma, axis=1) * self.k_res
        return kx

    def get_kperp(self, E, kpara):
        energy = (E + self.Vi) * sc.e
        k0 = np.sqrt(2 * sc.m_e * energy) / sc.hbar  # k0
        kpara = kpara.clip(max=k0)
        # see Thomas Schmidt, hering3.c, L600
        # kpara^2 + kperp^2 = K
        # k0^2 = (kperp - k0)^2 + kpara^2
        kperp = k0 + np.sqrt(k0**2 - kpara**2)  # kpara^2 + kperp^2 = k0^2
        return np.nan_to_num(kperp, 0)


# class LEEMCurve(Loadable):

#     """
#     Intensitiy curve of LEEM stack.
#     It receives a stack, ROI and xaxis.
#     The Intensity of ROI is extracted along the given xaxis and presented
#     as attributes afterwards
#         - save()/load is inherited from Loadable and saves/loads as pickle object
#         - savecsv saves the x,y data as .csv file
#     """

#     _pickle_extension = ".lc"

#     def __init__(self, stacks, xaxis, rois):

#         stacks = [stacks] if not isinstance(stacks, abc.Iterable) else stacks
#         rois = roify(rois)

#         if len(rois) != len(stacks):
#             raise ValueError(
#                 "Number of Stacks and ROIs does not match. "
#                 f"Expected {len(stacks)} list of ROIs, got {len(rois)} instead."
#             )

#         self._fullstack = None

#         self._roi = rois

#         self._fnames = [stack.fnames for stack in stacks]
#         try:
#             self._dark_counts = [stack.dark_counts for stack in stacks]
#             self._mcp = [stack.mcp for stack in stacks]
#         except AttributeError:
#             pass
#         try:
#             self._alignment = [stack.alignment for stack in stacks]
#         except AttributeError:
#             pass

#         self._xaxis = xaxis
#         if len(stacks) == 1:
#             self._x, self._y = self._get_intensity(
#                 self.simplify(stacks), self.simplify(rois)
#             )
#         else:
#             print(
#                 f"Received {len(stacks)} Stacks. Trying to stitch the curves togeher."
#             )
#             self._x, self._y = self._stitch_curves(stacks, rois)

#     def _get_intensity(self, stack, roi):
#         x = getattr(stack, self._xaxis)
#         y = [roi.apply(img.data).sum() / roi.area for img in stack]
#         return np.array(x), np.array(y)

#     def _stitch_curves(self, stacks, rois):
#         """
#         Acquire intensity curves sitched together from different stacks.
#         - stacks: A list of stacks, e.g. (stack1, stack2, ...).
#             Stacks have to be in the right order and need an overlapping region,
#             that will be fitted.
#         - rois: List of ROIs, e.g. (roi1, roi2, ..)
#             The Number of ROIs has to match the number of stacks
#             given. The first list of ROIs is applied
#             to the first stack, the second to the second stack. The
#             intensity curves of the ROIs are stitched together.
#         """

#         x_data = []
#         y_data = []
#         for stack, rois in zip(stacks, rois):
#             x, y = self._get_intensity(stack, rois)
#             x_data.append(x)
#             y_data.append(y)

#         new_y_data = [curve for curve in y_data]

#         for jj, line in enumerate(y_data[1:]):

#             # Overlap assuming ordered x values
#             # jj+1 because the first line remains unmodified. y_data[1] corresponds to jj=0
#             _, x1_ind, x2_ind = np.intersect1d(
#                 x_data[jj], x_data[jj + 1], return_indices=True
#             )
#             fit_y, fit_x = (
#                 line[x2_ind[0] : x2_ind[-1]],
#                 x_data[jj + 1][x2_ind[0] : x2_ind[-1]],
#             )
#             int_y, int_x = (
#                 new_y_data[jj][x1_ind[0] : x1_ind[-1]],
#                 x_data[jj][x1_ind[0] : x1_ind[-1]],
#             )

#             # Interpolate previous line and fit new line

#             # pylint: disable=unbalanced-tuple-unpacking, cell-var-from-loop

#             spline = scipy.interpolate.interp1d(int_x, int_y, kind="cubic")
#             f = lambda x, a: a * spline(x)
#             popt, _ = scipy.optimize.curve_fit(f, fit_x, fit_y)

#             new_y_data[jj + 1] *= 1 / popt

#         # Sort everything to new stitched x,y data

#         stitched_x_data = x_data[0]
#         stitched_y_data = new_y_data[0]

#         for x, y in zip(x_data[1:], new_y_data[1:]):
#             _, _, ind = np.intersect1d(stitched_x_data, x, return_indices=True)
#             stitched_x_data = np.append(stitched_x_data, x[ind[-1] + 1 :])
#             stitched_y_data = np.append(stitched_y_data, y[ind[-1] + 1 :])

#         return stitched_x_data, stitched_y_data

#     @property
#     def x(self):
#         return self._x

#     @property
#     def y(self):
#         return self._y

#     @property
#     def roi(self):
#         return self.simplify(self._roi)

#     @property
#     def xaxis(self):
#         return self._xaxis

#     @property
#     def stack(self):
#         if self._fullstack is not None:
#             return self.simplify(self._fullstack)
#         return self.simplify(self.reconstruct())

#     def reconstruct(self):
#         stacks = [LEEMStack(fnames) for fnames in self._fnames]
#         for ii, _ in enumerate(stacks):
#             try:
#                 stacks[ii] = driftnorm.normalize(
#                     stacks[ii], mcp=self._mcp[ii], dark_counts=self._dark_counts[ii]
#                 )
#             except AttributeError:
#                 pass
#             try:
#                 stacks[ii] = driftnorm.apply_alignment_matrices(
#                     stacks[ii], self._alignment[ii]
#                 )
#             except AttributeError:
#                 pass

#         return stacks

#     def simplify(self, value):
#         if len(value) == 1:
#             return value[0]
#         return value

#     def full_save(self, path):
#         fullstack = self._fullstack is None
#         self._fullstack = self.stack  # Add stack to the object
#         super().save(path)  # Save the Object
#         if fullstack:
#             self._fullstack = None  # Delete stack again if it was None before

#     def save_csv(self, ofile):
#         data = np.vstack((self._x, self._y))
#         np.savetxt(ofile, data, delimiter=",")


def poisson_correction(line: IntensityLine, plot=False) -> IntensityLine:
    stack = line.stack
    roi = line.roi

    # mask image for each image in stack, so mean and variance can be calulated
    mstack = LEEMStack([roi.apply(img) for img in stack])
    mean = np.array([np.mean((img.image)) for img in mstack])
    var = np.array([np.var((img.image)) / np.sum(~img.image.mask) for img in mstack])

    # Fit line to Variance-Mean
    # To calculate Darkcounts and gain of CCD per Electron and Pixel

    f = lambda x, a, b: a * (x) + b
    popt, pcov = scipy.optimize.curve_fit(f, mean, var)
    a, b = popt
    x0 = -b / a
    gain = np.sqrt(a)
    err = np.sqrt(np.diag(pcov))
    print(popt, err)
    print(f"Dark Counts={x0}±{(err[0]/a-err[1]/b)*gain}")
    print(f"gain={gain}±{0.5*err[0]/a*gain}")

    if plot:
        plt.scatter(mean, var)
        plt.plot(mean, f(mean, *popt), color="r")
        plt.xlabel("Mean")
        plt.ylabel("Variance")
        plt.show()

    # Return Poission corrected line. Intensity is now electrons per pixel.

    return (line - x0) / gain
