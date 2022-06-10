"""
Basic classes for Elmitec LEEM ".dat"-file parsing and data visualization.
"""
# pylint: disable=missing-docstring
# pylint: disable=attribute-defined-outside-init
# plyint: disable=access-member-before-definition

from __future__ import annotations
from typing import Any, Union, Optional
from collections.abc import Iterable
from numbers import Number
from datetime import datetime
import glob
from pathlib import Path
import warnings
from tqdm.auto import tqdm
import scipy.stats
import pandas as pd

import numpy as np
import cv2 as cv

from uspy.utility import parse_bytes, parse_cp1252_until_null
from uspy.dataobject import Image, ImageStack, StitchedLine
from uspy.leem.utility import imgify
from uspy.roi import ROI


class TimeOrigin:
    def __init__(self, value: Optional[Union[Number, TimeOrigin]] = None):
        if isinstance(value, TimeOrigin):
            value = value.value
        elif value is None:
            value = np.nan
        self.value = float(value)


class LEEMImg(Image):
    """
    LEEM image that exposes metadata as attributes.
    Default attributes are:
    - image: numpy array containing the image
    - energy (in eV), temperature (in °C), fov (in µm), timestamp (in s)
    """

    _meta_defaults = {
        "temperature": np.nan,
        "pressure": np.nan,
        "energy": np.nan,
        "fov": "Unknown FoV",
        "timestamp": np.nan,
        "mcp": None,
        "dark_counts": 0,
        "warp_matrix": np.eye(3),
    }
    _unit_defaults = {
        "energy": "eV",
        "temperature": "°C",
        "pressure": "Torr",
        "objective": "mA",
        "timestamp": "s",
        "exposure": "s",
        "rel_time": "s",
        "dose": "L",
        "emission": "µA",
        "resolution": "µm/px",
        "x_position": "µm",
        "y_position": "µm",
        "binding_energy": "eV",
    }
    default_fields = ("temperature", "pressure", "energy", "fov")

    def __init__(
        self, *args, time_origin: Union[TimeOrigin, Number] = None, **kwargs
    ) -> None:
        if not isinstance(time_origin, TimeOrigin):
            time_origin = TimeOrigin(time_origin)
        super().__init__(*args, **kwargs)
        self._time_origin = time_origin  # is a list so it can be mutable
        self.default_mask = ROI.circle(
            x0=self.width // 2, y0=self.height // 2, radius=self.width // 2
        )

    def get_field_string(self, field: str, fmt: Optional[str] = None) -> str:
        """Get a string that contains the value and its unit. Some LEEM-specific fields
        have different default formats than the pure DataObject method."""
        if fmt is None:
            if field == "temperature":
                fmt = ".0f"
            elif field == "energy":
                fmt = ".3g"
        return super().get_field_string(field, fmt)

    def warp(self, warp_matrix=None, inplace: bool = False):

        if warp_matrix is None:
            warp_matrix = self.warp_matrix

        image = cv.warpPerspective(
            self.image,
            warp_matrix,
            self.image.shape[::-1],
            flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
        )
        if inplace:
            self.image = image
            self.warp_matrix = warp_matrix
            return self

        result = self.copy()
        result.image = image
        result.warp_matrix = warp_matrix
        return result

    def normalize(
        self,
        mcp: Image = None,
        dark_counts: Union[int, float, Image] = 0,
        inplace: bool = False,
    ) -> LEEMImg:
        """Normalization of LEEM Images

        Parameters
        ----------
        mcp : Image, optional
            [description], by default None
        dark_counts : Union[int, float, Image], optional
            [description], by default 100
        inplace : bool, optional
            [description], by default False

        Returns
        -------
        LEEMImg
            [description]
        """
        if mcp is None:
            mcp = self.mcp  # pylint: disable=access-member-before-definition
        mcp = imgify(mcp)
        if not isinstance(dark_counts, (int, float, complex)):
            dark_counts = imgify(dark_counts)

        result = (self - dark_counts) / (mcp - dark_counts)
        result.image = np.nan_to_num(result.image, posinf=0, neginf=0)

        if inplace:
            self.image = result.image
            self.mcp = mcp
            self.dark_counts = dark_counts
            return self

        result.mcp = mcp
        result.dark_counts = dark_counts
        return result

    def find_warp_matrix(
        self, template: Image, algorithm="ecc", **kwargs
    ) -> np.ndarray:
        """Calculates a warp matrix between the image and a template.

        The algorithm applied is specified by the *algorithm* argument, which is *ecc* by default.
        currently only *ecc* is implemented

        Parameters
        ----------
        template : Image
            Template image against which is aligned
        algorithm : str, optional
            algorithm used for registration, by default "ecc"

        **kwargs
            Additional keyword arguments are passed through to the registration function. Can specifiy further options like convergence criteria or transformations

        Returns
        -------
        np.ndarray
            3x3 matrix containing the alignment parameters

        Raises
        ------
        ValueError
            Raised when not implemented alogrithm is called.

        See Also
        --------
        do_ecc_align
        """
        if algorithm == "ecc":
            return do_ecc_align(self, template, **kwargs)

        raise ValueError("Unknown Algorithm")

    def parse(self, source: str) -> dict[str, Any]:
        if isinstance(source, Image):
            self._source = None
            return dict(source.meta, **source.data)
        try:
            if source.endswith(".dat"):
                self._source = source
                idict = parse_dat(source)
            else:
                raise AttributeError
        except AttributeError:
            try:
                idict = super().parse(source)
            except:
                raise FileNotFoundError(
                    f"{source} does not exist or can't read."
                ) from None
        if idict.get("temperature", 0) > 3000:
            idict["temperature"] = np.nan
        return idict

    def __json_encode__(self) -> dict:
        return {
            "source": self.source,
            "mcp": self.mcp,
            "warp_matrix": self.warp_matrix,
            "dark_counts": self.dark_counts,
        }

    def __json_decode__(self, **attrs) -> None:
        self.__init__(source=attrs["source"])
        if attrs["mcp"]:
            self.normalize(
                mcp=attrs["mcp"], dark_counts=attrs["dark_counts"], inplace=True
            )

    @property
    def pressure(self) -> Number:
        for k in ("pressure", "pressure1", "pressure2", "MCH", "PCH"):
            pressure_ = self._meta.get(k, np.nan)
            if not np.isnan(pressure_):
                return pressure_
        return np.nan

    @pressure.setter
    def pressure(self, value: Number) -> None:
        self._meta["pressure"] = value

    # @property
    # def resolution(self) -> Number:
    #     fov_ = self._meta.get("fov", np.nan)
    #     if fov_ < 0:
    #         fov_ = np.nan
    #     return fov_ / self._meta.get("fov_cal", np.nan)

    @property
    def time_origin(self) -> Number:
        return self._time_origin.value

    @time_origin.setter
    def time_origin(self, value: Number) -> None:
        self._time_origin.value = value

    @property
    def rel_time(self) -> Number:
        return self.timestamp - self._time_origin.value

    @property
    def isotime(self) -> str:
        if np.isnan(self.timestamp):
            return "??-??-?? ??:??:??"
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")

    @property
    def unwarped(self) -> LEEMImg:
        return self.warp(np.linalg.inv(self.warp_matrix))


class LEEMStack(ImageStack):
    _type = LEEMImg

    def __init__(
        self, *args, time_origin: Optional[Union[TimeOrigin, Number]] = None, **kwargs
    ) -> None:
        # initialize with np.nan for the first calls to self._single_construct
        self._time_origin = TimeOrigin(time_origin)
        super().__init__(*args, **kwargs)
        # now we can access the 0th element and set the value
        if time_origin is None:
            self._time_origin.value = self[0].timestamp

    def _split_source(self, source: Union[str, Iterable]) -> list:
        if isinstance(source, str):
            fnames = []
            if source.endswith(".dat"):  # .../path/*.dat
                fnames = sorted(glob.glob(f"{source}"))
            if not fnames:  # .../path/
                fnames = sorted(glob.glob(f"{source}*.dat"))
            if not fnames:  # .../path
                fnames = sorted(glob.glob(f"{source}/*.dat"))

            if fnames:
                return fnames

        try:
            return super()._split_source(source)
        except ValueError:
            raise ValueError(f"No files were found at path {source}")

            if not fnames:  # if nothing works it might just be a list of filenames
                try:
                    return super()._split_source(source)
                except ValueError:
                    raise ValueError(f"No files were found at path {source}")
            return fnames
        return source

    def _single_construct(self, source: Any) -> LEEMImg:
        """Construct a single DataObject."""
        return LEEMImg(source, time_origin=self._time_origin)

    @property
    def time_origin(self) -> Number:
        return self._time_origin.value

    @time_origin.setter
    def time_origin(self, value: Number) -> None:
        self._time_origin.value = value

    def __getitem__(self, index: Union[int, slice]) -> Union[LEEMImg, LEEMStack]:
        if isinstance(index, np.ndarray):
            elements = [self._elements[i] for i in np.where(index == True)[0]]
        else:
            elements = self._elements[index]

        if isinstance(index, int):
            if (
                self.virtual
            ):  # if virtual elements contains just sources not DataObjects
                return self._single_construct(elements)
            return elements
        return type(self)(elements, virtual=self.virtual, time_origin=self._time_origin)

    def align(
        self,
        inplace: bool = False,
        mask: Union[bool, np.ndarray, None] = True,
        template=None,
        sanity=True,
        **kwargs,
    ) -> LEEMStack:
        """
        Image registration of the stack

        The images of the stack are registered subsequently by calling Image.find_warp_matrix() for
        each image and finally warping them using Image.warp(). After inital registration the
        results are checked for sanity by checking if some translation shifts are more than
        3 standard deviations away from the mean shift. If so, the values are interpolated from a
        spline, over the whole stack. The same is true if the initial alignment fails.

        Parameters
        ----------
        inplace : bool, default False
            If 'True' the stack itself will be registered
            If 'False' a copy of the stack will be registered
        mask : bool, np.ndarray or None, default True
            Mask that will be used during registration

            - If 'True' a circular mask with a radius=0.9*Image.width is used as mask
            - If 'False' or 'None' no mask is used during registration
            - If np.ndarray the array must be of dtype=np.uint8 and contain the mask that should be
              applied

        **kwargs
            Additional keyword arguments are passed through to Image.find_warp_matrix

        Returns
        -------
        LEEMStack
            LEEMStack with aligned images. The stack itself is returned when 'inplace' = True

        See Also
        --------
        find_warp_matrix
        LEEMImg.warp
        """

        if inplace:
            stack = self
        else:
            stack = self.copy()

        if mask:
            roi = ROI.circle(
                x0=stack[0].width // 2,
                y0=stack[0].height // 2,
                radius=stack[0].width // 2 * 9 // 10,
            )
            mask = (~np.ma.getmaskarray(roi.apply(stack[0]).image)).astype(np.uint8)
        else:
            mask = None
        # List of all warp matrices
        warp_matrices = [np.eye(3, dtype=np.float32)]

        if template is None:
            for index, (img1, img2) in enumerate(zip(tqdm(stack[1:]), stack)):
                try:
                    warp_matrix = img1.find_warp_matrix(img2, mask=mask, **kwargs)
                except:
                    print(f"Alginment failed on Image {index}. Will interpolate later")
                    # warp_matrix = warp_matrices[-1]
                    warp_matrix = np.eye(3, 3)
                    warp_matrix[0:2, 2] = np.nan

                # img1.warp(warp_matrix, inplace = True)
                warp_matrices.append(warp_matrix)
        else:
            warnings.warn("Aligning against template is considered unstable!")
            try:
                template = stack[template]
            except:
                pass
            for index, img in enumerate(stack):
                try:
                    warp_matrix = img.find_warp_matrix(template, mask=mask, **kwargs)
                except:
                    print(f"Alginment failed on Image {index}. Will interpolate later")
                    # warp_matrix = warp_matrices[-1]
                    warp_matrix = np.eye(3, 3)
                    warp_matrix[0:2, 2] = np.nan
                if len(warp_matrices) > 0:
                    warp_matrix = np.linalg.inv(warp_matrices[-1]) @ warp_matrix
                else:
                    warp_matrix = np.eye(3, 3)
                warp_matrices.append(warp_matrix)

            # When aligning against template the translations are absolute to the template, so
            # they are relative to first image
            # T = M_1 * img_1 => M_2*img2 = M_2*img2 => img2 = M_2^-1*M_1*img1

        dx = np.array([matrix[0, 2] for matrix in warp_matrices])
        dy = np.array([matrix[1, 2] for matrix in warp_matrices])

        if sanity:

            # Check sanity. Are some aligns outside 3 std dev, then they are most likely wrong.
            # Outliners are replaced with np.nan
            # Calculate zscore of dx,dy

            dx_z = np.abs(scipy.stats.zscore(dx, nan_policy="omit"))
            dy_z = np.abs(scipy.stats.zscore(dy, nan_policy="omit"))

            for index, value in enumerate((dx_z > 3) | (dy_z > 3)):
                if value:
                    print(f"Align {index} to far out")

            dx[(dx_z > 3) | (dy_z > 3)] = np.nan
            dy[(dx_z > 3) | (dy_z > 3)] = np.nan

        # Interpolate over valid points and recalculate invalid points.
        # Interpolation cannot handle nan, so first the nan values have to be converted to a
        # numerical value and then the weight is set to zero.

        # Check if dx and dy have nan values at same positions
        np.testing.assert_array_equal(np.isnan(dx), np.isnan(dy))

        # Pandas interpolates over nan values
        dx = pd.Series(dx).interpolate("spline", order=3)
        dy = pd.Series(dy).interpolate("spline", order=3)

        for index, (x, y) in enumerate(zip(dx, dy)):
            warp_matrices[index][0, 2] = x
            warp_matrices[index][1, 2] = y

        # calculate the warp matrices with respect to the position of first image
        for index, matrix in enumerate(warp_matrices[1:]):
            warp_matrices[index + 1] = matrix @ warp_matrices[index]

        # apply warp matrices to image
        for warp_matrix, img in zip(warp_matrices, stack):
            img.warp(warp_matrix=warp_matrix, inplace=True)

        return stack

    def normalize(
        self, mcp: Union[Image, str, None] = None, inplace: bool = False, **kwargs
    ) -> LEEMStack:
        """
        Normalization of images in stack

        The images in the stack are normalized by appling Image.normalize(mcp) to each image

        Parameters
        ----------
        mcp : Image, str or None
            Image or filename of image if str. If 'None' the mcp attribute of the images will be
            used
        inplace : bool, default False
            If 'True' the stack itself will be normalized
            If 'False' a copy of the stack will be normalized
        **kwargs
            Additional keyword arguments are passed through to Image.normalize


        .. note:: Consider passing 'dark_counts' as a keyword argument, to specify non-default
            dark counts of the images

        Returns
        -------
        LEEMStack
            LEEMStack with registered images. The stack itself is returned when 'inplace'=True

        See Also
        --------
        LEEMImg.normalize

        """
        # if not a copy funny things might happen when mcp is part of stack
        mcp = imgify(mcp).copy()
        if inplace:
            stack = self
        else:
            stack = self.copy()

        for index, img in enumerate(tqdm(stack)):
            # if an image is multiple times in a stack and inplace is True, it should not be
            # normalized twice, so if the mcp is already the current mcp, it will not be processed
            if img.mcp is None or not img.mcp == mcp:
                stack[index] = img.normalize(mcp=mcp, **kwargs)

        return stack


class IVCurve(StitchedLine):
    """
    Convenience Class for generating LEEM IV-Curves.
    """

    def __init__(self, *args, **kwargs):
        """
        It is called and behaves exactly like StitchedLine but is called with *xaxis* = "energy" by
        default.

        See Also
        --------
        dataobject.StitchedLine
        """
        print(*args)
        print(**kwargs)
        super().__init__(*args, **kwargs, xaxis="energy")


# Format: meta_key: (byte_position, encoding)
HEADER_ONE = {
    "_id": (0, "cp1252"),
    "_size": (20, "short"),
    "_version": (22, "short"),
    "_bitsperpix": (24, "short"),
    "width": (40, "short"),
    "height": (42, "short"),
    "_noimg": (44, "short"),
    "_recipe_size": (46, "short"),
}
HEADER_TWO = {
    "_isize": (0, "short"),
    "_iversion": (2, "short"),
    "_colorscale_low": (4, "short"),
    "_colorscale_high": (6, "short"),
    "time": (8, "time"),
    "_mask_xshift": (16, "short"),
    "_mask_yshift": (18, "short"),
    "_usemask": (20, "bool"),
    "_att_markupsize": (22, "short"),
    "_spin": (24, "short"),
}
ATTR_NAMES = {
    "timestamp": "time",
    "energy": "Start Voltage",
    "temperature": "Sample Temp.",
    "pressure1": "Gauge #1",
    "pressure2": "Gauge #2",
    "objective": "Objective",
    "emission": "Emission Cur.",
}
# Format: byte_position: (block_length, field_dict)
# where field_dict is formatted like the above HEADER_ONE and HEADER_TWO
VARIABLE_HEADER = {
    255: (0, None),  # stop byte
    100: (8, {"x_position": (0, "float"), "y_position": (4, "float")}),
    228: (
        8,
        {"x_position": (0, "float"), "y_position": (4, "float")},
    ),  # Image Version 8 of Elettra
    # Average Images: 0 means no averaging, 255 means sliding average
    104: (6, {"exposure": (0, "float"), "averaging": (4, "short")}),
    105: (0, {"_img_title": (0, "cp1252")}),
    242: (2, {"mirror_state": (0, "bool")}),
    243: (4, {"screen_voltage": (0, "float")}),
    244: (4, {"mcp_voltage": (0, "float")}),
}
UNIT_CODES = {
    "1": "V",
    "2": "mA",
    "3": "A",
    "4": "°C",
    "5": "K",
    "6": "mV",
    "7": "pA",
    "8": "nA",
    "9": "\xb5A",
    "B": "µm",
}


def parse_dat(fname: str, debug: bool = False) -> dict[str, Any]:
    """Parse a UKSOFT2001 file."""
    data = {}

    def parse_block(block, field_dict):
        for key, (pos, encoding) in field_dict.items():
            data[key] = parse_bytes(block, pos, encoding)
            if debug:
                print(f"\t{key} -> {data[key]}")

    with Path(fname).open("rb") as uk_file:
        parse_block(uk_file.read(104), HEADER_ONE)  # first fixed header

        if data["_recipe_size"] > 0:  # optional recipe
            data["recipe"] = parse_bytes(
                uk_file.read(data["_recipe_size"]), 0, "cp1252"
            )
            uk_file.seek(128 - data["_recipe_size"], 1)

        parse_block(uk_file.read(26), HEADER_TWO)  # second fixed header

        leemdata_version = parse_bytes(uk_file.read(2), 0, "short")
        if leemdata_version != 2:
            uk_file.seek(388, 1)
        bit = uk_file.read(1)[0]
        while bit != 255:
            if debug:
                print(bit)
            if bit in VARIABLE_HEADER:  # fixed byte codes
                block_length, field_dict = VARIABLE_HEADER[bit]
                buffer = uk_file.read(block_length)
                parse_block(buffer, field_dict)
                if debug:
                    print("\tknown")
            elif bit in (106, 107, 108, 109, 234, 235, 236, 237):  # varian pressures
                key = parse_cp1252_until_null(uk_file, debug)
                data[f"{key}_unit"] = parse_cp1252_until_null(uk_file, debug)
                data[key] = parse_bytes(uk_file.read(4), 0, "float")
                if debug:
                    print(f"\tknown: pressure {key} -> {data[key]}")
            elif bit in (110, 238):  # field of view
                fov_str = parse_cp1252_until_null(uk_file, debug)
                try:
                    data["fov"], data["preset"] = fov_str.split("\t")
                except ValueError:
                    data["fov"], data["preset"] = fov_str, ""
                data["fov_cal"] = parse_bytes(uk_file.read(4), 0, "float")
                if debug:
                    print(f"\tfov: {fov_str}\n\tfov_cal: {data['fov_cal']}")
            elif bit in (0, 1, 63, 66, 113, 128, 176, 216, 240, 232, 233):
                if debug:
                    print(f"unknown byte {bit}")
            elif bit:  # self-labelled stuff
                keyunit = parse_cp1252_until_null(uk_file, debug)
                # For some b, the string is empty. They should go in the tuple above.
                if not keyunit:
                    bit = uk_file.read(1)[0]
                    continue
                data[f"{keyunit[:-1]}_unit"] = UNIT_CODES.get(keyunit[-1], "")
                data[keyunit[:-1]] = parse_bytes(uk_file.read(4), 0, "float")
                if debug:
                    print(f"\tunknown: {keyunit[:-1]} -> {data[keyunit[:-1]]}")
            bit = uk_file.read(1)[0]

        size = data["width"] * data["height"]
        uk_file.seek(-2 * size, 2)
        image = np.fromfile(uk_file, dtype=np.uint16, sep="", count=size)
        image = np.array(image, dtype=np.float32)
        image = np.flipud(image.reshape((data["height"], data["width"])))
        data["image"] = image

    for new, old in ATTR_NAMES.items():
        data[new] = data.pop(old, np.nan)
        data[f"{new}_unit"] = data.pop(f"{old}_unit", "")
    try:
        if data["averaging"] == 0:
            data["averaging"] = 1
        elif data["averaging"] == 255:
            data["averaging"] = 0
    except KeyError:
        pass
    data["energy_unit"] = "eV"

    return data


def do_ecc_align(
    input_img: Image,
    template_img: Image,
    max_iter: int = 500,
    eps: float = 1e-4,
    trafo: str = "translation",
    mask: Union[np.ndarray, None] = None,
) -> np.ndarray:
    """Finds warp matrix between two Images using ECC

    Takes two images and calculates the warp matrix between both applying the transformation
    specified by 'trafo' using the ECC Algorithm. A optional mask can be given to masked areas of
    the images that are not taken into account during registration

    Parameters
    ----------
    input_img : Image
        Image that will be warped to match the tempalate image
    template_img : Image
        The template image
    max_iter : int, optional
        Number of iterations to find the warp matrix. Regstration fails if max_iter is
        exceeded, by default 500
    eps : float, optional
        Abortion criterion to determine successfull registration, by default 1e-4
    trafo : str, optional
        The transformation that is applied to the images. 'trafo' must be either

        * "translation" for only x,y shifting of images
        * "rigid" or "euclidean" for translation, rotation and scaling
        * "affine" for translation, rotation, scaling and shearing

        , by default "translation"
    mask : Union[np.ndarray, None], optional
        The mask must be a 2D numpy array of dtype=np.uint8 containing '0' where pixels in the
        images will not be taken into account during registration and '1' when they are, by
        default 'None'

    Returns
    -------
    np.ndarray
        3x3 numpy array representing the warp matrix
    """

    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, max_iter, eps)
    if trafo == "translation":
        warp_mode = cv.MOTION_TRANSLATION
    elif trafo in ("euclidean", "rigid"):
        warp_mode = cv.MOTION_EUCLIDEAN
    elif trafo == "affine":
        warp_mode = cv.MOTION_AFFINE
    else:
        print("Unrecognized transformation. Using Translation.")
        warp_mode = cv.MOTION_TRANSLATION

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    template_arr = cv.normalize(
        template_img.image,
        None,
        0,
        255,
        cv.NORM_MINMAX,
        -1,
        mask,
    )
    input_arr = cv.normalize(
        input_img.image,
        None,
        0,
        255,
        cv.NORM_MINMAX,
        -1,
        mask,
    )

    for sigma in [11, 5]:

        _, warp_matrix = cv.findTransformECC(  # template = warp_matrix * input
            # template_img.image,
            # input_img.image,
            template_arr,
            input_arr,
            warp_matrix,
            warp_mode,
            criteria,
            mask,  # hide everything that is not in ROI
            sigma,  # gaussian blur to apply before
        )
    # Expand to 3x3 matrix
    warp_matrix = np.append(warp_matrix, [[0, 0, 1]], axis=0)

    return warp_matrix
