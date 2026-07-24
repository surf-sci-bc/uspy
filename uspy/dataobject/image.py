"""
2D spatial data containers (images and image stacks).
"""

# pylint: disable=abstract-method
from __future__ import annotations
from typing import Any, Union
from collections.abc import Iterable

import numpy as np
import imageio
import tifffile
from tifffile.tifffile import TiffFileError
import cv2 as cv

from uspy.dataobject.base import DataObject, DataObjectStack, _ArithmeticMixin


class Image(_ArithmeticMixin, DataObject):
    """
    Base class for all spatial 2D data.
    """

    # pylint: disable=no-member
    _data_keys = ("image",)
    _meta_keys = ("width", "height")
    _op_attr = "image"

    def parse(self, source: Union(str, np.ndarray)) -> dict[str, Any]:
        if isinstance(source, np.ndarray):
            # if the object given already is a numpy array:
            image = source
            self._source = None
        elif source.lower().endswith((".tiff", ".tif")):
            image = tifffile.imread(source)
            self._source = source
        else:
            image = np.float32(imageio.imread(source))
            self._source = source

        if image.ndim != 2:
            raise ValueError(f"{source} is not a single image")

        return_val = {"image": image, "width": image.shape[1], "height": image.shape[0]}
        return return_val

    def save(self, fname: str, **kwargs) -> None:
        if fname.lower().endswith((".tiff", ".tif")):
            tifffile.imwrite(fname, self.image, **kwargs)
        else:
            try:
                imageio.imwrite(fname, np.uint16(self.image))
            except ValueError:
                super().save(fname)

    def filter(self, method="gaussian", **kwargs):
        """Apply 2D Filter to image

        Applies a 2D Filter to the image, specified by the *method* argument. Valid methods are
        "gaussian", "blur", "median" and "kernel". Depending on the selected method different kwargs
        can be passed.


        Parameters
        ----------
        method : str, optional
            Applied filter type. Either "gaussian", "blur", "median" or "kernel", by default "gaussian".
        kwargs : Depending on the passed method

        Methods: possible kwargs are specified below

        - gaussian: Applies a gaussian filter to the image

            - size : int or tuple, by default (3, 3)
                size of the applied kernel
            - sigma: int, by default -1
                width of the applied gaussian. When negative, sigma is calculated from kernel

        - blur: Applies a normlized box filter with size of kernel

            - size: int or tuple, by default (3,3)
                size of applied kernel

        - media: Applies median filter

            - size: int, by default 3
                size of kernel

        - kernel: Convolutes image with kernel

            - kernel: np.ndarray, by default None
                kernel to be applied to the image

        Returns
        -------
        Image
            Returns image with applied filter.

        Raises
        ------
        ValueError
            If specified method is not valid.
        """

        img = self.copy()

        if method == "gaussian":
            sigma = kwargs.pop("sigma", -1)
            size = kwargs.pop("size", (3, 3))
            if isinstance(size, int):
                size = (size, size)
            img.image = cv.GaussianBlur(self.image, size, sigma, **kwargs)
        elif method == "blur":
            size = kwargs.pop("size", (3, 3))
            if isinstance(size, int):
                size = (size, size)
            img.image = cv.blur(self.image, size, **kwargs)
        elif method == "median":
            size = kwargs.pop("size", 3)
            img.image = cv.medianBlur(self.image, size)
        elif method == "kernel":
            kernel = kwargs.pop("kernel", None)
            img.image = cv.filter2D(self.image, ddepth=-1, kernel=kernel, **kwargs)
        else:
            raise ValueError("Unkown Filter")

        return img

    def contrast(self, method="equal"):
        img = self.copy()
        if method == "auto":
            pass

        if method == "equal":
            img.image = cv.equalizeHist(img.image.astype(np.uint8))
        elif method == "clahe":
            clahe = cv.createCLAHE()
            img.image = clahe.apply(img.image)

        return img

    def is_compatible(self, other: DataObject) -> bool:
        if not super().is_compatible(other):
            return False
        try:
            return self.image.shape == other.image.shape
        except AttributeError:
            return False


class ImageStack(DataObjectStack):
    """DataObjectStack for Images which provides a default parsing mechanism for
    common image formats like png or tiff."""

    _type = Image

    def _split_source(self, source: Union[str, Iterable]) -> list:
        if isinstance(source, list):
            return super()._split_source(source)

        if isinstance(source, np.ndarray):
            # if the object given already is a numpy array:
            images = source
        else:
            try:
                images = tifffile.imread(source)
            except (
                FileNotFoundError,
                TypeError,
                TiffFileError,
            ):  # if not check if list of images
                return super()._split_source(source)

        if len(images.shape) != 3:
            raise ValueError(f"{source} is not a stack")

        return [images[i, :, :] for i in range(images.shape[0])]

    def average(self, avg: int) -> ImageStack:
        """Averages over consecutive images in a stack.

        Parameters
        ----------
        avg : int
            Number of consecutive image that will be averaged. The total number of images must be
            dividable by *avg*.

        Returns
        -------
        ImageStack
            Stack with averaged images. The metadata of the first element in every averaged slice of
            images is preserved.

        Raises
        ------
        ValueError
            If number of images in stack is not dividable by *avg*

        Examples
        --------
        >>> stack = ImageStack("testdata/test_stack_IV_RuO2_normed_aligned_80-140.tif")[0:60]
        >>> len(stack)
        60
        >>> avg_stack = stack.average(4)
        >>> len(avg_stack)
        15

        """
        if len(self) % avg != 0:
            raise ValueError(
                f"Number of Images in Stack ({len(self)}) is not dividable by {avg}"
            )

        imgs = []

        for i in range(0, len(self) // avg):
            img = np.mean(self[i * avg : i * avg + avg])
            imgs.append(img)

        return type(self)(imgs)

    def save(self, fname: str) -> None:
        if fname.lower().endswith((".tiff", ".tif")):
            array = np.stack([image.image for image in self])
            tifffile.imwrite(fname, array)
        else:
            super().save(fname)

    def filter(self, *args, **kwargs):
        return type(self)([img.filter(*args, **kwargs) for img in self])
