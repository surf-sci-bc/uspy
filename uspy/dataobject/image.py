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

        if method == "gaussian":
            sigma = kwargs.pop("sigma", -1)
            size = kwargs.pop("size", (3, 3))
            if isinstance(size, int):
                size = (size, size)
            filtered = cv.GaussianBlur(self.image, size, sigma, **kwargs)
        elif method == "blur":
            size = kwargs.pop("size", (3, 3))
            if isinstance(size, int):
                size = (size, size)
            filtered = cv.blur(self.image, size, **kwargs)
        elif method == "median":
            size = kwargs.pop("size", 3)
            filtered = cv.medianBlur(self.image, size)
        elif method == "kernel":
            kernel = kwargs.pop("kernel", None)
            filtered = cv.filter2D(self.image, ddepth=-1, kernel=kernel, **kwargs)
        else:
            raise ValueError("Unkown Filter")

        return self._with_data(image=filtered)

    def contrast(self, method="equal"):
        if method == "equal":
            adjusted = cv.equalizeHist(self.image.astype(np.uint8))
        elif method == "clahe":
            clahe = cv.createCLAHE()
            adjusted = clahe.apply(self.image)
        else:
            # "auto" (unimplemented) or any other value: mirror the original's
            # copy-only fallback (an independent array, not shared with self.image).
            return self.copy()

        return self._with_data(image=adjusted)

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

    def __new__(cls, *args, lazy: bool = False, **kwargs):
        if lazy:
            # Deferred import: uspy.dataobject.lazy imports Image/ImageStack from this module,
            # so importing it at module level here would be circular.
            from uspy.dataobject.lazy import LazyImageStack

            kwargs.pop("virtual", None)
            return LazyImageStack(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self, *args, lazy: bool = False, **kwargs) -> None:
        if lazy:
            # __new__ already built and returned a standalone LazyImageStack above; since it
            # isn't an ImageStack subclass, Python won't actually call this __init__ on it --
            # this branch only guards against __init__ being invoked directly.
            return
        super().__init__(*args, **kwargs)

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
        if not fname.lower().endswith((".tiff", ".tif")):
            return super().save(fname)

        # Streamed frame-by-frame instead of `np.stack([image.image for image in
        # self])` (which builds a second full (N, H, W) copy of the whole stack,
        # alongside the already-resident per-frame arrays) -- bounds memory to one
        # frame at a time. Still matches `np.stack`'s dtype promotion across frames
        # of differing dtype, so the written file is identical either way.
        first = self[0]
        dtype = first.image.dtype
        for image in self[1:]:
            dtype = np.promote_types(dtype, image.image.dtype)
        shape = (len(self),) + first.image.shape

        def frames():
            for image in self:
                yield image.image.astype(dtype, copy=False)

        # Passing a generator + explicit shape/dtype (rather than building one big
        # array) keeps this streamed. photometric="minisblack" avoids tifffile
        # misreading a 3-frame single-channel stack as RGB (its shape-guessing
        # heuristic is ambiguous for exactly 3 frames) -- see LazyImageStack.save.
        tifffile.imwrite(
            fname,
            frames(),
            shape=shape,
            dtype=dtype,
            bigtiff=True,
            photometric="minisblack",
        )
        return None

    def filter(self, *args, **kwargs):
        return type(self)([img.filter(*args, **kwargs) for img in self])

    def to_lazy(self, chunks: int = 1):
        """Rebuild this stack as an out-of-core `LazyImageStack` (a lazy dask.array of pixel
        data) over the same per-element sources, instead of the fully in-memory objects this
        stack already holds. Requires every element to have a source (e.g. built from files,
        not from raw in-memory arrays) -- this doesn't retroactively free the memory already
        held by `self`, it's for handing sources off to the lazy backend."""
        from uspy.dataobject.lazy import LazyImageStack

        sources = self.sources
        if None in sources:
            raise ValueError("Can't build a lazy stack without source information.")
        return LazyImageStack(sources, chunks=chunks)
