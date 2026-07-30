"""
2D spatial data containers (images and image stacks).
"""

# pylint: disable=abstract-method
from __future__ import annotations
from typing import Any, Union
from collections.abc import Iterable
import glob
import os

import numpy as np
import imageio
import natsort
import tifffile
from tifffile.tifffile import TiffFileError
import cv2 as cv

from uspy.dataobject.base import DataObject, DataObjectStack, _ArithmeticMixin
from uspy.dataobject.deferred import DeferredArray, materialize

try:
    # Extensions imageio itself knows how to write, keyed lowercase-with-dot (e.g. ".png").
    # Used by Image.save to decide whether an unrecognized extension should go to imageio or
    # raise -- older imageio releases without this registry fall back to attempting the write
    # unconditionally (see Image.save).
    from imageio.config import known_extensions as _IMAGEIO_KNOWN_EXTENSIONS
except ImportError:  # pragma: no cover -- depends on installed imageio version
    _IMAGEIO_KNOWN_EXTENSIONS = None


# Recognized per-frame image file suffixes for glob/folder expansion (list_image_files below)
# -- deliberately mirrors what uspy.leem.readers.elmitec/elettra recognize for their own
# formats, just for the generic image types tifffile/imageio can read.
IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def _has_glob_magic(pattern: str) -> bool:
    return any(char in pattern for char in "*?[")


def _ext(fname: str) -> str:
    """Lowercase file extension including the dot, e.g. "x.PNG" -> ".png"."""
    return os.path.splitext(fname)[1].lower()


def list_image_files(source: Any) -> list:
    """Expand a glob pattern or an existing directory into a natural-sorted list of
    per-frame image file paths (".tif"/".tiff"/".png"/".jpg"/".jpeg"/".bmp"), mirroring
    `LEEMReader.list_files`. Returns `[]` for anything else (a plain file path, a list,
    an ndarray, ...); callers fall back to their own handling in that case."""
    if not isinstance(source, str):
        return []
    if _has_glob_magic(source):
        return natsort.natsorted(path for path in glob.glob(source) if os.path.isfile(path))
    if os.path.isdir(source):
        return natsort.natsorted(
            entry.path for entry in os.scandir(source)
            if entry.is_file() and entry.name.lower().endswith(IMAGE_EXTENSIONS)
        )
    return []


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

    def parse_deferred(self, source: Any) -> dict[str, Any]:
        if isinstance(source, np.ndarray):
            # No file to defer from -- there's nothing cheaper than the array already in
            # hand, and (mirroring `parse`'s ndarray branch) no source to record either.
            # width/height are already known, so seed them directly rather than through a
            # (redundant) metadata loader.
            self._source = None
            self._meta_store["width"] = source.shape[1]
            self._meta_store["height"] = source.shape[0]
            return {"image": source}
        return super().parse_deferred(source)

    def save(self, fname: str, **kwargs) -> Image:
        # Dispatches on an explicit extension whitelist rather than trying imageio first:
        # imageio (2.37.2) silently brute-forces an unrecognized extension (e.g. ".pickle",
        # ".json") through its TIFF plugin instead of raising, so a try/except fallback
        # would never trigger and would silently write the wrong format.
        from uspy.hdf5 import is_hdf5_path

        if is_hdf5_path(fname):
            super().save(fname, **kwargs)
        elif fname.lower().endswith((".tiff", ".tif")):
            tifffile.imwrite(fname, self.image, **kwargs)
        elif self._is_serialize_path(fname):
            super().save(fname, **kwargs)
        elif _IMAGEIO_KNOWN_EXTENSIONS is None or _ext(fname) in _IMAGEIO_KNOWN_EXTENSIONS:
            # kwargs forwarded (e.g. PNG compress_level=/JPEG quality=); unlike the
            # tifffile/HDF5 branches, imageio silently ignores a typo'd kwarg.
            imageio.imwrite(fname, np.uint16(self.image), **kwargs)
        else:
            raise ValueError(
                f"Unrecognized file extension for {fname!r}: expected an HDF5 (.h5/.hdf5/"
                ".nxs), TIFF (.tif/.tiff), serialize (.pickle/.pickle.bz2/.json), or a "
                "format imageio can write (.png, .jpg, ...)."
            )
        return self

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

    @property
    def shape(self) -> tuple:
        """`(height, width)` of the image, resolved as cheaply as possible: a resident array's
        own shape; else a deferred handle's already-known shape (no load); else `height`/
        `width` metadata (a bounded, cheap resolve -- e.g. one file header -- for a deferred
        element, never a pixel read); only as a last resort, load the pixels to find out.
        A resident array always wins even if metadata is stale (e.g. after `img.image =
        smaller_array`, before anything re-derives width/height)."""
        data = self._data.get("image")
        if isinstance(data, np.ndarray):
            return data.shape
        if isinstance(data, DeferredArray) and data._shape is not None:  # pylint: disable=protected-access
            return data._shape  # pylint: disable=protected-access
        height, width = self._meta.get("height"), self._meta.get("width")
        if height is not None and width is not None:
            return (int(height), int(width))
        return materialize(data).shape

    @property
    def _dtype(self):
        """dtype of the image, with the same cheap-first precedence as `shape`. Deliberately
        *not* a public `dtype` property: numpy's internal `np.mean`/`np.sum` reduction over an
        object array of `Image`s (see `ImageStack.average`, `test_stack_img_meansum`) checks
        `hasattr(reduced_value, "dtype")` to decide whether the reduced value is a bare numpy
        scalar that needs unwrapping -- a *public* `Image.dtype` makes that check misfire and
        numpy tries (and fails) to call `float_dtype.type(some_image)`."""
        data = self._data.get("image")
        if isinstance(data, np.ndarray):
            return data.dtype
        if isinstance(data, DeferredArray) and data._dtype is not None:  # pylint: disable=protected-access
            return data._dtype  # pylint: disable=protected-access
        return materialize(data).dtype

    def is_compatible(self, other: DataObject) -> bool:
        if not super().is_compatible(other):
            return False
        try:
            return self.shape == other.shape
        except AttributeError:
            return False

    def incompatibility_reason(self, other: DataObject) -> str:
        if not isinstance(other, Image):
            return super().incompatibility_reason(other)
        try:
            other_shape = other.shape
        except AttributeError:
            return "the other element has no image data"
        return f"image shape {other_shape} != {self.shape}"


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
            if len(images.shape) != 3:
                raise ValueError(f"{source} is not a stack")
            return [images[i, :, :] for i in range(images.shape[0])]

        # A glob/directory matching more than one file returns file paths directly, so
        # per-frame construction keeps each element's own shape/source -- tifffile's
        # FileSequence machinery would otherwise force-reshape mismatched frames instead
        # of raising. A single matched file still needs the probe below (may be one 3D TIFF).
        fnames = list_image_files(source)
        if len(fnames) > 1:
            return fnames
        probe_source = fnames[0] if fnames else source

        try:
            images = tifffile.imread(probe_source)
        except (
            FileNotFoundError,
            TypeError,
            TiffFileError,
            ValueError,
            IsADirectoryError,
        ):  # if not check if list of images
            if fnames:
                return fnames
            return super()._split_source(source)

        if len(images.shape) != 3:
            if fnames:  # a single 2D image matched by a glob/folder -- keep its source
                return fnames
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
        if avg < 1:
            # Caught before the modulo check below: `avg <= 0` would slip through it
            # (e.g. 10 % -2 == 0) and then divide by zero or silently return an empty stack.
            raise ValueError(f"avg must be a positive integer, got {avg}")
        if len(self) % avg != 0:
            raise ValueError(
                f"Number of Images in Stack ({len(self)}) is not dividable by {avg}"
            )

        imgs = []

        for i in range(0, len(self) // avg):
            img = np.mean(self[i * avg : i * avg + avg])
            imgs.append(img)

        return type(self)(imgs)

    def save(self, fname: str, **kwargs) -> None:
        if not fname.lower().endswith((".tiff", ".tif")):
            return super().save(fname, **kwargs)
        if kwargs:
            raise TypeError(
                f"save() got unexpected keyword argument(s) {list(kwargs)!r}: these are only "
                "accepted for an HDF5 target (.h5/.hdf5/.nxs)."
            )
        if len(self) == 0:
            # An empty stack can't be constructed directly (see DataObjectStack.__init__), but
            # can still arise from e.g. `del stack[:]` -- mirrors uspy.hdf5.write's identical
            # empty-stack refusal for the HDF5 target.
            raise ValueError(f"Cannot export an empty stack to TIFF ({fname!r}).")

        # Streamed frame-by-frame instead of `np.stack(...)`, which would build a second
        # full (N, H, W) copy alongside the already-resident per-frame arrays.
        first = self[0]
        dtype = first.image.dtype
        for image in self[1:]:
            dtype = np.promote_types(dtype, image.image.dtype)
        shape = (len(self),) + first.image.shape

        def frames():
            for image in self:
                yield image.image.astype(dtype, copy=False)

        # Generator + explicit shape/dtype keeps this streamed; photometric="minisblack"
        # avoids tifffile misreading a 3-frame stack as RGB.
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
