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


# Recognized per-frame image file suffixes for glob/folder expansion (list_image_files below)
# -- deliberately mirrors what uspy.leem.readers.elmitec/elettra recognize for their own
# formats, just for the generic image types tifffile/imageio can read.
IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def _has_glob_magic(pattern: str) -> bool:
    return any(char in pattern for char in "*?[")


def list_image_files(source: Any) -> list:
    """Expand a glob pattern or an existing directory into a natural-sorted list of per-frame
    image file paths (".tif"/".tiff"/".png"/".jpg"/".jpeg"/".bmp"). Mirrors
    `LEEMReader.list_files` (see `uspy.leem.readers`) so glob/folder discovery behaves the
    same way everywhere in the codebase -- including natural sort order, so e.g.
    "frame_2.tiff" sorts before "frame_10.tiff".

    Returns `[]` for anything that isn't a glob pattern or an existing directory (a plain
    file path, a list, an ndarray, ...); callers fall back to their own handling in that case.
    """
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

    def incompatibility_reason(self, other: DataObject) -> str:
        if not isinstance(other, Image):
            return super().incompatibility_reason(other)
        try:
            other_shape = other.image.shape
        except AttributeError:
            return "the other element has no image data"
        return f"image shape {other_shape} != {self.image.shape}"


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
            if len(images.shape) != 3:
                raise ValueError(f"{source} is not a stack")
            return [images[i, :, :] for i in range(images.shape[0])]

        # A glob pattern or an existing directory expands to one or more per-frame image
        # files -- returned directly (as file paths, not pixel arrays) when it matches more
        # than one file, so per-frame construction keeps each element's own shape and
        # `source`. Previously any such string was handed straight to `tifffile.imread`,
        # whose FileSequence machinery silently force-reshapes every matched file into frame
        # 0's shape (corrupting mismatched frames instead of raising) and discards each
        # frame's path. A single matched/contained file still needs the probe below, since it
        # might itself be one 3D multi-frame TIFF rather than a lone 2D image.
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
