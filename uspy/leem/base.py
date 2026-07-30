"""
Basic classes for LEEM data parsing and data visualization.

Parsing of specific file formats/endstations is delegated to a pluggable "reader" (see
uspy.leem.readers) -- the Elmitec UKSOFT2001 ".dat" format is the default.
"""

# pylint: disable=missing-docstring
# pylint: disable=attribute-defined-outside-init
# plyint: disable=access-member-before-definition

from __future__ import annotations
from typing import Any, Union, Optional
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from numbers import Number
from datetime import datetime
import os
import warnings
from tqdm.auto import tqdm
import pandas as pd

import numpy as np
import cv2 as cv

import tifffile
from tifffile.tifffile import TiffFileError

from uspy.dataobject import Image, ImageStack
from uspy.dataobject.image import list_image_files
from uspy.dataobject.deferred import DeferredArray, ReaderPixels, is_deferred
from uspy.derived import StitchedLine
from uspy.leem.utility import imgify
from uspy.roi import ROI
from uspy.leem.readers import LEEMReader, resolve_reader, resolve_plotting_fields
# Re-exported for backward compatibility -- the Elmitec binary parser and its lookup
# tables now live in uspy.leem.readers.elmitec.
from uspy.leem.readers.elmitec import (  # noqa: F401  pylint: disable=unused-import
    parse_dat,
    ATTR_NAMES,
    HEADER_ONE,
    HEADER_TWO,
    VARIABLE_HEADER,
    UNIT_CODES,
)


def _nan_to_num_zeroed(arr: np.ndarray) -> np.ndarray:
    """`np.nan_to_num(arr, posinf=0, neginf=0)`, as a module-level (picklable) function so it
    can be used with `DeferredArray.map` -- see `LEEMImg.normalize`."""
    return np.nan_to_num(arr, posinf=0, neginf=0)


def _warp_perspective(warp_matrix: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """`cv.warpPerspective(arr, warp_matrix, ...)`, as a module-level function bound to
    `warp_matrix` via `functools.partial` (picklable, unlike a closure/lambda) so it can be
    used with `DeferredArray.map` -- see `LEEMImg.warp`."""
    return cv.warpPerspective(
        arr,
        warp_matrix,
        arr.shape[::-1],
        flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
    )


@dataclass(frozen=True)
class _LEEMMetaLoader:
    """Metadata loader for a deferred `LEEMImg`: `reader.parse_metadata(source)`, plus the
    same temperature sanitization `LEEMImg.parse` applies for the eager path -- so a deferred
    and an eager LEEMImg built from the same source resolve to identical metadata."""

    reader: Any
    source: Any

    def __call__(self) -> dict:
        idict = dict(self.reader.parse_metadata(self.source))
        if idict.get("temperature", 0) > 3000:
            idict["temperature"] = np.nan
        return idict


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
        # Photon energy and a manual energy offset, used to compute binding_energy below.
        # Only meaningful for readers that populate photon_energy (e.g. a synchrotron
        # beamline reader); default to a LEED/LEEM electron gun with no photon energy.
        "photon_energy": np.nan,
        "energy_offset": 0,
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
        "photon_energy": "eV",
    }
    default_fields = ("temperature", "pressure", "energy", "fov")
    meta_fields = ("temperature",)
    print_fields = (
        "energy",
        "temperature",
        "pressure1",
        "pressure2",
        "objective",
        "fov",
        "exposure",
        "averaging",
        "width",
        "height",
    )

    def __init__(
        self,
        *args,
        reader: Union[str, LEEMReader, None] = None,
        time_origin: Union[TimeOrigin, Number] = None,
        defer: bool = False,
        **kwargs,
    ) -> None:
        source = args[0] if args else kwargs.get("source")
        self._reader = resolve_reader(source, reader)
        # object.__setattr__ needed: default_fields/meta_fields/print_fields are class
        # attributes, and a normal `self.x = ...` would route through DataObject.__setattr__
        # into self._meta instead of overriding the class attribute.
        #
        # resolve_plotting_fields does a uspy.toml search-path walk, but the expensive part
        # of that (the resolve()-based candidate-folder computation) is cached at the
        # uspy.config layer (see config.config_search_paths) keyed on the data folder/cwd/
        # global config dir -- so calling it unconditionally, once per LEEMImg, stays cheap
        # for repeat frames from the same folder without LEEMImg/LEEMStack needing to know
        # anything about plotting fields themselves.
        plotting_fields = resolve_plotting_fields(
            source,
            {
                "default_fields": type(self).default_fields,
                "meta_fields": type(self).meta_fields,
                "print_fields": type(self).print_fields,
            },
        )
        for key, value in plotting_fields.items():
            object.__setattr__(self, key, value)
        # gap-fill any reader-specific defaults (e.g. a beamline reader's photon_energy)
        # that aren't already covered by this class' own _meta_defaults/_unit_defaults.
        # Writes the stores directly, not through the _meta/_units properties: at this point
        # in construction no loader has been installed yet either way (DataObject.__new__
        # always runs first and seeds it to None), so this is equivalent either way -- but
        # spelling it this way keeps this gap-fill correct regardless of ordering.
        for key, value in self._reader.meta_defaults.items():
            self._meta_store.setdefault(key, value)
        for key, value in self._reader.unit_defaults.items():
            self._units_store.setdefault(key, value)
        if not isinstance(time_origin, TimeOrigin):
            time_origin = TimeOrigin(time_origin)
        super().__init__(*args, defer=defer, **kwargs)
        self._time_origin = time_origin  # is a list so it can be mutable
        # Writes the store directly, not `self.default_mask = None` (which would go through
        # the _meta property and immediately resolve/consume the metadata loader installed
        # just above by a deferred construction -- defeating the entire point of deferring).
        self._meta_store["default_mask"] = None  # ROI.circle(
        #     x0=self.width // 2, y0=self.height // 2, radius=self.width // 2
        # )

    def parse_deferred(self, source: Any) -> dict[str, Any]:
        """Deferred (`defer=True`) counterpart of `parse`: builds a cheap, reader-backed
        `image` handle and metadata loader when this reader recognizes `source` (the common
        case), falling back to the generic re-parse machinery (`Image`/`DataObject`'s
        `parse_deferred`) for anything it doesn't -- e.g. a plain TIFF/PNG opened without a
        LEEM-specific reader. Mirrors the `reader.handles(source)` gate `parse` itself uses,
        so the same sources route to the same reader either way."""
        if self._reader.handles(source):
            self._source = source
            reader = self._reader
            return {
                # All current readers convert pixels to float32 -- hinting the dtype here
                # means Image._dtype never needs to force a load just to learn it.
                "image": DeferredArray(ReaderPixels(reader, source), dtype=np.float32),
                "_meta_loader": _LEEMMetaLoader(reader, source),
            }
        return super().parse_deferred(source)

    def get_field_string(self, field: str, fmt: Optional[str] = None) -> str:
        """Get a string that contains the value and its unit. Some LEEM-specific fields
        have different default formats than the pure DataObject method."""
        if fmt is None:
            if field == "temperature":
                fmt = ".0f"
            elif field == "energy":
                fmt = ".3g"
            elif field == "rel_time":
                fmt = ".0f"
        return super().get_field_string(field, fmt)

    def warp(self, warp_matrix=None, inplace: bool = False):

        if warp_matrix is None:
            warp_matrix = self.warp_matrix

        current = self._data["image"]  # pylint: disable=protected-access
        fn = partial(_warp_perspective, warp_matrix)
        if is_deferred(current):
            # Compose into a new DeferredArray instead of loading -- this is what keeps
            # `LEEMStack.align()` out-of-core on a virtual stack (see module docstring on
            # `_warp_perspective`). The warp doesn't change shape/dtype, so both hints carry
            # over from the pre-warp handle, straight from its own (already-known-or-not)
            # cache rather than the possibly-load-triggering `.shape`/`dtype` properties.
            image = current.map(fn, shape=current._shape, dtype=current._dtype)  # pylint: disable=protected-access
        else:
            image = fn(current)

        if inplace:
            self._data["image"] = image  # pylint: disable=protected-access
            self.warp_matrix = warp_matrix
            return self

        result = self.copy()
        result._data["image"] = image  # pylint: disable=protected-access
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

        # In-place `/=` and `nan_to_num(..., copy=False)` avoid two transient full-image
        # allocations here -- this loop runs once per image in `LEEMStack.normalize`.
        result = self - dark_counts
        result /= mcp - dark_counts
        result_image = result._data["image"]  # pylint: disable=protected-access
        if is_deferred(result_image):
            # `result.image` would materialize a *fresh copy* here -- mutating it with
            # nan_to_num(copy=False) would fix nothing on the object actually returned.
            # Compose the cleanup into the deferred value instead, so it survives.
            result._data["image"] = result_image.map(_nan_to_num_zeroed)  # pylint: disable=protected-access
        else:
            np.nan_to_num(result_image, copy=False, posinf=0, neginf=0)

        if inplace:
            # Direct _data write (not `self.image = result.image`, which would materialize
            # `result` and then pin the resident copy onto `self`) so a deferred `self` stays
            # deferred, holding `result`'s composed DeferredArray instead.
            self._data["image"] = result._data["image"]  # pylint: disable=protected-access
            self.mcp = mcp.source
            self.dark_counts = dark_counts
            return self

        result.mcp = mcp.source
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
            if self._reader.handles(source):
                self._source = source
                idict = self._reader.parse_image(source)
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
            "reader": self._reader.name,
        }

    def __json_decode__(self, **attrs) -> None:
        self.__init__(source=attrs["source"], reader=attrs.get("reader"))
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

    @property
    def kinetic_energy(self) -> Number:
        """Alias of 'energy', the kinetic energy of the imaging electrons (in eV)."""
        return self.energy

    @property
    def binding_energy(self) -> Number:
        """Binding energy (in eV), meaningful when the image was recorded with a known
        'photon_energy' (e.g. at a synchrotron beamline; see uspy.leem.readers)."""
        return self.photon_energy - self.energy - self.energy_offset

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
    """Stack of `LEEMImg`s. `virtual=True` builds elements with deferred pixel data (see
    `uspy.dataobject.base.DataObjectStack`) -- construction is then bounded by a directory
    listing plus one frame's metadata (to seed `time_origin`), and every operation (indexing,
    metadata gathers, arithmetic, `.normalize()`, `.align()`) works the same as on a fully
    resident stack, reading pixel data only where it's actually needed.
    """

    _type = LEEMImg

    def __init__(
        self,
        *args,
        reader: Union[str, LEEMReader, None] = None,
        time_origin: Optional[Union[TimeOrigin, Number]] = None,
        **kwargs,
    ) -> None:
        source = args[0] if args else kwargs.get("source")
        # If built directly from a list of already-built LEEMImgs (as slicing, mask
        # filtering, and .average() all do internally), inherit their reader instead of
        # defaulting to Elmitec, and skip post_process_stack below -- those elements'
        # metadata is already correct, and re-running it would re-read the reader's source
        # data against a differently-sized stack.
        built_from_objects = False
        try:
            if not isinstance(source, str) and isinstance(source[0], LEEMImg):
                built_from_objects = True
                if reader is None:
                    reader = source[0]._reader
        except (TypeError, IndexError):
            pass
        self._reader = resolve_reader(source, reader)
        # An already-built TimeOrigin (e.g. from __getitem__) is reused by reference so the
        # new stack shares its source's mutable origin.
        if isinstance(time_origin, TimeOrigin):
            self._time_origin = time_origin
        else:
            self._time_origin = TimeOrigin(time_origin)
        super().__init__(*args, **kwargs)
        if not built_from_objects:
            self._reader.post_process_stack(self)
        # now we can access the 0th element and set the value
        if time_origin is None:
            self._time_origin.value = self[0].timestamp

    def _split_source(self, source: Union[str, Iterable]) -> list:
        fnames = self._reader.list_files(source)
        if fnames:
            return fnames

        try:
            return super()._split_source(source)
        except ValueError:
            raise ValueError(f"No files were found at path {source}")

    def _single_construct(self, source: Any, defer: bool = False) -> LEEMImg:
        """Construct a single DataObject, resident or deferred."""
        return LEEMImg(source, reader=self._reader, time_origin=self._time_origin, defer=defer)

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
            # _elements always holds real LEEMImgs now (virtual or not) -- no re-parsing.
            return elements
        return type(self)(
            elements,
            virtual=self.virtual,
            time_origin=self._time_origin,
            reader=self._reader,
        )

    def extend(self, other: Union[LEEMImg, LEEMStack]) -> None:
        """Add new LEEMImg(s) to the end of the stack, reconciling `time_origin`: if `other`
        has a different one (e.g. two halves of a split/resumed measurement), a warning
        names the mismatch and the appended elements are rebound to this stack's
        `_time_origin` (whether this stack is virtual or not -- elements are real, stable
        DataObjects either way now) so `rel_time` shares one origin. `timestamp` itself is
        never touched."""
        if isinstance(other, (LEEMImg, LEEMStack)):
            self_origin, other_origin = self.time_origin, other.time_origin
            if not (np.isnan(self_origin) or np.isnan(other_origin)) and (
                self_origin != other_origin
            ):
                warnings.warn(
                    UserWarning(
                        f"Appending elements with a time origin ({other_origin} s) that "
                        f"differs from this stack's ({self_origin} s) -- their rel_time will "
                        "now be computed relative to this stack's time origin."
                    )
                )
        n_before = len(self._elements)
        super().extend(other)
        for element in self._elements[n_before:]:
            element._time_origin = self._time_origin

    def align(
        self,
        inplace: bool = False,
        mask: Union[bool, np.ndarray, None] = True,
        template: Union[int, Image, None] = None,
        sanity: bool = True,
        **kwargs,
    ) -> LEEMStack:
        """
        Image registration of the stack.

        By default (``template=None``) images are registered sequentially, each against
        its predecessor via ``Image.find_warp_matrix()``, with shifts accumulated so every
        warp matrix is relative to the first image. Passing ``template`` instead aligns
        every image directly against that one fixed image.

        Recovered translation shifts are then checked with a robust (median/MAD-based)
        outlier score (`_modified_zscore`); outliers and failed-registration frames are
        replaced by interpolation/extrapolation over the remaining valid shifts (printed,
        not warned -- this is the expected, routine path). If too few valid frames remain
        to interpolate, affected frames fall back to identity shift and a ``UserWarning``
        names them. The resulting warp matrices are applied via ``Image.warp()``.

        On a virtual stack (``virtual=True``), the warp is composed onto each element's
        deferred pixel data instead of being applied and stored -- the returned stack (or
        ``self``, for ``inplace=True``) stays virtual, and every frame is re-warped from
        ``source`` on each subsequent pixel read (the usual never-cached deferred trade-off;
        see ``uspy.dataobject.deferred``).

        Parameters
        ----------
        inplace : bool, default False
            If 'True' the stack itself will be registered
            If 'False' a copy of the stack will be registered. For an already-resident stack,
            peak memory is briefly ~2x the stack's size, since the whole stack is deep-copied
            up front (prefer 'inplace=True' for a stack close to the memory limit). For a
            virtual stack this copy is free (deferred pixel data is never actually
            duplicated) and the result stays virtual either way.
        mask : bool, np.ndarray or None, default True
            Mask that will be used during registration

            - If 'True' a circular mask with a radius=0.9*Image.width is used as mask
            - If 'False' or 'None' no mask is used during registration
            - If np.ndarray the array must be of dtype=np.uint8 and contain the mask that should be
              applied
        template : int, Image or None, default None
            If given, every image is aligned directly against this fixed reference image
            instead of sequentially against its predecessor. An int indexes 'stack'
            ('template=0' = first image); an 'Image' is used directly. Less
            battle-tested than sequential alignment -- sanity-check the result.
        sanity : bool, default True
            Whether to run the z-score outlier check described above.

        **kwargs
            Additional keyword arguments are passed through to Image.find_warp_matrix

        Returns
        -------
        LEEMStack
            LEEMStack with aligned images. The stack itself is returned when 'inplace' = True

        Raises
        ------
        ValueError
            If the stack is empty.
        TypeError
            If 'template' is neither an int, an Image, nor None.
        IndexError
            If 'template' is an int outside the stack's range.

        See Also
        --------
        find_warp_matrix
        LEEMImg.warp
        """
        if len(self) == 0:
            raise ValueError("Cannot align an empty stack.")

        if inplace:
            stack = self
        else:
            stack = self.copy()

        if len(stack) == 1:
            # Nothing to register against -- leave the single frame as-is (identity
            # warp) instead of feeding a 1-point array into the interpolation code
            # below, which needs at least 2 valid points.
            stack[0].warp(warp_matrix=np.eye(3, dtype=np.float32), inplace=True)
            return stack

        if sanity and kwargs.get("trafo", "translation") != "translation":
            warnings.warn(
                "align(): sanity-checking only inspects the recovered translation "
                f"(dx, dy) -- rotation/scale/shear from trafo={kwargs.get('trafo')!r} "
                "are not sanity-checked."
            )

        if mask:
            roi = ROI.circle(
                x0=stack[0].width // 2,
                y0=stack[0].height // 2,
                radius=stack[0].width // 2 * 9 // 10,
            )
            mask = (~np.ma.getmaskarray(roi.apply(stack[0]).image)).astype(np.uint8)
        else:
            mask = None

        if template is None:
            # List of all warp matrices, relative to the first image. The first
            # image's own warp is always the trivial identity -- it is never run
            # through find_warp_matrix.
            warp_matrices = [np.eye(3, dtype=np.float32)]
            for index, (img1, img2) in enumerate(zip(tqdm(stack[1:]), stack)):
                try:
                    warp_matrix = img1.find_warp_matrix(img2, mask=mask, **kwargs)
                except Exception:
                    print(
                        f"Alignment failed on Image {index + 1}. Will interpolate later"
                    )
                    warp_matrix = np.eye(3, dtype=np.float32)
                    warp_matrix[0:2, 2] = np.nan
                warp_matrices.append(warp_matrix)
        else:
            if isinstance(template, (int, np.integer)):
                template = stack[template]
            elif not isinstance(template, Image):
                raise TypeError(
                    "template must be an int index into the stack, an Image, or "
                    f"None, got {type(template)}"
                )
            warnings.warn(
                "align(): aligning against a fixed template is less battle-tested "
                "than the default sequential alignment -- sanity-check the result."
            )
            # Every image is aligned directly and independently against the template, so
            # each warp matrix is already absolute -- must NOT be cumulatively composed.
            warp_matrices = []
            for index, img in enumerate(stack):
                try:
                    warp_matrix = img.find_warp_matrix(template, mask=mask, **kwargs)
                except Exception:
                    print(f"Alignment failed on Image {index}. Will interpolate later")
                    warp_matrix = np.eye(3, dtype=np.float32)
                    warp_matrix[0:2, 2] = np.nan
                warp_matrices.append(warp_matrix)

        dx = np.array([matrix[0, 2] for matrix in warp_matrices])
        dy = np.array([matrix[1, 2] for matrix in warp_matrices])

        dx, dy = _resolve_shifts(dx, dy, sanity=sanity)

        for index, (x, y) in enumerate(zip(dx, dy)):
            warp_matrices[index][0, 2] = x
            warp_matrices[index][1, 2] = y

        if template is None:
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
            # Skip an image already normalized against this mcp (relevant if inplace=True
            # and the same image appears twice in the stack). img.mcp holds the *source* of
            # the mcp it was last normalized against, hence comparing to mcp.source.
            if img.mcp is None or img.mcp != mcp.source:
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
        uspy.derived.StitchedLine
        """
        super().__init__(*args, **kwargs, xaxis="energy")


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


#: Threshold on the robust (modified) z-score above which a shift is rejected as a
#: registration outlier -- see `_modified_zscore`. Deliberately high (rather than the
#: textbook 3.5): a genuine failure scores in the tens-hundreds, so this stays
#: well-separated from normal frame-to-frame shifts. Empirically chosen.
_OUTLIER_ZSCORE_THRESHOLD = 8.0


def _modified_zscore(x: np.ndarray) -> np.ndarray:
    """Robust (median/MAD-based) z-score, used by `_resolve_shifts` to flag registration
    outliers. Unlike a mean/std z-score, a single huge outlier can't inflate the scale
    enough to mask itself or other outliers. Falls back to mean-absolute-deviation scaling
    if MAD collapses to 0, or to all zeros if every value is identical.

    Parameters
    ----------
    x : np.ndarray
        Values to score. nan entries stay nan (so they never exceed a threshold).

    Returns
    -------
    np.ndarray
        Non-negative robust z-scores, same shape as 'x'.
    """
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    abs_dev = np.abs(x - med)
    mad = np.nanmedian(abs_dev)
    if mad > 0:
        return np.abs(0.6745 * (x - med) / mad)
    meanad = np.nanmean(abs_dev)
    if meanad > 0:
        return np.abs((x - med) / (1.253314 * meanad))
    return np.zeros_like(x)


def _resolve_shifts(
    dx: np.ndarray, dy: np.ndarray, sanity: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Cleans up raw x/y translation shifts from pairwise/template registration, used by
    `LEEMStack.align()`. A failed registration marking only one of dx/dy as nan is
    unified to both. When 'sanity' is set, shifts whose robust z-score (`_modified_zscore`)
    exceeds `_OUTLIER_ZSCORE_THRESHOLD` are flagged as outliers and set to nan too
    (printed, not warned -- this is the expected, routine path).

    All nan gaps are then filled by interpolation/extrapolation, with spline order
    adapted to the number of valid frames so short stacks don't crash. If too few valid
    points remain (0 or 1) to interpolate at all, affected frames fall back to identity
    (zero shift) and a warning names them.

    Parameters
    ----------
    dx, dy : np.ndarray
        Raw per-frame x/y translation shifts, with nan marking a failed registration.
    sanity : bool, optional
        Whether to run the z-score outlier check, by default True.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Cleaned up dx, dy with no remaining nan.
    """
    dx = np.array(dx, dtype=float)
    dy = np.array(dy, dtype=float)
    n = len(dx)

    # A failed registration should mark both components -- unify the two nan masks
    # instead of asserting they already match, so a registration that only marked one
    # component as nan doesn't crash the whole alignment.
    bad = np.isnan(dx) | np.isnan(dy)
    dx[bad] = np.nan
    dy[bad] = np.nan

    if sanity:
        n_valid = int((~np.isnan(dx)).sum())
        # A robust z-score needs a handful of points to be meaningful -- skip on short input.
        if n_valid >= 4:
            dx_z = _modified_zscore(dx)
            dy_z = _modified_zscore(dy)
            outliers = (dx_z > _OUTLIER_ZSCORE_THRESHOLD) | (
                dy_z > _OUTLIER_ZSCORE_THRESHOLD
            )
            for index in np.where(outliers)[0]:
                print(f"Align {index} too far out")
            dx[outliers] = np.nan
            dy[outliers] = np.nan

    valid = ~np.isnan(dx)
    n_valid = int(valid.sum())

    if n_valid == n:
        return dx, dy

    if n_valid == 0:
        warnings.warn(
            "align(): no frame could be registered -- falling back to identity (no "
            "shift) for the entire stack."
        )
        return np.zeros(n), np.zeros(n)

    if n_valid == 1:
        # A single valid point can't be interpolated or extrapolated (even a linear
        # fit needs >= 2 points) -- keep that one frame's own shift and fall back to
        # identity for the rest.
        bad_idx = list(np.where(~valid)[0])
        warnings.warn(
            f"align(): only one frame could be registered -- frame(s) {bad_idx} "
            "could not be recovered and fall back to identity (no shift)."
        )
        dx = np.where(valid, dx, 0.0)
        dy = np.where(valid, dy, 0.0)
        return dx, dy

    # Interpolate (and extrapolate leading/trailing gaps) over the valid points.
    # Pandas' spline interpolation needs at least 'order + 1' points, so the order
    # is capped by the number of valid points instead of a hard-coded 3.
    order = min(3, n_valid - 1)
    try:
        dx_filled = (
            pd.Series(dx)
            .interpolate("spline", order=order, limit_direction="both")
            .to_numpy()
        )
        dy_filled = (
            pd.Series(dy)
            .interpolate("spline", order=order, limit_direction="both")
            .to_numpy()
        )
    except Exception:
        # Defensive fallback in case the spline fit is numerically unhappy with a
        # particular arrangement of points: plain linear interpolation with flat
        # (nearest-valid-value) extrapolation at the boundaries never raises.
        idx = np.arange(n)
        valid_idx = idx[valid]
        dx_filled = np.interp(idx, valid_idx, dx[valid])
        dy_filled = np.interp(idx, valid_idx, dy[valid])

    residual = np.isnan(dx_filled) | np.isnan(dy_filled)
    if residual.any():
        bad_idx = list(np.where(residual)[0])
        warnings.warn(
            f"align(): frame(s) {bad_idx} could not be recovered by interpolation -- "
            "falling back to identity (no shift) for them."
        )
        dx_filled[residual] = 0.0
        dy_filled[residual] = 0.0

    return dx_filled, dy_filled
