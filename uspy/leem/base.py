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
from numbers import Number
from datetime import datetime
import os
import warnings
from tqdm.auto import tqdm
import pandas as pd

import numpy as np
import cv2 as cv

from uspy.dataobject import Image, ImageStack
from uspy.dataobject.lazy import LazyImageStack, da as _da
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
        **kwargs,
    ) -> None:
        source = args[0] if args else kwargs.get("source")
        self._reader = resolve_reader(source, reader)
        # resolve per-instance plotting-field defaults from a uspy.toml, if present --
        # object.__setattr__ is required here since default_fields/meta_fields/print_fields
        # are plain class attributes: a normal `self.x = ...` would be routed by
        # DataObject.__setattr__ into self._meta instead of overriding the class attribute
        # (which normal attribute lookup would keep returning, bypassing __getattr__).
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
        # that aren't already covered by this class' own _meta_defaults/_unit_defaults
        for key, value in self._reader.meta_defaults.items():
            self._meta.setdefault(key, value)
        for key, value in self._reader.unit_defaults.items():
            self._units.setdefault(key, value)
        if not isinstance(time_origin, TimeOrigin):
            time_origin = TimeOrigin(time_origin)
        super().__init__(*args, **kwargs)
        self._time_origin = time_origin  # is a list so it can be mutable
        self.default_mask = None  # ROI.circle(
        #     x0=self.width // 2, y0=self.height // 2, radius=self.width // 2
        # )

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

        # In-place `/=` and `nan_to_num(..., copy=False)` avoid two of the four
        # transient full-image allocations the naive `(a) / (b)` + reassignment would
        # produce here (`self - dark_counts` and `mcp - dark_counts` still each need
        # one copy, to avoid mutating `self`/`mcp`) -- this loop runs once per image in
        # `LEEMStack.normalize`.
        result = self - dark_counts
        result /= mcp - dark_counts
        np.nan_to_num(result.image, copy=False, posinf=0, neginf=0)

        if inplace:
            self.image = result.image
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
    """Stack of `LEEMImg`s, held fully in memory as a list of independent per-frame
    objects (see `uspy.dataobject.base.DataObjectStack`). For a stack close to or
    larger than available RAM, prefer building it lazily instead --
    ``LEEMStack(path, lazy=True)`` (or ``.to_lazy()`` on an existing stack) returns a
    `LazyLEEMStack` backed by a chunked, out-of-core `dask.array`: loading, slicing,
    `+ - * /`, `.average()`, and `.normalize()` all stay streamed/chunked there instead
    of materializing every frame at once. `.align()` and anything that touches
    `stack.image` (which always rebuilds a full `(N, H, W)` array on access, see
    `DataObjectStack.__getattr__`) still require the eager, fully in-memory `LEEMStack`
    -- call `.materialize()` on a (small enough) slice of a lazy stack for those.
    """

    _type = LEEMImg

    def __new__(cls, *args, lazy: bool = False, **kwargs):
        if lazy:
            # Deferred to call time: LazyLEEMStack is defined later in this module, but that's
            # fine since this body only runs when LEEMStack(..., lazy=True) is actually called.
            kwargs.pop("virtual", None)
            return LazyLEEMStack(*args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        *args,
        reader: Union[str, LEEMReader, None] = None,
        time_origin: Optional[Union[TimeOrigin, Number]] = None,
        lazy: bool = False,
        **kwargs,
    ) -> None:
        if lazy:
            # __new__ already built and returned a standalone LazyLEEMStack above; since it
            # isn't a LEEMStack subclass, Python won't actually call this __init__ on it --
            # this branch only guards against __init__ being invoked directly.
            return
        source = args[0] if args else kwargs.get("source")
        # If constructed directly from a list of already-built LEEMImgs (rather than from a
        # path/glob) -- as slicing, boolean-mask filtering, and .average() all do internally,
        # by rebuilding a LEEMStack from the previous stack's own elements -- inherit their
        # reader instead of silently defaulting to Elmitec (relevant if the stack is later
        # virtualized/reconstructed), and skip post_process_stack below: it mutates each
        # element's own metadata in place on the *original* construction (e.g. distributing a
        # shared tracefile's per-image values), so already-built elements carried over into a
        # slice/filter/average already have correct metadata -- there's nothing left to
        # post-process, and re-running it would re-read the reader's source data (e.g. the
        # full, now-mismatched-length tracefile) against a differently-sized stack.
        built_from_objects = False
        try:
            if not isinstance(source, str) and isinstance(source[0], LEEMImg):
                built_from_objects = True
                if reader is None:
                    reader = source[0]._reader
        except (TypeError, IndexError):
            pass
        self._reader = resolve_reader(source, reader)
        # initialize with np.nan for the first calls to self._single_construct
        # An already-built TimeOrigin (e.g. passed in by __getitem__ when slicing, or by
        # to_lazy()) is reused by reference rather than rewrapped, so the new stack keeps
        # sharing the same mutable origin as its source stack -- mirrors LEEMImg.__init__.
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

    def _single_construct(self, source: Any) -> LEEMImg:
        """Construct a single DataObject."""
        return LEEMImg(source, reader=self._reader, time_origin=self._time_origin)

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
        return type(self)(
            elements,
            virtual=self.virtual,
            time_origin=self._time_origin,
            reader=self._reader,
        )

    def extend(self, other: Union[LEEMImg, LEEMStack]) -> None:
        """Add new LEEMImg(s) to the end of the stack.

        Unlike a plain `DataObjectStack.extend`, this reconciles `time_origin`: `other` may
        come from an independently-constructed stack (e.g. the two halves of a measurement
        that was split and resumed), which by default has its own time origin (its own first
        frame's timestamp) rather than this stack's. Appending it as-is would silently leave
        part of the merged stack's `rel_time` computed against a different, unrelated origin
        -- so a mismatch is warned about, and (for non-virtual stacks) the newly appended
        elements are rebound to this stack's `_time_origin`, restoring one shared origin
        across the whole stack. `timestamp` itself is an absolute instrument-clock value and
        is never touched -- only which origin `rel_time` is measured from.
        """
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
        if not self.virtual:
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
        Image registration of the stack

        By default (``template=None``) the images of the stack are registered
        sequentially: each image is aligned to its predecessor by calling
        ``Image.find_warp_matrix()``, and the resulting shifts are accumulated so
        every warp matrix is relative to the first image. Alternatively, passing
        ``template`` aligns every image directly against that one fixed image
        instead -- useful when drift between neighboring frames is small relative to
        noise but the whole stack drifts relative to a known reference.

        After initial registration, the recovered translation shifts (dx, dy) are
        checked for sanity using a robust (median/MAD-based) outlier score (see
        `_modified_zscore`) -- unlike a classic mean/std z-score, this does not get
        skewed (and so does not miss outliers) when one shift is extremely far off.
        Outliers, together with any frame where registration raised an exception,
        are replaced by interpolating (and, for boundary
        frames, extrapolating) over the remaining valid shifts -- this is the
        expected, routine path (not an error), so it is only printed, never raised
        as a warning; the spline order automatically adapts to the number of valid
        frames so short stacks don't crash. If too few valid frames remain to
        interpolate at all, this is the unexpected, unrecoverable case: the affected
        frames fall back to identity (no shift) and a ``UserWarning`` names them and
        the fallback used. Finally the resulting warp matrices are applied via
        ``Image.warp()``.

        Parameters
        ----------
        inplace : bool, default False
            If 'True' the stack itself will be registered
            If 'False' a copy of the stack will be registered

            .. note:: With 'False' (the default) the whole stack is deep-copied up
                front, so peak memory is briefly ~2x the stack's size. For a stack
                close to the memory limit, pass 'inplace=True' (registering the stack
                in place) instead.
        mask : bool, np.ndarray or None, default True
            Mask that will be used during registration

            - If 'True' a circular mask with a radius=0.9*Image.width is used as mask
            - If 'False' or 'None' no mask is used during registration
            - If np.ndarray the array must be of dtype=np.uint8 and contain the mask that should be
              applied
        template : int, Image or None, default None
            If given, every image is aligned directly against this fixed reference
            image instead of sequentially against its predecessor. An int is used as
            an index into 'stack' (e.g. 'template=0' aligns against the first
            image); an 'Image' is used directly. Template alignment is less
            battle-tested than the default sequential alignment, so a caution is
            raised -- sanity-check the result.
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
            # Every image (including the template's own frame) is aligned directly
            # and independently against the template, so each warp matrix is already
            # absolute -- unlike sequential mode, these must NOT be cumulatively
            # composed further down.
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
            # if an image is multiple times in a stack and inplace is True, it should not be
            # normalized twice, so if the mcp is already the current mcp, it will not be
            # processed. img.mcp holds the *source* of the mcp it was last normalized
            # against (see LEEMImg.normalize), so compare against mcp.source rather than
            # the Image itself -- comparing to the Image would compare a str to an Image.
            if img.mcp is None or img.mcp != mcp.source:
                stack[index] = img.normalize(mcp=mcp, **kwargs)

        return stack

    def to_lazy(self, chunks: int = 1) -> LazyLEEMStack:
        """Rebuild this stack as an out-of-core `LazyLEEMStack` over the same per-element
        sources. See `ImageStack.to_lazy`."""
        sources = self.sources
        if None in sources:
            raise ValueError("Can't build a lazy stack without source information.")
        return LazyLEEMStack(
            sources, reader=self._reader, time_origin=self._time_origin, chunks=chunks
        )


class LazyLEEMStack(LazyImageStack):
    """
    Out-of-core counterpart to `LEEMStack`, backed by a lazy `dask.array` of pixel data
    instead of a fully in-memory list of `LEEMImg`s. See `uspy.dataobject.lazy.LazyImageStack`
    for the general design (metadata kept resident, pixels lazy/chunked).

    Only a subset of `LEEMStack`'s API is supported: construction from a `.dat` folder/glob or
    a 3D TIFF, indexing/slicing, per-frame metadata access, `average()`, elementwise arithmetic
    (+ - * /), and `.normalize()`. `.align()` is intentionally NOT provided here -- ECC
    registration needs pairwise access to full-resolution frames and doesn't parallelize onto
    dask cleanly; call `.materialize()` (or build a `LEEMStack` over a smaller slice) and use
    the regular `.align()` there.

    Requires the optional 'dask'/'zarr' dependencies: ``pip install uspy[bigdata]``.
    """

    _type = LEEMImg
    _materialized_type = LEEMStack

    def __init__(
        self,
        source: Union[str, Iterable],
        reader: Union[str, LEEMReader, None] = None,
        time_origin: Optional[Union[TimeOrigin, Number]] = None,
        chunks: int = 1,
        lazy: bool = True,  # accepted and ignored, see LazyImageStack.__init__
    ) -> None:
        self._reader = resolve_reader(source, reader)
        # Reuse an already-built TimeOrigin by reference (e.g. passed in by
        # LEEMStack.to_lazy()) instead of rewrapping it, so the lazy stack keeps sharing
        # the same mutable origin as its source stack -- mirrors LEEMImg.__init__.
        if isinstance(time_origin, TimeOrigin):
            self._time_origin = time_origin
        else:
            self._time_origin = TimeOrigin(time_origin)
        super().__init__(source, chunks=chunks)
        # Mirrors LEEMStack.__init__: readers that need a whole-stack pass to fill in
        # metadata only known once every frame exists (e.g. ElettraReader distributing a
        # shared tracefile's per-frame columns) get that chance before time_origin is seeded.
        self._reader.post_process_stack(self)
        if time_origin is None:
            self._time_origin.value = self._meta_list[0].get("timestamp", np.nan)

    def _build(self, source: Union[str, Iterable], chunks: int) -> tuple:
        # Mirrors LEEMStack._split_source: try the reader's own file discovery first (a
        # folder, glob, or explicit list of per-frame files -- including ".tif"/".tiff",
        # e.g. Elettra's per-frame TIFFs with a shared tracefile). Only a source the reader
        # doesn't recognize *and* that is itself an actual single 3D TIFF file falls back to
        # being read as one multi-frame TIFF stack -- previously any ".tif"/".tiff"-suffixed
        # string (a bare glob pattern, or a folder holding both .dat and .tif per frame) was
        # unconditionally routed here, which either crashed or silently loaded the wrong
        # (whole-file, single-frame-metadata) data instead of raising or reading per-frame.
        fnames = self._reader.list_files(source)
        if fnames:
            return self._build_from_files(fnames, chunks)
        if isinstance(source, list) and source:
            return self._build_from_files(source, chunks)
        if (
            isinstance(source, str)
            and source.lower().endswith((".tif", ".tiff"))
            and os.path.isfile(source)
        ):
            return self._build_from_tiff_stack(source, chunks)
        raise ValueError(f"No files were found at path {source}")

    def _read_pixels(self, fname: str) -> np.ndarray:
        return np.asarray(self._reader.parse_image(fname)["image"], dtype=np.float32)

    def _read_metadata(self, fname: str) -> dict:
        """Per-frame metadata dict, built the same way a real `LEEMImg(fname, reader=...)`
        would populate its `_meta`/`_units` (see `LEEMImg.__init__`/`DataObject.__init__`):
        class-level `_meta_defaults`/`_unit_defaults`, gap-filled by the reader's own
        `meta_defaults`/`unit_defaults`, then overwritten by the reader's actually-parsed
        values, with the same `temperature > 3000 -> nan` sentinel rule `LEEMImg.parse`
        applies. Without this, defaulted keys (e.g. 'mcp', 'photon_energy') would raise
        AttributeError on a lazy stack instead of returning their eager default."""
        meta = dict(self._type._meta_defaults)  # pylint: disable=protected-access
        units = dict(self._type._unit_defaults)  # pylint: disable=protected-access
        for key, value in self._reader.meta_defaults.items():
            meta.setdefault(key, value)
        for key, value in self._reader.unit_defaults.items():
            units.setdefault(key, value)

        parsed = dict(self._reader.parse_metadata(fname))
        if parsed.get("temperature", 0) > 3000:
            parsed["temperature"] = np.nan
        for key, value in parsed.items():
            if key.startswith("_"):
                continue
            if "_unit" in key:
                units[key.replace("_unit", "")] = value
            else:
                meta[key] = value

        result = meta
        for key, value in units.items():
            result[f"{key}_unit"] = value
        return result

    def _make_element(self, pixels: np.ndarray, meta: dict, index: int) -> LEEMImg:
        element = LEEMImg(pixels, reader=self._reader, time_origin=self._time_origin)
        self._apply_meta(element, meta)
        if self._sources is not None:
            element._source = self._sources[index]
        return element

    def _wrap(self, dask_array, meta_list: list, sources) -> "LazyLEEMStack":
        new = super()._wrap(dask_array, meta_list, sources)
        new._reader = self._reader
        new._time_origin = self._time_origin
        return new

    @property
    def time_origin(self) -> Number:
        return self._time_origin.value

    @time_origin.setter
    def time_origin(self, value: Number) -> None:
        self._time_origin.value = value

    def normalize(
        self, mcp: Union[Image, str, None] = None, dark_counts: Union[int, float, Image] = 0
    ) -> LazyLEEMStack:
        """Lazy counterpart to `LEEMStack.normalize`: computes ``(img - dark_counts) /
        (mcp - dark_counts)`` as a dask expression, so it stays chunked/streamed instead of
        materializing the whole stack. Unlike `LEEMStack.normalize`, `mcp` must be given
        explicitly -- falling back to each frame's stored '.mcp' metadata would require
        resolving it per frame, which isn't meaningfully cheaper than materializing."""
        if mcp is None:
            raise ValueError(
                "LazyLEEMStack.normalize() requires an explicit 'mcp' image (unlike "
                "LEEMStack.normalize, it can't fall back to each frame's stored '.mcp' "
                "metadata)."
            )
        mcp_img = imgify(mcp)
        mcp_arr = np.asarray(mcp_img.image, dtype=np.float32)
        if isinstance(dark_counts, (int, float, complex)):
            dark_value, dark_arr = dark_counts, dark_counts
        else:
            dark_img = imgify(dark_counts)
            dark_value, dark_arr = dark_img, np.asarray(dark_img.image, dtype=np.float32)

        result = (self._dask - dark_arr) / (mcp_arr - dark_arr)
        # Match np.nan_to_num(result, posinf=0, neginf=0) (default nan=0 too) -- the installed
        # dask version's nan_to_num doesn't accept posinf/neginf kwargs.
        result = _da.where(_da.isfinite(result), result, 0).astype(np.float32)

        meta_list = []
        for meta in self._meta_list:
            new_meta = dict(meta)
            new_meta["mcp"] = mcp_img.source
            new_meta["dark_counts"] = dark_value
            meta_list.append(new_meta)
        return self._wrap(result, meta_list, self._sources)


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
#: textbook 3.5): a genuine registration failure lands orders of magnitude beyond the
#: robust scale (score in the tens-hundreds), so a high, well-separated threshold
#: catches real failures while keeping the false-positive rate on clean data low, even
#: for the large, mutually-independent frame-to-frame shifts a stack can legitimately
#: have. Empirically chosen; adjust if real-world data needs more/less sensitivity.
_OUTLIER_ZSCORE_THRESHOLD = 8.0


def _modified_zscore(x: np.ndarray) -> np.ndarray:
    """Robust (median/MAD-based) z-score, used by `_resolve_shifts` to flag
    registration outliers.

    Unlike the classic mean/std z-score, this is not inflated -- and so not
    self-masked -- by the very outliers it's meant to detect: a single huge outlier
    drags the mean and inflates the std enough that both its own z-score and other,
    smaller outliers' z-scores can fall back under the detection threshold
    ("masking"/"swamping"). The median and MAD (median absolute deviation) are far
    less sensitive to extreme values, so this keeps working regardless of an
    outlier's magnitude or how many there are.

    If more than half the values are identical, the MAD itself collapses to 0 (a
    plain division would produce inf/nan); the mean absolute deviation is used as a
    fallback scale in that case, and only if all values are exactly identical does
    this return all zeros (no possible outliers).

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
    """Cleans up raw x/y translation shifts coming out of pairwise/template
    registration, used by `LEEMStack.align()`.

    A failed registration is expected to mark both components of a frame as nan;
    this is unified (rather than asserted) so a partial failure can't crash. When
    'sanity' is set, shifts whose robust (median/MAD-based) z-score (see
    `_modified_zscore`) exceeds `_OUTLIER_ZSCORE_THRESHOLD` are additionally flagged
    as outliers and replaced with nan -- this is the expected, routine path (not an
    error), so it is only printed, never raised as a warning.

    All nan gaps (whether from failed registration or outlier rejection) are then
    filled by interpolation, with the spline order adapted to the number of valid
    (non-nan) frames so that short stacks don't crash pandas' interpolation (which
    needs at least 'order + 1' points). Leading/trailing nan (boundary frames) are
    extrapolated rather than left as nan. If there are too few valid points to
    interpolate/extrapolate at all (0 or 1), this is the unexpected, unrecoverable
    case: the affected frames fall back to identity (zero shift) and a warning names
    the frames and the fallback used.

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
        # A robust z-score needs a handful of points to be meaningful -- skip it on
        # very short input. Unlike the classic mean/std z-score, the degenerate
        # (constant/near-constant) case is already handled inside _modified_zscore
        # itself (falls back to mean-absolute-deviation scaling, or returns all
        # zeros if truly constant), so no separate std > 0 guard is needed here.
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
