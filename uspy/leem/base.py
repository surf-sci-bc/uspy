"""
Basic classes for LEEM data parsing and data visualization.

Parsing of specific file formats/endstations is delegated to a pluggable "reader" (see
uspy.leem.readers) -- the Elmitec UKSOFT2001 ".dat" format is the default.
"""

# pylint: disable=missing-docstring
# pylint: disable=attribute-defined-outside-init
# plyint: disable=access-member-before-definition

from __future__ import annotations
from typing import Any, Callable, Union, Optional
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from numbers import Number
from datetime import datetime
import copy
import hashlib
import os
import warnings
from tqdm.auto import tqdm
import pandas as pd

import numpy as np
import cv2 as cv
from skimage.registration import phase_cross_correlation

import tifffile
from tifffile.tifffile import TiffFileError

from uspy.dataobject import Image, ImageStack
from uspy.dataobject.image import list_image_files
from uspy.dataobject.deferred import DeferredArray, ReaderPixels, is_deferred
from uspy.derived import StitchedLine
from uspy.leem.utility import imgify
from uspy.roi import ROI
from uspy.leem.readers import (
    LEEMReader,
    resolve_reader,
    resolve_plotting_fields,
    resolve_default_mask,
)
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


_IN_MEMORY = "<in-memory:"


def _identify(img: Image) -> Any:
    """Identifier of an mcp/dark-counts image: its `source`, or -- for a source-less
    (in-memory) image -- a content digest, so two different in-memory images never compare
    equal and the same one always does."""
    if img.source is not None:
        return img.source
    digest = hashlib.sha1(np.ascontiguousarray(img.image).tobytes()).hexdigest()
    return f"{_IN_MEMORY}{digest[:16]}>"


def _dark_id(dark_counts: Union[Number, Image]) -> Any:
    """Comparable identifier of a `dark_counts` value (number, filename or image)."""
    if isinstance(dark_counts, Number):
        return float(dark_counts)
    if isinstance(dark_counts, (str, os.PathLike)):
        return dark_counts  # a filename identifies itself
    return _identify(dark_counts)


def _is_in_memory_id(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(_IN_MEMORY)


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
        # True once `normalize` ran; `mcp`/`dark_counts` then record what it used. Before that,
        # `mcp` is merely the filename `normalize(mcp=None)` would use.
        "normalized": False,
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
        # The instrument's default mask (see the `default_mask` property below), resolved
        # to an unresolved spec/ROI/None here (cheap -- reuses the same cached uspy.toml
        # search-path walk as plotting_fields above) but not yet converted to a `ROI`: that
        # conversion needs `self.shape`, which for a deferred element could mean a file-header
        # read, so it stays lazy until `default_mask` is actually asked for.
        self._mask_spec = resolve_default_mask(source, self._reader)
        self._default_mask_cache = None  # (shape, ROI) once resolved -- see the property

    @property
    def default_mask(self) -> Optional[ROI]:
        """The instrument's default mask (e.g. the channelplate aperture), used by
        `LEEMStack.align()`/`uspy.plotting.plot_img()`/`uspy.plotting.make_video()` when
        called with `mask=True`. Resolved at construction time from (most specific first) a
        session override (`uspy.leem.readers.set_default_mask`), a `uspy.toml` `[uspy.mask]`
        table, or the reader's own `default_mask` -- see
        `uspy.leem.readers.resolve_default_mask`. `None` if no mask applies.

        A spec dict is converted to a `ROI` lazily, on first access, and cached per
        `self.shape` (re-resolved if `self.shape` changes, e.g. after `_with_data` produces
        a differently-shaped copy) -- this avoids resolving `self.shape` (a potential
        file-header read for a deferred element) until a mask is actually asked for.
        """
        if self._mask_spec is None or isinstance(self._mask_spec, ROI):
            return self._mask_spec
        shape = self.shape
        if self._default_mask_cache is not None and self._default_mask_cache[0] == shape:
            return self._default_mask_cache[1]
        roi = ROI.from_spec(self._mask_spec, shape=shape)
        self._default_mask_cache = (shape, roi)
        return roi

    @default_mask.setter
    def default_mask(self, value: Union[ROI, dict, None]) -> None:
        self._mask_spec = value
        self._default_mask_cache = None

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

        # The pixels are warped again on top of any earlier warp, so the recorded matrix must
        # compose (dst(x) = src(M x): total = earlier @ new) for `unwarped` to stay correct.
        total = self.warp_matrix @ warp_matrix
        if inplace:
            self._data["image"] = image  # pylint: disable=protected-access
            self.warp_matrix = total
            return self

        result = self.copy()
        result._data["image"] = image  # pylint: disable=protected-access
        result.warp_matrix = total
        return result

    def normalize(
        self,
        mcp: Image = None,
        dark_counts: Union[int, float, Image] = 0,
        inplace: bool = False,
    ) -> LEEMImg:
        """Normalization of LEEM Images: ``(mcp - dark) / ...`` -> ``(img - dark) / (mcp - dark)``.

        Parameters
        ----------
        mcp : Image or str, optional
            MCP (channelplate) image or filename. If None, the image's own ``mcp`` attribute
            (a filename) is used.
        dark_counts : int, float or Image, default 0
            Dark counts, a constant or an image.
        inplace : bool, default False
            Normalize this image itself instead of returning a normalized copy.

        Returns
        -------
        LEEMImg
            Normalized image; ``mcp`` and ``dark_counts`` record what it was normalized with
            (a source-less MCP is recorded as a content digest).

        Raises
        ------
        ValueError
            If ``mcp`` is None and the image has no usable ``mcp`` attribute.
        """
        if mcp is None:
            mcp = self.mcp  # pylint: disable=access-member-before-definition
            if mcp is None or _is_in_memory_id(mcp):
                raise ValueError(
                    "normalize(): no mcp given and the image has no 'mcp' filename to use"
                )
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
            self.mcp = _identify(mcp)
            self.dark_counts = dark_counts
            self.normalized = True
            return self

        result.mcp = _identify(mcp)
        result.dark_counts = dark_counts
        result.normalized = True
        return result

    def find_warp_matrix(self, template: Image, algorithm="ecc", **kwargs) -> np.ndarray:
        """Calculates a warp matrix between the image and a template.

        The algorithm applied is specified by the *algorithm* argument, which is *ecc* by
        default. Both algorithms accept *mask* (an `ROI` or an ndarray); only *ecc* accepts
        transformations other than *translation* (via the *trafo* kwarg).

        - "ecc" (`do_ecc_align`): OpenCV's enhanced correlation coefficient. Fast and
          accurate, but assumes a positive-gain relationship between image intensities and
          fails outright if contrast inverts between *self* and *template* (as LEEM images
          can across a wide energy sweep).
        - "phasecorr" (`do_phasecorr_align`): FFT phase correlation
          (`skimage.registration.phase_cross_correlation`). Translation only. Structurally
          invariant to contrast inversion. Can lock onto the wrong period for images
          dominated by repeating structure (e.g. a wide-FOV terrace array) -- prefer a tight
          *mask* in that regime.

        Parameters
        ----------
        template : Image
            Template image against which is aligned
        algorithm : str, optional
            algorithm used for registration: "ecc" (default) or "phasecorr"

        **kwargs
            Additional keyword arguments are passed through to the registration function. Can specifiy further options like convergence criteria or transformations

        Returns
        -------
        np.ndarray
            3x3 matrix containing the alignment parameters

        Raises
        ------
        ValueError
            Raised when an unknown *algorithm*, or an unsupported *trafo* for the chosen
            *algorithm*, is given.

        See Also
        --------
        do_ecc_align
        do_phasecorr_align
        """
        fn = _resolve_algorithm(algorithm, kwargs.get("trafo", "translation"))
        return fn(self, template, **kwargs)

    def parse(self, source: str) -> dict[str, Any]:
        if isinstance(source, Image):
            self._source = None
            return copy.deepcopy(dict(source.meta, **source.data))
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
        dark = self.dark_counts
        if self.normalized and (
            _is_in_memory_id(self.mcp) or (isinstance(dark, Image) and dark.source is None)
        ):
            raise ValueError(
                "Can't JSON-serialize an image normalized with an in-memory (source-less) "
                "mcp/dark_counts image: the normalization can't be reproduced from a source."
            )
        return {
            "source": self.source,
            "mcp": self.mcp,
            "warp_matrix": self.warp_matrix,
            "dark_counts": self.dark_counts,
            "normalized": self.normalized,
            "reader": self._reader.name,
        }

    def __json_decode__(self, **attrs) -> None:
        self.__init__(source=attrs["source"], reader=attrs.get("reader"))
        # JSON written before the `normalized` flag existed: a set `mcp` meant normalized.
        if attrs.get("normalized", bool(attrs["mcp"])):
            self.normalize(
                mcp=attrs["mcp"], dark_counts=attrs["dark_counts"], inplace=True
            )
        else:
            self.mcp = attrs["mcp"]
            self.dark_counts = attrs["dark_counts"]
        # Normalization first, then warp: the order `align` is meant to be used in.
        if not np.array_equal(attrs["warp_matrix"], np.eye(3)):
            self.warp(attrs["warp_matrix"], inplace=True)

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
            if time_origin is None:
                # now we can access the 0th element and set the value
                self._time_origin.value = self[0].timestamp
            return
        if time_origin is None:
            # Elements already sharing one origin (a slice, a re-spun stack) keep it;
            # otherwise the stack starts at its first frame.
            shared = {id(e._time_origin): e._time_origin for e in self._elements}
            if len(shared) == 1:
                self._time_origin = next(iter(shared.values()))
            else:
                self._time_origin.value = self[0].timestamp
        self._elements = [self._with_origin(e) for e in self._elements]

    def _with_origin(self, element: LEEMImg) -> LEEMImg:
        """`element` itself if it already uses this stack's `TimeOrigin`, else a copy that does
        (the caller's / another stack's element is never rebound in place)."""
        if element._time_origin is self._time_origin:
            return element
        rebound = element._with_data()
        rebound._time_origin = self._time_origin
        return rebound

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

    def extend(self, other: Union[LEEMImg, LEEMStack]) -> None:
        """Add new LEEMImg(s) to the end of the stack, reconciling `time_origin`: if `other`
        has a different one (e.g. two halves of a split/resumed measurement), a warning
        names the mismatch and the appended elements are replaced by copies bound to this
        stack's `_time_origin` (`other` itself is left untouched) so `rel_time` shares one
        origin. `timestamp` itself is never touched."""
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
        # Copies, so `other`'s own elements keep their origin.
        self._elements[n_before:] = [
            self._with_origin(element) for element in self._elements[n_before:]
        ]

    def align(
        self,
        inplace: bool = False,
        mask: Union[bool, np.ndarray, ROI, None] = True,
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
        mask : bool, np.ndarray, ROI or None, default True
            Mask that will be used during registration

            - If 'True', 'stack[0].default_mask' is used -- the instrument's default mask
              (e.g. the channelplate aperture; a radius=0.9*Image.width circle unless
              overridden via a 'uspy.toml' '[uspy.mask]' table or
              'uspy.leem.readers.set_default_mask', see 'LEEMImg.default_mask')
            - If 'False' or 'None' no mask is used during registration
            - If an 'ROI' (e.g. 'ROI.circle(x0, y0, radius=...)'), it is converted internally
              (zero pixel I/O -- see 'ROI.pad_to')
            - If np.ndarray, any 2D array matching the stack's frame shape; cast to uint8, so
              a boolean array (e.g. 'roi.array') works too
        template : int, Image or None, default None
            If given, every image is aligned directly against this fixed reference image
            instead of sequentially against its predecessor. An int indexes 'stack'
            ('template=0' = first image); an 'Image' is used directly. Less
            battle-tested than sequential alignment -- sanity-check the result.
        sanity : bool, default True
            Whether to run the z-score outlier check described above.

        **kwargs
            Additional keyword arguments are passed through to Image.find_warp_matrix --
            notably 'algorithm' ("ecc" default or "phasecorr"; see 'find_warp_matrix'). For
            an energy sweep with strong contrast inversion, prefer 'algorithm="phasecorr"'
            over plain "ecc".

        Returns
        -------
        LEEMStack
            LEEMStack with aligned images. The stack itself is returned when 'inplace' = True

        Raises
        ------
        ValueError
            If the stack is empty, or 'algorithm'/'trafo' is not recognized or not supported
            together.
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

        # Resolve up front (rather than letting a bad 'algorithm'/'trafo' combination be
        # caught, per frame, by the broad `except Exception` below) so a typo surfaces as an
        # immediate error instead of silently degrading every frame to an interpolated guess.
        _resolve_algorithm(kwargs.get("algorithm", "ecc"), kwargs.get("trafo", "translation"))

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

        # `isinstance` first, rather than testing `mask` for truthiness directly: an ndarray
        # mask's truth value is ambiguous (raises ValueError) once it has more than one
        # element, so only the bool-or-None `True`/`False`/`None` shorthand is ever tested
        # that way.
        if isinstance(mask, bool):
            mask = stack[0].default_mask if mask else None
        mask = _resolve_mask(mask, (stack[0].height, stack[0].width))

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

        The images in the stack are normalized by appling Image.normalize(mcp) to each image.
        Each frame records the ``mcp`` and ``dark_counts`` it was normalized with.

        Parameters
        ----------
        mcp : Image, str or None
            Image or filename of image if str. If 'None' each image's own ``mcp`` attribute
            (a filename, e.g. set via ``stack.mcp = ...``) is used; no repeat check is made
            then, so calling it twice divides twice.
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

        Raises
        ------
        ValueError
            If a frame was already normalized with a different ``mcp`` or ``dark_counts``
            (checked before any frame is touched), or ``mcp`` is None and a frame has no
            ``mcp`` attribute.

        Warns
        -----
        UserWarning
            Frames already normalized with identical ``mcp`` and ``dark_counts`` are skipped.

        See Also
        --------
        LEEMImg.normalize

        """
        done = [False] * len(self)
        if mcp is None:
            if any(img.mcp is None or _is_in_memory_id(img.mcp) for img in self):
                raise ValueError(
                    "normalize(): no mcp given and not every image has an 'mcp' filename"
                )
        else:
            # if not a copy funny things might happen when mcp is part of stack
            mcp = imgify(mcp).copy()
            key = (_identify(mcp), _dark_id(kwargs.get("dark_counts", 0)))
            done = [img.normalized for img in self]
            for img, normalized in zip(self, done):
                if normalized and (img.mcp, _dark_id(img.dark_counts)) != key:
                    raise ValueError(
                        f"Image already normalized with mcp={img.mcp!r}, "
                        f"dark_counts={img.dark_counts!r}; refusing to normalize again with "
                        f"mcp={key[0]!r}, dark_counts={key[1]!r}."
                    )
            if any(done):
                warnings.warn(
                    UserWarning(
                        f"{sum(done)} image(s) already normalized with the same mcp and "
                        "dark_counts -- skipped."
                    )
                )
        stack = self if inplace else self.copy()

        for index, img in enumerate(tqdm(stack)):
            if not done[index]:
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


def _as_array(img: Union[Image, np.ndarray]) -> np.ndarray:
    """`img.image` for an `Image`, or `img` itself coerced via `np.asarray` -- lets every
    registration backend (`do_ecc_align`/`do_phasecorr_align`) accept either an `Image`/
    `LEEMImg` or a raw ndarray."""
    return img.image if isinstance(img, Image) else np.asarray(img)


def _resolve_mask(
    mask: Union[np.ndarray, ROI, None], shape: tuple[int, int]
) -> Optional[np.ndarray]:
    """Normalizes a registration `mask` argument to a uint8 ndarray or `None`. Accepts `None`,
    an `ROI` (converted via `ROI.pad_to` -- zero pixel I/O, unlike `roi.apply(img).image`), or
    an ndarray of any dtype (cast to uint8, so a boolean array -- e.g. `roi.array` -- also just
    works). `shape` is the target `(height, width)`, matching `Image.image.shape` (note
    `ROI.pad_to`'s own parameter names are, confusingly, `(width, height)` but are actually
    consumed in `(rows, cols)` = `(height, width)` order -- passing `*shape` here sidesteps
    that directly rather than re-deriving it at each call site)."""
    if mask is None:
        return None
    if isinstance(mask, ROI):
        return mask.pad_to(*shape).astype(np.uint8)
    return np.asarray(mask).astype(np.uint8)


def do_ecc_align(
    input_img: Union[Image, np.ndarray],
    template_img: Union[Image, np.ndarray],
    max_iter: int = 500,
    eps: float = 1e-4,
    trafo: str = "translation",
    mask: Union[np.ndarray, ROI, None] = None,
) -> np.ndarray:
    """Finds warp matrix between two Images using ECC

    Takes two images and calculates the warp matrix between both applying the transformation
    specified by 'trafo' using the ECC Algorithm. A optional mask can be given to masked areas of
    the images that are not taken into account during registration

    ECC assumes a positive-gain relationship between the two images' intensities and fails
    outright (raises `cv2.error`) if contrast has inverted between them -- as LEEM images
    routinely do across a wide energy sweep. For that case, use 'algorithm="phasecorr"' on
    `LEEMImg.find_warp_matrix`/`LEEMStack.align` instead.

    Parameters
    ----------
    input_img : Image or np.ndarray
        Image that will be warped to match the tempalate image
    template_img : Image or np.ndarray
        The template image
    max_iter : int, optional
        Number of iterations to find the warp matrix. Regstration fails if max_iter is
        exceeded, by default 500
    eps : float, optional
        Abortion criterion to determine successfull registration, by default 1e-4
    trafo : str, optional
        The transformation that is applied to the images. 'trafo' must be either

        * "translation" for only x,y shifting of images
        * "rigid" or "euclidean" for translation and rotation
        * "affine" for translation, rotation, scaling and shearing

        , by default "translation"
    mask : np.ndarray, ROI or None, optional
        Pixels excluded from registration. An `ROI` (e.g. `ROI.circle(x0, y0, radius=...)`) is
        converted internally; an ndarray must be 2D and is cast to uint8, with '0' where
        pixels are excluded and any nonzero value where they are taken into account. By
        default 'None' (no mask).

    Returns
    -------
    np.ndarray
        3x3 numpy array representing the warp matrix

    See Also
    --------
    do_phasecorr_align
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

    template_pixels = _as_array(template_img)
    input_pixels = _as_array(input_img)
    mask = _resolve_mask(mask, template_pixels.shape)

    template_arr = cv.normalize(
        template_pixels,
        None,
        0,
        255,
        cv.NORM_MINMAX,
        -1,
        mask,
    )
    input_arr = cv.normalize(
        input_pixels,
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


def do_phasecorr_align(
    input_img: Union[Image, np.ndarray],
    template_img: Union[Image, np.ndarray],
    upsample_factor: int = 20,
    mask: Union[np.ndarray, ROI, None] = None,
    trafo: str = "translation",
    normalization: Optional[str] = "phase",
    mask_feather: float = 15,
) -> np.ndarray:
    """Finds a translation warp matrix between two images using FFT phase correlation.

    Wraps `skimage.registration.phase_cross_correlation`. Structurally invariant to a
    contrast inversion between 'input_img' and 'template_img' (negating one image negates
    the whole cross-power spectrum; the peak is located on its magnitude), unlike
    `do_ecc_align`, which fails outright in that case.

    Being a single global FFT rather than a local iterative optimization, phase correlation
    can lock onto the wrong period for images dominated by repeating structure (e.g. a
    wide-FOV terrace array) instead of the true shift -- prefer a tight 'mask' if that is a
    concern.

    Parameters
    ----------
    input_img : Image or np.ndarray
        Image that will be warped to match the template image
    template_img : Image or np.ndarray
        The template image
    upsample_factor : int, optional
        Registers to within '1/upsample_factor' of a pixel, by default 20
    mask : np.ndarray, ROI or None, optional
        Pixels excluded from registration. An `ROI` (e.g. `ROI.circle(x0, y0, radius=...)`) is
        converted internally; an ndarray must be 2D and is cast to uint8, with '0' where
        pixels are excluded and any nonzero value where they are taken into account. By
        default 'None' (no mask). Applied by mean-subtracting each image within the mask,
        then multiplying by a Gaussian-feathered ('mask_feather' px sigma) version of the
        mask, rather than zeroing outside it sharply: a hard 0/1 edge is bit-identical
        between both images and is otherwise itself a strong feature that a global FFT
        correlation reliably locks onto at zero shift, regardless of the images' actual
        content (this is not a hypothetical edge case -- it reproducibly beat the real
        registration signal on real LEEM fixtures during development). skimage's
        masked-normalized-correlation path is not used instead: that mode picks the *signed*
        correlation maximum rather than its magnitude, so it is not contrast-inversion
        invariant.
    trafo : str, optional
        Must be "translation" -- phase correlation cannot express rotation/scale/shear; by
        default "translation"
    normalization : str or None, optional
        Passed to `phase_cross_correlation`; by default "phase"
    mask_feather : float, optional
        Gaussian sigma, in pixels, used to feather 'mask''s edge -- see 'mask' above. By
        default 15; the exact value is not sensitive (verified over sigma 3-30 px on real
        data) since its only job is breaking the hard edge's bit-for-bit self-similarity.

    Returns
    -------
    np.ndarray
        3x3 numpy array representing the warp matrix

    Raises
    ------
    ValueError
        If 'trafo' is not "translation".

    See Also
    --------
    do_ecc_align
    """
    if trafo != "translation":
        raise ValueError(
            f"do_phasecorr_align only supports trafo='translation', got {trafo!r}. "
            "Use algorithm='ecc' for rotation/scale/shear."
        )

    template_arr = _as_array(template_img).astype(np.float64)
    input_arr = _as_array(input_img).astype(np.float64)
    mask = _resolve_mask(mask, template_arr.shape)

    if mask is not None:
        valid = mask > 0
        taper = np.clip(cv.GaussianBlur(mask.astype(np.float32), (0, 0), mask_feather), 0, 1)
        template_arr = (template_arr - template_arr[valid].mean()) * taper
        input_arr = (input_arr - input_arr[valid].mean()) * taper

    # template = warp_matrix * input (see do_ecc_align) <-> input(u) = template(u - shift),
    # i.e. warp_matrix's translation is the negative of the (row, col) shift skimage returns.
    shift, _error, _phasediff = phase_cross_correlation(
        template_arr,
        input_arr,
        upsample_factor=upsample_factor,
        normalization=normalization,
    )
    warp_matrix = np.eye(3, dtype=np.float32)
    warp_matrix[0, 2] = -shift[1]
    warp_matrix[1, 2] = -shift[0]
    return warp_matrix


@dataclass(frozen=True)
class _Algorithm:
    """One registration backend for `find_warp_matrix`/`LEEMStack.align`: 'fn' is
    `do_ecc_align`/`do_phasecorr_align`, 'trafos' the set of 'trafo' values it can express."""

    fn: Callable[..., np.ndarray]
    trafos: tuple[str, ...]


#: Registry of `find_warp_matrix`'s `algorithm` option.
ALIGNMENT_ALGORITHMS: dict[str, _Algorithm] = {
    "ecc": _Algorithm(do_ecc_align, ("translation", "rigid", "euclidean", "affine")),
    "phasecorr": _Algorithm(do_phasecorr_align, ("translation",)),
}


def _resolve_algorithm(algorithm: str, trafo: str) -> Callable[..., np.ndarray]:
    """Looks up 'algorithm' in `ALIGNMENT_ALGORITHMS` and checks it supports 'trafo', raising
    `ValueError` immediately on either mismatch -- used by both `find_warp_matrix` (per call)
    and `LEEMStack.align` (once, up front, so a bad combination doesn't get caught by the
    per-frame `except Exception` there and silently degraded to an interpolated guess)."""
    try:
        entry = ALIGNMENT_ALGORITHMS[algorithm]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Algorithm {algorithm!r}. Must be one of {tuple(ALIGNMENT_ALGORITHMS)}."
        ) from exc
    if trafo not in entry.trafos:
        raise ValueError(
            f"algorithm={algorithm!r} does not support trafo={trafo!r}; supported: "
            f"{entry.trafos}."
        )
    return entry.fn


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
