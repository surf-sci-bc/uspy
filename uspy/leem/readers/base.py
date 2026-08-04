"""
Base class and registry for pluggable LEEM data readers.

A reader encapsulates everything specific to a data format/endstation: recognizing a
file, parsing a single image, discovering a stack's files, and filling in metadata only
known once a whole stack exists. New endstations are added by writing a `LEEMReader`
subclass -- no subclassing of `LEEMImg`/`LEEMStack` needed.

See `uspy.leem.readers.resolve_reader` for how a reader is selected for a given source.
"""
# pylint: disable=missing-docstring

from __future__ import annotations
from typing import Any, Iterable, Optional, Union


class LEEMReader:
    """Base class for pluggable LEEM data readers. Subclasses must set a unique `name`,
    which auto-registers them for selection via `reader=`, `set_reader(...)`, or a
    `uspy.toml` file (see `resolve_reader`).

    Parameters
    ----------
    **options
        Reader-specific options (e.g. from a config file's `[uspy.options]` table),
        stored on `self.options`.
    """

    name: str = None
    extensions: tuple = ()
    meta_defaults: dict = {}
    unit_defaults: dict = {}
    # Built-in default mask, as a spec dict for `uspy.roi.ROI.from_spec` (see
    # `uspy.leem.readers.resolve_default_mask`) -- overridable per-instrument via a
    # `[uspy.mask]` uspy.toml table. Defaults to the channelplate aperture shared by every
    # built-in reader (all channelplate LEEM instruments); a reader for a different kind of
    # detector should override this, e.g. to `None` to disable by default.
    default_mask: Optional[dict] = {"shape": "circle", "x0": "50%", "y0": "50%", "radius": "45%"}

    _registry: dict = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name is not None:
            LEEMReader._registry[cls.name] = cls

    def __init__(self, **options: Any) -> None:
        self.options = options

    def handles(self, source: Any) -> bool:
        """Return True if this reader can parse `source` (a single-image source)."""
        return isinstance(source, str) and source.endswith(self.extensions)

    def parse_image(self, source: str) -> dict[str, Any]:
        """
        Parse a single file and return the flat dict expected by `DataObject.__init__`
        (see `uspy.dataobject`): data keys (e.g. "image"), metadata keys, and "<key>_unit"
        keys for units. Keys starting with "_" are treated as private and discarded.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot parse single images")

    def parse_metadata(self, source: str) -> dict[str, Any]:
        """Parse just the metadata of a single file source, without necessarily loading pixel
        data -- used to populate a virtual (`defer=True`) `LEEMImg`'s metadata on first touch
        (see `uspy.dataobject.deferred.ReaderMeta`). Default falls back to `parse_image` and
        discards "image"; override for a real speedup."""
        idict = dict(self.parse_image(source))
        idict.pop("image", None)
        return idict

    def parse_pixels(self, source: str) -> Any:
        """Read just the pixel data of a single file source -- used to build a virtual
        (`defer=True`) `LEEMImg`'s deferred `image` handle (see
        `uspy.dataobject.deferred.ReaderPixels`), so this runs on every access and should stay
        cheap. Default falls back to `parse_image` and discards everything else; override for
        a real speedup (e.g. `ElmitecReader.parse_pixels` skips the metadata header walk
        entirely)."""
        return self.parse_image(source)["image"]

    def list_files(self, source: Union[str, Iterable]) -> list:
        """
        Turn a directory/glob-pattern/... into a sorted list of per-image sources
        belonging to a stack. Return an empty list if `source` is not recognized by this
        reader -- the caller then falls back to the generic `ImageStack` source-splitting
        (multi-page TIFF stacks, or an explicit list of sources/objects).
        """
        return []

    def post_process_stack(self, stack) -> None:
        """
        Hook called once, after a stack has been fully constructed (or virtualized), for
        readers that need to fill in metadata that is only known once the whole stack
        exists (e.g. a tracefile shared across all images of a stack). No-op by default.
        """
