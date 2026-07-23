"""
Base class and registry for pluggable LEEM data readers.

A "reader" encapsulates everything that is specific to a data format or endstation: how
to recognize a file, how to parse a single image, how to discover the files that belong
to a stack, and how to fill in metadata that can only be determined once a whole stack is
known (e.g. a beamline "tracefile" shared by all images of a stack). New endstations are
added by writing a small, self-contained `LEEMReader` subclass -- no subclassing of
`LEEMImg`/`LEEMStack` is necessary.

See `uspy.leem.readers.resolve_reader` for how a reader is selected for a given source.
"""
# pylint: disable=missing-docstring

from __future__ import annotations
from typing import Any, Iterable, Union


class LEEMReader:
    """
    Base class for pluggable LEEM data readers.

    Subclasses must set a unique `name`; that automatically registers the subclass so it
    can be selected by that name (via the `reader=` argument of `LEEMImg`/`LEEMStack`, or a
    `uspy.toml` config file placed in the data folder -- see `resolve_reader`).

    Parameters
    ----------
    **options
        Reader-specific options (e.g. from a config file's `[uspy.options]` table). Stored
        on `self.options`; subclasses that need options should read them from there.
    """

    name: str = None
    extensions: tuple = ()
    meta_defaults: dict = {}
    unit_defaults: dict = {}

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
