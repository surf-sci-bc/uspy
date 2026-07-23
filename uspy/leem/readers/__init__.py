"""
Pluggable LEEM data readers.

`uspy` loads LEEM data through `LEEMImg`/`LEEMStack` (see `uspy.leem.base`), which delegate
all format/endstation-specific parsing to a `LEEMReader`. A reader is chosen for a given
source, in order:

1. An explicit `reader=` argument to `LEEMImg`/`LEEMStack` -- either a `LEEMReader`
   instance, or the registered name of one (e.g. `reader="elettra"`).
2. A `uspy.toml` config file placed in the data folder, e.g.::

       [uspy]
       reader = "elettra"

       [uspy.options]      # optional, passed as keyword arguments to the reader
       energy_offset = 0.0

3. The default `ElmitecReader` (Elmitec UKSOFT2001 ".dat" files).

There is no automatic content-based format detection -- a non-default reader must be
requested explicitly or declared via a `uspy.toml` file. To support a new endstation,
write a `LEEMReader` subclass (see `uspy.leem.readers.base.LEEMReader`) and give it a
unique `name`; it registers itself automatically and can then be selected by that name.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Union

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from uspy.leem.readers.base import LEEMReader
from uspy.leem.readers.elmitec import ElmitecReader
from uspy.leem.readers.elettra import ElettraReader

__all__ = ["LEEMReader", "ElmitecReader", "ElettraReader", "resolve_reader"]

CONFIG_FILENAME = "uspy.toml"


def _config_folder(source: Any) -> Optional[Path]:
    """Best-effort guess at the data folder a `source` argument refers to, so a
    `uspy.toml` placed there can be found."""
    if isinstance(source, (list, tuple)) and source:
        source = source[0]
    if not isinstance(source, str):
        return None
    path = Path(source)
    if path.is_dir():
        return path
    return path.parent


def _load_config(source: Any) -> tuple:
    """Look for a `uspy.toml` in the data folder of `source`. Returns a
    `(reader_name_or_None, options_dict)` tuple."""
    folder = _config_folder(source)
    if folder is None:
        return None, {}
    config_path = folder / CONFIG_FILENAME
    if not config_path.is_file():
        return None, {}
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    section = config.get("uspy", {})
    return section.get("reader"), section.get("options", {})


def resolve_reader(source: Any, reader: Union[str, LEEMReader, None] = None) -> LEEMReader:
    """
    Resolve which `LEEMReader` to use for `source`. See the module docstring for the
    selection order.

    Raises
    ------
    KeyError
        If `reader` (or the reader named in a `uspy.toml`) is not a registered reader name.
    """
    if isinstance(reader, LEEMReader):
        return reader
    if isinstance(reader, str):
        return LEEMReader._registry[reader]()  # pylint: disable=protected-access

    name, options = _load_config(source)
    if name is not None:
        return LEEMReader._registry[name](**options)  # pylint: disable=protected-access

    return ElmitecReader()
