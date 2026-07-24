"""
Pluggable LEEM data readers.

`uspy` loads LEEM data through `LEEMImg`/`LEEMStack` (see `uspy.leem.base`), which delegate
all format/endstation-specific parsing to a `LEEMReader`. A reader is chosen for a given
source, in order:

1. An explicit `reader=` argument to `LEEMImg`/`LEEMStack` -- either a `LEEMReader`
   instance, or the registered name of one (e.g. `reader="elettra"`).
2. A programmatic session-wide override set via `set_reader(...)`, e.g. in a notebook::

       from uspy.leem.readers import set_reader
       set_reader("smart")          # every load for the rest of the session uses this
       set_reader(None)             # or reset_reader() -- clears the override

   Useful for a machine/beamline that only ever produces one file format, without
   needing a `uspy.toml` at all.
3. `uspy.toml` config file(s), e.g.::

       [uspy]
       reader = "elettra"

       [uspy.options]      # optional, passed as keyword arguments to the reader
       energy_offset = 0.0

       [uspy.plotting]     # optional, overrides the (Elmitec-oriented) default
       default_fields = ["temperature", "pressure", "energy", "fov"]  # plot_img/make_video overlay
       meta_fields = ["temperature"]                                  # plot_meta
       print_fields = ["energy", "temperature", "fov"]                # print_meta

   Config files are searched for at multiple locations, most specific first:

   a. the data folder (the folder `source` points into, or `source` itself if it's
      already a folder)
   b. that folder's parent
   c. the current working directory
   d. a global config directory: `$XDG_CONFIG_HOME/uspy/uspy.toml`, or
      `~/.config/uspy/uspy.toml` if `XDG_CONFIG_HOME` is unset

   The `reader`/`[uspy.options]` pair is taken as a unit from the first (most specific)
   file that declares a `reader`. `[uspy.plotting]` fields are merged across all found
   files on a per-key basis, with more specific files winning -- so, for example, a
   global `uspy.toml` can fix the reader for a whole machine while a per-folder
   `uspy.toml` only overrides plotting fields for one measurement.

4. The default `ElmitecReader` (Elmitec UKSOFT2001 ".dat" files).

Ready-to-copy baseline `uspy.toml` files for each built-in reader (with sensible
default plotting fields for that endstation's metadata) live under
`examples/uspy_toml/<reader name>/uspy.toml`.

There is no automatic content-based format detection -- a non-default reader must be
requested explicitly or declared via `set_reader`/a `uspy.toml` file. To support a new
endstation, write a `LEEMReader` subclass (see `uspy.leem.readers.base.LEEMReader`) and
give it a unique `name`; it registers itself automatically and can then be selected by
that name.
"""

from __future__ import annotations
import functools
import os
from pathlib import Path
from typing import Any, Optional, Union

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from uspy.leem.readers.base import LEEMReader
from uspy.leem.readers.elmitec import ElmitecReader
from uspy.leem.readers.elettra import ElettraReader
from uspy.leem.readers.smart import SmartReader

__all__ = [
    "LEEMReader",
    "ElmitecReader",
    "ElettraReader",
    "SmartReader",
    "resolve_reader",
    "resolve_plotting_fields",
    "set_reader",
    "get_reader",
    "reset_reader",
]

CONFIG_FILENAME = "uspy.toml"

# Programmatic session-wide reader override, set via `set_reader()` -- see the module
# docstring. Ranks below an explicit `reader=` argument but above every `uspy.toml`.
_session_reader: dict = {"reader": None, "options": {}}


def set_reader(reader: Union[str, LEEMReader, None], **options: Any) -> None:
    """
    Set a session-wide reader override (e.g. from a notebook), used by `resolve_reader`
    for every subsequent `LEEMImg`/`LEEMStack` load that doesn't pass an explicit
    `reader=` argument -- taking precedence over any `uspy.toml`.

    `reader` may be a registered reader name, a `LEEMReader` instance, or `None` to clear
    the override (equivalent to `reset_reader()`). `**options` are stored alongside a
    name and passed as keyword arguments when the reader is instantiated; they're
    ignored if `reader` is already a `LEEMReader` instance.
    """
    if reader is None:
        reset_reader()
        return
    _session_reader["reader"] = reader
    _session_reader["options"] = options


def get_reader() -> tuple:
    """Returns the current `(reader, options)` session override set via `set_reader`,
    or `(None, {})` if none is set."""
    return _session_reader["reader"], _session_reader["options"]


def reset_reader() -> None:
    """Clears the session-wide reader override set via `set_reader`."""
    _session_reader["reader"] = None
    _session_reader["options"] = {}


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


def _global_config_dir() -> Path:
    """The global config directory: `$XDG_CONFIG_HOME/uspy`, or `~/.config/uspy` if
    `XDG_CONFIG_HOME` is unset -- for a `uspy.toml` that applies machine-/user-wide
    (e.g. a measurement machine that only ever produces one file format)."""
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"
    return base / "uspy"


def _config_search_paths(source: Any) -> list:
    """
    Ordered list of folders to look for a `uspy.toml` in, most specific first (see the
    module docstring): the data folder, its parent, the current working directory, and
    the global config directory. Entries are de-duplicated by resolved path (e.g. when
    the data folder IS the current working directory) while preserving the first,
    most-specific occurrence.
    """
    candidates = []
    data_folder = _config_folder(source)
    if data_folder is not None:
        candidates.append(data_folder)
        candidates.append(data_folder.parent)
    candidates.append(Path.cwd())
    candidates.append(_global_config_dir())

    seen = set()
    paths = []
    for folder in candidates:
        resolved = folder.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(folder)
    return paths


@functools.lru_cache(maxsize=32)
def _parse_toml_cached(config_path_str: str, mtime_ns: int) -> dict:  # pylint: disable=unused-argument
    """Parse a uspy.toml file, cached on its path+mtime so re-instantiating many
    `LEEMImg`s from the same folder (e.g. one per frame of a stack) doesn't re-open
    and re-parse the file each time."""
    with open(config_path_str, "rb") as f:
        return tomllib.load(f)


def _uspy_tables(source: Any) -> list:
    """Returns the `[uspy]` table of every `uspy.toml` found along `source`'s config
    search path (see `_config_search_paths`), most specific first. Folders without a
    `uspy.toml` are silently skipped."""
    tables = []
    for folder in _config_search_paths(source):
        config_path = folder / CONFIG_FILENAME
        if not config_path.is_file():
            continue
        config = _parse_toml_cached(str(config_path), config_path.stat().st_mtime_ns)
        tables.append(config.get("uspy", {}))
    return tables


def _load_config(source: Any) -> tuple:
    """Returns a `(reader_name_or_None, options_dict)` tuple taken as a unit from the
    first (most specific) `uspy.toml` on `source`'s config search path that declares a
    `reader` -- see `_uspy_tables`."""
    for section in _uspy_tables(source):
        if "reader" in section:
            return section["reader"], section.get("options", {})
    return None, {}


def resolve_plotting_fields(source: Any, defaults: dict[str, tuple]) -> dict[str, tuple]:
    """
    Resolve default plotting field-name tuples from the `[uspy.plotting]` tables of
    every `uspy.toml` on `source`'s config search path (see the module docstring),
    merged per-key with more specific files winning, falling back per-key to `defaults`
    if no file sets that key.
    """
    merged: dict = {}
    for section in reversed(_uspy_tables(source)):
        merged.update(section.get("plotting", {}))
    return {
        key: tuple(merged[key]) if key in merged else default
        for key, default in defaults.items()
    }


def resolve_reader(source: Any, reader: Union[str, LEEMReader, None] = None) -> LEEMReader:
    """
    Resolve which `LEEMReader` to use for `source`. See the module docstring for the
    selection order.

    Raises
    ------
    KeyError
        If `reader` (or the reader named by a session override or a `uspy.toml`) is not
        a registered reader name.
    """
    if isinstance(reader, LEEMReader):
        return reader
    if isinstance(reader, str):
        return LEEMReader._registry[reader]()  # pylint: disable=protected-access

    session_reader, session_options = get_reader()
    if isinstance(session_reader, LEEMReader):
        return session_reader
    if isinstance(session_reader, str):
        return LEEMReader._registry[session_reader](**session_options)  # pylint: disable=protected-access

    name, options = _load_config(source)
    if name is not None:
        return LEEMReader._registry[name](**options)  # pylint: disable=protected-access

    return ElmitecReader()
