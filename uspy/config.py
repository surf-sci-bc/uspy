"""
Shared `uspy.toml` config machinery.

`uspy` can be configured per-site/per-project through one or more `uspy.toml` files.
This module implements the generic parts (locating and parsing those files) plus the
`[uspy.paths]` table, which supplies `uspy.DATADIR`/`uspy.LEEMDIR` (see `uspy.__init__`)
and any other named data-root a site wants to declare. The reader-selection tables
(`[uspy]`/`[uspy.options]`/`[uspy.plotting]`) are documented and resolved in
`uspy.leem.readers`, which builds on the search/parse primitives here.

Config files are searched for at multiple locations, most specific first:

1. the data folder a given `source` points into (only when resolving something that
   has a `source`, e.g. a reader for `LEEMImg(source)` -- see `uspy.leem.readers`)
2. that folder's parent
3. the current working directory
4. a global config directory: `$XDG_CONFIG_HOME/uspy/uspy.toml`, or
   `~/.config/uspy/uspy.toml` if `XDG_CONFIG_HOME` is unset

`[uspy.paths]` has no `source` to anchor to (the data folder is precisely what it
resolves), so only steps 3 and 4 apply -- current working directory, then the global
config directory.

## `[uspy.paths]`

```toml
[uspy.paths]
datadir = "/mnt/surfer/data"
leemdir = "/mnt/surfer/data/LEEM"   # optional; defaults to "<datadir>/LEEM"
xps     = "~/data/XPS"              # arbitrary extra keys are allowed
```

Values may use `~` (expanded to the user's home directory) and `$VAR`/`${VAR}`
environment variable references; a relative value is anchored to the folder of the
`uspy.toml` that declared it. Every resolved path is a `str` ending in a path
separator, e.g. `"/mnt/surfer/data/"`, matching the historical `uspy.DATADIR` format.

Multiple `uspy.toml` files are layered, most specific first (cwd, then the global
config dir): for a given key, the *first layer that mentions it* (either directly, or
via its `_DERIVED` parent -- see `_layer_value`) wins, in its entirety. This is
deliberately not a flat per-key merge: if a project-local file sets only `datadir`, its
`leemdir` must move with it (default to `<that datadir>/LEEM`) even if a less-specific
file also declared `leemdir` explicitly -- otherwise `LEEMDIR` could silently point into
an unrelated tree.

A session override, set via `set_path`/`set_datadir`/`set_leemdir`, ranks above every
`uspy.toml` layer -- useful for one-off notebook work without writing a config file::

    from uspy.config import set_datadir, reset_paths
    set_datadir("/mnt/surfer/data")   # every uspy.DATADIR access for the rest of the
                                       # session uses this, regardless of any uspy.toml
    reset_paths()                     # clears all session path overrides again
"""

from __future__ import annotations

import functools
import os
import warnings
from pathlib import Path
from typing import Any, Optional, Union

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

__all__ = [
    "CONFIG_FILENAME",
    "config_folder",
    "global_config_dir",
    "config_search_paths",
    "uspy_tables",
    "get_path",
    "resolve_paths",
    "set_path",
    "set_datadir",
    "set_leemdir",
    "get_path_overrides",
    "reset_paths",
]

CONFIG_FILENAME = "uspy.toml"

# `leemdir` (and, historically, other endstation-technique dirs) defaults to a fixed
# subfolder of `datadir` when not given explicitly.
_DERIVED = {"leemdir": ("datadir", "LEEM")}

# Session-wide path overrides, set via `set_path`/`set_datadir`/`set_leemdir` -- see the
# module docstring. Ranks above every `uspy.toml` layer.
_session_paths: dict = {}


# --------------------------------------------------------------------------------
# Locating and parsing uspy.toml files (shared with uspy.leem.readers)
# --------------------------------------------------------------------------------


def config_folder(source: Any) -> Optional[Path]:
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


def global_config_dir() -> Path:
    """The global config directory: `$XDG_CONFIG_HOME/uspy`, or `~/.config/uspy` if
    `XDG_CONFIG_HOME` is unset -- for a `uspy.toml` that applies machine-/user-wide
    (e.g. a measurement machine that only ever produces one file format, or a site-wide
    `[uspy.paths]`)."""
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"
    return base / "uspy"


def config_search_paths(source: Any) -> list:
    """
    Ordered list of folders to look for a `uspy.toml` in, most specific first (see the
    module docstring): the data folder (if `source` is given), its parent, the current
    working directory, and the global config directory. Entries are de-duplicated by
    resolved path (e.g. when the data folder IS the current working directory) while
    preserving the first, most-specific occurrence.
    """
    candidates = []
    data_folder = config_folder(source)
    if data_folder is not None:
        candidates.append(data_folder)
        candidates.append(data_folder.parent)
    candidates.append(Path.cwd())
    candidates.append(global_config_dir())

    seen = set()
    paths = []
    for folder in candidates:
        resolved = folder.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(folder)
    return paths


@functools.lru_cache(maxsize=64)
def _parse_toml_cached(config_path_str: str, mtime_ns: int) -> dict:  # pylint: disable=unused-argument
    """Parse a uspy.toml file, cached on its path+mtime so re-instantiating many
    objects from the same folder (e.g. one per frame of a stack) doesn't re-open and
    re-parse the file each time."""
    with open(config_path_str, "rb") as f:
        return tomllib.load(f)


def _uspy_tables_located(source: Any) -> list:
    """Returns `(uspy_table, containing_folder)` pairs for every `uspy.toml` found
    along `source`'s config search path (see `config_search_paths`), most specific
    first. Folders without a `uspy.toml` are silently skipped. `containing_folder` is
    the resolved folder the config file lives in, used to anchor relative path values
    (see `_normalize_path_value`)."""
    located = []
    for folder in config_search_paths(source):
        config_path = folder / CONFIG_FILENAME
        if not config_path.is_file():
            continue
        config = _parse_toml_cached(str(config_path), config_path.stat().st_mtime_ns)
        located.append((config.get("uspy", {}), folder.resolve()))
    return located


def uspy_tables(source: Any) -> list:
    """Returns the `[uspy]` table of every `uspy.toml` found along `source`'s config
    search path (see `config_search_paths`), most specific first. Folders without a
    `uspy.toml` are silently skipped."""
    return [table for table, _folder in _uspy_tables_located(source)]


# --------------------------------------------------------------------------------
# [uspy.paths] -- DATADIR/LEEMDIR/... resolution
# --------------------------------------------------------------------------------


def _normalize_path_value(value: Any, anchor: Optional[Path], key: str, origin: str) -> str:
    """Turn a raw `[uspy.paths]` value into the historical `str`-with-trailing-slash
    form: validate, expand `$VAR`/`~`, anchor if relative, normalize separators."""
    if not isinstance(value, str):
        raise TypeError(
            f"{origin}: [uspy.paths] {key!r} must be a string, got {type(value).__name__}"
        )
    if not value.strip():
        raise ValueError(f"{origin}: [uspy.paths] {key!r} must not be empty")

    expanded = os.path.expanduser(os.path.expandvars(value))
    if "$" in expanded or ("%" in expanded and os.name == "nt"):
        warnings.warn(
            f"{origin}: [uspy.paths] {key!r} = {value!r} contains an environment "
            "variable reference that could not be expanded; using it literally.",
            UserWarning,
            stacklevel=2,
        )

    path = Path(expanded)
    if not path.is_absolute():
        base = anchor if anchor is not None else Path.cwd()
        path = base / path
    return str(path) + "/"


def _layer_value(layer: dict, key: str, anchor: Optional[Path], origin: str) -> Optional[str]:
    """Resolve `key` from a single `[uspy.paths]` layer, following the `_DERIVED` rule
    (e.g. `leemdir` defaults to `<that layer's datadir>/LEEM` when not given
    explicitly). Returns `None` if this layer says nothing about `key`."""
    if key in layer:
        return _normalize_path_value(layer[key], anchor, key, origin)
    derived = _DERIVED.get(key)
    if derived is not None:
        parent_key, subfolder = derived
        if parent_key in layer:
            parent = _normalize_path_value(layer[parent_key], anchor, parent_key, origin)
            return str(Path(parent) / subfolder) + "/"
    return None


def _default_datadir() -> Path:
    return Path.home() / "data"


def get_path(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Resolve a named data path (e.g. `"datadir"`, `"leemdir"`, or any site-specific key
    declared under `[uspy.paths]`), most-specific layer first: a session override set
    via `set_path`/`set_datadir`/`set_leemdir`, then each `uspy.toml`'s `[uspy.paths]`
    table found on the search path (cwd, then the global config dir) -- see the module
    docstring for why this is layered rather than merged per-key.

    Falls back to `~/data/` for `"datadir"`, `<datadir>/LEEM/` for `"leemdir"`, and
    `default` (normalized the same way as a config value, if given) for anything else.
    """
    if name in _session_paths:
        return _normalize_path_value(_session_paths[name], None, name, "session override")

    if name in _DERIVED:
        parent_key, _subfolder = _DERIVED[name]
        if parent_key in _session_paths:
            parent = _normalize_path_value(
                _session_paths[parent_key], None, parent_key, "session override"
            )
            return str(Path(parent) / _DERIVED[name][1]) + "/"

    for layer, folder in _uspy_tables_located(None):
        paths = layer.get("paths", {})
        if not isinstance(paths, dict):
            raise TypeError(
                f"{folder / CONFIG_FILENAME}: [uspy.paths] must be a table, "
                f"got {type(paths).__name__}"
            )
        value = _layer_value(paths, name, folder, str(folder / CONFIG_FILENAME))
        if value is not None:
            return value

    if name == "datadir":
        return str(_default_datadir()) + "/"
    if name == "leemdir":
        return get_path("datadir") + "LEEM/"
    if default is not None:
        return _normalize_path_value(default, None, name, "default")
    return None


def resolve_paths() -> dict:
    """Returns `{"datadir": ..., "leemdir": ...}` for introspection/repr purposes."""
    return {"datadir": get_path("datadir"), "leemdir": get_path("leemdir")}


def set_path(name: str, value: Union[str, "os.PathLike", None]) -> None:
    """
    Set a session-wide override for a named data path (e.g. `"datadir"`), used by
    `get_path` for every subsequent access -- taking precedence over any `uspy.toml`.
    Mirrors `uspy.leem.readers.set_reader`.

    `value` may be a `str`, a `os.PathLike`, or `None` to clear just this key
    (equivalent to not having set it).
    """
    if value is None:
        _session_paths.pop(name, None)
        return
    _session_paths[name] = os.fspath(value)


def set_datadir(value: Union[str, "os.PathLike", None]) -> None:
    """Shorthand for `set_path("datadir", value)`."""
    set_path("datadir", value)


def set_leemdir(value: Union[str, "os.PathLike", None]) -> None:
    """Shorthand for `set_path("leemdir", value)`."""
    set_path("leemdir", value)


def get_path_overrides() -> dict:
    """Returns a copy of the current session-wide path overrides set via `set_path`."""
    return dict(_session_paths)


def reset_paths() -> None:
    """Clears all session-wide path overrides set via `set_path`/`set_datadir`/
    `set_leemdir`."""
    _session_paths.clear()
