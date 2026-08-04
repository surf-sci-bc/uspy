"""
Pluggable LEEM data readers.

`LEEMImg`/`LEEMStack` (see `uspy.leem.base`) delegate all format/endstation-specific
parsing to a `LEEMReader`. A reader is chosen for a given source, in order:

1. An explicit `reader=` argument to `LEEMImg`/`LEEMStack` -- a `LEEMReader` instance,
   or the registered name of one (e.g. `reader="elettra"`).
2. A programmatic session-wide override::

       from uspy.leem.readers import set_reader
       set_reader("smart")          # every load for the rest of the session uses this
       set_reader(None)             # or reset_reader() -- clears the override

3. `uspy.toml` config file(s), e.g.::

       [uspy]
       reader = "elettra"

       [uspy.options]      # optional, passed as keyword arguments to the reader
       energy_offset = 0.0

       [uspy.plotting]     # optional, overrides the (Elmitec-oriented) default
       default_fields = ["temperature", "pressure", "energy", "fov"]  # plot_img/make_video overlay
       meta_fields = ["temperature"]                                  # plot_meta
       print_fields = ["energy", "temperature", "fov"]                # print_meta

       [uspy.mask]     # optional, overrides the reader's built-in default mask
       shape  = "circle"   # circle | ellipse | square | rectangle | polygon | point | "none"
       x0     = "50%"      # percent strings scale with frame width  (x0, width, radius)
       y0     = "50%"      # percent strings scale with frame height (y0, height)
       radius = "45%"      # plain numbers are absolute pixels; may be mixed with percents

   searched most-specific first (data folder, its parent, cwd, global config dir --
   see `uspy.config` for the full search/parse machinery, shared via that module). The
   `reader`/`[uspy.options]` pair, and separately the whole `[uspy.mask]` table, are each
   taken as a unit from the first file that declares them; `[uspy.plotting]` fields are
   merged per-key across all found files, more specific files winning.

4. The default `ElmitecReader` (Elmitec UKSOFT2001 ".dat" files); its `default_mask` is a
   channelplate-aperture circle (`x0="50%", y0="50%", radius="45%"`), shared by every
   built-in reader.

Ready-to-copy baseline `uspy.toml` files for each built-in reader live under
`examples/uspy_toml/<reader name>/uspy.toml`.

There is no content-based format detection -- a non-default reader must be requested
explicitly or declared via `set_reader`/a `uspy.toml` file. To support a new endstation,
write a `LEEMReader` subclass (see `uspy.leem.readers.base.LEEMReader`) with a unique
`name`; it registers itself automatically.

The same `uspy.toml` files also carry the `[uspy.paths]` table (`datadir`, `leemdir`,
...) documented in `uspy.config`.

`[uspy.mask]` resolves to `LEEMImg.default_mask` (see `uspy.leem.base`), which
`LEEMStack.align()`/`uspy.plotting.plot_img()`/`uspy.plotting.make_video()` use when passed
`mask=True`. It never applies itself automatically -- see those functions' `mask` parameter.
"""

from __future__ import annotations
from typing import Any, Union

from uspy.config import (  # noqa: F401  (re-exported for backwards compatibility)
    CONFIG_FILENAME,
    config_folder as _config_folder,
    global_config_dir as _global_config_dir,
    config_search_paths as _config_search_paths,
    uspy_tables as _uspy_tables,
)
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
    "resolve_default_mask",
    "set_reader",
    "get_reader",
    "reset_reader",
    "set_default_mask",
    "get_default_mask",
    "reset_default_mask",
]

# Programmatic session-wide reader override, set via `set_reader()` -- see the module
# docstring. Ranks below an explicit `reader=` argument but above every `uspy.toml`.
_session_reader: dict = {"reader": None, "options": {}}


def set_reader(reader: Union[str, LEEMReader, None], **options: Any) -> None:
    """Set a session-wide reader override, taking precedence over any `uspy.toml` for
    every subsequent load without an explicit `reader=`. `reader=None` clears it
    (equivalent to `reset_reader()`). `**options` are passed to the reader on
    instantiation; ignored if `reader` is already a `LEEMReader` instance."""
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


def _load_config(source: Any) -> tuple:
    """Returns a `(reader_name_or_None, options_dict)` tuple taken as a unit from the
    first (most specific) `uspy.toml` on `source`'s config search path that declares a
    `reader` -- see `_uspy_tables`."""
    for section in _uspy_tables(source):
        if "reader" in section:
            return section["reader"], section.get("options", {})
    return None, {}


def resolve_plotting_fields(source: Any, defaults: dict[str, tuple]) -> dict[str, tuple]:
    """Resolve default plotting field-name tuples from `[uspy.plotting]` across every
    `uspy.toml` on `source`'s search path, merged per-key (more specific wins),
    falling back to `defaults` for keys no file sets."""
    merged: dict = {}
    for section in reversed(_uspy_tables(source)):
        merged.update(section.get("plotting", {}))
    return {
        key: tuple(merged[key]) if key in merged else default
        for key, default in defaults.items()
    }


# Programmatic session-wide default-mask override, set via `set_default_mask()` -- see the
# module docstring. Ranks below an explicit `mask=` argument to `align()`/`plot_img()`/
# `make_video()`, but above every `uspy.toml` and the reader's own `default_mask`.
_session_mask: dict = {"active": False, "mask": None}


def set_default_mask(mask: Any) -> None:
    """Set a session-wide default-mask override, taking precedence over any `uspy.toml`
    `[uspy.mask]` table and the reader's own `default_mask` for every subsequent `LEEMImg`.
    `mask` may be a `uspy.roi.ROI`, a spec dict (see `uspy.roi.ROI.from_spec`), `False` to
    disable the default mask, or `None` to clear the override (equivalent to
    `reset_default_mask()`)."""
    if mask is None:
        reset_default_mask()
        return
    _session_mask["active"] = True
    _session_mask["mask"] = mask if mask is not False else None


def get_default_mask() -> tuple:
    """Returns `(active, mask)`: whether a session-wide default-mask override is set via
    `set_default_mask`, and its value (`None` if the override disables the mask).
    `(False, None)` if no override is set."""
    return _session_mask["active"], _session_mask["mask"]


def reset_default_mask() -> None:
    """Clears the session-wide default-mask override set via `set_default_mask`."""
    _session_mask["active"] = False
    _session_mask["mask"] = None


def resolve_default_mask(source: Any, reader: LEEMReader) -> Any:
    """Resolve the default mask for `source`/`reader` -- a `uspy.roi.ROI`, a spec dict for
    `uspy.roi.ROI.from_spec`, or `None` -- most specific first:

    1. A session-wide override set via `set_default_mask`.
    2. The `[uspy.mask]` table of the first `uspy.toml` on `source`'s search path that
       declares one, taken as a unit (like `reader`/`[uspy.options]` -- not merged per-key
       like `[uspy.plotting]`); `shape = "none"` disables the mask.
    3. `reader.default_mask`.
    """
    active, session_mask = get_default_mask()
    if active:
        return session_mask
    for section in _uspy_tables(source):
        if "mask" in section:
            spec = section["mask"]
            if isinstance(spec, dict) and spec.get("shape") == "none":
                return None
            return spec
    return reader.default_mask


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
