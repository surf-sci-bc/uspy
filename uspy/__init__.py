"""
uspy -- surface-science microscopy/spectroscopy data analysis.

`uspy.DATADIR`/`uspy.LEEMDIR` are the default data roots (`~/data/` and `~/data/LEEM/`
respectively, each a `str` ending in a path separator). They are resolved lazily on
every access from a `uspy.toml` `[uspy.paths]` table (current working directory, then
the global config directory) or a session override -- see `uspy.config` for the full
resolution order, and `uspy.config.set_datadir`/`set_leemdir` to override them from a
notebook without a config file. With no config anywhere, they keep their historical
`~/data/...` default.
"""

from typing import TYPE_CHECKING

from uspy.config import get_path as _get_path
from uspy.version import __version__

# `from uspy.config import get_path` above also binds `uspy.config` as a submodule
# attribute of `uspy`, so `uspy.config.set_datadir(...)` works after a bare `import uspy`.

__all__ = ["DATADIR", "LEEMDIR", "__version__", "config"]

_PATH_ATTRS = {"DATADIR": "datadir", "LEEMDIR": "leemdir"}

if TYPE_CHECKING:  # pragma: no cover -- for type checkers / IDE completion only
    DATADIR: str
    LEEMDIR: str


def __getattr__(name: str) -> str:
    key = _PATH_ATTRS.get(name)
    if key is not None:
        return _get_path(key)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | set(_PATH_ATTRS))
