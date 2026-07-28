"""Data directories."""

from pathlib import Path

from uspy.version import __version__

DATADIR = str(Path.home() / "data") + "/"
LEEMDIR = DATADIR + "LEEM/"
