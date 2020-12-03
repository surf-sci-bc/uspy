"""Data directories."""

from pathlib import Path

from agfalta.version import __version__

DATADIR = str(Path.home() / "/data/") + "/"
LEEMDIR = DATADIR + "LEEM/"
XPSDIR = DATADIR + "XPS/"
STMDIR = DATADIR + "STM/home/stmwizard/Documents/"
