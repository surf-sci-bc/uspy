"""Unmaintained, non-functional legacy code (XPS, STM, and pre-dataobject LEEM API).

Quarantined here on 2026-07-28: none of it runs against the current uspy API (see the
individual modules for what broke and why), and it is excluded from test collection
(tests/conftest.py). Do not use it as a reference for new work; do not re-export from here.
"""

from uspy import DATADIR

XPSDIR = DATADIR + "XPS/"
STMDIR = DATADIR + "STM/home/stmwizard/Documents/"
