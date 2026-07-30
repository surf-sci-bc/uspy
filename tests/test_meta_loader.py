"""Tests for the lazily-populated `_meta`/`_units` mechanism on `DataObject`.

This pins down the internal `_meta_loader` plumbing in isolation, before anything in
`uspy` actually installs a loader at construction time (that comes later, once deferred
construction is wired up). The mechanism must:

- resolve on the *first touch of the container* (a read OR a write), not on a per-key miss --
  a per-field DataObject like LEEMImg already ships every field in `_meta_defaults`
  (temperature/energy/timestamp/... all default to nan), so a miss-triggered loader would
  never fire and would silently return the wrong (default) value instead. See
  `test_loader_not_triggered_by_key_presence_in_defaults` below for the direct regression.
- resolve exactly once, ever, and cache permanently after that
- never fire just from printing/repr'ing the object
- leave `_meta`/`_units`/`data`/`_with_data`'s existing dict semantics unchanged when no
  loader is installed at all (the common case for the whole rest of the test suite)
"""
# pylint: disable=missing-docstring
# pylint: disable=protected-access

import numpy as np
import pytest

from uspy.dataobject import DataObject


class Counting(DataObject):
    """Minimal DataObject subclass with a settable, countable metadata loader."""

    _data_keys = ("value",)
    _meta_defaults = {"energy": np.nan, "label": "default"}
    _unit_defaults = {"energy": "eV"}

    def parse(self, source):
        self._source = source
        return {"value": np.asarray(source)}


def install_loader(obj, meta: dict, calls: list):
    def loader():
        calls.append(1)
        return dict(meta)

    obj._meta_loader = loader
    return obj


### No loader installed: behaves exactly like a plain dict-backed DataObject


def test_no_loader_behaves_like_plain_dict():
    obj = Counting(np.array([1, 2, 3]))
    assert obj.label == "default"
    obj.label = "changed"
    assert obj.label == "changed"
    assert obj.meta["label"] == "changed"


### Resolution triggers on container touch, not key miss


def test_read_resolves_loader_once():
    calls = []
    obj = install_loader(Counting(np.array([1])), {"energy": 5.0}, calls)
    assert obj.energy == 5.0
    assert calls == [1]
    # further reads must not call the loader again
    assert obj.energy == 5.0
    assert calls == [1]


def test_write_resolves_loader_and_persists():
    calls = []
    obj = install_loader(Counting(np.array([1])), {"energy": 5.0}, calls)
    obj.label = "overridden"
    assert calls == [1]  # writing also resolves
    assert obj.label == "overridden"
    assert obj.energy == 5.0  # parsed value still present alongside the write


def test_loader_not_triggered_by_key_presence_in_defaults():
    """The regression this whole design exists to prevent: `energy` is already a key in
    `_meta_defaults` (nan), so a naive "resolve on __getattr__ miss" implementation would
    never fire the loader at all and `obj.energy` would silently be nan forever."""
    calls = []
    obj = install_loader(Counting(np.array([1])), {"energy": 42.0}, calls)
    assert not np.isnan(obj.energy)
    assert obj.energy == 42.0
    assert calls == [1]


def test_repr_does_not_resolve():
    """`repr()` (the default object repr, DataObject defines no `__repr__`) must not resolve
    -- unlike `str()`, which deliberately prints real metadata values and so is expected (and
    fine: a single bounded header read, not the "excessive I/O" this design guards against) to
    resolve the loader."""
    calls = []
    obj = install_loader(Counting(np.array([1])), {"energy": 5.0}, calls)
    repr(obj)
    assert calls == []


def test_str_shows_resolved_metadata_but_not_deferred_pixel_values():
    """Printing an object legitimately shows its real metadata (one bounded resolve), but
    must never load pixel/data values just to print them -- a deferred `_data` entry should
    show its DeferredArray repr instead."""
    calls = []
    obj = install_loader(Counting(np.array([1])), {"energy": 5.0}, calls)
    text = str(obj)
    assert calls == [1]
    assert "5.0" in text


def test_units_resolution_shares_the_same_loader():
    calls = []
    obj = Counting(np.array([1]))
    obj._meta_loader = lambda: (calls.append(1), {"energy": 5.0, "energy_unit": "meV"})[1]
    assert obj.unit["energy"] == "meV"
    assert calls == [1]
    assert obj.energy == 5.0
    assert calls == [1]  # already resolved by the .unit access above


### `_with_data` keeps independent, loader-carrying stores


def test_with_data_copies_loader_and_stays_independent():
    calls = []
    obj = install_loader(Counting(np.array([1])), {"energy": 5.0}, calls)
    new = obj._with_data(value=np.array([2]))
    assert new.energy == 5.0
    assert calls == [1]
    new.meta["probe"] = "new"
    assert "probe" not in obj.meta
