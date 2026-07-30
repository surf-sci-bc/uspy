"""Tests for deferred (`defer=True`) construction of `DataObject` and arithmetic composition
on deferred `_data` values.

Covers:
- the generic re-parse fallback (`DataObject.parse_deferred`'s default implementation),
  which must work for *any* subclass with zero extra code -- this is what lets `virtual=True`
  work on every DataObjectStack element type, not just the ones with a cheap reader path
- pixel data is never cached: reading a deferred value twice re-runs the loader twice
- a write replaces the deferred handle with a resident value (localized -- other fields/
  elements are unaffected)
- arithmetic on a deferred `Image` composes into a new `DeferredArray` (doesn't load), while
  a resident `Image` still mutates its array in place for `+=`/`-=`/etc.
"""
# pylint: disable=missing-docstring
# pylint: disable=protected-access

import numpy as np
import pytest

from uspy.dataobject import DataObject, Image
from uspy.dataobject.deferred import DeferredArray


class CountingReparse(DataObject):
    """DataObject whose `parse` counts how many times it actually ran, via a counter carried
    in a mutable `source` dict -- lets tests prove `defer=True` construction performs zero
    parses, and that each subsequent read re-parses (no caching)."""

    _data_keys = ("value",)
    _meta_defaults = {"label": "default"}

    def parse(self, source):
        self._source = source
        source["calls"] += 1
        return {"value": np.array([source["calls"]]), "label": f"call-{source['calls']}"}


class _CountingArrayLoader:
    def __init__(self, array: np.ndarray):
        self.array = array
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.array.copy()


### Generic re-parse fallback (works for any DataObject subclass, zero extra code)


def test_defer_construction_performs_zero_parses():
    source = {"calls": 0}
    obj = CountingReparse(source, defer=True)
    assert source["calls"] == 0
    assert obj.is_virtual


def test_deferred_pixel_value_is_never_cached():
    source = {"calls": 0}
    obj = CountingReparse(source, defer=True)
    first = obj.value
    assert source["calls"] == 1
    second = obj.value
    assert source["calls"] == 2
    # each access re-parsed from scratch -- the counter it captured differs each time
    assert first[0] != second[0]


def test_deferred_metadata_resolves_once_and_caches():
    source = {"calls": 0}
    obj = CountingReparse(source, defer=True)
    assert obj.label == "call-1"
    assert source["calls"] == 1
    # metadata is cached permanently -- rereading it must not reparse
    assert obj.label == "call-1"
    assert source["calls"] == 1
    # ... independent of how many times the (never-cached) pixel value is read meanwhile
    _ = obj.value
    _ = obj.value
    assert obj.label == "call-1"


def test_write_to_deferred_field_replaces_handle_and_stops_reparsing():
    source = {"calls": 0}
    obj = CountingReparse(source, defer=True)
    with pytest.warns(UserWarning, match="deferred"):
        obj.value = np.array([999])
    assert not obj.is_virtual
    assert source["calls"] == 0  # writing must not itself trigger a parse
    assert np.array_equal(obj.value, [999])
    assert source["calls"] == 0  # reading the now-resident value doesn't reparse either


def test_write_to_deferred_field_warns_once_then_stays_quiet(recwarn):
    """`DataObject.__setattr__` warns the moment a resident write pins a still-deferred
    element in memory (see tests/test_base.py for the Image-specific coverage) -- generic
    across any DataObject subclass, not just the ones with a cheaper deferred path. A second
    write, now resident-to-resident, must not warn again."""
    source = {"calls": 0}
    obj = CountingReparse(source, defer=True)

    with pytest.warns(UserWarning, match="deferred"):
        obj.value = np.array([1])
    recwarn.clear()

    obj.value = np.array([2])
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]


def test_defer_raises_without_a_source():
    obj = CountingReparse({"calls": 0}, defer=True)
    obj.materialize()
    obj._source = None
    with pytest.raises(ValueError):
        obj.defer()


def test_defer_then_reload_roundtrip():
    source = {"calls": 0}
    obj = CountingReparse(source, defer=True)
    obj.materialize()
    assert not obj.is_virtual
    assert source["calls"] == 1
    obj.defer()
    assert obj.is_virtual
    assert source["calls"] == 1  # defer() itself must not eagerly reparse
    _ = obj.value
    assert source["calls"] == 2


### Arithmetic composition on deferred Image data


def test_image_arithmetic_composes_without_loading():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    loader = _CountingArrayLoader(arr)
    img = Image(np.zeros((2, 2), dtype=np.float32))
    img._data["image"] = DeferredArray(loader, shape=arr.shape, dtype=arr.dtype)

    result = (img * 2) - 3

    assert loader.calls == 0  # composing (out-of-place) must not load
    assert isinstance(result._data["image"], DeferredArray)
    assert np.allclose(result.image, (arr * 2) - 3)


def test_image_iadd_composes_when_deferred():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    loader = _CountingArrayLoader(arr)
    img = Image(np.zeros((2, 2), dtype=np.float32))
    img._data["image"] = DeferredArray(loader, shape=arr.shape, dtype=arr.dtype)

    img += 5

    assert loader.calls == 0  # in-place op on deferred data composes, doesn't load
    assert isinstance(img._data["image"], DeferredArray)
    assert np.allclose(img.image, arr + 5)


def test_image_iadd_mutates_resident_array_in_place():
    img = Image(np.array([[1.0, 1.0]], dtype=np.float32))
    original_array = img._data["image"]

    img += 1

    assert img._data["image"] is original_array  # same object, mutated in place
    assert np.allclose(img.image, [[2.0, 2.0]])
