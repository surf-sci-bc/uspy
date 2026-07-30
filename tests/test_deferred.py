"""Tests for uspy.dataobject.deferred (DeferredArray and its loaders).

These are pure unit tests for the deferral primitive itself -- no LEEM/reader fixtures
needed. They pin down the contract the rest of the refactor depends on:

- shape/dtype/ndim resolve without loading pixel data
- pixel data is NEVER cached: .load() re-runs the loader every time
- arithmetic (both operand orders) COMPOSES into a new DeferredArray instead of
  materializing -- this is what keeps stack.normalize()/arithmetic out-of-core
- copying (deepcopy/copy) never triggers a load
- pickling a picklable loader round-trips without loading; an unpicklable loader
  (e.g. built from a lambda) materializes with a warning instead of failing
"""
# pylint: disable=missing-docstring
# pylint: disable=protected-access

import copy
import pickle

import numpy as np
import pytest

from uspy.dataobject.deferred import DeferredArray, is_deferred, materialize


class CountingLoader:
    """Picklable loader that counts how many times it has been called."""

    def __init__(self, array: np.ndarray):
        self.array = array
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.array.copy()


@pytest.fixture
def arr():
    return np.arange(12, dtype=np.float32).reshape(3, 4)


@pytest.fixture
def loader(arr):
    return CountingLoader(arr)


@pytest.fixture
def deferred(loader, arr):
    return DeferredArray(loader, shape=arr.shape, dtype=arr.dtype)


### Shape/dtype resolve without loading


def test_shape_dtype_known_up_front_no_load(loader, arr):
    d = DeferredArray(loader, shape=arr.shape, dtype=arr.dtype)
    assert d.shape == (3, 4)
    assert d.dtype == arr.dtype
    assert d.ndim == 2
    assert loader.calls == 0


def test_shape_dtype_unknown_falls_back_to_load(loader, arr):
    d = DeferredArray(loader)  # no shape/dtype given
    assert d.shape == arr.shape
    assert loader.calls == 1
    # resolved shape/dtype are then cached -- second access costs nothing extra
    assert d.dtype == arr.dtype
    assert loader.calls == 1


def test_is_deferred(deferred, arr):
    assert is_deferred(deferred)
    assert not is_deferred(arr)
    assert not is_deferred(5)


def test_materialize_helper(deferred, arr):
    assert np.array_equal(materialize(deferred), arr)
    assert materialize(arr) is arr


### Pixel data is never cached


def test_load_twice_calls_loader_twice(deferred, loader):
    deferred.load()
    deferred.load()
    assert loader.calls == 2


def test_array_dunder_calls_loader(deferred, arr, loader):
    result = np.asarray(deferred)
    assert np.array_equal(result, arr)
    assert loader.calls == 1


def test_repr_and_shape_access_never_load(deferred, loader):
    repr(deferred)
    _ = deferred.shape
    _ = deferred.dtype
    _ = deferred.ndim
    assert loader.calls == 0


### Composition: arithmetic never loads eagerly, only .load() does


@pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv"])
def test_binary_op_with_number_composes(deferred, arr, loader, op):
    import operator

    pyop = getattr(operator, op)
    composed = pyop(deferred, 2.0)
    assert loader.calls == 0  # composing must not load
    assert is_deferred(composed)
    expected = pyop(arr, 2.0)
    result = composed.load()
    assert np.allclose(result, expected)


@pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv"])
def test_reflected_binary_op_with_ndarray_composes(deferred, arr, loader, op):
    """`ndarray <op> deferred` must be intercepted by DeferredArray's __array_priority__/
    __array_ufunc__ = None, not by numpy -- otherwise numpy calls __array__ on the deferred
    operand and materializes it right there. This is exactly the shape of
    LEEMImg.normalize(): `(mcp_array - deferred_image)`."""
    import operator

    pyop = getattr(operator, op)
    other = np.full(arr.shape, 5.0, dtype=np.float32)
    composed = pyop(other, deferred)
    assert loader.calls == 0
    assert is_deferred(composed)
    expected = pyop(other, arr)
    assert np.allclose(composed.load(), expected)


def test_chained_ops_compose_and_match_eager(deferred, arr, loader):
    composed = (deferred * 2 - 3) / 2
    assert loader.calls == 0
    expected = (arr * 2 - 3) / 2
    assert np.allclose(composed.load(), expected)
    # load() is called once per leaf DeferredArray per top-level .load() -- but must still
    # never cache: calling .load() again re-runs the whole chain.
    calls_after_first = loader.calls
    assert calls_after_first >= 1
    composed.load()
    assert loader.calls > calls_after_first


def test_map_composes(deferred, arr, loader):
    mapped = deferred.map(lambda a: a + 100)
    assert loader.calls == 0
    assert np.allclose(mapped.load(), arr + 100)


def test_operand_order_and_two_deferred_operands(arr):
    loader_a = CountingLoader(arr)
    loader_b = CountingLoader(arr + 1)
    a = DeferredArray(loader_a, shape=arr.shape, dtype=arr.dtype)
    b = DeferredArray(loader_b, shape=arr.shape, dtype=arr.dtype)
    composed = a + b
    assert loader_a.calls == 0
    assert loader_b.calls == 0
    assert np.allclose(composed.load(), arr + (arr + 1))


### Copying never loads


def test_deepcopy_never_loads(deferred, loader):
    other = copy.deepcopy(deferred)
    assert loader.calls == 0
    assert is_deferred(other)
    assert np.allclose(other.load(), loader.array)


def test_copy_never_loads(deferred, loader):
    other = copy.copy(deferred)
    assert loader.calls == 0
    assert is_deferred(other)


### Pickling


def test_pickle_roundtrip_picklable_loader_does_not_load(deferred, loader, arr):
    blob = pickle.dumps(deferred)
    assert loader.calls == 0  # dumping must not load
    restored = pickle.loads(blob)
    assert is_deferred(restored)
    assert loader.calls == 0  # loading the pickle must not load either
    assert np.allclose(restored.load(), arr)


def test_pickle_unpicklable_loader_materializes_with_warning(arr):
    d = DeferredArray(lambda: arr, shape=arr.shape, dtype=arr.dtype)
    with pytest.warns(UserWarning):
        blob = pickle.dumps(d)
    restored = pickle.loads(blob)
    # restored is a real ndarray (or an eagerly-backed DeferredArray) with the right values --
    # either way, it survives the round trip instead of raising a PicklingError.
    assert np.allclose(np.asarray(restored), arr)


### No elementwise __eq__ (would otherwise break DeepDiff-based DataObject.__eq__)


def test_no_elementwise_eq(deferred):
    other = DeferredArray(lambda: None)
    # identity comparison, not elementwise -- must not raise/broadcast a numpy bool array
    assert (deferred == deferred) is True
    assert (deferred == other) is False
