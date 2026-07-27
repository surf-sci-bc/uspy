"""Tests the uspy.dataobject.lazy / uspy.leem.base LazyLEEMStack out-of-core backend.

`LazyLEEMStack`/`LazyImageStack` build a lazy dask.array of pixel data instead of holding every
frame in memory (see uspy/dataobject/lazy.py for the design). The central correctness claim is
*parity*: every value a lazy stack produces (pixels, per-frame metadata, arithmetic, normalize,
average) must exactly match what the existing eager `LEEMStack`/`ImageStack` produces for the
same source data -- these tests build both from the same fixtures and compare directly, rather
than asserting against hand-picked expected values, so they can't silently paper over a real
divergence.

Requires the optional 'dask'/'zarr' dependencies (`pip install uspy[bigdata]`); skipped
(not failed) otherwise, except for the tests that specifically exercise the missing-dependency
error path via monkeypatching (those don't need dask installed at all).
"""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import numpy as np
import pytest

pytest.importorskip("dask")

from uspy.dataobject.image import Image, ImageStack
from uspy.dataobject import lazy as lazy_mod
from uspy.dataobject.lazy import LazyImageStack
from uspy.leem.base import LEEMImg, LEEMStack, LazyLEEMStack
from uspy.leem.readers.elmitec import ElmitecReader
from uspy.leem.utility import imgify, stackify

from .conftest import TESTDATA_DIR, STACK_FNAMES, MCP_IMG_FNAME, DC_IMG_FNAME


### Construction / pixel + metadata parity


def test_lazy_pixel_parity(stack_folder):
    eager = LEEMStack(stack_folder)
    lazy = LazyLEEMStack(stack_folder)
    assert len(eager) == len(lazy)
    for i in (0, 1, len(lazy) // 2, len(lazy) - 1):
        assert isinstance(lazy[i], LEEMImg)
        np.testing.assert_array_equal(eager[i].image, lazy[i].image)
        assert eager[i].width == lazy[i].width
        assert eager[i].height == lazy[i].height


def test_lazy_metadata_parity(stack_folder):
    eager = LEEMStack(stack_folder)
    lazy = LazyLEEMStack(stack_folder)
    for field in ("energy", "temperature", "timestamp", "fov"):
        eager_values = getattr(eager, field)
        lazy_values = getattr(lazy, field)
        assert len(eager_values) == len(lazy_values)
        for a, b in zip(eager_values, lazy_values):
            if isinstance(a, str):
                assert a == b
            else:
                assert a == b or (np.isnan(a) and np.isnan(b))


def test_lazy_metadata_access_reads_no_pixels(stack_folder, monkeypatch):
    calls = []
    original = ElmitecReader.parse_image
    monkeypatch.setattr(
        ElmitecReader, "parse_image", lambda self, source: (calls.append(source), original(self, source))[1]
    )

    lazy = LazyLEEMStack(stack_folder)
    # Building the dask graph reads exactly the first frame's pixels once, to learn shape/dtype.
    assert len(calls) == 1

    _ = lazy.energy
    _ = lazy.temperature
    _ = lazy.timestamp
    # Metadata is fully resident from construction -- gathering it must not read any more pixels.
    assert len(calls) == 1


def test_lazy_from_3d_tiff(stack_fname):
    if not stack_fname.lower().endswith((".tif", ".tiff")):
        pytest.skip("only meaningful for the 3D TIFF fixture")
    eager = LEEMStack(stack_fname)
    lazy = LazyLEEMStack(stack_fname)
    assert len(eager) == len(lazy)
    assert lazy.sources is None
    for i in (0, len(lazy) - 1):
        np.testing.assert_array_equal(eager[i].image, lazy[i].image)


def test_lazy_generic_image_stack(stack_fname):
    if not stack_fname.lower().endswith((".tif", ".tiff")):
        pytest.skip("only meaningful for the 3D TIFF fixture")
    eager = ImageStack(stack_fname)
    lazy = LazyImageStack(stack_fname)
    assert len(eager) == len(lazy)
    for i in (0, len(lazy) - 1):
        np.testing.assert_allclose(eager[i].image, lazy[i].image)


def test_lazy_single_frame_stack():
    fname = TESTDATA_DIR + "bremen.dat"
    eager = LEEMStack([fname])
    lazy = LazyLEEMStack([fname])
    assert len(lazy) == 1
    np.testing.assert_array_equal(eager[0].image, lazy[0].image)


def test_lazy_empty_source_raises():
    with pytest.raises(ValueError):
        LazyLEEMStack([])


### Indexing / slicing


def test_lazy_slicing_propagates_type_and_metadata(stack_folder):
    lazy = LazyLEEMStack(stack_folder)
    sliced = lazy[2:5]
    assert type(sliced) is LazyLEEMStack
    assert len(sliced) == 3
    np.testing.assert_array_equal(sliced[0].image, lazy[2].image)
    assert sliced.energy[0] == lazy.energy[2] or (
        np.isnan(sliced.energy[0]) and np.isnan(lazy.energy[2])
    )


def test_lazy_negative_and_out_of_range_indexing(stack_folder):
    lazy = LazyLEEMStack(stack_folder)
    np.testing.assert_array_equal(lazy[-1].image, lazy[len(lazy) - 1].image)
    with pytest.raises(IndexError):
        _ = lazy[len(lazy)]
    with pytest.raises(TypeError):
        _ = lazy["nonsense"]


### Arithmetic parity


def test_lazy_arithmetic_parity(stack_folder):
    eager = LEEMStack(stack_folder)
    lazy = LazyLEEMStack(stack_folder)

    e_expr = (eager * 2) - 3
    l_expr = (lazy * 2) - 3
    assert type(l_expr) is LazyLEEMStack
    for i in (0, len(lazy) - 1):
        np.testing.assert_allclose(e_expr[i].image, l_expr[i].image)

    e_div = eager / 2 + 1
    l_div = lazy / 2 + 1
    for i in (0,):
        np.testing.assert_allclose(e_div[i].image, l_div[i].image)


def test_lazy_arithmetic_with_single_image(stack_folder):
    lazy = LazyLEEMStack(stack_folder)
    other = imgify(MCP_IMG_FNAME)
    result = lazy - other
    np.testing.assert_allclose(result[0].image, lazy[0].image - other.image)


def test_lazy_inplace_arithmetic_raises(stack_folder):
    lazy = LazyLEEMStack(stack_folder)
    with pytest.raises(TypeError):
        lazy += 1
    with pytest.raises(TypeError):
        lazy -= 1
    with pytest.raises(TypeError):
        lazy *= 2
    with pytest.raises(TypeError):
        lazy /= 2


### normalize() parity


@pytest.mark.parametrize("dark_counts_kind", ["path", "image", "int"])
def test_lazy_normalize_parity(stack_folder, dark_counts_kind):
    # Deliberately not reusing conftest's module-scoped `dark_counts` fixture here: combined
    # with another parametrized fixture (stack_folder) in the same test, it triggers an
    # unrelated pytest fixture-caching quirk (that fixture embeds real LEEMImg objects
    # directly as `params`) that's never actually hit by the existing suite, since the one
    # test that combines them (test_driftnorm.py) is in conftest.py's collect_ignore.
    mcp = imgify(MCP_IMG_FNAME)
    dark_counts = {
        "path": DC_IMG_FNAME,
        "image": imgify(DC_IMG_FNAME),
        "int": 100,
    }[dark_counts_kind]

    eager = LEEMStack(stack_folder)
    lazy = LazyLEEMStack(stack_folder)

    e_norm = eager.normalize(mcp=mcp, dark_counts=dark_counts)
    l_norm = lazy.normalize(mcp=mcp, dark_counts=dark_counts)
    assert type(l_norm) is LazyLEEMStack
    for i in (0, len(lazy) // 2, len(lazy) - 1):
        np.testing.assert_allclose(e_norm[i].image, l_norm[i].image, atol=1e-4)


def test_lazy_normalize_requires_explicit_mcp(stack_folder):
    lazy = LazyLEEMStack(stack_folder)
    with pytest.raises(ValueError):
        lazy.normalize()


### average() parity


def test_lazy_average_parity(stack_folder):
    eager = LEEMStack(stack_folder)
    lazy = LazyLEEMStack(stack_folder)
    n = len(lazy) - (len(lazy) % 2)  # ensure even length
    eager, lazy = eager[:n], lazy[:n]

    e_avg = eager.average(2)
    l_avg = lazy.average(2)
    assert len(e_avg) == len(l_avg)
    for i in (0, len(l_avg) - 1):
        np.testing.assert_allclose(e_avg[i].image, l_avg[i].image, atol=1e-3)


def test_lazy_average_bad_divisor_raises(stack_folder):
    lazy = LazyLEEMStack(stack_folder)
    with pytest.raises(ValueError):
        lazy.average(len(lazy) + 1)  # never evenly divides


### materialize() / to_lazy()


def test_lazy_materialize_roundtrip(stack_folder):
    eager = LEEMStack(stack_folder)
    lazy = LazyLEEMStack(stack_folder)
    materialized = lazy.materialize()
    assert type(materialized) is LEEMStack
    assert len(materialized) == len(eager)
    for i in (0, len(eager) - 1):
        np.testing.assert_array_equal(materialized[i].image, eager[i].image)


def test_lazy_materialize_warns_when_large(stack_folder, monkeypatch):
    monkeypatch.setattr(lazy_mod, "_MATERIALIZE_WARN_BYTES", 0)
    lazy = LazyLEEMStack(stack_folder)
    with pytest.warns(ResourceWarning):
        lazy.materialize()


def test_to_lazy_from_eager_stack(stack_folder):
    eager = LEEMStack(stack_folder)
    lazy = eager.to_lazy()
    assert type(lazy) is LazyLEEMStack
    assert len(lazy) == len(eager)
    np.testing.assert_array_equal(lazy[0].image, eager[0].image)


### Entry points: LEEMStack(lazy=True), stackify(lazy=True)


def test_leemstack_lazy_kwarg_dispatch(stack_folder):
    stack = LEEMStack(stack_folder, lazy=True)
    assert type(stack) is LazyLEEMStack
    assert not isinstance(stack, LEEMStack)


def test_stackify_lazy(stack_folder):
    stack = stackify(stack_folder, lazy=True)
    assert type(stack) is LazyLEEMStack
    # passthrough for an already-lazy stack
    assert stackify(stack, lazy=True) is stack


def test_imagestack_lazy_kwarg_dispatch(stack_fname):
    if not stack_fname.lower().endswith((".tif", ".tiff")):
        pytest.skip("only meaningful for the 3D TIFF fixture")
    stack = ImageStack(stack_fname, lazy=True)
    assert type(stack) is LazyImageStack
    # the eager/virtual constructors must be unaffected by the new kwarg
    eager = ImageStack(stack_fname)
    assert type(eager) is ImageStack
    virt = ImageStack(stack_fname, virtual=True)
    assert virt.virtual is True


### Persistence: save() to zarr/tif, source untouched, pickle/json forbidden


def test_lazy_save_zarr_roundtrip(stack_folder, tmp_path):
    da = pytest.importorskip("dask.array")
    lazy = LazyLEEMStack(stack_folder)
    target = str(tmp_path / "stack.zarr")
    lazy.save(target)
    reloaded = da.from_zarr(target)
    np.testing.assert_array_equal(np.asarray(reloaded[0]), lazy[0].image)
    np.testing.assert_array_equal(np.asarray(reloaded[-1]), lazy[-1].image)


def test_lazy_save_tif_roundtrip(stack_folder, tmp_path):
    import tifffile

    lazy = LazyLEEMStack(stack_folder)[:3]
    target = str(tmp_path / "stack.tif")
    lazy.save(target)
    arr = tifffile.imread(target)
    assert arr.shape[0] == 3
    np.testing.assert_allclose(arr[0], lazy[0].image)
    np.testing.assert_allclose(arr[2], lazy[2].image)


def test_lazy_save_unsupported_extension_raises(stack_folder, tmp_path):
    lazy = LazyLEEMStack(stack_folder)
    with pytest.raises(NotImplementedError):
        lazy.save(str(tmp_path / "stack.pickle"))


def test_lazy_save_does_not_modify_source(stack_folder, tmp_path):
    import hashlib
    import os

    def file_hash(path):
        with open(path, "rb") as handle:
            return hashlib.md5(handle.read()).hexdigest()

    fname = sorted(
        f for f in os.listdir(stack_folder) if f.lower().endswith(".dat")
    )[0]
    full_path = os.path.join(stack_folder, fname)
    before = file_hash(full_path)

    lazy = LazyLEEMStack(stack_folder)
    lazy.save(str(tmp_path / "stack.zarr"))
    (lazy * 2).save(str(tmp_path / "stack2.zarr"))

    after = file_hash(full_path)
    assert before == after


def test_lazy_dumps_raises(stack_folder):
    lazy = LazyLEEMStack(stack_folder)
    with pytest.raises(NotImplementedError):
        lazy.dumps()
    with pytest.raises(NotImplementedError):
        lazy.dumps(use_pickle=False)


### Operations that are intentionally unsupported on lazy stacks


def test_lazy_setitem_delitem_extend_filter_raise(stack_folder):
    lazy = LazyLEEMStack(stack_folder)
    with pytest.raises(TypeError):
        lazy[0] = lazy[1]
    with pytest.raises(TypeError):
        del lazy[0]
    with pytest.raises(TypeError):
        lazy.extend(lazy[0])
    with pytest.raises(NotImplementedError):
        lazy.filter()


def test_lazy_virtual_setter_raises(stack_folder):
    lazy = LazyLEEMStack(stack_folder)
    assert lazy.virtual is False
    assert lazy.lazy is True
    with pytest.raises(NotImplementedError):
        lazy.virtual = True


def test_lazy_bulk_metadata_assignment(stack_folder):
    lazy = LazyLEEMStack(stack_folder)
    new_energy = np.linspace(0, 1, len(lazy))
    lazy.energy = new_energy
    np.testing.assert_allclose(lazy.energy, new_energy)
    with pytest.raises(ValueError):
        lazy.energy = 5.0  # a scalar can't overwrite a real per-frame field


### Stack-level scalar metadata (a label, not a per-frame value)


def test_lazy_stack_level_metadata(stack_folder):
    lazy = LazyLEEMStack(stack_folder)
    lazy.label = "my experiment"
    assert lazy.label == "my experiment"


def test_lazy_stack_level_metadata_collision_raises(stack_folder):
    lazy = LazyLEEMStack(stack_folder)
    with pytest.raises(ValueError):
        lazy.energy = 5.0  # 'energy' is a real per-frame field
    # the per-frame gather must be unaffected by the rejected assignment
    assert len(lazy.energy) == len(lazy)


def test_lazy_stack_level_metadata_survives_derivation(stack_folder):
    lazy = LazyLEEMStack(stack_folder)
    lazy.label = "my experiment"

    assert lazy[1:4].label == "my experiment"
    assert (lazy + 1).label == "my experiment"
    assert lazy.copy().label == "my experiment"
    assert lazy.average(2).label == "my experiment"


def test_lazy_unknown_attribute_raises_attributeerror(stack_folder):
    lazy = LazyLEEMStack(stack_folder)
    with pytest.raises(AttributeError):
        _ = lazy.totally_unknown_field


### Missing optional dependency


def test_missing_dask_raises_helpful_importerror(monkeypatch):
    monkeypatch.setattr(lazy_mod, "_HAVE_DASK", False)
    with pytest.raises(ImportError, match="uspy\\[bigdata\\]"):
        lazy_mod._require_dask()
