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
from uspy.leem.readers.elettra import ElettraReader
from uspy.leem.readers.elmitec import ElmitecReader
from uspy.leem.readers.smart import SmartReader
from uspy.leem.utility import imgify, stackify

from .conftest import (
    TESTDATA_DIR,
    STACK_FNAMES,
    MCP_IMG_FNAME,
    DC_IMG_FNAME,
    ELETTRA_TIF_DIR,
    BESSY_SCAN_FNAME,
)


def assert_field_parity(eager, lazy, fields):
    """Assert `getattr(eager, field)` and `getattr(lazy, field)` agree element-wise for every
    field in `fields` -- nan-aware (nan == nan here) and string-aware (exact match)."""
    for field in fields:
        eager_values = getattr(eager, field)
        lazy_values = getattr(lazy, field)
        assert len(eager_values) == len(lazy_values), field
        for a, b in zip(eager_values, lazy_values):
            if isinstance(a, str):
                assert a == b, field
            else:
                assert a == b or (np.isnan(a) and np.isnan(b)), field


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
    assert_field_parity(eager, lazy, ("energy", "temperature", "timestamp", "fov"))


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


### Metadata parity for readers where per-frame metadata isn't (all) in the per-image file:
### ElettraReader distributes energy/temperature/timestamp/... from a shared tracefile via
### post_process_stack, and computed LEEMImg properties (rel_time, binding_energy, ...) aren't
### stored in _meta_list at all -- neither path is exercised by the plain-per-file Elmitec
### ".dat" fixtures the tests above use, which is why the parity gap on these two readers went
### unnoticed (see the module docstring's parity claim).


def test_lazy_metadata_parity_elettra_tracefile(elettra_tif_fnames):
    eager = LEEMStack(elettra_tif_fnames, reader="elettra")
    lazy = LazyLEEMStack(elettra_tif_fnames, reader="elettra")
    assert len(eager) == len(lazy)
    # Plain per-frame meta keys, all sourced from the shared tracefile by post_process_stack.
    assert_field_parity(
        eager,
        lazy,
        ("energy", "temperature", "timestamp", "fov", "objective", "mesh", "emission"),
    )
    # Computed LEEMImg properties -- not stored in _meta_list, must be resolved separately.
    assert_field_parity(
        eager, lazy, ("rel_time", "binding_energy", "kinetic_energy", "pressure", "isotime")
    )


def test_lazy_metadata_elettra_not_constant_or_nan(elettra_tif_fnames):
    # Direct (eager-independent) regression guard: catches the original bug even in the
    # hypothetical case both backends were wrong the same way. Before the fix, lazy 'energy'
    # was the constant sidecar value (the microscope's Start Voltage, not the beamline energy
    # ramp) and 'temperature' was nan for every frame.
    lazy = LazyLEEMStack(elettra_tif_fnames, reader="elettra")
    assert not np.allclose(lazy.energy, lazy.energy[0])
    assert not np.isnan(lazy.temperature).all()
    assert not np.isnan(lazy.rel_time).any()


def test_lazy_metadata_elettra_construction_reads_pixels_once(elettra_tif_fnames, monkeypatch):
    # ElettraReader.parse_metadata (unlike the LEEMReader base's default, which falls back to
    # parse_image and discards "image") never calls tifffile.imread/reads a ".dat" pixel
    # block -- Elettra's metadata is entirely sidecar/header-based. And post_process_stack's
    # tracefile distribution goes straight into _meta_list (LazyImageStack.set_unit), not
    # through per-frame `stack[i]` access. So, like the plain Elmitec case, constructing a
    # lazy Elettra stack must read pixels for exactly one frame (frame 0, to learn shape/dtype
    # for the dask array) no matter how many frames or tracefile columns there are.
    calls = []
    original = ElettraReader.parse_image
    monkeypatch.setattr(
        ElettraReader,
        "parse_image",
        lambda self, source: (calls.append(source), original(self, source))[1],
    )
    LazyLEEMStack(elettra_tif_fnames, reader="elettra")
    assert len(calls) == 1


def test_lazy_metadata_parity_smart_scan():
    eager = LEEMStack(BESSY_SCAN_FNAME, reader="smart")
    lazy = LazyLEEMStack(BESSY_SCAN_FNAME, reader="smart")
    assert len(eager) == len(lazy) > 1
    assert_field_parity(eager, lazy, ("energy", "photon_energy", "timestamp", "width", "height"))
    # 'rel_time' used to raise AttributeError outright on a lazy stack (the reported bug).
    assert_field_parity(eager, lazy, ("rel_time", "binding_energy", "kinetic_energy"))
    # SMART has no temperature/fov in its metadata -- both should still resolve to LEEMImg's
    # own defaults (via the reader/meta_defaults gap-fill), not raise.
    assert np.isnan(lazy.temperature).all()
    assert (lazy.fov == "Unknown FoV").all()


def test_lazy_metadata_smart_construction_reads_one_frame_of_pixels(monkeypatch):
    # SmartReader has no cheap header-only parse_image, so building _meta_list must go
    # through the dedicated parse_metadata (sourcing width/height from _read_smart_dims)
    # instead of falling back to parse_image + discarding "image" -- the fallback would read
    # every frame's pixels just to build metadata, defeating out-of-core loading for exactly
    # the multi-frame scan files SMART is built around.
    calls = []
    import uspy.leem.readers.smart as smart_mod

    original = smart_mod._read_smart_frame
    monkeypatch.setattr(
        smart_mod, "_read_smart_frame", lambda path, index: (calls.append(index), original(path, index))[1]
    )
    lazy = LazyLEEMStack(BESSY_SCAN_FNAME, reader="smart")
    # Only frame 0's pixels are read, to learn shape/dtype for the dask array.
    assert calls == [0]
    assert len(lazy) > 1


def test_lazy_tif_glob_string_loads_per_frame_not_as_3d_stack():
    # A "*.tif" glob string used to be unconditionally routed to _build_from_tiff_stack
    # (single 3D TIFF), which would either crash (tifffile can't open a glob pattern) or,
    # for a real match, silently discard per-frame metadata. It must load one LEEMImg per
    # matched file, same as the reader's own list_files()-based discovery.
    lazy = LazyLEEMStack(ELETTRA_TIF_DIR + "*.tif", reader="elettra")
    assert len(lazy) > 1
    assert lazy.sources is not None
    assert not np.allclose(lazy.energy, lazy.energy[0])


def test_lazy_single_3d_tiff_still_routes_to_tiff_stack(stack_fname):
    # Guards the fallback branch of the _build reordering: a source the reader doesn't
    # recognize (default ElmitecReader.list_files returns [] for a ".tif") that IS an actual
    # single 3D TIFF file must still load as one multi-frame TIFF stack, not raise.
    if not stack_fname.lower().endswith((".tif", ".tiff")):
        pytest.skip("only meaningful for the 3D TIFF fixture")
    lazy = LazyLEEMStack(stack_fname)
    assert lazy.sources is None
    assert len(lazy) > 1


### Loading a lazy stack from per-frame ".tiff" files (glob / folder / explicit list), with
### the *default* Elmitec reader -- as opposed to test_lazy_tif_glob_string_loads_per_frame_
### not_as_3d_stack above, which only exercises this via the Elettra reader's own list_files.
### Must match the eager LEEMStack/ImageStack pixel-for-pixel (see tests/test_leembase.py and
### tests/test_base.py for the eager-side regression tests of the same bug).


def test_lazy_imagestack_glob_matches_eager(tiff_frame_stack):
    eager = ImageStack(tiff_frame_stack["glob"])
    lazy = LazyImageStack(tiff_frame_stack["glob"])
    assert len(lazy) == len(eager)
    assert lazy.sources == eager.sources
    for i in range(len(lazy)):
        np.testing.assert_array_equal(lazy[i].image, eager[i].image)


def test_lazy_leemstack_glob_matches_eager_with_default_reader(tiff_frame_stack):
    # Regression: LazyLEEMStack._read_pixels/_read_metadata used to call the reader's
    # parse_image/parse_metadata unconditionally, bypassing the reader.handles() check that
    # LEEMImg.parse does -- with the default (Elmitec) reader, a ".tiff" frame was misparsed
    # as an Elmitec ".dat" file, producing garbage pixel data with no error.
    eager = LEEMStack(tiff_frame_stack["glob"])
    lazy = LEEMStack(tiff_frame_stack["glob"], lazy=True)
    assert isinstance(lazy, LazyLEEMStack)
    assert len(lazy) == len(eager)
    assert lazy.sources == eager.sources
    for i in range(len(lazy)):
        np.testing.assert_array_equal(lazy[i].image, eager[i].image)
        assert lazy[i].image.shape == tiff_frame_stack["arrays"][i].shape


def test_lazy_leemstack_glob_bare_folder(tiff_frame_stack):
    lazy_dir = LazyLEEMStack(tiff_frame_stack["dir"])
    lazy_glob = LazyLEEMStack(tiff_frame_stack["glob"])
    assert lazy_dir.sources == lazy_glob.sources
    for i in range(len(lazy_dir)):
        np.testing.assert_array_equal(lazy_dir[i].image, lazy_glob[i].image)


def test_lazy_leemstack_mixed_shape_tiff_glob_raises(tiff_stack_glob):
    # Eager/lazy parity for the reported bug: a set of frames that can't form one stack must
    # fail the same way on both backends, not silently build an internally-inconsistent dask
    # array (LazyImageStack._build_from_files previously trusted frame 0's shape for the
    # entire array).
    with pytest.raises(ValueError, match=r"(?i)shape|compatible"):
        LazyLEEMStack(tiff_stack_glob)


def test_to_lazy_preserves_explicit_time_origin(elettra_tif_fnames):
    # ElettraReader.post_process_stack used to unconditionally reset time_origin to frame 0's
    # timestamp, which silently clobbered a caller-supplied time_origin when going through
    # to_lazy() (which reuses the source stack's TimeOrigin by reference). Only an *unset*
    # time_origin should be seeded from frame 0.
    eager = LEEMStack(elettra_tif_fnames, reader="elettra")
    eager.time_origin = 12345.0
    lazy = eager.to_lazy()
    assert lazy.time_origin == 12345.0


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

    # A lazy stack has no per-frame metadata of its own besides `_stack_meta`: a scalar
    # assigned to the stack means "this value, for every frame" -- mirrors the eager
    # DataObjectStack, where the same scalar is set as metadata on every element.
    lazy.energy = 5.0
    np.testing.assert_allclose(lazy.energy, 5.0)


def test_lazy_mismatched_iterable_collision_raises(stack_folder):
    """A mismatched-length iterable can't be distributed one-per-frame, and can't attach as
    stack-level metadata either without shadowing 'energy's per-frame gather."""
    lazy = LazyLEEMStack(stack_folder)
    mismatched = [1.0] * (len(lazy) + 1)
    with pytest.raises(ValueError):
        lazy.energy = mismatched
    # the rejected assignment must not have taken effect
    assert len(lazy.energy) == len(lazy)


### Stack-level scalar metadata (a label, not a per-frame value)


def test_lazy_scalar_broadcasts_to_every_frame(stack_folder):
    """A scalar set on the stack -- even for a genuinely new name -- becomes metadata on
    every frame, not a plain stack-level attribute; a string is not split character-wise."""
    lazy = LazyLEEMStack(stack_folder)
    lazy.label = "my experiment"
    assert (lazy.label == "my experiment").all()
    assert len(lazy.label) == len(lazy)


def test_lazy_scalar_setattr_gets_default_unit(stack_folder):
    """A scalar assigned to a new field must get a default unit too -- units live inline as
    '<attr>_unit' entries in _meta_list on the lazy backend, mirroring
    DataObject.__setattr__ on the eager one. Parity: the same assignment on the eager stack
    must give the same `unit` dict entry."""
    lazy = LazyLEEMStack(stack_folder)
    eager = LEEMStack(stack_folder)

    lazy.brand_new_field = 5.0
    eager.brand_new_field = 5.0

    assert lazy[0].unit["brand_new_field"] == ""
    assert lazy[0].unit["brand_new_field"] == eager[0].unit["brand_new_field"]


def test_lazy_mismatched_iterable_stack_level_metadata(stack_folder):
    """A mismatched-length iterable for a genuinely new name (no per-frame field to shadow)
    still attaches to the stack itself, via `_stack_meta`."""
    lazy = LazyLEEMStack(stack_folder)
    mismatched = ["x"] * (len(lazy) + 1)
    lazy.label = mismatched
    assert lazy.label == mismatched


def test_lazy_energy_offset_scalar_parity_with_eager(stack_folder):
    """Eager/lazy parity for the reported case: `stack.energy_offset = <scalar>` must
    broadcast onto every frame the same way on both backends, and shift `binding_energy`
    identically."""
    eager = LEEMStack(stack_folder)
    lazy = LazyLEEMStack(stack_folder)
    for s in (eager, lazy):
        s.photon_energy = 100.0  # binding_energy needs a real photon_energy to be meaningful

    eager.energy_offset = 3.4934
    lazy.energy_offset = 3.4934

    np.testing.assert_allclose(lazy.energy_offset, eager.energy_offset)
    np.testing.assert_allclose(lazy.binding_energy, eager.binding_energy)


def test_lazy_stack_level_metadata_survives_derivation(stack_folder):
    lazy = LazyLEEMStack(stack_folder)
    mismatched = ["x"] * (len(lazy) + 1)
    lazy.label = mismatched  # stored in _stack_meta (see test above)

    assert lazy[1:4].label == mismatched
    assert (lazy + 1).label == mismatched
    assert lazy.copy().label == mismatched
    assert lazy.average(2).label == mismatched


def test_lazy_unknown_attribute_raises_attributeerror(stack_folder):
    lazy = LazyLEEMStack(stack_folder)
    with pytest.raises(AttributeError):
        _ = lazy.totally_unknown_field


### Missing optional dependency


def test_missing_dask_raises_helpful_importerror(monkeypatch):
    monkeypatch.setattr(lazy_mod, "_HAVE_DASK", False)
    with pytest.raises(ImportError, match="uspy\\[bigdata\\]"):
        lazy_mod._require_dask()
