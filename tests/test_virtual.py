"""Tests the out-of-core (`virtual=True`) backend for LEEMStack/ImageStack.

Historical note: this file used to test a separate `LazyLEEMStack`/`LazyImageStack` backend
(`uspy/dataobject/lazy.py`, backed by a lazy `dask.array`) that reimplemented most of the
eager stack API in parallel and had already drifted from it in several ways (a missing
`average()` guard, shared-by-reference per-frame metadata after copy/slice, `stack.image`
raising, `.align()` silently no-op'ing). That backend has been deleted; `virtual=True` now
builds ordinary, stable `DataObject`s whose pixel data is a `DeferredArray` handle (see
`uspy.dataobject.deferred`) instead of a resident array, so every existing stack operation
(indexing, metadata gathers, arithmetic, `.normalize()`, `.average()`, `.align()`, `.save()`)
works unchanged on a virtual stack -- no separate implementation to keep in sync.

The central correctness claim carried over from the old suite is still *parity*: every value
a virtual stack produces (pixels, per-frame metadata, arithmetic, normalize, average) must
exactly match what the existing eager `LEEMStack`/`ImageStack` produces for the same source
data -- these tests build both from the same fixtures and compare directly, rather than
asserting against hand-picked expected values, so they can't silently paper over a real
divergence.
"""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import numpy as np
import pytest

from uspy.dataobject.image import Image, ImageStack
from uspy.leem.base import LEEMImg, LEEMStack
from uspy.leem.readers.elettra import ElettraReader
from uspy.leem.readers.elmitec import ElmitecReader
from uspy.leem.utility import imgify, stackify

from .conftest import (
    TESTDATA_DIR,
    MCP_IMG_FNAME,
    DC_IMG_FNAME,
    ELETTRA_TIF_DIR,
    BESSY_SCAN_FNAME,
)


def assert_field_parity(eager, virtual, fields):
    """Assert `getattr(eager, field)` and `getattr(virtual, field)` agree element-wise for
    every field in `fields` -- nan-aware (nan == nan here) and string-aware (exact match)."""
    for field in fields:
        eager_values = getattr(eager, field)
        virtual_values = getattr(virtual, field)
        assert len(eager_values) == len(virtual_values), field
        for a, b in zip(eager_values, virtual_values):
            if isinstance(a, str):
                assert a == b, field
            else:
                assert a == b or (np.isnan(a) and np.isnan(b)), field


### Construction / pixel + metadata parity


def test_virtual_pixel_parity(stack_folder):
    eager = LEEMStack(stack_folder)
    virtual = LEEMStack(stack_folder, virtual=True)
    assert len(eager) == len(virtual)
    for i in (0, 1, len(virtual) // 2, len(virtual) - 1):
        assert isinstance(virtual[i], LEEMImg)
        np.testing.assert_array_equal(eager[i].image, virtual[i].image)
        assert eager[i].width == virtual[i].width
        assert eager[i].height == virtual[i].height


def test_virtual_metadata_parity(stack_folder):
    eager = LEEMStack(stack_folder)
    virtual = LEEMStack(stack_folder, virtual=True)
    assert_field_parity(eager, virtual, ("energy", "temperature", "timestamp", "fov"))


def test_virtual_metadata_access_reads_no_pixels(stack_folder, monkeypatch):
    calls = []
    original = ElmitecReader.parse_image
    monkeypatch.setattr(
        ElmitecReader, "parse_image", lambda self, source: (calls.append(source), original(self, source))[1]
    )

    virtual = LEEMStack(stack_folder, virtual=True)
    # Construction is bounded by a directory listing plus one frame's header (to seed the
    # shared TimeOrigin) -- neither goes through parse_image (parse_metadata/parse_pixels do).
    assert len(calls) == 0

    _ = virtual.energy
    _ = virtual.temperature
    _ = virtual.timestamp
    # Metadata is resolved once per frame and cached permanently -- gathering it must not
    # read any pixels.
    assert len(calls) == 0


### Metadata parity for readers where per-frame metadata isn't (all) in the per-image file:
### ElettraReader distributes energy/temperature/timestamp/... from a shared tracefile via
### post_process_stack, and computed LEEMImg properties (rel_time, binding_energy, ...) are
### computed properties, not stored metadata -- neither path is exercised by the plain
### per-file Elmitec ".dat" fixtures above, which is why a parity gap on these two readers
### could go unnoticed.


def test_virtual_metadata_parity_elettra_tracefile(elettra_tif_fnames):
    eager = LEEMStack(elettra_tif_fnames, reader="elettra")
    virtual = LEEMStack(elettra_tif_fnames, reader="elettra", virtual=True)
    assert len(eager) == len(virtual)
    # Plain per-frame meta keys, all sourced from the shared tracefile by post_process_stack.
    assert_field_parity(
        eager,
        virtual,
        ("energy", "temperature", "timestamp", "fov", "objective", "mesh", "emission"),
    )
    # Computed LEEMImg properties.
    assert_field_parity(
        eager, virtual, ("rel_time", "binding_energy", "kinetic_energy", "pressure", "isotime")
    )


def test_virtual_metadata_elettra_not_constant_or_nan(elettra_tif_fnames):
    # Direct (eager-independent) regression guard: catches the original bug even in the
    # hypothetical case both backends were wrong the same way.
    virtual = LEEMStack(elettra_tif_fnames, reader="elettra", virtual=True)
    assert not np.allclose(virtual.energy, virtual.energy[0])
    assert not np.isnan(virtual.temperature).all()
    assert not np.isnan(virtual.rel_time).any()


def test_virtual_metadata_elettra_construction_reads_no_pixels(elettra_tif_fnames, monkeypatch):
    calls = []
    original = ElettraReader.parse_image
    monkeypatch.setattr(
        ElettraReader,
        "parse_image",
        lambda self, source: (calls.append(source), original(self, source))[1],
    )
    LEEMStack(elettra_tif_fnames, reader="elettra", virtual=True)
    assert len(calls) == 0


def test_virtual_metadata_parity_smart_scan():
    eager = LEEMStack(BESSY_SCAN_FNAME, reader="smart")
    virtual = LEEMStack(BESSY_SCAN_FNAME, reader="smart", virtual=True)
    assert len(eager) == len(virtual) > 1
    assert_field_parity(eager, virtual, ("energy", "photon_energy", "timestamp", "width", "height"))
    assert_field_parity(eager, virtual, ("rel_time", "binding_energy", "kinetic_energy"))
    # SMART has no temperature/fov in its metadata -- both should still resolve to LEEMImg's
    # own defaults (via the reader/meta_defaults gap-fill), not raise.
    assert np.isnan(virtual.temperature).all()
    assert (virtual.fov == "Unknown FoV").all()


def test_virtual_tif_glob_string_loads_per_frame_not_as_3d_stack():
    # A "*.tif" glob string must load one LEEMImg per matched file, same as the reader's own
    # list_files()-based discovery -- not get misrouted to a single 3D-TIFF interpretation.
    virtual = LEEMStack(ELETTRA_TIF_DIR + "*.tif", reader="elettra", virtual=True)
    assert len(virtual) > 1
    assert virtual.sources is not None
    assert not np.allclose(virtual.energy, virtual.energy[0])


def test_virtual_single_3d_tiff_still_routes_to_tiff_stack(stack_fname):
    if not stack_fname.lower().endswith((".tif", ".tiff")):
        pytest.skip("only meaningful for the 3D TIFF fixture")
    virtual = LEEMStack(stack_fname, virtual=True)
    assert len(virtual) > 1


### Loading a virtual stack from per-frame ".tiff" files (glob / folder / explicit list)


def test_virtual_imagestack_glob_matches_eager(tiff_frame_stack):
    eager = ImageStack(tiff_frame_stack["glob"])
    virtual = ImageStack(tiff_frame_stack["glob"], virtual=True)
    assert len(virtual) == len(eager)
    assert virtual.sources == eager.sources
    for i in range(len(virtual)):
        np.testing.assert_array_equal(virtual[i].image, eager[i].image)


def test_virtual_leemstack_glob_matches_eager_with_default_reader(tiff_frame_stack):
    # Regression: a virtual LEEMImg's deferred pixel handle must go through the same
    # reader.handles() gate LEEMImg.parse does -- with the default (Elmitec) reader, a
    # ".tiff" frame must not be misparsed as an Elmitec ".dat" file.
    eager = LEEMStack(tiff_frame_stack["glob"])
    virtual = LEEMStack(tiff_frame_stack["glob"], virtual=True)
    assert len(virtual) == len(eager)
    assert virtual.sources == eager.sources
    for i in range(len(virtual)):
        np.testing.assert_array_equal(virtual[i].image, eager[i].image)
        assert virtual[i].image.shape == tiff_frame_stack["arrays"][i].shape


def test_virtual_leemstack_glob_bare_folder(tiff_frame_stack):
    virtual_dir = LEEMStack(tiff_frame_stack["dir"], virtual=True)
    virtual_glob = LEEMStack(tiff_frame_stack["glob"], virtual=True)
    assert virtual_dir.sources == virtual_glob.sources
    for i in range(len(virtual_dir)):
        np.testing.assert_array_equal(virtual_dir[i].image, virtual_glob[i].image)


def test_virtual_leemstack_mixed_shape_tiff_glob_raises_at_first_bulk_use(tiff_stack_glob):
    # Eager/virtual parity for the reported bug: a set of frames that can't form one stack
    # must eventually fail the same way on both backends. Unlike the eager path (which checks
    # pairwise compatibility as it constructs, since every element is loaded anyway), a virtual
    # stack skips that check at construction to stay zero-I/O -- so construction itself
    # succeeds, and the mismatch is only caught at first bulk use (check_compatible(), or
    # anything that calls it: virtual = False, average(), a data-key gather, ...).
    virtual = LEEMStack(tiff_stack_glob, virtual=True)  # must not raise
    with pytest.raises(ValueError, match=r"(?i)shape|compatible"):
        virtual.check_compatible()
    with pytest.raises(ValueError, match=r"(?i)shape|compatible"):
        virtual.virtual = False


def test_virtual_preserves_explicit_time_origin(elettra_tif_fnames):
    # An explicitly-passed time_origin must not be clobbered by post_process_stack's tracefile
    # distribution or by the "seed from frame 0" default -- only an *unset* time_origin is
    # seeded that way.
    virtual = LEEMStack(
        elettra_tif_fnames, reader="elettra", virtual=True, time_origin=12345.0
    )
    assert virtual.time_origin == 12345.0


def test_virtual_from_3d_tiff(stack_fname):
    if not stack_fname.lower().endswith((".tif", ".tiff")):
        pytest.skip("only meaningful for the 3D TIFF fixture")
    eager = LEEMStack(stack_fname)
    virtual = LEEMStack(stack_fname, virtual=True)
    assert len(eager) == len(virtual)
    for i in (0, len(virtual) - 1):
        np.testing.assert_array_equal(eager[i].image, virtual[i].image)


def test_virtual_generic_image_stack(stack_fname):
    if not stack_fname.lower().endswith((".tif", ".tiff")):
        pytest.skip("only meaningful for the 3D TIFF fixture")
    eager = ImageStack(stack_fname)
    virtual = ImageStack(stack_fname, virtual=True)
    assert len(eager) == len(virtual)
    for i in (0, len(virtual) - 1):
        np.testing.assert_allclose(eager[i].image, virtual[i].image)


def test_virtual_single_frame_stack():
    fname = TESTDATA_DIR + "bremen.dat"
    eager = LEEMStack([fname])
    virtual = LEEMStack([fname], virtual=True)
    assert len(virtual) == 1
    np.testing.assert_array_equal(eager[0].image, virtual[0].image)


def test_virtual_empty_source_raises():
    with pytest.raises(ValueError):
        LEEMStack([], virtual=True)


### Indexing / slicing


def test_virtual_slicing_propagates_type_and_metadata(stack_folder):
    virtual = LEEMStack(stack_folder, virtual=True)
    sliced = virtual[2:5]
    assert type(sliced) is LEEMStack
    assert sliced.virtual is True
    assert len(sliced) == 3
    np.testing.assert_array_equal(sliced[0].image, virtual[2].image)
    assert sliced.energy[0] == virtual.energy[2] or (
        np.isnan(sliced.energy[0]) and np.isnan(virtual.energy[2])
    )


def test_virtual_negative_and_out_of_range_indexing(stack_folder):
    virtual = LEEMStack(stack_folder, virtual=True)
    np.testing.assert_array_equal(virtual[-1].image, virtual[len(virtual) - 1].image)
    with pytest.raises(IndexError):
        _ = virtual[len(virtual)]
    with pytest.raises(TypeError):
        _ = virtual["nonsense"]


def test_virtual_getitem_is_stable_not_reparsed(stack_folder):
    """A virtual stack's elements are real, stable DataObjects now -- repeated access returns
    the *same* object, not a fresh re-parse (the old lazy/virtual behavior)."""
    virtual = LEEMStack(stack_folder, virtual=True)
    assert virtual[0] is virtual[0]


### Arithmetic parity -- composes, stays virtual


def test_virtual_arithmetic_parity(stack_folder):
    eager = LEEMStack(stack_folder)
    virtual = LEEMStack(stack_folder, virtual=True)

    e_expr = (eager * 2) - 3
    v_expr = (virtual * 2) - 3
    assert v_expr.virtual is True
    for i in (0, len(virtual) - 1):
        np.testing.assert_allclose(e_expr[i].image, v_expr[i].image)

    e_div = eager / 2 + 1
    v_div = virtual / 2 + 1
    for i in (0,):
        np.testing.assert_allclose(e_div[i].image, v_div[i].image)


def test_virtual_arithmetic_with_single_image(stack_folder):
    virtual = LEEMStack(stack_folder, virtual=True)
    other = imgify(MCP_IMG_FNAME)
    result = virtual - other
    np.testing.assert_allclose(result[0].image, virtual[0].image - other.image)


def test_virtual_inplace_arithmetic_persists_and_stays_virtual(stack_folder):
    """In-place arithmetic on a virtual stack used to raise outright (elements were
    throwaway); it now composes into each element's deferred data and persists."""
    eager = LEEMStack(stack_folder)
    virtual = LEEMStack(stack_folder, virtual=True)

    virtual += 1
    assert virtual.virtual is True
    np.testing.assert_allclose(virtual[0].image, eager[0].image + 1)


### normalize() parity -- composes, stays virtual, no explicit mcp required


@pytest.mark.parametrize("dark_counts_kind", ["path", "image", "int"])
def test_virtual_normalize_parity(stack_folder, dark_counts_kind):
    mcp = imgify(MCP_IMG_FNAME)
    dark_counts = {
        "path": DC_IMG_FNAME,
        "image": imgify(DC_IMG_FNAME),
        "int": 100,
    }[dark_counts_kind]

    eager = LEEMStack(stack_folder)
    virtual = LEEMStack(stack_folder, virtual=True)

    e_norm = eager.normalize(mcp=mcp, dark_counts=dark_counts)
    v_norm = virtual.normalize(mcp=mcp, dark_counts=dark_counts)
    assert v_norm.virtual is True
    for i in (0, len(virtual) // 2, len(virtual) - 1):
        np.testing.assert_allclose(e_norm[i].image, v_norm[i].image, atol=1e-4)


def test_virtual_img_normalize_without_explicit_mcp_matches_eager(stack_folder):
    """Formerly a lazy-backend-only limitation (`LazyLEEMStack.normalize()` required an
    explicit mcp, unlike `LEEMImg.normalize()` which falls back to `self.mcp`); a single
    deferred `LEEMImg.normalize()` supports the same fallback as an eager one. (Note:
    `LEEMStack.normalize(mcp=None)` itself has a separate, pre-existing limitation --
    unrelated to virtual/eager -- so this is checked at the single-image level.)"""
    eager = LEEMStack(stack_folder)
    virtual = LEEMStack(stack_folder, virtual=True)
    mcp = imgify(MCP_IMG_FNAME)

    e_img = eager[0].normalize(mcp=mcp)
    v_img = virtual[0].normalize(mcp=mcp)
    assert e_img.mcp == v_img.mcp == mcp.source

    e_norm = e_img.normalize()  # falls back to img.mcp
    v_norm = v_img.normalize()
    np.testing.assert_allclose(e_norm.image, v_norm.image, atol=1e-4)


### align() parity -- warp composes, stays virtual (regression: used to materialize the
### whole stack via LEEMImg.warp's `self.image = cv.warpPerspective(...)`, see CLAUDE.md)


def test_virtual_align_stays_virtual(stack_folder):
    virtual = LEEMStack(stack_folder, virtual=True)[:5]
    aligned = virtual.align()  # default mask=True -- also exercises ROI.apply's compose path
    assert aligned.virtual is True
    assert all(element.is_virtual for element in aligned.elements)


def test_virtual_align_does_not_trigger_materialization_warning(stack_folder, recwarn):
    virtual = LEEMStack(stack_folder, virtual=True)[:5]
    virtual.align()
    materializing = [w for w in recwarn.list if "deferred data with a resident" in str(w.message)]
    assert not materializing


def test_virtual_align_inplace_stays_virtual(stack_folder):
    virtual = LEEMStack(stack_folder, virtual=True)[:4]
    result = virtual.align(inplace=True)
    assert result is virtual
    assert virtual.virtual is True


def test_virtual_align_pixel_and_warp_matrix_parity(stack_folder):
    eager = LEEMStack(stack_folder)[:5]
    virtual = LEEMStack(stack_folder, virtual=True)[:5]

    e_aligned = eager.align()
    v_aligned = virtual.align()

    np.testing.assert_allclose(e_aligned.warp_matrix, v_aligned.warp_matrix, atol=1e-3)
    for i in range(len(v_aligned)):
        np.testing.assert_allclose(e_aligned[i].image, v_aligned[i].image, atol=1e-2)


### LEEMImg.warp / ROI.apply / Image.filter -- compose onto a deferred element instead of
### materializing it (the individual pieces `align()`/`make_video()` are built from)


def test_virtual_warp_composes_and_matches_eager(stack_folder):
    eager = LEEMStack(stack_folder)[0]
    virtual = LEEMStack(stack_folder, virtual=True)[0]
    warp = np.array([[1, 0, 3], [0, 1, -2], [0, 0, 1]], dtype=np.float64)

    e_warped = eager.warp(warp, inplace=False)
    v_warped = virtual.warp(warp, inplace=False)
    assert v_warped.is_virtual
    np.testing.assert_allclose(e_warped.image, v_warped.image, atol=1e-3)

    v_inplace = LEEMStack(stack_folder, virtual=True)[1]
    v_inplace.warp(warp, inplace=True)
    assert v_inplace.is_virtual


def test_virtual_roi_apply_stays_virtual_and_matches_eager(stack_folder):
    from uspy.roi import ROI

    eager = LEEMStack(stack_folder)[0]
    virtual = LEEMStack(stack_folder, virtual=True)[0]
    roi = ROI.circle(x0=eager.width // 2, y0=eager.height // 2, radius=eager.width // 4)

    e_masked = roi.apply(eager)
    v_masked = roi.apply(virtual)
    assert v_masked.is_virtual

    np.testing.assert_array_equal(
        np.ma.getmaskarray(e_masked.image), np.ma.getmaskarray(v_masked.image)
    )
    np.testing.assert_allclose(
        np.ma.filled(e_masked.image, 0), np.ma.filled(v_masked.image, 0), atol=1e-3
    )


def test_virtual_image_filter_stays_virtual_and_matches_eager(stack_folder):
    eager = LEEMStack(stack_folder)[0]
    virtual = LEEMStack(stack_folder, virtual=True)[0]

    e_filtered = eager.filter(method="gaussian")
    v_filtered = virtual.filter(method="gaussian")
    assert v_filtered.is_virtual
    np.testing.assert_allclose(e_filtered.image, v_filtered.image, atol=1e-3)


def test_virtual_image_filter_unknown_method_raises_without_reading_pixels(
    stack_folder, monkeypatch
):
    """Method/kwarg validation must stay eager -- an unknown method should raise immediately,
    not get deferred into a DeferredArray.map() that only fails on first pixel read."""
    from uspy.leem.readers.elmitec import ElmitecReader

    calls = []
    original = ElmitecReader.parse_pixels
    monkeypatch.setattr(
        ElmitecReader,
        "parse_pixels",
        lambda self, source: (calls.append(source), original(self, source))[1],
    )
    virtual = LEEMStack(stack_folder, virtual=True)[0]
    with pytest.raises(ValueError):
        virtual.filter(method="nonsense")
    assert calls == []


### average() / filter() parity -- stay virtual


def test_virtual_average_parity(stack_folder):
    eager = LEEMStack(stack_folder)
    virtual = LEEMStack(stack_folder, virtual=True)
    n = len(virtual) - (len(virtual) % 2)  # ensure even length
    eager, virtual = eager[:n], virtual[:n]

    e_avg = eager.average(2)
    v_avg = virtual.average(2)
    assert len(e_avg) == len(v_avg)
    assert v_avg.virtual is True
    assert all(element.is_virtual for element in v_avg.elements)
    for i in (0, len(v_avg) - 1):
        np.testing.assert_allclose(e_avg[i].image, v_avg[i].image, atol=1e-3)


def test_virtual_stack_filter_stays_virtual(stack_folder):
    virtual = LEEMStack(stack_folder, virtual=True)[:3]
    filtered = virtual.filter(method="gaussian")
    assert filtered.virtual is True
    assert all(element.is_virtual for element in filtered.elements)


def test_virtual_average_bad_divisor_raises(stack_folder):
    virtual = LEEMStack(stack_folder, virtual=True)
    with pytest.raises(ValueError):
        virtual.average(len(virtual) + 1)  # never evenly divides
    with pytest.raises(ValueError):
        virtual.average(0)
    with pytest.raises(ValueError):
        virtual.average(-2)


### virtual <-> eager materialization roundtrip


def test_virtual_materialize_roundtrip(stack_folder):
    eager = LEEMStack(stack_folder)
    virtual = LEEMStack(stack_folder, virtual=True)
    virtual.virtual = False
    assert not virtual.virtual
    assert len(virtual) == len(eager)
    for i in (0, len(eager) - 1):
        np.testing.assert_array_equal(virtual[i].image, eager[i].image)


def test_eager_stack_can_go_virtual(stack_folder):
    eager = LEEMStack(stack_folder)
    eager.virtual = True
    assert eager.virtual
    assert all(element.is_virtual for element in eager.elements)


def test_materialize_then_defer_discards_composed_processing(stack_folder):
    """Documents a deliberate, *accepted* contract, not a bug: `defer()`/`.virtual = True`
    always re-derives data from `source` from scratch (see `DataObject.defer`'s docstring), so
    a materialize -> defer round trip on a stack holding *composed, unsaved* processing (e.g. a
    `.normalize()` result) discards that processing and comes back as the raw source data --
    exactly as if the intermediate result had never been computed. This is intentional: saving
    (`.save()`) before re-deferring is the way to keep composed results."""
    mcp = imgify(MCP_IMG_FNAME)
    raw = LEEMStack(stack_folder, virtual=True)

    v_norm = raw.normalize(mcp=mcp)  # composes onto the DeferredArray, still virtual
    assert v_norm.virtual is True

    v_norm.virtual = False  # materializes the *normalized* pixels
    normalized_pixels = v_norm[0].image.copy()
    assert not np.allclose(normalized_pixels, raw[0].image)

    v_norm.virtual = True  # re-derives from `source` -- the normalization is gone
    np.testing.assert_array_equal(v_norm[0].image, raw[0].image)


def test_leemstack_virtual_kwarg_dispatch(stack_folder):
    stack = LEEMStack(stack_folder, virtual=True)
    assert type(stack) is LEEMStack
    assert stack.virtual is True


### Slicing a virtual stack is a view (shares elements with the parent); a virtual<->eager
### transition on the slice must not reach back into the parent's shared elements. Regression
### for: `sub = virt[:5]; sub.virtual = False` used to mutate `virt`'s own frames in place,
### leaving `virt.virtual` reporting `True` while its first frames were actually resident.


def test_virtual_stack_slice_materialize_does_not_affect_parent(stack_folder):
    virt = LEEMStack(stack_folder, virtual=True)
    sub = virt[:2]

    assert sub[0] is virt[0]  # slicing shares element objects with the parent
    sub.virtual = False

    assert virt.virtual is True
    assert all(element.is_virtual for element in virt.elements)
    assert sub[0] is not virt[0]
    assert sub.virtual is False


def test_eager_stack_slice_defer_does_not_affect_parent(stack_folder):
    eager = LEEMStack(stack_folder)
    sub = eager[:2]

    assert sub[0] is eager[0]
    sub.virtual = True

    assert eager.virtual is False
    assert not any(element.is_virtual for element in eager.elements)
    assert sub[0] is not eager[0]
    assert sub.virtual is True


def test_virtual_stack_normalize_inplace_after_slice_materialization(stack_folder):
    """Regression for the concrete data-corruption case: before elements were rebound instead
    of mutated in place, a virtual stack whose first frames had been pinned resident via a
    materialized slice would silently *discard* an in-place normalize on those frames --
    `LEEMStack.normalize(inplace=True)` trusted the (stale) `self.virtual` flag and rebuilt
    each resident result from `source`, throwing away the just-computed normalization."""
    mcp = imgify(MCP_IMG_FNAME)

    eager_expected = LEEMStack(stack_folder).normalize(mcp=mcp)

    virt = LEEMStack(stack_folder, virtual=True)
    sub = virt[:2]
    sub.virtual = False  # used to pin virt's first frames in place

    virt.normalize(mcp=mcp, inplace=True)

    for i in range(len(virt)):
        assert virt[i].mcp == mcp.source
        np.testing.assert_allclose(virt[i].image, eager_expected[i].image, atol=1e-4)


def test_stackify_virtual(stack_folder):
    stack = stackify(stack_folder, virtual=True)
    assert type(stack) is LEEMStack
    assert stack.virtual is True
    # passthrough for an already-virtual stack
    assert stackify(stack, virtual=True) is stack


def test_imagestack_virtual_kwarg_dispatch(stack_fname):
    if not stack_fname.lower().endswith((".tif", ".tiff")):
        pytest.skip("only meaningful for the 3D TIFF fixture")
    stack = ImageStack(stack_fname, virtual=True)
    assert type(stack) is ImageStack
    # Known, pre-existing limitation, not a regression: `ImageStack._split_source` reads a
    # single 3D TIFF into memory in full up front (`tifffile.imread`) to split it into
    # per-frame arrays, and `Image.parse_deferred`'s ndarray branch has no cheaper deferred
    # path than the array already in hand -- so every element here actually ends up resident.
    # `stack.virtual` (derived from the elements, see `DataObjectStack.virtual`) now reports
    # that honestly instead of rubber-stamping the `virtual=True` request.
    assert stack.virtual is False
    assert not any(element.is_virtual for element in stack.elements)
    eager = ImageStack(stack_fname)
    assert type(eager) is ImageStack
    assert eager.virtual is False


### Persistence: save() to tif/hdf5, source untouched


def test_virtual_save_tif_roundtrip(stack_folder, tmp_path):
    import tifffile

    virtual = LEEMStack(stack_folder, virtual=True)[:3]
    target = str(tmp_path / "stack.tif")
    virtual.save(target)
    arr = tifffile.imread(target)
    assert arr.shape[0] == 3
    np.testing.assert_allclose(arr[0], virtual[0].image)
    np.testing.assert_allclose(arr[2], virtual[2].image)


def test_virtual_save_does_not_modify_source(stack_folder, tmp_path):
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

    virtual = LEEMStack(stack_folder, virtual=True)
    virtual.save(str(tmp_path / "stack.h5"))
    (virtual * 2).save(str(tmp_path / "stack2.h5"))

    after = file_hash(full_path)
    assert before == after


def test_virtual_dumps_and_pickle_roundtrip(stack_folder):
    """Formerly a lazy-backend-only limitation (`dumps()`/pickling raised outright on a lazy
    stack); a virtual stack pickles like any other DataObjectStack, staying virtual."""
    virtual = LEEMStack(stack_folder, virtual=True)
    blob = virtual.dumps(use_pickle=True)
    restored = LEEMStack.__new__(LEEMStack)
    import pickle

    restored = pickle.loads(blob)
    assert restored.virtual is True
    np.testing.assert_array_equal(restored[0].image, virtual[0].image)


### Operations that used to be unsupported on the old lazy backend, now work


def test_virtual_setitem_delitem_extend_filter_work(stack_folder):
    virtual = LEEMStack(stack_folder, virtual=True)
    n = len(virtual)

    virtual[0] = virtual[1]
    assert virtual[0].source == virtual[1].source

    del virtual[0]
    assert len(virtual) == n - 1

    virtual.extend(LEEMStack(stack_folder, virtual=True)[0])
    assert len(virtual) == n

    filtered = virtual.filter()
    assert len(filtered) == len(virtual)


def test_virtual_setter_now_works_in_both_directions(stack_folder):
    virtual = LEEMStack(stack_folder, virtual=True)
    assert virtual.virtual is True
    virtual.virtual = False
    assert virtual.virtual is False
    virtual.virtual = True
    assert virtual.virtual is True


def test_virtual_bulk_metadata_assignment(stack_folder):
    virtual = LEEMStack(stack_folder, virtual=True)
    new_energy = np.linspace(0, 1, len(virtual))
    virtual.energy = new_energy
    np.testing.assert_allclose(virtual.energy, new_energy)

    virtual.energy = 5.0
    np.testing.assert_allclose(virtual.energy, 5.0)


def test_virtual_mismatched_iterable_collision_raises(stack_folder):
    virtual = LEEMStack(stack_folder, virtual=True)
    mismatched = [1.0] * (len(virtual) + 1)
    with pytest.raises(ValueError):
        virtual.energy = mismatched
    assert len(virtual.energy) == len(virtual)


### Stack-level scalar metadata (a label, not a per-frame value)


def test_virtual_scalar_broadcasts_to_every_frame(stack_folder):
    virtual = LEEMStack(stack_folder, virtual=True)
    virtual.label = "my experiment"
    assert (virtual.label == "my experiment").all()
    assert len(virtual.label) == len(virtual)


def test_virtual_scalar_setattr_gets_default_unit(stack_folder):
    virtual = LEEMStack(stack_folder, virtual=True)
    eager = LEEMStack(stack_folder)

    virtual.brand_new_field = 5.0
    eager.brand_new_field = 5.0

    assert virtual[0].unit["brand_new_field"] == ""
    assert virtual[0].unit["brand_new_field"] == eager[0].unit["brand_new_field"]


def test_virtual_mismatched_iterable_stack_level_metadata(stack_folder):
    virtual = LEEMStack(stack_folder, virtual=True)
    mismatched = ["x"] * (len(virtual) + 1)
    virtual.label = mismatched
    assert virtual.label == mismatched


def test_virtual_energy_offset_scalar_parity_with_eager(stack_folder):
    eager = LEEMStack(stack_folder)
    virtual = LEEMStack(stack_folder, virtual=True)
    for s in (eager, virtual):
        s.photon_energy = 100.0  # binding_energy needs a real photon_energy to be meaningful

    eager.energy_offset = 3.4934
    virtual.energy_offset = 3.4934

    np.testing.assert_allclose(virtual.energy_offset, eager.energy_offset)
    np.testing.assert_allclose(virtual.binding_energy, eager.binding_energy)


def test_virtual_unknown_attribute_raises_attributeerror(stack_folder):
    virtual = LEEMStack(stack_folder, virtual=True)
    with pytest.raises(AttributeError):
        _ = virtual.totally_unknown_field


### copy() independence -- the aliasing bug the old lazy backend had


def test_virtual_copy_does_not_share_metadata(stack_folder):
    virtual = LEEMStack(stack_folder, virtual=True)
    other = virtual.copy()
    other[0].energy = 999.0
    assert virtual[0].energy != 999.0


### stack.image on a virtual stack: works (used to raise on the old lazy backend), warns first


def test_virtual_stack_image_gather_warns_and_materializes(stack_folder):
    virtual = LEEMStack(stack_folder, virtual=True)
    with pytest.warns(UserWarning):
        images = virtual.image
    assert images.shape[0] == len(virtual)
