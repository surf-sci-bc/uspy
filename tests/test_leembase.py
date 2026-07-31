"""Tests the uspy.leem.base module."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import warnings

import cv2 as cv
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from uspy.leem import base
from uspy.leem.base import LEEMImg, LEEMStack, IVCurve, TimeOrigin
from uspy.leem.readers.elmitec import ElmitecReader
from uspy.derived import IntensityLine
from uspy.roi import ROI

from .conftest import (
    MCP_IMG_FNAME,
    BREMEN_IMG_FNAME,
    BREMEN2_IMG_FNAME,
    GCU_STACK_DIR,
    RUO2_STACK_DIR,
    dark_counts,
    same_or_nan,
    IMG_FNAMES_COMPATIBLE,
    STACK_FNAMES,
    STACK_INCOMPATIBLE_IMG_FNAME,
    single_stack,
)

### Image loading


# @pytest.mark.parametrize("nolazy", [True, False])
def test_img_constructor(img_fname):
    img = base.LEEMImg(img_fname)
    assert img.source == img_fname
    assert isinstance(img.energy, (int, float))
    assert isinstance(img.meta, dict)
    assert isinstance(img.image, np.ndarray)
    assert img.image.shape == (img.height, img.width)


def test_img_constructor_array(img):
    img2 = base.LEEMImg(img.image)
    assert img2.image.shape == img.image.shape


# @pytest.mark.skip(reason="Takes long and seems unproblematic")
@pytest.mark.slow
def test_img_copying(img):
    img2 = img.copy()
    assert img2 == img
    assert img2 is not img


def test_img_filter_contrast_carries_over_state(img):
    # LEEMImg carries extra per-instance state beyond base Image (_reader, _time_origin,
    # default_mask) that DataObject._with_data (which filter()/contrast() are built on)
    # intentionally shares by reference rather than deep-copying, unlike copy(). Lock that
    # contract in, alongside the basic "still a correctly-transformed, correctly-typed
    # LEEMImg" check -- this is what a first, buggy _with_data got silently wrong (it never
    # actually substituted the new pixel array at all).
    filtered = img.filter("gaussian")
    contrasted = img.contrast("equal")

    for result in (filtered, contrasted):
        assert isinstance(result, LEEMImg)
        assert result is not img
        assert not np.array_equal(result.image, img.image)
        assert result.data is not img.data
        assert result.meta is not img.meta
        assert same_or_nan(result.energy, img.energy)
        assert same_or_nan(result.temperature, img.temperature)
        # documented _with_data contract: shared, not deep-copied, by reference
        assert result._reader is img._reader
        assert result._time_origin is img._time_origin


def test_imge_constructor_nonsense():
    with pytest.raises(FileNotFoundError):
        _ = base.LEEMImg("nonsense")
    with pytest.raises(FileNotFoundError):
        _ = base.LEEMImg("nonsense.dat")
    # img = base.LEEMImg("nonsense.dat")
    # with pytest.raises(FileNotFoundError):
    #    assert img.image


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_constructor(stack_fname, virtual):
    stack = base.LEEMStack(stack_fname, virtual=virtual)
    assert isinstance(stack[0], base.LEEMImg)
    assert stack[0].image.shape == stack[-1].image.shape


@pytest.mark.slow
def test_stack_constructor_globbing(stack_folder):
    stack1 = base.LEEMStack(stack_folder, virtual=True)
    stack2 = base.LEEMStack(stack_folder + "/*.dat", virtual=True)
    stack3 = base.LEEMStack(stack_folder + "/", virtual=True)
    assert len(stack1) == len(stack2) == len(stack3)
    for i in [0, -1]:
        assert stack1[i] == stack2[i]
        assert stack1[i] == stack3[i]
    stack3 = base.LEEMStack(IMG_FNAMES_COMPATIBLE[0])
    assert len(stack3) == 1


def test_stack_constructor_nonsense():
    with pytest.raises(ValueError):
        _ = base.LEEMStack("nonsense")
    with pytest.raises(FileNotFoundError):
        _ = base.LEEMStack(["nonsense.dat"])


### Loading a stack from per-frame ".tiff" files (glob / folder), as opposed to a single 3D
### TIFF. Reproduces a reported bug where globbing per-frame TIFFs silently misread frames
### (via tifffile's FileSequence machinery force-fitting every file into frame 0's shape) and
### lost each element's `source`, plus a lazy/eager asymmetry (glob raised on the lazy path).


def test_leemstack_glob_loads_one_frame_per_file_in_natural_order(tiff_frame_stack):
    stack = LEEMStack(tiff_frame_stack["glob"])
    assert len(stack) == len(tiff_frame_stack["paths"])
    assert stack.sources == tiff_frame_stack["paths"]
    for element, expected in zip(stack, tiff_frame_stack["arrays"]):
        assert element.image.shape == expected.shape
        assert_array_equal(element.image, expected)


def test_leemstack_glob_folder_and_explicit_list_agree(tiff_frame_stack):
    from_glob = LEEMStack(tiff_frame_stack["glob"])
    from_dir = LEEMStack(tiff_frame_stack["dir"])
    from_dir_trailing_slash = LEEMStack(tiff_frame_stack["dir"] + "/")
    from_list = LEEMStack(tiff_frame_stack["paths"])

    assert from_glob.sources == from_dir.sources == from_dir_trailing_slash.sources
    assert from_glob.sources == from_list.sources
    for a, b in zip(from_glob, from_list):
        assert_array_equal(a.image, b.image)


def test_leemstack_glob_can_virtualize(tiff_frame_stack):
    # Before the fix, a glob-built stack lost every element's `source`, so virtualizing
    # raised "Can't virtualize stack without source information."
    stack = LEEMStack(tiff_frame_stack["glob"])
    stack.defer()
    assert stack.virtual
    assert len(stack) == len(tiff_frame_stack["paths"])
    stack.materialize()
    assert not stack.virtual


def test_leemstack_tiff_glob_loads(tiff_stack_glob):
    # testdata/bremen/tiff_stack/ holds real per-frame LEED ".tiff" exports, all one shape
    # (600, 624) -- a plain smoke test that the real-world fixture still loads as one stack.
    stack = base.LEEMStack(tiff_stack_glob)
    assert len(stack) == 3
    assert all(element.image.shape == (600, 624) for element in stack)


def test_leemstack_mixed_shape_tiff_glob_raises(mixed_shape_tiff_stack):
    # The original bug report: a folder/glob of per-frame ".tiff" files with two different
    # pixel shapes. Before the fix, this silently loaded all frames with several of them
    # holding another file's pixels reshaped into the wrong dimensions.
    with pytest.raises(ValueError, match=r"(?i)shape|compatible"):
        base.LEEMStack(mixed_shape_tiff_stack["glob"])


def test_leemstack_mixed_shape_tiff_frames_load_correctly_when_selected(mixed_shape_tiff_stack):
    paths = mixed_shape_tiff_stack["paths"]
    arrays = mixed_shape_tiff_stack["arrays"]
    shapes = mixed_shape_tiff_stack["shapes"]
    reference_shape = shapes[0]
    same_shape = [p for p, s in zip(paths, shapes) if s == reference_shape]
    assert len(same_shape) >= 2, "fixture must contain multiple same-shape frames"

    stack = LEEMStack(same_shape)
    for fname, element in zip(same_shape, stack):
        expected = arrays[paths.index(fname)]
        assert element.image.shape == expected.shape
        assert_array_equal(element.image, expected)

    # Guard the corruption mechanism itself: no loaded frame should equal a *different*
    # shape's data reshaped into the reference shape.
    other_arrays = [a for a, s in zip(arrays, shapes) if s != reference_shape]
    for element in stack:
        for other in other_arrays:
            reshaped = other.reshape(reference_shape)
            assert not np.array_equal(element.image, reshaped)


def test_leemstack_incompatible_elements_raise_instead_of_silently_dropping():
    # DataObjectStack._construct used to build an unraised Warning(...) instance instead of
    # calling warnings.warn(...), so an incompatible frame was silently skipped -- the
    # resulting stack was shorter than the input list with no indication why.
    sources = IMG_FNAMES_COMPATIBLE + [STACK_INCOMPATIBLE_IMG_FNAME]
    with pytest.raises(ValueError, match=STACK_INCOMPATIBLE_IMG_FNAME.split("/")[-1]):
        base.LEEMStack(sources)


def test_img_attr_types(img):
    if ".dat" in img.source:
        assert img.energy > -6
        # assert img.rel_time > 0
    else:
        assert np.isnan(img.energy)
        assert np.isnan(img.rel_time)
    assert np.isnan(img.temperature) or img.temperature > -280
    assert isinstance(img.timestamp, (int, float))


@pytest.mark.parametrize("inplace", [True, False])
def test_img_normalize(inplace, normed_img):
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    ref = img
    mcp = MCP_IMG_FNAME
    if inplace:
        img.normalize(mcp=mcp, inplace=inplace, dark_counts=100)
    else:
        img = img.normalize(mcp=mcp, inplace=inplace, dark_counts=100)
    assert (img is ref) == inplace
    assert_array_equal(img.image, normed_img.image)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("copy", [True, False])
def test_stack_normalize(inplace, copy, normed_img):
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    if copy:
        stack = base.LEEMStack([img.copy(), img.copy()])
    else:
        stack = base.LEEMStack([img, img])
    ref = stack
    if inplace:
        stack.normalize(mcp=MCP_IMG_FNAME, inplace=inplace, dark_counts=100)
    else:
        stack = stack.normalize(mcp=MCP_IMG_FNAME, inplace=inplace, dark_counts=100)
    assert (stack is ref) == inplace
    for img, ref_img in zip(stack, ref):
        assert (img is ref_img) == inplace
        assert_array_equal(img.image, normed_img.image)


def test_stack_align(short_stack):
    stack = short_stack
    template = stack[0].copy()

    # create Stack with artificial shifts -- seeded (rather than the previous bare
    # np.random.rand) so this is deterministic: the sanity check's robust z-score
    # (see _modified_zscore) can actually fire on a stack this short (unlike the old
    # classic mean/std z-score, which is mathematically bounded below its own
    # threshold for <= 10 points and so was a silent no-op here). These synthetic
    # shifts are large and mutually independent -- an adversarial pattern for any
    # global outlier detector -- so an unseeded run could occasionally flag a
    # legitimate frame as an outlier and fail the atol=0.5 assertion below. Seed 7
    # was checked (both against real ECC recovery on this fixture and against the
    # outlier threshold) to have a comfortable margin on both counts.
    rng = np.random.default_rng(7)
    x = rng.random(len(stack)) * 20 - 10
    y = rng.random(len(stack)) * 20 - 10
    x[0] = y[0] = 0

    for dx, dy, img in zip(x, y, stack):
        warp = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
        img.image = cv.warpPerspective(
            template.image,
            warp,
            img.image.shape[::-1],
            flags=cv.INTER_CUBIC,
        )
    stack = stack.align(mask=True)

    # extract aligned shifts
    dx = stack.warp_matrix[:, 0, 2]
    dy = stack.warp_matrix[:, 1, 2]

    np.testing.assert_allclose(x, dx, atol=0.5)
    np.testing.assert_allclose(y, dy, atol=0.5)


def test_stack_align_no_mask(short_stack):
    """mask=False/None should skip masking entirely and still return a full stack of
    identity-shaped warp matrices without raising."""
    stack = short_stack.copy()
    aligned = stack.align(mask=False)
    assert aligned.warp_matrix.shape == (len(stack), 3, 3)
    aligned_none = stack.align(mask=None)
    assert aligned_none.warp_matrix.shape == (len(stack), 3, 3)


def test_stack_align_outlier_rejection(monkeypatch):
    """A single gross misregistration should be flagged (z-score > 3), replaced with
    nan, and recovered via spline interpolation instead of leaking into the final
    accumulated warp matrices -- see base.py:485-514."""
    n = 15
    stack = base.LEEMStack(GCU_STACK_DIR)[0:n]

    rng = np.random.default_rng(0)
    raw_dx = rng.normal(0, 1, n)
    raw_dy = rng.normal(0, 1, n)
    raw_dx[0] = raw_dy[0] = 0.0
    outlier_idx = 7
    raw_dx[outlier_idx] = 30.0
    raw_dy[outlier_idx] = -25.0

    calls = {"i": 0}

    def fake_find_warp_matrix(self, template, **kwargs):
        i = calls["i"]
        calls["i"] += 1
        matrix = np.eye(3, dtype=np.float32)
        matrix[0, 2] = raw_dx[i + 1]
        matrix[1, 2] = raw_dy[i + 1]
        return matrix

    monkeypatch.setattr(base.LEEMImg, "find_warp_matrix", fake_find_warp_matrix)
    aligned = stack.align(mask=False)

    dx = aligned.warp_matrix[:, 0, 2]
    naive_cumulative = np.cumsum(raw_dx)[outlier_idx]
    # the interpolated value must land near its neighbors, not near the raw outlier
    assert abs(dx[outlier_idx] - dx[outlier_idx - 1]) < 3
    assert abs(dx[outlier_idx] - naive_cumulative) > 10


def test_stack_align_template_warns(short_stack):
    """Template alignment is fixed (see test_stack_align_template_recovers_shifts) but
    still less battle-tested than sequential alignment, so it keeps a soft caution --
    no longer the old "unstable" wording, since the underlying math bug is fixed."""
    stack = short_stack.copy()
    with pytest.warns(UserWarning, match="template"):
        aligned = stack.align(template=0, mask=False)
    assert aligned.warp_matrix.shape == (len(stack), 3, 3)


def test_stack_align_template_recovers_shifts():
    """Aligning against a fixed template should recover each frame's absolute shift
    directly -- unlike sequential mode, these must NOT be cumulatively composed.
    Regression test for the double-composition bug (inv(prev) @ current, then
    additionally cumulatively multiplied by the sequential-mode loop)."""
    n = 10
    stack = base.LEEMStack(GCU_STACK_DIR)[0:n]
    template = stack[0].copy()

    rng = np.random.default_rng(1)
    x = rng.uniform(-10, 10, n)
    y = rng.uniform(-10, 10, n)
    x[0] = y[0] = 0

    for dx, dy, img in zip(x, y, stack):
        warp = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
        img.image = cv.warpPerspective(
            template.image,
            warp,
            img.image.shape[::-1],
            flags=cv.INTER_CUBIC,
        )

    with pytest.warns(UserWarning, match="template"):
        aligned_by_index = stack.copy().align(template=0, mask=False)
    dx = aligned_by_index.warp_matrix[:, 0, 2]
    dy = aligned_by_index.warp_matrix[:, 1, 2]
    np.testing.assert_allclose(x, dx, atol=0.5)
    np.testing.assert_allclose(y, dy, atol=0.5)

    # Passing the template Image directly should give the same result as an index.
    with pytest.warns(UserWarning, match="template"):
        aligned_by_image = stack.copy().align(template=template, mask=False)
    dx_img = aligned_by_image.warp_matrix[:, 0, 2]
    dy_img = aligned_by_image.warp_matrix[:, 1, 2]
    np.testing.assert_allclose(x, dx_img, atol=0.5)
    np.testing.assert_allclose(y, dy_img, atol=0.5)


def test_stack_align_template_out_of_range():
    """An out-of-range template index should raise a clear error instead of being
    swallowed by a bare except and passed through to OpenCV as a raw int."""
    stack = base.LEEMStack(GCU_STACK_DIR)[0:5]
    with pytest.raises(IndexError):
        stack.align(template=999, mask=False)


def test_stack_align_template_bad_type():
    stack = base.LEEMStack(GCU_STACK_DIR)[0:5]
    with pytest.raises(TypeError):
        stack.align(template="not a template", mask=False)


def test_stack_align_single_frame():
    """A 1-frame stack has nothing to register against -- align() should return it
    unchanged with an identity warp instead of crashing (a 1-point array previously
    reached pandas' spline interpolation, which requires >= 2 points)."""
    stack = base.LEEMStack(GCU_STACK_DIR)[0:1]
    original = stack[0].image.copy()
    aligned = stack.align(mask=False)
    assert len(aligned) == 1
    np.testing.assert_array_equal(aligned.warp_matrix[0], np.eye(3))
    np.testing.assert_allclose(aligned[0].image, original, atol=1e-3)


@pytest.mark.parametrize("n", [2, 3])
def test_stack_align_small_stack(n):
    """Stacks too short for a 4-point cubic spline (the sanity/interpolation code's
    former hard-coded order=3) should still align without raising, using real ECC
    registration (typically no gaps at all -- see test_stack_align_small_stack_with_gap
    for the case that actually forces < 4 valid points)."""
    stack = base.LEEMStack(GCU_STACK_DIR)[0:n]
    aligned = stack.align(mask=False)
    assert aligned.warp_matrix.shape == (n, 3, 3)
    assert not np.isnan(aligned.warp_matrix).any()


def test_stack_align_small_stack_with_gap(monkeypatch):
    """A 3-frame stack with one interior failure has only 2 valid points (frame 0's
    trivial identity + frame 2's successful registration) -- pandas' spline with a
    hard-coded order=3 needs >= 4 points and raises 'error (m>k)'. The order must
    adapt to the number of valid points instead."""
    n = 3
    stack = base.LEEMStack(GCU_STACK_DIR)[0:n]

    def fake_find_warp_matrix(self, template, **kwargs):
        # First pairwise call (frame 1 vs frame 0) fails; frame 2 vs frame 1 succeeds.
        if not hasattr(fake_find_warp_matrix, "called"):
            fake_find_warp_matrix.called = True
            raise RuntimeError("simulated ECC failure")
        matrix = np.eye(3, dtype=np.float32)
        matrix[0, 2] = 2.0
        matrix[1, 2] = 3.0
        return matrix

    monkeypatch.setattr(base.LEEMImg, "find_warp_matrix", fake_find_warp_matrix)
    aligned = stack.align(mask=False)
    assert aligned.warp_matrix.shape == (n, 3, 3)
    assert not np.isnan(aligned.warp_matrix).any()


def test_stack_align_boundary_failure_extrapolated(monkeypatch):
    """A failure on the first or last frame previously left NaN in the final warp
    matrix (pandas' default interpolate() does not extrapolate leading NaNs). This is
    only reachable via template mode: in sequential mode frame 0's warp is always the
    trivial (never-computed) identity, so a genuine *leading* failure can't occur.
    Boundary frames should now be extrapolated instead of leaking NaN."""
    n = 10
    stack = base.LEEMStack(GCU_STACK_DIR)[0:n]

    rng = np.random.default_rng(2)
    raw_dx = rng.normal(0, 1, n)
    raw_dy = rng.normal(0, 1, n)
    fail_indices = {0, n - 1}  # first and last frame, both aligned to the template

    calls = {"i": 0}

    def fake_find_warp_matrix(self, template, **kwargs):
        i = calls["i"]
        calls["i"] += 1
        if i in fail_indices:
            raise RuntimeError("simulated ECC failure")
        matrix = np.eye(3, dtype=np.float32)
        matrix[0, 2] = raw_dx[i]
        matrix[1, 2] = raw_dy[i]
        return matrix

    monkeypatch.setattr(base.LEEMImg, "find_warp_matrix", fake_find_warp_matrix)
    with pytest.warns(UserWarning, match="template"):
        aligned = stack.align(template=0, mask=False)

    assert not np.isnan(aligned.warp_matrix).any()


def test_stack_align_recoverable_failure_prints_not_warns(monkeypatch, capfd, recwarn):
    """A single ECC failure that interpolation can still recover from is the routine,
    expected path (like statistical outlier rejection) -- it should be printed, not
    surfaced as a UserWarning that would clutter notebooks."""
    n = 10
    stack = base.LEEMStack(GCU_STACK_DIR)[0:n]

    rng = np.random.default_rng(3)
    raw_dx = rng.normal(0, 1, n)
    raw_dy = rng.normal(0, 1, n)
    raw_dx[0] = raw_dy[0] = 0.0
    fail_index = 5

    calls = {"i": 0}

    def fake_find_warp_matrix(self, template, **kwargs):
        i = calls["i"] + 1
        calls["i"] += 1
        if i == fail_index:
            raise RuntimeError("simulated ECC failure")
        matrix = np.eye(3, dtype=np.float32)
        matrix[0, 2] = raw_dx[i]
        matrix[1, 2] = raw_dy[i]
        return matrix

    monkeypatch.setattr(base.LEEMImg, "find_warp_matrix", fake_find_warp_matrix)
    aligned = stack.align(mask=False)

    out, _err = capfd.readouterr()
    assert "failed" in out.lower()
    assert not any(issubclass(w.category, UserWarning) for w in recwarn.list)
    assert not np.isnan(aligned.warp_matrix).any()


def test_stack_align_sequential_all_fail_falls_back_to_identity(monkeypatch):
    """If every pairwise registration fails, only the (trivial, never-computed)
    identity of the first frame remains valid -- a single valid point can't be
    interpolated, so the rest should warn and fall back to identity rather than
    crash or silently propagate nan."""
    n = 5
    stack = base.LEEMStack(GCU_STACK_DIR)[0:n]

    def always_fail(self, template, **kwargs):
        raise RuntimeError("simulated ECC failure")

    monkeypatch.setattr(base.LEEMImg, "find_warp_matrix", always_fail)
    with pytest.warns(UserWarning, match="identity"):
        aligned = stack.align(mask=False)

    dx = aligned.warp_matrix[:, 0, 2]
    dy = aligned.warp_matrix[:, 1, 2]
    np.testing.assert_allclose(dx, 0)
    np.testing.assert_allclose(dy, 0)


def test_stack_align_template_all_fail_returns_zeros(monkeypatch):
    """In template mode every frame (including the template's own trivial
    self-alignment) goes through find_warp_matrix -- if all of them fail, there is no
    valid point at all, and the whole stack should fall back to identity with a
    warning rather than crash or produce all-nan warp matrices."""
    n = 5
    stack = base.LEEMStack(GCU_STACK_DIR)[0:n]

    def always_fail(self, template, **kwargs):
        raise RuntimeError("simulated ECC failure")

    monkeypatch.setattr(base.LEEMImg, "find_warp_matrix", always_fail)
    with pytest.warns(UserWarning):
        aligned = stack.align(template=0, mask=False)

    assert not np.isnan(aligned.warp_matrix).any()
    np.testing.assert_allclose(aligned.warp_matrix[:, 0, 2], 0)
    np.testing.assert_allclose(aligned.warp_matrix[:, 1, 2], 0)


def test_stack_align_does_not_modify_source():
    """align(inplace=False) (the default) must never mutate the original stack's
    image data -- project rule: never modify source data."""
    stack = base.LEEMStack(GCU_STACK_DIR)[0:5]
    originals = [img.image.copy() for img in stack]
    original_warp_matrices = [img.warp_matrix.copy() for img in stack]

    stack.align(mask=False)

    for img, original in zip(stack, originals):
        np.testing.assert_array_equal(img.image, original)
    for img, original_wm in zip(stack, original_warp_matrices):
        np.testing.assert_array_equal(img.warp_matrix, original_wm)


def test_leemstack_zero_width_slice_raises():
    """A stack is a container for one or more objects -- an empty stack is not something
    __init__ can build, so LEEMStack.__getitem__ (which re-enters __init__ for a slice) must
    surface a clear ValueError for a zero-width slice instead of the opaque IndexError it used
    to raise via DataObjectStack.__init__'s unchecked `source[0]`."""
    stack = base.LEEMStack(GCU_STACK_DIR)[0:5]
    with pytest.raises(ValueError):
        stack[0:0]
    with pytest.raises(ValueError):
        stack[5:5]


def test_leemstack_all_false_boolean_mask_raises():
    """LEEMStack's boolean-mask __getitem__ (unlike plain ImageStack, which has none) can
    itself select zero elements -- must raise the same clear ValueError as a zero-width slice,
    not IndexError."""
    stack = base.LEEMStack(GCU_STACK_DIR)[0:5]
    mask = np.zeros(len(stack), dtype=bool)
    with pytest.raises(ValueError):
        stack[mask]


def test_stack_align_empty_raises():
    """align() on an empty stack must raise a clear ValueError (uspy/leem/base.py's own
    len(self) == 0 guard) rather than crash somewhere downstream. An empty LEEMStack can't be
    *constructed* (see test_leemstack_zero_width_slice_raises above), but it can still *arise*
    from manipulating an existing one -- del stack[:] doesn't go through __init__, so that route
    stays open and is what this test exercises."""
    stack = base.LEEMStack(GCU_STACK_DIR)[0:5]
    del stack[:]
    assert len(stack) == 0
    with pytest.raises(ValueError, match="empty"):
        stack.align()


### _resolve_shifts (shift cleanup helper used by align())


def test_resolve_shifts_no_nan_passthrough():
    """Small, non-outlying variation with no failures should pass through unchanged
    (no spline interpolation should even run)."""
    rng = np.random.default_rng(4)
    dx = rng.normal(0, 1, 10)
    dy = rng.normal(0, 1, 10)
    dx_out, dy_out = base._resolve_shifts(dx.copy(), dy.copy(), sanity=True)
    np.testing.assert_array_equal(dx_out, dx)
    np.testing.assert_array_equal(dy_out, dy)


def test_resolve_shifts_interior_nan_interpolated():
    dx = np.array([0.0, 1.0, np.nan, 3.0, 4.0, 5.0])
    dy = np.array([0.0, 2.0, np.nan, 6.0, 8.0, 10.0])
    dx_out, dy_out = base._resolve_shifts(dx, dy, sanity=False)
    assert not np.isnan(dx_out).any()
    assert not np.isnan(dy_out).any()
    assert abs(dx_out[2] - 2.0) < 0.5
    assert abs(dy_out[2] - 4.0) < 0.5


def test_resolve_shifts_leading_trailing_extrapolated():
    """Boundary NaNs (first/last frame failed) should be extrapolated, not left as
    NaN -- pandas' default interpolate() does not extrapolate on its own."""
    dx = np.array([np.nan, 1.0, np.nan, np.nan, 4.0])
    dy = np.array([np.nan, 10.0, np.nan, np.nan, 40.0])
    dx_out, dy_out = base._resolve_shifts(dx, dy, sanity=False)
    np.testing.assert_allclose(dx_out, [0.0, 1.0, 2.0, 3.0, 4.0], atol=1e-8)
    np.testing.assert_allclose(dy_out, [0.0, 10.0, 20.0, 30.0, 40.0], atol=1e-8)


@pytest.mark.parametrize("n_valid", [2, 3])
def test_resolve_shifts_few_valid_points_no_crash(n_valid):
    """2 or 3 valid points used to reach pandas' hard-coded spline order=3, which
    requires >= 4 points and raises. The order must adapt instead."""
    n = 6
    dx = np.full(n, np.nan)
    dy = np.full(n, np.nan)
    valid_idx = np.linspace(0, n - 1, n_valid, dtype=int)
    dx[valid_idx] = np.arange(n_valid, dtype=float)
    dy[valid_idx] = np.arange(n_valid, dtype=float) * 2
    dx_out, dy_out = base._resolve_shifts(dx, dy, sanity=False)
    assert not np.isnan(dx_out).any()
    assert not np.isnan(dy_out).any()
    # the originally-valid points must be preserved exactly
    np.testing.assert_allclose(dx_out[valid_idx], dx[valid_idx])
    np.testing.assert_allclose(dy_out[valid_idx], dy[valid_idx])


def test_resolve_shifts_single_valid_point_identity_fallback():
    """A single valid point can't be interpolated or extrapolated (needs >= 2 points
    even for a linear fit) -- the rest must fall back to identity, with a warning
    naming the fallback used."""
    n = 5
    dx = np.full(n, np.nan)
    dy = np.full(n, np.nan)
    dx[2] = 7.0
    dy[2] = -3.0
    with pytest.warns(UserWarning, match="identity"):
        dx_out, dy_out = base._resolve_shifts(dx, dy, sanity=False)
    expected_dx = np.zeros(n)
    expected_dx[2] = 7.0
    expected_dy = np.zeros(n)
    expected_dy[2] = -3.0
    np.testing.assert_array_equal(dx_out, expected_dx)
    np.testing.assert_array_equal(dy_out, expected_dy)


def test_resolve_shifts_all_nan_returns_zeros_with_warning():
    n = 5
    dx = np.full(n, np.nan)
    dy = np.full(n, np.nan)
    with pytest.warns(UserWarning):
        dx_out, dy_out = base._resolve_shifts(dx, dy, sanity=False)
    np.testing.assert_array_equal(dx_out, np.zeros(n))
    np.testing.assert_array_equal(dy_out, np.zeros(n))


def test_resolve_shifts_constant_input_no_false_outliers(capfd, recwarn):
    """Zero standard deviation (e.g. a perfectly static stack) must not divide by
    zero in the z-score sanity check or raise/warn/print spuriously."""
    n = 8
    dx = np.zeros(n)
    dy = np.zeros(n)
    dx_out, dy_out = base._resolve_shifts(dx, dy, sanity=True)
    out, _err = capfd.readouterr()
    assert out == ""
    assert not any(issubclass(w.category, UserWarning) for w in recwarn.list)
    np.testing.assert_array_equal(dx_out, dx)
    np.testing.assert_array_equal(dy_out, dy)


def test_resolve_shifts_outlier_rejected_and_interpolated(capfd):
    """A gross outlier among enough valid points should be rejected by the sanity
    check (printed, not warned -- outlier rejection is the expected, routine path,
    not an error) and recovered by interpolation. Uses the robust (median/MAD-based)
    outlier score, which -- unlike a classic mean/std z-score -- does not get
    inflated (and thereby self-masked) by the outlier's own magnitude, so a clearly
    huge outlier is used here rather than a deliberately moderate one."""
    n = 15
    rng = np.random.default_rng(5)
    dx = rng.normal(0, 1, n)
    dy = rng.normal(0, 1, n)
    dx[5] = 1000.0
    dy[5] = -1000.0
    dx_out, dy_out = base._resolve_shifts(dx, dy, sanity=True)
    out, _err = capfd.readouterr()
    assert "5" in out
    assert abs(dx_out[5]) < 5
    assert abs(dy_out[5]) < 5


### Robust outlier detection (_modified_zscore / masking-resistant sanity check)


def test_modified_zscore_huge_outlier_not_self_masked():
    """The classic mean/std z-score can be inflated by an extreme value enough that
    the value's own z-score drops back below the detection threshold ("self-
    masking"). The median/MAD-based modified z-score should not have this problem:
    a huge outlier should score very high, and the rest of the (well-behaved) data
    should score low."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 20)
    x[10] = 1e6
    scores = base._modified_zscore(x)
    assert scores[10] > 100
    other = np.delete(scores, 10)
    assert (other < 5).all()


def test_modified_zscore_constant_and_mad_zero():
    """All-identical input has MAD == 0 -- must not divide by zero, and (correctly)
    reports no outliers. Mostly-identical input plus one clear outlier collapses MAD
    to 0 too (the median absolute deviation is dominated by the many zero
    deviations), so the helper must fall back to mean-absolute-deviation scaling
    instead of returning nonsense/nan/inf."""
    constant = np.full(10, 3.0)
    scores = base._modified_zscore(constant)
    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(scores, np.zeros(10))

    mostly_constant = np.zeros(10)
    mostly_constant[3] = 50.0
    scores = base._modified_zscore(mostly_constant)
    assert np.isfinite(scores).all()
    assert scores[3] > scores[np.arange(10) != 3].max()


def test_resolve_shifts_multiple_outliers_not_masked(capfd):
    """Regression test for masking/swamping: with a classic mean/std z-score, one
    huge outlier inflates the standard deviation enough to also hide a second,
    smaller-but-still-genuine outlier (verified: with this seed, a classic |z| > 3
    check flags index 3 but not index 11, even at a magnitude of 8 -- almost 3x the
    detection threshold's nominal "sigma"). The robust score must catch both, with
    comfortable margin above `_OUTLIER_ZSCORE_THRESHOLD` for each."""
    n = 20
    rng = np.random.default_rng(6)
    dx = rng.normal(0, 1, n)
    dy = rng.normal(0, 1, n)
    dx[3], dy[3] = 40.0, 35.0
    dx[11], dy[11] = 15.0, -13.5

    dx_out, dy_out = base._resolve_shifts(dx, dy, sanity=True)
    out, _err = capfd.readouterr()

    assert "3" in out
    assert "11" in out
    assert abs(dx_out[3]) < 5 and abs(dy_out[3]) < 5
    assert abs(dx_out[11]) < 5 and abs(dy_out[11]) < 5


def test_resolve_shifts_mad_zero_outlier_detected(capfd):
    """When most shifts are exactly equal (MAD == 0, e.g. a very static stack with
    integer-pixel-quantized ECC output), a single gross outlier must still be
    detected via the mean-absolute-deviation fallback, not silently pass through."""
    n = 12
    dx = np.zeros(n)
    dy = np.zeros(n)
    dx[6] = 200.0
    dy[6] = -200.0
    dx_out, dy_out = base._resolve_shifts(dx, dy, sanity=True)
    out, _err = capfd.readouterr()
    assert "6" in out
    assert abs(dx_out[6]) < 5
    assert abs(dy_out[6]) < 5


### Normalize edge cases


def test_img_normalize_shape_mismatch():
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    bad_mcp = base.LEEMImg(np.ones((5, 5), dtype=np.float32))
    assert not img.is_compatible(bad_mcp)
    with pytest.raises(ValueError):
        img.normalize(mcp=bad_mcp, dark_counts=0)


def test_img_normalize_no_nan_or_inf():
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    mcp = base.LEEMImg(MCP_IMG_FNAME)
    result = img.normalize(mcp=mcp, dark_counts=100)
    assert not np.isnan(result.image).any()
    assert not np.isinf(result.image).any()
    # unlike the legacy uspy.legacy.driftnorm.normalize_image, LEEMImg.normalize does not
    # clip negative values -- pin that behavior explicitly.
    assert result.image.min() < 0


def test_img_normalize_division_by_zero_becomes_zero():
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    zero_mcp = base.LEEMImg(np.zeros_like(img.image))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = img.normalize(mcp=zero_mcp, dark_counts=0)
    assert not np.isnan(result.image).any()
    assert not np.isinf(result.image).any()
    assert (result.image == 0).all()


def test_img_normalize_division_by_zero_becomes_zero_when_deferred():
    """Regression test: `normalize`'s nan_to_num cleanup used to run on `result.image`, which
    for a deferred (composed) image is a *freshly materialized copy* -- mutating that copy
    with `copy=False` fixed nothing on the object actually returned, so the correction was
    silently lost whenever the image was still virtual at normalize() time."""
    stack = base.LEEMStack(RUO2_STACK_DIR, virtual=True)
    img = stack[0]
    assert img.is_virtual
    zero_mcp = base.LEEMImg(np.zeros_like(img.image))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = img.normalize(mcp=zero_mcp, dark_counts=0)
    assert not np.isnan(result.image).any()
    assert not np.isinf(result.image).any()
    assert (result.image == 0).all()


def test_img_normalize_stores_mcp_source_and_dark_counts():
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    mcp = base.LEEMImg(MCP_IMG_FNAME)
    result = img.normalize(mcp=mcp, dark_counts=100)
    assert result.mcp == mcp.source
    assert result.dark_counts == 100


def test_stack_normalize_skips_already_normalized_image():
    """If the same image (by identity) appears twice in a stack and is already
    normalized against the given mcp, it should not be normalized a second time --
    base.py:569."""
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    mcp = base.LEEMImg(MCP_IMG_FNAME)
    stack = base.LEEMStack([img, img])
    normed_once = stack.normalize(mcp=mcp, dark_counts=100, inplace=True)
    # normalizing again in place must be a no-op since img.mcp already equals mcp
    before = normed_once[0].image.copy()
    normed_twice = normed_once.normalize(mcp=mcp, dark_counts=100, inplace=True)
    assert_array_equal(normed_twice[0].image, before)


### TimeOrigin and time semantics


def test_time_origin_constructor():
    assert np.isnan(TimeOrigin(None).value)
    assert TimeOrigin(5).value == 5.0
    assert isinstance(TimeOrigin(5).value, float)
    assert TimeOrigin(TimeOrigin(3.5)).value == 3.5


def test_stack_time_origin_shared_with_images():
    """A stack directly constructed from file sources shares one mutable TimeOrigin
    object with the LEEMImgs it builds (base.py:354-356, LEEMImg.__init__ reuses an
    existing TimeOrigin instance rather than rewrapping it), so mutating
    stack.time_origin is visible on every element."""
    stack = base.LEEMStack(GCU_STACK_DIR)
    assert stack[0]._time_origin is stack._time_origin
    stack.time_origin = 100.0
    assert stack[0].time_origin == 100.0
    assert stack[0].rel_time == stack[0].timestamp - 100.0
    stack.time_origin = 50.0
    # mutating the stack's time origin must be visible through the shared TimeOrigin
    assert stack[0].time_origin == 50.0
    assert stack[1].time_origin == 50.0


def test_stack_time_origin_defaults_to_first_timestamp():
    stack = base.LEEMStack(GCU_STACK_DIR)
    assert same_or_nan(stack.time_origin, stack[0].timestamp)


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_getitem_slice_shares_time_origin_identity(virtual):
    """LEEMStack.__init__ reuses an already-built TimeOrigin passed in by __getitem__
    instead of rewrapping it, so a sliced stack shares the *same* TimeOrigin object with
    its parent (not just an equal copied value) -- mutating either afterwards is visible
    on both, and on the (reused, not reconstructed) elements too."""
    stack = base.LEEMStack(GCU_STACK_DIR, virtual=virtual)
    stack.time_origin = 123.0
    sliced = stack[1:4]
    assert sliced._time_origin is stack._time_origin
    # A virtual stack's elements are real, stable LEEMImgs too now -- built with the stack's
    # own TimeOrigin at construction, so this identity holds regardless of virtuality.
    assert sliced[0]._time_origin is sliced._time_origin

    # mutating the *sliced* stack's time origin after the fact must propagate back to the
    # parent and to the (shared-identity) elements -- proving real identity, not a value
    # copied once at slice time.
    sliced.time_origin = 456.0
    assert stack.time_origin == 456.0
    assert stack[1].time_origin == 456.0
    assert sliced[0].time_origin == 456.0


### Combining stacks (extend / + / +=) and TimeOrigin reconciliation


def test_stack_extend_warns_on_time_origin_mismatch():
    stack_a = base.LEEMStack(GCU_STACK_DIR)
    stack_b = base.LEEMStack(GCU_STACK_DIR)
    stack_b.time_origin = stack_a.time_origin + 1000.0

    with pytest.warns(UserWarning, match="time origin"):
        stack_a.extend(stack_b)


def test_stack_extend_no_warning_when_origins_match():
    stack_a = base.LEEMStack(GCU_STACK_DIR)
    stack_b = base.LEEMStack(GCU_STACK_DIR)
    assert stack_a.time_origin == stack_b.time_origin

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stack_a.extend(stack_b)


def test_stack_extend_rebinds_appended_elements_time_origin():
    stack_a = base.LEEMStack(GCU_STACK_DIR)
    stack_b = base.LEEMStack(GCU_STACK_DIR)
    stack_b.time_origin = stack_a.time_origin + 1000.0
    n_a = len(stack_a)

    with pytest.warns(UserWarning):
        stack_a.extend(stack_b)

    appended = stack_a[n_a:]
    for element in appended:
        assert element._time_origin is stack_a._time_origin
        assert element.rel_time == element.timestamp - stack_a.time_origin

    # mutating the merged stack's time origin afterwards must propagate to every
    # element, including the ones that came from stack_b.
    stack_a.time_origin = 42.0
    assert stack_a[n_a].time_origin == 42.0


@pytest.mark.parametrize("op", ["add", "iadd"])
def test_stack_add_and_iadd_operators_rebind_time_origin(op):
    stack_a = base.LEEMStack(GCU_STACK_DIR)
    stack_b = base.LEEMStack(GCU_STACK_DIR)
    stack_b.time_origin = stack_a.time_origin + 1000.0
    n_a = len(stack_a)

    with pytest.warns(UserWarning):
        combined = (stack_a + stack_b) if op == "add" else stack_a.__iadd__(stack_b)

    assert len(combined) == 2 * n_a
    for element in combined[n_a:]:
        assert element._time_origin is combined._time_origin


def test_stack_extend_virtual_self_nonvirtual_other_no_crash():
    """Regression test: DataObjectStack.extend previously left `elements` unassigned (an
    UnboundLocalError) when self was virtual and other was not; it should fall back to
    other's element sources, same as defer() does."""
    stack_a = base.LEEMStack(GCU_STACK_DIR, virtual=True)
    stack_b = base.LEEMStack(GCU_STACK_DIR, virtual=False)
    n_a = len(stack_a)

    stack_a.extend(stack_b)

    assert len(stack_a) == n_a + len(stack_b)


def test_img_isotime_nan_for_missing_timestamp():
    img = base.LEEMImg(np.zeros((4, 4)))
    assert np.isnan(img.timestamp)
    assert img.isotime == "??-??-?? ??:??:??"


### pressure property


def test_img_pressure_fallback():
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    img._meta["pressure"] = np.nan
    img._meta["pressure1"] = np.nan
    img._meta["pressure2"] = 5.5
    assert img.pressure == 5.5


def test_img_pressure_all_nan():
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    for key in ("pressure", "pressure1", "pressure2", "MCH", "PCH"):
        img._meta[key] = np.nan
    assert np.isnan(img.pressure)


def test_img_pressure_setter():
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    img.pressure = 42.0
    assert img._meta["pressure"] == 42.0
    assert img.pressure == 42.0


def test_img_time_origin_setter_runs_not_silently_swallowed():
    """Before DataObject.__setattr__ was fixed to honor property setters,
    `img.time_origin = x` silently wrote `_meta["time_origin"]` (never read back, since the
    property getter always wins) instead of running `time_origin`'s setter."""
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    img.time_origin = 555.0
    assert img.time_origin == 555.0
    assert "time_origin" not in img._meta


def test_img_binding_energy_setattr_raises():
    """binding_energy is a read-only property (derived from photon_energy, energy, and
    energy_offset) -- assigning to it must raise, not silently write a dead _meta entry."""
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    with pytest.raises(AttributeError):
        img.binding_energy = 5.0


def test_stack_energy_offset_scalar_broadcasts_and_shifts_binding_energy():
    """The reported case: a scalar energy_offset set on a stack must apply to every element,
    since a stack has no metadata of its own -- and it must actually affect the derived
    binding_energy on every element by exactly that amount."""
    stack = base.LEEMStack(GCU_STACK_DIR)
    stack.photon_energy = 100.0  # binding_energy needs a real photon_energy to be meaningful
    before = stack.binding_energy.copy()

    stack.energy_offset = 3.4934

    assert (stack.energy_offset == 3.4934).all()
    assert "energy_offset" not in stack.__dict__  # not shadowing the per-element gather
    assert all(img.energy_offset == 3.4934 for img in stack)
    np.testing.assert_allclose(stack.binding_energy, before - 3.4934)


def test_stack_binding_energy_setattr_raises():
    stack = base.LEEMStack(GCU_STACK_DIR)
    with pytest.raises(AttributeError):
        stack.binding_energy = 5.0


### warp / unwarped


def test_img_warp_inplace_vs_copy():
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    identity = np.eye(3)
    warp = np.array([[1, 0, 3], [0, 1, -2], [0, 0, 1]], dtype=np.float64)

    copied = img.warp(warp, inplace=False)
    assert copied is not img
    assert_array_equal(img.warp_matrix, identity)
    assert_array_equal(copied.warp_matrix, warp)

    ref = img
    warped = img.warp(warp, inplace=True)
    assert warped is ref
    assert_array_equal(img.warp_matrix, warp)


def test_img_unwarped_roundtrip():
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    warp = np.array([[1, 0, 4], [0, 1, -3], [0, 0, 1]], dtype=np.float64)
    warped = img.warp(warp, inplace=False)
    roundtripped = warped.unwarped
    # unwarped applies the inverse warp matrix; comparing against the (cropped) original
    # is only approximately equal near the center due to interpolation/edge effects.
    h, w = img.image.shape
    center = np.s_[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    np.testing.assert_allclose(
        roundtripped.image[center], img.image[center], atol=1000
    )


### parse() edge cases


def test_img_constructor_from_image_copies_and_clears_source():
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    img2 = base.LEEMImg(img)
    assert img2.source is None
    assert_array_equal(img2.image, img.image)
    assert img2.energy == img.energy


def test_img_temperature_sanitized_above_3000(monkeypatch):
    """parse() forces temperature to nan when the raw value is > 3000 -- base.py:242,
    presumably guarding against a sentinel/garbage value from the instrument."""
    orig_parse_image = ElmitecReader.parse_image

    def fake_parse_image(self, source):
        idict = orig_parse_image(self, source)
        idict["temperature"] = 5000.0
        return idict

    monkeypatch.setattr(ElmitecReader, "parse_image", fake_parse_image)
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    assert np.isnan(img.temperature)


### find_warp_matrix


def test_find_warp_matrix_unknown_algorithm():
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    template = base.LEEMImg(BREMEN2_IMG_FNAME)
    with pytest.raises(ValueError):
        img.find_warp_matrix(template, algorithm="nonsense")


### do_ecc_align trafo variants


@pytest.mark.parametrize("trafo", ["translation", "rigid", "euclidean", "affine"])
def test_do_ecc_align_trafo_variants(trafo):
    img1 = base.LEEMImg(BREMEN_IMG_FNAME)
    img2 = base.LEEMImg(BREMEN2_IMG_FNAME)
    warp_matrix = base.do_ecc_align(img1, img2, trafo=trafo, max_iter=50)
    assert warp_matrix.shape == (3, 3)


def test_do_ecc_align_unknown_trafo_falls_back_to_translation(capfd):
    img1 = base.LEEMImg(BREMEN_IMG_FNAME)
    img2 = base.LEEMImg(BREMEN2_IMG_FNAME)
    warp_matrix = base.do_ecc_align(img1, img2, trafo="nonsense", max_iter=50)
    out, _err = capfd.readouterr()
    assert "Unrecognized transformation" in out
    assert warp_matrix.shape == (3, 3)


### json round-trip


def test_img_json_roundtrip():
    import json_tricks

    img = base.LEEMImg(BREMEN_IMG_FNAME)
    dumped = json_tricks.dumps(img)
    loaded = json_tricks.loads(dumped)
    assert isinstance(loaded, base.LEEMImg)
    assert loaded.source == img.source
    assert_array_equal(loaded.image, img.image)


def test_img_json_roundtrip_with_normalization():
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    normed = img.normalize(mcp=MCP_IMG_FNAME, dark_counts=100)
    import json_tricks

    dumped = json_tricks.dumps(normed)
    loaded = json_tricks.loads(dumped)
    assert_array_equal(loaded.image, normed.image)


### get_field_string default formats


def test_img_get_field_string_defaults():
    img = base.LEEMImg(BREMEN_IMG_FNAME)
    # temperature: ".0f", energy: ".3g", rel_time: ".0f" -- base.py:114-124
    assert "." not in img.get_field_string("temperature").split()[0]
    energy_str = img.get_field_string("energy").split()[0]
    assert len(energy_str.replace("-", "").replace(".", "")) <= 3
    assert "." not in img.get_field_string("rel_time").split()[0]


### LEEMStack structure


def test_stack_getitem_boolean_mask():
    stack = base.LEEMStack(GCU_STACK_DIR)
    mask = np.zeros(len(stack), dtype=bool)
    mask[[0, 2, 4]] = True
    substack = stack[mask]
    assert len(substack) == 3
    assert substack[0] == stack[0]
    assert substack[1] == stack[2]
    assert substack[2] == stack[4]


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_getitem_slice_propagates_state(virtual):
    stack = base.LEEMStack(GCU_STACK_DIR, virtual=virtual)
    stack.time_origin = 123.0
    sliced = stack[1:4]
    assert isinstance(sliced, base.LEEMStack)
    assert len(sliced) == 3
    assert sliced.virtual == stack.virtual
    assert sliced.time_origin == 123.0
    assert sliced._reader is stack._reader
    assert sliced._time_origin is stack._time_origin


def test_stack_inherits_reader_from_image_list():
    images = [
        base.LEEMImg(BREMEN_IMG_FNAME),
        base.LEEMImg(BREMEN2_IMG_FNAME),
    ]
    stack = base.LEEMStack(images)
    assert stack._reader is images[0]._reader


### IVCurve


def test_ivcurve_constructs_with_energy_xaxis():
    stack = base.LEEMStack(GCU_STACK_DIR)
    roi = ROI.circle(x0=stack[0].width // 2, y0=stack[0].height // 2, radius=10)
    curve = base.IVCurve(stack, roi)
    assert curve.xdim == "energy"
    assert len(curve.x) == len(stack)


def test_ivcurve_accepts_keyword_arguments():
    """Regression test: IVCurve.__init__ used to call print(**kwargs), which raised
    TypeError for any keyword argument other than sep/end/file/flush."""
    stack = base.LEEMStack(GCU_STACK_DIR)
    roi = ROI.circle(x0=stack[0].width // 2, y0=stack[0].height // 2, radius=10)
    curve = base.IVCurve(stack, rois=roi)
    assert len(curve.x) == len(stack)


def test_ivcurve_stitches_overlapping_stacks():
    """Regression test: StitchedLine._stitch_curves calls scipy.optimize.curve_fit on
    the real-overlap path, which used to raise AttributeError because scipy.optimize
    was never imported in uspy.dataobject."""
    stack = base.LEEMStack(GCU_STACK_DIR)
    roi = ROI.circle(x0=stack[0].width // 2, y0=stack[0].height // 2, radius=10)
    half = len(stack) // 2
    stack1 = stack[0 : half + 5]
    stack2 = stack[half - 5 :]
    curve = base.IVCurve([stack1, stack2], [roi, roi])
    assert len(curve.x) == len(stack)


def test_intensity_line_new_field_axis_does_not_crash():
    """The reported failure mode: after adding brand-new metadata to a stack (via the
    scalar-broadcast __setattr__), using it as the x-axis of an IntensityLine must not
    KeyError just because there's no unit recorded for it yet -- it gets the default ""."""
    stack = base.LEEMStack(GCU_STACK_DIR)
    roi = ROI.circle(x0=stack[0].width // 2, y0=stack[0].height // 2, radius=10)
    stack.some_new_field = 5.0

    line = IntensityLine(stack, roi, "some_new_field")
    assert line.unit["x"] == ""
    assert len(line.x) == len(stack)


def test_intensity_line_computed_axis_without_unit_entry_does_not_crash():
    """A computed (property-backed) axis that never enters _meta/_units at all -- e.g.
    kinetic_energy, a pure alias for energy with no _unit_defaults entry -- must not KeyError
    either. This covers the `.get(xaxis, "")` fallback in IntensityLine.parse, which the
    metadata-write fix alone can't reach since nothing ever calls __setattr__ for a
    property."""
    stack = base.LEEMStack(GCU_STACK_DIR)
    roi = ROI.circle(x0=stack[0].width // 2, y0=stack[0].height // 2, radius=10)
    assert "kinetic_energy" not in stack[0].unit

    line = IntensityLine(stack, roi, "kinetic_energy")
    assert line.unit["x"] == ""
