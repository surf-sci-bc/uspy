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
from uspy.roi import ROI

from .conftest import (
    MCP_IMG_FNAME,
    dark_counts,
    same_or_nan,
    IMG_FNAMES_COMPATIBLE,
    STACK_FNAMES,
    STACK_INCOMPATIBLE_IMG_FNAME,
    TESTDATA_DIR,
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
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
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
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
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

    # create Stack with artificial shifts
    x = np.random.rand(len(stack)) * 20 - 10
    y = np.random.rand(len(stack)) * 20 - 10
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
    stack = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu")[0:n]

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
    stack = short_stack.copy()
    with pytest.warns(UserWarning, match="unstable"):
        aligned = stack.align(template=0, mask=False)
    assert aligned.warp_matrix.shape == (len(stack), 3, 3)


### Normalize edge cases


def test_img_normalize_shape_mismatch():
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
    bad_mcp = base.LEEMImg(np.ones((5, 5), dtype=np.float32))
    assert not img.is_compatible(bad_mcp)
    with pytest.raises(ValueError):
        img.normalize(mcp=bad_mcp, dark_counts=0)


def test_img_normalize_no_nan_or_inf():
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
    mcp = base.LEEMImg(MCP_IMG_FNAME)
    result = img.normalize(mcp=mcp, dark_counts=100)
    assert not np.isnan(result.image).any()
    assert not np.isinf(result.image).any()
    # unlike the legacy uspy.leem.driftnorm.normalize_image, LEEMImg.normalize does not
    # clip negative values -- pin that behavior explicitly.
    assert result.image.min() < 0


def test_img_normalize_division_by_zero_becomes_zero():
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
    zero_mcp = base.LEEMImg(np.zeros_like(img.image))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = img.normalize(mcp=zero_mcp, dark_counts=0)
    assert not np.isnan(result.image).any()
    assert not np.isinf(result.image).any()
    assert (result.image == 0).all()


def test_img_normalize_stores_mcp_source_and_dark_counts():
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
    mcp = base.LEEMImg(MCP_IMG_FNAME)
    result = img.normalize(mcp=mcp, dark_counts=100)
    assert result.mcp == mcp.source
    assert result.dark_counts == 100


def test_stack_normalize_skips_already_normalized_image():
    """If the same image (by identity) appears twice in a stack and is already
    normalized against the given mcp, it should not be normalized a second time --
    base.py:569."""
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
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
    stack = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu")
    assert stack[0]._time_origin is stack._time_origin
    stack.time_origin = 100.0
    assert stack[0].time_origin == 100.0
    assert stack[0].rel_time == stack[0].timestamp - 100.0
    stack.time_origin = 50.0
    # mutating the stack's time origin must be visible through the shared TimeOrigin
    assert stack[0].time_origin == 50.0
    assert stack[1].time_origin == 50.0


def test_stack_time_origin_defaults_to_first_timestamp():
    stack = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu")
    assert same_or_nan(stack.time_origin, stack[0].timestamp)


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_getitem_slice_shares_time_origin_identity(virtual):
    """LEEMStack.__init__ reuses an already-built TimeOrigin passed in by __getitem__
    instead of rewrapping it, so a sliced stack shares the *same* TimeOrigin object with
    its parent (not just an equal copied value) -- mutating either afterwards is visible
    on both, and on the (reused, not reconstructed) elements too."""
    stack = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu", virtual=virtual)
    stack.time_origin = 123.0
    sliced = stack[1:4]
    assert sliced._time_origin is stack._time_origin
    if not virtual:
        assert sliced[0]._time_origin is sliced._time_origin

    # mutating the *sliced* stack's time origin after the fact must propagate back to the
    # parent and to the (shared-identity) elements -- proving real identity, not a value
    # copied once at slice time.
    sliced.time_origin = 456.0
    assert stack.time_origin == 456.0
    if not virtual:
        assert stack[1].time_origin == 456.0
        assert sliced[0].time_origin == 456.0


def test_stack_to_lazy_shares_time_origin_identity():
    stack = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu")
    stack.time_origin = 789.0
    lazy = stack.to_lazy()
    assert lazy._time_origin is stack._time_origin
    lazy.time_origin = 111.0
    assert stack.time_origin == 111.0


### Combining stacks (extend / + / +=) and TimeOrigin reconciliation


def test_stack_extend_warns_on_time_origin_mismatch():
    stack_a = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu")
    stack_b = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu")
    stack_b.time_origin = stack_a.time_origin + 1000.0

    with pytest.warns(UserWarning, match="time origin"):
        stack_a.extend(stack_b)


def test_stack_extend_no_warning_when_origins_match():
    stack_a = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu")
    stack_b = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu")
    assert stack_a.time_origin == stack_b.time_origin

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stack_a.extend(stack_b)


def test_stack_extend_rebinds_appended_elements_time_origin():
    stack_a = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu")
    stack_b = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu")
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
    stack_a = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu")
    stack_b = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu")
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
    other's element sources, same as _deconstruct() does."""
    stack_a = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu", virtual=True)
    stack_b = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu", virtual=False)
    n_a = len(stack_a)

    stack_a.extend(stack_b)

    assert len(stack_a) == n_a + len(stack_b)


def test_img_isotime_nan_for_missing_timestamp():
    img = base.LEEMImg(np.zeros((4, 4)))
    assert np.isnan(img.timestamp)
    assert img.isotime == "??-??-?? ??:??:??"


### pressure property


def test_img_pressure_fallback():
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
    img._meta["pressure"] = np.nan
    img._meta["pressure1"] = np.nan
    img._meta["pressure2"] = 5.5
    assert img.pressure == 5.5


def test_img_pressure_all_nan():
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
    for key in ("pressure", "pressure1", "pressure2", "MCH", "PCH"):
        img._meta[key] = np.nan
    assert np.isnan(img.pressure)


def test_img_pressure_setter():
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
    img.pressure = 42.0
    assert img._meta["pressure"] == 42.0
    assert img.pressure == 42.0


### warp / unwarped


def test_img_warp_inplace_vs_copy():
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
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
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
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
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
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
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
    assert np.isnan(img.temperature)


### find_warp_matrix


def test_find_warp_matrix_unknown_algorithm():
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
    template = base.LEEMImg(TESTDATA_DIR + "bremen2.dat")
    with pytest.raises(ValueError):
        img.find_warp_matrix(template, algorithm="nonsense")


### do_ecc_align trafo variants


@pytest.mark.parametrize("trafo", ["translation", "rigid", "euclidean", "affine"])
def test_do_ecc_align_trafo_variants(trafo):
    img1 = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
    img2 = base.LEEMImg(TESTDATA_DIR + "bremen2.dat")
    warp_matrix = base.do_ecc_align(img1, img2, trafo=trafo, max_iter=50)
    assert warp_matrix.shape == (3, 3)


def test_do_ecc_align_unknown_trafo_falls_back_to_translation(capfd):
    img1 = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
    img2 = base.LEEMImg(TESTDATA_DIR + "bremen2.dat")
    warp_matrix = base.do_ecc_align(img1, img2, trafo="nonsense", max_iter=50)
    out, _err = capfd.readouterr()
    assert "Unrecognized transformation" in out
    assert warp_matrix.shape == (3, 3)


### json round-trip


def test_img_json_roundtrip():
    import json_tricks

    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
    dumped = json_tricks.dumps(img)
    loaded = json_tricks.loads(dumped)
    assert isinstance(loaded, base.LEEMImg)
    assert loaded.source == img.source
    assert_array_equal(loaded.image, img.image)


def test_img_json_roundtrip_with_normalization():
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
    normed = img.normalize(mcp=MCP_IMG_FNAME, dark_counts=100)
    import json_tricks

    dumped = json_tricks.dumps(normed)
    loaded = json_tricks.loads(dumped)
    assert_array_equal(loaded.image, normed.image)


### get_field_string default formats


def test_img_get_field_string_defaults():
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
    # temperature: ".0f", energy: ".3g", rel_time: ".0f" -- base.py:114-124
    assert "." not in img.get_field_string("temperature").split()[0]
    energy_str = img.get_field_string("energy").split()[0]
    assert len(energy_str.replace("-", "").replace(".", "")) <= 3
    assert "." not in img.get_field_string("rel_time").split()[0]


### LEEMStack structure


def test_stack_getitem_boolean_mask():
    stack = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu")
    mask = np.zeros(len(stack), dtype=bool)
    mask[[0, 2, 4]] = True
    substack = stack[mask]
    assert len(substack) == 3
    assert substack[0] == stack[0]
    assert substack[1] == stack[2]
    assert substack[2] == stack[4]


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_getitem_slice_propagates_state(virtual):
    stack = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu", virtual=virtual)
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
        base.LEEMImg(TESTDATA_DIR + "bremen.dat"),
        base.LEEMImg(TESTDATA_DIR + "bremen2.dat"),
    ]
    stack = base.LEEMStack(images)
    assert stack._reader is images[0]._reader


### IVCurve


def test_ivcurve_constructs_with_energy_xaxis():
    stack = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu")
    roi = ROI.circle(x0=stack[0].width // 2, y0=stack[0].height // 2, radius=10)
    curve = base.IVCurve(stack, roi)
    assert curve.xdim == "energy"
    assert len(curve.x) == len(stack)


def test_ivcurve_accepts_keyword_arguments():
    """Regression test: IVCurve.__init__ used to call print(**kwargs), which raised
    TypeError for any keyword argument other than sep/end/file/flush."""
    stack = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu")
    roi = ROI.circle(x0=stack[0].width // 2, y0=stack[0].height // 2, radius=10)
    curve = base.IVCurve(stack, rois=roi)
    assert len(curve.x) == len(stack)


def test_ivcurve_stitches_overlapping_stacks():
    """Regression test: StitchedLine._stitch_curves calls scipy.optimize.curve_fit on
    the real-overlap path, which used to raise AttributeError because scipy.optimize
    was never imported in uspy.dataobject."""
    stack = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu")
    roi = ROI.circle(x0=stack[0].width // 2, y0=stack[0].height // 2, radius=10)
    half = len(stack) // 2
    stack1 = stack[0 : half + 5]
    stack2 = stack[half - 5 :]
    curve = base.IVCurve([stack1, stack2], [roi, roi])
    assert len(curve.x) == len(stack)
