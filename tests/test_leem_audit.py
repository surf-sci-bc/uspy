"""Regression tests for the LEEM audit (uspy-w7a): LEEMImg/LEEMStack bookkeeping."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=protected-access

import json_tricks
import numpy as np
import pytest
import scipy.spatial
from numpy.testing import assert_array_equal

from uspy.dataobject import Image
from uspy.leem.base import LEEMImg, LEEMStack

from .conftest import BREMEN_IMG_FNAME, BREMEN2_IMG_FNAME, GCU_STACK_DIR, MCP_IMG_FNAME


def _translation(dx, dy):
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)


def _memory_mcp():
    """A source-less (in-memory) MCP image, same frame shape as the Bremen fixture."""
    return LEEMImg(LEEMImg(MCP_IMG_FNAME).image.copy())


### uspy-w7a.5: LEEMImg(Image) must not alias the input's pixels


def test_leemimg_from_image_does_not_share_pixels():
    src = Image(np.arange(12, dtype=np.float64).reshape(3, 4))
    expected = src.image.copy()
    leem = LEEMImg(src)
    leem.image += 5
    assert_array_equal(src.image, expected)
    assert_array_equal(leem.image, expected + 5)


### uspy-w7a.10: re-warping composes warp_matrix


def test_img_warp_twice_composes_matrix_and_unwarps_to_original():
    img = LEEMImg(BREMEN_IMG_FNAME)
    first, second = _translation(5, -3), _translation(-2, 4)

    twice = img.warp(first).warp(second)

    assert_array_equal(twice.warp_matrix, first @ second)
    h, w = img.shape
    interior = np.s_[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    assert_array_equal(twice.unwarped.image[interior], img.image[interior])


### uspy-w7a.3: JSON round-trip keeps warp, refuses in-memory normalisation


def test_img_json_roundtrip_keeps_warp():
    img = LEEMImg(BREMEN_IMG_FNAME).warp(_translation(5, -3))

    loaded = json_tricks.loads(json_tricks.dumps(img))

    assert_array_equal(loaded.warp_matrix, img.warp_matrix)
    assert_array_equal(loaded.image, img.image)


def test_img_json_roundtrip_keeps_normalisation_and_warp():
    img = LEEMImg(BREMEN_IMG_FNAME).normalize(mcp=MCP_IMG_FNAME, dark_counts=100)
    img = img.warp(_translation(-4, 2))

    loaded = json_tricks.loads(json_tricks.dumps(img))

    assert_array_equal(loaded.image, img.image)
    assert_array_equal(loaded.warp_matrix, img.warp_matrix)


def test_img_json_encode_raises_if_normalised_with_in_memory_mcp():
    img = LEEMImg(BREMEN_IMG_FNAME).normalize(mcp=_memory_mcp(), dark_counts=100)
    with pytest.raises(ValueError, match="in-memory"):
        json_tricks.dumps(img)


### uspy-w7a.4 / .9: LEEMStack.normalize bookkeeping


@pytest.fixture()
def two_frames():
    return LEEMStack([LEEMImg(BREMEN_IMG_FNAME), LEEMImg(BREMEN2_IMG_FNAME)])


def test_stack_normalize_identical_repeat_warns_and_skips(two_frames):
    two_frames.normalize(mcp=MCP_IMG_FNAME, dark_counts=100, inplace=True)
    before = two_frames.image.copy()

    with pytest.warns(UserWarning, match="already normalized"):
        two_frames.normalize(mcp=MCP_IMG_FNAME, dark_counts=100, inplace=True)

    assert_array_equal(two_frames.image, before)


def test_stack_normalize_different_dark_counts_raises_and_leaves_stack_untouched(two_frames):
    two_frames.normalize(mcp=MCP_IMG_FNAME, dark_counts=100, inplace=True)
    before = two_frames.image.copy()

    with pytest.raises(ValueError, match="already normalized"):
        two_frames.normalize(mcp=MCP_IMG_FNAME, dark_counts=50, inplace=True)

    assert_array_equal(two_frames.image, before)


def test_stack_normalize_different_mcp_raises(two_frames):
    two_frames.normalize(mcp=MCP_IMG_FNAME, dark_counts=100, inplace=True)
    with pytest.raises(ValueError, match="already normalized"):
        two_frames.normalize(mcp=_memory_mcp(), dark_counts=100, inplace=True)


def test_stack_normalize_in_memory_mcp_does_not_divide_twice(two_frames):
    mcp = _memory_mcp()
    once = two_frames.normalize(mcp=mcp, dark_counts=100)
    reference = once.image.copy()

    with pytest.warns(UserWarning, match="already normalized"):
        twice = once.normalize(mcp=mcp, dark_counts=100)

    assert_array_equal(twice.image, reference)


def test_stack_normalize_without_mcp_uses_each_frames_own_mcp(two_frames):
    expected = two_frames.normalize(mcp=MCP_IMG_FNAME, dark_counts=100)

    two_frames.mcp = MCP_IMG_FNAME
    result = two_frames.normalize(dark_counts=100)

    assert_array_equal(result.image, expected.image)


def test_normalize_without_any_mcp_raises_clear_error(two_frames):
    with pytest.raises(ValueError, match="mcp"):
        two_frames.normalize()
    with pytest.raises(ValueError, match="mcp"):
        two_frames[0].normalize()


### uspy-w7a.7: time origin of stacks built from / extended with LEEMImgs


def test_stack_from_leemimg_list_has_finite_rel_time_and_leaves_inputs_alone():
    imgs = [LEEMImg(BREMEN_IMG_FNAME), LEEMImg(BREMEN2_IMG_FNAME)]
    stack = LEEMStack(imgs)

    assert stack[0].rel_time == 0
    assert np.isfinite(stack[1].rel_time)
    assert np.isnan(imgs[0].rel_time)  # caller's objects keep their own (NaN) origin


def test_stack_slice_shares_time_origin_and_elements():
    stack = LEEMStack(GCU_STACK_DIR)
    sub = stack[2:5]
    assert sub._time_origin is stack._time_origin
    assert sub[0] is stack[2]


def test_extend_does_not_mutate_other_stack():
    a = LEEMStack(GCU_STACK_DIR)[:3]
    b = LEEMStack(GCU_STACK_DIR)[3:6]
    b.time_origin = a.time_origin + 100.0
    before = b.rel_time.copy()

    with pytest.warns(UserWarning, match="time origin"):
        a.extend(b)

    assert_array_equal(b.rel_time, before)
    assert_array_equal(a.rel_time[3:], b.timestamp - a.time_origin)


### uspy-w7a.8: LEEMStack indexing with numpy integers / integer sequences


def test_stack_getitem_numpy_int_and_int_sequences():
    stack = LEEMStack(GCU_STACK_DIR)
    assert stack[np.int64(2)] is stack[2]
    for index in ([0, 2], np.array([0, 2])):
        sub = stack[index]
        assert isinstance(sub, LEEMStack)
        assert [e is stack[i] for e, i in zip(sub, (0, 2))] == [True, True]
        assert sub._time_origin is stack._time_origin


### uspy-w7a.13: Elettra reader, ".tiff" and tracefile/file-count mismatch


def _elettra_dir(tmp_path, n_files, n_trace_rows, suffix=".tif"):
    """`n_files` real Elettra frames (+ sidecars) and a tracefile cut to `n_trace_rows`."""
    import glob
    import shutil

    from .conftest import ELETTRA_TIF_DIR

    for i, fname in enumerate(sorted(glob.glob(ELETTRA_TIF_DIR + "*.tif"))[:n_files]):
        shutil.copy(fname, tmp_path / f"frame_{i:03d}{suffix}")
    for sidecar in ("camera", "microscope", "beamline"):
        shutil.copy(glob.glob(ELETTRA_TIF_DIR + f"*{sidecar}.txt")[0], tmp_path)
    trace = glob.glob(ELETTRA_TIF_DIR + "*trace.txt")[0]
    with open(trace, encoding="utf-8") as fin:
        lines = fin.readlines()[: n_trace_rows + 1]
    (tmp_path / "x_trace.txt").write_text("".join(lines), encoding="utf-8")
    return str(tmp_path)


def test_elettra_reader_accepts_tiff_extension(tmp_path):
    folder = _elettra_dir(tmp_path, 3, 3, suffix=".tiff")
    img = LEEMImg(folder + "/frame_000.tiff", reader="elettra")
    assert np.isfinite(img.energy)
    assert img.image.shape == (img.height, img.width)


def test_elettra_trace_row_count_matching_files_is_accepted(tmp_path):
    stack = LEEMStack(_elettra_dir(tmp_path, 3, 3), reader="elettra")
    assert len(stack) == 3
    assert np.all(np.isfinite(stack.energy))


@pytest.mark.parametrize("n_rows", [2, 4])
def test_elettra_trace_row_count_mismatch_raises_clear_error(tmp_path, n_rows):
    with pytest.raises(ValueError, match="trace"):
        LEEMStack(_elettra_dir(tmp_path, 3, n_rows), reader="elettra")


### uspy-w7a.14: truncated Elmitec file must not hang


def test_parse_cp1252_until_null_raises_on_truncated_input():
    import io

    from uspy.utility import parse_cp1252_until_null

    assert parse_cp1252_until_null(io.BytesIO(b"abc\x00rest")) == "abc"
    with pytest.raises(EOFError):
        parse_cp1252_until_null(io.BytesIO(b"abc"))


def test_truncated_elmitec_file_fails_instead_of_hanging(tmp_path):
    raw = open(BREMEN_IMG_FNAME, "rb").read()
    for cut in range(24, 600, 37):  # cuts landing in header, key/unit strings, ...
        path = tmp_path / f"cut_{cut}.dat"
        path.write_bytes(raw[:cut])
        with pytest.raises(Exception):  # any error, but it must terminate
            LEEMImg(str(path))


### uspy-w7a.15: correlation map must be float for integer images


def test_calc_correlation_is_float_for_integer_images():
    from uspy.leem import processing

    rng = np.random.default_rng(1)
    data = rng.integers(0, 1000, size=(8, 4, 5))
    stack = LEEMStack(data.astype(np.float32))
    for element, frame in zip(stack, data):  # integer-valued frames, as e.g. raw detector data
        element._data["image"] = frame
    assert np.issubdtype(stack[0].image.dtype, np.integer)
    from uspy.dataobject import Line

    line = Line(np.stack([np.arange(8.0), data[:, 1, 2].astype(float)]))

    corr = processing.calc_correlation(stack, line)

    expected = np.array(
        [[scipy.spatial.distance.correlation(data[:, i, j], line.y) for j in range(5)]
         for i in range(4)]
    )
    assert np.issubdtype(corr.dtype, np.floating)
    np.testing.assert_allclose(corr, expected)
    assert corr[1, 2] == pytest.approx(0.0, abs=1e-12)
    assert 0 < corr[0, 0] < 2 and not float(corr[0, 0]).is_integer()


### uspy-w7a.16: leem plot_img accepts fields positionally


def test_leem_plot_img_accepts_positional_fields():
    import matplotlib.pyplot as plt

    from uspy.leem import plotting

    img = LEEMImg(BREMEN_IMG_FNAME)
    ax = plotting.plot_img(img, "energy")
    assert ax is not None
    plt.close("all")


### uspy-w7a.17: explicit `normalized` flag; dark_counts as filename


def test_stack_with_mcp_attribute_set_is_still_normalized_by_explicit_mcp(two_frames):
    expected = two_frames.normalize(mcp=MCP_IMG_FNAME, dark_counts=100)
    two_frames.mcp = MCP_IMG_FNAME

    result = two_frames.normalize(MCP_IMG_FNAME, dark_counts=100)  # no skip, no warning

    assert all(result.normalized)
    assert_array_equal(result.image, expected.image)


def test_stack_with_other_mcp_attribute_set_is_not_treated_as_normalized(two_frames):
    two_frames.mcp = "something_else.dat"
    result = two_frames.normalize(MCP_IMG_FNAME, dark_counts=100)
    assert all(result.normalized)
    assert result[0].mcp == MCP_IMG_FNAME


def test_normalize_with_dark_counts_filename_records_and_repeats():
    from .conftest import DC_IMG_FNAME

    stack = LEEMStack([LEEMImg(BREMEN_IMG_FNAME), LEEMImg(BREMEN2_IMG_FNAME)])
    once = stack.normalize(MCP_IMG_FNAME, dark_counts=DC_IMG_FNAME)
    with pytest.warns(UserWarning, match="already normalized"):
        once.normalize(MCP_IMG_FNAME, dark_counts=DC_IMG_FNAME)


def test_img_json_roundtrip_with_mcp_set_but_not_normalized():
    img = LEEMImg(BREMEN_IMG_FNAME)
    img.mcp = MCP_IMG_FNAME

    loaded = json_tricks.loads(json_tricks.dumps(img))

    assert not loaded.normalized
    assert loaded.mcp == MCP_IMG_FNAME
    assert_array_equal(loaded.image, img.image)


def test_img_json_roundtrip_normalized_flag():
    img = LEEMImg(BREMEN_IMG_FNAME).normalize(MCP_IMG_FNAME, dark_counts=100)
    assert img.normalized
    loaded = json_tricks.loads(json_tricks.dumps(img))
    assert loaded.normalized
    assert_array_equal(loaded.image, img.image)


def test_normalized_flag_survives_hdf5_roundtrip(tmp_path):
    stack = LEEMStack([LEEMImg(BREMEN_IMG_FNAME), LEEMImg(BREMEN2_IMG_FNAME)])
    stack = stack.normalize(MCP_IMG_FNAME, dark_counts=100)
    target = str(tmp_path / "x.h5")
    stack.save(target)
    reloaded = LEEMStack.load(target)
    assert all(reloaded.normalized)
    assert reloaded[0].mcp == MCP_IMG_FNAME
