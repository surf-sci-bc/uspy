"""Regression tests for the data-core audit fixes (uspy-evt.*)."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=protected-access

import numpy as np
import pytest
import tifffile

from uspy.dataobject import Image, ImageStack, Line

from .conftest import TESTIMAGE_BASE


### uspy-evt.1: arithmetic never mutates the caller's arrays


def test_inplace_arithmetic_does_not_mutate_source_array():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    expected = src.copy()
    img = Image(src)
    img += 5
    img *= 2
    img -= 1
    img /= 4
    np.testing.assert_array_equal(src, expected)
    np.testing.assert_allclose(img.image, ((expected + 5) * 2 - 1) / 4)


def test_inplace_arithmetic_does_not_mutate_stack_source_array():
    src = np.ones((3, 4, 5), dtype=np.float32)
    stack = ImageStack(src)
    stack += 2
    np.testing.assert_array_equal(src, 1)
    np.testing.assert_array_equal(stack.image, 3)


def test_inplace_arithmetic_does_not_mutate_line_source_array():
    src = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    expected = src.copy()
    line = Line(src)
    line += 1
    np.testing.assert_array_equal(src, expected)
    np.testing.assert_array_equal(line.y, expected[1] + 1)


def test_virtual_inplace_arithmetic_stays_deferred(tiff_frame_stack):
    stack = ImageStack(tiff_frame_stack["paths"], virtual=True)
    stack += 1
    assert stack.virtual


### uspy-evt.2: integer pixels become float32 at parse


@pytest.fixture()
def uint16_tif(tmp_path):
    path = str(tmp_path / "u16.tif")
    tifffile.imwrite(path, np.full((4, 5), 60000, dtype=np.uint16))
    return path


def test_tif_uint16_arithmetic_does_not_wrap(uint16_tif):
    img = Image(uint16_tif)
    assert img.image.dtype == np.float32
    img += 1
    assert img.image.max() == 60001
    img /= 2
    assert img.image.max() == 30000.5


def test_shipped_16bit_tif_is_float32():
    assert Image(TESTIMAGE_BASE + ".tif").image.dtype == np.float32


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32])
def test_int_ndarray_becomes_float32(dtype):
    arr = np.full((3, 4), 7, dtype=dtype)
    assert Image(arr).image.dtype == np.float32
    assert Image(arr, defer=True).image.dtype == np.float32
    assert arr.dtype == dtype  # source untouched


def test_float_ndarray_keeps_dtype():
    assert Image(np.ones((3, 4), dtype=np.float64)).image.dtype == np.float64


def test_virtual_average_of_uint16_does_not_overflow(tmp_path):
    paths = []
    for i in range(2):
        path = str(tmp_path / f"f{i}.tif")
        tifffile.imwrite(path, np.full((4, 5), 38550, dtype=np.uint16))
        paths.append(path)
    averaged = ImageStack(paths, virtual=True).average(2)
    np.testing.assert_allclose(averaged[0].image, 38550)


### uspy-evt.9: resident operand-with-virtual-operand loads the operand


def test_resident_image_minus_virtual_image_stays_resident(tiff_frame_stack):
    img = Image(np.full((5, 4), 10.0, dtype=np.float32))
    dark = Image(tiff_frame_stack["paths"][0], defer=True)  # all ones
    result = img - dark
    assert not result.is_virtual
    np.testing.assert_allclose(result.image, 9)
    img -= dark
    assert not img.is_virtual
    np.testing.assert_allclose(img.image, 9)


def test_virtual_image_minus_virtual_image_stays_deferred(tiff_frame_stack):
    first, second = (Image(p, defer=True) for p in tiff_frame_stack["paths"][:2])
    result = second - first
    assert result.is_virtual
    np.testing.assert_allclose(result.image, 1)


### uspy-evt.3: virtual stack setitem/extend refuses resident elements


@pytest.fixture()
def virtual_stack(tiff_frame_stack):
    return ImageStack(tiff_frame_stack["paths"], virtual=True)


def _processed_resident(path):
    img = Image(path)  # resident, has a source
    img += 100  # in-memory edit the source doesn't reflect
    return img


def test_virtual_setitem_resident_element_raises(virtual_stack, tiff_frame_stack):
    img = _processed_resident(tiff_frame_stack["paths"][0])
    with pytest.raises(ValueError, match="deferred"):
        virtual_stack[0] = img
    assert virtual_stack.virtual


def test_virtual_setitem_slice_with_resident_stack_raises(virtual_stack, tiff_frame_stack):
    resident = ImageStack(tiff_frame_stack["paths"][:2])
    with pytest.raises(ValueError, match="deferred"):
        virtual_stack[:2] = resident


def test_virtual_extend_resident_element_raises(virtual_stack, tiff_frame_stack):
    with pytest.raises(ValueError, match="deferred"):
        virtual_stack.extend(_processed_resident(tiff_frame_stack["paths"][0]))
    assert len(virtual_stack) == 3


def test_virtual_setitem_and_extend_deferred_element_ok(virtual_stack, tiff_frame_stack):
    img = _processed_resident(tiff_frame_stack["paths"][0]).deferred()  # explicit discard
    virtual_stack[1] = img
    virtual_stack.extend(img)
    assert virtual_stack.virtual and len(virtual_stack) == 4
    np.testing.assert_allclose(virtual_stack[1].image, 1)  # processing discarded, not silent


### uspy-evt.6: stack indexing with numpy ints / int sequences


@pytest.fixture()
def frame_stack(tiff_frame_stack):
    return ImageStack(tiff_frame_stack["paths"])


@pytest.mark.parametrize("index", [np.int64(1), np.int32(-1)])
def test_stack_getitem_numpy_int(frame_stack, index):
    assert frame_stack[index] is frame_stack._elements[int(index)]


@pytest.mark.parametrize("index", [[0, 2], np.array([0, 2]), (0, 2)])
def test_stack_getitem_int_sequence(frame_stack, index):
    sub = frame_stack[index]
    assert isinstance(sub, ImageStack)
    assert sub.sources == [frame_stack.sources[0], frame_stack.sources[2]]


def test_stack_getitem_bool_mask(frame_stack):
    sub = frame_stack[np.array([True, False, True])]
    assert sub.sources == [frame_stack.sources[0], frame_stack.sources[2]]


### uspy-evt.7: stack attribute assignment of 2-D arrays


def test_stack_setattr_square_array_broadcasts_with_warning(frame_stack):
    matrix = np.eye(3)
    with pytest.warns(UserWarning, match="broadcast"):
        frame_stack.wm = matrix
    for img in frame_stack:
        np.testing.assert_array_equal(img.wm, matrix)
    assert frame_stack[0].wm is not frame_stack[1].wm  # independent copies


def test_stack_setattr_2d_array_other_shape_broadcasts_silently(frame_stack, recwarn):
    frame_stack.wm = np.ones((2, 5))
    assert not [w for w in recwarn if "broadcast" in str(w.message)]
    for img in frame_stack:
        assert img.wm.shape == (2, 5)


def test_stack_setattr_list_of_arrays_stays_per_element(frame_stack):
    frame_stack.wm = list(np.arange(12).reshape(3, 4))
    np.testing.assert_array_equal(frame_stack[2].wm, [8, 9, 10, 11])


def test_stack_setattr_1d_array_stays_per_element(frame_stack):
    frame_stack.temperature = np.array([1.0, 2.0, 3.0])
    assert list(frame_stack.temperature) == [1.0, 2.0, 3.0]


### uspy-evt.8: slicing a virtual stack must not re-parse pixels


def test_virtual_slice_does_not_reparse(tiff_frame_stack, monkeypatch):
    stack = ImageStack(tiff_frame_stack["paths"], virtual=True)
    calls = []
    original = Image.parse
    monkeypatch.setattr(
        Image, "parse", lambda self, source: calls.append(source) or original(self, source)
    )
    sub = stack[1:]
    assert sub.virtual and len(sub) == 2
    assert not calls


def test_virtual_stack_check_compatible_still_catches_mismatch(tmp_path):
    paths = []
    for i, shape in enumerate([(4, 5), (6, 5)]):
        paths.append(str(tmp_path / f"f{i}.tif"))
        tifffile.imwrite(paths[-1], np.zeros(shape, dtype=np.float32))
    stack = ImageStack(paths, virtual=True)[0:2]  # no I/O, no error
    with pytest.raises(ValueError, match="not compatible"):
        stack.check_compatible()


### uspy-evt.10: virtual=True on resident-only sources warns


def test_virtual_multipage_tiff_warns(tmp_path):
    path = str(tmp_path / "multi.tif")
    tifffile.imwrite(path, np.zeros((3, 4, 5), dtype=np.float32), photometric="minisblack")
    with pytest.warns(UserWarning, match="virtual"):
        stack = ImageStack(path, virtual=True)
    assert not stack.virtual


def test_virtual_ndarray_stack_warns():
    with pytest.warns(UserWarning, match="virtual"):
        ImageStack(np.zeros((3, 4, 5)), virtual=True)


def test_virtual_file_list_does_not_warn(tiff_frame_stack, recwarn):
    ImageStack(tiff_frame_stack["paths"], virtual=True)
    assert not [w for w in recwarn if "virtual" in str(w.message)]


### uspy-evt.11: one-frame stack saves to TIFF


def test_one_frame_stack_saves_tif(tmp_path):
    frame = np.arange(20, dtype=np.float32).reshape(4, 5)
    stack = ImageStack([Image(frame)])
    target = str(tmp_path / "one.tif")
    stack.save(target)
    np.testing.assert_array_equal(np.squeeze(tifffile.imread(target)), frame)


### uspy-evt.12: save/contrast clip with a warning; unknown contrast method raises


def test_png_save_clips_out_of_range_with_warning(tmp_path):
    import imageio

    img = Image(np.array([[-5.0, 100.0], [70000.0, 65535.0]], dtype=np.float32))
    target = str(tmp_path / "x.png")
    with pytest.warns(UserWarning, match="clip"):
        img.save(target)
    saved = imageio.imread(target)
    np.testing.assert_array_equal(saved, [[0, 100], [65535, 65535]])
    assert img.image.max() == 70000  # data object untouched


def test_png_save_in_range_does_not_warn(tmp_path, recwarn):
    Image(np.array([[0.0, 100.0], [65535.0, 3.0]], dtype=np.float32)).save(str(tmp_path / "x.png"))
    assert not [w for w in recwarn if "clip" in str(w.message)]


def test_contrast_equal_clips_with_warning(testimgpng):
    # fixture holds 16-bit values (38550), far beyond uint8
    with pytest.warns(UserWarning, match="clip"):
        result = testimgpng.contrast("equal")
    assert result.image.dtype == np.uint8


def test_contrast_clahe_on_16bit_float_image(testimgpng, recwarn):
    result = testimgpng.contrast("clahe")  # values fit uint16: no clipping
    assert result.image.shape == testimgpng.image.shape
    assert not [w for w in recwarn if "clip" in str(w.message)]


@pytest.mark.parametrize("method", ["auto", "nonsense"])
def test_contrast_unknown_method_raises(testimgpng, method):
    with pytest.raises(ValueError, match=method):
        testimgpng.contrast(method)


### uspy-evt.13: JSON save of a source-less object raises (and leaves the target alone)


def test_json_save_of_sourceless_image_raises_and_keeps_existing_file(tmp_path):
    target = tmp_path / "x.json"
    target.write_text("precious")
    with pytest.raises(ValueError, match=r"\.h5|pickle"):
        Image(np.ones((3, 4), dtype=np.float32)).save(str(target))
    assert target.read_text() == "precious"


def test_json_save_of_sourceless_line_raises(tmp_path):
    with pytest.raises(ValueError, match="source"):
        Line(np.array([[0.0, 1.0], [0.0, 1.0]])).save(str(tmp_path / "l.json"))


def test_json_save_of_sourceless_stack_raises(tmp_path):
    target = tmp_path / "s.json"
    with pytest.raises(ValueError, match="source"):
        ImageStack(np.ones((2, 3, 4), dtype=np.float32)).save(str(target))
    assert not target.exists()


def test_json_roundtrip_with_source_still_works(tiff_frame_stack, tmp_path):
    img = Image(tiff_frame_stack["paths"][0])
    target = str(tmp_path / "i.json")
    img.save(target)
    np.testing.assert_array_equal(Image.load(target).image, img.image)


def test_pickle_save_of_sourceless_image_still_works(tmp_path):
    img = Image(np.ones((3, 4), dtype=np.float32))
    target = str(tmp_path / "i.pickle")
    img.save(target)
    np.testing.assert_array_equal(Image.load(target).image, img.image)


### uspy-evt.14: Line.area (scipy>=1.14 keyword-only x) and Line.smooth edges


def test_line_area_passes_x_by_keyword(monkeypatch):
    import scipy.integrate

    real = scipy.integrate.simpson
    monkeypatch.setattr(
        scipy.integrate, "simpson", lambda y, *, x=None, dx=1.0: real(y, x=x, dx=dx)
    )  # scipy>=1.14 signature
    x = np.array([0.0, 1.0, 3.0, 4.0, 6.0])
    assert Line(np.array([x, 2 * x])).area == pytest.approx(36.0)  # integral of 2x on [0, 6]


def test_line_smooth_constant_line_stays_constant():
    line = Line(np.array([np.arange(7.0), np.ones(7)]))
    np.testing.assert_allclose(line.smooth(5).y, 1.0)


def test_line_smooth_linear_line_stays_linear_away_from_edges():
    x = np.arange(9.0)
    smoothed = Line(np.array([x, 2 * x])).smooth(5)
    np.testing.assert_allclose(smoothed.y[2:-2], 2 * x[2:-2])
    assert smoothed.y[0] == 0 and smoothed.y[-1] == 16  # endpoints stay pinned


def test_line_smooth_does_not_alias_x_or_modify_original():
    x, y = np.arange(7.0), np.arange(7.0) ** 2
    line = Line(np.array([x, y]))
    smoothed = line.smooth(3)
    assert smoothed.x is not line.x
    smoothed.x[0] = -99
    assert line.x[0] == 0
    np.testing.assert_array_equal(line.y, y)


### uspy-evt.15: old pickles lacking a newly added _meta_defaults field


class _DefaultsObject(Line):
    """Line subclass whose class defaults gain a mutable field (as if added after pickling)."""

    _meta_defaults = {**Line._meta_defaults, "new_matrix": np.eye(2)}


def test_missing_meta_default_is_served_after_pickle_roundtrip(tmp_path):
    obj = _DefaultsObject(np.array([[0.0, 1.0], [0.0, 1.0]]))
    del obj._meta["new_matrix"]  # state of a pickle written before the field existed
    target = str(tmp_path / "old.pickle")
    obj.save(target)
    loaded = _DefaultsObject.load(target)
    assert "new_matrix" not in loaded._meta
    np.testing.assert_array_equal(loaded.new_matrix, np.eye(2))
    assert "new_matrix" not in loaded._meta  # not written back on read


def test_missing_meta_default_is_a_copy():
    obj = _DefaultsObject(np.array([[0.0, 1.0], [0.0, 1.0]]))
    del obj._meta["new_matrix"]
    obj.new_matrix[0, 0] = 99
    np.testing.assert_array_equal(_DefaultsObject._meta_defaults["new_matrix"], np.eye(2))


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        Line(np.array([[0.0, 1.0], [0.0, 1.0]])).no_such_field  # pylint: disable=expression-not-assigned


def test_old_leemimg_pickle_without_normalized(tmp_path):
    from uspy.leem.base import LEEMImg

    from .conftest import BREMEN_IMG_FNAME

    img = LEEMImg(BREMEN_IMG_FNAME)
    expected = img.normalized
    del img._meta["normalized"]
    target = str(tmp_path / "old_leem.pickle")
    img.save(target)
    assert LEEMImg.load(target).normalized == expected
