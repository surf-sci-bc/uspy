"""Tests the uspy.base module."""

from __future__ import annotations
import numbers

import cv2 as cv
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
from uspy.dataobject import DataObject, DataObjectStack, Image, ImageStack, Line
from deepdiff.diff import DeepDiff
import os

# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=attribute-defined-outside-init

### Test DataObject

from .conftest import (
    TESTDATA_DIR,
    TESTIMAGE,
    TESTIMAGE_ENDING,
    TESTIMAGE_NAME,
    ARRAY_1D,
    ARRAY_2D,
    quad_line,
)


def array2d():
    np.random.rand(2, 2)


class MinimalObject(DataObject):
    """A Class used for Tests. Source is converted to a numpy array"""

    _data_keys = ("matrix",)
    _meta_defaults = {"some_meta": "nothing"}

    def parse(self, source: str) -> dict[str, Any]:
        if not isinstance(source, (np.ndarray, numbers.Number)):
            print(source, type(source))
            raise TypeError
        data = {"matrix": source, "string": "TestMeta"}
        self._source = source
        return data

    def is_compatible(self, other: DataObject) -> bool:
        if isinstance(self.matrix, np.ndarray):
            print(self.matrix.shape, other.matrix.shape)
            return self.matrix.shape == other.matrix.shape
        if isinstance(self.matrix, numbers.Number) and isinstance(
            other.matrix, numbers.Number
        ):
            return True
        return False

    ### Property to set Source to None when needed
    # @property
    # def source(self) -> Optional[Any]:
    #     return super().source

    # @source.setter
    # def source(self, val):
    #     self._source = val


### Test DataObjectStack


class MinimalObjectStack(DataObjectStack):
    _type = MinimalObject


@pytest.mark.parametrize("source", ARRAY_2D + ARRAY_1D)
def test_dataobject_generation(source):
    ### DataObjects cannot be created itself
    with pytest.raises(NotImplementedError):
        DataObject("nosource")

    obj = MinimalObject(source)

    assert hasattr(obj, "matrix")
    assert hasattr(obj, "string")
    assert hasattr(obj, "some_meta")
    assert not hasattr(obj, "bogus")

    assert (obj.matrix == source).all()
    assert obj.string is "TestMeta"
    assert obj.some_meta is "nothing"
    assert (obj.source == source).all()

    assert not DeepDiff(
        obj.meta, {"string": "TestMeta", "some_meta": "nothing"}, ignore_order=True
    )
    assert not DeepDiff(obj.data, {"matrix": np.array(source)})


@pytest.mark.parametrize("val", [1, "bogus"])
@pytest.mark.parametrize("source", ARRAY_2D)
def test_dataobj_setattr(source, val):
    obj = MinimalObject(source)
    with pytest.raises(AttributeError):
        print(obj.attr)
    obj.attr = val
    assert obj.attr == val
    obj.attr = 1
    assert obj.attr == 1


@pytest.mark.parametrize(("source", "other_source"), [(ARRAY_2D[0], ARRAY_2D[1])])
def test_datobject_comparision(source, other_source):
    obj = MinimalObject(source)
    same_obj = MinimalObject(source)
    other_obj = MinimalObject(other_source)

    assert other_obj is not obj
    assert same_obj is not obj
    assert same_obj == obj
    assert same_obj != other_obj


@pytest.mark.parametrize("source", ARRAY_2D)
def test_dataobject_datadir_not_overridable(source):
    obj = MinimalObject(source)

    obj.data = None
    obj.meta = None
    assert obj.data is not None
    assert obj.meta is not None

    obj._data = None
    obj._meta = None

    assert obj._data is not None
    assert obj._meta is not None


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_generation_from_sources(list2darrays, virtual):
    sources = list2darrays

    stack = MinimalObjectStack(sources, virtual=virtual)
    assert len(stack) == len(sources)

    data = np.array([element.matrix for element in stack])
    assert (data == sources).all()

    objs = [MinimalObject(source) for source in sources]

    stack = MinimalObjectStack(objs)


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_generation_from_objects(list1darrays, virtual):
    sources = list1darrays

    objs = [MinimalObject(source) for source in sources]

    stack = MinimalObjectStack(objs, virtual=virtual)

    assert len(stack) == len(sources)
    for index, element in enumerate(stack):
        assert element is objs[index]  # Elements should be the same not equal


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_generation_with_not_compatible_objs(virtual):
    sources = [np.eye(2), np.eye(3)]

    objs = [MinimalObject(source) for source in sources]
    with pytest.raises(TypeError):
        MinimalObjectStack(objs, virtual=virtual)


def test_stack_virtual(list1darrays):
    sources = list1darrays
    virt_stack = MinimalObjectStack(sources, virtual=True)
    real_stack = MinimalObjectStack(sources, virtual=False)

    # For Virtual stacks the elements are sources
    assert (virt_stack.elements == np.asarray(sources)).all()
    for elv, elr in zip(virt_stack, real_stack):
        assert elv == elr
    # reifiy
    virt_stack.virtual = False
    assert not virt_stack.virtual
    assert virt_stack.elements == real_stack.elements

    # virtualization
    virt_stack.virtual = True
    assert virt_stack.virtual
    assert (virt_stack.elements == np.asarray(sources)).all()


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_setattr(list1darrays, virtual):
    sources = list1darrays
    stack = MinimalObjectStack(sources, virtual=virtual)

    # setting of per element attributes

    energy = np.linspace(0, 20, len(sources))
    if virtual:
        with pytest.raises(ValueError):
            stack.energy = energy
            stack.bogus = "Bogus"
            stack.array = [1, 2]
            stack.number = 1
    else:
        stack.energy = energy
        assert (stack.energy == energy).all()
        stack.bogus = "Bogus"
        stack.array = [1, 2]
        stack.number = 1
        """
        TODO:
        When a string is assigned to a stack and len(string) == len(stack)
        the chars are assigned to the elements not the stack. I am not sure if
        this is useful behaivor. It should be considered how Attributes are assigned
        to stacks and are passed to elements.
        """
        assert stack.bogus is "Bogus"
        assert stack.array == [1, 2]
        assert stack.number == 1


def test_stack_setattr_scalar_collision_raises(list1darrays):
    """A non-iterable scalar assignment must not silently shadow a real per-element
    attribute's gather: stack.some_meta = 5.0 previously became a plain instance attribute
    with no signal at all, so afterwards stack.some_meta returned 5.0 instead of the
    per-element array."""
    sources = list1darrays
    stack = MinimalObjectStack(sources, virtual=False)

    assert len(stack.some_meta) == len(sources)  # baseline: per-element gather works

    with pytest.raises(ValueError):
        stack.some_meta = 5.0

    # the rejected assignment must not have taken effect
    assert len(stack.some_meta) == len(sources)


def test_stack_setattr_mismatched_iterable_collision_raises(list1darrays):
    """A mismatched-length *iterable* assignment must be refused the same way as a scalar
    when it collides with a real per-element field -- this used to be allowed through (after
    printing a notice), attaching the whole iterable as a plain instance attribute and
    shadowing the per-element gather the same way a scalar would. (Originally added to
    accommodate uspy/leem/readers/elettra.py's post_process_stack attaching a mismatched-
    length tracefile column to a sliced stack -- that turned out to be a bug, not a real use
    case, and is fixed at its source; see uspy/leem/base.py's LEEMStack.__init__.)"""
    sources = list1darrays
    stack = MinimalObjectStack(sources, virtual=False)

    assert len(stack.some_meta) == len(sources)  # baseline: per-element gather works

    mismatched = ["x"] * (len(sources) + 1)  # deliberately wrong length
    with pytest.raises(ValueError):
        stack.some_meta = mismatched

    # the rejected assignment must not have taken effect
    assert len(stack.some_meta) == len(sources)

    # a genuinely non-colliding mismatched-length iterable is still allowed (existing
    # behavior, e.g. test_stack_setattr's `stack.array = [1, 2]` above)
    stack.totally_new_field = mismatched
    assert stack.totally_new_field == mismatched

    # a genuinely new, non-colliding scalar label is still allowed (existing behavior)
    stack.totally_new_label = "fine"
    assert stack.totally_new_label == "fine"


@pytest.mark.parametrize("virtual", [False])
@pytest.mark.parametrize(("source", "source_ext"), [(ARRAY_1D[0], ARRAY_1D[1])])
def test_stack_extend(source, source_ext, virtual):
    sources = source.tolist()
    sources_ext = source_ext.tolist()
    stack = MinimalObjectStack(sources, virtual=virtual)
    stack_ext = MinimalObjectStack(sources_ext, virtual=virtual)
    objs = [element for element in stack]
    objs.extend([element for element in stack_ext])

    stack.extend(stack_ext)

    for ii, jj in zip(stack.elements, objs):
        assert ii is jj


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_getitem(list1darrays, virtual):
    sources = list1darrays
    stack = MinimalObjectStack(sources, virtual=virtual)

    assert isinstance(stack[:3], MinimalObjectStack)
    assert isinstance(stack[0], MinimalObject)


@pytest.mark.parametrize("virtual", [False, True])
@pytest.mark.parametrize(("source", "source_add"), [(ARRAY_1D[0], ARRAY_1D[1])])
def test_stack_setitem(source, source_add, virtual):
    sources = source.tolist()
    sources_add = source_add.tolist()
    add_obj = MinimalObject(-10)
    add_stack = MinimalObjectStack(sources_add, virtual=virtual)
    load_stack = MinimalObjectStack(sources, virtual=virtual)

    # set one element to element
    stack = load_stack.copy()
    stack[-1] = add_obj
    if virtual:
        assert stack[-1].source == add_obj.source
        with pytest.raises(ValueError):
            add_obj._source = None
            assert add_obj.source is None
            stack[-1] = add_obj
    else:
        assert stack[-1] is add_obj

    add_obj._source = -10

    # set one element to list
    stack = load_stack.copy()
    with pytest.raises(ValueError):
        stack[0] = add_stack

    # set list to element
    stack = load_stack.copy()
    print(add_obj.source)
    with pytest.raises(TypeError):
        stack[:] = add_obj

    # set list to list

    stack = load_stack.copy()
    stack[0:1] = add_stack
    assert len(stack) == len(sources) + len(sources_add) - 1
    for index, obj in enumerate(stack[: len(add_stack)]):
        assert obj == add_stack[index]
    for index, obj in enumerate(stack[len(add_stack) :]):
        assert obj == load_stack[index + 1]

    stack = load_stack.copy()
    stack[:] = add_stack
    for obj1, obj2 in zip(stack, add_stack):
        assert obj1 == obj2

    # set element to integer

    with pytest.raises(ValueError):
        stack[0] = 1


### Image Class Tests


@pytest.mark.parametrize("fileending", TESTIMAGE_ENDING)
def test_image_generation_from_common_filetypes(fileending):
    img = Image(TESTDATA_DIR + TESTIMAGE_NAME + fileending)
    # pylint: disable = no-member
    assert (img.image == TESTIMAGE).all()
    assert (img.width, img.height) == TESTIMAGE.shape


@pytest.mark.parametrize("source", ARRAY_2D)
def test_image_generation_from_numpy_array(source):
    array = source
    img = Image(array)
    assert (img.image == array).all()
    assert img.width, img.height == source.shape
    assert img.source is None


# Integer Calculation Tests


@pytest.mark.parametrize("x", [10, 2.0, -5.0])
def test_image_add_integer(x, testimgpng):
    img2 = testimgpng + x
    assert (img2.image == TESTIMAGE + x).all()
    assert img2 is not testimgpng

    img2 = x + testimgpng
    assert (img2.image == TESTIMAGE + x).all()
    assert img2 is not testimgpng

    img2 = testimgpng - x
    assert (img2.image == TESTIMAGE - x).all()
    assert img2 is not testimgpng

    img2 = testimgpng * x
    assert (img2.image == TESTIMAGE * x).all()
    assert img2 is not testimgpng

    img2 = x * testimgpng
    assert (img2.image == TESTIMAGE * x).all()
    assert img2 is not testimgpng

    img2 = testimgpng / x
    assert (img2.image == TESTIMAGE / x).all()
    assert img2 is not testimgpng

    with pytest.raises(TypeError):
        _ = x / testimgpng


def test_image_iadd_integer(testimgpng):
    img = testimgpng
    testimgpng += 100
    assert (testimgpng.image == TESTIMAGE + 100).all()
    assert img is testimgpng


def test_image_imul_integer(testimgpng):
    img = testimgpng
    testimgpng *= 0.5
    assert (testimgpng.image == TESTIMAGE * 0.5).all()
    assert img is testimgpng


# Image Calculation Tests


def test_image_add_image(testimgpng):
    img2 = testimgpng + testimgpng / 2
    assert (img2.image == TESTIMAGE * 1.5).all()
    assert img2 is not testimgpng


def test_image_sub_integer(testimgpng):
    img2 = testimgpng - testimgpng / 2
    assert (img2.image == TESTIMAGE * 0.5).all()
    assert img2 is not testimgpng


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_image_div_image(testimgpng):
    img2 = testimgpng / testimgpng
    # 0/0 gives NaN so we have to get rid of it before testing
    assert (np.nan_to_num(img2.image) == np.eye(10)).all()
    assert img2 is not testimgpng


### Image Filter/Contrast Tests
#
# Regression coverage for `DataObject._with_data`, the helper `filter`/`contrast` build their
# result with: a first version silently no-op'd `new._data = ...` because of the pre-existing
# `__setattr__` guard against replacing `_data`/`_meta` once already set on an instance, so
# filtered/contrasted images silently kept their *original* pixels. These tests assert on
# exactly what that bug violated: the result differs from the input as expected, the input is
# untouched, and the result's `_data`/`_meta` are independent dict objects (not the same ones
# the original still holds) -- not just an independent wrapper pointing at shared containers.

FILTER_METHODS = [
    ("gaussian", {}),
    ("blur", {}),
    ("median", {}),
    ("kernel", {"kernel": np.ones((3, 3), dtype=np.float32) / 9}),
]


@pytest.mark.parametrize(
    "method, kwargs", FILTER_METHODS, ids=[case[0] for case in FILTER_METHODS]
)
def test_image_filter(method, kwargs, testimgpng):
    orig = testimgpng.image.copy()

    result = testimgpng.filter(method, **kwargs)

    if method == "gaussian":
        expected = cv.GaussianBlur(orig, (3, 3), -1)
    elif method == "blur":
        expected = cv.blur(orig, (3, 3))
    elif method == "median":
        expected = cv.medianBlur(orig, 3)
    else:
        expected = cv.filter2D(orig, ddepth=-1, kernel=kwargs["kernel"])

    assert np.allclose(result.image, expected)
    assert result is not testimgpng
    assert result.image is not testimgpng.image
    assert np.array_equal(testimgpng.image, orig)  # original untouched
    assert result.data is not testimgpng.data
    assert result.meta is not testimgpng.meta


def test_image_filter_unknown_method_raises(testimgpng):
    with pytest.raises(ValueError):
        testimgpng.filter("nonsense")


def test_image_contrast_equal(testimgpng):
    orig = testimgpng.image.copy()

    result = testimgpng.contrast("equal")
    expected = cv.equalizeHist(orig.astype(np.uint8))

    assert np.allclose(result.image, expected)
    assert result is not testimgpng
    assert result.image is not testimgpng.image
    assert np.array_equal(testimgpng.image, orig)
    assert result.data is not testimgpng.data
    assert result.meta is not testimgpng.meta


def test_image_contrast_clahe():
    # CLAHE requires an 8-/16-bit single-channel source -- unlike the float32 `testimgpng`
    # fixture, which .contrast("clahe") can't currently handle (a separate, pre-existing
    # limitation of the "clahe" branch, not something this test is about); use a directly
    # constructed uint8 image so this actually exercises the _with_data-based code path.
    array = np.random.default_rng(0).integers(0, 256, size=(12, 12), dtype=np.uint8)
    img = Image(array)
    orig = img.image.copy()

    result = img.contrast("clahe")
    expected = cv.createCLAHE().apply(orig)

    assert np.allclose(result.image, expected)
    assert result is not img
    assert result.image is not img.image
    assert np.array_equal(img.image, orig)
    assert result.data is not img.data
    assert result.meta is not img.meta


def test_image_contrast_fallback_returns_independent_copy(testimgpng):
    # "auto" (and any other unrecognized method) hits contrast()'s fallback, which mirrors
    # the pre-_with_data behavior of returning an unmodified but independent copy.
    orig = testimgpng.image.copy()

    result = testimgpng.contrast("auto")

    assert result is not testimgpng
    assert result.image is not testimgpng.image
    assert np.array_equal(result.image, orig)
    assert np.array_equal(testimgpng.image, orig)
    assert result.data is not testimgpng.data
    assert result.meta is not testimgpng.meta


def test_dataobject_with_data_independent_of_original():
    """Direct unit test for `DataObject._with_data`, independent of any specific caller: the
    exact invariant a prior bug broke (a plain `new._data = ...` was silently swallowed by
    `__setattr__`'s guard against replacing `_data`/`_meta` once already set, so the
    substituted array never actually took effect)."""
    original = Image(np.zeros((3, 3)))
    new_array = np.ones((3, 3))

    result = original._with_data(image=new_array)

    assert result.image is new_array
    assert result is not original
    assert result.data is not original.data
    assert result.meta is not original.meta
    assert np.array_equal(original.image, np.zeros((3, 3)))  # original untouched

    # mutating one's meta afterward must not affect the other
    result.meta["probe"] = "result"
    original.meta["probe"] = "original"
    assert result.meta["probe"] == "result"
    assert original.meta["probe"] == "original"


### Stack Arithmetic Tests


def test_stack_addsub_stack():
    sources = list(range(0, 10))
    sources_add = list(range(9, 12))

    stack = MinimalObjectStack(sources)
    stack_add = MinimalObjectStack(sources_add)

    comb_stack = stack + stack_add
    comb_sources = sources + sources_add

    for obj, source in zip(comb_stack, comb_sources):
        assert obj == MinimalObject(source)

    with pytest.raises(TypeError):
        comb_stack = stack - stack_add


# Further Tests have to be done with a Stack of a Class that implements calc itself


# @pytest.mark.parametrize("x", [Image(TESTDATA_DIR + TESTIMAGE_NAME + ".png"), 10, 10.0])
# @pytest.mark.parametrize(
#    "virtual", [False, pytest.param(True, marks=pytest.mark.xfail(raises=ValueError))]
# )
@pytest.mark.parametrize("x", [Image(TESTDATA_DIR + TESTIMAGE_NAME + ".png"), 10, 10.0])
@pytest.mark.parametrize("virtual", [False, True])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_stack_calc_imgNumber(x, virtual, testimgpng):
    sources = [testimgpng * i for i in range(1, 3)]
    if virtual:
        with pytest.warns(UserWarning):
            stack = ImageStack(sources, virtual=virtual)
    else:
        stack = ImageStack(sources, virtual=virtual)
    stack_calc = stack + x

    for index, img in enumerate(stack_calc):
        assert img == sources[index] + x

    stack_calc = stack - x

    for index, img in enumerate(stack_calc):
        assert img == sources[index] - x

    stack_calc = stack / x
    for index, img in enumerate(stack_calc):
        assert img == sources[index] / x

    if isinstance(x, numbers.Number):
        stack_calc = stack * x
        for index, img in enumerate(stack_calc):
            assert img == sources[index] * x


@pytest.mark.parametrize("virtual", [False, True])
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_stack_img_meansum(testimgpng, virtual):
    sources = [testimgpng * i for i in range(1, 4)]
    if virtual:
        with pytest.warns(UserWarning):
            stack = ImageStack(sources, virtual=virtual)
    else:
        stack = ImageStack(sources, virtual=virtual)

    # pylint: disable = no-member
    assert (np.mean(stack).image == TESTIMAGE * 2).all()
    assert (np.sum(stack).image == TESTIMAGE * 6).all()


### Test data saving and loading


def test_image_pickling(testimgpng):
    testimgpng.save(TESTDATA_DIR + "test_image.pickle")
    img = Image.load(TESTDATA_DIR + "test_image.pickle")
    os.remove(TESTDATA_DIR + "test_image.pickle")
    assert (img.image == TESTIMAGE).all()


def test_image_compression(testimgpng):
    testimgpng.save(TESTDATA_DIR + "test_image.pickle.bz2")
    img = Image.load(TESTDATA_DIR + "test_image.pickle.bz2")
    os.remove(TESTDATA_DIR + "test_image.pickle.bz2")
    assert (img.image == TESTIMAGE).all()


def test_image_save_json(testimgpng):
    testimgpng.save(TESTDATA_DIR + "test_image.json")
    img = Image.load(TESTDATA_DIR + "test_image.json")
    os.remove(TESTDATA_DIR + "test_image.json")
    assert (img.image == TESTIMAGE).all()


# @pytest.mark.parametrize(
#     "fileext",
#     [
#         ".thin.pickle",
#         pytest.param(".pickle", marks=pytest.mark.xfail(raises=AttributeError, strict = True)),
#     ],
# )
# def test_thin_obj_is_newly_generated(fileext, testimgpng):
#     testimgpng.save(TESTDATA_DIR + "test_image" + fileext)
#     Image.test = 1
#     img = Image.load(TESTDATA_DIR + "test_image" + fileext)
#     os.remove(TESTDATA_DIR + "test_image" + fileext)
#     assert img.test == 1


@pytest.mark.parametrize("fileext", [".pickle", ".pickle.bz2", ".json"])
def test_imagestack_save_full(fileext, testimgpng):
    stack = ImageStack([testimgpng, testimgpng])
    stack.save(TESTDATA_DIR + "test_image" + fileext)
    load_stack = Image.load(TESTDATA_DIR + "test_image" + fileext)
    os.remove(TESTDATA_DIR + "test_image" + fileext)
    assert len(stack) == len(load_stack)
    assert all(
        [(img1.image == img2.image).all for img1, img2 in zip(stack, load_stack)]
    )


@pytest.mark.parametrize("fileext", [".png", ".tiff"])
def test_image_save(testimgpng, fileext):
    testimgpng.save(TESTDATA_DIR + "test_image" + fileext)
    img = Image(TESTDATA_DIR + "test_image" + fileext)
    os.remove(TESTDATA_DIR + "test_image" + fileext)
    assert (img.image == testimgpng.image).all()


def test_stack_save():
    # array = [np.random.rand(10,10) for i in np.linspace(0,10,10)]
    array = [np.eye(10) * i for i in np.linspace(0, 65000, 10)]
    stack = ImageStack(array)
    stack.save(TESTDATA_DIR + "test_image.tiff")
    load_stack = ImageStack(TESTDATA_DIR + "test_image.tiff")
    os.remove(TESTDATA_DIR + "test_image.tiff")
    assert all([(img1.image == img2).all() for img1, img2 in zip(load_stack, array)])


@pytest.mark.parametrize("source", ARRAY_2D)
def test_line_generation(source):
    line = Line(source)
    assert_array_equal(line.dataframe.to_numpy(), source.T)
    assert line.xdim == "x"
    assert line.ydim == "y"


# @pytest.mark.xfail(raises = ValueError, strict = True)
@pytest.mark.parametrize("source", ARRAY_1D)
def test_line_incompatible_generation(source):
    with pytest.raises(ValueError):
        Line(source)


@pytest.mark.parametrize("source", ARRAY_2D)
def test_line_xy_alias(source):
    line = Line(source)
    line.xdim = "a"
    line.ydim = "b"
    # assert (line.a == source[0,:]).all()
    # assert (line.b == source[1,:]).all()
    assert_array_equal(line.a, source[0, :])
    assert_array_equal(line.b, source[1, :])


@pytest.mark.parametrize("source", ARRAY_2D)
def test_line_saveload_csv(source):
    line = Line(source)
    line.xdim = "a"
    line.ydim = "b"
    line.save(TESTDATA_DIR + "test.csv")
    line2 = Line(TESTDATA_DIR + "test.csv")
    # assert line.dataframe.equals(line2.dataframe)
    pd.testing.assert_frame_equal(line.dataframe, line2.dataframe)


def test_line_interpolate(quad_line):
    line = quad_line
    np.testing.assert_almost_equal(line.interpolate(line.x), line.y)
    assert line.interpolate(1.0).size == 1


def test_line_integration(quad_line):
    integral = quad_line.integral()
    assert 10 ** 3 / 3 - 10 ** -5 < integral.interpolate(10) < 10 ** 3 / 3 + 10 ** -5


def test_line_area(quad_line):
    assert quad_line.area == 1000 / 3


def test_line_derivative(quad_line):
    diff = quad_line.derivative()
    assert diff.length == quad_line.length - 1
    assert_array_equal(diff.x, quad_line.x[1:])

    # derivative of x^2 is 2x
    dydx = 2 * quad_line.x[1:]
    assert_almost_equal(dydx, diff.y)


def test_line_calc(quad_line):
    # calc with Numbers
    assert_array_equal((2 * quad_line).y, 2 * quad_line.y)
    assert_array_equal((quad_line / 2).y, quad_line.y / 2)
    assert_array_equal((quad_line + 2).y, quad_line.y + 2)
    assert_array_equal((quad_line - 2).y, quad_line.y - 2)
    # calc with Lines
    assert_array_equal((quad_line + quad_line).y, 2 * quad_line.y)
    assert_array_equal((quad_line - quad_line).y, np.zeros(quad_line.length))
    assert_array_equal((quad_line * quad_line).y, quad_line.y ** 2)
    assert_array_equal(
        np.nan_to_num((quad_line / quad_line).y, nan=1), np.ones(quad_line.length)
    )
