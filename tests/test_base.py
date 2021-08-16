"""Tests the agfalta.base module."""

from __future__ import annotations
from typing import Any
import numbers

import numpy as np
import pytest
from agfalta.base import DataObject, DataObjectStack, Image
from deepdiff.diff import DeepDiff

# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=attribute-defined-outside-init

### Test DataObject

from .conftest import TESTDATA_DIR


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


### Test DataObjectStack


class MinimalObjectStack(DataObjectStack):
    _type = MinimalObject


### ImageStack


class ImageStack(DataObjectStack):
    _type = Image


@pytest.mark.parametrize("source", [np.array([[1, 2], [3, 4]])])
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

    assert not DeepDiff(
        obj.meta, {"string": "TestMeta", "some_meta": "nothing"}, ignore_order=True
    )
    assert not DeepDiff(obj.data, {"matrix": np.array(source)})


@pytest.mark.parametrize("val", [1, "bogus"])
@pytest.mark.parametrize("source", [np.array([[1, 2], [3, 4]])])
def test_dataobj_setattr(source, val):
    obj = MinimalObject(source)
    with pytest.raises(AttributeError):
        print(obj.attr)
    obj.attr = val
    assert obj.attr == val


@pytest.mark.parametrize(
    ("source", "other_source"), [(np.array([[1, 2], [3, 4]]), np.eye(2))]
)
def test_datobject_comparision(source, other_source):
    obj = MinimalObject(source)
    same_obj = MinimalObject(source)
    other_obj = MinimalObject(other_source)

    assert other_obj is not obj
    assert same_obj is not obj
    assert same_obj == obj
    assert same_obj != other_obj


@pytest.mark.parametrize("source", [np.array([[1, 2], [3, 4]])])
def test_dataobject_data_mutability(source):
    obj = MinimalObject(source)

    # data and meta should be immutable
    with pytest.raises(AttributeError):
        obj.data = None
        obj.meta = None

    # _data and _meta are not immutable
    obj._data = None
    obj._meta = None

    assert obj._data is None
    assert obj._meta is None


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_generation_from_sources(virtual):
    sources = np.linspace(0, 100, 10)

    stack = MinimalObjectStack(sources, virtual=virtual)
    assert len(stack) == len(sources)

    data = np.array([element.matrix for element in stack])
    assert (data == sources).all()

    objs = [MinimalObject(source) for source in sources]

    stack = MinimalObjectStack(objs)


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_generation_from_objects(virtual):
    sources = np.linspace(0, 100, 11)

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


def test_stack_virtual():
    sources = np.linspace(0, 100, 11)
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
def test_stack_setattr(virtual):
    sources = np.linspace(0, 100, 11)
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
        assert stack.bogus is "Bogus"
        assert stack.array == [1, 2]
        assert stack.number == 1


@pytest.mark.parametrize("virtual", [False])
def test_stack_extend(virtual):
    sources = np.linspace(0, 100, 11)
    sources_ext = np.linspace(100, 200, 11)
    stack = MinimalObjectStack(sources, virtual=virtual)
    stack_ext = MinimalObjectStack(sources_ext, virtual=virtual)
    objs = [element for element in stack]
    objs.extend([element for element in stack_ext])

    stack.extend(stack_ext)

    for ii, jj in zip(stack.elements, objs):
        assert ii is jj


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_getitem(virtual):
    sources = np.linspace(0, 100, 11)
    stack = MinimalObjectStack(sources, virtual=virtual)

    assert isinstance(stack[:3], MinimalObjectStack)
    assert isinstance(stack[0], MinimalObject)


### Image Class Tests


@pytest.mark.parametrize("fileending", [".png", ".tif"])
def test_image_generation_from_common_filetypes(fileending):
    img = Image(TESTDATA_DIR + f"test_gray_16bit{fileending}")
    # pylint: disable = no-member
    assert (img.image == np.eye(10, dtype=np.float32) * 38550).all()


# Integer Calculation Tests


@pytest.fixture
def testimgpng():
    return Image(TESTDATA_DIR + "test_gray_16bit.png")


@pytest.mark.parametrize("x", [10, 2.0, -5.0])
def test_image_add_integer(x, testimgpng):
    img2 = testimgpng + x
    assert (img2.image == np.eye(10) * 38550 + x).all()
    assert img2 is not testimgpng

    img2 = x + testimgpng
    assert (img2.image == np.eye(10) * 38550 + x).all()
    assert img2 is not testimgpng

    img2 = testimgpng - x
    assert (img2.image == np.eye(10) * 38550 - x).all()
    assert img2 is not testimgpng
    """
    img2 = x - testimgpng
    assert (img2.image == np.eye(10) * 38550 - x).all()
    assert img2 is not testimgpng
    """
    img2 = testimgpng * x
    assert (img2.image == np.eye(10) * 38550 * x).all()
    assert img2 is not testimgpng

    img2 = x * testimgpng
    assert (img2.image == np.eye(10) * 38550 * x).all()
    assert img2 is not testimgpng

    img2 = testimgpng / x
    assert (img2.image == np.eye(10) * 38550 / x).all()
    assert img2 is not testimgpng

    with pytest.raises(TypeError):
        _ = x / testimgpng


def test_image_iadd_integer(testimgpng):
    img = testimgpng
    testimgpng += 100
    assert (testimgpng.image == np.eye(10) * 38550 + 100).all()
    assert img is testimgpng


def test_image_imul_integer(testimgpng):
    img = testimgpng
    testimgpng *= 0.5
    assert (testimgpng.image == np.eye(10) * 38550 * 0.5).all()
    assert img is testimgpng


# Image Calculation Tests


def test_image_add_image(testimgpng):
    img2 = testimgpng + testimgpng / 2
    assert (img2.image == np.eye(10) * 38550 * 1.5).all()
    assert img2 is not testimgpng


def test_image_sub_integee(testimgpng):
    img2 = testimgpng - testimgpng / 2
    assert (img2.image == np.eye(10) * 38550 * 0.5).all()
    assert img2 is not testimgpng


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_image_div_image(testimgpng):
    img2 = testimgpng / testimgpng
    # 0/0 gives NaN so we have to get rid of it before testing
    assert (np.nan_to_num(img2.image) == np.eye(10)).all()
    assert img2 is not testimgpng


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


@pytest.mark.parametrize("x", [Image(TESTDATA_DIR + "test_gray_16bit.png"), 10, 10.0])
@pytest.mark.parametrize(
    "virtual", [False, pytest.param(True, marks=pytest.mark.xfail(raises=ValueError))]
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_stack_calc_imgNumber(x, virtual, testimgpng):
    sources = [testimgpng * i for i in range(1, 3)]

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


@pytest.mark.parametrize(
    "virtual", [False, pytest.param(True, marks=pytest.mark.xfail(raises=ValueError))]
)
def test_stack_img_meansum(testimgpng, virtual):
    sources = [testimgpng * i for i in range(1, 4)]
    stack = ImageStack(sources, virtual=virtual)

    # pylint: disable = no-member
    assert (np.mean(stack).image == np.eye(10) * 38550 * 2).all()
    assert (np.sum(stack).image == np.eye(10) * 38550 * 6).all()
