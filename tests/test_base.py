"""Tests the agfalta.base module."""

from __future__ import annotations
from typing import Any, Optional
import numbers

import numpy as np
import pytest
from agfalta.dataobject import DataObject, DataObjectStack, Image
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
    @property
    def source(self) -> Optional[Any]:
        return super().source

    @source.setter
    def source(self, val):
        self._source = val


### Test DataObjectStack


class MinimalObjectStack(DataObjectStack):
    _type = MinimalObject


### ImageStack


class ImageStack(DataObjectStack):
    _type = Image


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
            add_obj.source = None
            assert add_obj.source is None
            stack[-1] = add_obj
    else:
        assert stack[-1] is add_obj

    add_obj.source = -10

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


@pytest.mark.parametrize("x", [Image(TESTDATA_DIR + TESTIMAGE_NAME + ".png"), 10, 10.0])
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
    assert (np.mean(stack).image == TESTIMAGE * 2).all()
    assert (np.sum(stack).image == TESTIMAGE * 6).all()


### Test data saving and loading


def test_image_pickling(testimgpng):
    testimgpng.save(TESTDATA_DIR + "test_image")
    img = Image.load(TESTDATA_DIR + "test_image.pickle")
    os.remove(TESTDATA_DIR + "test_image.pickle")
    assert (img.image == TESTIMAGE).all()


def test_image_compression(testimgpng):
    testimgpng.save(TESTDATA_DIR + "test_image.bz2")
    img = Image.load(TESTDATA_DIR + "test_image.pickle.bz2")
    os.remove(TESTDATA_DIR + "test_image.pickle.bz2")
    assert (img.image == TESTIMAGE).all()


def test_image_save_thin(testimgpng):
    testimgpng.save(TESTDATA_DIR + "test_image.thin")
    img = Image.load(TESTDATA_DIR + "test_image.thin.pickle")
    os.remove(TESTDATA_DIR + "test_image.thin.pickle")
    assert (img.image == TESTIMAGE).all()
