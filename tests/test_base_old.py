"""Tests the uspy.leem.base module."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import os
import pytest
import numpy as np

from uspy.leem import base

from .conftest import (
    same_or_nan,
    IMG_FNAMES_COMPATIBLE, STACK_FNAMES,
    STACK_INCOMPATIBLE_IMG_FNAME,
    TESTDATA_DIR
)


### Image loading

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

def test_img_copying(img):
    img2 = img.copy()
    assert img2 == img
    assert img2 is not img

def test_img_pickling(img):
    import os
    path = TESTDATA_DIR + "test_img.pickle"
    img.save(path)
    img2 = base.LEEMImg.load(path)
    os.remove(path)
    assert img.image.shape == img2.image.shape
    assert img.source == img2.source
    assert same_or_nan(img.energy, img2.energy)
    assert same_or_nan(img.timestamp, img2.timestamp)

def test_imge_constructor_nonsense():
    with pytest.raises(FileNotFoundError):
        _ = base.LEEMImg("nonsense")
    with pytest.raises(FileNotFoundError):
        _ = base.LEEMImg("nonsense.dat")


### Stack loading

@pytest.mark.parametrize("virtual", [False, True])
def test_stack_constructor(stack_fname, virtual):
    stack = base.LEEMStack(stack_fname, virtual=virtual)
    assert isinstance(stack[0], base.LEEMImg)
    assert stack[0].image.shape == stack[-1].image.shape

def test_stack_constructor_globbing():
    stack = base.LEEMStack(TESTDATA_DIR + "*.dat")
    stack2 = base.LEEMStack(TESTDATA_DIR + "*")
    assert len(stack2) >= len(stack)
    stack3 = base.LEEMStack(IMG_FNAMES_COMPATIBLE[0])
    assert len(stack3) == 1

def test_stack_constructor_lists():
    stack = base.LEEMStack(IMG_FNAMES_COMPATIBLE)
    stack2 = base.LEEMStack([base.LEEMImg(ifn) for ifn in IMG_FNAMES_COMPATIBLE])
    assert stack == stack2

def test_stack_constructor_array():
    stack = base.LEEMStack(STACK_FNAMES[0])
    data = np.stack([img.image for img in stack], axis=0)
    stack2 = base.LEEMStack(data)
    assert len(stack2) == len(stack)

def test_stack_copying(stack):
    stack2 = stack.copy()
    assert stack2 == stack
    assert stack2 is not stack

def test_stack_pickling(stack):
    import os
    path = TESTDATA_DIR + "test_stack.pickle"
    stack.save(path)
    stack2 = base.LEEMStack.load(path)
    os.remove(path)
    assert stack == stack2

def test_stack_constructor_nonsense():
    with pytest.raises(ValueError):
        _ = base.LEEMStack("nonsense")
    with pytest.raises(FileNotFoundError):
        _ = base.LEEMStack(["nonsense.dat"])

def test_stack_virtuality(stack_fname):
    stack2 = base.LEEMStack(stack_fname, virtual=True)
    sources = stack2.sources
    if not isinstance(sources[0], str) or not sources[0].endswith(".dat"):
        assert stack2.virtual
        return
    stack = base.LEEMStack(stack_fname, virtual=False)
    assert stack2.virtual
    assert len(stack) == len(stack2)
    assert stack[0].image.shape == stack2[0].image.shape
    assert stack[-1] == stack2[-1]

def test_substacks(stack):
    end_idx = len(stack) // 2
    substack = stack[0:end_idx]
    assert isinstance(substack, base.LEEMStack)
    assert len(substack) == end_idx
    assert substack[0] == stack[0]


### Stack container functionality

def test_stack_indexing(stack):
    for img in stack[10:20]:
        assert isinstance(img, base.LEEMImg)
    assert isinstance(stack[22], base.LEEMImg)
    with pytest.raises(IndexError):
        _ = stack[len(stack) + 1]
    with pytest.raises(TypeError):
        _ = stack["nonsense"]

def test_stack_setitem(stack):
    img = stack[0]
    stack[len(stack) // 2] = img
    with pytest.raises(IndexError):
        stack[len(stack) + 1] = img
    with pytest.raises((TypeError, AttributeError, ValueError)):
        stack[len(stack) // 2] = "nonsense"

def test_stack_delitem(stack):
    length = len(stack)
    del stack[length // 2 - 1: length // 2 + 1]
    assert len(stack) == length - 2


### Data integrity

def test_img_attr_types(img):
    if ".dat" in img.source:
        assert img.energy > -6
        assert img.timestamp > 0
    else:
        assert np.isnan(img.energy)
        assert np.isnan(img.timestamp)
    assert np.isnan(img.temperature) or img.temperature > -280
    assert isinstance(img.timestamp, float)

def test_img_setattr(img):
    img.energy = 7
    assert img.energy == 7
    with pytest.raises(AttributeError):
        img.rel_time = 5
    img.image = np.ones_like(img.image)

def test_stack_getattr(stack):
    assert stack.energy.shape == (len(stack),)
    test_idx = len(stack) // 2
    assert stack.timestamp[test_idx] == stack[test_idx].timestamp
    assert same_or_nan(stack.energy[test_idx], stack[test_idx].energy)
    assert same_or_nan(stack.rel_time[test_idx], stack[test_idx].rel_time)

def test_stack_setattr(stack):
    stack.energy = np.linspace(0, 10, len(stack))
    assert stack.energy[-1] == 10


def test_stack_dimension_consistency(stack):
    img = base.LEEMImg(STACK_INCOMPATIBLE_IMG_FNAME)
    with pytest.raises(ValueError):
        stack[0] = img

def test_stack_set_virtual(stack_fname):
    if "tif" in stack_fname:
        return
    stack = base.LEEMStack(stack_fname, virtual=True)
    assert stack.virtual
    stack.virtual = False
    assert len(stack._elements) == len(stack)

### Image arithmetic

def test_img_mul_int():
    test_data = np.array([[1,2],[3,4]], dtype=float)
    img1 = 2*base.LEEMImg(test_data)
    img2 = base.LEEMImg(test_data)*2
    assert np.array_equal(img1.image, 2*test_data)
    assert np.array_equal(img2.image, 2*test_data)

def test_img_add_int():
    test_data = np.array([[1,2],[3,4]], dtype=float)
    img1 = base.LEEMImg(test_data)+100
    img2 = base.LEEMImg(test_data)-100
    assert np.array_equal(img1.image, test_data+100)
    assert np.array_equal(img2.image, test_data-100)

def test_img_div_int():
    test_data = np.array([[1,2],[3,4]], dtype=float)
    img = base.LEEMImg(test_data)/2
    assert np.array_equal(img.image, test_data/2)

def test_img_add_img():
    test_data_1 = np.array([[1,2],[3,4]], dtype=float)
    test_data_2 = np.array([[5,6],[7,8]], dtype=float)
    img1 = base.LEEMImg(test_data_1)
    img2 = base.LEEMImg(test_data_2)
    img12 = img1+img2
    img21 = img2+img1
    assert np.array_equal(img12.image, test_data_1+test_data_2)
    assert np.array_equal(img21.image, test_data_1+test_data_2)

def test_img_sub_img():
    test_data_1 = np.array([[1,2],[3,4]], dtype=float)
    test_data_2 = np.array([[5,6],[7,8]], dtype=float)
    img1 = base.LEEMImg(test_data_1)
    img2 = base.LEEMImg(test_data_2)
    img12 = img1-img2
    img21 = img2-img1
    assert np.array_equal(img12.image, test_data_1-test_data_2)
    assert np.array_equal(img21.image, test_data_2-test_data_1)

def test_img_div_img():
    test_data_1 = np.array([[1,2],[3,4]], dtype=float)
    test_data_2 = np.array([[5,6],[7,8]], dtype=float)
    img1 = base.LEEMImg(test_data_1)
    img2 = base.LEEMImg(test_data_2)
    img12 = img1/img2
    img21 = img2/img1
    assert np.array_equal(img12.image, test_data_1/test_data_2)
    assert np.array_equal(img21.image, test_data_2/test_data_1)