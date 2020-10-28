"""Tests the agfalta.leem.base module."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import os
from datetime import datetime

import pytest
import numpy as np

from agfalta.leem import base

from .conftest import IMG_FNAMES_COMPATIBLE, STACK_FNAMES, same_or_nan


TESTDATA_DIR = os.path.dirname(__file__) + "/../testdata/"


### General

def test_version():
    assert base.LEEMBASE_VERSION > 1.0


### Image loading

@pytest.mark.parametrize("nolazy", [True, False])
def test_img_constructor(img_fname, nolazy):
    img = base.LEEMImg(img_fname, nolazy=nolazy)
    assert img.path == img_fname
    assert isinstance(img.energy, (int, float))
    assert isinstance(img.meta, dict)
    assert isinstance(img.data, np.ndarray)
    assert img.data.shape == (img.height, img.width)

def test_img_constructor_array(img):
    img2 = base.LEEMImg(img.data)
    assert img2.data.shape == img.data.shape

def test_img_copying(img):
    img2 = img.copy()
    assert img2 == img
    assert img2 is not img

def test_img_pickling(img):
    img.save(TESTDATA_DIR + "test_img.limg")
    img2 = base.LEEMImg(TESTDATA_DIR + "test_img.limg")
    assert img.data.shape == img2.data.shape
    assert img.path == img2.path
    assert img.meta == img2.meta

def test_imge_constructor_nonsense():
    with pytest.raises(ValueError):
        _ = base.LEEMImg("nonsense")
    with pytest.raises(FileNotFoundError):
        _ = base.LEEMImg("nonsense.dat", nolazy=True)
    img = base.LEEMImg("nonsense.dat")
    with pytest.raises(FileNotFoundError):
        assert img.data


### Stack loading

@pytest.mark.parametrize("virtual", [False, True])
# @pytest.mark.parametrize("stack_fname", stack_fnames+tif_stack_fnames)
def test_stack_constructor(stack_fname, virtual):
    stack = base.LEEMStack(stack_fname, virtual=virtual)
    assert isinstance(stack[0], base.LEEMImg)
    assert stack[0].data.shape == stack[-1].data.shape

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
    assert stack.path == "NO_PATH"

def test_stack_constructor_array():
    stack = base.LEEMStack(STACK_FNAMES[0])
    stack2 = base.LEEMStack(stack.data)
    assert len(stack2) == len(stack)

def test_stack_copying(stack):
    stack2 = stack.copy()
    assert stack2 == stack
    assert stack2 is not stack

def test_stack_pickling(stack):
    stack.save(TESTDATA_DIR + "test_stack.lstk")
    stack2 = base.LEEMStack(TESTDATA_DIR + "test_stack.lstk")
    assert stack == stack2

def test_stack_constructor_nonsense():
    with pytest.raises(ValueError):
        _ = base.LEEMStack("nonsense")
    with pytest.raises(FileNotFoundError):
        _ = base.LEEMStack(["nonsense.dat"], nolazy=True)
    stack = base.LEEMStack(["nonsense.dat"])
    with pytest.raises(FileNotFoundError):
        assert stack[0]

def test_stack_virtuality(stack_fname):
    stack2 = base.LEEMStack(stack_fname, virtual=True)
    if not isinstance(stack2.fnames[0], str) or not stack2.fnames[0].endswith(".dat"):
        assert not stack2._virtual
        return
    stack = base.LEEMStack(stack_fname, virtual=False)
    assert stack2._images is None
    assert len(stack) == len(stack2)
    assert stack[0].data.shape == stack2[0].data.shape
    assert stack[-1].meta == stack2[-1].meta

def test_stack_dimension_consistency(stack):
    pass

def test_stack_set_virtual(stack):
    pass

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
    with pytest.raises((TypeError, AttributeError)):
        stack[len(stack) // 2] = "nonsense"

def test_stack_delitem(stack):
    length = len(stack)
    del stack[length // 2 - 1: length // 2 + 1]
    assert len(stack) == length - 2


### Data integrity

def test_img_attr_types(img):
    if ".dat" in img.path:
        assert img.energy > -6
        assert img.rel_time > 0
    else:
        assert np.isnan(img.energy)
        assert np.isnan(img.rel_time)
    assert np.isnan(img.temperature) or img.temperature > -280
    assert isinstance(img.time_dtobject, datetime)

def test_img_setattr(img):
    img.energy = 7
    assert img.energy == 7
    with pytest.raises(AttributeError):
        img.meta = {}
    with pytest.raises(AttributeError):
        img.rel_time = 5
    img.data = np.ones_like(img.data)

def test_stack_getattr(stack):
    assert stack.energy.shape == (len(stack),)
    test_idx = len(stack) // 2
    assert stack.time[test_idx] == stack[test_idx].time
    assert same_or_nan(stack.energy[test_idx], stack[test_idx].energy)
    assert same_or_nan(stack.rel_time[test_idx], stack[test_idx].rel_time)

def test_stack_setattr(stack):
    with pytest.raises(ValueError):
        stack.energy = np.linspace(0, 10, len(stack) + 1)
    with pytest.raises(ValueError):
        stack.energy = "bla"
    stack.energy = np.linspace(0, 10, len(stack))
    assert stack.energy[-1] == 10
