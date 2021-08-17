"""Tests the agfalta.leem.base module."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import pytest
import numpy as np

from agfalta.leem import base

from .conftest import (
    same_or_nan,
    IMG_FNAMES_COMPATIBLE,
    STACK_FNAMES,
    STACK_INCOMPATIBLE_IMG_FNAME,
    TESTDATA_DIR,
)
### Image loading


#@pytest.mark.parametrize("nolazy", [True, False])
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
@pytest.mark.skip(reason="Takes long and seems unproblematic")
def test_img_copying(img):
    img2 = img.copy()
    assert img2 == img
    assert img2 is not img

def test_imge_constructor_nonsense():
    with pytest.raises(FileNotFoundError):
        _ = base.LEEMImg("nonsense")
    with pytest.raises(FileNotFoundError):
        _ = base.LEEMImg("nonsense.dat")
    #img = base.LEEMImg("nonsense.dat")
    #with pytest.raises(FileNotFoundError):
    #    assert img.image

@pytest.mark.parametrize("virtual", [False, True])
def test_stack_constructor(stack_fname, virtual):
    stack = base.LEEMStack(stack_fname, virtual=virtual)
    assert isinstance(stack[0], base.LEEMImg)
    assert stack[0].image.shape == stack[-1].image.shape

def test_stack_constructor_globbing(stack_folder):
    stack1 = base.LEEMStack(stack_folder, virtual=True)
    stack2 = base.LEEMStack(stack_folder+"/*.dat", virtual=True)
    stack3 = base.LEEMStack(stack_folder+"/", virtual=True)
    assert len(stack1) == len(stack2) == len(stack3)
    for i in [0, -1]:
        assert stack1[i] == stack2[i]
        assert stack1[i] == stack3[i]
    stack3 = base.LEEMStack(IMG_FNAMES_COMPATIBLE[0])
    assert len(stack3) == 1