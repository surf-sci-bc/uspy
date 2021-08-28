"""Tests the agfalta.leem.base module."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import cv2 as cv
import pytest
import numpy as np

from agfalta.leem import base

from .conftest import (
    MCP_IMG_FNAME,
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


def test_img_normalize(normed_img):
    img = base.LEEMImg(TESTDATA_DIR + "bremen.dat")
    mcp = MCP_IMG_FNAME
    img = img.normalize(mcp=mcp, dark_counts=100)
    assert (img.image == normed_img.image).all()


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
