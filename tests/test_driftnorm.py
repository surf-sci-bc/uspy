"""Tests the uspy.leem.driftnorm module."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import numpy as np
import pytest

from uspy.leem import driftnorm


### Normalization

def test_normalize_image(img, mcp, dark_counts):
    """dark_counts_img and mcp_img have the same shape in all fixtures, so it
    is not tested here!
    """
    if img.image.shape != mcp.image.shape:
        with pytest.raises(ValueError):
            img_normed = driftnorm.normalize_image(img, mcp, dark_counts=dark_counts)
    else:
        img_normed = driftnorm.normalize_image(img, mcp, dark_counts=dark_counts)
        assert img_normed.image.shape == img.image.shape
        assert img_normed.image.mean() <= img.image.mean()
        assert not np.isnan(img_normed.image).any()
        assert not (img_normed.image < 0).any()

def test_normalize_stack(stack, mcp, dark_counts):
    stack = stack[10:15]
    if stack[0].image.shape != mcp.image.shape:
        with pytest.raises(ValueError):
            stack_normed = driftnorm.normalize_stack(stack, mcp, dark_counts=dark_counts)
    stack_normed = driftnorm.normalize_stack(stack, mcp, dark_counts=dark_counts)
    assert stack_normed[0].image.shape == stack[0].image.shape
    assert stack_normed[0].image.mean() <= stack[0].image.mean()
    assert not (stack_normed[0].image < 0).any()

@pytest.mark.parametrize("trafo", ["translation", "affine"])
@pytest.mark.parametrize("mask_outer", [0.2, 0.4])
def test_find_alignment_matrices_sift(stack, trafo, mask_outer):
    stack = stack[10:15]
    alignment = driftnorm.find_alignment_matrices(stack, trafo=trafo)
    assert np.array(alignment).shape == (len(stack), 3, 3)

@pytest.mark.parametrize("mask_outer", [0.2, 0.4])
def test_find_alignment_matrices_ecc(stack, mask_outer):
    stack = stack[10:15]
    alignment = driftnorm.find_alignment_matrices(stack)
    assert np.array(alignment).shape == (len(stack), 3, 3)

def test_apply_alignment_matrices(stack):
    stack = stack[10:15]
    alignment = driftnorm.find_alignment_matrices(stack)
    stack2 = driftnorm.apply_alignment_matrices(stack, alignment)
    assert stack2 != stack
    assert len(stack2) == len(stack)