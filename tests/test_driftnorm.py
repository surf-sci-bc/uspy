"""Tests the agfalta.leem.driftnorm module."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import numpy as np
import pytest

from agfalta.leem import driftnorm


### Normalization

def test_normalize_image(img, mcp, dark_counts):
    """dark_counts_img and mcp_img have the same shape in all fixtures, so it
    is not tested here!
    """
    if img.data.shape != mcp.data.shape:
        with pytest.raises(ValueError):
            img_normed = driftnorm.normalize_image(img, mcp, dark_counts=dark_counts)
    else:
        img_normed = driftnorm.normalize_image(img, mcp, dark_counts=dark_counts)
        assert img_normed.data.shape == img.data.shape
        assert img_normed.data.mean() <= img.data.mean()
        assert not np.isnan(img_normed.data).any()
        assert not (img_normed.data < 0).any()

def normalize_stack(stack, mcp, dark_counts):
    stack = stack[10:20]
    if stack[0].data.shape != mcp.data.shape:
        with pytest.raises(ValueError):
            stack_normed = driftnorm.normalize_stack(stack, mcp, dark_counts=dark_counts)
    stack_normed = driftnorm.normalize_stack(stack, mcp, dark_counts=dark_counts)
    assert stack_normed[0].data.shape == stack[0].data.shape
    assert stack_normed[0].data.mean() <= stack[0].data.mean()
    assert not (stack_normed[0].data < 0).any()
