"""Tests the agfalta.leem.driftnorm module."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

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
