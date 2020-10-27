"""Tests the agfalta.leem.driftnorm module."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import os

import pytest

from agfalta.leem import base
from agfalta.leem import driftnorm


TESTDATA_DIR = os.path.dirname(__file__) + "/../testdata/"


########## Fixtures

stack_fnames = [
    TESTDATA_DIR + "test_stack_ca",
    TESTDATA_DIR + "Stack",
]
tif_stack_fnames = [
    TESTDATA_DIR + "normed_aligned_stack.tif",
]
image_fnames = [
    TESTDATA_DIR + "channelplate.dat",
    TESTDATA_DIR + "bremen.dat",
    TESTDATA_DIR + "alba.dat",
    TESTDATA_DIR + "elettra.dat",
    TESTDATA_DIR + "elettra.tif",
    TESTDATA_DIR + "Reinigung002000.dat",
]

@pytest.fixture(params=stack_fnames+tif_stack_fnames)
def stack(request):
    return base.LEEMStack(request.param)

@pytest.fixture(params=image_fnames)
def img(request):
    return base.LEEMImg(request.param)

@pytest.fixture
def mcp():
    return base.LEEMImg(TESTDATA_DIR + "channelplate.dat")

dark_counts_img_fname = TESTDATA_DIR + "dark_counts_bremen.dat"
@pytest.fixture
def dark_counts_img():
    return base.LEEMImg(dark_counts_img_fname)


########### Test functions

### normalization

@pytest.mark.parametrize("dark_counts", [dark_counts_img_fname, 100, 500, 1e6])
def test_normalize_image(img, mcp, dark_counts):
    # dark_counts_img and mcp_img have the same shape in all fixtures, so it
    # is not tested here!
    if img.data.shape != mcp.data.shape:
        with pytest.raises(ValueError):
            img_normed = driftnorm.normalize_image(img, mcp, dark_counts=dark_counts)
    else:
        img_normed = driftnorm.normalize_image(img, mcp, dark_counts=dark_counts)
        assert img_normed.data.shape == img.data.shape
        assert img_normed.data.mean() <= img.data.mean()
