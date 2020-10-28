"""Fixtures for testing the agfalta module."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import os

import pytest
import numpy as np

from agfalta.leem.base import LEEMImg, LEEMStack


######### Utility

def same_or_nan(x, y):
    return np.isnan([x, y]).any() or x == y


######### Fixtures

TESTDATA_DIR = os.path.dirname(__file__) + "/../testdata/"

STACK_FNAMES = [
    TESTDATA_DIR + "test_stack_IV_RuO2",
    TESTDATA_DIR + "test_stack_IV_g-Cu",
    TESTDATA_DIR + "test_stack_IV_RuO2_normed_aligned.tif",
]
STACK_INCOMPATIBLE_IMG_FNAME = TESTDATA_DIR + "elettra.dat"

IMG_FNAMES = [
    TESTDATA_DIR + "bremen.dat",
    TESTDATA_DIR + "channelplate.dat",
    TESTDATA_DIR + "alba.dat",
    TESTDATA_DIR + "elettra.dat",
    TESTDATA_DIR + "elettra.tif",
    TESTDATA_DIR + "bremen2.dat",
]
IMG_FNAMES_COMPATIBLE = [
    TESTDATA_DIR + "bremen.dat",
    TESTDATA_DIR + "bremen2.dat",
]

DC_IMG_FNAME = TESTDATA_DIR + "dark_counts_bremen.dat"
MCP_IMG_FNAME = TESTDATA_DIR + "channelplate.dat"


@pytest.fixture(scope="module", params=IMG_FNAMES)
def img_fname(request):
    return request.param

@pytest.fixture(scope="module")
def img(img_fname):
    return LEEMImg(img_fname)


@pytest.fixture(scope="module", params=STACK_FNAMES)
def stack_fname(request):
    return request.param

@pytest.fixture(scope="module")
def stack(stack_fname):
    return LEEMStack(stack_fname)


@pytest.fixture(scope="module", params=(DC_IMG_FNAME, LEEMImg(DC_IMG_FNAME), 100, 439))
def dark_counts(request):
    return request.param

@pytest.fixture(scope="module", params=(LEEMImg(MCP_IMG_FNAME), ))
def mcp(request):
    return request.param
