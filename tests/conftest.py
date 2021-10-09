"""Fixtures for testing the uspy module."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import os

import pytest
import numpy as np

from uspy.leem.base import LEEMImg, LEEMStack
from uspy.dataobject import Line
from uspy.dataobject import Image


######### Utility


def same_or_nan(x, y):
    return np.isnan([x, y]).any() or x == y


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


######### Fixtures

TESTDATA_DIR = os.path.dirname(__file__) + "/../testdata/"

STACK_FNAMES = [
    TESTDATA_DIR + "test_stack_IV_RuO2",
    TESTDATA_DIR + "test_stack_IV_g-Cu",
    TESTDATA_DIR + "test_stack_IV_RuO2_normed_aligned_80-140.tif",
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

ARRAY_2D = [np.array([[1, 2], [3, 4]]), np.ones((2, 2)), np.eye(2)]

ARRAY_1D = [
    np.linspace(0, 100, 11),
    np.zeros(3),
    np.array([-1.0, -2.0, -3.0, -4.0, -5.0]),
]

TESTIMAGE = np.eye(10) * 38550
TESTIMAGE_NAME = "test_gray_16bit"
TESTIMAGE_ENDING = [".png", ".tif"]

# Create Testimages


@pytest.fixture(scope="module", params=ARRAY_1D)
def list1darrays(request):
    return request.param.tolist()


@pytest.fixture()
def list2darrays(request):
    return ARRAY_2D


@pytest.fixture()
def testimgpng():
    return Image(TESTDATA_DIR + TESTIMAGE_NAME + ".png")

@pytest.fixture()
def quad_line():
    a = np.linspace(0,10,11)
    b = a**2
    c = np.asarray([a,b])

    return Line(c)

@pytest.fixture(scope="module", params=IMG_FNAMES)
def img_fname(request):
    return request.param


@pytest.fixture(scope="module")
def img(img_fname):
    return LEEMImg(img_fname)

@pytest.fixture()
def normed_img():
    return LEEMImg(TESTDATA_DIR+"bremen_normed.tif")


@pytest.fixture(scope="module", params=STACK_FNAMES)
def stack_fname(request):
    return request.param


@pytest.fixture(scope="module", params=STACK_FNAMES[0:2])
def stack_folder(request):
    return request.param


@pytest.fixture(scope="module")
def stack(stack_fname):
    return LEEMStack(stack_fname)


@pytest.fixture(scope="module")
def single_stack():
    stack = LEEMStack(TESTDATA_DIR + "test_stack_IV_RuO2_normed_aligned_80-140.tif")
    stack.energy = np.linspace(3.0, 50.0, len(stack))
    return stack


@pytest.fixture(scope="module")
def short_stack():
    stack = LEEMStack(TESTDATA_DIR + "test_stack_IV_RuO2_normed_aligned_80-140.tif")
    stack.energy = np.linspace(19, 28.81, len(stack))
    return stack[10:20]


@pytest.fixture(scope="module", params=(DC_IMG_FNAME, LEEMImg(DC_IMG_FNAME), 100, 439))
def dark_counts(request):
    return request.param


@pytest.fixture(scope="module", params=(LEEMImg(MCP_IMG_FNAME),))
def mcp(request):
    return request.param
