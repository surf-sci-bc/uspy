"""Fixtures for testing the uspy module."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import os

import pytest
import numpy as np
import tifffile

from uspy.leem.base import LEEMImg, LEEMStack
from uspy.dataobject import Line
from uspy.dataobject import Image


# tests/legacy/ tests the unmaintained uspy.legacy package (XPS, STM, pre-dataobject LEEM
# base_old, driftnorm, rsm) -- none of it runs against the current uspy API. See
# uspy/legacy/__init__.py; normalize/align functionality lives on LEEMImg/LEEMStack directly
# (see test_leembase.py).
collect_ignore = ["legacy"]


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

# An Elettra beamline stack whose per-frame energy/temperature/timestamp are NOT in the
# per-image ".tif" sidecars but distributed from a single shared "*trace.txt" tracefile by
# ElettraReader.post_process_stack -- exercises a metadata path the plain per-file Elmitec
# ".dat" fixtures above (STACK_FNAMES) never touch. See tests/test_virtual.py.
ELETTRA_TIF_DIR = TESTDATA_DIR + "elettra/2024_09_26_004_XPS/"

# Per-frame ".tiff" fixtures (one 2D image per file, as opposed to a single 3D-TIFF stack)
# reproducing a reported bug: frames here deliberately have TWO DIFFERENT pixel shapes
# ((600, 624) and (624, 600)), so a correct loader must reject the whole glob/folder rather
# than silently reshaping frames into a single, wrong shape (see tests/test_leembase.py,
# tests/test_virtual.py).
TIFF_STACK_DIR = TESTDATA_DIR + "tiff_stack/"
TIFF_STACK_GLOB = TIFF_STACK_DIR + "*.tiff"

# A SMART/BESSY multi-frame HDF4 scan file (an I(V) energy series, 161 points in one file --
# see uspy.leem.readers.smart) -- exercises the "(path, frame_index)" tuple-source path and
# computed properties (e.g. rel_time) that aren't themselves stored metadata.
BESSY_SCAN_FNAME = TESTDATA_DIR + "bessy/0409aa.hdf"

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


@pytest.fixture()
def tiff_stack_glob():
    """Glob pattern for the mixed-shape per-frame ".tiff" fixture at `testdata/tiff_stack/`
    (see TIFF_STACK_DIR above). Skips if the testdata submodule/fixture isn't present, like
    `elettra_tif_fnames` does."""
    import glob

    if not glob.glob(TIFF_STACK_GLOB):
        pytest.skip(f"no tiff_stack fixture files found at {TIFF_STACK_DIR}")
    return TIFF_STACK_GLOB


@pytest.fixture()
def tiff_frame_stack(tmp_path):
    """A small set of same-shape, single-frame ".tiff" files on disk, standing in for a real
    per-frame TIFF export. Filenames (`frame_1`, `frame_2`, `frame_10`) deliberately diverge
    between natural-sort and lexicographic-sort order, so tests can tell natsort-based file
    discovery apart from a plain `sorted()`. Each frame holds a distinct constant pixel value
    so mis-ordering or a wrong frame being read is directly observable, without depending on
    the (large, binary) testdata submodule.

    Returns a dict with the containing directory, a "*.tiff" glob pattern into it, the file
    paths in natural-sort order, and the pixel array written to each (same order).
    """
    names = ["frame_1.tiff", "frame_2.tiff", "frame_10.tiff"]
    shape = (5, 4)
    paths, arrays = [], []
    for i, name in enumerate(names):
        arr = np.full(shape, float(i + 1), dtype=np.float32)
        path = tmp_path / name
        tifffile.imwrite(str(path), arr)
        paths.append(str(path))
        arrays.append(arr)
    return {
        "dir": str(tmp_path),
        "glob": str(tmp_path / "*.tiff"),
        "paths": paths,
        "arrays": arrays,
    }


@pytest.fixture(scope="module")
def elettra_tif_fnames():
    """Explicit, sorted list of per-frame ".tif" paths for the Elettra tracefile fixture --
    not the bare folder (which also holds ".dat" siblings that ElettraReader.list_files
    prefers, so the folder alone wouldn't exercise the tracefile-only metadata) and not a
    "*.tif" glob string (kept as a separate, deliberately-glob test case instead)."""
    import glob

    fnames = sorted(glob.glob(ELETTRA_TIF_DIR + "*.tif"))
    if not fnames:
        pytest.skip(f"no Elettra .tif fixture files found at {ELETTRA_TIF_DIR}")
    return fnames


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
