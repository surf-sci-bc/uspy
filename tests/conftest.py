"""Fixtures for testing the uspy module."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import os

import matplotlib

# Headless test runs must never resolve to a GUI backend (macosx/qt/tk...) -- set before any
# module under test gets the chance to `import matplotlib.pyplot` and trigger backend
# resolution. Individual test files (e.g. test_plotting.py) also set this themselves; doing
# it once here too covers modules like uspy.leem.cluster that import pyplot as a side effect
# of module import, before any test-local `matplotlib.use("Agg")` would run.
matplotlib.use("Agg")

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

# testdata/ is grouped one folder per instrument/beamline (see testdata/dataset_info.txt for
# provenance of the ALBA fixture).
BREMEN_DIR = TESTDATA_DIR + "bremen/"  # Elmitec LEEM, Bremen home lab -- 600x624
ALBA_DIR = TESTDATA_DIR + "alba/"  # ALBA beamtime, Elmitec -- 1024x1024
ELETTRA_DIR = TESTDATA_DIR + "elettra/"  # Elettra Nanospectroscopy -- 464x464
BESSY_DIR = TESTDATA_DIR + "bessy/"  # SMART @ BESSY, HDF4
SYNTH_DIR = TESTDATA_DIR + "synthetic/"  # instrument-independent test images

BREMEN_IMG_FNAME = BREMEN_DIR + "bremen.dat"
BREMEN2_IMG_FNAME = BREMEN_DIR + "bremen2.dat"
BREMEN_NORMED_FNAME = BREMEN_DIR + "bremen_normed.tif"
MCP_IMG_FNAME = BREMEN_DIR + "channelplate.dat"
DC_IMG_FNAME = BREMEN_DIR + "dark_counts_bremen.dat"
RUO2_STACK_DIR = BREMEN_DIR + "test_stack_IV_RuO2"
GCU_STACK_DIR = BREMEN_DIR + "test_stack_IV_g-Cu"
RUO2_STACK_TIF = BREMEN_DIR + "test_stack_IV_RuO2_normed_aligned_80-140.tif"

ALBA_IMG_FNAME = ALBA_DIR + "alba.dat"

ELETTRA_DAT_FNAME = ELETTRA_DIR + "elettra.dat"
ELETTRA_TIF_FNAME = ELETTRA_DIR + "elettra.tif"

STACK_FNAMES = [RUO2_STACK_DIR, GCU_STACK_DIR, RUO2_STACK_TIF]
STACK_INCOMPATIBLE_IMG_FNAME = ELETTRA_DAT_FNAME

# An Elettra beamline stack whose per-frame energy/temperature/timestamp are NOT in the
# per-image ".tif" sidecars but distributed from a single shared "*trace.txt" tracefile by
# ElettraReader.post_process_stack -- exercises a metadata path the plain per-file Elmitec
# ".dat" fixtures above (STACK_FNAMES) never touch. See tests/test_virtual.py.
ELETTRA_TIF_DIR = ELETTRA_DIR + "2024_09_26_004_XPS/"

# Per-frame ".tiff" fixtures (one 2D image per file, as opposed to a single 3D-TIFF stack),
# all sharing one pixel shape (600, 624). The mixed-shape case ("a loader must reject frames
# that can't form one stack") is covered by the synthetic `mixed_shape_tiff_stack` fixture
# below instead -- see its docstring for why.
TIFF_STACK_DIR = BREMEN_DIR + "tiff_stack/"
TIFF_STACK_GLOB = TIFF_STACK_DIR + "*.tiff"

# A SMART/BESSY multi-frame HDF4 scan file (a handful of leading frames of a real I(V) energy
# series, shrunk by scripts/shrink_bessy_fixtures.py -- see uspy.leem.readers.smart) --
# exercises the "(path, frame_index)" tuple-source path and computed properties (e.g. rel_time)
# that aren't themselves stored metadata.
BESSY_SCAN_FNAME = BESSY_DIR + "0409aa.hdf"

IMG_FNAMES = [
    BREMEN_IMG_FNAME,
    MCP_IMG_FNAME,
    ALBA_IMG_FNAME,
    ELETTRA_DAT_FNAME,
    ELETTRA_TIF_FNAME,
    BREMEN2_IMG_FNAME,
]
IMG_FNAMES_COMPATIBLE = [
    BREMEN_IMG_FNAME,
    BREMEN2_IMG_FNAME,
]

ARRAY_2D = [np.array([[1, 2], [3, 4]]), np.ones((2, 2)), np.eye(2)]

ARRAY_1D = [
    np.linspace(0, 100, 11),
    np.zeros(3),
    np.array([-1.0, -2.0, -3.0, -4.0, -5.0]),
]

TESTIMAGE = np.eye(10) * 38550
TESTIMAGE_BASE = SYNTH_DIR + "test_gray_16bit"
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
    return Image(TESTIMAGE_BASE + ".png")

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
    return LEEMImg(BREMEN_NORMED_FNAME)


@pytest.fixture(scope="module", params=STACK_FNAMES)
def stack_fname(request):
    return request.param


@pytest.fixture(scope="module", params=STACK_FNAMES[0:2])
def stack_folder(request):
    return request.param


@pytest.fixture()
def tiff_stack_glob():
    """Glob pattern for the real per-frame ".tiff" fixture at `testdata/bremen/tiff_stack/`
    (see TIFF_STACK_DIR above), all sharing one pixel shape. Skips if the testdata
    submodule/fixture isn't present, like `elettra_tif_fnames` does. For the mixed-shape
    regression case, see `mixed_shape_tiff_stack` below instead."""
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


@pytest.fixture()
def mixed_shape_tiff_stack(tmp_path):
    """Per-frame ".tiff" files whose pixel shapes deliberately disagree -- (5, 4) and (4, 5),
    two frames each. The two shapes hold the same number of elements on purpose: that is what
    made the original bug possible (a frame silently `reshape`d into another frame's shape
    instead of the mismatched set being rejected outright), so a test can still prove no
    loaded frame equals a differently-shaped frame reshaped. Synthetic and in `tmp_path`
    rather than a testdata fixture: `testdata/bremen/tiff_stack/` used to hold real frames at
    two shapes, but that mismatch was an export accident, not a deliberate fixture, so fixing
    it upstream would have silently deleted this regression coverage.

    Returns a dict with the containing directory, a "*.tiff" glob pattern into it, the file
    paths, the pixel array written to each, and each array's shape (all parallel lists).
    """
    shapes = [(5, 4), (5, 4), (4, 5), (4, 5)]
    paths, arrays = [], []
    for i, shape in enumerate(shapes):
        arr = np.full(shape, float(i + 1), dtype=np.float32)
        path = tmp_path / f"frame_{i}.tiff"
        tifffile.imwrite(str(path), arr)
        paths.append(str(path))
        arrays.append(arr)
    return {
        "dir": str(tmp_path),
        "glob": str(tmp_path / "*.tiff"),
        "paths": paths,
        "arrays": arrays,
        "shapes": shapes,
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
    stack = LEEMStack(RUO2_STACK_TIF)
    stack.energy = np.linspace(3.0, 50.0, len(stack))
    return stack


@pytest.fixture(scope="module")
def short_stack():
    stack = LEEMStack(RUO2_STACK_TIF)
    stack.energy = np.linspace(19, 28.81, len(stack))
    return stack[10:20]


@pytest.fixture(scope="module", params=(DC_IMG_FNAME, LEEMImg(DC_IMG_FNAME), 100, 439))
def dark_counts(request):
    return request.param


@pytest.fixture(scope="module", params=(LEEMImg(MCP_IMG_FNAME),))
def mcp(request):
    return request.param


@pytest.fixture()
def synthetic_leem_stack():
    """Factory building a small in-memory LEEMStack from synthetic pixel data, with metadata
    fields settable by keyword. Complements `single_stack`/`short_stack` (which only set
    `energy`): `uspy.leem.cluster`/`uspy.leem.processing`/`uspy.plotting` tests often need
    `pressure1`, `rel_time`, or `binding_energy` too, none of which those real-data fixtures
    carry.

    Parameters (all keyword, passed to the returned factory)
    ----------
    n : int
        Number of frames (default 5).
    shape : tuple[int, int]
        Per-frame pixel shape (default (20, 24)).
    data : np.ndarray, optional
        Explicit (n, *shape) pixel array; overrides `n`/`shape` if given.
    **meta
        Extra metadata, broadcast onto every element per
        `DataObjectStack.__setattr__`'s scalar/length-matched-sequence rules (see
        uspy/dataobject/base.py) -- e.g. `energy=np.linspace(...)`, `pressure1=1e-6`.

    Returns
    -------
    callable
        `factory(n=5, shape=(20, 24), data=None, **meta) -> LEEMStack`
    """

    def _make(n=5, shape=(20, 24), data=None, **meta):
        if data is None:
            data = np.random.default_rng(0).normal(size=(n, *shape)).astype(np.float64)
        stack = LEEMStack(np.asarray(data, dtype=np.float64))
        for key, value in meta.items():
            setattr(stack, key, value)
        return stack

    return _make


@pytest.fixture()
def blob_vectors():
    """Factory for a deterministic (n_pixels, n_frames) design matrix with well-separated,
    known clusters -- for uspy.leem.cluster tests that need clustering/decomposition to
    recover a ground truth. Each cluster's rows are a shared base curve plus a per-cluster
    offset far larger than the added noise, plus a small per-cluster label array.

    Returns
    -------
    callable
        `factory(n_clusters=3, n_per_cluster=40, n_frames=6, spacing=50.0, seed=0)
        -> (X, labels)`
    """

    def _make(n_clusters=3, n_per_cluster=40, n_frames=6, spacing=50.0, seed=0):
        rng = np.random.default_rng(seed)
        base = np.linspace(1.0, 2.0, n_frames)
        rows, labels = [], []
        for k in range(n_clusters):
            cluster_rows = base + k * spacing + rng.normal(
                scale=0.01, size=(n_per_cluster, n_frames)
            )
            rows.append(cluster_rows)
            labels.extend([k] * n_per_cluster)
        X = np.vstack(rows)
        return X, np.array(labels)

    return _make
