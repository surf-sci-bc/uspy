"""Tests the pluggable LEEM reader mechanism (uspy.leem.readers / uspy.leem.base)."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import datetime
import os
import shutil
import struct

import numpy as np
import pytest

from uspy.leem import base, elettra
from uspy.leem.readers import (
    ElettraReader,
    ElmitecReader,
    resolve_reader,
    set_reader,
    get_reader,
    reset_reader,
)
from uspy.leem.readers.elmitec import parse_dat, ATTR_NAMES
from uspy.leem.readers.smart import SmartReader
from uspy.utility import parse_bytes

from .conftest import TESTDATA_DIR

ELETTRA_DAT = TESTDATA_DIR + "elettra.dat"


@pytest.fixture(autouse=True)
def _no_session_reader_leak():
    """The session-wide `set_reader` override is deliberately global mutable state --
    make sure no test can leak it into another."""
    reset_reader()
    yield
    reset_reader()


### Default resolution (no reader given, no config file present) -> Elmitec


def test_default_img_reader_is_elmitec(img_fname):
    img = base.LEEMImg(img_fname)
    assert isinstance(img._reader, ElmitecReader)


def test_default_stack_reader_is_elmitec(stack_folder):
    stack = base.LEEMStack(stack_folder)
    assert isinstance(stack._reader, ElmitecReader)


### Explicit reader selection


def test_explicit_reader_by_name():
    img = base.LEEMImg(ELETTRA_DAT, reader="elettra")
    assert isinstance(img._reader, ElettraReader)
    assert isinstance(img.image, np.ndarray)
    assert img.image.shape == (img.height, img.width)


def test_explicit_reader_instance_is_used_directly():
    reader = ElettraReader()
    img = base.LEEMImg(ELETTRA_DAT, reader=reader)
    assert img._reader is reader


def test_explicit_reader_propagates_through_stack(tmp_path):
    # two copies of the same .dat file so a 2-element stack can be built
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_000.dat")
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_001.dat")

    stack = base.LEEMStack(str(tmp_path), reader="elettra")
    assert isinstance(stack._reader, ElettraReader)
    assert len(stack) == 2
    for img in stack:
        assert isinstance(img._reader, ElettraReader)


### Config-file-driven resolution (uspy.toml dropped into the data folder)


def test_config_file_resolves_reader(tmp_path):
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_000.dat")
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_001.dat")
    (tmp_path / "uspy.toml").write_text('[uspy]\nreader = "elettra"\n')

    stack = base.LEEMStack(str(tmp_path))  # no explicit reader=
    assert isinstance(stack._reader, ElettraReader)
    assert len(stack) == 2


def test_config_file_options_passed_to_reader(tmp_path):
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_000.dat")
    (tmp_path / "uspy.toml").write_text(
        '[uspy]\nreader = "elettra"\n\n[uspy.options]\nenergy_offset = 2.5\n'
    )

    img = base.LEEMImg(str(tmp_path / "elettra_000.dat"))
    assert isinstance(img._reader, ElettraReader)
    assert img._reader.options == {"energy_offset": 2.5}


def test_no_config_file_defaults_to_elmitec(tmp_path):
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_000.dat")
    img = base.LEEMImg(str(tmp_path / "elettra_000.dat"))
    assert isinstance(img._reader, ElmitecReader)


### Config search path: data-dir parent, cwd, and a global config directory (in addition
### to the data folder itself, already covered above), plus their precedence and merging.


def test_config_file_resolved_from_data_folder_parent(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(ELETTRA_DAT, data_dir / "elettra_000.dat")
    (tmp_path / "uspy.toml").write_text('[uspy]\nreader = "elettra"\n')

    img = base.LEEMImg(str(data_dir / "elettra_000.dat"))
    assert isinstance(img._reader, ElettraReader)


def test_config_file_resolved_from_cwd(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(ELETTRA_DAT, data_dir / "elettra_000.dat")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "uspy.toml").write_text('[uspy]\nreader = "elettra"\n')
    monkeypatch.chdir(cwd)

    img = base.LEEMImg(str(data_dir / "elettra_000.dat"))
    assert isinstance(img._reader, ElettraReader)


def test_config_file_resolved_from_global_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(ELETTRA_DAT, data_dir / "elettra_000.dat")
    global_dir = tmp_path / "xdg" / "uspy"
    global_dir.mkdir(parents=True)
    (global_dir / "uspy.toml").write_text('[uspy]\nreader = "elettra"\n')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)  # cwd itself must not carry a config file

    img = base.LEEMImg(str(data_dir / "elettra_000.dat"))
    assert isinstance(img._reader, ElettraReader)


def test_config_search_path_precedence_data_dir_beats_others(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(ELETTRA_DAT, data_dir / "elettra_000.dat")
    (data_dir / "uspy.toml").write_text('[uspy]\nreader = "elettra"\n')
    (tmp_path / "uspy.toml").write_text('[uspy]\nreader = "smart"\n')
    global_dir = tmp_path / "xdg" / "uspy"
    global_dir.mkdir(parents=True)
    (global_dir / "uspy.toml").write_text('[uspy]\nreader = "smart"\n')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    img = base.LEEMImg(str(data_dir / "elettra_000.dat"))
    assert isinstance(img._reader, ElettraReader)


def test_config_search_path_precedence_parent_beats_cwd_and_global(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(ELETTRA_DAT, data_dir / "elettra_000.dat")
    (tmp_path / "uspy.toml").write_text('[uspy]\nreader = "elettra"\n')  # parent
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "uspy.toml").write_text('[uspy]\nreader = "smart"\n')
    global_dir = tmp_path / "xdg" / "uspy"
    global_dir.mkdir(parents=True)
    (global_dir / "uspy.toml").write_text('[uspy]\nreader = "smart"\n')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(cwd)

    img = base.LEEMImg(str(data_dir / "elettra_000.dat"))
    assert isinstance(img._reader, ElettraReader)


def test_config_search_path_layered_merge_reader_and_plotting(tmp_path, monkeypatch):
    # global uspy.toml fixes the reader; the (more specific) data folder's uspy.toml
    # only overrides plotting fields -- both should take effect together.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(ELETTRA_DAT, data_dir / "elettra_000.dat")
    (data_dir / "uspy.toml").write_text(
        '[uspy.plotting]\ndefault_fields = ["fov"]\n'
    )
    global_dir = tmp_path / "xdg" / "uspy"
    global_dir.mkdir(parents=True)
    (global_dir / "uspy.toml").write_text('[uspy]\nreader = "elettra"\n')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)

    img = base.LEEMImg(str(data_dir / "elettra_000.dat"))
    assert isinstance(img._reader, ElettraReader)
    assert img.default_fields == ("fov",)


### Programmatic session override (set_reader/get_reader/reset_reader)


def test_set_reader_overrides_config_file(tmp_path):
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_000.dat")
    (tmp_path / "uspy.toml").write_text('[uspy]\nreader = "elmitec"\n')

    set_reader("elettra")
    img = base.LEEMImg(str(tmp_path / "elettra_000.dat"))
    assert isinstance(img._reader, ElettraReader)


def test_set_reader_accepts_instance_and_options():
    reader = ElettraReader(energy_offset=1.5)
    set_reader(reader)
    assert get_reader() == (reader, {})
    assert resolve_reader(ELETTRA_DAT) is reader

    set_reader("elettra", energy_offset=2.5)
    resolved = resolve_reader(ELETTRA_DAT)
    assert isinstance(resolved, ElettraReader)
    assert resolved.options == {"energy_offset": 2.5}


def test_explicit_reader_argument_beats_set_reader():
    set_reader("smart")
    img = base.LEEMImg(ELETTRA_DAT, reader="elettra")
    assert isinstance(img._reader, ElettraReader)


def test_reset_reader_restores_config_resolution(tmp_path):
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_000.dat")
    (tmp_path / "uspy.toml").write_text('[uspy]\nreader = "elettra"\n')

    set_reader("smart")
    reset_reader()
    assert get_reader() == (None, {})
    img = base.LEEMImg(str(tmp_path / "elettra_000.dat"))
    assert isinstance(img._reader, ElettraReader)


### Config-file-driven plotting field defaults (uspy.toml [uspy.plotting])


def test_no_config_file_keeps_elmitec_plotting_defaults(img_fname):
    img = base.LEEMImg(img_fname)
    assert img.default_fields == ("temperature", "pressure", "energy", "fov")
    assert img.meta_fields == ("temperature",)
    assert img.print_fields == (
        "energy",
        "temperature",
        "pressure1",
        "pressure2",
        "objective",
        "fov",
        "exposure",
        "averaging",
        "width",
        "height",
    )


def test_config_file_overrides_default_fields(tmp_path):
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_000.dat")
    (tmp_path / "uspy.toml").write_text(
        '[uspy.plotting]\ndefault_fields = ["energy", "fov"]\n'
    )
    img = base.LEEMImg(str(tmp_path / "elettra_000.dat"))
    assert img.default_fields == ("energy", "fov")
    # keys not set in the config file keep their Elmitec fallback
    assert img.meta_fields == ("temperature",)


def test_config_file_partial_override_keeps_other_defaults(tmp_path):
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_000.dat")
    (tmp_path / "uspy.toml").write_text(
        '[uspy.plotting]\nmeta_fields = ["pressure"]\n'
    )
    img = base.LEEMImg(str(tmp_path / "elettra_000.dat"))
    assert img.meta_fields == ("pressure",)
    assert img.default_fields == ("temperature", "pressure", "energy", "fov")
    assert img.print_fields[0] == "energy"


def test_config_file_plotting_fields_propagate_through_stack(tmp_path):
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_000.dat")
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_001.dat")
    (tmp_path / "uspy.toml").write_text(
        '[uspy.plotting]\ndefault_fields = ["fov"]\nprint_fields = ["energy"]\n'
    )

    stack = base.LEEMStack(str(tmp_path))
    assert len(stack) == 2
    for img in stack:
        assert img.default_fields == ("fov",)
        assert img.print_fields == ("energy",)


### Registry


def test_resolve_reader_by_name():
    reader = resolve_reader(ELETTRA_DAT, reader="elmitec")
    assert isinstance(reader, ElmitecReader)


def test_resolve_reader_unknown_name_raises():
    with pytest.raises(KeyError):
        resolve_reader(ELETTRA_DAT, reader="not-a-registered-reader")


def test_resolve_reader_default_is_elmitec():
    reader = resolve_reader("some/nonexistent/path.dat")
    assert isinstance(reader, ElmitecReader)


### Elettra .tif + sidecar-metadata parsing (synthetic sidecars, since the checked-in
### testdata fixture for elettra.tif has none)


@pytest.fixture()
def elettra_tif_folder(tmp_path):
    shutil.copy(TESTDATA_DIR + "elettra.tif", tmp_path / "test_000.tif")
    shutil.copy(TESTDATA_DIR + "elettra.tif", tmp_path / "test_001.tif")
    (tmp_path / "test.camera.txt").write_text(
        "width 464x464\nexposure info 500\naveraging 4\n"
    )
    (tmp_path / "test.microscope.txt").write_text("Operation Mode, LEEM10\n")
    (tmp_path / "test.beamline.txt").write_text(
        "Monochromator energy,a,b,74.5\nMesh,a,b,1.2,V\n"
    )
    return tmp_path


def test_elettra_tif_parses_sidecar_metadata(elettra_tif_folder):
    img = base.LEEMImg(str(elettra_tif_folder / "test_000.tif"), reader="elettra")
    assert isinstance(img.image, np.ndarray)
    assert img.photon_energy == pytest.approx(74.5)
    assert img.unit["photon_energy"] == "eV"
    # a lone .tif has no "Start Voltage"/energy of its own (only known once a stack's
    # tracefile is read, see test_elettra_tracefile_distributes_stack_metadata), so
    # binding_energy is legitimately nan here -- just confirm it computes without error.
    assert np.isnan(img.binding_energy)


def test_elettra_tracefile_distributes_stack_metadata(elettra_tif_folder):
    (elettra_tif_folder / "test.trace.txt").write_text(
        "time_ms STV_eV\n1000 10.0\n2000 20.0\n"
    )

    stack = base.LEEMStack(str(elettra_tif_folder), reader="elettra")
    assert len(stack) == 2
    # tracefile timestamps are in ms -> converted to s
    assert list(stack.timestamp) == pytest.approx([1.0, 2.0])
    assert list(stack.energy) == pytest.approx([10.0, 20.0])
    assert stack.time_origin == pytest.approx(1.0)
    # now that energy is known (from the tracefile) and photon_energy from the sidecar
    # beamline file, binding_energy is finite for every image in the stack
    assert np.all(np.isfinite(stack.binding_energy))


### Leak fix: binding_energy/kinetic_energy/photon_energy are universal on LEEMImg,
### regardless of which reader was used.


def test_binding_energy_available_on_default_reader(img_fname):
    img = base.LEEMImg(img_fname)
    assert np.isnan(img.photon_energy)  # no beamline reader -> default nan
    np.testing.assert_equal(img.kinetic_energy, img.energy)
    assert np.isnan(img.binding_energy)  # nan - energy - 0 is still nan

    img.photon_energy = 100.0
    np.testing.assert_equal(
        img.binding_energy, 100.0 - img.energy - img.energy_offset
    )


### Deprecated uspy.leem.elettra shim


def test_elettra_module_shim_still_works():
    with pytest.deprecated_call():
        img = elettra.LEEMImg(ELETTRA_DAT)
    assert isinstance(img, base.LEEMImg)
    assert isinstance(img._reader, ElettraReader)


### Real XPEEM fixture (testdata/elettra/2024_09_26_004_XPS): 51 .dat files, 51 .tif
### files, and all four sidecar files (camera/microscope/beamline/trace) from a real
### measurement at the Nanospectroscopy beamline -- an end-to-end check of the reading
### routine against real data, rather than synthetic sidecars.

XPEEM_FOLDER = TESTDATA_DIR + "elettra/2024_09_26_004_XPS"
XPEEM_DAT_000 = XPEEM_FOLDER + "/2024_09_26_004_000.dat"
XPEEM_TIF_000 = XPEEM_FOLDER + "/2024_09_26_004_000.tif"


@pytest.fixture(scope="module")
def xpeem_dat_stack():
    return base.LEEMStack(XPEEM_FOLDER, reader="elettra")


@pytest.fixture(scope="module")
def xpeem_tif_stack():
    return base.LEEMStack(XPEEM_FOLDER + "/*.tif", reader="elettra")


def test_xpeem_default_reader_prefers_dat():
    # a bare folder with both .dat and .tif present prefers .dat (more robust format)
    files = ElettraReader().list_files(XPEEM_FOLDER)
    assert len(files) == 51
    assert all(f.endswith(".dat") for f in files)


def test_xpeem_single_dat_image_without_tracefile():
    # a single file parsed in isolation reflects that file's own embedded metadata --
    # not yet corrected by the stack-level tracefile (see test_xpeem_dat_stack_* below)
    img = base.LEEMImg(XPEEM_DAT_000, reader="elettra")
    assert img.image.shape == (560, 560)
    assert img.energy == pytest.approx(578.0)
    assert img.timestamp == pytest.approx(1727332959.58)
    assert img.photon_energy == pytest.approx(249.989686773)


def test_xpeem_single_tif_image_without_tracefile():
    img = base.LEEMImg(XPEEM_TIF_000, reader="elettra")
    assert img.image.shape == (560, 560)
    assert img.energy == pytest.approx(222.0)  # from the microscope.txt sidecar
    assert img.photon_energy == pytest.approx(249.989686773)
    # no tracefile yet, but a sibling .dat with the same basename exists, so its real
    # embedded timestamp is used rather than the (unreliable) file mtime
    assert img.timestamp == pytest.approx(1727332959.58)


def test_xpeem_dat_stack_length_and_reader(xpeem_dat_stack):
    assert isinstance(xpeem_dat_stack._reader, ElettraReader)
    assert len(xpeem_dat_stack) == 51
    assert all(img.image.shape == (560, 560) for img in xpeem_dat_stack)


def test_xpeem_dat_stack_tracefile_overrides_metadata(xpeem_dat_stack):
    # the tracefile's logged per-image values take priority over each .dat file's own
    # embedded metadata (contrast with the raw single-file energy of 578.0 above)
    assert xpeem_dat_stack[0].energy == pytest.approx(216.0)
    assert xpeem_dat_stack[0].temperature == pytest.approx(19.2317790985)
    assert xpeem_dat_stack[-1].energy == pytest.approx(226.0)
    assert xpeem_dat_stack[-1].temperature == pytest.approx(20.7337303162)


def test_xpeem_dat_stack_timestamp_is_a_real_epoch_time(xpeem_dat_stack):
    # regression test: this tracefile has no "time" column, so the per-image timestamp
    # (already correct, from parse_dat) must NOT be divided by 1000 (see
    # ElettraReader.post_process_stack)
    ts = xpeem_dat_stack[0].timestamp
    assert ts == pytest.approx(1727332959.58)
    assert datetime.datetime.fromtimestamp(ts).year == 2024
    assert xpeem_dat_stack.time_origin == pytest.approx(ts)


def test_xpeem_dat_stack_binding_energy_finite(xpeem_dat_stack):
    assert np.all(np.isfinite(xpeem_dat_stack.binding_energy))


def test_xpeem_tif_stack_matches_dat_stack_after_tracefile(xpeem_dat_stack, xpeem_tif_stack):
    # both formats read the same underlying measurement and share the same tracefile, so
    # post-tracefile metadata should agree regardless of source format
    assert len(xpeem_tif_stack) == len(xpeem_dat_stack)
    np.testing.assert_allclose(xpeem_tif_stack.energy, xpeem_dat_stack.energy)
    np.testing.assert_allclose(xpeem_tif_stack.temperature, xpeem_dat_stack.temperature)


def test_xpeem_default_reader_still_loads_dat_without_beamline_metadata():
    # even without reader="elettra", the .dat files are still valid Elmitec binaries and
    # load fine -- just without photon_energy/tracefile enrichment (graceful degradation)
    stack = base.LEEMStack(XPEEM_FOLDER)
    assert isinstance(stack._reader, ElmitecReader)
    assert len(stack) == 51
    assert stack[0].energy == pytest.approx(578.0)  # raw .dat value, no tracefile applied
    assert np.isnan(stack[0].photon_energy)


def test_xpeem_virtual_stack_preserves_reader(tmp_path):
    # virtual=True with a tracefile-having folder is not supported (see
    # test_xpeem_virtual_stack_with_tracefile_raises below), so use a copy without one
    for i in range(2):
        name = f"2024_09_26_004_{i:03d}"
        shutil.copy(f"{XPEEM_FOLDER}/{name}.dat", tmp_path / f"{name}.dat")
    shutil.copy(
        f"{XPEEM_FOLDER}/2024_09_26_004_beamline.txt",
        tmp_path / "2024_09_26_004_beamline.txt",
    )

    stack = base.LEEMStack(str(tmp_path), reader="elettra", virtual=True)
    assert isinstance(stack._reader, ElettraReader)
    img = stack[0]  # lazily constructed
    assert isinstance(img._reader, ElettraReader)
    assert img.photon_energy == pytest.approx(249.989686773)


def test_xpeem_virtual_stack_with_tracefile_raises():
    # Known limitation (pre-existing, not introduced by the reader refactor): the
    # tracefile mechanism distributes metadata by broadcasting setattr() across all
    # stack elements, which DataObjectStack explicitly disallows for virtual stacks.
    # Documented here so this doesn't silently change; not fixed as part of this change
    # since it would require reworking lazy per-image construction for virtual mode.
    with pytest.raises(ValueError):
        base.LEEMStack(XPEEM_FOLDER, reader="elettra", virtual=True)


def test_xpeem_stack_slice_preserves_reader(xpeem_dat_stack):
    sliced = xpeem_dat_stack[:5]
    assert len(sliced) == 5
    assert isinstance(sliced._reader, ElettraReader)

    # Slicing must not re-run post_process_stack against the full (now mismatched-length)
    # tracefile: that used to attach the *whole* un-sliced trace column to the slice as a
    # plain instance attribute, silently shadowing the per-element gather (sliced.energy
    # would return all 51 values instead of the correctly-sliced 5). Confirm the gather
    # matches the corresponding subset of the full stack instead.
    np.testing.assert_allclose(sliced.energy, xpeem_dat_stack.energy[:5])
    np.testing.assert_allclose(sliced.temperature, xpeem_dat_stack.temperature[:5])
    np.testing.assert_array_equal(sliced.ImageNum, xpeem_dat_stack.ImageNum[:5])
    for i in range(5):
        assert sliced[i].energy == xpeem_dat_stack[i].energy


def test_xpeem_config_file_resolution(tmp_path):
    # copy a few real files (with their shared sidecars, but no tracefile) into a
    # scratch folder and let a plain LEEMStack (no explicit reader=) discover the reader
    # via uspy.toml
    for i in range(3):
        name = f"2024_09_26_004_{i:03d}"
        shutil.copy(f"{XPEEM_FOLDER}/{name}.dat", tmp_path / f"{name}.dat")
    for sidecar in ("beamline", "camera", "microscope"):
        shutil.copy(
            f"{XPEEM_FOLDER}/2024_09_26_004_{sidecar}.txt",
            tmp_path / f"2024_09_26_004_{sidecar}.txt",
        )
    (tmp_path / "uspy.toml").write_text('[uspy]\nreader = "elettra"\n')

    stack = base.LEEMStack(str(tmp_path))
    assert isinstance(stack._reader, ElettraReader)
    assert len(stack) == 3
    # beamline sidecar metadata is picked up per-image even without a tracefile
    assert stack[0].photon_energy == pytest.approx(249.989686773)


### Real SMART fixture (testdata/bessy/0409ac*.hdf): a "SCAN_MODE = TIME" series that didn't
### fit in a single HDF4 file, split by SMART into "0409ac.hdf" (first part) + "0409ac_1.hdf"
### + "0409ac_2.hdf" (continuations) -- an end-to-end check that globbing/reading a split scan
### by its basename reassembles one correctly-ordered stack. These fixtures are large (~1.15GB
### each) and not tracked in the testdata submodule, so this section skips cleanly when absent.

BESSY_SPLIT_BASE = TESTDATA_DIR + "bessy/0409ac"

pytestmark_bessy_split = pytest.mark.skipif(
    not os.path.exists(BESSY_SPLIT_BASE + ".hdf"),
    reason="testdata/bessy/0409ac*.hdf fixtures not present locally",
)


@pytestmark_bessy_split
@pytest.mark.slow
def test_bessy_split_scan_basename_merges_all_parts_in_order():
    # the un-suffixed base file sorts before its "_1"/"_2" siblings under natsort (since
    # "." < "_"), so a bare basename globs and concatenates the parts in acquisition order
    sources = SmartReader().list_files(BESSY_SPLIT_BASE)
    paths = list(dict.fromkeys(path for path, _index in sources))
    assert paths == [
        BESSY_SPLIT_BASE + ".hdf",
        BESSY_SPLIT_BASE + "_1.hdf",
        BESSY_SPLIT_BASE + "_2.hdf",
    ]
    assert len(sources) == 900  # 300 frames/file * 3 files


@pytestmark_bessy_split
@pytest.mark.slow
def test_bessy_split_scan_exact_base_file_loads_only_that_file():
    # known, documented limitation: the exact "<base>.hdf" path is not expanded to its
    # siblings -- only the basename/glob form loads a whole split scan (see the module
    # docstring of uspy.leem.readers.smart)
    sources = SmartReader().list_files(BESSY_SPLIT_BASE + ".hdf")
    assert len(sources) == 300
    assert {path for path, _index in sources} == {BESSY_SPLIT_BASE + ".hdf"}


@pytestmark_bessy_split
@pytest.mark.slow
def test_bessy_split_scan_virtual_stack_length_and_order():
    # virtual=True keeps elements as unloaded (path, frame_index) sources, so this check
    # doesn't have to load ~7GB of pixel data just to confirm the merge is correct
    stack = base.LEEMStack(BESSY_SPLIT_BASE, reader="smart", virtual=True)
    assert isinstance(stack._reader, SmartReader)
    assert len(stack) == 900
    # timestamps are per-file (not per-frame), so they step up once per file boundary but
    # must never go backwards across the seams between parts
    timestamps = [stack[i].timestamp for i in (0, 299, 300, 599, 600, 899)]
    assert all(a <= b for a, b in zip(timestamps, timestamps[1:]))


### parse_dat: the Elmitec UKSOFT2001 binary format parser (uspy.leem.readers.elmitec)

DAT_FNAMES = [
    TESTDATA_DIR + "bremen.dat",
    TESTDATA_DIR + "channelplate.dat",
    TESTDATA_DIR + "alba.dat",
    TESTDATA_DIR + "elettra.dat",
    TESTDATA_DIR + "dark_counts_bremen.dat",
]


@pytest.fixture(scope="module", params=DAT_FNAMES)
def dat_fname(request):
    return request.param


def test_parse_dat_image_shape_and_dtype(dat_fname):
    data = parse_dat(dat_fname)
    assert data["image"].dtype == np.float32
    assert data["image"].shape == (data["height"], data["width"])
    # a real 16-bit-per-pixel LEEM frame should be within uint16 range and finite
    assert np.isfinite(data["image"]).all()
    assert data["image"].min() >= 0
    assert data["image"].max() <= 65535


def test_parse_dat_energy_unit_is_ev(dat_fname):
    data = parse_dat(dat_fname)
    assert data["energy_unit"] == "eV"


def test_parse_dat_canonical_attr_names_present(dat_fname):
    data = parse_dat(dat_fname)
    # every canonical name in ATTR_NAMES must be present, even if nan (base.py:150-152)
    for canonical in ATTR_NAMES:
        assert canonical in data
    assert isinstance(data["timestamp"], (int, float))


def test_parse_dat_averaging_remap(dat_fname):
    data = parse_dat(dat_fname)
    if "averaging" in data:
        # raw 0 -> 1 (non-averaged), raw 255 -> 0 (sliding average); elmitec.py:153-159
        assert isinstance(data["averaging"], (int, float))
        assert data["averaging"] != 255


def test_parse_dat_fov_parsing(dat_fname):
    data = parse_dat(dat_fname)
    if "fov" in data:
        assert isinstance(data["fov"], str)
        assert isinstance(data.get("preset", ""), str)


def test_parse_dat_nonexistent_file_raises():
    with pytest.raises((FileNotFoundError, OSError)):
        parse_dat(TESTDATA_DIR + "nonexistent_file.dat")


### parse_bytes: the low-level binary field decoder (uspy.utility)


def test_parse_bytes_short():
    buffer = struct.pack("<h", -1234)
    assert parse_bytes(buffer, 0, "short") == -1234


def test_parse_bytes_float():
    buffer = struct.pack("<f", 3.5)
    assert parse_bytes(buffer, 0, "float") == pytest.approx(3.5)


def test_parse_bytes_bool():
    assert parse_bytes(struct.pack("<?", True), 0, "bool") is True
    assert parse_bytes(struct.pack("<?", False), 0, "bool") is False


def test_parse_bytes_string_encoding():
    buffer = "hello".encode("cp1252") + b"\x00trailing garbage"
    assert parse_bytes(buffer, 0, "cp1252") == "hello"


def test_parse_bytes_time_windows_filetime_to_unix():
    # WIN FILETIME epoch is 1601-01-01 UTC; unix epoch is 11644473600s later. Encode a
    # FILETIME value of exactly the unix epoch (in 100ns units) and confirm parse_bytes
    # decodes it back to unix timestamp 0.
    win_100ns_ticks = 11644473600 * 10**7
    buffer = struct.pack("<Q", win_100ns_ticks)
    assert parse_bytes(buffer, 0, "time") == pytest.approx(0.0, abs=1e-3)


def test_parse_bytes_position_offset():
    # a leading pad byte before the real value must be skipped via pos
    buffer = b"\x00" + struct.pack("<h", 42)
    assert parse_bytes(buffer, 1, "short") == 42
