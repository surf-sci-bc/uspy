"""Tests the SMART/BESSY HDF4 reader (uspy.leem.readers.smart).

These files are HDF4. The reader uses the optional `pyhdf` package when it is installed, but
otherwise falls back to the built-in pure-Python HDF4 reader (`uspy.leem.readers._hdf4`), so --
unlike before -- the file-reading tests run with or without `pyhdf`. A dedicated parity test
(`test_fallback_matches_pyhdf`) additionally asserts the two backends agree wherever `pyhdf`
happens to be installed, and `test_fallback_used_when_pyhdf_absent` forces the fallback path so
it is exercised even in an environment that *does* have `pyhdf`.
"""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import datetime
import shutil

import numpy as np
import pytest

from uspy.leem import base
from uspy.leem.readers import ElmitecReader, SmartReader
from uspy.leem.readers import smart
from uspy.leem.readers.smart import (
    _parse_fd_text,
    _n_frames,
    _energy_axis,
    _smart_timestamp,
    _metadata_dict,
)
from uspy.leem.readers._hdf4 import Hdf4File, Hdf4Error

from .conftest import TESTDATA_DIR

BESSY_DIR = TESTDATA_DIR + "bessy/"
SMART_IMG = BESSY_DIR + "0409a001.hdf"  # 1 frame, "Einzelbild", MICRO 2.0V
SMART_IMG2 = BESSY_DIR + "0409a002.hdf"  # 1 frame
SMART_STACK = BESSY_DIR + "0409aa.hdf"  # 161 frames, "EDC" energy scan, 10..40V
# A "TIME" scan split across 3 files (300 frames each) -- see the module docstring on
# "<base>.hdf"/"<base>_1.hdf"/"<base>_2.hdf" splitting.
SMART_SPLIT_GLOB = BESSY_DIR + "0409ac*.hdf"


### Registration works even without pyhdf installed


def test_smart_reader_is_registered():
    # SmartReader must self-register at import time regardless of whether pyhdf is
    # installed -- only *parsing* a real file should require it.
    from uspy.leem.readers.base import LEEMReader  # pylint: disable=import-outside-toplevel

    assert LEEMReader._registry["smart"] is SmartReader


def test_smart_handles():
    reader = SmartReader()
    assert reader.handles(SMART_IMG) is True
    assert reader.handles((SMART_IMG, 0)) is True
    assert reader.handles((SMART_IMG, 5)) is True
    assert reader.handles("somefile.dat") is False
    assert reader.handles((SMART_IMG, "0")) is False  # index must be an int
    assert reader.handles(("somefile.dat", 0)) is False
    assert reader.handles(123) is False


### Pure-Python helpers (no pyhdf/HDF4 needed -- these operate on plain strings/dicts)


def test_parse_fd_text_multiple_pairs_per_line():
    text = "IDENT = SMART;            VERSION = 0;\nX_SIZE = 1600;\n"
    fd = _parse_fd_text(text)
    assert fd == {"IDENT": "SMART", "VERSION": "0", "X_SIZE": "1600"}


def test_n_frames_from_points():
    assert _n_frames({"POINTS_1": "161"}) == 161
    assert _n_frames({"POINTS_1": "1"}) == 1
    assert _n_frames({}) == 1  # missing -> assume a single image


def test_energy_axis_linear_interpolation():
    fd = {"MICRO_START_1": "10.0", "MICRO_END_1": "40.0"}
    axis = _energy_axis(fd, "MICRO", 161)
    assert axis[0] == pytest.approx(10.0)
    assert axis[-1] == pytest.approx(40.0)
    assert len(axis) == 161
    np.testing.assert_allclose(axis, np.linspace(10.0, 40.0, 161))


def test_energy_axis_single_frame():
    fd = {"MICRO_START_1": "2.0", "MICRO_END_1": "2.0"}
    axis = _energy_axis(fd, "MICRO", 1)
    assert list(axis) == pytest.approx([2.0])


def test_energy_axis_missing_key_is_nan():
    axis = _energy_axis({}, "MICRO", 3)
    assert np.isnan(axis).all()


def test_energy_axis_mono_unused_is_nan():
    # SMART writes -1/-1 for MONO_START_1/MONO_END_1 on plain LEEM (no monochromator)
    # acquisitions -- must resolve to nan when negative_means_unused=True.
    fd = {"MONO_START_1": "-1.0", "MONO_END_1": "-1.0"}
    axis = _energy_axis(fd, "MONO", 5, negative_means_unused=True)
    assert np.isnan(axis).all()


def test_energy_axis_negative_start_voltage_not_treated_as_unused():
    # A genuinely negative MICRO start voltage (mirror-electron-microscopy mode) must NOT
    # be nan'd out -- negative_means_unused only applies to MONO (see smart.py docstring).
    fd = {"MICRO_START_1": "-5.0", "MICRO_END_1": "-2.0"}
    axis = _energy_axis(fd, "MICRO", 2)
    np.testing.assert_allclose(axis, [-5.0, -2.0])


def test_smart_timestamp_parses_date_and_time():
    ts = _smart_timestamp("9.Apr 2025", "9:32:01")
    dt = datetime.datetime.fromtimestamp(ts)
    assert (dt.year, dt.month, dt.day) == (2025, 4, 9)
    assert (dt.hour, dt.minute, dt.second) == (9, 32, 1)


def test_smart_timestamp_invalid_input_is_nan():
    assert np.isnan(_smart_timestamp("not a date", "not a time"))


def test_metadata_dict_timestamp_ramps_per_frame_within_a_file():
    # SMART writes only one DATE/TIME per file (the acquisition start), not a per-frame
    # clock -- _metadata_dict must estimate per-frame timestamps as start + index * exposure
    # so a multi-frame scan's rel_time (timestamp - time_origin) isn't degenerate (all the
    # same value). See the module docstring and _metadata_dict's "timestamp" comment.
    fd = {"DATE": "9.Apr 2025", "TIME": "16:26:49", "INTERATION_TIME": "1000", "POINTS_1": "5"}
    base_ts = _smart_timestamp(fd["DATE"], fd["TIME"])
    timestamps = [_metadata_dict(fd, i, 10, 10)["timestamp"] for i in range(5)]
    assert timestamps[0] == pytest.approx(base_ts)  # frame 0 is exactly the file's start time
    np.testing.assert_allclose(np.diff(timestamps), 1.0)  # one 1.0 s exposure apart
    assert timestamps == sorted(timestamps)


def test_metadata_dict_timestamp_falls_back_without_exposure():
    # Without a usable INTERATION_TIME there is nothing to estimate per-frame spacing from --
    # every frame in the file keeps the same (file-start) timestamp, same as before this fix.
    fd = {"DATE": "9.Apr 2025", "TIME": "16:26:49", "POINTS_1": "5"}
    base_ts = _smart_timestamp(fd["DATE"], fd["TIME"])
    timestamps = [_metadata_dict(fd, i, 10, 10)["timestamp"] for i in range(5)]
    assert all(ts == pytest.approx(base_ts) for ts in timestamps)


### The pure-Python HDF4 fallback reader (uspy.leem.readers._hdf4), tested directly. This runs
### regardless of whether pyhdf is installed -- it never uses pyhdf.


def test_hdf4file_reads_dims_and_dtype():
    hdf4 = Hdf4File(SMART_IMG)
    assert hdf4.shape == (1, 1200, 1600)  # single-frame file
    assert hdf4.dtype == np.dtype(">i2")  # SMART stores big-endian int16


def test_hdf4file_reads_annotations():
    hdf4 = Hdf4File(SMART_IMG)
    fd = _parse_fd_text(hdf4.file_description())
    assert fd["IDENT"] == "SMART"
    assert fd["SCAN_MODE"] == "Einzelbild"
    assert hdf4.file_label().endswith("0409a001.hdf")  # original acquisition path


def test_hdf4file_read_frame():
    frame = Hdf4File(SMART_IMG).read_frame(0)
    assert frame.shape == (1200, 1600)
    # A real 16-bit camera image: all counts non-negative, well below the int16 ceiling
    # (a wrong byte order would smear values across the full +-32768 range).
    assert frame.min() >= 0
    assert 0 < frame.max() < 32768


def test_hdf4file_frame_index_out_of_range():
    with pytest.raises(IndexError):
        Hdf4File(SMART_IMG).read_frame(1)  # single-frame file has only frame 0


def test_hdf4file_rejects_non_hdf4(tmp_path):
    bogus = tmp_path / "not.hdf"
    bogus.write_bytes(b"this is not an HDF4 file")
    with pytest.raises(Hdf4Error):
        Hdf4File(str(bogus))


def test_fallback_used_when_pyhdf_absent(monkeypatch):
    # Force the pure-Python fallback even where pyhdf is installed, so this path is always
    # exercised. Clearing the annotation cache avoids reusing a pyhdf-parsed result.
    monkeypatch.setattr(smart, "_pyhdf", lambda: None)
    smart._read_smart_annotations.cache_clear()

    img = base.LEEMImg(SMART_IMG, reader="smart")
    assert img.image.shape == (1200, 1600)
    assert img.image.dtype == np.float32
    assert img.energy == pytest.approx(2.0)
    assert img.ident == "SMART"
    smart._read_smart_annotations.cache_clear()


def test_fallback_matches_pyhdf():
    # Wherever pyhdf *is* installed, the fallback must produce byte-identical pixels and the
    # same metadata. Skipped (not failed) when pyhdf is unavailable.
    pytest.importorskip("pyhdf")
    from pyhdf.SD import SD  # pylint: disable=import-outside-toplevel
    from pyhdf.HDF import HDF  # pylint: disable=import-outside-toplevel
    from pyhdf.AN import AN_FILE_DESC  # pylint: disable=import-outside-toplevel

    sd_file = SD(SMART_IMG)
    ref = np.asarray(sd_file.select("Range0")[0, :, :])
    sd_file.end()

    hdf_file = HDF(SMART_IMG)
    annot = hdf_file.astart()
    ann = annot.select(0, AN_FILE_DESC)
    ref_desc = ann.annotation()
    ann.endaccess()
    annot.end()
    hdf_file.close()

    hdf4 = Hdf4File(SMART_IMG)
    assert np.array_equal(np.asarray(hdf4.read_frame(0)), ref)  # equal regardless of byteorder
    assert _parse_fd_text(hdf4.file_description()) == _parse_fd_text(ref_desc)


### Reading whole fixtures through the LEEMImg/LEEMStack API. These use whichever backend is
### active (pyhdf if installed, otherwise the pure-Python fallback).


def test_smart_reader_selected_by_name():
    img = base.LEEMImg(SMART_IMG, reader="smart")
    assert isinstance(img._reader, SmartReader)


def test_smart_single_image_shape_and_dtype():
    img = base.LEEMImg(SMART_IMG, reader="smart")
    assert img.image.shape == (1200, 1600)
    assert img.image.shape == (img.height, img.width)
    assert img.image.dtype == np.float32


def test_smart_single_image_metadata():
    img = base.LEEMImg(SMART_IMG, reader="smart")
    assert img.energy == pytest.approx(2.0)
    assert img.unit["energy"] == "eV"
    assert np.isnan(img.photon_energy)
    assert img.exposure == pytest.approx(1.0)  # INTERATION_TIME = 1000 ms -> 1.0 s
    dt = datetime.datetime.fromtimestamp(img.timestamp)
    assert (dt.year, dt.month, dt.day) == (2025, 4, 9)
    assert (dt.hour, dt.minute, dt.second) == (9, 32, 1)
    # not present in SMART metadata -- left at LEEMImg's own defaults
    assert np.isnan(img.temperature)
    assert np.isnan(img.pressure)


def test_smart_second_image_loads():
    img = base.LEEMImg(SMART_IMG2, reader="smart")
    assert img.image.shape == (1200, 1600)
    assert np.isfinite(img.energy)


def test_smart_parse_image_dict_directly():
    idict = SmartReader().parse_image(SMART_IMG)
    assert idict["image"].shape == (idict["height"], idict["width"])
    assert idict["image"].dtype == np.float32
    assert idict["energy_unit"] == "eV"
    assert idict["scan_mode"] == "Einzelbild"
    assert idict["ident"] == "SMART"


def test_smart_parse_metadata_matches_parse_image():
    # parse_metadata is a virtual (defer=True) LEEMImg's per-frame metadata source -- it must
    # return exactly what parse_image does, minus "image" itself,
    # for both a single-frame file and one frame of a multi-frame scan. nan-aware since e.g.
    # 'photon_energy' is nan for these fixtures (no monochromator energy).
    for source in (SMART_IMG, (SMART_STACK, 0), (SMART_STACK, 42)):
        image_dict = dict(SmartReader().parse_image(source))
        image_dict.pop("image")
        metadata_dict = SmartReader().parse_metadata(source)
        assert metadata_dict.keys() == image_dict.keys()
        for key, value in image_dict.items():
            other = metadata_dict[key]
            if isinstance(value, float) and np.isnan(value):
                assert np.isnan(other), key
            else:
                assert other == value, key


def test_smart_parse_metadata_reads_no_pixels(monkeypatch):
    calls = []
    original = smart._read_smart_frame
    monkeypatch.setattr(
        smart, "_read_smart_frame", lambda path, index: (calls.append(index), original(path, index))[1]
    )
    SmartReader().parse_metadata((SMART_STACK, 42))
    assert calls == []


def test_smart_binding_energy():
    img = base.LEEMImg(SMART_IMG, reader="smart")
    assert np.isnan(img.photon_energy)
    assert np.isnan(img.binding_energy)  # nan photon_energy -> nan binding_energy

    img.photon_energy = 100.0
    assert img.binding_energy == pytest.approx(100.0 - img.energy - img.energy_offset)


def test_smart_config_file_resolution(tmp_path):
    shutil.copy(SMART_IMG, tmp_path / "0409a001.hdf")
    (tmp_path / "uspy.toml").write_text('[uspy]\nreader = "smart"\n')

    img = base.LEEMImg(str(tmp_path / "0409a001.hdf"))  # no explicit reader=
    assert isinstance(img._reader, SmartReader)
    assert img.image.shape == (1200, 1600)


def test_smart_without_reader_does_not_parse():
    # No content-based auto-detection (see uspy.leem.readers module docstring): without an
    # explicit reader="smart" or a uspy.toml, a ".hdf" file falls through to the default
    # ElmitecReader (which only handles ".dat"), then Image.parse's generic image loader,
    # neither of which can read this binary format.
    with pytest.raises((FileNotFoundError, OSError)):
        base.LEEMImg(SMART_IMG)


def test_smart_folder_loads_multiple_files(tmp_path):
    shutil.copy(SMART_IMG, tmp_path / "a.hdf")
    shutil.copy(SMART_IMG2, tmp_path / "b.hdf")

    stack = base.LEEMStack(str(tmp_path), reader="smart")
    assert isinstance(stack._reader, SmartReader)
    assert len(stack) == 2
    for img in stack:
        assert isinstance(img, base.LEEMImg)
        assert img.image.shape == (1200, 1600)


def test_smart_list_files_expands_single_file_into_frames():
    # A folder containing one single-frame file expands to exactly one (path, 0) source --
    # exercised without touching the large 161-frame stack fixture.
    files = SmartReader().list_files(SMART_IMG)
    assert files == [(SMART_IMG, 0)]


def test_smart_rel_time_ramps_within_a_multiframe_file():
    # Regression test for the reported bug: loading a SMART multi-frame scan file
    # (LEEMStack(..., reader="smart")) gave an all-zero rel_time, since every frame shared
    # one per-file DATE/TIME. Uses parse_metadata (no pixel reads) to stay fast on the
    # 161-frame fixture -- mirrors test_smart_parse_metadata_matches_parse_image's use of
    # SMART_STACK without @pytest.mark.slow.
    indices = [0, 1, 80, 160]
    metas = [SmartReader().parse_metadata((SMART_STACK, i)) for i in indices]
    timestamps = [m["timestamp"] for m in metas]
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)  # not degenerate/constant anymore
    exposure = metas[0]["exposure"]
    assert timestamps[2] - timestamps[0] == pytest.approx(80 * exposure)  # index 80 vs 0


def test_smart_list_files_expands_split_scan_glob():
    # natsort places the un-suffixed base file before its numbered continuations (see module
    # docstring): "0409ac.hdf" < "0409ac_1.hdf" < "0409ac_2.hdf".
    sources = SmartReader().list_files(SMART_SPLIT_GLOB)
    assert len(sources) == 900  # 3 files x 300 frames
    assert sources[0] == (BESSY_DIR + "0409ac.hdf", 0)
    assert sources[300] == (BESSY_DIR + "0409ac_1.hdf", 0)
    assert sources[600] == (BESSY_DIR + "0409ac_2.hdf", 0)


def test_smart_rel_time_monotonic_across_split_scan_files():
    # Regression test for the reported bug: loading the whole split-scan glob gave rel_time
    # only 3 distinct values (one per file), since every frame within a file shared that
    # file's single DATE/TIME. Uses parse_metadata (no pixel reads) so this stays fast even
    # across all three 300-frame files.
    sources = SmartReader().list_files(SMART_SPLIT_GLOB)
    timestamps = [SmartReader().parse_metadata(s)["timestamp"] for s in sources]
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) > 3  # more than just the 3 per-file start times

    # Each file's own frame 0 still lands exactly on that file's recorded start time --
    # the per-frame estimate re-anchors to the real clock at each file boundary.
    assert timestamps[0] == pytest.approx(_smart_timestamp("9.Apr 2025", "17:45:51"))
    assert timestamps[300] == pytest.approx(_smart_timestamp("9.Apr 2025", "17:50:54"))
    assert timestamps[600] == pytest.approx(_smart_timestamp("9.Apr 2025", "17:55:56"))


def test_smart_default_reader_is_still_elmitec(img_fname):
    # Sanity check that adding SmartReader doesn't change default resolution for existing
    # (non-SMART) fixtures.
    img = base.LEEMImg(img_fname)
    assert isinstance(img._reader, ElmitecReader)


### The 161-frame, 618 MB I(V) stack -- marked slow (needs `pytest --runslow`).


@pytest.mark.slow
def test_smart_stack_length_and_energy_axis():
    stack = base.LEEMStack(SMART_STACK, reader="smart")
    assert len(stack) == 161
    assert stack[0].energy == pytest.approx(10.0)
    assert stack[-1].energy == pytest.approx(40.0)
    np.testing.assert_allclose(list(stack.energy), np.linspace(10.0, 40.0, 161))
    assert all(img.image.shape == (1200, 1600) for img in stack)


@pytest.mark.slow
def test_smart_stack_virtual_mode():
    stack = base.LEEMStack(SMART_STACK, reader="smart", virtual=True)
    assert stack.virtual is True
    img = stack[0]  # lazily constructed
    assert isinstance(img, base.LEEMImg)
    assert isinstance(img._reader, SmartReader)
    assert img.energy == pytest.approx(10.0)


@pytest.mark.slow
def test_smart_parse_frame_index_directly():
    # midpoint frame of the 10..40V, 161-point scan
    idict = SmartReader().parse_image((SMART_STACK, 80))
    assert idict["energy"] == pytest.approx(25.0)
    assert idict["image"].shape == (1200, 1600)
