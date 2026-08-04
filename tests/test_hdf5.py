"""Tests for uspy.hdf5 -- NeXus-flavoured HDF5 export/import.

These tests are written against the design agreed before any implementation existed (see the
approved plan), so at first they fail to even collect (uspy.hdf5 doesn't exist yet). That's
intentional -- implementation is added afterwards, in phases, until each group below goes green.

Guiding claims under test:

1. `obj.save("x.h5")` / `Cls.load("x.h5")` round-trips pixels, every metadata value (with its
   Python type), and every unit -- for `Image`, `LEEMImg`, `Line`, `ImageStack`, `LEEMStack`,
   and the lazy out-of-core stacks.
2. Derived types (`IntensityLine`, `StitchedLine`, `ProfileLine`, `IVCurve`, `Waterfall`) export
   *as their parent class* (`Line`/`Image`) -- an archive holds data, not the analysis object
   graph. Whatever can't be represented (upstream stacks/ROIs/profiles) is dropped with a loud
   warning at write time, and leaves no trace in the file.
2. The archive is deliberately uspy-agnostic: two groups (`/entry/data`, `/entry/metadata`) plus
   standard NeXus scaffolding, and exactly three `uspy_*` attributes in the whole file
   (`uspy_class`, `uspy_version`, `uspy_format_version`). No dropped-field stubs, no per-dataset
   type tags -- the Python type is inferred from the HDF5 dtype itself.
3. Never modify source data (rule 1), never silently change/lose information (rule 3) -- every
   loss (a dropped field, a missing key filled with nan, a unit that differs across frames) is
   warned about loudly rather than swallowed.

Prefer synthetic fixtures (`tmp_path`, in-memory arrays) throughout -- only the one test at the
bottom marked `slow` touches the `testdata` git submodule, and only to read it.
"""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import h5py

from uspy.dataobject.base import DataObject
from uspy.dataobject.image import Image, ImageStack
from uspy.dataobject.line import Line
from uspy.leem.base import LEEMImg, LEEMStack
from uspy.leem.readers.base import LEEMReader
from uspy.derived import IntensityLine, ProfileLine, Waterfall
from uspy import roi
from uspy.serialize import Loadable

import uspy.hdf5 as h5mod

from .conftest import STACK_FNAMES


######### Synthetic fixtures (no testdata submodule needed) ####################################


def make_leem_img(value=1.0, shape=(6, 5), dtype=np.float32, **meta):
    arr = np.full(shape, value, dtype=dtype)
    img = LEEMImg(arr)
    for key, val in meta.items():
        setattr(img, key, val)
    return img


def make_line(n=5, xdim="U", ydim="I", x_unit="V", y_unit="A"):
    x = np.linspace(0.0, 1.0, n)
    y = x**2
    line = Line(np.stack([x, y]))
    line.xdim = xdim
    line.ydim = ydim
    line._units["x"] = x_unit
    line._units["y"] = y_unit
    return line


def make_leem_stack(n=4, shape=(6, 5), energies=None):
    if energies is None:
        energies = np.linspace(10.0, 10.0 + n - 1, n)
    imgs = []
    for i in range(n):
        img = make_leem_img(
            value=float(i),
            shape=shape,
            energy=float(energies[i]),
            temperature=25.0,
            fov="10 µm",
            averaging=4,
            mirror_state=(i % 2 == 0),
            timestamp=1_700_000_000.0 + 10.0 * i,
        )
        imgs.append(img)
    return LEEMStack(imgs)


@pytest.fixture()
def leem_img():
    return make_leem_img(energy=30.0, temperature=25.5, fov="10 µm")


@pytest.fixture()
def line():
    return make_line()


@pytest.fixture()
def leem_stack():
    return make_leem_stack()


######### Section A: extension dispatch / regression ###########################################


class TestDispatch:
    def test_is_hdf5_path_recognizes_extensions(self):
        for ext in (".h5", ".hdf5", ".nxs", ".H5", ".HDF5", ".NXS"):
            assert h5mod.is_hdf5_path("x" + ext)

    def test_is_hdf5_path_rejects_hdf4_and_others(self):
        for ext in (".hdf", ".he5", ".tif", ".pickle", ".json", ".csv"):
            assert not h5mod.is_hdf5_path("x" + ext)

    def test_hdf4_smart_reader_not_confused_with_hdf5(self):
        from uspy.leem.readers.smart import SmartReader

        reader = SmartReader()
        assert reader.handles("x.h5") is False
        assert "x.h5" not in reader.extensions

    def test_loadable_load_unknown_extension_raises_valueerror(self, tmp_path):
        target = tmp_path / "thing.unknownext"
        target.write_bytes(b"not a real file")
        with pytest.raises(ValueError):
            Loadable.load(str(target))

    def test_loadable_save_kwargs_on_non_h5_target_raises_typeerror(self, line, tmp_path):
        # Goes through Line.save's else-branch -> Loadable.save directly. Image.save has its
        # own kwargs behavior -- see TestImageSaveDispatch below.
        with pytest.raises(TypeError):
            line.save(str(tmp_path / "x.pickle"), compression="gzip")

    def test_leemstack_save_kwargs_on_non_h5_target_raises_typeerror(self, leem_stack, tmp_path):
        with pytest.raises(TypeError):
            leem_stack.save(str(tmp_path / "x.pickle"), compression="gzip")

    def test_image_save_png_unchanged(self, tmp_path):
        img = Image(np.full((4, 4), 100, dtype=np.uint16))
        target = tmp_path / "x.png"
        img.save(str(target))
        assert target.exists()

    @pytest.mark.parametrize("ext", (".h5", ".hdf5", ".nxs"))
    def test_extension_roundtrip_all_three(self, leem_img, tmp_path, ext):
        target = tmp_path / ("archive" + ext)
        leem_img.save(str(target))
        reloaded = LEEMImg.load(str(target))
        np.testing.assert_array_equal(reloaded.image, leem_img.image)


######### Section A2: Image.save extension dispatch (bug fix) ###################################

# Regression coverage for a confirmed bug: imageio 2.37.2's imwrite() brute-forces every
# registered plugin for an unrecognized extension instead of raising, so Image.save's old
# `try: imageio.imwrite(...) except ValueError: super().save(...)` fallback never actually
# reached the pickle/JSON fallback -- it silently wrote TIFF bytes under any filename, including
# ".pickle"/".json". Image.save now dispatches on an explicit extension whitelist instead of
# relying on imageio to fail.

_TIFF_MAGICS = (b"II*\x00", b"MM\x00*")


class TestImageSaveDispatch:
    def _img(self):
        return Image(np.full((4, 4), 12345, dtype=np.uint16))

    def _file_backed_img(self, tmp_path):
        # DataObject.__json_encode__ (uspy/dataobject/base.py:299-301) stores only `_source`
        # and re-parses on load ("by design", per CLAUDE.md: .json is not self-contained for
        # Image, unlike .pickle/.h5) -- so a JSON round-trip needs an Image backed by a real
        # file path, not one built directly from an ndarray (_source is None there).
        src = tmp_path / "source.png"
        self._img().save(str(src))
        return Image(str(src))

    @pytest.mark.parametrize("ext", (".pickle", ".pickle.bz2"))
    def test_serialize_extensions_are_not_silently_written_as_tiff(self, tmp_path, ext):
        img = self._img()
        target = tmp_path / ("x" + ext)
        img.save(str(target))
        assert target.exists()
        head = target.read_bytes()[:4]
        assert head not in _TIFF_MAGICS, (
            f"Image.save({ext!r}) wrote TIFF-magic bytes {head!r} -- the imageio fallback bug "
            "is back."
        )
        reloaded = Image.load(str(target))
        np.testing.assert_array_equal(reloaded.image, img.image)

    def test_json_extension_is_not_silently_written_as_tiff(self, tmp_path):
        img = self._file_backed_img(tmp_path)
        target = tmp_path / "x.json"
        img.save(str(target))
        assert target.exists()
        head = target.read_bytes()[:4]
        assert head not in _TIFF_MAGICS, (
            f"Image.save('.json') wrote TIFF-magic bytes {head!r} -- the imageio fallback bug "
            "is back."
        )
        reloaded = Image.load(str(target))
        np.testing.assert_array_equal(reloaded.image, img.image)

    def test_image_save_junk_extension_raises(self, tmp_path):
        img = self._img()
        target = tmp_path / "x.notarealformat"
        with pytest.raises(ValueError):
            img.save(str(target))
        assert not target.exists()

    def test_image_save_png_roundtrip(self, tmp_path):
        img = self._img()
        target = tmp_path / "x.png"
        img.save(str(target))
        reloaded = Image(str(target))
        np.testing.assert_array_equal(reloaded.image, img.image)

    def test_image_save_h5_accepts_kwargs(self, tmp_path):
        img = self._img()
        target = tmp_path / "x.h5"
        img.save(str(target), compression="gzip")
        reloaded = Image.load(str(target))
        np.testing.assert_array_equal(reloaded.image, img.image)


######### Section B: coercion (private _encode_value / _decode_value) ##########################


class TestCoercion:
    @pytest.mark.parametrize(
        "value",
        [0.0, 1.5, -3.25, float("nan"), float("inf"), float("-inf")],
    )
    def test_float_roundtrip(self, tmp_path, value):
        with h5py.File(tmp_path / "t.h5", "w") as f:
            h5mod._encode_value(f, "v", value)
        with h5py.File(tmp_path / "t.h5", "r") as f:
            out = h5mod._decode_value(f["v"])
        assert type(out) is float
        if np.isnan(value):
            assert np.isnan(out)
        else:
            assert out == value

    @pytest.mark.parametrize("value", [0, 1, -7, 12345])
    def test_int_roundtrip(self, tmp_path, value):
        with h5py.File(tmp_path / "t.h5", "w") as f:
            h5mod._encode_value(f, "v", value)
        with h5py.File(tmp_path / "t.h5", "r") as f:
            out = h5mod._decode_value(f["v"])
        assert type(out) is int
        assert out == value

    @pytest.mark.parametrize("value", [True, False])
    def test_bool_roundtrip(self, tmp_path, value):
        with h5py.File(tmp_path / "t.h5", "w") as f:
            h5mod._encode_value(f, "v", value)
        with h5py.File(tmp_path / "t.h5", "r") as f:
            out = h5mod._decode_value(f["v"])
        assert type(out) is bool
        assert out is value

    @pytest.mark.parametrize(
        "value", ["", "plain", "10 µm", "°C", "Å", "e⁻/s"]
    )
    def test_str_roundtrip(self, tmp_path, value):
        with h5py.File(tmp_path / "t.h5", "w") as f:
            h5mod._encode_value(f, "v", value)
        with h5py.File(tmp_path / "t.h5", "r") as f:
            ds = f["v"]
            # bare h5py: a vlen string DATASET decodes to bytes unless .asstr() is used --
            # this is the exact asymmetry the design docs called out.
            assert isinstance(ds[()], bytes)
            assert ds[()].decode("utf-8") == value
            out = h5mod._decode_value(ds)
        assert type(out) is str
        assert out == value

    def test_ndarray_roundtrip_exact_dtype(self, tmp_path):
        arr = np.eye(3, dtype=np.float64)
        with h5py.File(tmp_path / "t.h5", "w") as f:
            h5mod._encode_value(f, "v", arr)
        with h5py.File(tmp_path / "t.h5", "r") as f:
            out = h5mod._decode_value(f["v"])
        assert isinstance(out, np.ndarray)
        assert out.dtype == arr.dtype
        np.testing.assert_array_equal(out, arr)

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.float32, np.float64])
    def test_ndarray_roundtrip_dtype_matrix(self, tmp_path, dtype):
        arr = np.arange(12, dtype=dtype).reshape(3, 4)
        with h5py.File(tmp_path / "t.h5", "w") as f:
            h5mod._encode_value(f, "v", arr)
        with h5py.File(tmp_path / "t.h5", "r") as f:
            out = h5mod._decode_value(f["v"])
        assert out.dtype == dtype
        np.testing.assert_array_equal(out, arr)

    def test_none_is_not_written(self, tmp_path):
        with h5py.File(tmp_path / "t.h5", "w") as f:
            written = h5mod._encode_value(f, "v", None)
            assert written is False
            assert "v" not in f

    def test_list_of_scalars_becomes_ndarray(self, tmp_path):
        with h5py.File(tmp_path / "t.h5", "w") as f:
            written = h5mod._encode_value(f, "v", [1.0, 2.0, 3.0])
            assert written is True
        with h5py.File(tmp_path / "t.h5", "r") as f:
            out = h5mod._decode_value(f["v"])
        np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))

    def test_list_of_incompatible_objects_is_not_written(self, tmp_path):
        class Thing:
            pass

        with h5py.File(tmp_path / "t.h5", "w") as f:
            written = h5mod._encode_value(f, "v", [Thing(), Thing()])
            assert written is False
            assert "v" not in f

    @pytest.mark.parametrize(
        "value",
        [
            roi.ROI.circle(2, 2, radius=1),
            roi.Profile([(0, 0), (1, 1)]),
            DataObject.__new__(DataObject),
            lambda: None,
            {"a": 1},
        ],
    )
    def test_arbitrary_objects_not_written(self, tmp_path, value):
        with h5py.File(tmp_path / "t.h5", "w") as f:
            written = h5mod._encode_value(f, "v", value)
            assert written is False
            assert "v" not in f

    def test_key_with_slash_is_sanitized_and_reversible(self, tmp_path):
        img = make_leem_img()
        setattr(img, "a/b", 42.0)
        target = tmp_path / "x.h5"
        with pytest.warns(UserWarning):
            img.save(str(target))
        reloaded = LEEMImg.load(str(target))
        assert reloaded._meta["a/b"] == 42.0


######### Section C: single-object round trip ###################################################


class TestObjectRoundtrip:
    @pytest.mark.parametrize(
        "dtype", [np.uint8, np.uint16, np.int32, np.float32, np.float64]
    )
    def test_image_pixel_dtype_preserved(self, tmp_path, dtype):
        arr = np.arange(20, dtype=dtype).reshape(4, 5)
        img = Image(arr)
        target = tmp_path / "x.h5"
        img.save(str(target))
        reloaded = Image.load(str(target))
        assert reloaded.image.dtype == dtype
        np.testing.assert_array_equal(reloaded.image, arr)

    def test_leemimg_full_meta_defaults_roundtrip(self, tmp_path):
        img = make_leem_img(
            temperature=123.5,
            pressure=1e-9,
            energy=45.0,
            fov="10 µm",
            timestamp=1700000000.0,
            dark_counts=7,
            photon_energy=90.0,
            energy_offset=1.5,
        )
        target = tmp_path / "x.h5"
        img.save(str(target))
        reloaded = LEEMImg.load(str(target))
        for key in LEEMImg._meta_defaults:
            if key in ("mcp", "warp_matrix"):
                continue
            assert key in reloaded._meta, key
            orig, new = img._meta[key], reloaded._meta[key]
            if isinstance(orig, float) and np.isnan(orig):
                assert np.isnan(new)
            else:
                assert new == orig, key
            assert type(new) is type(orig), key

    def test_warp_matrix_roundtrip(self, tmp_path):
        img = make_leem_img()
        img.warp_matrix = np.array([[1.0, 0.1, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
        target = tmp_path / "x.h5"
        img.save(str(target))
        reloaded = LEEMImg.load(str(target))
        np.testing.assert_array_equal(reloaded.warp_matrix, img.warp_matrix)
        assert reloaded.warp_matrix.dtype == img.warp_matrix.dtype

    def test_none_valued_fields_survive_as_none(self, tmp_path):
        img = make_leem_img()
        assert img.mcp is None
        target = tmp_path / "x.h5"
        img.save(str(target))
        reloaded = LEEMImg.load(str(target))
        assert reloaded.mcp is None

    def test_line_default_label_none_roundtrip(self, tmp_path, line):
        assert line.label is None
        target = tmp_path / "x.h5"
        line.save(str(target))
        reloaded = Line.load(str(target))
        assert reloaded.label is None

    def test_line_roundtrip_dataframe_equal(self, tmp_path, line):
        target = tmp_path / "x.h5"
        line.save(str(target))
        reloaded = Line.load(str(target))
        pd.testing.assert_frame_equal(line.dataframe, reloaded.dataframe)

    def test_nan_inf_meta_roundtrip(self, tmp_path):
        img = make_leem_img(energy=float("nan"), temperature=float("inf"), custom=float("-inf"))
        target = tmp_path / "x.h5"
        img.save(str(target))
        reloaded = LEEMImg.load(str(target))
        assert np.isnan(reloaded.energy)
        assert np.isinf(reloaded.temperature) and reloaded.temperature > 0
        assert np.isinf(reloaded.custom) and reloaded.custom < 0

    def test_bool_and_int_stay_bool_and_int(self, tmp_path):
        img = make_leem_img(mirror_state=True, averaging=4)
        target = tmp_path / "x.h5"
        img.save(str(target))
        reloaded = LEEMImg.load(str(target))
        assert type(reloaded.mirror_state) is bool
        assert reloaded.mirror_state is True
        assert type(reloaded.averaging) is int
        assert reloaded.averaging == 4

    @pytest.mark.parametrize(
        "unit", ["°C", "µm", "µm/px", "µA", "Å"]
    )
    def test_nonascii_units_roundtrip(self, tmp_path, unit):
        img = make_leem_img()
        img.custom = 1.23
        img._units["custom"] = unit
        target = tmp_path / "x.h5"
        img.save(str(target))
        reloaded = LEEMImg.load(str(target))
        assert reloaded._units["custom"] == unit
        assert type(reloaded._units["custom"]) is str

    def test_orphan_units_return_via_constructor(self, tmp_path, leem_img):
        # binding_energy/rel_time/objective have units but (usually) no value -- these must
        # not need any storage in the file at all; they come back from _unit_defaults.
        target = tmp_path / "x.h5"
        leem_img.save(str(target))
        reloaded = LEEMImg.load(str(target))
        for key in ("binding_energy", "rel_time", "objective"):
            assert reloaded._units.get(key) == LEEMImg._unit_defaults[key]

    def test_source_metadata_roundtrip(self, tmp_path):
        src_path = tmp_path / "original.tiff"
        import tifffile

        arr = np.ones((4, 4), dtype=np.float32)
        tifffile.imwrite(str(src_path), arr)
        img = Image(str(src_path))
        assert img.source == str(src_path)

        target = tmp_path / "x.h5"
        img.save(str(target))
        reloaded = Image.load(str(target))
        assert reloaded._meta.get("source") == str(src_path)
        # the reloaded object is self-contained; its own `.source` property is None, not the
        # original acquisition path (that's only reachable via `.meta["source"]`).
        assert reloaded.source is None

    def test_save_load_save_idempotent_no_key_accretion(self, tmp_path, leem_img):
        target1 = tmp_path / "x1.h5"
        target2 = tmp_path / "x2.h5"
        leem_img.save(str(target1))
        reloaded = LEEMImg.load(str(target1))
        reloaded.save(str(target2))
        reloaded_again = LEEMImg.load(str(target2))
        assert set(reloaded.meta.keys()) == set(reloaded_again.meta.keys())

    def test_dataobject_eq_smoke(self, tmp_path, leem_img):
        # DataObject.__eq__ is a weak smoke check only (5 significant digits, numeric-type
        # insensitive) -- exact dtype/value assertions above are the real test.
        target = tmp_path / "x.h5"
        leem_img.save(str(target))
        reloaded = LEEMImg.load(str(target))
        assert reloaded == leem_img

    def test_meta_key_containing_unit_substring_is_not_this_features_problem(self, tmp_path):
        # DataObject.__init__ routes any key *containing* "_unit" into _units, but only when
        # parse()'s flat dict is routed through __init__ (see base.py:129) -- not for a plain
        # setattr() after construction, which just lands in _meta normally. Since this writer
        # walks obj._meta directly and never re-invokes parse()/`__init__` on read, that
        # pre-existing parse-time hazard (documented in the approved plan) simply doesn't
        # arise here: a key that happens to contain "_unit" round-trips like any other.
        img = make_leem_img()
        img.beam_unit_cell = 3.14
        target = tmp_path / "x.h5"
        img.save(str(target))
        reloaded = LEEMImg.load(str(target))
        assert reloaded._meta["beam_unit_cell"] == 3.14


######### Section D: bare-h5py structural (the archival requirement) ###########################


class TestStructural:
    def test_nexus_attrs_present(self, tmp_path, leem_img):
        target = tmp_path / "x.h5"
        leem_img.save(str(target))
        with h5py.File(target, "r") as f:
            assert f.attrs["NX_class"] == "NXroot"
            assert f["entry"].attrs["NX_class"] == "NXentry"
            data_grp = f["entry/data"]
            assert data_grp.attrs["NX_class"] == "NXdata"
            signal_name = data_grp.attrs["signal"]
            assert isinstance(signal_name, str)
            signal = data_grp[signal_name]
            axes = list(data_grp.attrs["axes"])
            assert len(axes) == signal.ndim

    def test_only_three_uspy_attributes_in_whole_file(self, tmp_path, leem_stack):
        target = tmp_path / "x.h5"
        leem_stack.save(str(target))
        found = set()

        def visit(name, obj):
            for k in obj.attrs:
                if k.startswith("uspy"):
                    found.add(f"{name}@{k}" if name else f"/@{k}")

        with h5py.File(target, "r") as f:
            for k in f.attrs:
                if k.startswith("uspy"):
                    found.add(f"/@{k}")
            f.visititems(visit)

        allowed = {"/@uspy_version", "/@uspy_format_version", "entry@uspy_class"}
        assert found == allowed, found

    def test_archive_has_only_data_and_metadata_groups(self, tmp_path, leem_img):
        target = tmp_path / "x.h5"
        leem_img.save(str(target))
        with h5py.File(target, "r") as f:
            entry_children = set(f["entry"].keys())
        assert entry_children <= {"data", "metadata", "instrument", "program_name", "title", "start_time"}
        assert "data" in entry_children
        assert "metadata" in entry_children

    def test_readable_with_bare_h5py_subprocess(self, tmp_path, leem_stack):
        target = tmp_path / "x.h5"
        leem_stack.save(str(target))
        script = f"""
import h5py, numpy as np
with h5py.File({str(target)!r}, "r") as f:
    img = f["entry/data/image"][()]
    energy = f["entry/metadata/energy"][()]
    assert img.shape[0] == {len(leem_stack)}
    assert np.allclose(sorted(energy.tolist()), sorted({[float(x) for x in leem_stack.energy]!r}))
print("OK")
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OK" in result.stdout

    def test_missing_uspy_class_falls_back_structurally(self, tmp_path, leem_img):
        target = tmp_path / "x.h5"
        leem_img.save(str(target))
        with h5py.File(target, "r+") as f:
            del f["entry"].attrs["uspy_class"]
        with pytest.warns(UserWarning):
            reloaded = Loadable.load(str(target))
        assert isinstance(reloaded, Image)
        np.testing.assert_array_equal(reloaded.image, leem_img.image)

    def test_refuses_non_uspy_class(self, tmp_path, leem_img):
        target = tmp_path / "x.h5"
        leem_img.save(str(target))
        with h5py.File(target, "r+") as f:
            f["entry"].attrs["uspy_class"] = "os.system"
        with pytest.raises(ValueError):
            Loadable.load(str(target))

    def test_unknown_uspy_class_falls_back_with_warning(self, tmp_path, leem_img):
        target = tmp_path / "x.h5"
        leem_img.save(str(target))
        with h5py.File(target, "r+") as f:
            f["entry"].attrs["uspy_class"] = "uspy.leem.cluster.NotARealClass"
        with pytest.warns(UserWarning):
            reloaded = Loadable.load(str(target))
        assert isinstance(reloaded, Image)


######### Section E: stack round trip and heterogeneity #########################################


class TestStackRoundtrip:
    def test_leemstack_roundtrip_basic(self, tmp_path, leem_stack):
        target = tmp_path / "x.h5"
        leem_stack.save(str(target))
        reloaded = LEEMStack.load(str(target))
        assert len(reloaded) == len(leem_stack)
        for i in range(len(leem_stack)):
            np.testing.assert_array_equal(reloaded[i].image, leem_stack[i].image)
        np.testing.assert_allclose(reloaded.energy, leem_stack.energy)
        np.testing.assert_allclose(reloaded.temperature, leem_stack.temperature)
        assert reloaded[0]._units["energy"] == "eV"

    def test_imagestack_roundtrip(self, tmp_path):
        imgs = [Image(np.full((3, 3), float(i))) for i in range(3)]
        stack = ImageStack(imgs)
        target = tmp_path / "x.h5"
        stack.save(str(target))
        reloaded = ImageStack.load(str(target))
        assert len(reloaded) == 3
        for i in range(3):
            np.testing.assert_array_equal(reloaded[i].image, stack[i].image)

    def test_single_frame_stack_roundtrips(self, tmp_path):
        stack = make_leem_stack(n=1)
        target = tmp_path / "x.h5"
        stack.save(str(target))
        reloaded = LEEMStack.load(str(target))
        assert len(reloaded) == 1

    def test_empty_stack_raises_not_corrupt_file(self, tmp_path):
        # A genuinely empty ImageStack can't be *constructed* via the public API (see
        # tests/test_base.py's empty-stack tests) -- but del stack[:] doesn't go through
        # __init__, so it's still the real, reachable route to a len()==0 stack, and this
        # module's own empty-stack guard (uspy/hdf5.py's write(), before the file is even
        # opened) must fire for it.
        empty = ImageStack([Image(np.full((4, 4), 1, dtype=np.uint16))])
        del empty[:]
        assert len(empty) == 0
        target = tmp_path / "x.h5"
        with pytest.raises(ValueError):
            empty.save(str(target))
        assert not target.exists()

    def test_read_stack_with_zero_frames_raises(self, tmp_path, leem_stack):
        # uspy never *writes* a 0-frame stack (see test_empty_stack_raises_not_corrupt_file
        # above), so a 0-frame signal in a file being read is necessarily foreign/truncated,
        # not a legitimate empty stack -- reading it must raise a clear ValueError rather than
        # silently yielding an (unconstructible-by-any-other-route) empty stack, or crashing
        # with a bare IndexError inside the stack constructor.
        target = tmp_path / "truncated.h5"
        leem_stack.save(str(target))
        with h5py.File(str(target), "r+") as f:
            data_grp = f["entry/data"]
            signal_name = data_grp.attrs.get("signal", "image")
            shape = data_grp[signal_name].shape
            del data_grp[signal_name]
            data_grp.create_dataset(signal_name, shape=(0,) + shape[1:], dtype=np.uint16)
        with pytest.raises(ValueError) as excinfo:
            LEEMStack.load(str(target))
        assert not isinstance(excinfo.value, IndexError)

    def test_missing_key_in_one_frame_warns_and_fills_nan(self, tmp_path, leem_stack):
        del leem_stack[2]._meta["energy"]
        target = tmp_path / "x.h5"
        with pytest.warns(h5mod.MissingFieldWarning, match="energy"):
            leem_stack.save(str(target))
        reloaded = LEEMStack.load(str(target))
        assert np.isnan(reloaded[2].energy)

    def test_unit_mismatch_across_frames_warns_and_picks_common(self, tmp_path, leem_stack):
        leem_stack[0]._units["energy"] = "keV"
        target = tmp_path / "x.h5"
        with pytest.warns(h5mod.UnitMismatchWarning, match="energy"):
            leem_stack.save(str(target))
        with h5py.File(target, "r") as f:
            unit = f["entry/metadata/energy"].attrs["units"]
        assert unit == "eV"  # the majority unit across the 4 frames

    def test_mixed_int_float_column_coerces_with_warning(self, tmp_path):
        stack = make_leem_stack(n=2)
        stack[0]._meta["custom"] = 1
        stack[1]._meta["custom"] = 2.5
        stack[0]._units.setdefault("custom", "")
        stack[1]._units.setdefault("custom", "")
        target = tmp_path / "x.h5"
        with pytest.warns(UserWarning):
            stack.save(str(target))
        reloaded = LEEMStack.load(str(target))
        assert reloaded[0]._meta["custom"] == 1.0
        assert reloaded[1]._meta["custom"] == 2.5

    def test_mixed_str_number_column_coerces_to_string(self, tmp_path):
        stack = make_leem_stack(n=2)
        stack[0]._meta["custom"] = "abc"
        stack[1]._meta["custom"] = 5
        stack[0]._units.setdefault("custom", "")
        stack[1]._units.setdefault("custom", "")
        target = tmp_path / "x.h5"
        with pytest.warns(UserWarning):
            stack.save(str(target))
        reloaded = LEEMStack.load(str(target))
        assert reloaded[0]._meta["custom"] == "abc"
        assert reloaded[1]._meta["custom"] == "5"

    def test_virtual_stack_export(self, tmp_path, tiff_frame_stack):
        stack = ImageStack(tiff_frame_stack["paths"], virtual=True)
        assert stack.virtual
        target = tmp_path / "x.h5"
        stack.save(str(target))
        reloaded = ImageStack.load(str(target))
        assert len(reloaded) == len(tiff_frame_stack["paths"])
        for i, arr in enumerate(tiff_frame_stack["arrays"]):
            np.testing.assert_array_equal(reloaded[i].image, arr)


######### Section F: source safety ###############################################################


class TestSourceSafety:
    def test_save_does_not_modify_source_files(self, tmp_path, tiff_frame_stack):
        def file_hash(path):
            with open(path, "rb") as handle:
                return hashlib.md5(handle.read()).hexdigest()

        before = {p: file_hash(p) for p in tiff_frame_stack["paths"]}
        stack = ImageStack(tiff_frame_stack["paths"])
        stack.save(str(tmp_path / "x.h5"))
        after = {p: file_hash(p) for p in tiff_frame_stack["paths"]}
        assert before == after

    def test_save_does_not_mutate_object(self, tmp_path, leem_img):
        import copy

        before_meta = copy.deepcopy(leem_img._meta)
        before_units = copy.deepcopy(leem_img._units)
        before_data = {k: v.copy() for k, v in leem_img._data.items()}
        leem_img.save(str(tmp_path / "x.h5"))
        assert set(leem_img._meta.keys()) == set(before_meta.keys())
        for k in before_meta:
            v1, v2 = before_meta[k], leem_img._meta[k]
            if isinstance(v1, float) and np.isnan(v1):
                assert np.isnan(v2)
            elif isinstance(v1, np.ndarray):
                np.testing.assert_array_equal(v1, v2)
            else:
                assert v1 == v2
        assert leem_img._units == before_units
        for k in before_data:
            np.testing.assert_array_equal(before_data[k], leem_img._data[k])


######### Section G: derived-as-parent, and omitted (unrepresentable) fields ####################


class TestDerivedAsParent:
    def _stack_and_roi(self, n=3):
        stack = make_leem_stack(n=n)
        r = roi.ROI.circle(2, 2, radius=2)
        return stack, r

    def test_intensityline_exports_as_line(self, tmp_path):
        stack, r = self._stack_and_roi()
        line = IntensityLine(stack, r, "energy")
        target = tmp_path / "x.h5"
        with pytest.warns(h5mod.DroppedFieldWarning):
            line.save(str(target))
        with h5py.File(target, "r") as f:
            assert f["entry"].attrs["uspy_class"] == "uspy.dataobject.line.Line"
        reloaded = Loadable.load(str(target))
        assert type(reloaded) is Line
        np.testing.assert_allclose(reloaded.x, line.x)
        np.testing.assert_allclose(reloaded.y, line.y)
        assert reloaded.xdim == line.xdim
        assert reloaded.ydim == line.ydim

    def test_stitchedline_and_ivcurve_export_as_line(self, tmp_path):
        from uspy.leem.base import IVCurve

        stack, r = self._stack_and_roi()
        curve = IVCurve(stack, r)
        target = tmp_path / "x.h5"
        with pytest.warns(h5mod.DroppedFieldWarning):
            curve.save(str(target))
        reloaded = Loadable.load(str(target))
        assert type(reloaded) is Line
        np.testing.assert_allclose(reloaded.x, curve.x)
        np.testing.assert_allclose(reloaded.y, curve.y)

    def test_profileline_exports_as_line_with_no_embedded_image(self, tmp_path):
        stack = make_leem_stack(n=2, shape=(6, 5))
        img = stack[0]
        profile = roi.Profile([(0, 0), (4, 5)])
        pline = ProfileLine(img, profile)
        target = tmp_path / "x.h5"
        with pytest.warns(h5mod.DroppedFieldWarning):
            pline.save(str(target))
        with h5py.File(target, "r") as f:
            # exactly one @signal array in the whole file -- no second (embedded) image
            signal_datasets = []

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and "image" in name:
                    signal_datasets.append(name)

            f.visititems(visit)
            assert len(signal_datasets) <= 1
        assert target.stat().st_size < 50_000  # small: no upstream image embedded
        reloaded = Loadable.load(str(target))
        assert type(reloaded) is Line
        np.testing.assert_allclose(reloaded.x, pline.x)
        np.testing.assert_allclose(reloaded.y, pline.y)

    def test_waterfall_exports_as_image_with_axes(self, tmp_path):
        stack = make_leem_stack(n=3, shape=(4, 4))
        profile = roi.Profile([(0, 0), (3, 3)])
        wf = Waterfall(stack, profile, "energy")
        target = tmp_path / "x.h5"
        with pytest.warns(h5mod.DroppedFieldWarning):
            wf.save(str(target))
        with h5py.File(target, "r") as f:
            assert f["entry"].attrs["uspy_class"] == "uspy.dataobject.image.Image"
        reloaded = Loadable.load(str(target))
        assert type(reloaded) is Image
        np.testing.assert_array_equal(reloaded.image, wf.image)
        np.testing.assert_allclose(reloaded._meta["y"], wf._meta["y"])

    def test_object_valued_fields_leave_no_trace_in_file(self, tmp_path):
        stack, r = self._stack_and_roi()
        line = IntensityLine(stack, r, "energy")
        target = tmp_path / "x.h5"
        with pytest.warns(h5mod.DroppedFieldWarning):
            line.save(str(target))
        reloaded = Loadable.load(str(target))
        assert "stack" not in reloaded._meta
        assert "roi" not in reloaded._meta
        with h5py.File(target, "r") as f:
            entry_children = set(f["entry"].keys())
        assert entry_children <= {"data", "metadata", "instrument", "program_name", "title", "start_time"}


######### Section H: LEEM specifics ##############################################################


class TestLEEMSpecifics:
    def test_shared_time_origin_and_rel_time(self, tmp_path, leem_stack):
        target = tmp_path / "x.h5"
        leem_stack.save(str(target))
        reloaded = LEEMStack.load(str(target))
        assert reloaded[0]._time_origin is reloaded[1]._time_origin
        assert reloaded[0].rel_time == pytest.approx(0.0)
        assert reloaded[1].rel_time == pytest.approx(
            leem_stack[1].timestamp - leem_stack[0].timestamp
        )

    def test_hand_set_high_temperature_survives_reload(self, tmp_path):
        img = make_leem_img(temperature=5000.0)
        target = tmp_path / "x.h5"
        img.save(str(target))
        reloaded = LEEMImg.load(str(target))
        assert reloaded.temperature == 5000.0

    def test_post_process_stack_never_runs_on_reload(self, tmp_path, leem_stack, monkeypatch):
        calls = []

        class MutatingReader(LEEMReader):
            name = "test_mutating_reader"
            extensions = (".dat",)

            def post_process_stack(self, stack):
                calls.append(stack)
                stack.energy = np.zeros(len(stack))

        (tmp_path / "uspy.toml").write_text('[uspy]\nreader = "test_mutating_reader"\n')
        target = tmp_path / "x.h5"
        leem_stack.save(str(target))
        reloaded = LEEMStack.load(str(target))
        assert not calls
        np.testing.assert_allclose(reloaded.energy, leem_stack.energy)


######### Section I: virtual stacks ###############################################################


class TestVirtual:
    def test_virtual_imagestack_roundtrip_keeps_metadata(self, tmp_path, tiff_frame_stack):
        virtual = ImageStack(tiff_frame_stack["paths"], virtual=True)
        target = tmp_path / "x.h5"
        virtual.save(str(target))
        reloaded = ImageStack.load(str(target))
        assert type(reloaded) is ImageStack
        assert len(reloaded) == len(tiff_frame_stack["paths"])
        for i, arr in enumerate(tiff_frame_stack["arrays"]):
            np.testing.assert_array_equal(reloaded[i].image, arr)

    def test_virtual_leemstack_roundtrip_matches_eager(self, tmp_path):
        eager = make_leem_stack(n=3)
        target = tmp_path / "x.h5"
        eager.save(str(target))
        reloaded = LEEMStack.load(str(target))
        np.testing.assert_allclose(reloaded.energy, eager.energy)

    def test_virtual_save_never_materializes_pixels(self, tmp_path, tiff_frame_stack, monkeypatch):
        virtual = ImageStack(tiff_frame_stack["paths"], virtual=True)

        def boom(self):
            raise AssertionError("materialize() should never be called by save()")

        monkeypatch.setattr(Image, "materialize", boom)
        target = tmp_path / "x.h5"
        virtual.save(str(target))  # must not raise
        assert all(element.is_virtual for element in virtual.elements)


######### Section J: real-data regression (submodule-dependent, slow) ###########################


@pytest.mark.slow
@pytest.mark.parametrize("stack_fname", STACK_FNAMES)
def test_real_leem_stack_roundtrip(tmp_path, stack_fname):
    if not os.path.exists(stack_fname) and not os.path.exists(
        os.path.dirname(stack_fname) or "."
    ):
        pytest.skip(f"no fixture at {stack_fname}")
    stack = LEEMStack(stack_fname)
    target = tmp_path / "x.h5"
    stack.save(str(target))
    reloaded = LEEMStack.load(str(target))
    assert len(reloaded) == len(stack)
    np.testing.assert_array_equal(reloaded[0].image, stack[0].image)
    np.testing.assert_array_equal(reloaded[-1].image, stack[-1].image)
    np.testing.assert_allclose(reloaded.energy, stack.energy, equal_nan=True)
