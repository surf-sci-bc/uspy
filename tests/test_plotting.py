"""Tests that plot_img/make_video/plot_meta/print_meta pick up their default fields
from LEEMImg.default_fields/meta_fields/print_fields (see test_readers.py for how those
get resolved from a uspy.toml [uspy.plotting] table), and that explicit `fields=`
arguments still override everything."""
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name

import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import uspy.plotting as base_plotting
from uspy.leem import base
from uspy.leem import plotting as leem_plotting

from .conftest import TESTDATA_DIR

ELETTRA_DAT = TESTDATA_DIR + "elettra.dat"


### plot_img / make_video (uspy/leem/plotting.py wrappers)


def test_plot_img_explicit_fields_overrides_default(img_fname, monkeypatch):
    captured = {}

    def fake_plot_img(img, *args, fields=None, **kwargs):  # pylint: disable=unused-argument
        captured["fields"] = fields

    monkeypatch.setattr(base_plotting, "plot_img", fake_plot_img)
    leem_plotting.plot_img(img_fname, fields=("fov",))
    assert captured["fields"] == ("fov",)


def test_plot_img_omitted_fields_passes_none_through(img_fname, monkeypatch):
    # plot_img's own fallback to img.default_fields lives in uspy.plotting.plot_img,
    # so the wrapper should just pass None through unchanged when fields isn't given.
    captured = {}

    def fake_plot_img(img, *args, fields=None, **kwargs):  # pylint: disable=unused-argument
        captured["fields"] = fields

    monkeypatch.setattr(base_plotting, "plot_img", fake_plot_img)
    leem_plotting.plot_img(img_fname)
    assert captured["fields"] is None


def test_make_video_uses_configured_default_fields_when_omitted(tmp_path, monkeypatch):
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_000.dat")
    (tmp_path / "uspy.toml").write_text('[uspy.plotting]\ndefault_fields = ["fov"]\n')

    captured = {}

    def fake_make_video(stack, *args, fields=None, **kwargs):  # pylint: disable=unused-argument
        captured["fields"] = fields

    monkeypatch.setattr(base_plotting, "make_video", fake_make_video)
    leem_plotting.make_video(str(tmp_path), ofile="unused.mp4")
    assert captured["fields"] == ("fov",)


def test_make_video_explicit_none_still_disables_overlay(tmp_path, monkeypatch):
    # regression guard for the _UNSET sentinel: an explicit fields=None must keep
    # meaning "no overlay" (see Analysis/Untitled.ipynb), not get resolved to defaults.
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_000.dat")
    (tmp_path / "uspy.toml").write_text('[uspy.plotting]\ndefault_fields = ["fov"]\n')

    captured = {}

    def fake_make_video(stack, *args, fields=None, **kwargs):  # pylint: disable=unused-argument
        captured["fields"] = fields

    monkeypatch.setattr(base_plotting, "make_video", fake_make_video)
    leem_plotting.make_video(str(tmp_path), ofile="unused.mp4", fields=None)
    assert captured["fields"] is None


def test_make_video_explicit_fields_overrides_default(tmp_path, monkeypatch):
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_000.dat")
    (tmp_path / "uspy.toml").write_text('[uspy.plotting]\ndefault_fields = ["fov"]\n')

    captured = {}

    def fake_make_video(stack, *args, fields=None, **kwargs):  # pylint: disable=unused-argument
        captured["fields"] = fields

    monkeypatch.setattr(base_plotting, "make_video", fake_make_video)
    leem_plotting.make_video(str(tmp_path), ofile="unused.mp4", fields=("energy",))
    assert captured["fields"] == ("energy",)


### plot_meta / print_meta (uspy/plotting.py, resolve via stack[0].meta_fields/print_fields)


def test_plot_meta_uses_configured_meta_fields(tmp_path):
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_000.dat")
    (tmp_path / "uspy.toml").write_text('[uspy.plotting]\nmeta_fields = ["energy"]\n')

    stack = base.LEEMStack(str(tmp_path))
    axes = base_plotting.plot_meta(stack)
    try:
        assert axes[0].get_ylabel().startswith("energy")
    finally:
        plt.close("all")


def test_plot_meta_explicit_fields_overrides_config(tmp_path):
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_000.dat")
    (tmp_path / "uspy.toml").write_text('[uspy.plotting]\nmeta_fields = ["energy"]\n')

    stack = base.LEEMStack(str(tmp_path))
    axes = base_plotting.plot_meta(stack, fields="fov")
    try:
        assert axes[0].get_ylabel().startswith("fov")
    finally:
        plt.close("all")


def test_print_meta_uses_configured_print_fields(tmp_path):
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_000.dat")
    (tmp_path / "uspy.toml").write_text('[uspy.plotting]\nprint_fields = ["energy"]\n')

    stack = base.LEEMStack(str(tmp_path))
    # only checking it resolves the configured field and doesn't raise
    base_plotting.print_meta(stack)


def test_print_meta_explicit_fields_overrides_config(tmp_path):
    shutil.copy(ELETTRA_DAT, tmp_path / "elettra_000.dat")
    (tmp_path / "uspy.toml").write_text('[uspy.plotting]\nprint_fields = ["energy"]\n')

    stack = base.LEEMStack(str(tmp_path))
    base_plotting.print_meta(stack, fields=("fov",))
