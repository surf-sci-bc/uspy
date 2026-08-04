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

import numpy as np

import uspy.dataobject as do
import uspy.plotting as base_plotting
from uspy.leem import base
from uspy.leem import plotting as leem_plotting
from uspy.roi import ROI

from .conftest import ELETTRA_DAT_FNAME, RUO2_STACK_DIR

ELETTRA_DAT = ELETTRA_DAT_FNAME


class _FakeWriter:
    """Records frames instead of invoking ffmpeg, so make_video()'s masking/contrast logic
    can be exercised for real without needing ffmpeg installed."""

    def __init__(self, *args, **kwargs):
        self.frames = []

    def writeFrame(self, data):
        self.frames.append(data.copy())

    def close(self):
        pass


def test_make_video_mask_is_actually_applied_for_a_path_source(tmp_path, monkeypatch, capsys):
    """Regression test: make_video() applies the ROI mask at each pixel read
    (`np.ma.array(img.image, mask=inv_mask)`) rather than writing a masked array back into the
    stack -- a write-back would (a) on a virtual stack, permanently pin every frame in memory
    (see CLAUDE.md's "Memory: large stacks" section) and (b) on the old virtual-stack backend
    predating that fix, silently do nothing at all, since `stack[i]` used to be a throwaway
    re-parsed object. Confirmed fixed here: contrast is computed only over the masked (in-ROI)
    region, matching a mask genuinely applied."""
    fake_writer = _FakeWriter()
    monkeypatch.setattr(base_plotting.skvideo.io, "FFmpegWriter", lambda *a, **kw: fake_writer)
    monkeypatch.setattr(base_plotting, "Video", lambda ofile: ofile)

    stack = base.LEEMStack(RUO2_STACK_DIR, virtual=True)
    height, width = stack[0].height, stack[0].width
    mask = ROI.circle(x0=width // 4, y0=height // 4, radius=min(height, width) // 8)
    mask_array = mask.pad_to(height, width)

    # Ground truth: the masked min/max across the (still eager, untouched) source stack.
    eager = base.LEEMStack(RUO2_STACK_DIR)
    expected_c0 = min(np.ma.min(np.ma.array(img.image, mask=~mask_array)) for img in eager)
    expected_c1 = max(np.ma.max(np.ma.array(img.image, mask=~mask_array)) for img in eager)

    base_plotting.make_video(
        stack, str(tmp_path / "out.mp4"), mask=mask, contrast="maximum", fields=None
    )

    printed = capsys.readouterr().out
    assert f"Set contrast to [{expected_c0}, {expected_c1}]" in printed
    assert len(fake_writer.frames) == len(stack)


def test_make_video_leaves_a_virtual_stack_virtual(tmp_path, monkeypatch):
    """Regression test for the reported bug: `make_video()` on a virtual stack used to
    materialize every frame (`img.image = np.ma.array(...)` written back into each deferred
    element, see CLAUDE.md's "Memory: large stacks" section) even though `stack.virtual`
    stayed True. Masking is now applied at each read instead of written back, so the stack
    (and every element) must still be virtual -- and holding no resident pixel data -- after
    the call, whether or not a mask was requested."""
    monkeypatch.setattr(base_plotting.skvideo.io, "FFmpegWriter", lambda *a, **kw: _FakeWriter())
    monkeypatch.setattr(base_plotting, "Video", lambda ofile: ofile)

    height, width = 0, 0
    for use_mask in (False, True):
        stack = base.LEEMStack(RUO2_STACK_DIR, virtual=True)
        height, width = stack[0].height, stack[0].width
        mask = (
            ROI.circle(x0=width // 4, y0=height // 4, radius=min(height, width) // 8)
            if use_mask
            else None
        )
        base_plotting.make_video(
            stack, str(tmp_path / "out.mp4"), mask=mask, fields=None
        )
        assert stack.virtual is True
        assert all(element.is_virtual for element in stack.elements)


def test_make_video_mask_true_uses_default_mask(tmp_path, monkeypatch, capsys):
    # regression: mask=True used to crash (`True.pad_to(...)` -- bool has no such method).
    monkeypatch.setattr(base_plotting.skvideo.io, "FFmpegWriter", lambda *a, **kw: _FakeWriter())
    monkeypatch.setattr(base_plotting, "Video", lambda ofile: ofile)

    stack = base.LEEMStack(RUO2_STACK_DIR, virtual=True)
    height, width = stack[0].height, stack[0].width
    default_mask_array = stack[0].default_mask.pad_to(height, width)

    eager = base.LEEMStack(RUO2_STACK_DIR)
    expected_c0 = min(
        np.ma.min(np.ma.array(img.image, mask=~default_mask_array)) for img in eager
    )
    expected_c1 = max(
        np.ma.max(np.ma.array(img.image, mask=~default_mask_array)) for img in eager
    )

    base_plotting.make_video(
        stack, str(tmp_path / "out.mp4"), mask=True, contrast="maximum", fields=None
    )
    printed = capsys.readouterr().out
    assert f"Set contrast to [{expected_c0}, {expected_c1}]" in printed


def test_make_video_mask_ndarray_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(base_plotting.skvideo.io, "FFmpegWriter", lambda *a, **kw: _FakeWriter())
    monkeypatch.setattr(base_plotting, "Video", lambda ofile: ofile)

    stack = base.LEEMStack(RUO2_STACK_DIR, virtual=True)
    height, width = stack[0].height, stack[0].width
    mask_array = ROI.circle(x0=width // 2, y0=height // 2, radius=10).pad_to(height, width)

    base_plotting.make_video(stack, str(tmp_path / "out.mp4"), mask=mask_array, fields=None)


def test_make_video_mask_false_and_none_equivalent_to_no_mask(tmp_path, monkeypatch):
    monkeypatch.setattr(base_plotting.skvideo.io, "FFmpegWriter", lambda *a, **kw: _FakeWriter())
    monkeypatch.setattr(base_plotting, "Video", lambda ofile: ofile)

    for mask in (False, None):
        stack = base.LEEMStack(RUO2_STACK_DIR, virtual=True)
        base_plotting.make_video(stack, str(tmp_path / "out.mp4"), mask=mask, fields=None)


### plot_img mask handling (uspy/plotting.py)


def test_plot_img_mask_true_uses_default_mask():
    img = base.LEEMImg(ELETTRA_DAT)
    ax = base_plotting.plot_img(img, mask=True)
    plt.close(ax.figure)


def test_plot_img_mask_true_on_plain_image_is_noop():
    # regression: do.Image has no default_mask attribute of its own -- mask=True used to
    # raise AttributeError instead of being a no-op.
    img = do.Image(np.arange(20 * 30).reshape(20, 30).astype(float))
    ax = base_plotting.plot_img(img, mask=True)
    plt.close(ax.figure)


def test_plot_img_ndarray_mask_does_not_raise():
    # regression: `if mask or invert:` raised ValueError for a >1-element ndarray mask
    # (ambiguous truth value); only bool/ROI/None were ever actually handled.
    img = base.LEEMImg(ELETTRA_DAT)
    mask_array = np.ones(img.shape, dtype=bool)
    ax = base_plotting.plot_img(img, mask=mask_array)
    plt.close(ax.figure)


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
        captured["virtual"] = stack.virtual

    monkeypatch.setattr(base_plotting, "make_video", fake_make_video)
    leem_plotting.make_video(str(tmp_path), ofile="unused.mp4")
    assert captured["fields"] == ("fov",)
    # regression guard: a path source used to be forced eager here (stackify()'s virtual=False
    # default), which meant a virtual stack could never reach uspy.plotting.make_video through
    # the LEEM-facing wrapper at all.
    assert captured["virtual"] is True


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
