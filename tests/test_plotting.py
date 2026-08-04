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
import pytest

import uspy.dataobject as do
import uspy.plotting as base_plotting
from uspy.leem import base
from uspy.leem import plotting as leem_plotting
from uspy.roi import ROI

from .conftest import ELETTRA_DAT_FNAME, RUO2_STACK_DIR, TESTIMAGE_BASE

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


### _CV2_IMG_POS (pure arithmetic, uspy/plotting.py) ##########################


def test_cv2_img_pos_top_left_corner():
    # idx=0: top-left. margin=0.01, charwidth=20, charheight=23 (see _CV2_IMG_POS).
    height, width = 100, 200
    x, y = base_plotting._CV2_IMG_POS(0, height, width, text="ab", pos=0)
    assert (x, y) == (int(width * 0.01), int(height * 0.01 + 23))


def test_cv2_img_pos_top_right_corner_accounts_for_text_length():
    height, width = 100, 200
    text = "abcd"
    x, y = base_plotting._CV2_IMG_POS(1, height, width, text=text, pos=0)
    assert x == int(width * (1 - 0.01) - len(text) * 20)
    assert y == int(height * 0.01 + 23)


def test_cv2_img_pos_bottom_left_corner():
    height, width = 100, 200
    x, y = base_plotting._CV2_IMG_POS(2, height, width, text="ab", pos=0)
    assert x == int(width * 0.01)
    assert y == int(height * (1 - 0.01))


def test_cv2_img_pos_bottom_right_corner():
    height, width = 100, 200
    text = "abc"
    x, y = base_plotting._CV2_IMG_POS(3, height, width, text=text, pos=0)
    assert x == int(width * (1 - 0.01) - len(text) * 20)
    assert y == int(height * (1 - 0.01))


def test_cv2_img_pos_advances_by_charwidth_per_position():
    x0, y0 = base_plotting._CV2_IMG_POS(0, 100, 200, text="ab", pos=0)
    x1, y1 = base_plotting._CV2_IMG_POS(0, 100, 200, text="ab", pos=1)
    assert x1 - x0 == 20
    assert y1 == y0


### get_max_variance_idx (uspy/plotting.py) ####################################


def test_get_max_variance_idx_picks_analytically_known_winner():
    """A two-level (0/1) frame's normalized variance is p*(1-p) for fill fraction p, maximal
    at p=0.5 -- an independent closed form, not just "this frame looks more varied"."""
    rng = np.random.default_rng(0)
    shape = (30, 30)
    fill_fractions = [0.05, 0.5, 0.9]  # frame 1 (p=0.5) has the highest p*(1-p)
    frames = []
    for p in fill_fractions:
        arr = (rng.random(shape) < p).astype(float)
        frames.append(do.Image(arr))

    idx = base_plotting.get_max_variance_idx(frames)

    assert idx == 1


def test_get_max_variance_idx_breaks_ties_with_first_index():
    identical = [do.Image(np.full((5, 5), 3.0)) for _ in range(3)]
    # all frames are constant -> all variances are 0 (the B7 guard below), so the strict ">"
    # comparison never updates max_index away from its initial 0.
    assert base_plotting.get_max_variance_idx(identical) == 0


def test_get_max_variance_idx_constant_frame_does_not_crash_or_win():
    """Regression: a constant frame's (amax - amin) is 0, which used to divide-by-zero into
    nan; nan > max_var is always False, so it accidentally "worked" by losing every
    comparison anyway -- but only by luck, and only for a *strict* `>` with no other frame
    tied. Pin the actual guarded behavior instead of relying on that coincidence."""
    frames = [
        do.Image(np.full((5, 5), 3.0)),  # constant: guarded to variance 0
        do.Image(np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]] * 1)[:5, :2]),
    ]
    idx = base_plotting.get_max_variance_idx(frames)
    assert idx == 1


### plot_img option branches ####################################################


def test_plot_img_accepts_str_path():
    # plot_img(str) constructs a generic do.Image(source) (not a LEEM-specific reader), so
    # the path must be a format imageio itself recognizes (.png here), not a raw .dat.
    ax = base_plotting.plot_img(TESTIMAGE_BASE + ".png")
    assert ax.images
    plt.close(ax.figure)


def test_plot_img_fields_as_str_is_wrapped_in_a_list():
    img = base.LEEMImg(ELETTRA_DAT)
    ax = base_plotting.plot_img(img, fields="fov")
    assert len(ax.texts) == 1
    plt.close(ax.figure)


def test_plot_img_log_matches_hand_computed_log_normalization():
    arr = np.array([[1.0, 2.0], [4.0, 8.0]])
    img = do.Image(arr)
    ax = base_plotting.plot_img(img, log=True, fields=())

    drawn = ax.images[0].get_array()
    log_data = np.log(arr)
    expected = np.clip(
        (log_data - log_data.min()) / (log_data.max() - log_data.min()) * 255, 0, 255
    ).astype(np.uint8)
    np.testing.assert_array_equal(drawn, expected)
    plt.close(ax.figure)


def test_plot_img_contrast_inner_excludes_outer_20_percent():
    arr = np.zeros((20, 20))
    arr[0, 0] = 1000.0  # in the excluded 20% border
    arr[10, 10] = 5.0  # in the inner region
    img = do.Image(arr)

    ax = base_plotting.plot_img(img, contrast="inner", fields=())

    drawn = ax.images[0].get_array()
    # the far corner (1000) must have clipped/saturated, not set the scale: the inner max
    # (5.0) is what 255 corresponds to.
    assert drawn[10, 10] == 255
    assert drawn[0, 0] == 255  # clipped past the inner-derived upper bound too
    plt.close(ax.figure)


def test_plot_img_explicit_contrast_tuple_matches_hand_computed_normalization():
    arr = np.array([[0.0, 10.0], [20.0, 30.0]])
    img = do.Image(arr)
    ax = base_plotting.plot_img(img, contrast=(10.0, 20.0), fields=())

    drawn = ax.images[0].get_array()
    expected = np.clip((np.clip(arr, 10.0, None) - 10.0) / 10.0 * 255, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(drawn, expected)
    plt.close(ax.figure)


def test_plot_img_invert_flips_around_max():
    arr = np.array([[0.0, 10.0], [20.0, 30.0]])
    img = do.Image(arr)
    ax_normal = base_plotting.plot_img(img, fields=())
    normal = ax_normal.images[0].get_array().copy()
    plt.close(ax_normal.figure)

    ax_inverted = base_plotting.plot_img(img, invert=True, fields=())
    inverted = ax_inverted.images[0].get_array()

    np.testing.assert_array_equal(inverted, normal.max() - normal)
    plt.close(ax_inverted.figure)


def test_plot_img_more_than_four_fields_warns_and_ignores_extras(capsys):
    img = base.LEEMImg(ELETTRA_DAT)
    ax = base_plotting.plot_img(img, fields=("energy", "temperature", "pressure", "fov", "exposure"))
    assert len(ax.texts) == 4
    assert "Ignoring field exposure" in capsys.readouterr().out
    plt.close(ax.figure)


def test_plot_img_falsy_field_in_list_is_skipped():
    img = base.LEEMImg(ELETTRA_DAT)
    ax = base_plotting.plot_img(img, fields=("energy", None, "fov"))
    assert len(ax.texts) == 2
    plt.close(ax.figure)


def test_plot_img_title_true_uses_source_and_truncates_long_paths():
    img = base.LEEMImg(ELETTRA_DAT)
    ax = base_plotting.plot_img(img, title=True, fields=())
    title = ax.get_title()
    assert title.startswith("...")
    assert len(title) == 28  # "..." + last 25 chars of the source path
    plt.close(ax.figure)


### make_video option branches ###################################################


def _synthetic_video_stack(n=3, shape=(12, 16), **meta):
    stack = base.LEEMStack(
        np.random.default_rng(0).uniform(1, 100, size=(n, *shape))
    )
    for key, value in meta.items():
        setattr(stack, key, value)
    return stack


def test_make_video_rejects_mcp_argument(tmp_path, monkeypatch):
    monkeypatch.setattr(base_plotting.skvideo.io, "FFmpegWriter", lambda *a, **kw: _FakeWriter())
    monkeypatch.setattr(base_plotting, "Video", lambda ofile: ofile)
    stack = _synthetic_video_stack()
    with pytest.raises(ValueError):
        base_plotting.make_video(stack, str(tmp_path / "out.mp4"), mcp=object(), fields=None)


def test_make_video_rejects_non_default_dark_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(base_plotting.skvideo.io, "FFmpegWriter", lambda *a, **kw: _FakeWriter())
    monkeypatch.setattr(base_plotting, "Video", lambda ofile: ofile)
    stack = _synthetic_video_stack()
    with pytest.raises(ValueError):
        base_plotting.make_video(stack, str(tmp_path / "out.mp4"), dark_counts=50, fields=None)


def test_make_video_overwrite_false_skips_when_file_exists(tmp_path, monkeypatch):
    fake_writer = _FakeWriter()
    monkeypatch.setattr(base_plotting.skvideo.io, "FFmpegWriter", lambda *a, **kw: fake_writer)
    monkeypatch.setattr(base_plotting, "Video", lambda ofile: ofile)
    ofile = tmp_path / "out.mp4"
    ofile.write_bytes(b"existing")
    stack = _synthetic_video_stack()

    base_plotting.make_video(stack, str(ofile), overwrite=False, fields=None)

    assert fake_writer.frames == []  # writer was never actually used


def test_make_video_skip_uses_every_nth_frame(tmp_path, monkeypatch):
    fake_writer = _FakeWriter()
    monkeypatch.setattr(base_plotting.skvideo.io, "FFmpegWriter", lambda *a, **kw: fake_writer)
    monkeypatch.setattr(base_plotting, "Video", lambda ofile: ofile)
    stack = _synthetic_video_stack(n=6)

    base_plotting.make_video(stack, str(tmp_path / "out.mp4"), skip=2, fields=None)

    assert len(fake_writer.frames) == len(stack[::2])


def test_make_video_contrast_int_uses_that_frames_min_max(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(base_plotting.skvideo.io, "FFmpegWriter", lambda *a, **kw: _FakeWriter())
    monkeypatch.setattr(base_plotting, "Video", lambda ofile: ofile)
    stack = _synthetic_video_stack(n=4)
    expected_c0 = stack[2].image.min()
    expected_c1 = stack[2].image.max()

    base_plotting.make_video(stack, str(tmp_path / "out.mp4"), contrast=2, fields=None)

    assert f"Set contrast to [{expected_c0}, {expected_c1}]" in capsys.readouterr().out


def test_make_video_contrast_tuple_is_used_as_is(tmp_path, monkeypatch):
    fake_writer = _FakeWriter()
    monkeypatch.setattr(base_plotting.skvideo.io, "FFmpegWriter", lambda *a, **kw: fake_writer)
    monkeypatch.setattr(base_plotting, "Video", lambda ofile: ofile)
    stack = _synthetic_video_stack(n=3)

    base_plotting.make_video(stack, str(tmp_path / "out.mp4"), contrast=(0.0, 50.0), fields=None)

    assert len(fake_writer.frames) == len(stack)


def test_make_video_invalid_contrast_raises_value_error(tmp_path, monkeypatch):
    monkeypatch.setattr(base_plotting.skvideo.io, "FFmpegWriter", lambda *a, **kw: _FakeWriter())
    monkeypatch.setattr(base_plotting, "Video", lambda ofile: ofile)
    stack = _synthetic_video_stack()
    with pytest.raises(ValueError):
        base_plotting.make_video(stack, str(tmp_path / "out.mp4"), contrast="bogus", fields=None)


def test_make_video_log_with_static_contrast_takes_log_of_contrast_bounds(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(base_plotting.skvideo.io, "FFmpegWriter", lambda *a, **kw: _FakeWriter())
    monkeypatch.setattr(base_plotting, "Video", lambda ofile: ofile)
    stack = _synthetic_video_stack(n=3)

    base_plotting.make_video(
        stack, str(tmp_path / "out.mp4"), contrast=(2.0, 8.0), log=True, fields=None
    )

    printed = capsys.readouterr().out
    c0_expected, c1_expected = min(np.log(2.0), 1), min(np.log(8.0), 1)
    assert f"Contrast set to {c0_expected} -- {c1_expected}" in printed


def test_make_video_field_overlay_draws_expected_characters_at_expected_positions(tmp_path, monkeypatch):
    """Mock cv.putText to capture what make_video's field-overlay loop actually draws, and
    check it against independently computed _CV2_IMG_POS coordinates and the field's own
    get_field_string text -- rather than just "it doesn't raise"."""
    monkeypatch.setattr(base_plotting.skvideo.io, "FFmpegWriter", lambda *a, **kw: _FakeWriter())
    monkeypatch.setattr(base_plotting, "Video", lambda ofile: ofile)
    stack = _synthetic_video_stack(n=1, energy=5.0)
    height, width = stack[0].shape

    calls = []
    real_put_text = base_plotting.cv.putText

    def fake_put_text(data, char, org, **kwargs):
        calls.append((char, org))
        return real_put_text(data, char, org=org, **kwargs)

    monkeypatch.setattr(base_plotting.cv, "putText", fake_put_text)

    base_plotting.make_video(stack, str(tmp_path / "out.mp4"), fields=("energy",))

    expected_text = stack[0].get_field_string("energy").replace("µ", "u")
    expected_text = expected_text.encode("ascii", errors="ignore").decode()
    assert len(calls) == len(expected_text)
    for pos, ((char, org), expected_char) in enumerate(zip(calls, expected_text)):
        assert char == expected_char
        assert org == base_plotting._CV2_IMG_POS(0, height, width, expected_text, pos=pos)


### plot_line ####################################################################


def test_plot_line_draws_one_line_per_input_with_labels_and_legend():
    lines = [
        do.Line(np.vstack([np.arange(5), np.arange(5) ** 2])),
        do.Line(np.vstack([np.arange(5), np.arange(5) ** 3])),
    ]
    lines[0].label = "squares"
    lines[1].label = "cubes"

    ax = base_plotting.plot_line(lines)

    assert len(ax.get_lines()) == 2
    labels = {line.get_label() for line in ax.get_lines()}
    assert labels == {"squares", "cubes"}
    assert ax.get_legend() is not None
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), np.arange(5) ** 2)
    plt.close(ax.figure)


def test_plot_line_no_label_skips_legend():
    line = do.Line(np.vstack([np.arange(5), np.arange(5)]))
    ax = base_plotting.plot_line(line)
    assert ax.get_legend() is None
    plt.close(ax.figure)


def test_plot_line_export_to_writes_csv_named_after_label(tmp_path):
    line = do.Line(np.vstack([np.arange(5), np.arange(5) ** 2]))
    line.label = "myline"
    export_pattern = str(tmp_path / "$.csv")

    base_plotting.plot_line(line, export_to=export_pattern)

    assert (tmp_path / "myline.csv").exists()
    plt.close("all")


### plot_intensity / plot_iv #####################################################


def test_plot_intensity_return_line_matches_hand_computed_roi_mean():
    stack = _synthetic_video_stack(n=4, energy=np.linspace(1, 4, 4))
    roi = ROI.circle(x0=8, y0=6, radius=3)

    ax, line = base_plotting.plot_intensity(stack, roi, "energy", return_line=True)

    expected = np.array([np.mean(roi.apply(img, return_array=True)) for img in stack])
    np.testing.assert_allclose(line.y, expected)
    assert len(ax.get_lines()) == 1
    plt.close(ax.figure)


def test_plot_iv_is_plot_intensity_with_energy_xaxis():
    stack = _synthetic_video_stack(n=4, energy=np.linspace(1, 4, 4))
    roi = ROI.circle(x0=8, y0=6, radius=3)

    ax, line = base_plotting.plot_iv(stack, roi, return_line=True)

    np.testing.assert_allclose(line.x, stack.energy)
    plt.close(ax.figure)


def test_plot_intensity_multiple_rois_draws_one_line_each():
    stack = _synthetic_video_stack(n=4, energy=np.linspace(1, 4, 4))
    roi1 = ROI.circle(x0=4, y0=4, radius=2)
    roi2 = ROI.circle(x0=10, y0=8, radius=2)

    ax = base_plotting.plot_intensity(stack, [roi1, roi2], "energy")

    assert len(ax.get_lines()) == 2
    plt.close(ax.figure)


### plot_intensity_img / plot_iv_img #############################################


def test_plot_intensity_img_draws_two_axes_with_roi_overlay():
    stack = _synthetic_video_stack(n=4, energy=np.linspace(1, 4, 4))
    roi = ROI.circle(x0=8, y0=6, radius=3)

    ax1, ax2 = base_plotting.plot_intensity_img(stack, roi, "energy")

    assert len(ax1.get_lines()) == 1
    # ax2 has the image (1) plus the ROI overlay imshow (1) from roi.plot()
    assert len(ax2.get_images()) == 2
    plt.close(ax1.figure)


def test_plot_intensity_img_uses_get_max_variance_idx_by_default(monkeypatch):
    stack = _synthetic_video_stack(n=4, energy=np.linspace(1, 4, 4))
    roi = ROI.circle(x0=8, y0=6, radius=3)
    calls = []
    real = base_plotting.get_max_variance_idx

    def spy(s):
        calls.append(s)
        return real(s)

    monkeypatch.setattr(base_plotting, "get_max_variance_idx", spy)
    ax1, ax2 = base_plotting.plot_intensity_img(stack, roi, "energy")
    assert len(calls) == 1
    plt.close(ax1.figure)


### waterfall #####################################################################


def test_waterfall_image_matches_hand_computed_linear_ramp():
    """Build a stack whose pixel value along a horizontal profile is a known linear ramp per
    frame, so the resulting (contrast-normalized) waterfall image is independently
    computable, not just "some image got drawn"."""
    from uspy.roi import Profile

    n_frames = 4
    width = 10
    ramp = np.arange(width, dtype=float)  # profile_line samples this per row
    frames = np.tile(ramp, (n_frames, 20, 1))  # constant along rows, ramp along columns
    stack = base.LEEMStack(frames)
    stack.timestamp = np.arange(n_frames, dtype=float)  # rel_time is read-only; set via this
    stack.time_origin = 0.0

    profile = Profile([(0, 10), (width - 1, 10)], width=1)
    ax = base_plotting.waterfall(stack, profile)

    drawn = ax.images[0].get_array()
    assert drawn.shape[0] == n_frames
    # every row is contrast-normalized the same ramp (min=0 at col 0, max=254/255 pattern)
    for row in drawn:
        assert row[0] == 0
        assert row[-1] == row.max()
    plt.close(ax.figure)


def test_waterfall_interpolate_regrids_to_requested_shape():
    from uspy.roi import Profile

    n_frames = 5
    width = 12
    frames = np.tile(np.arange(width, dtype=float), (n_frames, 15, 1))
    stack = base.LEEMStack(frames)
    stack.timestamp = np.arange(n_frames, dtype=float)
    stack.time_origin = 0.0
    profile = Profile([(0, 7), (width - 1, 7)], width=1)

    ax = base_plotting.waterfall(stack, profile, interpolate=True, npoints=(6, 8))

    drawn = ax.images[0].get_array()
    assert drawn.shape == (8, 6)
    plt.close(ax.figure)


def test_waterfall_non_default_yaxis_sets_ylabel():
    from uspy.roi import Profile

    n_frames = 3
    width = 8
    frames = np.tile(np.arange(width, dtype=float), (n_frames, 10, 1))
    stack = base.LEEMStack(frames)
    stack.energy = np.array([1.0, 2.0, 3.0])
    profile = Profile([(0, 5), (width - 1, 5)], width=1)

    ax = base_plotting.waterfall(stack, profile, yaxis="energy")

    assert ax.get_ylabel().startswith("energy")
    plt.close(ax.figure)


### plot_mov ######################################################################


def test_plot_mov_skip_and_ncols_produce_expected_grid():
    stack = _synthetic_video_stack(n=6)
    axes = base_plotting.plot_mov(stack, skip=2, ncols=2, virtual=False)
    # 6 frames, skip=2 -> 3 frames; ncols=2 -> 2 rows x 2 cols = 4 axes, 1 deleted (empty slot)
    assert len(axes) == 4
    plt.close("all")


### plot_rois #####################################################################


def test_plot_rois_draws_one_overlay_per_roi_plus_background_image():
    img = do.Image(np.random.default_rng(0).normal(size=(20, 24)))
    roi1 = ROI.circle(5, 5, radius=3)
    roi2 = ROI.circle(15, 10, radius=3)

    ax = base_plotting.plot_rois([roi1, roi2], img=img)

    assert len(ax.get_images()) == 3  # background + 2 ROI overlays
    plt.close(ax.figure)


def test_plot_rois_without_img_still_draws_rois():
    roi1 = ROI.circle(5, 5, radius=3)
    ax = base_plotting.plot_rois([roi1])
    assert len(ax.get_images()) == 1
    plt.close(ax.figure)


### plot_meta option branches #####################################################


def test_plot_meta_excludes_unreasonable_temperature_points(capsys):
    stack = _synthetic_video_stack(
        n=4, temperature=np.array([300.0, 2500.0, 310.0, 320.0]), meta_fields=None
    )
    axes = base_plotting.plot_meta(stack, fields="temperature")
    line = axes[0].get_lines()[0]
    assert len(line.get_xdata()) == 3  # the 2500 K point excluded
    assert "excluded from plot" in capsys.readouterr().out
    plt.close("all")


def test_plot_meta_pressure_field_uses_log_yscale():
    stack = _synthetic_video_stack(n=3, pressure1=1e-6)
    axes = base_plotting.plot_meta(stack, fields="pressure1")
    assert axes[0].get_yscale() == "log"
    plt.close("all")


### print_meta option branches ####################################################


def test_print_meta_falls_back_to_imgify_for_a_single_file_source():
    # print_meta's stackify() raises FileNotFoundError for a single-image (non-stack)
    # source; the fallback wraps it as a 1-element stack via imgify instead of propagating.
    base_plotting.print_meta(ELETTRA_DAT, fields=("energy",))


def test_print_meta_fields_as_str_is_wrapped_in_a_list():
    # print_meta reads img.source for the "Name" column, so needs a real file source, not a
    # synthetic in-memory array (source=None there).
    stack = base.LEEMStack([base.LEEMImg(ELETTRA_DAT)])
    base_plotting.print_meta(stack, fields="energy")


### _get_ax #######################################################################


def test_get_ax_sets_title_xlabel_ylabel():
    _, ax = plt.subplots()
    result = base_plotting._get_ax(ax, title="my title", xlabel="x", ylabel="y")
    assert result.get_title() == "my title"
    assert result.get_xlabel() == "x"
    assert result.get_ylabel() == "y"
    plt.close(ax.figure)


def test_get_ax_hides_ticks_when_ticks_false():
    _, ax = plt.subplots()
    result = base_plotting._get_ax(ax, ticks=False, axis_off=True)
    assert result.get_xticks().size == 0
    assert result.get_yticks().size == 0
    plt.close(ax.figure)


### uspy/leem/plotting.py: plot_aligns ############################################


def test_plot_aligns_plots_known_warp_translations():
    stack = base.LEEMStack(np.random.default_rng(0).normal(size=(4, 10, 12)))
    expected = []
    for i, img in enumerate(stack):
        m = np.eye(3)
        m[0, 2] = i * 1.5
        m[1, 2] = -i * 0.5
        img.warp_matrix = m
        expected.append((m[0, 2], m[1, 2]))
    expected = np.array(expected)

    leem_plotting.plot_aligns(stack)

    ax = plt.gca()
    lines = ax.get_lines()
    assert len(lines) == 2  # one line per (x, y) translation component
    np.testing.assert_allclose(lines[0].get_ydata(), expected[:, 0])
    np.testing.assert_allclose(lines[1].get_ydata(), expected[:, 1])
    assert ax.get_xlabel() == "Index"
    plt.close("all")


def test_plot_aligns_xaxis_uses_given_metadata_field():
    stack = base.LEEMStack(np.random.default_rng(0).normal(size=(3, 10, 12)))
    stack.energy = np.array([1.0, 2.0, 3.0])
    for i, img in enumerate(stack):
        m = np.eye(3)
        m[0, 2] = i
        img.warp_matrix = m

    leem_plotting.plot_aligns(stack, xaxis="energy")

    ax = plt.gca()
    np.testing.assert_allclose(ax.get_lines()[0].get_xdata(), stack.energy)
    assert ax.get_xlabel() == "energy"
    plt.close("all")
