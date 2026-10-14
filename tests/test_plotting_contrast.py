"""Static-contrast/log/inner handling (uspy-cu7.1, cu7.2) and plot_line kwargs (cu7.3).

Assertions are on rendered gray levels (imshow's norm+cmap applied to the drawn array, or the
frames handed to the video writer), not on intermediate contrast tuples."""
# pylint: disable=missing-docstring

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

import uspy.dataobject as do
import uspy.plotting as plotting
from uspy.leem import base
from uspy.roi import Profile, ROI

from .test_plotting import _FakeWriter


def _gray(ax):
    """The gray level (0-255) each pixel is actually rendered with."""
    im = ax.images[0]
    return im.to_rgba(im.get_array(), bytes=True)[..., 0].astype(int)


def _plot(arr, **kwargs):
    ax = plotting.plot_img(do.Image(np.asarray(arr)), fields=(), **kwargs)
    gray = _gray(ax)
    clim = ax.images[0].get_clim()
    plt.close(ax.figure)
    return gray, clim


def _video(monkeypatch, tmp_path, frames, **kwargs):
    writer = _FakeWriter()
    monkeypatch.setattr(plotting.skvideo.io, "FFmpegWriter", lambda *a, **kw: writer)
    monkeypatch.setattr(plotting, "Video", lambda ofile: ofile)
    kwargs.setdefault("fields", None)
    plotting.make_video(base.LEEMStack(np.asarray(frames)), str(tmp_path / "o.mp4"), **kwargs)
    return writer.frames


def _linear(arr, lo, hi):
    return np.clip((np.clip(arr, lo, hi) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


# frame 1 is frame 0 dimmed to half brightness: a genuine intensity change.
BRIGHT = np.array([[0.0, 25.0], [50.0, 100.0]])
DIM = BRIGHT / 2


### uspy-cu7.1: a value maps to the same gray level in every frame #################


@pytest.mark.parametrize("contrast", [(0.0, 100.0), np.array([0.0, 100.0])])
def test_plot_img_static_contrast_keeps_intensity_changes(contrast):
    (g_bright, clim), (g_dim, _) = _plot(BRIGHT, contrast=contrast), _plot(DIM, contrast=contrast)
    assert clim == (0, 255)
    np.testing.assert_allclose(g_bright, _linear(BRIGHT, 0, 100), atol=1)
    np.testing.assert_allclose(g_dim, _linear(DIM, 0, 100), atol=1)
    assert g_dim.max() < 140  # was re-stretched to 255 before
    assert g_bright[0, 1] == g_dim[1, 0]  # 25 -> same gray in both frames


def test_plot_img_static_contrast_invert_is_255_minus_mapped():
    gray, _ = _plot(DIM, contrast=(0.0, 100.0), invert=True)
    np.testing.assert_allclose(gray, 255 - _linear(DIM, 0, 100).astype(int), atol=1)


def test_plot_img_auto_still_autoscales_each_frame():
    np.testing.assert_array_equal(_plot(BRIGHT)[0], _plot(DIM)[0])


@pytest.mark.parametrize("contrast", [(0.0, 100.0), "maximum", 0])
def test_make_video_static_contrast_maps_linearly(monkeypatch, tmp_path, contrast):
    frames = np.stack([BRIGHT, DIM, DIM / 5 + 3])
    # "maximum" -> (0, 100) over the whole stack; int 0 -> frame 0's range (0, 100)
    out = _video(monkeypatch, tmp_path, frames, contrast=contrast)
    assert len(out) == 3
    for frame, ref in zip(out, frames):
        assert frame.dtype == np.uint8
        np.testing.assert_array_equal(frame, _linear(ref, 0, 100))


def test_make_video_static_contrast_with_fields_is_not_restretched(monkeypatch, tmp_path):
    frame = np.full((120, 160), 30.0)  # uniform, well below contrast max
    frame[0, 0] = 100.0
    out = _video(monkeypatch, tmp_path, [frame, frame], contrast=(0.0, 100.0), fields=("energy",))
    centre = out[1][60, 80]
    np.testing.assert_array_equal(centre, [int(30 / 100 * 255)] * 3)


def test_make_video_auto_still_autoscales_each_frame(monkeypatch, tmp_path):
    out = _video(monkeypatch, tmp_path, np.stack([BRIGHT, DIM]), contrast="auto")
    np.testing.assert_array_equal(out[0], out[1])
    assert out[1].max() == 255


def test_make_video_static_contrast_keeps_mask_white_and_linear_inside(monkeypatch, tmp_path):
    frames = np.stack([np.full((20, 20), 40.0), np.full((20, 20), 80.0)])
    mask = ROI.circle(x0=10, y0=10, radius=4)
    inside = mask.pad_to(20, 20).astype(bool)
    out = _video(monkeypatch, tmp_path, frames, contrast=(0.0, 100.0), mask=mask)
    for frame, value in zip(out, (40.0, 80.0)):
        assert (frame[~inside] == 255).all()
        assert (frame[inside] == int(value / 100 * 255)).all()


### uspy-cu7.2: ndarray / int / log / inner ##########################################


def test_plot_img_int_contrast_with_log_is_converted_before_log():
    arr = np.array([[1, 10], [100, 1000]])  # integer image, integer contrast
    gray, _ = _plot(arr, log=True, contrast=(1, 100))
    expected = _linear(np.log(arr.astype(float)), 0.0, np.log(100.0))
    np.testing.assert_allclose(gray, expected, atol=1)
    assert gray[1, 0] == 255 and abs(gray[0, 1] - 127) <= 1  # log(10)/log(100) = 1/2


def test_plot_img_inner_uses_the_displayed_log_data():
    arr = np.full((20, 20), 2.0)
    arr[0, 0] = 1e6  # outside the inner 60 %
    arr[10, 10] = 100.0
    gray, _ = _plot(arr, log=True, contrast="inner")
    assert gray[10, 10] == 255 and gray[0, 0] == 255
    assert gray[5, 5] == 0  # inner minimum, log(2)


def test_plot_img_inner_ignores_nan():
    arr = np.tile(np.arange(20.0), (20, 1)) + 1
    arr[10, 10] = np.nan
    gray, _ = _plot(arr, contrast="inner")
    assert gray[10, 14] == 255 or gray[10, 15] == 255  # inner column range is finite
    assert gray.max() == 255


def test_plot_img_inner_ignores_masked_pixels():
    arr = np.full((40, 40), 10.0)
    arr[20, 20] = 50.0
    arr[20, 28] = 900.0  # inside the inner region but outside the mask
    mask = ROI.circle(x0=20, y0=20, radius=6)
    ax = plotting.plot_img(do.Image(arr), fields=(), contrast="inner", mask=mask)
    gray = _gray(ax)
    plt.close(ax.figure)
    assert gray[20, 20] == 255
    assert gray[20, 21] == 0


def test_plot_img_inner_on_tiny_image_does_not_crash():
    gray, _ = _plot(np.arange(16.0).reshape(4, 4), contrast="inner")  # 20 % of 4 px = 0
    assert gray.max() == 255


def test_plot_img_invalid_string_contrast_raises():
    with pytest.raises(ValueError):
        _plot(BRIGHT, contrast="bogus")


def _ramp_stack(n_frames=4, width=10):
    frames = np.tile(np.arange(width, dtype=float) + 1, (n_frames, 20, 1))
    stack = base.LEEMStack(frames * (np.arange(n_frames)[:, None, None] + 1))
    stack.timestamp = np.arange(n_frames, dtype=float)
    stack.time_origin = 0.0
    return stack


@pytest.mark.parametrize("contrast", [(1.0, 20.0), np.array([1.0, 20.0])])
def test_waterfall_log_with_explicit_contrast(contrast):
    stack = _ramp_stack()
    profile = Profile([(0, 10), (9, 10)], width=1)
    ax = plotting.waterfall(stack, profile, log=True, contrast=contrast)
    drawn = ax.images[0].get_array()
    ref = plotting.derived.Waterfall(stack, profile, "rel_time").image
    np.testing.assert_array_equal(drawn, _linear(np.log(ref), 0.0, np.log(20.0)))
    assert ax.images[0].get_clim() == (0, 255)
    plt.close(ax.figure)


def test_waterfall_fixed_contrast_keeps_intensity_changes():
    stack = _ramp_stack()
    profile = Profile([(0, 10), (9, 10)], width=1)
    ax = plotting.waterfall(stack, profile, contrast=(0.0, 40.0))
    gray = _gray(ax)
    plt.close(ax.figure)
    assert (gray.max(axis=1) == np.sort(gray.max(axis=1))).all()  # brighter frames -> brighter
    assert len(set(gray.max(axis=1))) == len(gray)


def test_make_video_log_limits_are_not_capped(monkeypatch, tmp_path):
    frame = np.array([[1.0, 10.0], [100.0, 1000.0]])
    out = _video(monkeypatch, tmp_path, [frame, frame], contrast=(1.0, 1000.0), log=True)
    expected = _linear(np.log(frame), 0.0, np.log(1000.0))
    np.testing.assert_array_equal(out[0], expected)
    assert out[0][1, 1] == 255 and out[0][0, 1] == 85  # log10/log1000 = 1/3


def test_make_video_log_maximum_and_auto_with_zero_pixel(monkeypatch, tmp_path):
    frame = np.array([[0.0, 10.0], [100.0, 1000.0]])
    for contrast in ("auto", "maximum"):
        out = _video(monkeypatch, tmp_path, [frame], contrast=contrast, log=True)[0]
        assert out[1, 1] == 255 and out[1, 0] > 0, contrast  # zero pixel must not zero the scale


def test_make_video_ndarray_contrast_and_inner_log(monkeypatch, tmp_path):
    out = _video(monkeypatch, tmp_path, [BRIGHT, DIM], contrast=np.array([0.0, 100.0]))
    np.testing.assert_array_equal(out[1], _linear(DIM, 0, 100))

    frame = np.full((20, 20), 2.0)
    frame[0, 0] = 1e6
    frame[10, 10] = 100.0
    out = _video(monkeypatch, tmp_path, [frame], contrast="inner", log=True)[0]
    assert out[10, 10] == 255 and out[5, 5] == 0


### uspy-cu7.3: plot_line #############################################################


def _lines():
    return [
        do.Line(np.vstack([np.arange(5), np.arange(5) ** 2])),
        do.Line(np.vstack([np.arange(5), np.arange(5) ** 3])),
    ]


def test_plot_line_scalar_label_and_color_apply_to_every_line_without_mutation():
    lines = _lines()
    before = [(li.label, li.color) for li in lines]
    ax = plotting.plot_line(lines, label="x", color="r", linewidth=3)
    assert [ln.get_label() for ln in ax.get_lines()] == ["x", "x"]
    assert [ln.get_color() for ln in ax.get_lines()] == ["r", "r"]
    assert all(ln.get_linewidth() == 3 for ln in ax.get_lines())
    assert [(li.label, li.color) for li in lines] == before
    plt.close(ax.figure)


def test_plot_line_per_line_label_and_color_lists():
    lines = _lines()
    ax = plotting.plot_line(lines, label=["a", "b"], color=["r", (0.0, 0.5, 0.0)])
    assert [ln.get_label() for ln in ax.get_lines()] == ["a", "b"]
    assert [ln.get_color() for ln in ax.get_lines()] == ["r", (0.0, 0.5, 0.0)]
    plt.close(ax.figure)


def test_plot_line_rgb_tuple_is_one_color_not_a_per_line_list():
    ax = plotting.plot_line(_lines(), color=(0.0, 0.5, 0.0))
    assert [ln.get_color() for ln in ax.get_lines()] == [(0.0, 0.5, 0.0)] * 2
    plt.close(ax.figure)


def test_plot_line_single_line_label_kwarg():
    line = _lines()[0]
    ax = plotting.plot_line(line, label="solo")
    assert ax.get_lines()[0].get_label() == "solo" and line.label is None
    plt.close(ax.figure)


def test_plot_line_label_list_length_mismatch_raises():
    with pytest.raises(ValueError):
        plotting.plot_line(_lines(), label=["only one"])
    plt.close("all")
