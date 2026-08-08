"""Tests for uspy.interactive.profile.ProfilePicker -- the ipywidgets shell wiring
Canvas to widget controls and the profile-line/waterfall preview panel. Mirrors
tests/test_interactive_picker.py's structure and testing technique (real ipympl
backend, synthetic matplotlib events) -- see that file's module docstring for why.
"""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=protected-access

import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

pytest.importorskip("ipywidgets")
pytest.importorskip("ipympl")

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent, KeyEvent

import uspy.dataobject as do
import uspy.derived as derived
from uspy.roi import ROI, Profile
from uspy.interactive.spec import Spec
from uspy.interactive.profile import ProfilePicker, pick_profiles

H, W, N = 60, 80, 6


@pytest.fixture(scope="module", autouse=True)
def _ipympl_backend():
    original = matplotlib.get_backend()
    matplotlib.use("module://ipympl.backend_nbagg")
    yield
    matplotlib.use(original)


def _img(h=H, w=W, seed=0):
    rng = np.random.default_rng(seed)
    return do.Image(rng.random((h, w)).astype("float32"))


def _stack(n=N, h=H, w=W, seed=0):
    rng = np.random.default_rng(seed)
    imgs = [do.Image(rng.random((h, w)).astype("float32")) for _ in range(n)]
    stack = do.ImageStack(imgs)
    for i, im in enumerate(stack):
        im.energy = float(i)
    return stack


@pytest.fixture
def img():
    return _img()


@pytest.fixture
def stack():
    return _stack()


@pytest.fixture
def line_picker(img):
    p = ProfilePicker(img)
    yield p
    p.canvas.close()
    plt.close(p.fig)


@pytest.fixture
def wf_picker(stack):
    p = ProfilePicker(stack, yaxis="energy")
    yield p
    p.canvas.close()
    plt.close(p.fig)


def _press(axes, xy, dblclick=False):
    x, y = axes.transData.transform(xy)
    ev = MouseEvent("button_press_event", axes.figure.canvas, x, y, button=1, dblclick=dblclick)
    axes.figure.canvas.callbacks.process("button_press_event", ev)


def _move(axes, xy):
    x, y = axes.transData.transform(xy)
    ev = MouseEvent("motion_notify_event", axes.figure.canvas, x, y)
    axes.figure.canvas.callbacks.process("motion_notify_event", ev)


def _release(axes, xy):
    x, y = axes.transData.transform(xy)
    ev = MouseEvent("button_release_event", axes.figure.canvas, x, y, button=1)
    axes.figure.canvas.callbacks.process("button_release_event", ev)


def _key(axes, key):
    ev = KeyEvent("key_press_event", axes.figure.canvas, key)
    axes.figure.canvas.callbacks.process("key_press_event", ev)


def _draw_profile(ax, start, end):
    """Click-chain a two-point profile, Fiji-style: click, then double-click to finish."""
    _press(ax, start)
    _press(ax, end, dblclick=True)


### construction ##############################################################


def test_picker_constructs_with_no_regions(line_picker):
    assert line_picker.specs == []
    assert line_picker.code == "# nothing picked yet"


def test_pick_profiles_returns_and_displays_picker(img):
    p = pick_profiles(img)
    assert isinstance(p, ProfilePicker)
    p.canvas.close()
    plt.close(p.fig)


def test_picker_has_no_ipython_display_hook(img):
    assert not hasattr(ProfilePicker, "_ipython_display_")


def test_single_image_has_no_frame_or_stack_widgets(line_picker):
    assert line_picker.stack is None
    assert not hasattr(line_picker, "w_frame")
    assert not hasattr(line_picker, "w_step")
    assert not hasattr(line_picker, "w_cmap")


def test_stack_gets_frame_and_cmap_widgets(wf_picker):
    assert hasattr(wf_picker, "w_frame")
    assert hasattr(wf_picker, "w_step")
    assert hasattr(wf_picker, "w_cmap")


### seeding #####################################################################


def test_seeding_from_existing_profiles(stack):
    prof = Profile([(5, 5), (20, 20)], width=2, label="cut")
    p = ProfilePicker(stack, profiles=[prof])
    assert len(p.specs) == 1
    assert "cut" in p.code
    p.canvas.close()
    plt.close(p.fig)


def test_seeding_with_an_roi_raises(stack):
    r = ROI.circle(20, 20, radius=5)
    with pytest.raises(ValueError, match="Profile"):
        ProfilePicker(stack, profiles=[r])


### single image: 1D line preview ###############################################


def test_drawing_a_profile_adds_a_line_to_the_preview(line_picker):
    _draw_profile(line_picker.ax_img, (10, 10), (50, 50))
    assert len(line_picker.specs) == 1
    assert len(line_picker.ax_preview.lines) == 1


def test_two_profiles_overlay_as_two_lines(line_picker):
    _draw_profile(line_picker.ax_img, (10, 10), (50, 10))
    _draw_profile(line_picker.ax_img, (10, 20), (50, 20))
    assert len(line_picker.specs) == 2
    assert len(line_picker.ax_preview.lines) == 2


def test_line_preview_matches_profileline(line_picker):
    _draw_profile(line_picker.ax_img, (10, 10), (50, 10))
    spec = line_picker.specs[0]
    canonical = derived.ProfileLine(line_picker.frames[0], spec.to_object())
    drawn = line_picker.ax_preview.lines[0]
    np.testing.assert_allclose(drawn.get_ydata(), canonical.y)


### stack: waterfall preview follows the selection ##############################


def test_drawing_a_profile_on_a_stack_shows_a_waterfall(wf_picker):
    _draw_profile(wf_picker.ax_img, (10, 10), (50, 50))
    assert len(wf_picker.ax_preview.images) == 1


def test_waterfall_matches_derived_waterfall(wf_picker):
    _draw_profile(wf_picker.ax_img, (10, 10), (50, 10))
    spec = wf_picker.specs[0]
    wf = derived.Waterfall(wf_picker.stack, spec.to_object(), "energy")
    drawn = wf_picker.ax_preview.images[0].get_array()
    assert drawn.shape == wf.image.shape


def test_second_profile_switches_the_waterfall_to_it(wf_picker):
    _draw_profile(wf_picker.ax_img, (10, 10), (50, 10))
    _draw_profile(wf_picker.ax_img, (5, 20), (55, 20))
    assert len(wf_picker.specs) == 2
    assert "1" in wf_picker.ax_preview.get_title()  # newest region auto-selected


def test_selecting_via_list_swaps_the_waterfall_back(wf_picker):
    _draw_profile(wf_picker.ax_img, (10, 10), (50, 10))
    _draw_profile(wf_picker.ax_img, (5, 20), (55, 20))
    wf_picker.canvas.select(0)
    assert wf_picker.ax_preview.get_title().startswith("profile 0")
    wf_picker.canvas.select(1)
    assert wf_picker.ax_preview.get_title().startswith("profile 1")


### live drag ####################################################################


def test_live_defaults_on_for_single_image(line_picker):
    assert line_picker.live is True


def test_live_defaults_on_for_resident_stack(wf_picker):
    assert wf_picker.live is True


def test_live_defaults_off_for_virtual_stack():
    from uspy.leem.base import LEEMStack
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vstack = LEEMStack("testdata/003_IV", virtual=True)
        p = ProfilePicker(vstack, yaxis="energy")
    assert p.live is False
    p.canvas.close()
    plt.close(p.fig)


def test_live_on_virtual_stack_warns():
    from uspy.leem.base import LEEMStack
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vstack = LEEMStack("testdata/003_IV", virtual=True)
    with pytest.warns(UserWarning, match="materialize"):
        p = ProfilePicker(vstack, yaxis="energy", live=True)
    p.canvas.close()
    plt.close(p.fig)


def test_drag_updates_preview_live(wf_picker):
    wf_picker.live = True
    wf_picker.canvas.new_shape = "profile"
    _draw_profile(wf_picker.ax_img, (10, 10), (50, 10))
    wf_picker.canvas.select(0)
    spec = wf_picker.specs[0]
    x0, y0 = dict(spec.params)["points"][1]
    _press(wf_picker.ax_img, (x0, y0))
    _move(wf_picker.ax_img, (60, 30))
    # a live drag recomputes a fresh Waterfall for the dragged vertex position
    assert len(wf_picker.ax_preview.images) == 1
    _release(wf_picker.ax_img, (60, 30))


def test_live_off_does_not_recompute_on_drag(wf_picker, monkeypatch):
    wf_picker.live = False
    calls = []
    real_waterfall = derived.Waterfall

    def counting(*args, **kwargs):
        calls.append(1)
        return real_waterfall(*args, **kwargs)

    monkeypatch.setattr(derived, "Waterfall", counting)
    spec = Spec("profile", (("points", ((10, 10), (50, 10))), ("width", 1)), "red")
    wf_picker._on_drag(0, spec)
    assert calls == []


### bounded live-drag preview for very large stacks (safety valve) ##########
# Shared with ROIPicker (see tests/test_interactive_picker.py) -- both go through
# PickerBase._live_frame_stride, so these tests exercise ProfilePicker's own use of
# it (Waterfall built from a subsampled stack) rather than the helper itself.


@pytest.fixture
def big_stack():
    return _stack(n=200)


@pytest.fixture
def big_wf_picker(big_stack):
    p = ProfilePicker(big_stack, yaxis="energy")
    yield p
    p.canvas.close()
    plt.close(p.fig)


def test_drag_on_a_large_stack_samples_at_most_the_cap(big_wf_picker):
    big_wf_picker.canvas.new_shape = "profile"
    _draw_profile(big_wf_picker.ax_img, (10, 10), (50, 10))
    big_wf_picker.canvas.select(0)
    spec = big_wf_picker.specs[0]
    x0, y0 = dict(spec.params)["points"][1]
    _press(big_wf_picker.ax_img, (x0, y0))
    _move(big_wf_picker.ax_img, (60, 30))
    drawn = big_wf_picker.ax_preview.images[0].get_array()
    assert drawn.shape[0] <= big_wf_picker._LIVE_MAX_FRAMES
    assert drawn.shape[0] < len(big_wf_picker.stack)
    _release(big_wf_picker.ax_img, (60, 30))


def test_drag_preview_on_a_large_stack_annotates_the_title(big_wf_picker):
    big_wf_picker.canvas.new_shape = "profile"
    _draw_profile(big_wf_picker.ax_img, (10, 10), (50, 10))
    big_wf_picker.canvas.select(0)
    spec = big_wf_picker.specs[0]
    x0, y0 = dict(spec.params)["points"][1]
    _press(big_wf_picker.ax_img, (x0, y0))
    _move(big_wf_picker.ax_img, (60, 30))
    assert "preview" in big_wf_picker.ax_preview.get_title()
    _release(big_wf_picker.ax_img, (60, 30))
    # released: the exact, full-resolution, cached Waterfall replaces the capped one
    drawn = big_wf_picker.ax_preview.images[0].get_array()
    assert drawn.shape[0] == len(big_wf_picker.stack)
    assert "preview" not in big_wf_picker.ax_preview.get_title()


def test_small_stack_drag_is_never_capped(wf_picker):
    # no behavior change for the common case -- the 6-frame fixture stack is well
    # under the cap, so the cap must never engage for it.
    wf_picker.canvas.new_shape = "profile"
    _draw_profile(wf_picker.ax_img, (10, 10), (50, 10))
    wf_picker.canvas.select(0)
    spec = wf_picker.specs[0]
    x0, y0 = dict(spec.params)["points"][1]
    _press(wf_picker.ax_img, (x0, y0))
    _move(wf_picker.ax_img, (20, 20))
    drawn = wf_picker.ax_preview.images[0].get_array()
    assert drawn.shape[0] == len(wf_picker.stack)
    assert "preview" not in wf_picker.ax_preview.get_title()
    _release(wf_picker.ax_img, (20, 20))


### width editing ################################################################


def test_constructor_width_seeds_the_widget_and_new_regions(img):
    # regression: ProfilePicker(width=N) used to be silently reset to 1 -- the
    # initial _sync_controls() call ran before Canvas.new_width was ever seeded
    # from the constructor argument.
    p = ProfilePicker(img, width=5)
    try:
        assert p.w_width.value == 5
        assert p.canvas.new_width == 5
        _draw_profile(p.ax_img, (10, 10), (50, 10))
        assert dict(p.canvas.specs[0].params)["width"] == 5
    finally:
        p.canvas.close()
        plt.close(p.fig)


def test_width_contextual_for_next_region(line_picker):
    line_picker.w_width.value = 5
    assert line_picker.canvas.new_width == 5


def test_width_contextual_for_selected_region(line_picker):
    _draw_profile(line_picker.ax_img, (10, 10), (50, 10))
    line_picker.canvas.select(0)
    line_picker.w_width.value = 7
    assert dict(line_picker.canvas.specs[0].params)["width"] == 7


### code / profiles properties ###################################################


def test_code_emits_profiles_block(line_picker):
    _draw_profile(line_picker.ax_img, (10, 10), (50, 10))
    assert "profiles = [" in line_picker.code
    assert "Profile(" in line_picker.code


def test_profiles_property_returns_fresh_objects(line_picker):
    _draw_profile(line_picker.ax_img, (10, 10), (50, 10))
    profs = line_picker.profiles
    assert len(profs) == 1
    assert isinstance(profs[0], Profile)


def test_picker_has_no_rois_property(line_picker):
    assert not hasattr(line_picker, "rois")


### does not mutate the source ###################################################


def test_picker_does_not_mutate_a_single_image(img):
    before = img.image.copy()
    p = ProfilePicker(img)
    _draw_profile(p.ax_img, (10, 10), (50, 50))
    p.canvas.select(0)
    p.w_width.value = 4
    p.canvas.delete_selected()
    assert_array_equal(img.image, before)
    p.canvas.close()
    plt.close(p.fig)


def test_picker_does_not_mutate_the_stack(stack):
    original = [im.image.copy() for im in stack]
    p = ProfilePicker(stack, yaxis="energy")
    _draw_profile(p.ax_img, (10, 10), (50, 50))
    p.w_frame.value = min(2, len(stack) - 1)
    p.canvas.select(0)
    p.canvas.delete_selected()
    for im, before in zip(stack, original):
        assert_array_equal(im.image, before)
    p.canvas.close()
    plt.close(p.fig)
