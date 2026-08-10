"""Tests for uspy.interactive.viewer.StackViewer -- the plain image + frame-slider
viewer built on FrameView (also the base PickerBase itself now sits on).

Gated on ipywidgets/ipympl being installed (the 'interactive' extra); skipped
entirely otherwise via pytest.importorskip. Uses the real ipympl backend (not Agg --
FrameView._check_backend rejects Agg on purpose), same technique as
tests/test_interactive_picker.py.
"""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=protected-access

import numpy as np
import pytest

pytest.importorskip("ipywidgets")
pytest.importorskip("ipympl")

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.text import Text

import uspy.dataobject as do
from uspy.dataobject.deferred import DeferredArray
from uspy.interactive.viewer import StackViewer, view_stack

H, W, N = 60, 80, 6


class _ArrayLoader:
    """Picklable-shaped stand-in for a real file loader -- see
    tests/test_deferred_dataobject.py's _CountingArrayLoader, same idea."""

    def __init__(self, array):
        self.array = array
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.array


@pytest.fixture(scope="module", autouse=True)
def _ipympl_backend():
    """See tests/test_interactive_picker.py's fixture of the same name -- switches the
    process-global matplotlib backend for this module's tests only."""
    original = matplotlib.get_backend()
    matplotlib.use("module://ipympl.backend_nbagg")
    yield
    matplotlib.use(original)


def _stack(n=N, h=H, w=W, seed=0, virtual=False):
    rng = np.random.default_rng(seed)
    arrays = [rng.random((h, w)).astype("float32") for _ in range(n)]
    if virtual:
        imgs = []
        for arr in arrays:
            img = do.Image(np.zeros((h, w), dtype="float32"))
            img._data["image"] = DeferredArray(_ArrayLoader(arr), shape=arr.shape, dtype=arr.dtype)
            imgs.append(img)
    else:
        imgs = [do.Image(arr) for arr in arrays]
    # virtual=True: ImageStack(objects) otherwise materializes deferred elements it's
    # handed, to keep a non-virtual stack's invariant that every element is resident.
    stack = do.ImageStack(imgs, virtual=virtual)
    for i, img in enumerate(stack):
        img.energy = float(i)
    return stack


@pytest.fixture
def stack():
    return _stack()


@pytest.fixture
def viewer(stack):
    v = StackViewer(stack)
    yield v
    plt.close(v.fig)


### construction #############################################################


def test_missing_backend_raises_actionable_error(stack):
    matplotlib.use("Agg")
    try:
        with pytest.raises(RuntimeError, match="matplotlib widget"):
            StackViewer(stack)
    finally:
        matplotlib.use("module://ipympl.backend_nbagg")


def test_single_image_rejected():
    img = do.Image(np.random.default_rng(0).random((H, W)).astype("float32"))
    with pytest.raises(ValueError, match="DataObjectStack"):
        StackViewer(img)


def test_initial_frame_is_middle_frame(viewer, stack):
    assert viewer._frame_idx == len(stack) // 2
    assert viewer.w_frame.value == len(stack) // 2


def test_explicit_frame_honored(stack):
    v = StackViewer(stack, frame=1)
    try:
        assert v._frame_idx == 1
        assert v.w_frame.value == 1
    finally:
        plt.close(v.fig)


def test_slider_bounds_match_stack(viewer, stack):
    assert viewer.w_frame.min == 0
    assert viewer.w_frame.max == len(stack) - 1


def test_view_stack_displays_once_and_returns_viewer(stack):
    v = view_stack(stack)
    try:
        assert isinstance(v, StackViewer)
    finally:
        plt.close(v.fig)


### frame/contrast redraw ####################################################


def test_frame_slider_updates_live_not_just_on_release(viewer):
    assert viewer.w_frame.continuous_update is True


def test_slider_change_redraws_displayed_frame(viewer, stack):
    before = viewer.ax_img.images[0].get_array().copy()
    viewer.w_frame.value = (viewer._frame_idx + 1) % len(stack)
    after = viewer.ax_img.images[0].get_array()
    assert not np.array_equal(before, after)


def test_frame_change_does_not_leak_base_artists(viewer, stack):
    n_before = len(viewer.ax_img.get_children())
    for i in range(min(3, len(stack))):
        viewer.w_frame.value = i
    assert len(viewer.ax_img.get_children()) == n_before


def test_contrast_change_redraws_without_leaking(viewer):
    n_before = len(viewer.ax_img.get_children())
    viewer.w_contrast.value = "inner"
    viewer.w_contrast.value = "auto"
    assert len(viewer.ax_img.get_children()) == n_before


def test_fixed_contrast_freezes_range_across_frames(viewer, stack):
    viewer.w_contrast.value = "fixed"
    frozen = viewer._contrast
    assert isinstance(frozen, tuple) and len(frozen) == 2

    viewer.w_frame.value = (viewer._frame_idx + 1) % len(stack)
    assert viewer._contrast == frozen


def test_contrast_options_include_fixed(viewer):
    assert "fixed" in viewer.w_contrast.options


### metadata overlay toggle ###################################################


def test_fields_toggle_removes_and_restores_corner_text(viewer):
    assert any(isinstance(a, Text) for a in viewer.ax_img.get_children())

    viewer.w_fields.value = False
    assert not any(isinstance(a, Text) and a.get_text() for a in viewer.ax_img.get_children())

    viewer.w_fields.value = True
    assert any(isinstance(a, Text) for a in viewer.ax_img.get_children())


### play control ##############################################################


def test_play_linked_to_frame_slider(viewer, stack):
    assert viewer.w_play.min == 0
    assert viewer.w_play.max == len(stack) - 1
    assert viewer.w_play.interval == 200

    viewer.w_play.value = 2
    assert viewer.w_frame.value == 2


def test_custom_interval_honored(stack):
    v = StackViewer(stack, interval=50)
    try:
        assert v.w_play.interval == 50
    finally:
        plt.close(v.fig)


### non-mutation / virtual stacks #############################################


def test_viewer_does_not_mutate_the_stack(viewer, stack):
    originals = [img.image.copy() for img in stack]
    for i in range(len(stack)):
        viewer.w_frame.value = i
    for img, original in zip(stack, originals):
        np.testing.assert_array_equal(img.image, original)


def test_virtual_stack_stays_virtual_after_scrubbing():
    vstack = _stack(virtual=True)
    assert vstack.virtual is True
    v = StackViewer(vstack)
    try:
        for i in range(len(vstack)):
            v.w_frame.value = i
        assert vstack.virtual is True
    finally:
        plt.close(v.fig)
