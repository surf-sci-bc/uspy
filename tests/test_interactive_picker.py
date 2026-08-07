"""Tests for uspy.interactive.picker.ROIPicker -- the ipywidgets shell wiring
Canvas to widget controls and the I(V) preview panel.

Gated on ipywidgets/ipympl being installed (the 'interactive' extra); skipped
entirely otherwise via pytest.importorskip. Uses the real ipympl backend (not Agg --
picker.py's _check_backend rejects Agg on purpose) but drives it headlessly with
synthetic matplotlib events, same technique as tests/test_interactive_canvas.py.
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
from matplotlib.colors import to_hex, to_rgba

import uspy.dataobject as do
import uspy.derived as derived
import uspy.roi as roi_mod
from uspy.roi import ROI, Profile, palette_color
from uspy.interactive.spec import ROI_SHAPES, Spec
from uspy.interactive.picker import (
    ROIPicker, pick_rois, _check_backend, _check_mpl_version, _SHAPE_BUTTON_CSS,
)

H, W, N = 60, 80, 6


@pytest.fixture(scope="module", autouse=True)
def _ipympl_backend():
    """Switch to the ipympl backend for this module's tests only, restoring whatever
    backend was active before -- matplotlib.use() is process-global, and pytest runs
    every test file in one process, so leaving this switched at module-import time (or
    forgetting to restore it) would silently change the backend for every test file
    that runs afterward (this bit tests/test_plotting.py once already: its plt.gca()
    plt.show() based tests need Agg's "non-interactive" warning path)."""
    original = matplotlib.get_backend()
    matplotlib.use("module://ipympl.backend_nbagg")
    yield
    matplotlib.use(original)


def _stack(n=N, h=H, w=W, seed=0):
    rng = np.random.default_rng(seed)
    imgs = [do.Image(rng.random((h, w)).astype("float32")) for _ in range(n)]
    stack = do.ImageStack(imgs)
    for i, img in enumerate(stack):
        img.energy = float(i)
    return stack


@pytest.fixture
def stack():
    return _stack()


@pytest.fixture
def picker(stack):
    p = ROIPicker(stack, xaxis="energy")
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


def _drag(axes, start, end):
    _press(axes, start)
    _move(axes, end)
    _release(axes, end)


def _key(axes, key):
    ev = KeyEvent("key_press_event", axes.figure.canvas, key)
    axes.figure.canvas.callbacks.process("key_press_event", ev)


### construction / backend guard #########################################


def test_missing_backend_raises_actionable_error(stack):
    matplotlib.use("Agg")
    try:
        with pytest.raises(RuntimeError, match="matplotlib widget"):
            ROIPicker(stack)
    finally:
        matplotlib.use("module://ipympl.backend_nbagg")


def test_low_mpl_version_raises(monkeypatch):
    import uspy.interactive.picker as picker_mod
    monkeypatch.setattr(picker_mod.mpl, "__version__", "3.4.0")
    with pytest.raises(RuntimeError, match="matplotlib"):
        _check_mpl_version()


def test_picker_constructs_with_no_regions(picker):
    assert picker.specs == []
    assert picker.code == "# nothing picked yet"


def test_pick_rois_returns_and_displays_picker(stack):
    p = pick_rois(stack, xaxis="energy")
    assert isinstance(p, ROIPicker)
    p.canvas.close()
    plt.close(p.fig)


def test_pick_rois_displays_the_widget_exactly_once(stack, monkeypatch):
    # Regression coverage: pick_rois() used to display its widget explicitly *and*
    # ROIPicker defined _ipython_display_, so an unassigned `pick_rois(...)` cell (the
    # common case) rendered the same widget twice via IPython's own echo. Assigning
    # (`p = pick_rois(...)`) happened to suppress the echo, masking the bug -- so this
    # test drives pick_rois() directly and counts calls to the module's own `display`,
    # rather than relying on IPython's cell-echo machinery (not present under pytest).
    import uspy.interactive.picker as picker_mod
    calls = []
    monkeypatch.setattr(picker_mod, "display", lambda obj: calls.append(obj))
    p = pick_rois(stack, xaxis="energy")
    widget_calls = [c for c in calls if c is p.widget]
    assert len(widget_calls) == 1
    p.canvas.close()
    plt.close(p.fig)


def test_roipicker_has_no_ipython_display_hook(stack):
    # the other half of the same fix: with _ipython_display_ still defined, IPython
    # would call it on every unassigned cell result regardless of what pick_rois()
    # itself already displayed -- so it must simply not exist.
    assert not hasattr(ROIPicker, "_ipython_display_")


def test_repr_reports_frames_regions_and_current_frame(picker):
    picker.canvas.new_shape = "circle"
    _drag(picker.ax_img, (15, 15), (25, 25))
    text = repr(picker)
    assert str(len(picker.stack)) in text
    assert "1" in text  # one region
    assert str(picker._frame_idx) in text


### seeding from existing ROIs ############################################


def test_seeding_from_existing_rois(stack):
    r1 = ROI.circle(30, 20, radius=8, color="red", label="Cu")
    r2 = ROI(50, 30, source="rectangle", width=10, height=6, color="blue")
    p = ROIPicker(stack, rois=[r1, r2])
    assert len(p.specs) == 2
    assert "Cu" in p.code
    assert_array_equal(p.rois[0].array, r1.array)
    p.canvas.close()
    plt.close(p.fig)


def test_seeding_with_a_profile_raises(stack):
    # the picker's whole right-hand panel is an I(V) plot, which a Profile cannot
    # produce (that's a Waterfall, a different tool) -- so it must refuse outright
    # rather than silently drop the region (CLAUDE.md: never silently change what's
    # shown to the user).
    profile = Profile([(5, 5), (20, 20)], width=2)
    with pytest.raises(ValueError, match="waterfall"):
        ROIPicker(stack, rois=[profile])


### live I(V) ##############################################################


def test_live_matches_intensityline(picker):
    spec = Spec("circle", (("x0", 30), ("y0", 25), ("radius", 10)), "red")
    live_y = picker._live_intensity(spec)
    canonical = derived.IntensityLine(picker.stack, spec.to_object(), "energy")
    np.testing.assert_allclose(live_y, canonical.y, rtol=1e-6)


def test_live_defaults_on_for_resident_stack(stack):
    p = ROIPicker(stack)
    assert p.live is True
    p.canvas.close()
    plt.close(p.fig)


def test_live_off_does_not_recompute_on_drag(picker):
    picker.live = False
    calls = []
    picker._live_intensity = lambda spec: (calls.append(1), np.zeros(N))[1]
    spec = Spec("circle", (("x0", 30), ("y0", 25), ("radius", 10)), "red")
    picker._on_drag(0, spec)
    assert calls == []


def test_live_motion_events_are_coalesced(picker):
    picker.live = True
    calls = []
    picker._live_intensity = lambda spec: (calls.append(1), np.zeros(N))[1]
    fake_time = [0.0]
    picker._clock = lambda: fake_time[0]
    spec = Spec("circle", (("x0", 30), ("y0", 25), ("radius", 10)), "red")
    picker.canvas.specs = [spec]  # _update_iv_preview's drag branch needs it present

    for _ in range(20):
        fake_time[0] += 0.001  # cumulative 0.02s, still under the 0.05s throttle
        picker._on_drag(0, spec)
    assert calls == []  # every one of the 20 rapid events was coalesced away

    fake_time[0] += 1.0  # advance well past the throttle interval
    picker._on_drag(0, spec)
    assert len(calls) == 1  # but it does eventually fire again


### virtual-stack handling #################################################


def test_live_defaults_off_for_virtual_stack():
    from uspy.leem.base import LEEMStack
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vstack = LEEMStack("testdata/003_IV", virtual=True)
        p = ROIPicker(vstack, xaxis="energy")
    assert p.live is False
    p.canvas.close()
    plt.close(p.fig)


def test_live_on_virtual_stack_warns():
    from uspy.leem.base import LEEMStack
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vstack = LEEMStack("testdata/003_IV", virtual=True)
    with pytest.warns(UserWarning, match="materialize"):
        p = ROIPicker(vstack, xaxis="energy", live=True)
    p.canvas.close()
    plt.close(p.fig)


def test_preview_never_materializes_a_virtual_stack():
    from uspy.leem.base import LEEMStack
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vstack = LEEMStack("testdata/003_IV", virtual=True)
        p = ROIPicker(vstack, xaxis="energy", live=False)
    p.canvas.specs = [Spec("circle", (("x0", 300), ("y0", 300), ("radius", 20)), "red")]
    p.canvas.redraw_overlays()
    with warnings.catch_warnings():
        # the *only* UserWarning that should ever be possible here is the "materializes
        # every element" one from bulk `stack.image` access, which this must NOT trigger
        warnings.simplefilter("error", UserWarning)
        p._update_iv_preview()
    p.canvas.close()
    plt.close(p.fig)


### preview caching #########################################################


def test_preview_cached_per_spec(picker, monkeypatch):
    calls = []
    real_intensity_line = derived.IntensityLine

    def counting(*args, **kwargs):
        calls.append(1)
        return real_intensity_line(*args, **kwargs)

    monkeypatch.setattr(derived, "IntensityLine", counting)
    spec = Spec("circle", (("x0", 30), ("y0", 25), ("radius", 10)), "red")
    picker.canvas.specs = [spec]
    picker.canvas.redraw_overlays()

    picker._update_iv_preview()
    picker._update_iv_preview()  # same spec, same step -- cache hit
    assert len(calls) == 1

    picker.canvas.specs = []
    picker.canvas.specs = [spec]  # delete + re-add an *identical* spec
    picker.canvas.redraw_overlays()
    picker._update_iv_preview()
    assert len(calls) == 1  # still a cache hit -- Spec is frozen/hashable/equal


### drag -> commit replaces preview with the canonical line #################


def test_release_replaces_preview_with_intensityline(picker):
    picker.live = True
    picker.canvas.new_shape = "circle"
    _drag(picker.ax_img, (20, 20), (30, 30))
    assert len(picker.canvas.specs) == 1

    picker.canvas.select(0)
    from uspy.interactive.spec import handle_position
    spec = picker.canvas.specs[0]
    if spec.shape == "circle":
        corner = handle_position(spec, 5)  # arbitrary nearby point to start a body move
    _press(picker.ax_img, tuple(dict(spec.params)[k] for k in ("x0", "y0")))
    _move(picker.ax_img, (35, 35))
    # mid-drag: the live preview line should be dashed
    dashed = [l for l in picker.ax_iv.lines if l.get_linestyle() == "--"]
    assert len(dashed) >= 1
    _release(picker.ax_img, (35, 35))
    # after release: no dashed preview line remains, replaced by the cached IntensityLine
    dashed_after = [l for l in picker.ax_iv.lines if l.get_linestyle() == "--"]
    assert dashed_after == []


### arrow-key nudging reaches the picker (code box + I(V) preview) ##########


def test_shift_arrow_nudge_updates_code_and_preview(picker):
    picker.canvas.new_shape = "circle"
    _drag(picker.ax_img, (20, 20), (28, 28))
    picker.canvas.select(0)
    code_before = picker.code
    x0_before = dict(picker.canvas.specs[0].params)["x0"]

    _key(picker.ax_img, "shift+right")

    assert dict(picker.canvas.specs[0].params)["x0"] == x0_before + 1
    assert picker.code != code_before
    assert picker.w_code.value == picker.code


### frame/contrast redraw hygiene ###########################################


def test_frame_slider_updates_live_not_just_on_release(picker):
    # Regression coverage: the slider used to have continuous_update=False, so the
    # displayed frame only updated once the mouse was released, making scrubbing
    # through the stack feel unresponsive. A single-frame read + plot_img is cheap
    # (unlike a canvas drag's per-pixel motion events), so it should update live.
    assert picker.w_frame.continuous_update is True


def test_frame_change_does_not_leak_base_artists(picker):
    n_before = len(picker.ax_img.get_children())
    for i in range(min(3, len(picker.stack))):
        picker.w_frame.value = i
    n_after = len(picker.ax_img.get_children())
    assert n_after == n_before


def test_contrast_change_redraws_without_leaking(picker):
    n_before = len(picker.ax_img.get_children())
    picker.w_contrast.value = "inner"
    picker.w_contrast.value = "auto"
    assert len(picker.ax_img.get_children()) == n_before


### widget <-> canvas wiring #################################################


def test_rotation_widget_and_handle_stay_in_sync(picker):
    picker.canvas.specs = [Spec("rectangle", (("x0", 30), ("y0", 25), ("width", 20), ("height", 10)),
                               "red")]
    picker.canvas.redraw_overlays()
    picker._refresh_list()

    picker.w_list.value = 0  # select via the list -> rotation widget should reflect it
    assert picker.w_rotation.value == 0

    picker.w_rotation.value = 45  # edit via the widget -> canvas spec should update
    assert dict(picker.canvas.specs[0].params)["rotation"] == 45


def test_shape_color_label_widgets_set_new_region_defaults(picker):
    picker.w_shape.value = "rectangle"
    picker.w_color.value = "#ff0000"
    picker.w_label.value = "Cu"
    assert picker.canvas.new_shape == "rectangle"
    assert picker.canvas.new_color == "#ff0000"
    assert picker.canvas.new_label == "Cu"


### profiles are not offered by the I(V) picker ###############################
# A profile over a stack yields a Waterfall, which the I(V) preview panel here
# cannot plot -- see uspy.plotting.waterfall for that separate tool instead.


def test_shape_dropdown_offers_no_profile_option(picker):
    assert "profile" not in picker.w_shape.options
    assert tuple(picker.w_shape.options) == ROI_SHAPES


### Fiji-style shape button row #################################################
# Fiji shows a row of small square buttons, one per shape, each glyph depicting the
# shape itself, rather than a dropdown -- see uspy.interactive.picker's ToggleButtons
# + injected CSS. The widget stays a single ToggleButtons (not one Button per shape)
# so .options/.value and the one-line wiring to canvas.new_shape are unchanged.


def test_shape_widget_is_a_toggle_buttons_row_with_no_profile_option(picker):
    import ipywidgets as w
    assert isinstance(picker.w_shape, w.ToggleButtons)
    assert tuple(picker.w_shape.options) == ROI_SHAPES
    assert "profile" not in picker.w_shape.options


def test_shape_widget_selection_still_drives_canvas_new_shape(picker):
    picker.w_shape.value = "rectangle"
    assert picker.canvas.new_shape == "rectangle"


def test_shape_widget_has_a_tooltip_per_shape(picker):
    assert tuple(picker.w_shape.tooltips) == ROI_SHAPES


def test_shape_button_css_has_an_icon_rule_per_shape(picker):
    # each shape's tooltip is the CSS hook the injected stylesheet targets (a
    # button[title="<shape>"] selector) -- if the icons ever silently stopped
    # rendering, this is the assertion that would catch a broken/missing selector.
    for shape in ROI_SHAPES:
        assert f'"{shape}"' in _SHAPE_BUTTON_CSS


def test_shape_css_widget_is_present_in_the_widget_tree(picker):
    assert picker._shape_css in picker.widget.children


def test_picker_has_no_profile_width_widget(picker):
    assert not hasattr(picker, "w_profile_width")


def test_picker_has_no_profiles_property(picker):
    assert not hasattr(picker, "profiles")


def test_canvas_shapes_are_roi_only(picker):
    assert picker.canvas.shapes == ROI_SHAPES
    with pytest.raises(ValueError):
        picker.canvas.new_shape = "profile"


### contextual label/color/rotation editing of an existing region ############


def test_selecting_via_list_loads_label_and_color_into_widgets(picker):
    picker.canvas.specs = [Spec("circle", (("x0", 20), ("y0", 20), ("radius", 5)), "red", "Cu")]
    picker.canvas.redraw_overlays()
    picker._refresh_list()

    picker.w_list.value = 0
    assert picker.w_label.value == "Cu"
    assert picker.w_color.value == "#ff0000"  # normalized to hex, see to_hex() usage


def test_selecting_via_canvas_click_loads_widgets_too(picker):
    picker.canvas.specs = [Spec("circle", (("x0", 20), ("y0", 20), ("radius", 5)), "red", "Cu")]
    picker.canvas.redraw_overlays()
    _press(picker.ax_img, (20, 20))  # body hit -> selects region 0
    _release(picker.ax_img, (20, 20))
    assert picker.canvas.selected == 0
    assert picker.w_label.value == "Cu"


def test_editing_label_widget_renames_the_selected_region(picker):
    picker.canvas.specs = [Spec("circle", (("x0", 20), ("y0", 20), ("radius", 5)), "red")]
    picker.canvas.redraw_overlays()
    picker.canvas.select(0)
    picker.w_label.value = "Cu"
    assert picker.canvas.specs[0].label == "Cu"
    assert "Cu" in picker.code


def test_editing_color_widget_recolors_the_selected_region(picker):
    picker.canvas.specs = [Spec("circle", (("x0", 20), ("y0", 20), ("radius", 5)), "red")]
    picker.canvas.redraw_overlays()
    picker.canvas.select(0)
    picker.w_color.value = "#00ff00"
    assert picker.canvas.specs[0].color == "#00ff00"


def test_deselecting_resets_controls_to_next_region_defaults(picker):
    picker.canvas.specs = [Spec("circle", (("x0", 20), ("y0", 20), ("radius", 5)), "red", "Cu")]
    picker.canvas.redraw_overlays()
    picker.canvas.select(0)
    assert picker.w_label.value == "Cu"

    picker.canvas.select(None)
    assert picker.w_label.value == ""
    assert picker.canvas.new_label is None


def test_explicit_next_color_does_not_leak_past_deselect(picker):
    picker.w_color.value = "#123456"  # nothing selected -> sets the next-region default
    assert picker.canvas.new_color == "#123456"
    picker.canvas.new_shape = "circle"
    _drag(picker.ax_img, (15, 15), (25, 25))  # uses the explicit color; region is auto-selected
    assert picker.canvas.specs[0].color == "#123456"

    picker.canvas.select(None)  # deselect -> resets so the auto-cycle resumes
    assert picker.canvas.new_color is None


def test_list_selection_survives_a_label_edit(picker):
    picker.canvas.specs = [
        Spec("circle", (("x0", 20), ("y0", 20), ("radius", 5)), "red"),
        Spec("circle", (("x0", 50), ("y0", 20), ("radius", 5)), "blue"),
    ]
    picker.canvas.redraw_overlays()
    picker._refresh_list()
    picker.w_list.value = 1
    picker.w_label.value = "Au"
    assert picker.w_list.value == 1  # still pointing at the region we just edited


### auto-cycling colors in the picker #########################################


def test_successive_drawn_regions_cycle_the_picker_palette(picker):
    picker.canvas.new_shape = "circle"
    _drag(picker.ax_img, (10, 10), (20, 20))
    _drag(picker.ax_img, (30, 30), (40, 40))
    assert [s.color for s in picker.canvas.specs] == [palette_color(0), palette_color(1)]


### color widget must accept every matplotlib color, not just HTML ones #######
# Regression coverage for a reported crash: TraitError from ipywidgets.ColorPicker
# when handed a color it -- unlike matplotlib -- doesn't understand (a float-RGB
# tuple, or names like "b"/"C0"/"tab:blue"). Two independent paths hit this: an
# auto-cycled color drawn from a non-hex axes.prop_cycle, and loading an existing
# region's spec.color (e.g. a notebook-authored ROI(..., color="b") -- the literal
# example in ROI.__repr__'s own docstring) into the widget on selection. Every test
# above this point ran under matplotlib's stock hex palette and "red"/"blue"/"#..."
# literals, so this whole failure mode went untested.


def _assert_html_color(value) -> None:
    """`value` must be something ipywidgets.ColorPicker itself would accept as a
    color -- not just "parseable by matplotlib" (a float-RGB tuple passes
    matplotlib's own to_rgba() fine but is exactly what crashed the widget)."""
    assert isinstance(value, str)
    to_rgba(value)  # raises if not even matplotlib-parseable


def _patch_prop_cycle(monkeypatch, colors):
    """Monkeypatch the actual rc source `uspy.roi.palette_color` reads
    (mpl.rc_params()), not palette_color itself, so this exercises the real path."""
    real_rc_params = roi_mod.mpl.rc_params

    def fake_rc_params(*a, **kw):
        rc = real_rc_params(*a, **kw)
        rc["axes.prop_cycle"] = roi_mod.mpl.cycler(color=colors)
        return rc

    monkeypatch.setattr(roi_mod.mpl, "rc_params", fake_rc_params)


def test_construction_survives_a_tuple_valued_prop_cycle(stack, monkeypatch):
    # regression pin 1: this raised TraitError on construction before the fix
    _patch_prop_cycle(monkeypatch, [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                                    (1.0, 0.4980392156862745, 0.054901960784313725)])
    p = ROIPicker(stack, xaxis="energy")
    _assert_html_color(p.w_color.value)
    p.canvas.close()
    plt.close(p.fig)


def test_selecting_a_non_html_seeded_region_does_not_raise(stack):
    # regression pin 2: this raised TraitError the moment the region was selected,
    # before the fix
    r = ROI.circle(20, 20, radius=6, color="b", label="Cu")
    p = ROIPicker(stack, xaxis="energy", rois=[r])
    p.canvas.select(0)
    assert p.w_color.value == "#0000ff"
    assert "color='b'" in p.code  # the authored literal is preserved, not rewritten
    p.canvas.close()
    plt.close(p.fig)


@pytest.mark.parametrize("color", ["b", "C0", "tab:blue", "red", "#1f77b4",
                                   (0.1, 0.2, 0.3)])
def test_selecting_any_matplotlib_color_loads_a_matching_html_color(picker, color):
    picker.canvas.specs = [Spec("circle", (("x0", 20), ("y0", 20), ("radius", 5)), color)]
    picker.canvas.redraw_overlays()
    picker.canvas.select(0)
    _assert_html_color(picker.w_color.value)
    # to_hex() quantizes each channel to 1/255 (an 8-bit hex digit), so an arbitrary
    # float tuple round-trips only approximately -- exact for every color actually
    # reachable via hex/named strings, which is every case here except the raw tuple
    assert to_rgba(picker.w_color.value) == pytest.approx(to_rgba(color), abs=1 / 255)


def test_color_widget_stays_valid_through_a_realistic_session(picker):
    """State-machine sweep -- the kind of test that would have caught this class of
    bug generically instead of one path at a time: after every step of a plausible
    session, the color widget must hold a value ipywidgets itself accepts."""
    _assert_html_color(picker.w_color.value)  # initial state, nothing drawn yet

    picker.canvas.specs = [Spec("circle", (("x0", 20), ("y0", 20), ("radius", 5)), "b"),
                           Spec("circle", (("x0", 50), ("y0", 20), ("radius", 5)),
                                (0.1, 0.2, 0.3))]
    picker.canvas.redraw_overlays()
    picker._refresh_list()

    picker.canvas.select(0)
    _assert_html_color(picker.w_color.value)

    picker.canvas.select(1)
    _assert_html_color(picker.w_color.value)

    picker.canvas.select(None)
    _assert_html_color(picker.w_color.value)

    picker.canvas.select(0)
    picker.w_color.value = "#123456"
    _assert_html_color(picker.w_color.value)
    assert picker.canvas.specs[0].color == "#123456"

    picker.canvas.delete_selected()
    _assert_html_color(picker.w_color.value)


### editing label/color must not re-read the stack ############################


def test_recoloring_a_region_reuses_the_iv_cache_entry(picker):
    spec = Spec("circle", (("x0", 30), ("y0", 25), ("radius", 10)), "red")
    picker.canvas.specs = [spec]
    picker.canvas.redraw_overlays()
    picker._update_iv_preview()
    assert len(picker._cache) == 1

    picker.canvas.select(0)
    picker.w_color.value = "#00ff00"
    picker.w_label.value = "Pt"
    assert len(picker._cache) == 1  # geometry unchanged -> same cache key, no re-read
    line = next(iter(picker._cache.values()))
    assert line.color == "#00ff00"
    assert line.label == "Pt"


### frame slider layout #########################################################


def test_frame_slider_has_its_own_full_width_row(picker):
    assert picker.w_frame.layout.width in ("99%", "100%")
    rows = picker.widget.children
    frame_row = next(row for row in rows if picker.w_frame in getattr(row, "children", ()))
    # it must not share its row with the other per-frame controls anymore -- that's
    # what squeezed it down to a few millimetres of usable range
    assert picker.w_contrast not in frame_row.children


### compact control sizing ######################################################
# Regression coverage for the layout tightening: captioned controls used to fall
# back to ipywidgets' default ~300px control box + ~80px description column,
# which is why the concise color swatch and rotation spinner looked oversized and
# the shape row wrapped polygon onto a second line.


def test_shape_row_does_not_wrap(picker):
    assert "nowrap" in _SHAPE_BUTTON_CSS
    assert picker.w_shape.layout.width is not None


def _px(width):
    assert width is not None and width.endswith("px")
    return float(width[:-2])


@pytest.mark.parametrize("name", ["w_label", "w_color", "w_rotation", "w_contrast", "w_step"])
def test_captioned_controls_are_sized_to_content(picker, name):
    widget = getattr(picker, name)
    assert _px(widget.layout.width) <= 220
    assert widget.style.description_width not in (None, "")


def test_checkboxes_are_not_indented(picker):
    assert picker.w_live.indent is False
    assert picker.w_auto.indent is False


def test_region_controls_and_actions_share_one_row(picker):
    rows = picker.widget.children
    tool_row = next(row for row in rows if picker.w_shape in getattr(row, "children", ()))
    assert picker.w_label in tool_row.children
    assert picker.w_color in tool_row.children
    assert picker.w_rotation in tool_row.children
    assert picker.w_delete in tool_row.children
    assert picker.w_clear in tool_row.children
    assert picker.w_contrast not in tool_row.children


def test_control_rows_wrap_instead_of_overflow(picker):
    rows = picker.widget.children
    tool_row = next(row for row in rows if picker.w_shape in getattr(row, "children", ()))
    preview_row = next(row for row in rows if picker.w_contrast in getattr(row, "children", ()))
    assert "wrap" in tool_row.layout.flex_flow
    assert "wrap" in preview_row.layout.flex_flow


def test_delete_and_clear_buttons(picker):
    picker.canvas.specs = [
        Spec("circle", (("x0", 20), ("y0", 20), ("radius", 5)), "red"),
        Spec("circle", (("x0", 50), ("y0", 20), ("radius", 5)), "blue"),
    ]
    picker.canvas.redraw_overlays()
    picker.canvas.select(0)
    picker.w_delete.click()
    assert len(picker.canvas.specs) == 1

    picker.w_clear.click()
    assert picker.canvas.specs == []


def test_code_and_list_stay_in_sync_after_drag(picker):
    picker.canvas.new_shape = "circle"
    _drag(picker.ax_img, (15, 15), (25, 25))
    assert len(picker.w_list.options) == 1
    assert "ROI.circle" in picker.code
    assert picker.w_code.value == picker.code


def test_copy_button_does_not_raise(picker):
    picker.canvas.specs = [Spec("circle", (("x0", 20), ("y0", 20), ("radius", 5)), "red")]
    picker.canvas.redraw_overlays()
    picker._refresh_code()
    picker.w_copy.click()
    assert "copied" in picker.w_status.value or "Ctrl+C" in picker.w_status.value


### does not mutate the stack ################################################


def test_picker_does_not_mutate_the_stack(stack):
    import copy
    original = [img.image.copy() for img in stack]
    p = ROIPicker(stack, xaxis="energy")
    p.canvas.new_shape = "circle"
    _drag(p.ax_img, (15, 15), (25, 25))
    p.w_frame.value = min(2, len(stack) - 1)
    p.canvas.select(0)
    p.w_rotation.value = 10  # a no-op for a circle, but exercises the path
    p.canvas.delete_selected()
    for img, before in zip(stack, original):
        assert_array_equal(img.image, before)
    p.canvas.close()
    plt.close(p.fig)
