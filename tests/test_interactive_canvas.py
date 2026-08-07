"""Tests for uspy.interactive.canvas.Canvas -- the draggable-artist / drag-state-
machine layer of the interactive picker. Headless under Agg, driven by synthetic
matplotlib MouseEvent/KeyEvent objects dispatched through the real event pipeline
(`fig.canvas.callbacks.process(...)`), not by calling the handler methods directly --
this exercises the actual matplotlib event API, not just this module's model of it.

No ipywidgets needed: Canvas is pure matplotlib + uspy.interactive.spec.
"""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=protected-access

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent, KeyEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from uspy.interactive.canvas import Canvas
from uspy.interactive.spec import ROI_SHAPES, Spec, vertex_positions
from uspy.roi import palette_color

H, W = 200, 200


def _force_agg(fig) -> None:
    """Attach a real `FigureCanvasAgg` to `fig`, independent of whatever backend
    `matplotlib.get_backend()` currently reports. `matplotlib.use("Agg")` above only
    sets the *default* for figures created from here on -- it does nothing once
    something has switched the process-global backend afterward, which is exactly
    what happens if tests/test_interactive_picker.py is collected in the same pytest
    session: `import ipympl` there has a documented side effect (see ipympl/
    __init__.py) of calling `matplotlib.use('module://ipympl.backend_nbagg')` if
    `matplotlib` is already imported -- true here, since conftest.py imports it first.
    Without this, `fig.canvas` could silently be ipympl's traitlets-typed `Canvas`
    instead of a plain Agg one, which rejects a test double assigned to `.toolbar`."""
    FigureCanvasAgg(fig)


@pytest.fixture
def ax():
    fig, axes = plt.subplots()
    _force_agg(fig)
    axes.set_xlim(-0.5, W - 0.5)
    axes.set_ylim(H - 0.5, -0.5)  # inverted, matches imshow's origin-upper convention
    yield axes
    plt.close(fig)


@pytest.fixture
def canvas(ax):
    c = Canvas(ax, height=H, width=W)
    yield c
    c.close()


def _press(axes, xy, dblclick=False, button=1, key=None):
    x, y = axes.transData.transform(xy)
    ev = MouseEvent("button_press_event", axes.figure.canvas, x, y, button=button,
                    dblclick=dblclick, key=key)
    axes.figure.canvas.callbacks.process("button_press_event", ev)


def _move(axes, xy, key=None):
    x, y = axes.transData.transform(xy)
    ev = MouseEvent("motion_notify_event", axes.figure.canvas, x, y, key=key)
    axes.figure.canvas.callbacks.process("motion_notify_event", ev)


def _release(axes, xy, button=1, key=None):
    x, y = axes.transData.transform(xy)
    ev = MouseEvent("button_release_event", axes.figure.canvas, x, y, button=button, key=key)
    axes.figure.canvas.callbacks.process("button_release_event", ev)


def _key(axes, key):
    ev = KeyEvent("key_press_event", axes.figure.canvas, key)
    axes.figure.canvas.callbacks.process("key_press_event", ev)


def _drag(axes, start, end):
    _press(axes, start)
    _move(axes, end)
    _release(axes, end)


### creating new regions ##################################################


def test_drag_creates_circle_region():
    # centred (default, no Ctrl): the press point is the centre, radius = distance
    # dragged -- hypot(40, 40) = 56.57, rounded to an even diameter of 114 -> radius 57.
    fig, axes = plt.subplots()
    _force_agg(fig)
    axes.set_xlim(-0.5, W - 0.5)
    axes.set_ylim(H - 0.5, -0.5)
    c = Canvas(axes, H, W)
    c.new_shape = "circle"
    c.new_color = "red"
    _drag(axes, (60, 60), (100, 100))
    assert len(c.specs) == 1
    assert c.specs[0].shape == "circle"
    geom = dict(c.specs[0].params)
    assert geom["x0"] == 60 and geom["y0"] == 60
    assert geom["radius"] == 57
    plt.close(fig)


def test_drag_creates_rectangle_region(ax, canvas):
    # centred (default, no Ctrl): the press point is the centre, so width/height are
    # double the dragged offset (60, 20) -> (120, 40), not the offset itself.
    canvas.new_shape = "rectangle"
    _drag(ax, (50, 50), (110, 70))
    assert len(canvas.specs) == 1
    geom = dict(canvas.specs[0].params)
    assert geom["x0"] == 50 and geom["y0"] == 50
    assert geom["width"] == 120 and geom["height"] == 40


def test_ctrl_drag_creates_corner_to_corner_circle(ax, canvas):
    # Ctrl reproduces the old bounding-box gesture: centre = midpoint, diameter = the
    # larger dragged extent (here a 40x40 square, forced even -> unchanged).
    canvas.new_shape = "circle"
    _press(ax, (60, 60))
    _move(ax, (100, 100), key="control")
    _release(ax, (100, 100))
    geom = dict(canvas.specs[0].params)
    assert geom["x0"] == 80 and geom["y0"] == 80
    assert geom["radius"] == 20


def test_ctrl_drag_creates_corner_to_corner_rectangle(ax, canvas):
    canvas.new_shape = "rectangle"
    _press(ax, (50, 50))
    _move(ax, (110, 70), key="control")
    _release(ax, (110, 70))
    geom = dict(canvas.specs[0].params)
    assert geom["x0"] == 80 and geom["y0"] == 60  # midpoint
    assert geom["width"] == 60 and geom["height"] == 20


def test_shift_drag_forces_equal_extents(ax, canvas):
    # centred + Shift: width == height, using the larger dragged offset (60 > 20).
    canvas.new_shape = "rectangle"
    _press(ax, (50, 50))
    _move(ax, (110, 70), key="shift")
    _release(ax, (110, 70), key="shift")
    geom = dict(canvas.specs[0].params)
    assert geom["width"] == 120 and geom["height"] == 120


def test_modifier_change_mid_drag_updates_the_draft_live(ax, canvas):
    canvas.new_shape = "rectangle"
    _press(ax, (50, 50))
    _move(ax, (110, 70))  # no modifier yet -> centred
    assert dict(canvas._drag["draft"].params)["width"] == 120
    _move(ax, (110, 70), key="control")  # Ctrl held now -> switches to corner-to-corner
    assert dict(canvas._drag["draft"].params)["width"] == 60
    _release(ax, (110, 70), key="control")


def test_click_creates_point_region(ax, canvas):
    canvas.new_shape = "point"
    _press(ax, (77, 88))
    _release(ax, (77, 88))
    assert len(canvas.specs) == 1
    assert canvas.specs[0].shape == "point"
    assert dict(canvas.specs[0].params) == {"x0": 77, "y0": 88}


def test_click_chain_creates_polygon(ax, canvas):
    canvas.new_shape = "polygon"
    _press(ax, (10, 10))
    _press(ax, (60, 10))
    _press(ax, (60, 60), dblclick=True)  # finishes with 3 points
    assert len(canvas.specs) == 1
    assert canvas.specs[0].shape == "polygon"
    assert vertex_positions(canvas.specs[0]) == ((10, 10), (60, 10), (60, 60))


def test_click_chain_creates_profile(ax, canvas):
    canvas.new_shape = "profile"
    canvas.new_width = 5
    _press(ax, (10, 10))
    _press(ax, (100, 100), dblclick=True)
    assert len(canvas.specs) == 1
    assert canvas.specs[0].shape == "profile"
    assert dict(canvas.specs[0].params)["width"] == 5


def test_offframe_region_rejected():
    # axes view wider than the underlying (H, W) frame -- e.g. after zooming out --
    # so a drag can be within the visible plot (inaxes=True) but outside the frame.
    fig, axes = plt.subplots()
    _force_agg(fig)
    axes.set_xlim(-80, W + 80)
    axes.set_ylim(H + 80, -80)
    fig.canvas.draw()
    c = Canvas(axes, H, W)
    c.new_shape = "circle"
    _drag(axes, (-60, -60), (-40, -40))
    assert c.specs == []
    assert "outside" in c.status
    c.close()
    plt.close(fig)


### shape allow-list (Canvas(..., shapes=...), for the I(V) picker's ROI-only use) ###


def test_default_canvas_accepts_every_shape(ax, canvas):
    canvas.new_shape = "profile"  # no raise -- the bare Canvas() is shape-generic
    assert canvas.new_shape == "profile"


def test_restricted_canvas_rejects_shape_outside_allow_list():
    fig, axes = plt.subplots()
    _force_agg(fig)
    axes.set_xlim(-0.5, W - 0.5)
    axes.set_ylim(H - 0.5, -0.5)
    c = Canvas(axes, H, W, shapes=ROI_SHAPES)
    assert "profile" not in c.shapes
    with pytest.raises(ValueError):
        c.new_shape = "profile"
    c.close()
    plt.close(fig)


def test_restricted_canvas_still_accepts_allowed_shapes():
    fig, axes = plt.subplots()
    _force_agg(fig)
    axes.set_xlim(-0.5, W - 0.5)
    axes.set_ylim(H - 0.5, -0.5)
    c = Canvas(axes, H, W, shapes=ROI_SHAPES)
    c.new_shape = "rectangle"
    assert c.new_shape == "rectangle"
    c.close()
    plt.close(fig)


### toolbar / navigation mode #################################################
# Regression coverage: reported bug -- using matplotlib's zoom/pan toolbar tool also
# created (or moved) a region, because no handler checked whether the toolbar owned
# the mouse. `_FakeToolbar` stands in for the real ipympl Toolbar (this module's tests
# run under Agg, which has no toolbar at all -- see the "idle" case below, which is
# what every other test in this file implicitly exercises); it needs the internal
# `_wait_cursor_for_draw_cm` hook because Agg's own `draw()` calls it whenever
# `canvas.toolbar` is set, real toolbar or not.


class _FakeToolbar:
    def __init__(self, mode):
        self.mode = mode

    def _wait_cursor_for_draw_cm(self):
        from contextlib import nullcontext
        return nullcontext()


def test_drag_creates_nothing_while_toolbar_zoom_is_active(ax, canvas):
    ax.figure.canvas.toolbar = _FakeToolbar("zoom rect")
    canvas.new_shape = "circle"
    _drag(ax, (60, 60), (100, 100))
    assert canvas.specs == []


def test_drag_does_not_move_a_region_while_toolbar_pan_is_active(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    ax.figure.canvas.toolbar = _FakeToolbar("pan/zoom")
    _drag(ax, (50, 50), (70, 70))
    assert dict(canvas.specs[0].params) == {"x0": 50, "y0": 50, "radius": 10}


def test_drag_creates_nothing_while_widgetlock_is_held(ax, canvas):
    canvas.new_shape = "circle"
    ax.figure.canvas.widgetlock(object())  # simulate some other tool owning the mouse
    _drag(ax, (60, 60), (100, 100))
    assert canvas.specs == []


def test_drag_works_normally_with_an_idle_toolbar(ax, canvas):
    ax.figure.canvas.toolbar = _FakeToolbar("")
    canvas.new_shape = "circle"
    _drag(ax, (60, 60), (100, 100))
    assert len(canvas.specs) == 1


def test_new_region_uses_new_color_and_label(ax, canvas):
    canvas.new_shape = "circle"
    canvas.new_color = "purple"
    canvas.new_label = "Cu"
    _drag(ax, (60, 60), (100, 100))
    assert canvas.specs[0].color == "purple"
    assert canvas.specs[0].label == "Cu"


### auto-cycling colors (new_color defaults to None -> auto_color()) ########


def test_new_color_defaults_to_none(ax, canvas):
    assert canvas.new_color is None


def test_successive_regions_cycle_the_palette(ax, canvas):
    canvas.new_shape = "circle"
    _drag(ax, (20, 20), (40, 40))
    _drag(ax, (60, 60), (80, 80))
    _drag(ax, (100, 100), (120, 120))
    assert [s.color for s in canvas.specs] == [palette_color(0), palette_color(1), palette_color(2)]


def test_deleting_a_region_frees_its_color(ax, canvas):
    canvas.new_shape = "circle"
    _drag(ax, (20, 20), (40, 40))   # C0
    _drag(ax, (60, 60), (80, 80))   # C1
    canvas.select(0)
    canvas.delete_selected()
    _drag(ax, (100, 100), (120, 120))
    assert canvas.specs[-1].color == palette_color(0)  # C0 reused, not C2


def test_explicit_new_color_overrides_auto_cycle(ax, canvas):
    canvas.new_shape = "circle"
    canvas.new_color = "purple"
    _drag(ax, (20, 20), (40, 40))
    assert canvas.specs[0].color == "purple"


def test_auto_color_reports_next_without_side_effects(ax, canvas):
    canvas.new_shape = "circle"
    assert canvas.auto_color() == palette_color(0)
    assert canvas.auto_color() == palette_color(0)  # calling it again is a no-op
    _drag(ax, (20, 20), (40, 40))
    assert canvas.auto_color() == palette_color(1)


### auto_color() must always return an HTML color, whatever the rc palette holds ####
# Regression coverage: reported bug -- a user's axes.prop_cycle held float-RGB tuples
# (a perfectly valid matplotlib color, just not an HTML one), and auto_color() handed
# that tuple straight to an ipywidgets.ColorPicker, which only accepts HTML colors
# (#rrggbb or CSS names) and raised TraitError. Every existing test above ran under
# matplotlib's stock hex palette, so this whole failure mode was untested.


def _patch_prop_cycle(monkeypatch, colors):
    """Monkeypatch the actual rc source `uspy.roi.palette_color` reads (mpl.rc_params()),
    not palette_color itself -- so these tests exercise the real code path, the same
    one that broke."""
    import uspy.roi as roi_mod
    real_rc_params = roi_mod.mpl.rc_params

    def fake_rc_params(*a, **kw):
        rc = real_rc_params(*a, **kw)
        rc["axes.prop_cycle"] = roi_mod.mpl.cycler(color=colors)
        return rc

    monkeypatch.setattr(roi_mod.mpl, "rc_params", fake_rc_params)


def test_auto_color_is_html_under_tuple_valued_palette(ax, canvas, monkeypatch):
    _patch_prop_cycle(monkeypatch, [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                                    (1.0, 0.4980392156862745, 0.054901960784313725)])
    import matplotlib.colors as mcolors
    color = canvas.auto_color()
    assert isinstance(color, str) and color.startswith("#")
    mcolors.to_rgba(color)  # does not raise -- a genuinely valid color


def test_auto_color_is_html_under_named_color_palette(ax, canvas, monkeypatch):
    _patch_prop_cycle(monkeypatch, ["b", "g", "tab:red"])
    assert canvas.auto_color() == "#0000ff"


def test_auto_color_dedupes_across_color_representations(ax, canvas):
    # a region seeded with the tuple form of C0 must still be recognized as "using"
    # C0 -- otherwise auto_color() would hand out an already-visually-identical color
    canvas.specs = [Spec("circle", (("x0", 20), ("y0", 20), ("radius", 5)),
                        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765))]
    assert canvas.auto_color() == palette_color(1)


def test_auto_color_skips_unparseable_spec_colors(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 20), ("y0", 20), ("radius", 5)), "not-a-real-color")]
    # the bogus color occupies no palette slot -- must not raise, must not block C0
    assert canvas.auto_color() == palette_color(0)


### editing existing regions ###############################################


def test_drag_body_moves_only_that_region(ax, canvas):
    canvas.specs = [
        Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red"),
        Spec("circle", (("x0", 150), ("y0", 150), ("radius", 10)), "blue"),
    ]
    canvas.redraw_overlays()
    _press(ax, (50, 50))     # select + start move on region 0 (body hit)
    _move(ax, (70, 55))
    _release(ax, (70, 55))
    assert dict(canvas.specs[0].params)["x0"] == 70
    assert dict(canvas.specs[0].params)["y0"] == 55
    assert dict(canvas.specs[1].params) == {"x0": 150, "y0": 150, "radius": 10}


def test_drag_corner_handle_resizes(ax, canvas):
    canvas.specs = [Spec("rectangle", (("x0", 100), ("y0", 100), ("width", 20), ("height", 20)),
                        "red")]
    canvas.redraw_overlays()
    canvas._selected = 0  # select first so handles are hit-testable, as a real click would
    canvas._draw_outlines()
    from uspy.interactive.spec import corner_positions
    corner = corner_positions(canvas.specs[0])[0]
    _press(ax, corner)
    _move(ax, (corner[0] + 20, corner[1] + 20))
    _release(ax, (corner[0] + 20, corner[1] + 20))
    geom = dict(canvas.specs[0].params)
    assert geom["width"] > 20
    assert geom["x0"] == 100 and geom["y0"] == 100  # centre stays fixed


def test_drag_corner_handle_with_shift_forces_square(ax, canvas):
    canvas.specs = [Spec("rectangle", (("x0", 100), ("y0", 100), ("width", 20), ("height", 20)),
                        "red")]
    canvas.redraw_overlays()
    canvas._selected = 0
    canvas._draw_outlines()
    from uspy.interactive.spec import corner_positions
    corner = corner_positions(canvas.specs[0])[0]
    _press(ax, corner)
    _move(ax, (corner[0] + 40, corner[1] + 10), key="shift")  # off-diagonal drag
    _release(ax, (corner[0] + 40, corner[1] + 10), key="shift")
    geom = dict(canvas.specs[0].params)
    assert geom["width"] == geom["height"]
    assert geom["x0"] == 100 and geom["y0"] == 100


def test_drag_corner_handle_with_shift_keeps_circle_a_circle(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 100), ("y0", 100), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    canvas._selected = 0
    canvas._draw_outlines()
    from uspy.interactive.spec import corner_positions
    corner = corner_positions(canvas.specs[0])[0]
    _press(ax, corner)
    _move(ax, (corner[0] + 40, corner[1] + 10), key="shift")  # off-diagonal drag
    _release(ax, (corner[0] + 40, corner[1] + 10), key="shift")
    assert canvas.specs[0].shape == "circle"


def test_shift_released_mid_resize_leaves_final_drag_unconstrained(ax, canvas):
    canvas.specs = [Spec("rectangle", (("x0", 100), ("y0", 100), ("width", 20), ("height", 20)),
                        "red")]
    canvas.redraw_overlays()
    canvas._selected = 0
    canvas._draw_outlines()
    from uspy.interactive.spec import corner_positions
    corner = corner_positions(canvas.specs[0])[0]
    _press(ax, corner)
    _move(ax, (corner[0] + 40, corner[1] + 10), key="shift")  # square while held
    _move(ax, (corner[0] + 40, corner[1] + 10))  # shift released, same endpoint
    _release(ax, (corner[0] + 40, corner[1] + 10))
    geom = dict(canvas.specs[0].params)
    assert geom["width"] != geom["height"]


def test_drag_rotate_handle_sets_degrees(ax, canvas):
    canvas.specs = [Spec("rectangle", (("x0", 100), ("y0", 100), ("width", 40), ("height", 10)),
                        "red")]
    canvas.redraw_overlays()
    canvas._selected = 0
    canvas._draw_outlines()
    from uspy.interactive.spec import handle_position
    handle = handle_position(canvas.specs[0], 20)
    _press(ax, handle)
    _move(ax, (100, 60))  # drag straight up -> screen angle -90 -> rotation 90
    _release(ax, (100, 60))
    assert dict(canvas.specs[0].params).get("rotation") == 90


def test_drag_vertex_moves_one_corner(ax, canvas):
    canvas.specs = [Spec("polygon", (("corners", ((0, 0), (50, 0), (50, 50), (0, 50))),), "red")]
    canvas.redraw_overlays()
    canvas._selected = 0
    canvas._draw_outlines()
    _drag(ax, (50, 0), (99, 1))  # drag the second vertex
    assert vertex_positions(canvas.specs[0]) == ((0, 0), (99, 1), (50, 50), (0, 50))


def test_all_regions_stay_editable(ax, canvas):
    canvas.specs = [
        Spec("circle", (("x0", 30), ("y0", 30), ("radius", 10)), "red"),
        Spec("rectangle", (("x0", 100), ("y0", 30), ("width", 20), ("height", 20)), "blue"),
        Spec("polygon", (("corners", ((150, 150), (180, 150), (165, 180))),), "green"),
    ]
    canvas.redraw_overlays()
    for i, inside in enumerate([(30, 30), (100, 30), (165, 160)]):
        _drag(ax, inside, (inside[0] + 5, inside[1] + 5))
        moved = canvas.specs[i]
        assert moved is not None  # still present, still a valid spec
    # every region actually moved (none left untouched / unhit)
    assert dict(canvas.specs[0].params)["x0"] == 35
    assert dict(canvas.specs[1].params)["x0"] == 105
    assert vertex_positions(canvas.specs[2])[0] == (155, 155)


def test_insert_and_remove_vertex(ax, canvas):
    canvas.specs = [Spec("polygon", (("corners", ((0, 0), (100, 0), (100, 100), (0, 100))),), "red")]
    canvas.redraw_overlays()
    canvas._selected = 0
    canvas._draw_outlines()
    # double-click near the midpoint of the first edge to insert a vertex there
    _press(ax, (50, 0), dblclick=True)
    assert vertex_positions(canvas.specs[0]) == ((0, 0), (50, 0), (100, 0), (100, 100), (0, 100))

    canvas._selected_vertex = 1  # as if that vertex had just been grabbed
    _key(ax, "delete")
    assert vertex_positions(canvas.specs[0]) == ((0, 0), (100, 0), (100, 100), (0, 100))


def test_remove_vertex_refuses_below_three(ax, canvas):
    canvas.specs = [Spec("polygon", (("corners", ((0, 0), (10, 0), (10, 10))),), "red")]
    canvas.redraw_overlays()
    canvas._selected = 0
    canvas._selected_vertex = 0
    _key(ax, "delete")
    assert len(vertex_positions(canvas.specs[0])) == 3  # refused, nothing removed


def test_delete_key_removes_whole_selected_region(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    canvas._selected = 0
    canvas._selected_vertex = None
    _key(ax, "delete")
    assert canvas.specs == []


### "d" is the delete key, not "backspace" ###################################
# Regression coverage: reported bug -- pressing Backspace deleted the selected region
# *and* rewound the plot to its previous view. Root cause: "backspace" is in
# matplotlib's own rcParams["keymap.back"], and FigureManagerBase connects its
# key_press_handler to the same key_press_event on every managed figure -- so binding
# "backspace" here made Canvas and matplotlib both act on the same keypress. Fixed by
# moving off "backspace" onto "d" (unbound in every default keymap) instead of trying
# to suppress matplotlib's own handler.


def test_d_key_removes_whole_selected_region(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    canvas._selected = 0
    canvas._selected_vertex = None
    _key(ax, "d")
    assert canvas.specs == []


def test_d_key_removes_only_selected_vertex(ax, canvas):
    canvas.specs = [Spec("polygon", (("corners", ((0, 0), (100, 0), (100, 100), (0, 100))),), "red")]
    canvas.redraw_overlays()
    canvas._selected = 0
    canvas._selected_vertex = 1
    _key(ax, "d")
    assert vertex_positions(canvas.specs[0]) == ((0, 0), (100, 100), (0, 100))


def test_d_key_is_noop_when_nothing_selected(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    _key(ax, "d")
    assert len(canvas.specs) == 1


def test_backspace_does_not_delete_the_selected_region(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    canvas._selected = 0
    _key(ax, "backspace")
    assert len(canvas.specs) == 1  # left alone -- matplotlib owns this key (view-back)


def test_escape_cancels_drag(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    _press(ax, (50, 50))
    _move(ax, (90, 90))
    _key(ax, "escape")
    _release(ax, (90, 90))
    # the drag was cancelled before release, so the spec is unchanged
    assert dict(canvas.specs[0].params) == {"x0": 50, "y0": 50, "radius": 10}


def test_escape_cancels_pending_polygon_chain(ax, canvas):
    canvas.new_shape = "polygon"
    _press(ax, (10, 10))
    _press(ax, (60, 10))
    _key(ax, "escape")
    _press(ax, (200, 200), dblclick=True)  # would finish a chain if one were still open
    assert canvas.specs == []  # nothing committed -- 200,200 alone is not >= 3 points


### arrow-key nudging #######################################################
# shift+arrow, not plain arrow: matplotlib's own key_press_handler matches event.key
# by exact string against rcParams["keymap.back"] (["left", "c", "backspace"]) /
# ["keymap.forward"] (["right", "v"]) -- a modified arrow arrives as "shift+left",
# which matches neither, so plain left/right keep walking the toolbar's view history
# undisturbed. See test_plain_arrow_keys_do_not_nudge below for the regression pin.


def test_shift_arrows_nudge_selected_region_by_one_pixel(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    canvas._selected = 0
    _key(ax, "shift+right")
    assert dict(canvas.specs[0].params)["x0"] == 51
    _key(ax, "shift+left")
    assert dict(canvas.specs[0].params)["x0"] == 50
    _key(ax, "shift+down")
    assert dict(canvas.specs[0].params)["y0"] == 51
    _key(ax, "shift+up")
    assert dict(canvas.specs[0].params)["y0"] == 50


def test_plain_arrow_keys_do_not_nudge(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    canvas._selected = 0
    for key in ("left", "right", "up", "down"):
        _key(ax, key)
    assert dict(canvas.specs[0].params) == {"x0": 50, "y0": 50, "radius": 10}


def test_nudge_is_noop_when_nothing_selected(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    _key(ax, "shift+right")
    assert dict(canvas.specs[0].params)["x0"] == 50


def test_nudge_fires_on_commit_once(ax):
    calls = []
    c = Canvas(ax, H, W, on_commit=lambda: calls.append(1))
    c.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    c.redraw_overlays()
    c._selected = 0
    _key(ax, "shift+right")
    assert len(calls) == 1
    c.close()


def test_nudge_shifts_every_polygon_corner(ax, canvas):
    canvas.specs = [Spec("polygon", (("corners", ((0, 0), (50, 0), (50, 50), (0, 50))),), "red")]
    canvas.redraw_overlays()
    canvas._selected = 0
    _key(ax, "shift+right")
    assert vertex_positions(canvas.specs[0]) == ((1, 0), (51, 0), (51, 50), (1, 50))


def test_nudge_with_vertex_selected_moves_only_that_vertex(ax, canvas):
    canvas.specs = [Spec("polygon", (("corners", ((0, 0), (50, 0), (50, 50), (0, 50))),), "red")]
    canvas.redraw_overlays()
    canvas._selected = 0
    canvas._selected_vertex = 1  # as if that vertex had just been grabbed
    _key(ax, "shift+down")
    assert vertex_positions(canvas.specs[0]) == ((0, 0), (50, 1), (50, 50), (0, 50))


### rendering invariants ####################################################


def test_overlay_artists_do_not_accumulate(ax, canvas):
    canvas.specs = [
        Spec("circle", (("x0", 30), ("y0", 30), ("radius", 10)), "red"),
        Spec("circle", (("x0", 80), ("y0", 30), ("radius", 10)), "blue"),
        Spec("circle", (("x0", 130), ("y0", 30), ("radius", 10)), "green"),
    ]
    canvas.redraw_overlays()
    n_images = len(ax.images)
    for _ in range(5):
        canvas.redraw_overlays()
        assert len(ax.images) == n_images


def test_zoomed_axes_still_produces_a_full_frame_overlay(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    ax.set_xlim(20, 80)
    ax.set_ylim(80, 20)
    canvas.redraw_overlays()
    assert canvas._overlay_artists[0].get_array().shape == (H, W)


def test_commit_rasterizes_from_spec_not_outline(ax, canvas):
    canvas.specs = [Spec("rectangle", (("x0", 100), ("y0", 100), ("width", 20), ("height", 20)),
                        "red")]
    canvas.redraw_overlays()
    canvas._selected = 0
    canvas._draw_outlines()
    from uspy.interactive.spec import corner_positions
    corner = corner_positions(canvas.specs[0])[0]
    _drag(ax, corner, (corner[0] + 15, corner[1] + 15))
    expected = canvas.specs[0].to_object().pad_to(H, W)
    assert_array_equal(canvas._overlay_artists[0].get_array().data.astype(bool), expected)


def test_frame_change_does_not_leak_base_artists(ax, canvas):
    # Canvas itself only manages overlay/outline artists (not the base frame -- that's
    # picker.py/plot_img's job), so repeated redraw_overlays() calls alone must not
    # touch or accumulate any *other* artist that happens to be on the same axes.
    from matplotlib.lines import Line2D
    marker = Line2D([10], [10])
    ax.add_line(marker)
    for _ in range(5):
        canvas.redraw_overlays()
    assert marker in ax.lines


### programmatic editing (picker.py's widget-driven API) ####################


def test_select_reveals_handles(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    assert canvas.selected is None
    canvas.select(0)
    assert canvas.selected == 0


def test_set_rotation_sets_exact_value(ax, canvas):
    canvas.specs = [Spec("rectangle", (("x0", 50), ("y0", 50), ("width", 20), ("height", 10)),
                        "red")]
    canvas.redraw_overlays()
    canvas.set_rotation(0, 45)
    assert dict(canvas.specs[0].params)["rotation"] == 45


def test_set_label_and_color(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    canvas.set_label(0, "Cu")
    canvas.set_color(0, "blue")
    assert canvas.specs[0].label == "Cu"
    assert canvas.specs[0].color == "blue"


def test_delete_selected_removes_region(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    canvas.select(0)
    canvas.delete_selected()
    assert canvas.specs == []
    assert canvas.selected is None


def test_delete_selected_is_noop_when_nothing_selected(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    canvas.delete_selected()
    assert len(canvas.specs) == 1


def test_clear_removes_everything(ax, canvas):
    canvas.specs = [
        Spec("circle", (("x0", 30), ("y0", 30), ("radius", 10)), "red"),
        Spec("circle", (("x0", 80), ("y0", 30), ("radius", 10)), "blue"),
    ]
    canvas.redraw_overlays()
    canvas.clear()
    assert canvas.specs == []
    assert canvas.selected is None


def test_editing_methods_fire_on_commit(ax):
    calls = []
    c = Canvas(ax, H, W, on_commit=lambda: calls.append(1))
    c.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    c.redraw_overlays()
    c.set_label(0, "Cu")
    c.set_color(0, "blue")
    c.set_rotation(0, 30)
    c.delete_selected()  # nothing selected -- no-op, no extra commit
    assert len(calls) == 3
    c.close()


### drag redraw throttling / per-region artist tracking #####################
# Regression coverage for the performance bug reported after the picker's first
# release: matplotlib's ipympl/webagg backend sends a full-figure re-render over the
# websocket on every canvas.draw_idle() call with no debouncing of its own, and
# _on_move used to call _draw_outlines() (which rebuilt *every* region's artists) on
# every single motion event, unthrottled -- a drag could queue dozens of full
# re-renders that only drained sequentially, making the widget appear to "vanish" for
# 20-30s. These tests pin the fix: a throttle on how often a motion event actually
# triggers a redraw (while the Spec draft itself still updates every event, so the
# final position is exact), and per-region artist tracking so a redraw during a drag
# only touches the dragged region, not every committed one.


def test_drag_redraw_is_throttled(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    fake_time = [0.0]
    canvas._clock = lambda: fake_time[0]
    calls = []
    real_draw_outlines = canvas._draw_outlines
    canvas._draw_outlines = lambda *a, **kw: (calls.append(1), real_draw_outlines(*a, **kw))[1]

    _press(ax, (50, 50))
    calls.clear()  # press itself does one full (unthrottled, one-shot) rebuild
    for i in range(20):
        fake_time[0] += 0.001  # cumulative 0.02s, well under _REDRAW_MIN_INTERVAL
        _move(ax, (50 + i, 50))
    assert calls == []  # every one of the 20 rapid motion events was throttled away
    # but the draft geometry itself was NOT throttled -- it tracks every event exactly
    assert dict(canvas._drag["draft"].params)["x0"] == 50 + 19

    fake_time[0] += 1.0  # advance well past the throttle interval
    _move(ax, (90, 90))
    assert len(calls) == 1  # now it does fire
    _release(ax, (90, 90))


def test_release_always_redraws_even_after_a_throttled_motion_event(ax, canvas):
    canvas.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    fake_time = [0.0]
    canvas._clock = lambda: fake_time[0]

    _press(ax, (50, 50))
    fake_time[0] += 0.001  # under the throttle interval -- this move's redraw is skipped
    _move(ax, (60, 60))
    _release(ax, (60, 60))

    geom = dict(canvas.specs[0].params)
    assert geom["x0"] == 60 and geom["y0"] == 60  # the real endpoint, not a stale one
    assert 0 in canvas._outline_artists and len(canvas._outline_artists[0]) > 0


def test_drag_does_not_touch_other_regions_artist_identity(ax, canvas):
    canvas.specs = [
        Spec("circle", (("x0", 30), ("y0", 30), ("radius", 10)), "red"),
        Spec("circle", (("x0", 100), ("y0", 30), ("radius", 10)), "blue"),
    ]
    canvas.redraw_overlays()
    fake_time = [0.0]
    canvas._clock = lambda: fake_time[0]

    _press(ax, (30, 30))  # one-shot full rebuild at drag start -- not the target here
    other_before = list(canvas._outline_artists[1])

    for i in range(5):
        fake_time[0] += 1.0  # past the throttle every time, so each move *does* redraw
        _move(ax, (30 + i, 30))

    # region 1's artists are the exact same objects -- never removed/recreated by a
    # drag on region 0, even across several *unthrottled* redraws
    assert canvas._outline_artists[1] == other_before
    _release(ax, (30 + 4, 30))
    assert dict(canvas.specs[0].params)["x0"] == 34
    assert dict(canvas.specs[1].params) == {"x0": 100, "y0": 30, "radius": 10}


def test_drag_hides_stale_outline_while_dragging(ax, canvas):
    # during an edit-drag the region's live position is shown via "_draft"; its old,
    # stale committed-position outline must not also still be on screen.
    canvas.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    canvas.redraw_overlays()
    _press(ax, (50, 50))
    _move(ax, (80, 80))
    assert canvas._outline_artists.get(0, []) == []
    assert len(canvas._outline_artists.get("_draft", [])) > 0
    _release(ax, (80, 80))
    # after release, the real outline is back and the draft is gone
    assert len(canvas._outline_artists[0]) > 0
    assert canvas._outline_artists.get("_draft", []) == []


def test_new_region_drag_is_also_throttled(ax, canvas):
    canvas.new_shape = "circle"
    fake_time = [0.0]
    canvas._clock = lambda: fake_time[0]
    calls = []
    real_draw_outlines = canvas._draw_outlines
    canvas._draw_outlines = lambda *a, **kw: (calls.append(1), real_draw_outlines(*a, **kw))[1]

    _press(ax, (20, 20))
    calls.clear()
    for i in range(10):
        fake_time[0] += 0.001
        _move(ax, (20 + i, 20 + i))
    assert calls == []
    _release(ax, (30, 30))
    assert len(canvas.specs) == 1


### callbacks ################################################################


def test_on_commit_fires_on_new_region(ax):
    fig, axes = ax.figure, ax
    calls = []
    c = Canvas(axes, H, W, on_commit=lambda: calls.append(1))
    c.new_shape = "circle"
    _drag(axes, (60, 60), (100, 100))
    assert len(calls) == 1
    c.close()


def test_on_commit_fires_once_per_edit(ax):
    calls = []
    c = Canvas(ax, H, W, on_commit=lambda: calls.append(1))
    c.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    c.redraw_overlays()
    _drag(ax, (50, 50), (60, 60))
    assert len(calls) == 1
    c.close()


def test_on_drag_fires_during_existing_region_drag_not_for_new_region(ax):
    drag_calls = []
    c = Canvas(ax, H, W, on_drag=lambda i, s: drag_calls.append((i, s)))
    c.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    c.redraw_overlays()
    _press(ax, (50, 50))
    _move(ax, (55, 50))
    _move(ax, (60, 50))
    _release(ax, (60, 50))
    assert len(drag_calls) == 2
    assert all(i == 0 for i, _ in drag_calls)

    drag_calls.clear()
    c.new_shape = "circle"
    _drag(ax, (150, 150), (170, 170))  # drawing a brand new region
    assert drag_calls == []  # on_drag is only for editing *existing* regions
    c.close()


def test_on_select_fires_on_click_select_and_click_empty_deselect(ax):
    calls = []
    c = Canvas(ax, H, W, on_select=lambda i: calls.append(i))
    c.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    c.redraw_overlays()

    _press(ax, (50, 50))  # hit the region's body -> selects it
    _release(ax, (50, 50))
    assert calls == [0]

    _press(ax, (150, 150))  # empty canvas -> deselects
    assert calls == [0, None]
    c.close()


def test_on_select_does_not_fire_when_selection_unchanged(ax):
    calls = []
    c = Canvas(ax, H, W, on_select=lambda i: calls.append(i))
    c.specs = [Spec("circle", (("x0", 50), ("y0", 50), ("radius", 10)), "red")]
    c.redraw_overlays()
    c.select(0)
    calls.clear()
    c.select(0)  # already selected -- no change
    assert calls == []
    c.close()


def test_on_select_fires_on_append_and_on_delete(ax):
    calls = []
    c = Canvas(ax, H, W, on_select=lambda i: calls.append(i))
    c.new_shape = "circle"
    _drag(ax, (20, 20), (40, 40))
    assert calls == [0]  # a freshly appended region is auto-selected

    c.delete_selected()
    assert calls == [0, None]
    c.close()
