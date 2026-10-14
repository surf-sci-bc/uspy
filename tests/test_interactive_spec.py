"""Tests for uspy.interactive.spec -- the pure geometry layer of the interactive ROI
picker (no matplotlib, no ipywidgets; see tests/test_interactive_canvas.py and
tests/test_interactive_picker.py for those layers).

All tests are pure in-memory, same convention as tests/test_roi.py.
"""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=protected-access

import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from uspy.roi import ROI, Profile
from uspy.interactive.spec import (
    Spec,
    SHAPES,
    specs_to_code,
    spec_from_bbox,
    spec_from_drag,
    spec_from_points,
    translate,
    resize_from_handle,
    rotation_from_handle,
    handle_position,
    corner_positions,
    vertex_positions,
    move_vertex,
    insert_vertex,
    remove_vertex,
    with_width,
    _screen_to_local,
    _local_to_screen,
)

H, W = 400, 400


### Spec: basic construction / hashing #################################


def test_spec_rejects_unknown_shape():
    with pytest.raises(ValueError):
        Spec("triangle", (("x0", 0), ("y0", 0)), "red")


def test_spec_hashable_and_equal():
    a = Spec("circle", (("x0", 1), ("y0", 2), ("radius", 5)), "red", "cp")
    b = Spec("circle", (("x0", 1), ("y0", 2), ("radius", 5)), "red", "cp")
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_spec_differs_by_color():
    a = Spec("circle", (("x0", 1), ("y0", 2), ("radius", 5)), "red")
    b = Spec("circle", (("x0", 1), ("y0", 2), ("radius", 5)), "blue")
    assert a != b


### Spec.to_object / to_dict / to_code ##################################


def test_to_object_returns_fresh_objects():
    spec = Spec("circle", (("x0", 10), ("y0", 10), ("radius", 5)), "red")
    a, b = spec.to_object(), spec.to_object()
    assert a is not b
    assert_array_equal(a.array, b.array)


@pytest.mark.parametrize(
    "spec",
    [
        Spec("point", (("x0", 33), ("y0", 44)), "red", "hotspot"),
        Spec("circle", (("x0", 300), ("y0", 350), ("radius", 40)), "b", "Cu"),
        Spec("ellipse", (("x0", 200), ("y0", 350), ("width", 80), ("height", 40)), "g"),
        Spec("rectangle", (("x0", 300), ("y0", 300), ("width", 20), ("height", 80)), "orange"),
        Spec("polygon", (("corners", ((210, 140), (268, 150), (250, 200))),), "r", "graphene"),
        Spec("ellipse", (("x0", 200), ("y0", 200), ("width", 60), ("height", 20),
                         ("rotation", 30)), "c"),
    ],
)
def test_code_roundtrips_for_every_roi_shape(spec):
    # eval(spec.to_code()) and spec.to_object() are two independent construction
    # paths (repr()-based vs ROI.from_spec-based) that must agree exactly.
    from_code = eval(spec.to_code(), {"ROI": ROI})  # pylint: disable=eval-used
    from_object = spec.to_object()
    assert_array_equal(from_code.array, from_object.array)
    assert from_code.color == spec.color
    assert from_code.label == spec.label


def test_code_roundtrips_for_profile():
    spec = Spec("profile", (("points", ((100, 100), (400, 300))), ("width", 10)), "purple")
    from_code = eval(spec.to_code(), {"Profile": Profile})  # pylint: disable=eval-used
    from_object = spec.to_object()
    assert from_code.points == from_object.points
    assert from_code.width == from_object.width
    assert from_code.color == spec.color


def test_to_dict_matches_from_spec_expectations():
    spec = Spec("circle", (("x0", 10), ("y0", 20), ("radius", 5)), "red")
    d = spec.to_dict()
    assert d == {"shape": "circle", "x0": 10, "y0": 20, "radius": 5}
    r = ROI.from_spec(d, color=spec.color)
    assert r.color == "red"


### spec_from_bbox #######################################################


def test_bbox_maps_to_centered_mask():
    # the single test that catches a centre/top-left mix-up: a box dragged from
    # (20,30) to (80,70) should give a shape centred at (50,50), not at (20,30).
    spec = spec_from_bbox("rectangle", x0=50, y0=50, width=60, height=40, rotation=0,
                          color="red")
    r = spec.to_object()
    ys, xs = np.nonzero(r.pad_to(H, W))
    assert abs(xs.mean() - 50) < 1
    assert abs(ys.mean() - 50) < 1


def test_no_y_flip():
    # a box in the *top* half of the image (small y) must yield a mask at *low* row
    # indices -- imshow's origin-upper convention, matched by ROI itself.
    spec = spec_from_bbox("rectangle", x0=100, y0=20, width=20, height=20, rotation=0,
                          color="red")
    ys, _ = np.nonzero(spec.to_object().pad_to(H, W))
    assert ys.max() < 50


def test_even_square_ellipse_becomes_circle():
    spec = spec_from_bbox("ellipse", 100, 100, 40, 40, 0, "red")
    assert spec.shape == "circle"
    assert dict(spec.params)["radius"] == 20
    assert_array_equal(spec.to_object().array, ROI.circle(100, 100, radius=20).array)


def test_odd_span_stays_ellipse():
    spec = spec_from_bbox("ellipse", 100, 100, 41, 41, 0, "red")
    assert spec.shape == "ellipse"


def test_nonsquare_ellipse_not_canonicalized():
    spec = spec_from_bbox("ellipse", 100, 100, 60, 20, 0, "red")
    assert spec.shape == "ellipse"


def test_rotation_zero_is_omitted_from_params():
    spec = spec_from_bbox("rectangle", 100, 100, 60, 20, 0, "red")
    assert "rotation" not in dict(spec.params)


def test_rotation_nonzero_is_kept():
    spec = spec_from_bbox("rectangle", 100, 100, 60, 20, 30, "red")
    assert dict(spec.params)["rotation"] == 30


def test_rotation_wraps_to_0_360():
    spec = spec_from_bbox("rectangle", 100, 100, 60, 20, -30, "red")
    assert dict(spec.params)["rotation"] == 330


### spec_from_drag (Fiji-style creation drag) ###########################


def test_drag_centered_rectangle_uses_press_point_as_center():
    # default (centered=True): the press point never moves, the drag only sets the
    # (doubled) half-extents -- unlike spec_from_bbox's corner-to-corner convention.
    spec = spec_from_drag("rectangle", (50, 50), (80, 60), "red")
    geom = dict(spec.params)
    assert geom["x0"] == 50 and geom["y0"] == 50
    assert geom["width"] == 60 and geom["height"] == 20


def test_drag_not_centered_reproduces_bbox_semantics():
    # centered=False is the old corner-to-corner gesture: same numbers spec_from_bbox
    # would give for the equivalent centre/extent box.
    drag = spec_from_drag("rectangle", (50, 50), (110, 70), "red", centered=False)
    bbox = spec_from_bbox("rectangle", 80, 60, 60, 20, 0, "red")
    assert drag == bbox


def test_drag_square_forces_equal_extents_centered():
    # centered + square: half-extent = the larger of |dx|, |dy| (60 > 20).
    spec = spec_from_drag("rectangle", (50, 50), (110, 70), "red", square=True)
    geom = dict(spec.params)
    assert geom["width"] == 120 and geom["height"] == 120


def test_drag_square_forces_equal_extents_not_centered():
    spec = spec_from_drag("rectangle", (50, 50), (110, 70), "red", centered=False, square=True)
    geom = dict(spec.params)
    assert geom["width"] == 60 and geom["height"] == 60


def test_drag_circle_centered_radius_is_drag_distance():
    # circle is always square; centered radius = the distance actually dragged
    # (hypot), not the larger axis-aligned extent -- so it grows straight out from
    # the click point regardless of drag direction.
    spec = spec_from_drag("circle", (100, 100), (103, 104), "red")  # 3-4-5 triangle
    assert spec.shape == "circle"
    assert dict(spec.params)["radius"] == 5


def test_drag_circle_not_centered_uses_larger_extent_as_diameter():
    # matches the pre-fix bounding-box behaviour exactly.
    spec = spec_from_drag("circle", (60, 60), (100, 100), "red", centered=False)
    assert spec.shape == "circle"
    assert dict(spec.params)["radius"] == 20


def test_drag_circle_center_stays_at_press_point():
    spec = spec_from_drag("circle", (60, 60), (100, 60), "red")
    geom = dict(spec.params)
    assert geom["x0"] == 60 and geom["y0"] == 60


def test_drag_minimum_extent_is_two_pixels():
    spec = spec_from_drag("rectangle", (50, 50), (50, 50), "red")  # no movement at all
    geom = dict(spec.params)
    assert geom["width"] >= 2 and geom["height"] >= 2


### spec_from_points (polygon / profile) ################################


def test_polygon_uses_absolute_coords_and_no_x0():
    corners = [(10, 10), (60, 10), (60, 40), (10, 40)]
    spec = spec_from_points("polygon", corners, "red")
    assert "x0" not in dict(spec.params)
    assert "y0" not in dict(spec.params)
    r = spec.to_object()  # must not raise TypeError (ROI.from_spec's polygon branch)
    ys, xs = np.nonzero(r.pad_to(100, 100))
    assert xs.min() >= 9 and xs.max() <= 60
    assert ys.min() >= 9 and ys.max() <= 40


def test_polygon_min_vertex_count_raises():
    with pytest.raises(ValueError):
        spec_from_points("polygon", [(0, 0), (10, 10)], "red")


def test_profile_min_point_count_raises():
    with pytest.raises(ValueError):
        spec_from_points("profile", [(0, 0)], "red")


def test_profile_deduplicates_consecutive_points():
    spec = spec_from_points("profile", [(0, 0), (0, 0), (10, 10)], "red")
    assert vertex_positions(spec) == ((0, 0), (10, 10))


### translate #############################################################


@pytest.mark.parametrize(
    "spec",
    [
        Spec("circle", (("x0", 100), ("y0", 100), ("radius", 20)), "red"),
        Spec("rectangle", (("x0", 100), ("y0", 100), ("width", 20), ("height", 10)), "red"),
        Spec("polygon", (("corners", ((0, 0), (10, 0), (10, 10))),), "red"),
        Spec("profile", (("points", ((0, 0), (10, 10))), ("width", 1)), "red"),
    ],
)
def test_translate_shifts_mask_by_delta(spec):
    # polygon/profile: check vertices directly -- a rasterized mean-of-mask check
    # would be thrown off whenever the translated shape's negative-coordinate part
    # gets clipped by pad_to, which dx=5, dy=-3 deliberately provokes here.
    if spec.shape in ("polygon", "profile"):
        before = vertex_positions(spec)
        after = vertex_positions(translate(spec, dx=5, dy=-3))
        assert after == tuple((x + 5, y - 3) for x, y in before)
        return
    before = spec.to_object()
    after = translate(spec, dx=5, dy=-3)
    before_ys, before_xs = np.nonzero(before.pad_to(H, W))
    after_ys, after_xs = np.nonzero(after.to_object().pad_to(H, W))
    assert abs((after_xs.mean() - before_xs.mean()) - 5) < 1
    assert abs((after_ys.mean() - before_ys.mean()) - -3) < 1


### resize_from_handle / rotation_from_handle / handle geometry ##########


def test_resize_keeps_center_fixed():
    spec = Spec("rectangle", (("x0", 100), ("y0", 100), ("width", 20), ("height", 20)), "red")
    resized = resize_from_handle(spec, (130, 100))  # drag right edge outward
    geom = dict(resized.params)
    assert geom["x0"] == 100 and geom["y0"] == 100
    assert geom["width"] > 20


def test_resize_under_rotation_changes_size_not_center():
    spec = Spec("ellipse", (("x0", 100), ("y0", 100), ("width", 40), ("height", 20),
                            ("rotation", 45)), "red")
    resized = resize_from_handle(spec, (140, 100))
    geom = dict(resized.params)
    assert geom["x0"] == 100 and geom["y0"] == 100
    assert geom.get("rotation") == 45


def test_resize_rejects_polygon():
    spec = Spec("polygon", (("corners", ((0, 0), (10, 0), (10, 10))),), "red")
    with pytest.raises(ValueError):
        resize_from_handle(spec, (5, 5))


def test_resize_square_forces_equal_width_and_height():
    spec = Spec("rectangle", (("x0", 100), ("y0", 100), ("width", 20), ("height", 20)), "red")
    # handle dragged off the diagonal -- free resize would give width != height
    resized = resize_from_handle(spec, (150, 110), square=True)
    geom = dict(resized.params)
    assert geom["x0"] == 100 and geom["y0"] == 100
    assert geom["width"] == geom["height"]
    assert geom["width"] == 2 * max(abs(150 - 100), abs(110 - 100))


def test_resize_square_false_is_unchanged_from_default():
    spec = Spec("rectangle", (("x0", 100), ("y0", 100), ("width", 20), ("height", 20)), "red")
    assert resize_from_handle(spec, (150, 110)) == resize_from_handle(spec, (150, 110), square=False)


def test_resize_square_preserves_center_and_rotation():
    # a rectangle stays a rectangle when squared (no ellipse->circle canonicalization),
    # so its rotation must survive
    spec = Spec("rectangle", (("x0", 100), ("y0", 100), ("width", 40), ("height", 20),
                              ("rotation", 45)), "red")
    resized = resize_from_handle(spec, (140, 105), square=True)
    geom = dict(resized.params)
    assert geom["x0"] == 100 and geom["y0"] == 100
    assert geom.get("rotation") == 45
    assert geom["width"] == geom["height"]


def test_resize_square_ellipse_that_becomes_equal_extents_canonicalizes_to_circle():
    # squaring an ellipse is exactly the "circle" case: width == height and even, so
    # spec_from_bbox's existing canonicalization fires and rotation is dropped as
    # rotationally meaningless -- same contract as rotation_from_handle on a circle.
    spec = Spec("ellipse", (("x0", 100), ("y0", 100), ("width", 40), ("height", 20),
                            ("rotation", 45)), "red")
    resized = resize_from_handle(spec, (140, 105), square=True)
    assert resized.shape == "circle"
    geom = dict(resized.params)
    assert geom["x0"] == 100 and geom["y0"] == 100


@pytest.mark.parametrize("xy", [(130, 100), (100 + 20 / math.sqrt(2), 100 + 20 / math.sqrt(2)),
                                (100 - 25, 100 + 5)])
def test_resize_square_circle_stays_circle_on_and_off_diagonal(xy):
    spec = Spec("circle", (("x0", 100), ("y0", 100), ("radius", 15)), "red")
    resized = resize_from_handle(spec, xy, square=True)
    assert resized.shape == "circle"
    geom = dict(resized.params)
    assert "radius" in geom


@pytest.mark.parametrize("angle_in,expected_rotation", [(0, 0), (90, 270), (180, 180),
                                                        (-90, 90), (270, 90)])
def test_rotation_from_handle_matches_calibrated_sign(angle_in, expected_rotation):
    # handle placed at a known screen angle from centre -> rotation. Calibrated
    # against ROI's own rotated masks in test_roi.py / the docstring of
    # uspy.interactive.spec._local_to_screen, not just self-consistency.
    spec = Spec("rectangle", (("x0", 100), ("y0", 100), ("width", 40), ("height", 20)), "red")
    distance = 50
    xy = (100 + distance * math.cos(math.radians(angle_in)),
         100 + distance * math.sin(math.radians(angle_in)))
    rotated = rotation_from_handle(spec, xy)
    # rotation=0 is intentionally omitted from params (spec_from_bbox), so use .get
    assert dict(rotated.params).get("rotation", 0) == expected_rotation


def test_rotation_from_handle_matches_actual_roi_rotation_direction():
    # end-to-end: the rotation this module computes, fed into ROI(), must place a
    # rectangle's marked long axis in the direction the handle was actually dragged
    # to -- not merely self-consistent within this module.
    spec = Spec("rectangle", (("x0", 200), ("y0", 200), ("width", 60), ("height", 6)), "red")
    # drag the handle to screen angle -90 (straight up: dx=0, dy=-50)
    rotated = rotation_from_handle(spec, (200, 150))
    roi = rotated.to_object()
    ys, xs = np.nonzero(roi.array)
    # long axis should now run vertically (row extent > col extent)
    assert (ys.max() - ys.min()) > (xs.max() - xs.min())


def test_rotate_circle_stays_circle_and_drops_rotation():
    # a circle is rotationally symmetric, so rotation_from_handle deliberately
    # canonicalizes it straight back to "circle" (via spec_from_bbox) regardless of
    # the dragged angle, rather than storing a rotation that would be visually
    # meaningless -- see spec_from_bbox's docstring/comment.
    spec = Spec("circle", (("x0", 100), ("y0", 100), ("radius", 20)), "red")
    rotated = rotation_from_handle(spec, (100 + 40, 100 - 40))  # 45 degrees
    assert rotated.shape == "circle"
    assert "rotation" not in dict(rotated.params)
    assert dict(rotated.params)["radius"] == 20


def test_handle_position_matches_rotation_from_handle_roundtrip():
    for rotation in (0, 30, 90, 200, 350):
        spec = Spec("rectangle", (("x0", 50), ("y0", 50), ("width", 20), ("height", 10),
                                  ("rotation", rotation)), "red")
        if rotation == 0:
            spec = Spec("rectangle", (("x0", 50), ("y0", 50), ("width", 20), ("height", 10)), "red")
        handle_xy = handle_position(spec, distance=30)
        recovered = rotation_from_handle(spec, handle_xy)
        assert dict(recovered.params).get("rotation", 0) == rotation


def test_corner_positions_returns_four_points_around_center():
    spec = Spec("rectangle", (("x0", 100), ("y0", 100), ("width", 40), ("height", 20)), "red")
    corners = corner_positions(spec)
    assert len(corners) == 4
    cx = sum(c[0] for c in corners) / 4
    cy = sum(c[1] for c in corners) / 4
    assert abs(cx - 100) < 1e-6 and abs(cy - 100) < 1e-6


def test_local_to_screen_and_screen_to_local_are_inverses():
    for rotation in (0, 37, 123, 270):
        lx, ly = 12.0, -5.0
        x, y = _local_to_screen(10, 20, lx, ly, rotation)
        back_lx, back_ly = _screen_to_local(10, 20, x, y, rotation)
        assert abs(back_lx - lx) < 1e-9
        assert abs(back_ly - ly) < 1e-9


### vertex editing (polygon / profile) ###################################


def test_move_vertex_moves_only_that_one():
    spec = Spec("polygon", (("corners", ((0, 0), (10, 0), (10, 10))),), "red")
    moved = move_vertex(spec, 1, (99, 99))
    assert vertex_positions(moved) == ((0, 0), (99, 99), (10, 10))


def test_insert_vertex_adds_after_edge_index():
    spec = Spec("polygon", (("corners", ((0, 0), (10, 0), (10, 10))),), "red")
    inserted = insert_vertex(spec, 0, (5, -5))
    assert vertex_positions(inserted) == ((0, 0), (5, -5), (10, 0), (10, 10))


def test_remove_vertex_drops_the_index():
    spec = Spec("polygon", (("corners", ((0, 0), (10, 0), (10, 10), (0, 10))),), "red")
    removed = remove_vertex(spec, 1)
    assert vertex_positions(removed) == ((0, 0), (10, 10), (0, 10))


def test_remove_vertex_refuses_below_three_for_polygon():
    spec = Spec("polygon", (("corners", ((0, 0), (10, 0), (10, 10))),), "red")
    with pytest.raises(ValueError):
        remove_vertex(spec, 0)


def test_remove_vertex_refuses_below_two_for_profile():
    spec = Spec("profile", (("points", ((0, 0), (10, 10))), ("width", 1)), "red")
    with pytest.raises(ValueError):
        remove_vertex(spec, 0)


def test_vertex_ops_reject_non_vertex_shapes():
    spec = Spec("circle", (("x0", 0), ("y0", 0), ("radius", 5)), "red")
    with pytest.raises(ValueError):
        move_vertex(spec, 0, (1, 1))


### with_width (profile linewidth editing) #################################


def test_with_width_updates_a_profile_spec():
    spec = Spec("profile", (("points", ((0, 0), (10, 10))), ("width", 1)), "red")
    wider = with_width(spec, 5)
    assert dict(wider.params)["width"] == 5
    assert dict(wider.params)["points"] == ((0, 0), (10, 10))  # points untouched


def test_with_width_round_trips_through_to_object():
    spec = Spec("profile", (("points", ((0, 0), (10, 10))), ("width", 1)), "red")
    wider = with_width(spec, 7)
    assert wider.to_object().width == 7


def test_with_width_rejects_non_profile_shapes():
    spec = Spec("circle", (("x0", 0), ("y0", 0), ("radius", 5)), "red")
    with pytest.raises(ValueError):
        with_width(spec, 5)


### Spec.from_roi (seeding from existing ROIs) ############################


@pytest.mark.parametrize(
    "roi",
    [
        ROI.point(3, 4, color="red", label="hotspot"),
        ROI.circle(300, 350, radius=40, color="b", label="Cu"),
        ROI(200, 350, source="ellipse", width=80, height=40, color="g"),
        ROI(300, 300, source="rectangle", width=20, height=80, color="orange"),
        ROI(300, 300, source="square", width=20, color="purple"),
        ROI.polygon([(210, 140), (268, 150), (250, 200)], color="r", label="graphene"),
        ROI(200, 200, source="ellipse", width=60, height=20, rotation=30, color="c"),
    ],
)
def test_from_roi_round_trips(roi):
    spec = Spec.from_roi(roi)
    rebuilt = spec.to_object()
    assert_array_equal(rebuilt.array, roi.array)
    assert rebuilt.color == roi.color
    assert rebuilt.label == roi.label


@pytest.mark.parametrize(
    "roi",
    [
        ROI(60, 60, source="square", width=20, color="r"),
        ROI(60, 60, source="square", width=20, rotation=30, color="r"),
        ROI(60, 60, source="ellipse", radius=7, color="r"),
        ROI(60, 60, source="circle", width=20, color="r"),
        ROI(60, 60, source="circle", width=20, height=10, color="r"),
        ROI(60, 60, source="rectangle", width=20, color="r"),
        ROI.circle(60, 60, radius=7.5, color="r"),
        ROI.circle(60, 60, width=15.5, color="r"),
        ROI(60, 60, source="ellipse", width=20.5, height=10.5, color="r"),
        ROI(60, 60, source="square", width=20.5, color="r"),
        ROI.polygon([(10.5, 14.5), (68.5, 15.5), (50.5, 60.5)], color="r"),
    ],
    ids=["square-w", "square-rot", "ellipse-radius", "circle-width", "circle-w-h",
         "rect-w-only", "circle-float-radius", "circle-float-width", "ellipse-float",
         "square-float", "polygon-float"],
)
def test_from_roi_accepts_every_parameterisation_and_keeps_floats(roi):
    spec = Spec.from_roi(roi)
    rebuilt = spec.to_object()  # must not raise (_build_outline-style KeyError)
    assert_array_equal(rebuilt.array, roi.array)
    # the code emitted for the untouched seed evaluates to the same region
    again = eval(spec.to_code(), {"ROI": ROI})  # pylint: disable=eval-used
    assert_array_equal(again.array, roi.array)


def test_from_roi_keeps_float_geometry_in_spec():
    spec = Spec.from_roi(ROI.circle(60, 61, radius=7.5, color="r"))
    assert dict(spec.params)["radius"] == 7.5
    spec = Spec.from_roi(ROI(60, 60, source="square", width=20.5, color="r"))
    assert dict(spec.params)["width"] == dict(spec.params)["height"] == 20.5
    spec = Spec.from_roi(ROI.polygon([(10.5, 14.5), (68.5, 15.5), (50.5, 60.5)], color="r"))
    assert dict(spec.params)["corners"][0] == (10.5, 14.5)
    assert all(type(v) in (int, float) for xy in dict(spec.params)["corners"] for v in xy)


def test_from_roi_float_position_is_not_truncated():
    spec = Spec.from_roi(ROI.circle(50.7, 40.2, radius=7, color="r"))
    assert dict(spec.params)["x0"] == 50.7 and dict(spec.params)["y0"] == 40.2


def test_from_roi_maps_square_to_rectangle_shape_name():
    roi = ROI(300, 300, source="square", width=20, color="purple")
    spec = Spec.from_roi(roi)
    assert spec.shape == "rectangle"


def test_from_roi_rejects_array_roi():
    roi = ROI.from_array([[True, False], [False, True]], x0=5, y0=3, color="b")
    with pytest.raises(ValueError):
        Spec.from_roi(roi)


def test_from_roi_rejects_add_result():
    union = ROI.circle(10, 10, radius=3) + ROI.circle(50, 50, radius=3)
    with pytest.raises(ValueError):
        Spec.from_roi(union)


def test_from_roi_profile_round_trips():
    profile = Profile([(100, 100), (400, 300)], width=10, color="#9467bd", label="edge")
    spec = Spec.from_roi(profile)
    rebuilt = spec.to_object()
    assert rebuilt.points == profile.points
    assert rebuilt.width == profile.width
    assert rebuilt.label == profile.label


def test_from_roi_label_override():
    roi = ROI.circle(10, 10, radius=5, color="red", label="old")
    spec = Spec.from_roi(roi, label="new")
    assert spec.label == "new"


### specs_to_code (whole-block emission) ##################################


def test_specs_to_code_empty():
    assert specs_to_code([]) == "# nothing picked yet"


def test_specs_to_code_block_execs_and_matches_objects():
    specs = [
        Spec("circle", (("x0", 300), ("y0", 350), ("radius", 40)), "b", "Cu"),
        Spec("polygon", (("corners", ((210, 140), (268, 150), (250, 200))),), "r", "graphene"),
        Spec("profile", (("points", ((100, 100), (400, 300))), ("width", 10)), "purple"),
    ]
    code = specs_to_code(specs)
    ns: dict = {}
    exec(code, ns)  # pylint: disable=exec-used
    assert len(ns["rois"]) == 2
    assert len(ns["profiles"]) == 1
    assert_array_equal(ns["rois"][0].array, specs[0].to_object().array)
    assert ns["rois"][1].label == "graphene"
    assert ns["profiles"][0].width == 10


def test_specs_to_code_omits_empty_group():
    specs = [Spec("circle", (("x0", 1), ("y0", 1), ("radius", 5)), "red")]
    code = specs_to_code(specs)
    assert "profiles" not in code
    assert "rois = [" in code


def test_specs_to_code_no_imports():
    specs = [Spec("circle", (("x0", 1), ("y0", 1), ("radius", 5)), "red")]
    assert "import" not in specs_to_code(specs, imports=False)
