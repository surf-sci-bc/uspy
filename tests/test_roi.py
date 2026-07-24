"""Tests the uspy.roi module (ROI, Profile, StyledObject).

All tests are pure in-memory: no dependency on the testdata submodule. ROI is a
purely geometric class, so Image inputs are built directly from numpy arrays
(see tests/test_base.py:325 for the same convention).
"""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend for plot smoke tests
import matplotlib.pyplot as plt
from numpy.testing import assert_array_equal

import uspy.dataobject as do
from uspy.roi import ROI, Profile, StyledObject


### StyledObject ###################################################


def test_styled_object_default_color_cycles():
    StyledObject._idx = 0
    o1 = StyledObject()
    o2 = StyledObject()
    assert o1.color != o2.color


def test_styled_object_explicit_color():
    o = StyledObject(color="red")
    assert o.color == "red"


def test_styled_object_style_dict_overrides():
    o = StyledObject(style={"alpha": 0.3}, color="blue")
    assert o.style["alpha"] == 0.3
    assert o.style["color"] == "blue"


def test_styled_object_cmap_transparent_to_color():
    o = StyledObject(color="blue")
    cmap = o.cmap
    assert isinstance(cmap, matplotlib.colors.ListedColormap)
    assert cmap.N == 2
    assert cmap(0)[-1] == 0  # fully transparent
    assert cmap(1)[-1] == 1  # fully opaque, matches color


### ROI: shape construction #########################################


def test_roi_circle_area_and_geometry():
    radius = 10
    c = ROI.circle(0, 0, radius=radius)
    expected_area = np.pi * radius**2
    assert abs(c.area - expected_area) / expected_area < 0.15
    assert c.array.shape == (28, 28)  # int(sqrt((2r)**2 + (2r)**2))
    assert_array_equal(c.ref_point, [radius, radius])
    # a circle's mask is symmetric, so its center of mass is exactly the ref point
    assert_array_equal(c.center_of_mass, [radius, radius])


def test_roi_ellipse_area():
    width, height = 10, 6
    e = ROI(0, 0, source="ellipse", width=width, height=height)
    expected_area = np.pi * (width / 2) * (height / 2)
    assert abs(e.area - expected_area) / expected_area < 0.25


def test_roi_square_area_exact():
    # regression: square used to draw with pt1 at the shape's center instead of a
    # corner, filling (almost) the whole canvas instead of a width x width square.
    width = 10
    s = ROI(0, 0, source="square", width=width)
    assert s.area == width * width


def test_roi_rectangle_area_exact():
    width, height = 10, 6
    r = ROI(0, 0, source="rectangle", width=width, height=height)
    assert r.area == width * height


def test_roi_point_via_classmethod():
    p = ROI.point(3, 4)
    assert p.array.shape == (1, 1)
    assert p.area == 1
    assert_array_equal(p.position, [3, 4])


def test_roi_point_via_source_string():
    # regression: source="point" used to raise AttributeError because
    # _apply_params never assigned self.array for the "point" branch.
    p = ROI(3, 4, source="point")
    assert p.array.shape == (1, 1)
    assert p.area == 1


def test_roi_polygon_axis_aligned_square():
    poly = ROI.polygon([[0, 0], [10, 0], [10, 10], [0, 10]])
    assert poly.area == 100


def test_roi_from_array_roundtrip():
    arr = np.array([[True, False, True], [False, True, False], [True, True, False]])
    r = ROI.from_array(arr, x0=2, y0=3)
    assert_array_equal(r.array, arr)
    assert r.area == arr.sum()
    assert_array_equal(r.position, [2, 3])
    # array-sourced ROIs default to a zero ref_point, so origin == position
    assert_array_equal(r.origin, r.position)


def test_roi_unknown_shape_raises():
    try:
        ROI(0, 0, source="triangle", radius=5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_roi_radius_and_width_conflict_raises():
    try:
        ROI(0, 0, source="circle", radius=5, width=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


### ROI: array storage (packbits round-trip) #########################


def test_roi_array_roundtrip_non_multiple_of_8():
    # _packed_array uses np.packbits, which pads to whole bytes; the getter must
    # trim back to the exact original shape (9 elements here, not a multiple of 8).
    arr = np.array([[True, False, True], [False, True, False], [True, True, False]])
    r = ROI.from_array(arr)
    assert r.array.shape == (3, 3)
    assert_array_equal(r.array, arr)


def test_roi_array_setter_accepts_non_bool_input():
    r = ROI.from_array([[1, 0], [0, 1]])
    assert r.array.dtype == bool
    assert_array_equal(r.array, [[True, False], [False, True]])


### ROI: origin/ref_point ############################################


def test_roi_origin_equals_position_minus_ref_point():
    c = ROI.circle(20, 20, radius=5)
    assert_array_equal(c.origin, c.position - c.ref_point)
    assert_array_equal(c.origin, [15, 15])


### ROI.pad_to ########################################################


def test_pad_to_shape_matches_requested_extent():
    c = ROI.circle(15, 15, radius=3)
    padded = c.pad_to(30, 40)
    assert padded.shape == (30, 40)


def test_pad_to_fully_inside_preserves_area():
    c = ROI.circle(15, 15, radius=3)
    padded = c.pad_to(30, 30)
    assert padded.sum() == c.area


def test_pad_to_partial_overlap_crops():
    # circle centered near the image edge: part of its mask must be cropped away
    c = ROI.circle(1, 1, radius=5)
    padded = c.pad_to(10, 10)
    assert 0 < padded.sum() < c.area


def test_pad_to_fully_outside_returns_empty():
    c = ROI.circle(-50, -50, radius=5)
    padded = c.pad_to(10, 10)
    assert padded.sum() == 0
    assert padded.shape == (10, 10)


def test_pad_to_caches_result_for_repeated_non_square_call():
    # regression: the cache guard compared self._mask_buffer.shape against
    # (height, width) while pad_to actually produces shape (width, height), so
    # the cache never hit unless width == height.
    c = ROI.circle(15, 15, radius=3)
    first = c.pad_to(30, 40)
    second = c.pad_to(30, 40)
    assert first is second


def test_pad_to_recomputes_after_different_size():
    c = ROI.circle(15, 15, radius=3)
    first = c.pad_to(30, 40)
    other = c.pad_to(20, 20)
    assert first is not other
    assert other.shape == (20, 20)


### ROI.apply #########################################################


def test_apply_return_array_masks_outside_roi():
    img = do.Image(np.ones((40, 30)))
    c = ROI.circle(15, 20, radius=5)
    masked = c.apply(img, return_array=True)
    assert masked.count() == c.area
    assert masked.shape == img.image.shape


def test_apply_returns_copied_image_without_mutating_original():
    img = do.Image(np.ones((40, 30)))
    c = ROI.circle(15, 20, radius=5)
    result = c.apply(img)
    assert isinstance(result, do.Image)
    assert np.ma.is_masked(result.image)
    assert not np.ma.is_masked(img.image)


### ROI.__add__ #######################################################


def test_add_identical_rois_same_area():
    a = ROI.circle(10, 10, radius=5)
    b = ROI.circle(10, 10, radius=5)
    union = a + b
    assert union.area == a.area


def test_add_concentric_rois_matches_larger():
    # regression: __add__ used to do self.array + other.array directly, which
    # raises ValueError whenever the two ROIs' masks have different shapes.
    small = ROI.circle(10, 10, radius=5)
    big = ROI.circle(10, 10, radius=8)
    union = small + big
    assert union.area == big.area


def test_add_disjoint_rois_sums_areas():
    # regression: __add__ used to ignore each ROI's position entirely.
    a = ROI.circle(5, 5, radius=3)
    b = ROI.circle(50, 50, radius=3)
    union = a + b
    assert union.area == a.area + b.area


### ROI: exact data extraction / positioning ##########################
# Spectra are extracted by masking an image with a ROI and averaging the
# unmasked pixels (see IntensityLine.parse in uspy/derived.py), so a shifted
# ROI silently shifts every spectrum. These tests pin the exact positions
# each ROI type extracts, computed by hand before the ROI is ever applied.
# Images are value-encoded (value == row * width + col, unique per pixel) so
# extracted values can be decoded back to the (row, col) they came from.


def _encoded_image(h=40, w=50):
    return do.Image(np.arange(h * w).reshape(h, w).astype(float)), w


def _extracted_positions(roi, img, w):
    """The set of (row, col) positions a ROI actually extracts from img."""
    values = roi.apply(img, return_array=True).compressed().astype(int)
    return {(int(v) // w, int(v) % w) for v in values}


def test_extract_point_exact_pixel():
    # position is (x0, y0): x is the column, y is the row.
    img, w = _encoded_image()
    x0, y0 = 12, 7
    p = ROI.point(x0, y0)
    assert _extracted_positions(p, img, w) == {(y0, x0)}


def test_extract_from_array_exact_positions():
    # array-sourced ROIs have ref_point [0, 0], so position is the mask's
    # top-left corner: a True cell at mask[row, col] lands at (y0+row, x0+col).
    arr = np.array([[True, False, True], [False, True, False]])
    x0, y0 = 5, 3
    r = ROI.from_array(arr, x0=x0, y0=y0)
    img, w = _encoded_image()
    expected = {(y0 + row, x0 + col) for row, col in zip(*np.nonzero(arr))}
    assert expected == {(3, 5), (3, 7), (4, 6)}  # spelled out by hand too
    assert _extracted_positions(r, img, w) == expected


def test_extract_rectangle_exact_block():
    # a width x height rectangle centered on (x0, y0) extracts exactly the
    # block rows [y0-h/2, y0-h/2+h) x cols [x0-w/2, x0-w/2+w).
    x0, y0, width, height = 12, 7, 6, 4
    r = ROI(x0, y0, source="rectangle", width=width, height=height)
    img, w = _encoded_image()
    expected = {
        (row, col)
        for row in range(y0 - height // 2, y0 - height // 2 + height)
        for col in range(x0 - width // 2, x0 - width // 2 + width)
    }
    assert _extracted_positions(r, img, w) == expected
    assert r.area == width * height


def test_extract_rectangle_mask_matches_hand_built_full_mask():
    # Independent check that doesn't rely on the value-encoding helper above:
    # build the expected full-image boolean mask by hand and compare directly.
    x0, y0, width, height = 12, 7, 6, 4
    r = ROI(x0, y0, source="rectangle", width=width, height=height)
    h_img, w_img = 20, 30
    img = do.Image(np.zeros((h_img, w_img)))
    expected_mask = np.zeros((h_img, w_img), dtype=bool)
    expected_mask[
        y0 - height // 2 : y0 - height // 2 + height,
        x0 - width // 2 : x0 - width // 2 + width,
    ] = True
    masked = r.apply(img, return_array=True)
    assert_array_equal(~masked.mask, expected_mask)


def test_extract_square_exact_block():
    x0, y0, width = 12, 7, 6
    s = ROI(x0, y0, source="square", width=width)
    img, w = _encoded_image()
    half = width // 2
    expected = {
        (row, col)
        for row in range(y0 - half, y0 - half + width)
        for col in range(x0 - half, x0 - half + width)
    }
    assert _extracted_positions(s, img, w) == expected


def test_extract_polygon_axis_aligned_square_exact_block():
    x0, y0, side = 5, 5, 8
    poly = ROI.polygon(
        [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]]
    )
    img, w = _encoded_image()
    expected = {
        (row, col) for row in range(y0, y0 + side) for col in range(x0, x0 + side)
    }
    assert _extracted_positions(poly, img, w) == expected


def test_extract_polygon_axis_aligned_nonsquare_exact_block():
    # regression: polygons never rotate, so their mask canvas doesn't need to
    # be square; padding it square (to max(width, height)) used to let
    # draw_polygon's boundary-inclusive rasterization leak an extra row/col
    # of extracted pixels on the smaller dimension whenever width != height.
    x0, y0, width, height = 5, 5, 8, 5
    poly = ROI.polygon(
        [[x0, y0], [x0 + width, y0], [x0 + width, y0 + height], [x0, y0 + height]]
    )
    img, w = _encoded_image()
    expected = {
        (row, col) for row in range(y0, y0 + height) for col in range(x0, x0 + width)
    }
    assert _extracted_positions(poly, img, w) == expected
    assert poly.area == width * height


def test_extract_circle_center_of_mass_matches_position():
    x0, y0, radius = 25, 20, 6
    c = ROI.circle(x0, y0, radius=radius)
    img, w = _encoded_image()
    positions = np.array(list(_extracted_positions(c, img, w)))
    rows, cols = positions[:, 0], positions[:, 1]
    # a circle's mask is symmetric, so its center of mass is exactly the
    # requested position (same reasoning as test_roi_circle_area_and_geometry)
    assert rows.mean() == y0
    assert cols.mean() == x0
    dist = np.sqrt((rows - y0) ** 2 + (cols - x0) ** 2)
    assert dist.max() <= radius


def test_extract_ellipse_axes_not_swapped():
    # width maps to x/columns, height maps to y/rows: a wide, short ellipse
    # must extract a wide, short block, not the other way around.
    x0, y0, width, height = 25, 20, 16, 8
    e = ROI(x0, y0, source="ellipse", width=width, height=height)
    img, w = _encoded_image()
    positions = np.array(list(_extracted_positions(e, img, w)))
    rows, cols = positions[:, 0], positions[:, 1]
    assert rows.mean() == y0
    assert cols.mean() == x0
    row_extent = rows.max() - rows.min() + 1
    col_extent = cols.max() - cols.min() + 1
    assert row_extent < col_extent
    assert row_extent <= height
    assert col_extent <= width


### Profile ############################################################


def test_profile_apply_matches_known_gradient():
    # image whose value equals the column index everywhere
    arr = np.tile(np.arange(20.0), (10, 1))
    img = do.Image(arr)
    p = Profile(points=[(0, 5), (19, 5)], width=1)
    values = p.apply(img)
    assert len(values) == 20
    assert values[0] == 0
    assert values[-1] == 19


def test_profile_plot_smoke():
    p = Profile(points=[(0, 0), (5, 5)], color="green")
    fig, ax = plt.subplots()
    p.plot(ax)
    plt.close(fig)
    assert p.color == "green"


def test_profile_plot_band_is_true_to_size_horizontal():
    # regression: plot() used to draw ax.plot(..., linewidth=self.width), where
    # linewidth is in typographic points, not data/pixel units, so the drawn
    # band's thickness had no relation to the width profile_line() samples over.
    width = 6
    p = Profile(points=[(0, 0), (10, 0)], width=width)
    fig, ax = plt.subplots()
    p.plot(ax)
    assert len(ax.patches) == 1
    verts = ax.patches[0].get_path().vertices
    # perpendicular (y) extent of the band must equal self.width data units,
    # centered on the line (y=0)
    assert_array_equal(sorted({verts[:, 1].min(), verts[:, 1].max()}), [-width / 2, width / 2])
    assert verts[:, 0].min() == 0
    assert verts[:, 0].max() == 10
    plt.close(fig)


def test_profile_plot_band_thickness_independent_of_orientation():
    width = 4
    p = Profile(points=[(0, 0), (10, 10)], width=width)
    fig, ax = plt.subplots()
    p.plot(ax)
    path = ax.patches[0].get_path()
    verts = path.vertices[:4]
    # distance between the midpoints of the two long edges == band thickness
    mid_a = (verts[0] + verts[1]) / 2
    mid_b = (verts[2] + verts[3]) / 2
    assert np.isclose(np.linalg.norm(mid_a - mid_b), width)
    plt.close(fig)


def test_profile_plot_multisegment_is_a_single_seamless_patch():
    # regression: plot() used to draw one Polygon per segment, which left gaps
    # on the outer side of a corner and, since alpha < 1, a darker double-blended
    # seam on the inner (overlapping) side. A single Path/patch avoids both.
    p = Profile(points=[(0, 0), (5, 0), (5, 5)], width=2)
    fig, ax = plt.subplots()
    p.plot(ax)
    assert len(ax.patches) == 1
    plt.close(fig)


def test_profile_plot_corner_join_has_no_gap():
    # L-shaped profile turning at (10, 0): a point just outside the sharp corner
    # (within width/2 of the joint) must be filled by the round join. Neither
    # segment's own rectangle covers this point on its own.
    width = 4
    p = Profile(points=[(0, 0), (10, 0), (10, 10)], width=width)
    fig, ax = plt.subplots()
    p.plot(ax)
    path = ax.patches[0].get_path()
    outer_gap_point = (11.2, -1.2)
    assert np.linalg.norm(np.subtract(outer_gap_point, (10, 0))) < width / 2
    assert path.contains_point(outer_gap_point)
    plt.close(fig)


def test_profile_plot_corner_join_has_no_hole_in_overlap():
    # inner-corner overlap region must stay filled: nonzero-winding fill must not
    # cancel the rectangle/disk overlap into a hole.
    p = Profile(points=[(0, 0), (10, 0), (10, 10)], width=4)
    fig, ax = plt.subplots()
    p.plot(ax)
    path = ax.patches[0].get_path()
    assert path.contains_point((9, 1))
    plt.close(fig)


def test_profile_plot_far_outside_band_not_filled():
    p = Profile(points=[(0, 0), (10, 0), (10, 10)], width=4)
    fig, ax = plt.subplots()
    p.plot(ax)
    path = ax.patches[0].get_path()
    assert not path.contains_point((20, 20))
    plt.close(fig)


def test_profile_plot_skips_zero_length_segment():
    # duplicate consecutive points must not raise (division by zero on a
    # zero-length direction vector) and must not emit a degenerate extra patch
    p = Profile(points=[(0, 0), (0, 0), (5, 0)], width=2)
    fig, ax = plt.subplots()
    p.plot(ax)
    assert len(ax.patches) == 1
    plt.close(fig)


def test_profile_plot_single_point_draws_no_band():
    # fewer than two distinct points means no well-defined segment to draw
    p = Profile(points=[(0, 0)], width=2)
    fig, ax = plt.subplots()
    p.plot(ax)
    assert len(ax.patches) == 0
    plt.close(fig)


def test_profile_plot_uses_color_for_band_and_centerline():
    p = Profile(points=[(0, 0), (10, 0)], width=4, color="red")
    fig, ax = plt.subplots()
    p.plot(ax)
    assert ax.patches[0].get_facecolor()[:3] == matplotlib.colors.to_rgb("red")
    (line,) = ax.lines
    assert line.get_color() == "red"
    plt.close(fig)
