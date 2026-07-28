"""Tests the uspy.leem.processing module.

Focused on `correct_img_distortion` (symmetry-guided distortion correction via
thin-plate splines) and `find_reflections` (LEED reflection auto-detection).
All tests are pure in-memory: synthetic LEED patterns (hexagonal, square,
rectangular) are rendered as Gaussian-blob images, put through a known affine
distortion, and corrected. The corrected reflections are then re-detected and
checked against the expected symmetry directly (equal radii / correct
inter-vertex angles), as well as against the ideal target lattice.

Correctness is deliberately checked on **non-square** image shapes (portrait
and landscape) in addition to square ones: `symmetrize.tps.tpsWarping`'s 2D
routine only preserves image shape/content by coincidence when the image is
square (see the comment in `correct_img_distortion`), so a fix that only
works for square inputs must not be able to pass these tests.

See tests/test_roi.py for the same build-Image-from-ndarray convention.
"""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name

import numpy as np
import pytest

pytest.importorskip("symmetrize")

from symmetrize import pointops as po, sym, tps

from uspy.leem import base
from uspy.leem import processing


### Synthetic LEED pattern helpers ##################################

# Anisotropic scale + shear: a "distortion" in the sense used throughout this
# module, i.e. not a pure rotation/translation (which correct_img_distortion
# cannot detect, since it only fixes shape, not overall scale/orientation).
DISTORTION_MATRIX = np.array([[1.15, 0.25], [0.10, 0.95]])

# Non-square shapes are the point: symmetrize's underlying TPS routine
# corrupts non-square images unless the (row, col) convention is handled
# consistently end-to-end (see the fix in correct_img_distortion).
SHAPES = [(220, 220), (220, 160), (160, 220)]


def _distort(coords, center, matrix=DISTORTION_MATRIX):
    """Apply a fixed affine distortion around `center` to a set of (row, col)
    points."""
    coords = np.asarray(coords, dtype=float)
    center = np.asarray(center, dtype=float)
    return (coords - center) @ matrix.T + center


def _regular_polygon(center, radius, n, angle0=0.0):
    """(row, col) coordinates of a regular n-gon's vertices around `center`."""
    center = np.asarray(center, dtype=float)
    angles = np.deg2rad(angle0 + np.arange(n) * (360.0 / n))
    return np.array(
        [[center[0] + radius * np.sin(a), center[1] + radius * np.cos(a)] for a in angles]
    )


def _hexagon(center, radius, angle0=0.0):
    return _regular_polygon(center, radius, 6, angle0=angle0)


def _rectangle(center, a, b, angle0=0.0):
    """(row, col) coordinates of a rectangle's 4 vertices (2 side lengths a, b)
    around `center`, ordered so opposite entries (0&2, 1&3) share a length."""
    center = np.asarray(center, dtype=float)
    theta0 = np.deg2rad(angle0)
    d0 = np.array([np.sin(theta0), np.cos(theta0)])
    d1 = np.array([np.sin(theta0 + np.pi / 2), np.cos(theta0 + np.pi / 2)])
    return np.array([center + a * d0, center + b * d1, center - a * d0, center - b * d1])


def _render_spots(shape, coords, sigma=5.0, amp=1000.0):
    """Render a black image (possibly non-square) with a 2D Gaussian blob at
    each (row, col) coord."""
    img = np.zeros(shape, dtype=np.float64)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for r, c in coords:
        img += amp * np.exp(-(((yy - r) ** 2 + (xx - c) ** 2) / (2 * sigma**2)))
    return img


def _center(shape):
    return np.array([shape[0] / 2.0, shape[1] / 2.0])


def _symmetry_metrics(coords):
    """From a set of (n_spots + 1) detected (row, col) points (incl. center),
    return the detected center, the surrounding points sorted ccw, their
    center-vertex distances ("radii"), and the angles between consecutive
    vertices as seen from the center (should sum to 360 deg)."""
    coords = np.asarray(coords, dtype=float)
    pcent, psur = po.pointset_center(coords, method="centroidnn")
    psur_ord = po.pointset_order(psur, direction="ccw")
    radii = po.cvdist(psur_ord, pcent)
    vecs = psur_ord - pcent
    n = len(vecs)
    angles = []
    for i in range(n):
        v1, v2 = vecs[i], vecs[(i + 1) % n]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angles.append(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
    return pcent, psur_ord, radii, np.array(angles)


def _radius_spread(radii):
    """Relative spread of the center-vertex distances (0 = perfectly regular)."""
    return (radii.max() - radii.min()) / radii.mean()


def _expected_regular_targets(pks, n_sides):
    """Independently recompute (using the public symmetrize API, the same way
    correct_img_distortion does) the ideal target positions a regular n-gon
    correction is supposed to reach: the surrounding points rotated around the
    center by exact n-gon angles, anchored at the (distorted) first ccw point.
    This is the algorithm's documented contract, not an implementation detail.
    """
    pcent, psur = po.pointset_center(pks, method="centroidnn")
    psur_ord = po.pointset_order(psur, direction="ccw")
    arot = np.full(n_sides - 1, 360.0 / n_sides)
    ptargs = sym.rotVertexGenerator(
        pcent,
        fixedvertex=psur_ord[0, :],
        cvd=po.cvdist(psur_ord, pcent).mean(),
        arot=arot,
        direction=-1,
        scale=1,
        ret="all",
    )
    return np.vstack([pcent, ptargs])


def _max_position_error(detected, expected):
    return max(np.linalg.norm(detected - e, axis=1).min() for e in expected)


def _swap(coords):
    """(row, col) <-> (x, y). All the geometry helpers above build points in
    (row, col) to match symmetrize's internal frame directly; uspy's public
    API (find_reflections/correct_img_distortion) is (x, y)."""
    return np.asarray(coords)[:, ::-1]


### find_reflections #################################################


def test_find_reflections_recovers_spots():
    shape = (200, 150)
    center = _center(shape)
    spots = _hexagon(center, 40)
    all_pts = np.vstack([center, spots])
    arr = _render_spots(shape, all_pts, sigma=4)
    img = base.LEEMImg(arr)

    detected = _swap(
        processing.find_reflections(img, len(all_pts), min_distance=8, threshold_rel=0.1)
    )
    assert len(detected) == len(all_pts)

    # every ground-truth spot should have a close detected match
    for pt in all_pts:
        dists = np.linalg.norm(detected - pt, axis=1)
        assert dists.min() < 2.0


def test_find_reflections_raises_when_not_enough_peaks():
    shape = (200, 150)
    center = _center(shape)
    spots = _hexagon(center, 40)
    all_pts = np.vstack([center, spots])
    arr = _render_spots(shape, all_pts, sigma=4)

    with pytest.raises(ValueError):
        processing.find_reflections(arr, len(all_pts) + 5, min_distance=8, threshold_rel=0.1)


def test_find_reflections_returns_xy():
    """find_reflections must return (x, y), not skimage's native (row, col):
    uspy's public coordinate convention throughout (see uspy.roi.ROI, whose
    constructors and properties are all (x0, y0)) is (x, y). Row and column
    are chosen deliberately unequal, and the column even exceeds the number of
    rows, so a YX/XY mix-up produces both a wrong and an out-of-bounds-looking
    result rather than an accidental pass."""
    shape = (200, 150)
    row, col = 30, 120
    arr = _render_spots(shape, [(row, col)], sigma=4)

    detected = processing.find_reflections(arr, 1, min_distance=8, threshold_rel=0.1)
    assert detected[0] == pytest.approx([col, row], abs=2)


### correct_img_distortion -- rotational symmetry, incl. non-square shapes ##


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_correct_hexagonal_restores_symmetry(shape):
    center = _center(shape)
    radius = min(shape) * 0.28
    ideal = _hexagon(center, radius)
    distorted = _distort(ideal, center)
    pks = np.vstack([center, distorted])

    distorted_arr = _render_spots(shape, pks, sigma=5)
    _, _, distorted_radii, distorted_angles = _symmetry_metrics(
        _swap(processing.find_reflections(distorted_arr, 7, min_distance=8, threshold_rel=0.1))
    )

    img = base.LEEMImg(distorted_arr)
    corrected = processing.correct_img_distortion(img, _swap(pks), symmetry="hexagonal")
    assert corrected.image.shape == shape

    detected = _swap(processing.find_reflections(corrected, 7, min_distance=8, threshold_rel=0.1))
    pcent, psur_ord, radii, angles = _symmetry_metrics(detected)

    # corrected pattern is (close to) a regular hexagon...
    assert _radius_spread(radii) < 0.05
    assert np.max(np.abs(angles - 60.0)) < 5.0
    # ...and substantially more so than the distorted input.
    assert _radius_spread(radii) < _radius_spread(distorted_radii) / 3
    assert np.max(np.abs(angles - 60.0)) < np.max(np.abs(distorted_angles - 60.0)) / 3

    # corrected spots land close to the algorithm's ideal target lattice
    expected = _expected_regular_targets(pks, n_sides=6)
    assert _max_position_error(np.vstack([pcent, psur_ord]), expected) < 3.0


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_correct_square_restores_symmetry(shape):
    center = _center(shape)
    radius = min(shape) * 0.28
    ideal = _regular_polygon(center, radius, 4, angle0=20)
    distorted = _distort(ideal, center)
    pks = np.vstack([center, distorted])

    distorted_arr = _render_spots(shape, pks, sigma=5)
    _, _, distorted_radii, distorted_angles = _symmetry_metrics(
        _swap(processing.find_reflections(distorted_arr, 5, min_distance=8, threshold_rel=0.1))
    )

    img = base.LEEMImg(distorted_arr)
    corrected = processing.correct_img_distortion(img, _swap(pks), symmetry="square")
    assert corrected.image.shape == shape

    detected = _swap(processing.find_reflections(corrected, 5, min_distance=8, threshold_rel=0.1))
    pcent, psur_ord, radii, angles = _symmetry_metrics(detected)

    assert _radius_spread(radii) < 0.05
    assert np.max(np.abs(angles - 90.0)) < 5.0
    assert _radius_spread(radii) < _radius_spread(distorted_radii) / 3
    assert np.max(np.abs(angles - 90.0)) < np.max(np.abs(distorted_angles - 90.0)) / 3

    expected = _expected_regular_targets(pks, n_sides=4)
    assert _max_position_error(np.vstack([pcent, psur_ord]), expected) < 3.0


### correct_img_distortion -- rectangular symmetry, incl. non-square shapes #


@pytest.mark.parametrize("shape", [(240, 240), (260, 180), (180, 260)], ids=lambda s: f"{s[0]}x{s[1]}")
def test_correct_rectangular_restores_symmetry(shape):
    center = _center(shape)
    a, b = min(shape) * 0.27, min(shape) * 0.17
    ideal = _rectangle(center, a, b, angle0=15)
    distorted = _distort(ideal, center)
    pks = np.vstack([center, distorted])

    distorted_arr = _render_spots(shape, pks, sigma=5)
    img = base.LEEMImg(distorted_arr)
    corrected = processing.correct_img_distortion(img, _swap(pks), symmetry="rectangular")
    assert corrected.image.shape == shape

    detected = _swap(
        processing.find_reflections(corrected, 5, min_distance=8, threshold_rel=0.1)
    ).astype(float)
    pcent, psur_ord, radii, angles = _symmetry_metrics(detected)
    vecs = psur_ord - pcent

    # opposite spots collinear through, and equidistant from, the center
    assert np.linalg.norm(vecs[0] + vecs[2]) < 4.0
    assert np.linalg.norm(vecs[1] + vecs[3]) < 4.0

    # adjacent edges (from the center) are perpendicular
    assert np.max(np.abs(angles - 90.0)) < 5.0

    # the rectangle's anisotropy (long side / short side) is preserved
    group1 = (radii[0] + radii[2]) / 2
    group2 = (radii[1] + radii[3]) / 2
    ratio_detected = max(group1, group2) / min(group1, group2)
    ratio_expected = max(a, b) / min(a, b)
    assert abs(ratio_detected - ratio_expected) / ratio_expected < 0.15


def test_correct_rectangular_requires_four_reflections():
    shape = (200, 150)
    center = _center(shape)
    pks = np.vstack([center, _hexagon(center, 40)])  # 6 surrounding spots, not 4

    with pytest.raises(ValueError):
        processing.correct_img_distortion(
            base.LEEMImg(_render_spots(shape, pks)), _swap(pks), symmetry="rectangular"
        )


### correct_img_distortion -- stack handling / API ###################


def test_correct_img_distortion_accepts_stack():
    shape = (200, 150)
    center = _center(shape)
    ideal = _hexagon(center, 40)
    distorted = _distort(ideal, center)
    pks = np.vstack([center, distorted])
    arr = _render_spots(shape, pks, sigma=4)

    stack = base.LEEMStack([base.LEEMImg(arr), base.LEEMImg(arr)])
    corrected_stack = processing.correct_img_distortion(stack, _swap(pks), symmetry="hexagonal")
    assert isinstance(corrected_stack, base.LEEMStack)
    assert len(corrected_stack) == 2
    for corrected_img in corrected_stack:
        assert corrected_img.image.shape == shape

    corrected_single = processing.correct_img_distortion(
        base.LEEMImg(arr), _swap(pks), symmetry="hexagonal"
    )
    assert isinstance(corrected_single, base.LEEMImg)
    assert corrected_single.image.shape == shape


def test_correct_img_distortion_requires_rot_for_custom_symmetry():
    shape = (200, 150)
    center = _center(shape)
    pks = np.vstack([center, _hexagon(center, 40)])
    img = base.LEEMImg(_render_spots(shape, pks))

    with pytest.raises(ValueError):
        processing.correct_img_distortion(img, _swap(pks), symmetry="custom")


### find_reflections + correct_img_distortion -- convention compatibility ###


def test_find_reflections_correct_img_distortion_roundtrip():
    """find_reflections' output must plug directly into correct_img_distortion
    with no manual coordinate swapping in between -- both are documented as
    (x, y), so they must be directly composable, which is the point of
    find_reflections's docstring example usage."""
    shape = (220, 160)
    center = _center(shape)
    radius = min(shape) * 0.28
    ideal = _hexagon(center, radius)
    distorted = _distort(ideal, center)
    pks_rc = np.vstack([center, distorted])
    arr = _render_spots(shape, pks_rc, sigma=5)

    detected_xy = processing.find_reflections(arr, 7, min_distance=8, threshold_rel=0.1)
    _, _, distorted_radii, distorted_angles = _symmetry_metrics(_swap(detected_xy))

    img = base.LEEMImg(arr)
    corrected = processing.correct_img_distortion(img, detected_xy, symmetry="hexagonal")
    assert corrected.image.shape == shape

    result_xy = processing.find_reflections(corrected, 7, min_distance=8, threshold_rel=0.1)
    _, _, radii, angles = _symmetry_metrics(_swap(result_xy))

    # corrected pattern is much closer to a regular hexagon than the input
    # was (pks here come from find_reflections' pixel-quantized detections
    # rather than exact analytic points, so use a relative rather than a tight
    # absolute tolerance).
    assert np.max(np.abs(angles - 60.0)) < 5.0
    assert _radius_spread(radii) < _radius_spread(distorted_radii) / 3
    assert np.max(np.abs(angles - 60.0)) < np.max(np.abs(distorted_angles - 60.0)) / 3


def test_correct_img_distortion_xy_matches_legacy_rowcol():
    """The (x, y) convention is purely a relabelling at the public boundary of
    correct_img_distortion: swapping pks to (x, y) and un-swapping it right
    back inside the function must reproduce exactly what calling the
    underlying symmetrize chain directly with the original (row, col) points
    produces (non-square shape, so a residual axis mix-up can't hide)."""
    shape = (220, 160)
    center = _center(shape)
    radius = min(shape) * 0.28
    ideal = _hexagon(center, radius)
    distorted = _distort(ideal, center)
    pks_rc = np.vstack([center, distorted])
    arr = _render_spots(shape, pks_rc, sigma=5)
    img = base.LEEMImg(arr)

    corrected_xy_api = processing.correct_img_distortion(img, _swap(pks_rc), symmetry="hexagonal")

    # Reference: replicate correct_img_distortion's body directly with the
    # original (row, col) points, bypassing the (x, y) -> (row, col) swap.
    pcent, psur = po.pointset_center(pks_rc, method="centroidnn")
    psur_ord = po.pointset_order(psur, direction="ccw")
    cvd = po.cvdist(psur_ord, pcent)
    ptargs = sym.rotVertexGenerator(
        pcent,
        fixedvertex=psur_ord[0, :],
        cvd=cvd.mean(),
        arot=np.full(5, 60.0),
        direction=-1,
        scale=1,
        ret="all",
    )
    nr, nc = arr.shape
    reference, _ = tps.tpsWarping(
        psur_ord, ptargs, arr, None, 3, ret="all", output_region=(0, 0, nr, nc)
    )

    assert np.allclose(corrected_xy_api.image, reference)
