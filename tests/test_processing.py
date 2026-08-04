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
import scipy.spatial

pytest.importorskip("symmetrize")

from symmetrize import pointops as po, sym, tps

import uspy.dataobject as do
import uspy.roi as rois
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


### custom symmetry -- success path (see test_correct_img_distortion_requires_rot_for_custom_symmetry
### above for the missing-rot error path) ####################################


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_correct_custom_symmetry_matches_hexagonal(shape):
    """symmetry="custom" with rot=[60]*5 (the exact arot the "hexagonal" branch hardcodes,
    see correct_img_distortion) must restore symmetry exactly as well as symmetry="hexagonal"
    itself -- the only case where a truly independent ground truth is available for the
    otherwise arbitrary custom-rotation path."""
    center = _center(shape)
    radius = min(shape) * 0.28
    ideal = _hexagon(center, radius)
    distorted = _distort(ideal, center)
    pks = np.vstack([center, distorted])
    arr = _render_spots(shape, pks, sigma=5)
    img = base.LEEMImg(arr)

    corrected = processing.correct_img_distortion(
        img, _swap(pks), symmetry="custom", rot=np.full(5, 60.0)
    )
    assert corrected.image.shape == shape

    detected = _swap(processing.find_reflections(corrected, 7, min_distance=8, threshold_rel=0.1))
    _, _, radii, angles = _symmetry_metrics(detected)
    assert _radius_spread(radii) < 0.05
    assert np.max(np.abs(angles - 60.0)) < 5.0


### correct_stack_distortion #################################################


def test_correct_stack_distortion_restores_symmetry():
    """Feeds correct_stack_distortion the spline_warp a single-image
    correct_img_distortion call already produced, applied to a *fresh* stack -- ground
    truth is the same "corrected pattern is (close to) a regular hexagon" check the
    single-image tests above use."""
    shape = (200, 150)
    center = _center(shape)
    radius = min(shape) * 0.28
    ideal = _hexagon(center, radius)
    distorted = _distort(ideal, center)
    pks = np.vstack([center, distorted])
    arr = _render_spots(shape, pks, sigma=5)

    single = base.LEEMImg(arr.copy())
    corrected_single = processing.correct_img_distortion(single, _swap(pks), symmetry="hexagonal")

    fresh_stack = base.LEEMStack([base.LEEMImg(arr.copy()), base.LEEMImg(arr.copy())])
    warped_stack = processing.correct_stack_distortion(fresh_stack, corrected_single)
    assert warped_stack.image.shape == (2, *shape)

    for frame in warped_stack:
        detected = _swap(processing.find_reflections(frame, 7, min_distance=8, threshold_rel=0.1))
        _, _, radii, angles = _symmetry_metrics(detected)
        assert _radius_spread(radii) < 0.05
        assert np.max(np.abs(angles - 60.0)) < 5.0


def test_correct_stack_distortion_accepts_raw_deform_field_tuple():
    """spline may be given as a bare (rdeform, cdeform) tuple instead of an object exposing
    .spline_warp -- same ground truth (symmetry restored) as the object-based test above."""
    shape = (200, 150)
    center = _center(shape)
    radius = min(shape) * 0.28
    ideal = _hexagon(center, radius)
    distorted = _distort(ideal, center)
    pks = np.vstack([center, distorted])
    arr = _render_spots(shape, pks, sigma=5)

    single = base.LEEMImg(arr.copy())
    corrected_single = processing.correct_img_distortion(single, _swap(pks), symmetry="hexagonal")

    fresh_stack = base.LEEMStack([base.LEEMImg(arr.copy())])
    warped_stack = processing.correct_stack_distortion(
        fresh_stack, corrected_single.spline_warp
    )

    detected = _swap(
        processing.find_reflections(warped_stack[0], 7, min_distance=8, threshold_rel=0.1)
    )
    _, _, radii, angles = _symmetry_metrics(detected)
    assert _radius_spread(radii) < 0.05
    assert np.max(np.abs(angles - 60.0)) < 5.0


### calc_correlation ##########################################################


def test_calc_correlation_matches_known_correlation_distances():
    """calc_correlation calls scipy.spatial.distance.correlation, but only
    scipy.interpolate/optimize/constants/signal/ndimage were imported (processing.py) --
    every call raised AttributeError. Ground truth: an identical time series has
    correlation-distance 0, a negated (anti-correlated) one has distance 2."""
    n = 6
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    frames = np.zeros((n, 3, 3))
    frames[:, 0, 0] = ref  # identical to the reference
    frames[:, 1, 1] = -ref  # anti-correlated
    frames[:, 2, 2] = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])  # unrelated
    stack = base.LEEMStack(frames)
    line = do.Line(np.vstack([np.arange(n), ref]))

    corr = processing.calc_correlation(stack, line)

    assert corr[0, 0] == pytest.approx(0.0, abs=1e-10)
    assert corr[1, 1] == pytest.approx(2.0, abs=1e-10)
    expected_unrelated = scipy.spatial.distance.correlation(
        np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0]), ref
    )
    assert corr[2, 2] == pytest.approx(expected_unrelated)


def test_calc_correlation_threshold_filters_values():
    n = 6
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    frames = np.zeros((n, 2, 2))
    frames[:, 0, 0] = ref
    frames[:, 1, 1] = -ref
    stack = base.LEEMStack(frames)
    line = do.Line(np.vstack([np.arange(n), ref]))

    filtered = processing.calc_correlation(stack, line, threshold=1.0)

    assert np.all(filtered < 1.0)
    assert 0.0 in filtered


### xps_correct ################################################################


def _xps_correct_setup(W=30, H=24, a=0.01, b=-0.02, c0=0.3, sigma=0.15, amp=50.0, bkg=2.0):
    """Build a synthetic XPS-like stack whose per-pixel Gaussian peak center follows a
    *known* plane center(x, y) = a*x + b*y + c0 in binding energy -- so xps_correct's
    output has an independently computable ground truth (the same plane, mean-centered
    over the ROI positions used to fit it)."""
    n_frames = 41
    be = np.linspace(-1.0, 1.0, n_frames)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    center_plane = a * xx + b * yy + c0
    frames = np.zeros((n_frames, H, W))
    for k, e in enumerate(be):
        frames[k] = amp * np.exp(-0.5 * ((e - center_plane) / sigma) ** 2) + bkg
    stack = base.LEEMStack(frames)
    # binding_energy = photon_energy - energy - energy_offset (uspy.leem.base.LEEMImg);
    # photon_energy=energy_offset=0 and energy=-be makes binding_energy exactly be.
    stack.photon_energy = 0.0
    stack.energy_offset = 0.0
    stack.energy = -be
    return stack, xx, yy, a, b, c0


def test_xps_correct_recovers_known_energy_plane():
    """Also regression-covers a real bug found while writing this test: xps_correct
    reconstructed the correction plane with x/y swapped (`normal_vector[0] * yy +
    normal_vector[1] * xx`, cluster.py -- sorry, processing.py), even though _fit_plane was
    called with (x, y, shifts) in that order. Fixed to `normal_vector[0] * xx +
    normal_vector[1] * yy`."""
    stack, xx, yy, a, b, c0 = _xps_correct_setup()
    roi_list = [rois.ROI.point(5, 5), rois.ROI.point(25, 3), rois.ROI.point(10, 20)]
    points = np.array([r.position for r in roi_list])

    zz_neg = processing.xps_correct(stack, roi_list, plot=False)

    plane_at_points = a * points[:, 0] + b * points[:, 1] + c0
    expected = (a * xx + b * yy + c0) - plane_at_points.mean()
    # xps_correct returns -zz with zz shaped (width, height) (np.mgrid[0:width, 0:height]);
    # expected is (height, width) to match the image/meshgrid convention used to build it.
    np.testing.assert_allclose(-zz_neg.T, expected, atol=1e-6)


def test_xps_correct_accepts_coordinate_list_and_uses_given_radius():
    """The `type(roi[0]) is np.array` check never matched (np.array is a function, not a
    type: see the bug regression test's docstring above), so a plain list of (x, y)
    coordinates always fell through to the ROI-object code path and crashed. Also checks
    the (until now separately hardcoded) `radius` kwarg is honored, not silently replaced by
    a literal 5."""
    stack, *_ = _xps_correct_setup()
    coords = [np.array([5, 5]), np.array([25, 3]), np.array([10, 20])]

    zz_neg = processing.xps_correct(stack, coords, radius=2, plot=False)

    assert zz_neg.shape == (stack[0].width, stack[0].height)


def test_xps_correct_rejects_malformed_coordinates():
    stack, *_ = _xps_correct_setup()
    with pytest.raises(ValueError):
        processing.xps_correct(stack, [np.array([5, 5, 5])], plot=False)


def test_xps_correct_duplet_recovers_known_plane():
    """duplet=True fits two Gaussians and uses the mean of their two centers as the shift --
    build data where both peaks sit exactly on the same known plane, so the duplet's fitted
    shift has the identical ground truth as the singlet case."""
    W, H = 24, 20
    a, b, c0 = 0.008, 0.015, 0.2
    sep = 0.6
    sigma = 0.08
    amp = 30.0
    bkg = 1.0
    n_frames = 81
    be = np.linspace(-1.5, 1.5, n_frames)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    center_plane = a * xx + b * yy + c0
    frames = np.zeros((n_frames, H, W))
    for k, e in enumerate(be):
        frames[k] = (
            amp * np.exp(-0.5 * ((e - (center_plane - sep / 2)) / sigma) ** 2)
            + amp * np.exp(-0.5 * ((e - (center_plane + sep / 2)) / sigma) ** 2)
            + bkg
        )
    stack = base.LEEMStack(frames)
    stack.photon_energy = 0.0
    stack.energy_offset = 0.0
    stack.energy = -be

    roi_list = [rois.ROI.point(4, 4), rois.ROI.point(18, 2), rois.ROI.point(8, 15)]
    points = np.array([r.position for r in roi_list])

    # xps_correct's own default p0 heuristic for duplet=True guesses sigma=1 and a 1eV peak
    # separation (processing.py) -- far from this test's deliberately narrow/close peaks
    # (sigma=0.08, separation=0.6), so curve_fit doesn't converge from it. Passing a p0
    # close to the known truth is about curve_fit's basin of convergence, not the code
    # under test, so it's supplied explicitly here rather than weakening the peaks.
    center_guess = c0 + a * points[0, 0] + b * points[0, 1]
    p0 = [amp, center_guess - sep / 2, sigma, amp, center_guess + sep / 2, sigma, bkg, 0.0]
    zz_neg = processing.xps_correct(stack, roi_list, plot=False, duplet=True, p0=p0)

    plane_at_points = a * points[:, 0] + b * points[:, 1] + c0
    expected = (a * xx + b * yy + c0) - plane_at_points.mean()
    np.testing.assert_allclose(-zz_neg.T, expected, atol=1e-3)


### calculate_dose #############################################################


def test_calculate_dose_matches_closed_form_ramp():
    """Constant pressure and evenly-spaced rel_time give a closed-form linear dose ramp,
    independent of calling np.gradient/np.cumsum the way calculate_dose itself does."""
    n = 6
    pressure = 2e-6
    dt = 1.0
    stack = base.LEEMStack(np.zeros((n, 2, 2)))
    stack.timestamp = np.arange(n, dtype=float) * dt
    stack.time_origin = 0.0
    stack.pressure1 = pressure

    dose = processing.calculate_dose(stack)

    expected = pressure * dt * np.arange(1, n + 1) * 1e6
    np.testing.assert_allclose(dose, expected)


def test_calculate_dose_approx_keeps_output_length():
    n = 9
    stack = base.LEEMStack(np.zeros((n, 2, 2)))
    stack.timestamp = np.arange(n, dtype=float)
    stack.time_origin = 0.0
    stack.pressure1 = 3.0

    dose = processing.calculate_dose(stack, approx=2)

    assert len(dose) == len(stack)


### center_of_mass / maximum / fit_2d_gaussian #################################


def _bright_pixel_image(shape=(20, 24), row=8, col=15, value=100.0):
    arr = np.zeros(shape)
    arr[row, col] = value
    return base.LEEMImg(arr), row, col


def test_center_of_mass_locates_single_bright_pixel():
    img, row, col = _bright_pixel_image()
    roi = rois.ROI.circle(col, row, radius=5)

    com = processing.center_of_mass(img, roi)

    # center_of_mass returns (x, y): scipy.ndimage.center_of_mass on img.image.T.
    assert com == pytest.approx((col, row))


def test_maximum_locates_single_bright_pixel():
    img, row, col = _bright_pixel_image()
    roi = rois.ROI.circle(col, row, radius=5)

    argmax = processing.maximum(img, roi)

    # maximum returns (row, col): np.unravel_index on the untransposed image.
    assert argmax == (row, col)


def test_fit_2d_gaussian_brightest_row_is_bright_pixel_xy():
    """fit_2d_gaussian does no fitting despite its name -- it column-stacks (col, row,
    intensity) for every pixel in the ROI's bounding box. Pinning current behaviour: the
    brightest row is the (x, y, intensity) of the pixel actually placed."""
    img, row, col = _bright_pixel_image()
    roi = rois.ROI.circle(col, row, radius=5)

    out = processing.fit_2d_gaussian(img, roi)
    brightest = out[np.argmax(out[:, 2])]

    assert tuple(brightest) == pytest.approx((col, row, 100.0))


def test_center_of_mass_ignores_bright_pixel_outside_roi():
    """A far-outside bright pixel must not pull the COM at all: with a second, fainter
    pixel inside a distant ROI as the only unmasked mass, the COM must land exactly on
    that second pixel, not somewhere between the two."""
    arr = np.zeros((20, 24))
    arr[8, 15] = 100.0  # bright, outside the ROI below
    arr[2, 3] = 10.0  # inside the ROI below -- the only pixel that should count
    img = base.LEEMImg(arr)
    far_roi = rois.ROI.circle(2, 2, radius=3)

    com = processing.center_of_mass(img, far_roi)

    assert com == pytest.approx((3.0, 2.0))


### Pure math: fermi_edge / fit_fermi_edge / _fit_plane / _1gaussian / _2gaussian ####


def test_fermi_edge_far_field_limits():
    """Far below fd_center, the Fermi factor -> 0 and only the (constant) background
    contributes: y -> const_bkg + offset. Far above, the background's `<= fd_center` gate
    turns it off and the Fermi factor -> const_fd: y -> const_fd + offset. Using distinct
    const_fd/const_bkg makes the two plateaus independently checkable ground truths (with
    equal defaults, as in uspy's own default kwargs, both sides coincide at the same value,
    which wouldn't distinguish a x/asymptote mix-up)."""
    x = np.linspace(-2.0, 2.0, 4001)
    y = processing.fermi_edge(
        x, fd_center=0.0, fd_width=0.02, conv_width=0.01, const_fd=3.0, const_bkg=1.0, offset=0.5
    )
    assert y[0] == pytest.approx(1.5, abs=0.01)
    assert y[-1] == pytest.approx(3.5, abs=0.01)


def test_fit_fermi_edge_recovers_known_ef():
    """Also regression-covers two real bugs found while writing this test (see the fix's
    comments in uspy/leem/processing.py): fit_fermi_edge's inner `fe` was missing the "+ 1"
    Fermi-Dirac denominator (fermi_edge, its sibling/reference implementation right above,
    has it), and `bg` was called with its (a, b, ef) arguments reordered to (ef, a, b)
    against its own definition. Both fixed to match fermi_edge's formula. Target `y` is
    generated from fermi_edge_func's own (now-correct) formula with known parameters,
    refit, and the recovered ef is checked against the true value."""
    import scipy.ndimage

    kb = 8.617e-5
    temp = 40.0
    true_ef = 0.05
    true_a, true_b, true_c = 0.0, 0.3, 2.0
    true_conv_width, true_offset = 0.01, 0.0
    x = np.linspace(-0.5, 0.5, 400)

    def fe(x, ef):
        return 1 / (np.exp((x - ef) / (kb * temp)) + 1)

    def bg(x, a, b, ef):
        return np.array([a * xx + b if xx <= ef else 0 for xx in x])

    x_scaling = x[1] - x[0]
    y = (
        scipy.ndimage.gaussian_filter(
            true_c * (fe(x, true_ef) + bg(x, true_a, true_b, true_ef)),
            sigma=true_conv_width / x_scaling,
        )
        + true_offset
    )

    popt, _ = processing.fit_fermi_edge(
        x, y, temp, p0=[true_ef + 0.02, 0.0, 0.2, 1.5, 0.01, 0.0]
    )
    fitted_ef = popt[0]
    assert fitted_ef == pytest.approx(true_ef, abs=0.01)


def test_fit_plane_recovers_known_coefficients():
    # z = 2x + 3y + 1
    x_points = np.array([0.0, 1.0, 0.0])
    y_points = np.array([0.0, 0.0, 1.0])
    z_points = 2 * x_points + 3 * y_points + 1

    normal_vector, constant_term = processing._fit_plane(x_points, y_points, z_points)

    x_test, y_test = 5.0, 7.0
    z_pred = -(normal_vector[0] * x_test + normal_vector[1] * y_test + constant_term) / normal_vector[2]
    assert z_pred == pytest.approx(2 * x_test + 3 * y_test + 1)


def test_1gaussian_peak_position_and_amplitude():
    x = np.linspace(-5, 5, 1001)
    y = processing._1gaussian(x, amp1=1.0, cen1=0.5, sigma1=0.2, a=1.0, b=0.1)
    peak_x = x[np.argmax(y - (1.0 + 0.1 * x))]
    assert peak_x == pytest.approx(0.5, abs=0.02)
    assert y[500] < y[np.argmax(y)]  # x=~0 is not the peak


def test_2gaussian_recovers_both_peak_positions():
    x = np.linspace(-5, 5, 2001)
    y = processing._2gaussian(
        x, amp1=1.0, cen1=-1.5, sigma1=0.15, amp2=1.0, cen2=1.5, sigma2=0.15, a=0.0, b=0.0
    )
    # two well-separated peaks -> two local maxima near +-1.5
    left = x[(x > -3) & (x < 0)]
    right = x[(x > 0) & (x < 3)]
    y_left = y[(x > -3) & (x < 0)]
    y_right = y[(x > 0) & (x < 3)]
    assert left[np.argmax(y_left)] == pytest.approx(-1.5, abs=0.02)
    assert right[np.argmax(y_right)] == pytest.approx(1.5, abs=0.02)


### _shift_z_cols / correct_energies ############################################


def test_shift_z_cols_fractional_shift_matches_hand_computed_interpolation():
    stack = np.arange(5, dtype=np.float64).reshape(5, 1, 1)
    z_corr = np.array([[0.5]])

    out = processing._shift_z_cols(stack, z_corr)

    # floor(0.5)=0, frac=0.5: out[zz] = 0.5*stack[zz] + 0.5*stack[zz+1] for zz=0..3;
    # out[4] is never written (loop only runs zz in range(shape[0]-1)) and stays 0.
    expected = np.array([0.5, 1.5, 2.5, 3.5, 0.0]).reshape(5, 1, 1)
    np.testing.assert_allclose(out, expected)


def test_shift_z_cols_integer_shift_moves_values_up_by_one_index():
    stack = np.arange(5, dtype=np.float64).reshape(5, 1, 1)
    z_corr = np.array([[1.0]])

    out = processing._shift_z_cols(stack, z_corr)

    # floor(1.0)=1, frac=0.0: out[zz+1] = stack[zz+1] exactly, for zz=0..3; out[0] unwritten.
    expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0]).reshape(5, 1, 1)
    np.testing.assert_allclose(out, expected)


def test_shift_z_cols_out_of_range_shift_skips_every_pixel():
    stack = np.arange(5, dtype=np.float64).reshape(5, 1, 1)
    z_corr = np.array([[10.0]])  # always shifts out of the [0, 4] index range

    out = processing._shift_z_cols(stack, z_corr)

    np.testing.assert_allclose(out, np.zeros_like(stack))


def test_correct_energies_applies_shift_per_pixel():
    stack = base.LEEMStack(np.arange(5, dtype=np.float64).reshape(5, 1, 1))
    stack.energy = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # dE = energy[0]-energy[1] = 1
    z_corr = np.array([[0.5]])  # divided by dE=1 inside correct_energies -> shift 0.5

    corrected = processing.correct_energies(stack, z_corr)

    expected = np.array([0.5, 1.5, 2.5, 3.5, 0.0]).reshape(5, 1, 1)
    np.testing.assert_allclose(corrected.image, expected)
