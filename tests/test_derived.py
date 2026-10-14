"""Tests for uspy.derived.ProfileLine/Waterfall -- previously covered only indirectly
via tests/test_hdf5.py's export round-trip and tests/test_plotting.py's `waterfall()`
smoke tests, neither of which pins the raw x/y/label content these two classes produce.
"""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring

import numpy as np
import pytest

import uspy.dataobject as do
from uspy.derived import ProfileLine, Waterfall
from uspy.roi import Profile


def _img(shape=(10, 20)):
    return do.Image(np.arange(np.prod(shape), dtype=float).reshape(shape))


def _stack(n=4, shape=(10, 20)):
    imgs = [do.Image(np.full(shape, float(i))) for i in range(n)]
    stack = do.ImageStack(imgs)
    for i, im in enumerate(stack):
        im.energy = float(i)
    return stack


### ProfileLine ##################################################################


def test_x_is_plain_pixel_index_not_stretched():
    # regression: x used to be np.linspace(0, len(y), len(y)), spacing samples
    # len(y)/(len(y)-1) px apart -- a ~1% stretch on an axis labelled "px"
    img = _img()
    profile = Profile([(0, 5), (9, 5)])
    line = ProfileLine(img, profile)
    np.testing.assert_array_equal(line.x, np.arange(len(line.y), dtype=float))


def test_xdim_ydim_and_x_unit():
    img = _img()
    profile = Profile([(0, 5), (9, 5)])
    line = ProfileLine(img, profile)
    assert line.xdim == "pixel"
    assert line.ydim == "intensity"
    assert line.unit["x"] == "px"


def test_label_propagates_from_profile():
    img = _img()
    profile = Profile([(0, 5), (9, 5)], label="step edge")
    line = ProfileLine(img, profile)
    assert line.label == "step edge"


def test_label_defaults_to_none():
    img = _img()
    profile = Profile([(0, 5), (9, 5)])
    line = ProfileLine(img, profile)
    assert line.label is None


def test_multisegment_sample_count_is_concatenation_of_segments():
    img = _img()
    seg1 = Profile([(0, 5), (9, 5)]).apply(img)
    seg2 = Profile([(9, 5), (9, 0)]).apply(img)
    multi = Profile([(0, 5), (9, 5), (9, 0)])
    line = ProfileLine(img, multi)
    assert len(line.y) == len(seg1) + len(seg2)
    np.testing.assert_array_equal(line.x, np.arange(len(line.y), dtype=float))


### Waterfall #####################################################################


def test_waterfall_shape_is_n_frames_by_n_samples():
    stack = _stack(n=5)
    profile = Profile([(0, 5), (19, 5)])
    n_samples = len(profile.apply(stack[0]))
    wf = Waterfall(stack, profile, "energy")
    assert wf.image.shape == (5, n_samples)
    assert wf.height == 5
    assert wf.width == n_samples


def test_waterfall_y_is_the_requested_stack_axis():
    stack = _stack(n=4)
    profile = Profile([(0, 5), (19, 5)])
    wf = Waterfall(stack, profile, "energy")
    np.testing.assert_array_equal(wf.y, np.array([0.0, 1.0, 2.0, 3.0]))


def test_waterfall_x_matches_a_profileline_over_one_frame():
    stack = _stack(n=3)
    profile = Profile([(0, 5), (19, 5)])
    wf = Waterfall(stack, profile, "energy")
    expected = ProfileLine(stack[0], profile).x
    np.testing.assert_array_equal(wf.x, expected)


def test_waterfall_rows_are_per_frame_profiles():
    stack = _stack(n=3)
    profile = Profile([(0, 5), (19, 5)])
    wf = Waterfall(stack, profile, "energy")
    for i, img in enumerate(stack):
        np.testing.assert_array_equal(wf.image[i], profile.apply(img))


def test_waterfall_calls_apply_exactly_once_per_frame(monkeypatch):
    # regression: Waterfall used to build a whole extra ProfileLine(stack[0],
    # profile).x just to get an arange -- one wasted Profile.apply call on top of
    # the N needed for the image rows. Pin exactly N calls, not N+1.
    stack = _stack(n=5)
    profile = Profile([(0, 5), (19, 5)])
    calls = []
    real_apply = Profile.apply

    def counting(self, obj):
        calls.append(1)
        return real_apply(self, obj)

    monkeypatch.setattr(Profile, "apply", counting)
    Waterfall(stack, profile, "energy")
    assert len(calls) == len(stack)


### StitchedLine ##################################################################


def _segment_stack(energies, scale):
    """Stack whose frame at energy E is the constant image scale * (2 + E / 10)."""
    stack = do.ImageStack([do.Image(np.full((6, 6), scale * (2 + e / 10))) for e in energies])
    for img, e in zip(stack, energies):
        img.energy = float(e)
    return stack


def _stitch(*segments):
    from uspy.derived import StitchedLine
    from uspy.roi import ROI

    stacks = [_segment_stack(*segment) for segment in segments]
    return StitchedLine(stacks, [ROI.from_array(np.ones((6, 6), dtype=bool))] * len(stacks), "energy")


@pytest.mark.parametrize(
    "segments",
    [
        # one-point overlaps (x of last == x of first of next), three segments scaled 1/.5/.25
        [(range(0, 5), 1.0), (range(4, 9), 0.5), (range(8, 13), 0.25)],
        # two segments, one-point overlap
        [(range(0, 5), 1.0), (range(4, 9), 0.5)],
        # genuine multi-point overlaps, three segments
        [(range(0, 7), 1.0), (range(4, 11), 0.5), (range(8, 15), 0.25)],
    ],
)
def test_stitched_line_rescales_segments_onto_the_first(segments):
    line = _stitch(*segments)
    np.testing.assert_allclose(line.y, 2 + line.x / 10, rtol=1e-6)
