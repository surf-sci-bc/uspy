"""Fixtures for testing the uspy.xps.spectrum module."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import os

import pytest
import numpy as np

from uspy.xps import spectrum

TESTDATA_DIR = os.path.dirname(__file__) + "/../testdata/xps/"

EIS_FNAMES = [
    TESTDATA_DIR + "2020-02-20_Ga2O3_new_check-oxidation.txt"
]

@pytest.fixture(scope="module", params=EIS_FNAMES)
def spectrum_fname(request):
    return request.param

@pytest.fixture(scope="module")
def spec(spectrum_fname):
    return spectrum.ModeledSpectrum(spectrum_fname)


def test_spectrum_init(spectrum_fname):
    s = spectrum.ModeledSpectrum(spectrum_fname)
    assert s.energy.shape == s.intensity.shape
    assert s.fit_intensity.shape == s.background.shape

def test_spectrum_norm(spec):
    spec.norm("highest")
    assert np.isclose(spec.intensity.max(), 1)
    spec.norm("low_energy")
    assert spec.intensity.max() > 1
    spec.norm("energy", 200)
    assert np.isclose(spec.intensity_at_E(200), 1)
    spec.norm_divisor = 5
    assert spec.norm_type == "manual"

def test_spectrum_fit(spec):
    spec.add_peak("A", position=510, fwhm=3, area=4000)
    assert not (spec.fit_intensity == 0).all()
    fi1 = spec.fit_intensity
    spec.do_fit()
    assert (spec.fit_intensity != fi1).any()
