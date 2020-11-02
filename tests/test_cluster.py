"""Test cluster calculation functionality."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import pytest
import numpy as np

from agfalta.leem import cluster
from agfalta.leem import base

from .conftest import TESTDATA_DIR


@pytest.fixture
def pendry_stack():
    loaded = False
    try:
        stack = base.LEEMStack(TESTDATA_DIR + "pendried_stack.lstk")
        loaded = True
    except FileNotFoundError:
        stack = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_RuO2_normed_aligned.tif")
        stack.energy = np.linspace(3.0, 50.0, len(stack))

    X, _, _ = cluster.stack2vectors(stack, mask_outer=0.2) # cut away 20% on every side

    if loaded:
        X = stack.pendry
    else:
        # cut out a few bad frames
        X = np.delete(X, list((range(10))) + [160], axis=1)
        energy = np.delete(stack.energy, list((range(10))) + [160])

        X = cluster.pendryfy(X, energy)
        stack.pendry = X
        stack.save(TESTDATA_DIR + "pendried_stack.lstk")
    return stack


@pytest.mark.parametrize("n_components", [3, 12])
def test_pca_pendry(pendry_stack, n_components):
    trafo, _, _ = cluster.component_analysis(
        pendry_stack.pendry, algorithm="pca", n_components=n_components
    )
    W = trafo(pendry_stack.pendry)
    assert W.shape == (pendry_stack.pendry.shape[0], n_components)

@pytest.mark.parametrize("algorithm", ["pca", "nmf"])
def test_pca_full(single_stack, algorithm):
    single_stack = single_stack[10:30]
    X, _, _ = cluster.stack2vectors(single_stack, mask_outer=0.2)
    trafo, _, model = cluster.component_analysis(
        X, algorithm=algorithm, n_components=7
    )
    W = trafo(X)
    cluster.save_model(model, TESTDATA_DIR + "pca.model")
    _, inv_trafo, _ = cluster.load_pca_model(TESTDATA_DIR + "pca.model")
    X2 = inv_trafo(W)
    assert X.shape == X2.shape
