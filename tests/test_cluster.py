"""Test cluster calculation functionality."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import pytest
import numpy as np

from uspy.leem import cluster
from uspy.leem import base

from .conftest import TESTDATA_DIR


def main():
    # pylint: disable=unused-variable, too-many-locals
    import matplotlib.pyplot as plt

    n_components = 15
    # for n_clusters in range(8, 9):
    n_clusters = 8

    loaded = False
    try:
        # raise FileNotFoundError
        stack = base.LEEMStack(TESTDATA_DIR + "pendried_stack.lstk")
        loaded = True
    except FileNotFoundError:
        stack = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_RuO2_normed_aligned_80-130.tif")
        stack.energy = np.linspace(3.0, 50.0, len(stack))
        stack = stack[10:]

    X, h, w = cluster.stack2vectors(stack, mask_outer=0.2) # cut away 20% on every side
    if loaded:
        X = stack.pendry
    else:
        X = cluster.pendryfy(X, stack.energy, smoothing_params={"both": True})
        stack.pendry = X
        stack.save(TESTDATA_DIR + "pendried_stack.lstk")

    trafo, inv_trafo, model = cluster.component_analysis(
        X, "pca",
        n_components=n_components
    )
    W = trafo(X)
    comps = cluster.vectors2stack(W, h, w)

    # labels, model = cluster.repeat_cluster_analysis(
    #     W, "pc-kmeans", n_iter=50, init="random",
    #     n_clusters=n_clusters, metric="euclidean_square"
    # )

    labels, model = cluster.elbow_cluster_analysis(
        W, "pc-kmeans", start=3, end=10, init="random",
        n_clusters=n_clusters, metric="euclidean_square"
    )

    fig, ax = plt.subplots()
    for Y in cluster.extract_IVs(X, labels):
        ax.plot(stack.energy, Y)
    fig3, ax3 = plt.subplots()
    ax3.imshow(labels.reshape(h, w))
    fig4, ax4 = plt.subplots()
    cluster.plot_IVs(stack, labels, ax=ax4)

        # fig.savefig(f"{n_clusters}_pendry.png")
        # fig3.savefig(f"{n_clusters}_clustermap.png")
        # fig4.savefig(f"{n_clusters}_IVs.png")
        # plt.close(fig)
        # plt.close(fig3)
        # plt.close(fig4)
    plt.show()


@pytest.fixture
def pendry_stack():
    loaded = False
    try:
        stack = base.LEEMStack(TESTDATA_DIR + "pendried_stack.lstk")
        loaded = True
    except FileNotFoundError:
        stack = base.LEEMStack(TESTDATA_DIR + "test_stack_IV_RuO2_normed_aligned_80-130.tif")
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

@pytest.mark.parametrize("algorithm", ["pca"])#, "nmf"])
def test_pca_full(short_stack, algorithm):
    X, _, _ = cluster.stack2vectors(short_stack, mask_outer=0.2)
    trafo, _, model = cluster.component_analysis(
        X, algorithm=algorithm, n_components=7
    )
    W = trafo(X)
    cluster.save_model(model, TESTDATA_DIR + "pca.model")
    _, inv_trafo, _ = cluster.load_pca_model(TESTDATA_DIR + "pca.model")
    X2 = inv_trafo(W)
    assert X.shape == X2.shape

@pytest.mark.parametrize("algorithm", ["pc-kmeans", "pc-xmeans", "sk-birch"])
def test_cluster(short_stack, algorithm):
    X, h, w = cluster.stack2vectors(short_stack, mask_outer=0.2)
    labels, _ = cluster.cluster_analysis(X, algorithm, metric="euclidean_square")
    assert len(labels) == h * w

if __name__ == "__main__":
    main()
