"""Test cluster calculation functionality."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import matplotlib.colors
import matplotlib.pyplot as plt
import pytest
import numpy as np

from uspy.leem import cluster
from uspy.leem import base

from .conftest import TESTDATA_DIR, RUO2_STACK_TIF


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


@pytest.fixture(scope="module")
def pendry_stack():
    # The original ...80-130.tif fixture this cache logic targeted no longer exists;
    # the only available equivalent is ...80-140.tif (61 frames, vs. the >160 frames the
    # original "cut out a few bad frames" indices assumed). Cache to .pickle -- the
    # only extension uspy.dataobject.Loadable.save()/load() actually recognizes; the
    # original ".lstk" extension is a leftover from the legacy uspy.legacy.base_old API
    # and is not supported by the current Loadable. Loading a cache also requires the
    # LEEMStack.load() classmethod, not the constructor (which treats its argument as
    # an image source/glob, not a pickle file, and raises ValueError for an unknown one).
    cache_fname = TESTDATA_DIR + "pendried_stack.pickle"
    loaded = False
    try:
        stack = base.LEEMStack.load(cache_fname)
        loaded = True
    except FileNotFoundError:
        stack = base.LEEMStack(RUO2_STACK_TIF)
        stack.energy = np.linspace(3.0, 50.0, len(stack))

    X, _, _ = cluster.stack2vectors(stack, mask_outer=0.2) # cut away 20% on every side

    if loaded:
        X = stack.pendry
    else:
        # cut out the first few frames (start of the energy ramp)
        cutoff = list(range(10))
        X = np.delete(X, cutoff, axis=1)
        energy = np.delete(stack.energy, cutoff)

        X = cluster.pendryfy(X, energy)
        stack.pendry = X
        stack.save(cache_fname)
    return stack


@pytest.mark.slow
@pytest.mark.parametrize("n_components", [3, 12])
def test_pca_pendry(pendry_stack, n_components):
    trafo, _, _ = cluster.component_analysis(
        pendry_stack.pendry, algorithm="pca", n_components=n_components
    )
    W = trafo(pendry_stack.pendry)
    assert W.shape == (pendry_stack.pendry.shape[0], n_components)

@pytest.mark.parametrize(
    "algorithm", ["pca", pytest.param("nmf", marks=pytest.mark.slow)]
)
def test_pca_full(short_stack, algorithm, tmp_path):
    X, _, _ = cluster.stack2vectors(short_stack, mask_outer=0.2)
    trafo, _, model = cluster.component_analysis(
        X, algorithm=algorithm, n_components=7
    )
    W = trafo(X)
    model_fname = str(tmp_path / "pca.model")
    cluster.save_model(model, model_fname)
    _, inv_trafo, _ = cluster.load_pca_model(model_fname)
    X2 = inv_trafo(W)
    assert X.shape == X2.shape

@pytest.mark.parametrize("algorithm", ["pc-kmeans", "pc-xmeans", "sk-birch"])
def test_cluster(short_stack, algorithm):
    X, h, w = cluster.stack2vectors(short_stack, mask_outer=0.2)
    labels, _ = cluster.cluster_analysis(X, algorithm, metric="euclidean_square")
    assert len(labels) == h * w


### --- Bug-fix regression tests (test-first, see CLAUDE.md #6) -----------------------


def test_enforce_clustershape_passes_matrix_not_stack2vectors_tuple(synthetic_leem_stack):
    """enforce_clustershape's whole purpose is letting a LEEMStack be passed directly to a
    vector-space function like normalize(); it must unpack stack2vectors' (X, h, w) return
    down to X, not forward the 3-tuple itself."""
    stack = synthetic_leem_stack(n=4, shape=(6, 8))
    expected_X, _, _ = cluster.stack2vectors(stack, mask_outer=0)
    expected_Xn, expected_integrals = cluster.normalize(expected_X)

    Xn, integrals = cluster.normalize(stack)

    np.testing.assert_allclose(Xn, expected_Xn)
    np.testing.assert_allclose(integrals, expected_integrals)


def test_smoothen_default_energy_falls_back_to_frame_count():
    """smoothen(X) with no energy given must guess an energy axis from X's own frame count
    (X.shape[1]), exactly like pendryfy does (cluster.py:303) -- not crash on len(None)."""
    X = np.tile(np.linspace(0, 1, 20), (5, 1))
    S = cluster.smoothen(X)
    assert S.shape == X.shape


def test_cutout_stack_reads_image_not_data(synthetic_leem_stack):
    """cutout_stack must read pixel data via img.image (an ndarray); img.data is a dict of
    every _data key in the current DataObject API and has no .shape."""
    stack = synthetic_leem_stack(n=3, shape=(10, 12))
    cut = cluster.cutout_stack(stack, mask_outer=0.2)
    dy, dx = int(0.2 * 10), int(0.2 * 12)
    expected = np.array([img.image[dy:-dy, dx:-dx] for img in stack])
    np.testing.assert_allclose(cut.image, expected)


def test_cutout_stack_mask_outer_zero_keeps_full_frame(synthetic_leem_stack):
    stack = synthetic_leem_stack(n=3, shape=(10, 12))
    cut = cluster.cutout_stack(stack, mask_outer=0)
    expected = np.array([img.image for img in stack])
    np.testing.assert_allclose(cut.image, expected)


def test_elbow_cluster_analysis_forwards_n_clusters(monkeypatch, blob_vectors):
    """Each sweep iteration of elbow_cluster_analysis must fit with its own n_clusters, not
    silently re-fit the same default model every time (which would make the 'elbow' search
    meaningless). Isolated from the real clusterers/kneed via monkeypatches, since this is
    purely about the forwarded argument."""
    X, _ = blob_vectors(n_clusters=2, n_per_cluster=5, n_frames=3)
    seen_n_clusters = []

    def fake_cluster_analysis(X, algorithm="pc-kmeans", **params):
        seen_n_clusters.append(params["n_clusters"])
        return np.zeros(X.shape[0]), params["n_clusters"]

    class FakeKnee:
        def __init__(self, x, y, **kwargs):  # pylint: disable=unused-argument
            self.elbow = list(x)[0]

    monkeypatch.setattr(cluster, "cluster_analysis", fake_cluster_analysis)
    monkeypatch.setattr(cluster, "goodness_of_model", lambda model: model)
    monkeypatch.setattr(cluster.kneed, "KneeLocator", FakeKnee)

    cluster.elbow_cluster_analysis(X, start=2, end=5)

    assert seen_n_clusters == [2, 3, 4, 5]


def test_plot_single_clusters_does_not_print_axes_array(capsys):
    """Leftover debug print(axes.flatten()) must not fire."""
    clustermap = np.array([[0, 1], [2, 3]])
    cluster.plot_single_clusters(clustermap)
    captured = capsys.readouterr()
    assert "Axes" not in captured.out
    plt.close("all")


def test_cluster_analysis_nltk_branch_is_reachable(blob_vectors):
    """The nltk- branch built `labels`/`model` (cluster.py:212-214) but the *separate*
    if/elif/else right below it (cluster.py:216-242) had no elif for "nltk-", so every
    nltk-* algorithm fell straight into the trailing `else: raise ValueError` -- the branch
    was unreachable dead code regardless of what it computed."""
    X, _ = blob_vectors(n_clusters=2, n_per_cluster=10, n_frames=3, spacing=100.0)
    labels, _ = cluster.cluster_analysis(X, algorithm="nltk-kmeans", num_means=2, repeats=3)
    assert len(labels) == X.shape[0]
    assert len(set(int(l) for l in labels)) == 2


### --- Pure math against ground truths ------------------------------------------------


def test_sort_labels_relabels_by_ascending_population():
    labels = np.array([5, 5, 5, 5, 2, 2, 7])
    # populations: 7 -> 1, 2 -> 2, 5 -> 4; ascending order gives new labels 7:0, 2:1, 5:2
    sorted_labels = cluster.sort_labels(labels)
    for old, new in {7: 0, 2: 1, 5: 2}.items():
        assert np.all(sorted_labels[labels == old] == new)


def test_combine_labels_merges_groups_into_first_member():
    labels = np.array([0, 1, 2, 2, 1, 0])
    combined = cluster.combine_labels(labels, [[1, 2]])
    np.testing.assert_array_equal(combined, np.array([0, 1, 1, 1, 1, 0]))


def test_make_cmap_from_explicit_phases():
    phases = {0: ("substrate", "#111111"), 1: ("film", "#222222")}
    labels = np.array([0, 1, 1, 0])
    cmap, legend = cluster.make_cmap(phases, labels)
    assert cmap.N == 2
    assert legend == {0: "substrate", 1: "film"}
    assert cmap(0) == matplotlib.colors.to_rgba("#111111")
    assert cmap(1) == matplotlib.colors.to_rgba("#222222")


def test_make_cmap_default_phases_uses_colormap():
    labels = np.array([0, 1, 2])
    cmap, legend = cluster.make_cmap({}, labels, default_phases=True)
    assert cmap.N == 3
    assert set(legend.values()) == {"0", "1", "2"}


def test_pendry_y_matches_closed_form_for_linear_intensity():
    """np.gradient is exact (not just approximate) for a linear function on a uniform grid,
    including at the edges -- so for I = a*E + b, gradient(I)/gradient(E) == a everywhere,
    giving a closed form for pendry_y independent of calling it."""
    E = np.linspace(3.0, 10.0, 15)
    a, b = 0.4, 2.0
    I = a * E + b
    dE = np.gradient(E)
    V0i_sq = E ** (2 / 3)

    Y = cluster.pendry_y(I, dE, V0i_sq)

    L_expected = a / I
    Y_expected = L_expected / (1 + L_expected**2 * V0i_sq)
    np.testing.assert_allclose(Y, Y_expected)


def test_smooth_pendry_y_matches_pendry_y_for_linear_intensity():
    """A Savitzky-Golay fit of order >= 1 reproduces a linear signal (and its derivative)
    exactly, so smooth_pendry_y should agree with the unsmoothed pendry_y on linear input."""
    E = np.linspace(3.0, 10.0, 21)
    a, b = 0.4, 2.0
    I = a * E + b
    dE = np.gradient(E)
    V0i_sq = E ** (2 / 3)

    Y_smooth = cluster.smooth_pendry_y(I, dE, V0i_sq, wl=7, p=3)
    Y_plain = cluster.pendry_y(I, dE, V0i_sq)

    np.testing.assert_allclose(Y_smooth, Y_plain, atol=1e-8)


def test_pendryfy_matches_pendry_y_applied_per_row():
    energy = np.linspace(3.0, 10.0, 15)
    X = np.vstack([np.linspace(1, 5, 15), np.linspace(2, 1, 15)])

    PY = cluster.pendryfy(X, energy=energy)

    dE = np.gradient(energy)
    V0i_sq = energy ** (2 / 3)
    expected = np.array([cluster.pendry_y(row, dE, V0i_sq) for row in X])
    np.testing.assert_allclose(PY, expected)


def test_normalize_divides_by_trapz_integral():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 0.0]])
    Xn, integrals = cluster.normalize(X)
    expected_integrals = np.trapz(X, axis=1)
    np.testing.assert_allclose(integrals, expected_integrals)
    np.testing.assert_allclose(Xn, X / expected_integrals[:, None])


def test_denormalize_inverts_normalize():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 0.5]])
    Xn, integrals = cluster.normalize(X)
    X2 = cluster.denormalize(Xn, integrals)
    np.testing.assert_allclose(X2, X)


def test_extract_ivs_returns_per_cluster_mean():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    labels = np.array([0, 0, 1, 1])
    means = cluster.extract_IVs(X, labels)
    np.testing.assert_allclose(means[0], X[:2].mean(axis=0))
    np.testing.assert_allclose(means[1], X[2:].mean(axis=0))


def test_stack2vectors_matches_hand_sliced_matrix(synthetic_leem_stack):
    stack = synthetic_leem_stack(n=3, shape=(10, 12))
    X, h, w = cluster.stack2vectors(stack, mask_outer=0.2)
    dy, dx = int(0.2 * 10), int(0.2 * 12)
    expected = np.array([img.image[dy:-dy, dx:-dx].flatten() for img in stack]).T
    np.testing.assert_allclose(X, expected)
    assert (h, w) == (10 - 2 * dy, 12 - 2 * dx)


def test_stack2vectors_mask_outer_zero_keeps_full_frame(synthetic_leem_stack):
    stack = synthetic_leem_stack(n=3, shape=(10, 12))
    X, h, w = cluster.stack2vectors(stack, mask_outer=0)
    expected = np.array([img.image.flatten() for img in stack]).T
    np.testing.assert_allclose(X, expected)
    assert (h, w) == (10, 12)


def test_vectors2stack_round_trips_stack2vectors(synthetic_leem_stack):
    stack = synthetic_leem_stack(n=3, shape=(10, 12))
    X, h, w = cluster.stack2vectors(stack, mask_outer=0)
    rebuilt = cluster.vectors2stack(X, h, w)
    for original, new in zip(stack, rebuilt):
        np.testing.assert_allclose(original.image, new.image)


### --- component_analysis / cluster_analysis branches ----------------------------------


def test_component_analysis_unknown_algorithm_raises_key_error():
    X = np.zeros((4, 3))
    with pytest.raises(KeyError):
        cluster.component_analysis(X, algorithm="bogus")


def test_cluster_analysis_unknown_registry_key_raises_key_error():
    X = np.zeros((4, 3))
    with pytest.raises(KeyError):
        cluster.cluster_analysis(X, algorithm="bogus-algo")


def test_cluster_analysis_unrecognized_prefix_raises_value_error(monkeypatch):
    """The trailing `else: raise ValueError` (cluster.py:241) only fires for an algorithm
    key that's present in CLUSTERING_DEFAULTS but not prefixed pc-/sk-/nltk- -- none of the
    built-in keys are like that (the lookup above raises KeyError first for anything not in
    the registry at all), so exercise it via a monkeypatched registry entry."""
    fake_defaults = dict(cluster.CLUSTERING_DEFAULTS)
    fake_defaults["unknown-algo"] = (dict, {})
    monkeypatch.setattr(cluster, "CLUSTERING_DEFAULTS", fake_defaults)
    X = np.zeros((4, 3))
    with pytest.raises(ValueError):
        cluster.cluster_analysis(X, algorithm="unknown-algo")


### --- goodness / repeat / save / load --------------------------------------------------


def test_goodness_of_model_pc_kmeans_returns_negative_wce(blob_vectors):
    X, _ = blob_vectors(n_clusters=2, n_per_cluster=10, n_frames=3)
    _, model = cluster.cluster_analysis(X, algorithm="pc-kmeans", n_clusters=2)
    assert cluster.goodness_of_model(model) == -model.get_total_wce()


def test_goodness_of_model_raises_for_unsupported_model():
    with pytest.raises(NotImplementedError):
        cluster.goodness_of_model(object())


def test_repeat_cluster_analysis_picks_best_goodness(monkeypatch):
    """Isolated from real clusterers: repeat_cluster_analysis's own job is picking the
    candidate with the highest goodness_of_model() value across n_iter fits."""

    class FakeModel:
        def __init__(self, goodness):
            self.goodness = goodness

    goodness_values = [1.0, 5.0, 3.0]
    calls = iter(goodness_values)

    def fake_cluster_analysis(X, algorithm="pc-kmeans", **params):  # pylint: disable=unused-argument
        g = next(calls)
        return np.zeros(3), FakeModel(g)

    monkeypatch.setattr(cluster, "cluster_analysis", fake_cluster_analysis)
    monkeypatch.setattr(cluster, "goodness_of_model", lambda model: model.goodness)

    _, model = cluster.repeat_cluster_analysis(np.zeros((3, 2)), n_iter=len(goodness_values))

    assert model.goodness == max(goodness_values)


def test_repeat_cluster_analysis_raises_if_no_model_found():
    """n_iter=0 means the fitting loop never runs, so no candidate model is ever found."""
    with pytest.raises(RuntimeError):
        cluster.repeat_cluster_analysis(np.zeros((3, 2)), n_iter=0)


def test_load_cluster_model_pyclustering_branch(tmp_path, blob_vectors):
    X, _ = blob_vectors(n_clusters=2, n_per_cluster=6, n_frames=3)
    labels, model = cluster.cluster_analysis(X, algorithm="pc-kmeans", n_clusters=2)
    fname = str(tmp_path / "pc.model")
    cluster.save_model(model, fname)

    loaded_labels, _ = cluster.load_cluster_model(fname)

    np.testing.assert_array_equal(
        np.ma.filled(loaded_labels, -1), np.ma.filled(labels, -1)
    )


def test_load_cluster_model_sklearn_branch(tmp_path, blob_vectors):
    X, _ = blob_vectors(n_clusters=2, n_per_cluster=6, n_frames=3)
    labels, model = cluster.cluster_analysis(X, algorithm="sk-kmeans", n_clusters=2)
    fname = str(tmp_path / "sk.model")
    cluster.save_model(model, fname)

    loaded_labels, _ = cluster.load_cluster_model(fname)

    np.testing.assert_array_equal(
        np.ma.filled(loaded_labels, -1), np.ma.filled(labels, -1)
    )


### --- AutoKMeans / OPTICS ---------------------------------------------------------------


def test_autokmeans_sse_matches_independently_fit_kmeans_inertia(blob_vectors):
    """self.sse must be exactly each k's own KMeans(...).inertia_ -- independently refit
    here with the same random_state. preferred="silhouette" sidesteps kneed's elbow
    detection on this test's short (max_clusters=4) SSE curve -- too few points for kneed
    to find an elbow at all (kl.elbow is None), a separate, real bug in AutoKMeans.fit's
    "SSE" branch (cluster.py) that's out of scope here since self.sse is populated either
    way before that branch runs."""
    X, _ = blob_vectors(n_clusters=3, n_per_cluster=20, n_frames=4, spacing=80.0)
    auto = cluster.AutoKMeans(
        max_clusters=4, preferred="silhouette", do_plot=False, n_init=3, random_state=0
    )
    auto.fit(X)

    for k, sse in zip(range(1, 4), auto.sse):
        expected = (
            cluster.sk_cluster.KMeans(n_clusters=k, n_init=3, random_state=0).fit(X).inertia_
        )
        assert sse == pytest.approx(expected)


def test_autokmeans_sse_branch_returns_one_of_the_fitted_models(blob_vectors):
    """preferred="SSE" (the default) runs the kneed-based elbow branch -- exercised here
    with enough points on the curve (max_clusters=6) for kneed to find an elbow at all.
    Not asserted against the true cluster count: kneed's elbow choice is a heuristic that,
    verified above (test_autokmeans_sse_matches_independently_fit_kmeans_inertia) and by
    hand, does not reliably land on the true k even for well-separated blobs -- so this
    only pins that the branch runs and returns a model actually produced by this fit."""
    X, _ = blob_vectors(n_clusters=3, n_per_cluster=20, n_frames=4, spacing=80.0)
    auto = cluster.AutoKMeans(
        max_clusters=6, preferred="SSE", do_plot=False, n_init=3, random_state=0
    )

    best = auto.fit(X)

    assert best in auto.models


def test_autokmeans_silhouette_recovers_known_cluster_count(blob_vectors):
    """Unlike the SSE branch (which goes through kneed's heuristic elbow detection), the
    silhouette branch directly argmaxes self.sil_coef -- verifiable against independently
    recomputed sklearn silhouette scores, and reliable enough on well-separated blobs to
    recover the true cluster count."""
    X, _ = blob_vectors(n_clusters=3, n_per_cluster=20, n_frames=4, spacing=80.0)
    auto = cluster.AutoKMeans(
        max_clusters=6, preferred="silhouette", do_plot=False, n_init=3, random_state=0
    )

    best = auto.fit(X)

    expected_sil = [
        cluster.sk_metrics.silhouette_score(
            X,
            cluster.sk_cluster.KMeans(n_clusters=k, n_init=3, random_state=0).fit(X).labels_,
        )
        for k in range(2, 6)
    ]
    np.testing.assert_allclose(auto.sil_coef, expected_sil)
    assert best.n_clusters == 3 == np.argmax(expected_sil) + 2


def test_remodel_optics_xi_returns_one_label_per_point(blob_vectors):
    X, _ = blob_vectors(n_clusters=2, n_per_cluster=30, n_frames=3, spacing=80.0)
    model = cluster.sk_cluster.OPTICS(min_samples=5).fit(X)
    labels = cluster._remodel_optics(
        model, target="xi", xi=0.05, min_samples=5, min_cluster_size=5
    )
    assert len(labels) == X.shape[0]


def test_remodel_optics_dbscan_returns_one_label_per_point(blob_vectors):
    X, _ = blob_vectors(n_clusters=2, n_per_cluster=30, n_frames=3, spacing=80.0)
    model = cluster.sk_cluster.OPTICS(min_samples=5).fit(X)
    labels = cluster._remodel_optics(model, target="dbscan", eps=20.0)
    assert len(labels) == X.shape[0]


def test_plot_optics_draws_one_series_per_label(blob_vectors):
    X, _ = blob_vectors(n_clusters=2, n_per_cluster=30, n_frames=3, spacing=80.0)
    model = cluster.sk_cluster.OPTICS(min_samples=5).fit(X)
    _, ax = plt.subplots()
    cluster.plot_optics(model, ax=ax)
    # one line per label found (>=1) plus one for the noise/-1 points
    assert len(ax.get_lines()) >= 1
    plt.close("all")


### --- Plot functions ---------------------------------------------------------------------


def test_plot_clustermap_draws_one_image():
    clustermap = np.array([[0, 1], [1, 2]])
    _, ax = plt.subplots()
    cluster.plot_clustermap(clustermap, ax=ax)
    assert len(ax.get_images()) == 1
    plt.close("all")


def test_plot_clustermap_ofile_saves_a_file(tmp_path):
    clustermap = np.array([[0, 1], [1, 2]])
    ofile = tmp_path / "clustermap.png"
    cluster.plot_clustermap(clustermap, ofile=str(ofile))
    assert ofile.exists()
    plt.close("all")


def test_plot_raw_draws_cropped_image(synthetic_leem_stack):
    stack = synthetic_leem_stack(n=1, shape=(10, 12))
    img = stack[0]
    _, ax = plt.subplots()
    cluster.plot_raw(img, mask_outer=0.2, ax=ax)
    drawn = ax.get_images()[0].get_array()
    dy, dx = int(0.2 * 10), int(0.2 * 12)
    np.testing.assert_allclose(drawn, img.image[dy:-dy, dx:-dx])
    plt.close("all")


def test_plot_raw_mask_outer_zero_keeps_full_frame(synthetic_leem_stack):
    stack = synthetic_leem_stack(n=1, shape=(10, 12))
    img = stack[0]
    _, ax = plt.subplots()
    cluster.plot_raw(img, mask_outer=0, ax=ax)
    drawn = ax.get_images()[0].get_array()
    np.testing.assert_allclose(drawn, img.image)
    plt.close("all")


def test_plot_ivs_draws_one_line_per_cluster(synthetic_leem_stack):
    stack = synthetic_leem_stack(n=6, shape=(10, 12), energy=np.linspace(1, 6, 6))
    X, _, _ = cluster.stack2vectors(stack, mask_outer=0.2)
    labels = np.zeros(X.shape[0])
    labels[: X.shape[0] // 2] = 1
    _, ax = plt.subplots()
    cluster.plot_IVs(stack, labels, ax=ax, mask_outer=0.2)
    assert len(ax.get_lines()) == 2
    plt.close("all")


def test_plot_ivs_uses_callable_cmap_and_legend_and_saves_ofile(synthetic_leem_stack, tmp_path):
    stack = synthetic_leem_stack(n=6, shape=(10, 12), energy=np.linspace(1, 6, 6))
    X, _, _ = cluster.stack2vectors(stack, mask_outer=0.2)
    labels = np.zeros(X.shape[0])
    labels[: X.shape[0] // 2] = 1
    cmap, legend = cluster.make_cmap({0: ("a", "#111111"), 1: ("b", "#222222")}, labels)
    ofile = tmp_path / "ivs.png"
    ax = cluster.plot_IVs(
        stack, labels, mask_outer=0.2, cmap=cmap, legend=legend, ofile=str(ofile)
    )
    assert len(ax.get_lines()) == 2
    assert ax.get_legend() is not None
    assert ofile.exists()
    plt.close("all")


def test_plot_ivs2_draws_one_line_per_curve():
    energy = np.linspace(0, 10, 5)
    IV_curves = [np.sin(energy), np.cos(energy), energy]
    _, ax = plt.subplots()
    cluster.plot_IVs2(energy, IV_curves, ax=ax)
    assert len(ax.get_lines()) == 3
    plt.close("all")


if __name__ == "__main__":
    main()


### --- Audit regressions: noise handling, defaults, elbow, unsupported algorithms ---------


def _blobs_with_noise(blob_vectors):
    """Two tight blobs (20 + 10 pixels) plus 3 isolated outliers DBSCAN calls noise."""
    X, _ = blob_vectors(n_clusters=2, n_per_cluster=10, n_frames=3, spacing=50.0)
    X = np.vstack([X, X[:10] + 0.0])  # blob 0 doubled -> 20 vs 10 pixels
    outliers = np.array([[500.0] * 3, [-500.0] * 3, [900.0, -900.0, 900.0]])
    return np.vstack([X, outliers])


def test_sort_labels_masks_noise_instead_of_ranking_it():
    labels = np.array([5, 5, 5, 5, 2, 2, 7, -1, -1, -1])  # noise is the most frequent "label"
    out = cluster.sort_labels(labels)
    assert list(np.ma.getmaskarray(out)) == [False] * 7 + [True] * 3
    np.testing.assert_array_equal(out[:7].data, [2, 2, 2, 2, 1, 1, 0])
    np.testing.assert_array_equal(labels[-3:], [-1, -1, -1])  # input untouched


def test_cluster_analysis_masks_noise_and_keeps_model_labels(blob_vectors):
    X = _blobs_with_noise(blob_vectors)
    labels, model = cluster.cluster_analysis(X, "sk-dbscan", eps=1.0, min_samples=3)

    assert (model.labels_ == -1).sum() == 3  # model.labels_ not mutated
    assert list(np.ma.getmaskarray(labels)) == list(model.labels_ == -1)
    # noise must not inflate cluster 0 (the smallest real cluster, 10 pixels)
    assert np.count_nonzero(labels == 0) == 10


def test_load_cluster_model_sklearn_masks_noise_and_keeps_model_labels(tmp_path, blob_vectors):
    X = _blobs_with_noise(blob_vectors)
    _, model = cluster.cluster_analysis(X, "sk-dbscan", eps=1.0, min_samples=3)
    fname = str(tmp_path / "db.model")
    cluster.save_model(model, fname)

    labels, loaded = cluster.load_cluster_model(fname)

    assert (loaded.labels_ == -1).sum() == 3
    assert np.ma.getmaskarray(labels).sum() == 3
    assert np.count_nonzero(labels == 0) == 10


def test_component_analysis_does_not_mutate_defaults():
    X = np.random.default_rng(0).normal(size=(30, 12))
    before = cluster.COMPONENTS_DEFAULTS["pca"][1]["n_components"]
    cluster.component_analysis(X, "pca", n_components=3)
    assert cluster.COMPONENTS_DEFAULTS["pca"][1]["n_components"] == before
    _, _, model = cluster.component_analysis(X, "pca")
    assert model.n_components_ == before


def test_elbow_cluster_analysis_finds_true_cluster_count_of_four_blobs():
    """-WCE rises concavely with k; the knee of that curve is the true k (4), whereas
    curve='convex' picks the sweep end (15)."""
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
    X = np.vstack([c + rng.normal(scale=0.5, size=(60, 2)) for c in centers])

    labels, _ = cluster.elbow_cluster_analysis(X, "pc-kmeans", start=2, end=15)

    assert len(np.unique(labels)) == 4


def test_cluster_analysis_sk_kmeans_auto_runs(blob_vectors):
    X, _ = blob_vectors(n_clusters=3, n_per_cluster=20, n_frames=4, spacing=80.0)
    labels, _ = cluster.cluster_analysis(X, "sk-kmeans-auto", do_plot=False)
    assert len(labels) == X.shape[0]


def test_cluster_analysis_pc_kmeans_elbow_raises_value_error(blob_vectors):
    X, _ = blob_vectors(n_clusters=2, n_per_cluster=10, n_frames=3)
    with pytest.raises(ValueError, match="elbow_cluster_analysis"):
        cluster.cluster_analysis(X, "pc-kmeans-elbow")
