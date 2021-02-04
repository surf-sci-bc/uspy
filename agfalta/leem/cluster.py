"""LEEM cluster analysis tools."""
# pylint:disable=invalid-name
# pylint:disable=missing-docstring

import os
from pathlib import Path
import pickle
import copy
import inspect

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
# clustering via scikit
from sklearn import decomposition as sk_decomposition
from sklearn import cluster as sk_cluster
from sklearn import mixture as sk_mixture
# clustering via nltk
from nltk import cluster as nltk_cluster
# clustering via pyclustering
from pyclustering.cluster.kmeans import kmeans as pc_kmeans
from pyclustering.cluster.kmedoids import kmedoids as pc_kmedoids
from pyclustering.cluster.xmeans import xmeans as pc_xmeans
from pyclustering.cluster.cure import cure as pc_cure
from pyclustering.cluster.agglomerative import agglomerative as pc_agglomerative
from pyclustering.cluster.center_initializer import (
    kmeans_plusplus_initializer, random_center_initializer)
from pyclustering.cluster.elbow import elbow as pc_elbow
from pyclustering.utils import metric as pc_metric
# measuring clustering quality
from sklearn import metrics as sk_metrics     # pylint: disable=ungrouped-imports
import kneed

from agfalta.leem.base import LEEMStack
from agfalta.utility import timing_notification, progress_bar, silence


def stack2vectors(stack, mask_outer=0.2):
    _, h0, w0 = len(stack), stack[0].data.shape[0], stack[0].data.shape[1]
    dy, dx = (int(mask_outer * h0), int(mask_outer * w0))
    h, w = (h0 - 2 * dy, w0 - 2 * dx)
    if mask_outer == 0:
        data = np.array([img.data.flatten() for img in stack]).T
    else:
        data = np.array([img.data[dy:-dy, dx:-dx].flatten() for img in stack]).T
    return data, h, w

def vectors2stack(X, h, w, mask_outer=0):
    h, w = (h - 2 * int(mask_outer * h), w - 2 * int(mask_outer * w))
    data = (X.T).reshape(X.shape[1], h, w)
    return LEEMStack(data)

def enforce_clustershape(mask_outer=0):
    def enforcer(wrapped):
        def wrapper(X, *args, **kwargs):
            if isinstance(X, LEEMStack):
                X = stack2vectors(X, mask_outer=mask_outer)
            return wrapped(X, *args, **kwargs)
        return wrapper
    return enforcer

def sort_labels(labels):
    labellist = list(np.unique(labels))
    labellist.sort(key=lambda l: np.count_nonzero(labels == l))
    sorted_labels = np.zeros_like(labels) - 1
    for i, ul in enumerate(labellist):
        sorted_labels[labels == ul] = i
    sorted_labels[sorted_labels < 0] = np.ma.masked
    return sorted_labels

def combine_labels(labels, combinations):
    combined_labels = np.copy(labels)
    for combination in combinations:
        for label in combination:
            combined_labels[labels == label] = combination[0]
    return combined_labels


@enforce_clustershape(0)
@timing_notification("component analysis")
def component_analysis(X, algorithm="pca", **params_):
    constructor, params = COMPONENTS_DEFAULTS[algorithm]
    params.update(params_)
    model = constructor(**params)
    model.fit(X)

    if algorithm == "pca":
        variance_list = ", ".join([f"{v*100:.2f}%" for v in model.explained_variance_ratio_])
        if len(model.explained_variance_ratio_) > 3:
            variance_list += ", ..."
        print(f"{algorithm.upper()}: {model.n_components_} components explain "
              f"{sum(model.explained_variance_ratio_)*100:.2f}% of variance: ({variance_list})")
    return model.transform, model.inverse_transform, model


def _pendry_distance(p1, p2):
    return np.divide(np.sum(np.square(p1 - p2)),
                     np.sum(np.square(p1) + np.square(p2)))
PC_METRICS = {
    None: pc_metric.distance_metric(pc_metric.type_metric.EUCLIDEAN),
    "euclidean": pc_metric.distance_metric(pc_metric.type_metric.EUCLIDEAN),
    "euclidean_square": pc_metric.distance_metric(pc_metric.type_metric.EUCLIDEAN_SQUARE),
    "canberra": pc_metric.distance_metric(pc_metric.type_metric.CANBERRA),
    "chi_square": pc_metric.distance_metric(pc_metric.type_metric.CHI_SQUARE),
    "pendry": pc_metric.distance_metric(
        pc_metric.type_metric.USER_DEFINED, func=_pendry_distance),
}
PC_INITIALIZERS = {
    None: kmeans_plusplus_initializer,
    "k-means++": kmeans_plusplus_initializer,
    "random": random_center_initializer,
}

@enforce_clustershape(0)
@timing_notification("cluster analysis")
def cluster_analysis(X, algorithm="pc-kmeans", **params_):
    """
    There are different models to choose from. The following table lists them
    and on the right, the keyword arguments that they take are listed with the
    default values.
    For more information look at the scikit-learn (sk), pyclustering (pc) and
    ntltk packages online.

    "sk-birch":         "threshold": 0.1, "n_clusters": 15
    "sk-optics":        "min_samples": 0.03, "xi": 0.00005, 
                        "min_cluster_size": 0.01, "n_jobs": 1
    "sk-dbscan":        "eps": 0.06, "min_samples": 400
    "sk-kmeans":        "init": "k-means++", "n_clusters": N_CLUSTERS, 
                        "n_init": 10, "max_iter": 300
    "sk-bgm":           "n_components": N_CLUSTERS, "n_init": 1, "max_iter": 200
    "sk-kmeans-auto":   "init": "k-means++", "n_init": 5, "max_iter": 300
    "pc-kmeans":        "init": "k-means++", "n_clusters": N_CLUSTERS
    "pc-kmeans-elbow":  "init": "k-means++"
    "pc-kmeans-iter":   "init": "random"
    "pc-kmedoids":      "init": "k-means++", "n_clusters": N_CLUSTERS
    "pc-xmeans":        "init": "k-means++", "n_clusters": N_CLUSTERS
    "pc-cure":          "init": "n_clusters", "n_clusters": N_CLUSTERS
    "pc-agglomerative": "init": "n_clusters", "n_clusters": N_CLUSTERS
    "nltk-kmeans":      "num_means": N_CLUSTERS, "repeats": 10,
                        "distance": nltk_cluster.euclidean_distance, 
    """
    # pylint: disable=too-many-locals
    algorithm = algorithm.lower()
    constructor, params = copy.deepcopy(CLUSTERING_DEFAULTS[algorithm])
    params.update(params_)
    metric_type = params.pop("metric", "euclidean")
    initializer = PC_INITIALIZERS[params.pop("init", None)]
    n_clusters = params.get("n_clusters", N_CLUSTERS)

    for kw in params.copy():
        if kw not in inspect.signature(constructor).parameters.keys():
            params.pop(kw)

    if algorithm.startswith("nltk-"):
        model = constructor(**params)
        labels = model.cluster(X, True)

    if algorithm.startswith("pc-"):
        if algorithm.startswith("pc-k") or algorithm == "pc-xmeans":
            if callable(metric_type):
                metric = pc_metric.distance_metric(
                    pc_metric.type_metric.USER_DEFINED, func=metric_type)
            else:
                metric = PC_METRICS[metric_type]
            use_indices = "medoids" in algorithm
            initial_centers = initializer(X, n_clusters).initialize(return_index=use_indices)
            model = constructor(X, initial_centers, metric=metric, **params)
        else:
            model = constructor(X, n_clusters, **params)
        model.process()
        clusters = model.get_clusters()
        labels = np.zeros((X.shape[0],)) - 1
        for i, cluster in enumerate(clusters):
            labels[cluster] = i

    elif algorithm.startswith("sk-"):
        model = constructor(**params)
        labels = model.fit_predict(X)

    else:
        raise ValueError(f"Unknown algorithm {algorithm}")

    print(f"{algorithm.upper()}: Found {len(set(labels))} clusters")
    labels[labels < 0] = np.ma.masked
    return sort_labels(labels), model


def goodness_of_model(model):
    if isinstance(model, pc_kmeans):
        return -model.get_total_wce()   # minus is important
    raise NotImplementedError("Repeat cluster analysis only possible for PC-KMeans")

@timing_notification("repeat cluster analysis")
def repeat_cluster_analysis(X, algorithm="pc-kmeans", n_iter=10, **params_):
    goodness = -np.inf   # maximize this
    model = None
    labels = None
    for i in range(n_iter):
        with silence():
            labels_cand, model_cand = cluster_analysis(X, algorithm=algorithm, **params_)
        goodness_cand = goodness_of_model(model_cand)
        if goodness_cand > goodness:
            model = model_cand
            goodness = goodness_cand
            labels = labels_cand
            print(f"Found better model, goodness = {goodness}")
        print(f"\033[KDid model iter: {i}", end="\r")
    if model is None:
        raise RuntimeError("Could not get a cluster model")
    return labels, model

@timing_notification("elbow cluster analysis")
def elbow_cluster_analysis(X, algorithm="pc-kmeans", start=2, end=15, **params_):
    labels_list = []
    models = []
    for n_clusters in range(start, end + 1):
        with silence():
            labels, model = cluster_analysis(X, algorithm=algorithm, **params_)
        labels_list.append(labels)
        models.append(model)
        print(f"\033[KDid model with n_clusters={n_clusters}", end="\r")
    goodnesses = [goodness_of_model(model) for model in models]
    kl = kneed.KneeLocator(
        range(start, end + 1), goodnesses, curve="convex", direction="increasing")
    n_clusters = kl.elbow
    print(f"ELBOW-{algorithm.upper()}: Found elbow at {n_clusters} clusters "
          f"(goodness: {goodnesses[n_clusters - start]:.2f})")
    return labels_list[n_clusters - start], models[n_clusters - start]



@enforce_clustershape(0)
def pendryfy(X, energy=None, smoothing_params=None):
    if energy is None or np.isnan(energy).any():
        print("WARNING: NaN values in energy, guessing 3eV + 0.2eV * idx")
        energy = np.linspace(3.0, 3.0 + len(energy) * 0.2, len(energy))
    PY = np.zeros_like(X)
    V0i_sq = energy**(2/3)      # square of energy**(1/3)
    dE = np.gradient(energy)
    for i, I in enumerate(progress_bar(X, "Pendryfying...")):
        if smoothing_params:
            PY[i] = smooth_pendry_y(I, dE, V0i_sq, **smoothing_params)
        else:
            PY[i] = pendry_y(I, dE, V0i_sq)
    return PY

def pendry_y(I, dE, V0i_sq, eps=1e-5):
    L = np.gradient(I) / dE / np.clip(I, eps, None)
    Y = L / (1 + L**2 * V0i_sq)
    return Y

def smooth_pendry_y(I, dE, V0i_sq, eps=1e-5, wl=17, p=4, both=True):
    """
    I: intensity,
    dE: gradient of energy
    V0i_sq: squared elelctron self energy
    eps: clipping value to avoid division by 0
    wl: Savitzky-Golay filter window length
    p: Savitzky-Golay filter polynom order
    """
    # pylint: disable=too-many-arguments
    if both:
        I = savgol_filter(I, window_length=wl, polyorder=p)
    dI = savgol_filter(I, window_length=wl, polyorder=p, deriv=1)
    L = dI / dE / np.clip(I, eps, None)
    Y = L / (1 + L**2 * V0i_sq)
    return Y

@enforce_clustershape(0)
def smoothen(X, energy=None, wl=17, p=4):
    if energy is None or np.isnan(energy).any():
        print("WARNING: NaN values in energy, guessing 3eV + 0.2eV * idx")
        energy = np.linspace(3.0, 3.0 + len(energy) * 0.2, len(energy))
    S = np.zeros_like(X)
    for i, spectrum in enumerate(progress_bar(X, "Smoothing...")):
        S[i] = savgol_filter(spectrum, window_length=wl, polyorder=p)
    return S

@enforce_clustershape(0)
@timing_notification("normalization")
def normalize(X):
    Xn = np.zeros_like(X)
    integrals = np.zeros_like(X[:, 0])
    for i, spectrum in enumerate(X):
        integrals[i] = np.trapz(X[i, :])
        Xn[i, :] = spectrum / integrals[i]
    return Xn, integrals

def denormalize(Xn, integrals):
    X = np.zeros_like(Xn)
    for i, _ in enumerate(Xn):
        X[i, :] = Xn[i, :] * integrals[i]
    return X

@enforce_clustershape(0)
def extract_IVs(X, labels):
    IV_means = []
    for klass in np.unique(labels):
        IVs = X[labels == klass, :]
        IV_means.append(IVs.mean(axis=0))
    return np.array(IV_means)

def save_model(model, fname):
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    with Path(fname).open("wb") as pfile:
        try:
            pickle.dump(model, pfile)
            print(f"Saved model to {fname}")
        except RecursionError:
            print("Did not save model due to recursion error (Too big?).")


def load_pca_model(fname):
    with Path(fname).open("rb") as pfile:
        model = pickle.load(pfile)
    print(f"Loaded PCA model from {fname}")
    return model.transform, model.inverse_transform, model

def load_cluster_model(fname):
    with Path(fname).open("rb") as pfile:
        model = pickle.load(pfile)
    labels = model.labels_
    labels[labels < 0] = np.ma.masked
    print(f"Loaded cluster model from {fname}")
    return sort_labels(labels), model

def plot_clustermap(clustermap, ax=None, out_prefix=None, cmap="seismic"):
    """Plot 2d numpy array with integer values to colormap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(clustermap, interpolation="none", origin="upper", cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    if out_prefix is not None:
        plt.imsave(f"{out_prefix}_clustermap.png", clustermap, origin="upper", cmap=cmap)

def plot_IVs(stack, labels, ax=None, mask_outer=0.2, cmap="seismic"):
    """Plot IV curves."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    X, _, _ = stack2vectors(stack, mask_outer=mask_outer)
    IV_curves = extract_IVs(X, labels)
    for i, IV_curve in enumerate(IV_curves):
        ax.plot(stack.energy, IV_curve, color=plt.get_cmap(cmap)(i / len(IV_curves)))
    return ax

def plot_IVs2(energy, IV_curves, ax=None, cmap="seismic"):
    """Plot IV curves."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for i, IV_curve in enumerate(IV_curves):
        ax.plot(energy, IV_curve, color=plt.get_cmap(cmap)(i / len(IV_curves)))

def plot_optics(model, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    N = len(set(model.labels_))
    colors = ["g.", "r.", "b.", "y.", "c."] * 10
    space = np.arange(len(model.labels_))
    reachability = model.reachability_[model.ordering_]
    for klass, color in zip(range(N), colors[:N]):
        Xk = space[model.labels_ == klass]
        Rk = reachability[model.labels_ == klass]
        ax.plot(Xk, Rk, color, alpha=0.3)
    ax.plot(space[model.labels_ == -1], reachability[model.labels_ == -1], "k.", alpha=0.3)

def _remodel_optics(model, target="xi", **kwargs):
    if target == "xi":
        xi = kwargs.get("xi", 0.03)
        min_cluster_size = kwargs.get("min_cluster_size", 0.01)
        min_samples = kwargs.get("min_samples", 0.03)
        labels = sk_cluster.cluster_optics_xi(
            min_samples=min_samples, min_cluster_size=min_cluster_size, xi=xi,
            reachability=model.reachability_, predecessor=model.predecessor_,
            ordering=model.ordering_
        )
    else:
        eps = kwargs.get("eps", 0.5)
        labels = sk_cluster.cluster_optics_dbscan(
            eps=eps,
            reachability=model.reachability_, core_distances=model.core_distances_,
            ordering=model.ordering_,
        )
    return sort_labels(labels)


class AutoKMeans:
    # pylint: disable=too-few-public-methods
    def __init__(self, *args, max_clusters=15, preferred="SSE", do_plot=True, **kwargs):
        self.preferred = preferred
        self.do_plot = do_plot
        self.sse = []
        self.sil_coef = []
        self.max_clusters = max_clusters
        self.models = []
        for k in range(1, max_clusters):
            self.models.append(sk_cluster.KMeans(*args, n_clusters=k, **kwargs))

    def fit(self, X):
        for k, model in enumerate(self.models, 1):
            model.fit(X)
            print(f"AutoKMeans: Finished cycle {k}")
            self.sse.append(model.inertia_)
            if k > 1:
                self.sil_coef.append(sk_metrics.silhouette_score(X, model.labels_))
        if self.do_plot:
            _, ax = plt.subplots()
            ax.plot(
                range(1, self.max_clusters),
                np.array(self.sse) / max(self.sse), label="sse"
            )
            ax.plot(
                range(2, self.max_clusters),
                np.array(self.sil_coef) / max(self.sil_coef), label="sil_coef"
            )
            ax.legend()
        kl = kneed.KneeLocator(
            range(1, self.max_clusters), self.sse,
            curve="convex", direction="decreasing"
        )
        if self.preferred == "SSE":
            print(f"SSE detection finds {kl.elbow} clusters "
                  f"(silhouette coefficients say {np.argmax(self.sil_coef) + 2} clusters)")
            return self.models[kl.elbow - 1]
        print(f"Silhouette coefficients find {np.argmax(self.sil_coef) + 2} clusters"
              f"(SSE detection says {kl.elbow} clusters)")
        return self.models[np.argmax(self.sil_coef) + 2 - 1]


# defaults
N_COMPONENTS = 7
N_CLUSTERS = 8
COMPONENTS_DEFAULTS = {
    "pca":      (sk_decomposition.PCA, {
        "n_components": N_COMPONENTS
    }),
    "nmf":      (sk_decomposition.NMF, {
        "n_components": N_COMPONENTS, "init": "nndsvda"
    })
}
CLUSTERING_DEFAULTS = {
    "sk-birch":     (sk_cluster.Birch, {"threshold": 0.1, "n_clusters": 15}),
    "sk-optics":    (sk_cluster.OPTICS, {
        "min_samples": 0.03, "xi": 0.00005, "min_cluster_size": 0.01, "n_jobs": 1}),
    "sk-dbscan":    (sk_cluster.DBSCAN, {"eps": 0.06, "min_samples": 400}),
    "sk-kmeans":    (sk_cluster.KMeans, {
        "init": "k-means++", "n_clusters": N_CLUSTERS, "n_init": 10, "max_iter": 300}),
    "sk-bgm":       (sk_mixture.BayesianGaussianMixture, {
        "n_components": N_CLUSTERS, "n_init": 1, "max_iter": 200}),
    "sk-kmeans-auto":  (AutoKMeans, {"init": "k-means++", "n_init": 5, "max_iter": 300}),

    "pc-kmeans":    (pc_kmeans, {"init": "k-means++", "n_clusters": N_CLUSTERS}),
    "pc-kmeans-elbow":  (pc_elbow, {"init": "k-means++"}),
    "pc-kmeans-iter":  (pc_kmeans, {"init": "random"}),
    "pc-kmedoids":  (pc_kmedoids, {"init": "k-means++", "n_clusters": N_CLUSTERS}),
    "pc-xmeans":  (pc_xmeans, {"init": "k-means++", "n_clusters": N_CLUSTERS}),
    "pc-cure":  (pc_cure, {"init": "n_clusters", "n_clusters": N_CLUSTERS}),
    "pc-agglomerative":  (pc_agglomerative, {"init": "n_clusters", "n_clusters": N_CLUSTERS}),

    "nltk-kmeans":  (nltk_cluster.KMeansClusterer, {
        "num_means": N_CLUSTERS, "distance": nltk_cluster.euclidean_distance, "repeats": 10}),
}
