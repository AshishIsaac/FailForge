"""Dimensionality reduction + density clustering of failure embeddings.

UMAP (or t-SNE / PCA) maps the features to a low-dimensional space where HDBSCAN finds dense groups
without being told how many there are, and leaves outliers unassigned (label -1) instead of forcing
every odd error into some cluster. A separate 2-D projection is made for the plots.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

log = logging.getLogger(__name__)


def reduce(x: np.ndarray, method: str, n_components: int, n_neighbors: int = 30, min_dist: float = 0.0,
           seed: int = 0) -> np.ndarray:
    n = len(x)
    if n <= n_components + 2:
        return np.pad(x[:, :n_components], ((0, 0), (0, max(0, n_components - x.shape[1]))))
    if method == "umap":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import umap

                return umap.UMAP(n_components=n_components, n_neighbors=min(n_neighbors, n - 1), min_dist=min_dist,
                                 metric="euclidean", random_state=seed, n_jobs=1).fit_transform(x)
        except ImportError:
            log.warning("umap-learn not installed: falling back to PCA")
            method = "pca"
    from sklearn.decomposition import PCA

    if method == "tsne":
        from sklearn.manifold import TSNE

        pre = PCA(n_components=min(50, x.shape[1], n - 1), random_state=seed).fit_transform(x)
        return TSNE(n_components=min(n_components, 3), perplexity=min(30.0, (n - 1) / 3), init="pca",
                    random_state=seed).fit_transform(pre)
    return PCA(n_components=min(n_components, x.shape[1], n), random_state=seed).fit_transform(x)


def hdbscan(z: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    from sklearn.cluster import HDBSCAN

    if len(z) < max(min_cluster_size, 2):
        return np.full(len(z), -1)
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                   cluster_selection_method="eom", copy=True).fit_predict(z)


def cluster(features: np.ndarray, method: str = "umap", n_components: int = 10, n_neighbors: int = 30,
            min_dist: float = 0.0, min_cluster_size: int = 40, min_samples: int = 10, seed: int = 0
            ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, 2-D coordinates for plotting)."""
    z = reduce(features, method, n_components, n_neighbors, min_dist, seed)
    labels = hdbscan(z, min_cluster_size, min_samples)
    xy = z[:, :2] if method == "tsne" or n_components == 2 else reduce(features, method, 2, n_neighbors, 0.1, seed)
    return labels, np.asarray(xy, dtype=np.float32)
