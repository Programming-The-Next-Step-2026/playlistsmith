"""Clustering subpackage.

This package turns a feature DataFrame (as produced by
:func:`playlistsmith.features.extract`) into cluster labels suitable for
playlist export. The pipeline is sequenced as:

1. :mod:`.preprocess` — logit / log transforms, median imputation,
   standard-scaling, and a record of what happened
   (:class:`.preprocess.TransformLog`).
2. :mod:`.algorithms` — the actual clustering (GMM is the default;
   K-Means and HDBSCAN are alternatives).
3. :mod:`.interpret` — short, deduplicated human-readable labels per
   cluster (e.g. ``"high energy, fast tempo, low valence"``).
"""

from __future__ import annotations

from playlistsmith.cluster.algorithms import (
    ClusteringResult,
    fit_gmm,
    fit_hdbscan,
    fit_kmeans,
)
from playlistsmith.cluster.interpret import describe_clusters
from playlistsmith.cluster.preprocess import (
    BOUNDED_FEATURES,
    LOG_FEATURES,
    TransformLog,
    prepare_matrix,
)
from playlistsmith.cluster.public import (
    ClusterDiagnostics,
    ClusterPipelineResult,
    cluster,
)

__all__ = [
    "BOUNDED_FEATURES",
    "ClusterDiagnostics",
    "ClusterPipelineResult",
    "ClusteringResult",
    "LOG_FEATURES",
    "TransformLog",
    "cluster",
    "describe_clusters",
    "fit_gmm",
    "fit_hdbscan",
    "fit_kmeans",
    "prepare_matrix",
]
