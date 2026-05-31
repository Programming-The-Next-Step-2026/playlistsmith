"""Human-readable summaries of clusters.

A bare cluster index (``0``, ``1``, ...) is unhelpful for a playlist
export: the user wants to know *why* a playlist hangs together. This
module turns a :class:`~playlistsmith.cluster.algorithms.ClusteringResult`
plus the modelling matrix into a short, comma-separated description for
each cluster — for example::

    "high energy, fast tempo, low valence"

The procedure is documented in
``plans/clustering_analysis_revised.md`` Section 4:

1. Compute the per-cluster z-profile. Because the modelling matrix is
   already z-scored, the z-profile is just the cluster's mean vector.
2. Rank features by ``|z|``.
3. Deduplicate correlated features (``|ρ| > 0.6``) so the summary does
   not say the same thing three different ways. ``energy`` and
   ``loudness`` typically correlate ~0.7–0.8 in the ReccoBeats space,
   so this matters in practice.
4. Render a short qualified string. ``tempo`` and ``loudness`` get
   genre-appropriate qualifiers; everything else uses ``high``/``low``.
"""

from __future__ import annotations

import pandas as pd

from playlistsmith.cluster.algorithms import ClusteringResult

__all__ = ["describe_clusters"]

#: Features with ``|z|`` below this threshold are considered too close
#: to the global mean to be a defining characteristic of the cluster.
_NEAR_ZERO_Z_THRESHOLD: float = 0.5

#: Maximum number of features in a single summary string. Three keeps
#: the label legible; the plan picks this number explicitly.
_MAX_FEATURES_IN_SUMMARY: int = 3

#: Absolute Pearson correlation above which two features are treated as
#: redundant for label purposes (e.g. energy ↔ loudness).
_CORRELATION_DEDUP_THRESHOLD: float = 0.6

#: Per-feature qualifier overrides. Anything not in this mapping uses
#: the default ``high X`` / ``low X``.
_QUALIFIER_OVERRIDES: dict[str, tuple[str, str]] = {
    "tempo": ("fast tempo", "slow tempo"),
    "loudness": ("loud", "quiet"),
}


def _select_top_features(
    z_profile: pd.Series, correlations: pd.DataFrame
) -> list[str]:
    """Pick up to three defining features for a cluster.

    Features are walked in order of decreasing ``|z|`` and added to the
    label only if (a) their ``|z|`` is above
    :data:`_NEAR_ZERO_Z_THRESHOLD` and (b) they do not correlate with
    any already-selected feature at
    :data:`_CORRELATION_DEDUP_THRESHOLD` or stronger.

    Args:
        z_profile: One cluster's z-profile across the nine features.
        correlations: Pairwise feature correlations on the modelling
            matrix (output of ``X.corr()``).

    Returns:
        Feature names in summary order; ``[]`` if every feature is near
        the global mean.
    """
    ranked = z_profile.abs().sort_values(ascending=False).index.tolist()
    chosen: list[str] = []
    for feature in ranked:
        if len(chosen) >= _MAX_FEATURES_IN_SUMMARY:
            break
        if abs(z_profile[feature]) < _NEAR_ZERO_Z_THRESHOLD:
            break  # The rest are even smaller; nothing more to add.
        if any(
            abs(correlations.loc[feature, other]) > _CORRELATION_DEDUP_THRESHOLD
            for other in chosen
        ):
            continue
        chosen.append(feature)
    return chosen


def _qualify(feature: str, z: float) -> str:
    """Render a single ``feature, z`` pair as a human-readable phrase.

    Args:
        feature: The feature column name.
        z: The cluster's z-profile value for that feature (sign matters).

    Returns:
        E.g. ``"high energy"``, ``"slow tempo"`` or ``"quiet"``.
    """
    if feature in _QUALIFIER_OVERRIDES:
        high_label, low_label = _QUALIFIER_OVERRIDES[feature]
        return high_label if z >= 0 else low_label
    return f"{'high' if z >= 0 else 'low'} {feature}"


def describe_clusters(
    result: ClusteringResult, X: pd.DataFrame
) -> pd.DataFrame:
    """Summarise each cluster as a short human-readable string.

    Builds the z-profile from ``result.feature_means_per_cluster`` (the
    modelling matrix is already z-scored, so cluster means *are* the
    z-profile), then picks up to three uncorrelated, non-trivial top
    features per cluster and renders them as ``"high energy, fast
    tempo, low valence"``-style summaries.

    Args:
        result: The clustering outcome to interpret.
        X: The transformed-and-scaled modelling matrix that was fed to
            the clusterer. Used to compute pairwise feature correlations
            for the dedup step.

    Returns:
        A frame with one row per cluster and the columns ``cluster``,
        ``size``, ``top_features`` (a ``list[str]``), ``z_profile`` (a
        ``dict[str, float]``) and ``summary`` (a ``str``). Cluster rows
        are ordered by ascending cluster id.

    Examples:
        Using a fit and modelling matrix ``X`` from
        :func:`~playlistsmith.cluster.fit_gmm` /
        :func:`~playlistsmith.cluster.prepare_matrix` (``X`` built as in
        the :func:`~playlistsmith.cluster.fit_gmm` example):

        >>> from playlistsmith.cluster import describe_clusters, fit_gmm
        >>> fit = fit_gmm(X, k_range=range(2, 6))
        >>> describe_clusters(fit, X)[["cluster", "size", "summary"]]
           cluster  size       summary
        0        0    15  high valence
        1        1    15   low valence
    """
    correlations = X.corr()
    means = result.feature_means_per_cluster
    sizes = pd.Series(result.labels).value_counts()

    rows: list[dict[str, object]] = []
    for c in range(result.k):
        z_profile = means.iloc[c]
        top = _select_top_features(z_profile, correlations)
        summary = ", ".join(_qualify(feat, z_profile[feat]) for feat in top)
        rows.append(
            {
                "cluster": c,
                "size": int(sizes.get(c, 0)),
                "top_features": top,
                "z_profile": z_profile.to_dict(),
                "summary": summary,
            }
        )

    return pd.DataFrame(
        rows, columns=["cluster", "size", "top_features", "z_profile", "summary"]
    )
