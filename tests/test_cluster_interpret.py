"""Tests for :func:`playlistsmith.cluster.interpret.describe_clusters`.

A bare cluster label is useless to a user — :func:`describe_clusters`
turns each cluster into a short, human-readable summary such as
``"high energy, fast tempo, low valence"`` (plan Section 4). The key
correctness criteria are:

- one row per cluster, with ``cluster``, ``size``, ``top_features``
  and ``summary`` columns;
- top features are picked by absolute z-profile, capped at three;
- correlated features are deduplicated (``|ρ| > 0.6``) so the summary
  does not say the same thing three ways;
- ``tempo`` and ``loudness`` get genre-appropriate qualifiers (fast/
  slow, loud/quiet) instead of bare ``high``/``low``.

The tests construct a :class:`ClusteringResult` by hand so the z-profile
is known exactly, rather than going through :func:`fit_gmm`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from playlistsmith.cluster.algorithms import ClusteringResult
from playlistsmith.cluster.interpret import describe_clusters
from playlistsmith.features.reccobeats import FEATURE_COLUMNS


def _zscore_matrix(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """A z-scored modelling matrix (per-column mean 0, SD 1)."""
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(
        rng.standard_normal((n, len(FEATURE_COLUMNS))),
        columns=list(FEATURE_COLUMNS),
    )
    return (raw - raw.mean()) / raw.std(ddof=0)


def _result_with_means(
    cluster_means: list[dict[str, float]], labels: np.ndarray
) -> ClusteringResult:
    """Build a minimal :class:`ClusteringResult` with controlled means."""
    means = pd.DataFrame(
        [
            {col: row.get(col, 0.0) for col in FEATURE_COLUMNS}
            for row in cluster_means
        ],
        columns=list(FEATURE_COLUMNS),
    )
    k = len(cluster_means)
    posteriors = np.eye(k)[labels]
    return ClusteringResult(
        labels=labels,
        posteriors=posteriors,
        k=k,
        bic_curve={k: 0.0},
        icl_curve={k: 0.0},
        silhouette=0.5,
        cohesion=1.0,
        feature_means_per_cluster=means,
        stability_ari=1.0,
        random_state=0,
        covariance_type="full",
    )


# --------------------------------------------------------------------------- #
# Shape & required columns                                                    #
# --------------------------------------------------------------------------- #


def test_returns_one_row_per_cluster() -> None:
    """Output has exactly one row per cluster in the input."""
    X = _zscore_matrix(n=60)
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    result = _result_with_means(
        [{"energy": 2.0}, {"energy": -1.5}, {}], labels
    )

    described = describe_clusters(result, X)

    assert len(described) == 3
    assert sorted(described["cluster"].tolist()) == [0, 1, 2]


def test_has_required_columns() -> None:
    """Plan Section 4: ``label, size, top features, summary string``."""
    X = _zscore_matrix(n=60)
    labels = np.array([0] * 30 + [1] * 30)
    result = _result_with_means(
        [{"energy": 2.0}, {"energy": -2.0}], labels
    )

    described = describe_clusters(result, X)

    for col in ("cluster", "size", "top_features", "summary"):
        assert col in described.columns


def test_size_column_matches_label_counts() -> None:
    """``size`` is the count of tracks in each cluster."""
    X = _zscore_matrix(n=60)
    labels = np.array([0] * 10 + [1] * 20 + [2] * 30)
    result = _result_with_means(
        [{"energy": 2.0}, {"energy": -2.0}, {"valence": 2.0}], labels
    )

    described = describe_clusters(result, X)

    sizes = dict(zip(described["cluster"], described["size"]))
    assert sizes == {0: 10, 1: 20, 2: 30}


# --------------------------------------------------------------------------- #
# Top-feature selection                                                       #
# --------------------------------------------------------------------------- #


def test_top_features_capped_at_three() -> None:
    """Even when every feature has a strong z, the label is at most three
    features long — the plan picks top-3 to keep summaries readable."""
    X = _zscore_matrix(n=40)
    labels = np.array([0] * 20 + [1] * 20)
    # All nine features extreme on cluster 0.
    extreme = {col: (i + 1) * 0.5 for i, col in enumerate(FEATURE_COLUMNS)}
    result = _result_with_means([extreme, {}], labels)

    described = describe_clusters(result, X)

    cluster_0 = described.set_index("cluster").loc[0]
    assert len(cluster_0["top_features"]) <= 3


def test_top_features_are_ranked_by_absolute_z() -> None:
    """The feature with the largest ``|z|`` lands first, regardless of sign."""
    X = _zscore_matrix(n=40)
    labels = np.array([0] * 20 + [1] * 20)
    # Cluster 0: largest |z| is valence (-3), then danceability (+2), then
    # liveness (+1). Everything else ~0.
    means_0 = {"valence": -3.0, "danceability": 2.0, "liveness": 1.0}
    result = _result_with_means([means_0, {}], labels)

    described = describe_clusters(result, X)

    cluster_0 = described.set_index("cluster").loc[0]
    assert cluster_0["top_features"][0] == "valence"
    assert cluster_0["top_features"][1] == "danceability"


def test_features_near_zero_are_excluded() -> None:
    """A feature whose ``|z|`` is near zero should not appear in the
    summary even if the cluster has fewer than 3 strong features."""
    X = _zscore_matrix(n=40)
    labels = np.array([0] * 20 + [1] * 20)
    # Only one notable feature; the other eight have means ~0.
    result = _result_with_means(
        [{"energy": 2.5}, {"energy": -2.5}], labels
    )

    described = describe_clusters(result, X)

    cluster_0 = described.set_index("cluster").loc[0]
    assert "energy" in cluster_0["top_features"]
    # No spurious near-average features have been padded in.
    assert all(
        abs(cluster_0["z_profile"][f]) >= 0.5
        for f in cluster_0["top_features"]
    )


# --------------------------------------------------------------------------- #
# Correlated-feature deduplication                                            #
# --------------------------------------------------------------------------- #


def test_correlated_features_are_deduplicated() -> None:
    """When two features correlate at ``|ρ| > 0.6``, only the one with
    higher ``|z|`` appears in the summary."""
    rng = np.random.default_rng(0)
    n = 80
    # Make energy and loudness strongly correlated (ρ ≈ 0.9).
    energy = rng.standard_normal(n)
    loudness = 0.9 * energy + 0.1 * rng.standard_normal(n)
    data = {col: rng.standard_normal(n) for col in FEATURE_COLUMNS}
    data["energy"] = energy
    data["loudness"] = loudness
    raw = pd.DataFrame(data, columns=list(FEATURE_COLUMNS))
    X = (raw - raw.mean()) / raw.std(ddof=0)

    labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
    # Both energy and loudness extreme in cluster 0; energy slightly larger.
    result = _result_with_means(
        [{"energy": 2.5, "loudness": 2.3, "valence": 1.5}, {}], labels
    )

    described = describe_clusters(result, X)

    cluster_0 = described.set_index("cluster").loc[0]
    top = cluster_0["top_features"]
    assert "energy" in top
    assert "loudness" not in top  # Suppressed by correlation with energy.
    # The deduplicated slot should be filled by the next-strongest
    # uncorrelated feature.
    assert "valence" in top


# --------------------------------------------------------------------------- #
# Summary string                                                              #
# --------------------------------------------------------------------------- #


def test_summary_uses_high_and_low_qualifiers_with_sign() -> None:
    """``+z`` → "high X"; ``-z`` → "low X" for ordinary features."""
    X = _zscore_matrix(n=40)
    labels = np.array([0] * 20 + [1] * 20)
    result = _result_with_means(
        [{"energy": 2.0, "valence": -2.0}, {"energy": -2.0, "valence": 2.0}],
        labels,
    )

    described = describe_clusters(result, X)

    s0 = described.set_index("cluster").loc[0]["summary"]
    s1 = described.set_index("cluster").loc[1]["summary"]
    assert "high energy" in s0
    assert "low valence" in s0
    assert "low energy" in s1
    assert "high valence" in s1


def test_tempo_summary_uses_fast_and_slow_qualifiers() -> None:
    """``tempo`` reads more naturally as fast/slow than high/low."""
    X = _zscore_matrix(n=40)
    labels = np.array([0] * 20 + [1] * 20)
    result = _result_with_means(
        [{"tempo": 2.0}, {"tempo": -2.0}], labels
    )

    described = describe_clusters(result, X)

    assert "fast tempo" in described.set_index("cluster").loc[0]["summary"]
    assert "slow tempo" in described.set_index("cluster").loc[1]["summary"]


def test_loudness_summary_uses_loud_and_quiet_qualifiers() -> None:
    """``loudness`` reads naturally as loud/quiet."""
    X = _zscore_matrix(n=40)
    labels = np.array([0] * 20 + [1] * 20)
    result = _result_with_means(
        [{"loudness": 2.0}, {"loudness": -2.0}], labels
    )

    described = describe_clusters(result, X)

    s0 = described.set_index("cluster").loc[0]["summary"]
    s1 = described.set_index("cluster").loc[1]["summary"]
    assert "loud" in s0 and "quiet" not in s0
    assert "quiet" in s1 and "loud" not in s1


def test_summary_is_comma_separated() -> None:
    """The summary is a comma-separated, lowercase, no-trailing-comma
    string built from the top features."""
    X = _zscore_matrix(n=40)
    labels = np.array([0] * 20 + [1] * 20)
    result = _result_with_means(
        [{"energy": 2.0, "tempo": 1.8, "valence": -1.5}, {}], labels
    )

    described = describe_clusters(result, X)

    s = described.set_index("cluster").loc[0]["summary"]
    assert ", " in s
    assert not s.endswith(",")
    assert s == s.lower()
