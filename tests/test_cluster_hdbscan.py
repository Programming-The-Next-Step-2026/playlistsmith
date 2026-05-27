"""Tests for :func:`playlistsmith.cluster.algorithms.fit_hdbscan`.

HDBSCAN is the density-based, outlier-aware alternative (plan §3). It
differs from GMM and K-Means in three ways the tests have to pin down:

- there is *no* ``k`` to sweep — the number of clusters is discovered
  from the data, so the relevant knobs are ``min_cluster_size`` and
  ``min_samples``;
- low-density points are labelled ``-1`` (noise) rather than being
  forced into a cluster — this naturally flows into the Unclassified
  bucket maintained by the public pipeline;
- BIC / ICL / inertia curves are not defined; only a chosen-fit
  silhouette (computed over non-noise points only) and a separately
  reported ``noise_rate`` make sense.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from playlistsmith.cluster.algorithms import (
    ClusteringResult,
    fit_hdbscan,
)
from playlistsmith.features.reccobeats import FEATURE_COLUMNS


def _two_blobs_plus_noise(
    per_blob: int = 80, n_noise: int = 10, seed: int = 0
) -> pd.DataFrame:
    """Two tight blobs plus a scatter of noise points far from both."""
    rng = np.random.default_rng(seed)
    n_features = len(FEATURE_COLUMNS)
    blob_a = rng.normal(loc=+5.0, scale=0.5, size=(per_blob, n_features))
    blob_b = rng.normal(loc=-5.0, scale=0.5, size=(per_blob, n_features))
    noise = rng.uniform(low=-20.0, high=20.0, size=(n_noise, n_features))
    arr = np.vstack([blob_a, blob_b, noise])
    rng.shuffle(arr)
    return pd.DataFrame(arr, columns=list(FEATURE_COLUMNS))


# --------------------------------------------------------------------------- #
# Return shape                                                                #
# --------------------------------------------------------------------------- #


def test_fit_hdbscan_returns_clustering_result() -> None:
    """HDBSCAN reuses the same dataclass as GMM and K-Means."""
    X = _two_blobs_plus_noise()

    result = fit_hdbscan(X, min_cluster_size=5)

    assert isinstance(result, ClusteringResult)


def test_posteriors_is_none_for_hdbscan() -> None:
    """HDBSCAN is hard-assignment with a noise label."""
    X = _two_blobs_plus_noise()

    result = fit_hdbscan(X, min_cluster_size=5)

    assert result.posteriors is None


def test_covariance_type_records_hdbscan() -> None:
    """The variant marker carries through the result."""
    X = _two_blobs_plus_noise()

    result = fit_hdbscan(X, min_cluster_size=5)

    assert result.covariance_type == "hdbscan"


def test_bic_icl_inertia_curves_are_empty_for_hdbscan() -> None:
    """HDBSCAN has no ``k`` sweep; the GMM/K-Means curves are empty."""
    X = _two_blobs_plus_noise()

    result = fit_hdbscan(X, min_cluster_size=5)

    assert result.bic_curve == {}
    assert result.icl_curve == {}
    assert result.inertia_curve == {}
    assert result.silhouette_curve == {}
    assert result.calinski_harabasz_curve == {}


# --------------------------------------------------------------------------- #
# Clustering behaviour                                                        #
# --------------------------------------------------------------------------- #


def test_finds_two_clusters_on_two_well_separated_blobs() -> None:
    """Two blobs + a sparse noise scatter → 2 clusters (plus noise)."""
    X = _two_blobs_plus_noise(per_blob=80, n_noise=10, seed=0)

    result = fit_hdbscan(X, min_cluster_size=10)

    assert result.k == 2


def test_noise_points_get_label_minus_one() -> None:
    """Low-density points are not forced into a cluster; they get ``-1``."""
    X = _two_blobs_plus_noise(per_blob=80, n_noise=10, seed=0)

    result = fit_hdbscan(X, min_cluster_size=10)

    assert (result.labels == -1).any()


def test_noise_rate_is_recorded_and_between_zero_and_one() -> None:
    """The noise-point share is reported on the result so the coverage
    summary can surface it (plan §3 step 2)."""
    X = _two_blobs_plus_noise(per_blob=80, n_noise=10, seed=0)

    result = fit_hdbscan(X, min_cluster_size=10)

    assert 0.0 <= result.noise_rate <= 1.0
    assert result.noise_rate > 0.0  # Some noise was injected.


def test_no_noise_when_data_has_no_outliers() -> None:
    """If the data is all dense clusters, the noise rate is zero."""
    X = _two_blobs_plus_noise(per_blob=80, n_noise=0, seed=0)

    result = fit_hdbscan(X, min_cluster_size=10)

    assert result.noise_rate == 0.0


def test_silhouette_excludes_noise_points() -> None:
    """Silhouette is computed only over non-noise labels — including
    noise as its own cluster would distort the score."""
    X = _two_blobs_plus_noise(per_blob=80, n_noise=10, seed=0)

    result = fit_hdbscan(X, min_cluster_size=10)

    # Silhouette is well-defined and high; the noise points were
    # excluded rather than collapsed into a degenerate cluster.
    assert -1.0 <= result.silhouette <= 1.0
    assert result.silhouette > 0.5


def test_feature_means_have_one_row_per_non_noise_cluster() -> None:
    """``feature_means_per_cluster`` has ``k`` rows (the discovered
    non-noise clusters); the noise bucket is not a row."""
    X = _two_blobs_plus_noise()

    result = fit_hdbscan(X, min_cluster_size=10)

    assert result.feature_means_per_cluster.shape == (result.k, len(FEATURE_COLUMNS))
    assert list(result.feature_means_per_cluster.columns) == list(FEATURE_COLUMNS)


def test_cohesion_is_non_negative() -> None:
    """Cohesion is a mean of non-negative squared distances."""
    X = _two_blobs_plus_noise()

    result = fit_hdbscan(X, min_cluster_size=10)

    assert result.cohesion >= 0.0


# --------------------------------------------------------------------------- #
# Parameter handling                                                          #
# --------------------------------------------------------------------------- #


def test_min_samples_defaults_to_min_cluster_size() -> None:
    """Plan §3: ``min_samples`` defaults to ``min_cluster_size`` but is
    exposable as its own argument. Calling without it should not error
    and should produce a sensible fit."""
    X = _two_blobs_plus_noise()

    result = fit_hdbscan(X, min_cluster_size=10)

    assert result.k >= 1


def test_min_samples_is_independently_settable() -> None:
    """Passing ``min_samples`` explicitly changes the noise sensitivity."""
    X = _two_blobs_plus_noise(per_blob=80, n_noise=10)

    strict = fit_hdbscan(X, min_cluster_size=10, min_samples=20)
    lax = fit_hdbscan(X, min_cluster_size=10, min_samples=1)

    # Higher min_samples → more points labelled as noise.
    assert strict.noise_rate >= lax.noise_rate


def test_fit_is_deterministic() -> None:
    """HDBSCAN has no randomness; two fits agree exactly."""
    X = _two_blobs_plus_noise()

    r1 = fit_hdbscan(X, min_cluster_size=10)
    r2 = fit_hdbscan(X, min_cluster_size=10)

    np.testing.assert_array_equal(r1.labels, r2.labels)


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #


def test_empty_matrix_raises_value_error() -> None:
    """Empty input is rejected with a user-facing error."""
    empty = pd.DataFrame(columns=list(FEATURE_COLUMNS))

    with pytest.raises(ValueError):
        fit_hdbscan(empty, min_cluster_size=5)
