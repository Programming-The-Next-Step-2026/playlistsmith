"""Tests for :func:`playlistsmith.cluster.algorithms.fit_kmeans`.

K-Means is the cheap, hard-assignment alternative to GMM.
Selection differs from the GMM path:

- there is no BIC / ICL, so :attr:`ClusteringResult.bic_curve` and
  :attr:`icl_curve` are empty;
- per-``k`` we instead record inertia (for elbow plotting), silhouette
  and Calinski–Harabasz on three new ``ClusteringResult`` fields;
- ``k`` is picked by *agreement* between silhouette and Calinski–
  Harabasz rather than by either metric alone (silhouette systematically
  prefers low ``k`` on real audio libraries);
- ``posteriors`` is ``None`` — callers that rank tracks within a
  cluster must fall back to distance-to-centroid.

These tests pin all of that against deterministic synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from playlistsmith.cluster.algorithms import (
    ClusteringResult,
    fit_kmeans,
)
from playlistsmith.features.reccobeats import FEATURE_COLUMNS


def _three_blob_matrix(
    per_blob: int = 60, seed: int = 0
) -> pd.DataFrame:
    """Three well-separated Gaussian blobs in the nine-feature space."""
    rng = np.random.default_rng(seed)
    n_features = len(FEATURE_COLUMNS)
    centres = np.array(
        [
            [+5.0] * n_features,
            [-5.0] * n_features,
            [+5.0 if i % 2 == 0 else -5.0 for i in range(n_features)],
        ]
    )
    rows = []
    for centre in centres:
        rows.append(rng.normal(loc=centre, scale=1.0, size=(per_blob, n_features)))
    arr = np.vstack(rows)
    rng.shuffle(arr)
    return pd.DataFrame(arr, columns=list(FEATURE_COLUMNS))


# --------------------------------------------------------------------------- #
# Return shape & K-Means specifics                                            #
# --------------------------------------------------------------------------- #


def test_fit_kmeans_returns_clustering_result() -> None:
    """K-Means reuses the same result dataclass as GMM (plan §2)."""
    X = _three_blob_matrix()

    result = fit_kmeans(X, k_range=range(2, 6), random_state=0)

    assert isinstance(result, ClusteringResult)


def test_posteriors_is_none_for_kmeans() -> None:
    """K-Means is hard-assignment: there are no posteriors. Downstream
    code is expected to fall back to distance-to-centroid."""
    X = _three_blob_matrix()

    result = fit_kmeans(X, k_range=range(2, 6), random_state=0)

    assert result.posteriors is None


def test_bic_and_icl_curves_are_empty_for_kmeans() -> None:
    """BIC/ICL are not defined for K-Means — those fields are present
    on the dataclass but empty here."""
    X = _three_blob_matrix()

    result = fit_kmeans(X, k_range=range(2, 6), random_state=0)

    assert result.bic_curve == {}
    assert result.icl_curve == {}


def test_kmeans_curves_have_one_entry_per_k_in_range() -> None:
    """``inertia_curve``, ``silhouette_curve`` and
    ``calinski_harabasz_curve`` populate the per-``k`` diagnostics
    used for selection (plan §2 step 1)."""
    X = _three_blob_matrix()
    k_range = range(2, 7)

    result = fit_kmeans(X, k_range=k_range, random_state=0)

    assert set(result.inertia_curve.keys()) == set(k_range)
    assert set(result.silhouette_curve.keys()) == set(k_range)
    assert set(result.calinski_harabasz_curve.keys()) == set(k_range)


def test_inertia_curve_is_non_increasing_in_k() -> None:
    """K-Means inertia (sum of squared distances to centroid) is
    monotonically non-increasing in ``k`` by construction."""
    X = _three_blob_matrix()
    result = fit_kmeans(X, k_range=range(2, 7), random_state=0)

    ks_sorted = sorted(result.inertia_curve)
    values = [result.inertia_curve[k] for k in ks_sorted]
    for prev, curr in zip(values, values[1:]):
        assert curr <= prev + 1e-9


def test_covariance_type_records_kmeans() -> None:
    """The ``covariance_type`` slot is reused as a model-variant marker
    so callers can tell GMM-full / GMM-diag / K-Means apart."""
    X = _three_blob_matrix()

    result = fit_kmeans(X, k_range=range(2, 6), random_state=0)

    assert result.covariance_type == "kmeans"


# --------------------------------------------------------------------------- #
# Selection behaviour                                                         #
# --------------------------------------------------------------------------- #


def test_selection_picks_three_blobs() -> None:
    """On three well-separated blobs, the silhouette-CH agreement rule
    picks ``k = 3``."""
    X = _three_blob_matrix(per_blob=60, seed=0)

    result = fit_kmeans(X, k_range=range(2, 8), random_state=0)

    assert result.k == 3


def test_silhouette_field_is_in_valid_range() -> None:
    """The chosen-``k`` silhouette is in ``[-1, 1]`` and positive on
    structured data."""
    X = _three_blob_matrix(per_blob=60, seed=0)

    result = fit_kmeans(X, k_range=range(2, 8), random_state=0)

    assert -1.0 <= result.silhouette <= 1.0
    assert result.silhouette > 0.5


def test_cohesion_is_non_negative() -> None:
    """Intra-cluster cohesion is a mean of non-negative squared
    distances."""
    X = _three_blob_matrix()

    result = fit_kmeans(X, k_range=range(2, 6), random_state=0)

    assert result.cohesion >= 0.0


def test_stability_ari_is_high_on_well_separated_blobs() -> None:
    """Re-fit with a different seed and check agreement."""
    X = _three_blob_matrix(per_blob=60, seed=0)

    result = fit_kmeans(X, k_range=range(2, 8), random_state=0)

    assert -1.0 <= result.stability_ari <= 1.0
    assert result.stability_ari > 0.7


def test_fit_is_reproducible_for_fixed_random_state() -> None:
    """Two fits with the same seed produce identical labels."""
    X = _three_blob_matrix(per_blob=40)

    r1 = fit_kmeans(X, k_range=range(2, 6), random_state=42)
    r2 = fit_kmeans(X, k_range=range(2, 6), random_state=42)

    assert r1.k == r2.k
    np.testing.assert_array_equal(r1.labels, r2.labels)


def test_random_state_is_recorded() -> None:
    """The seed used is part of the result."""
    X = _three_blob_matrix()

    result = fit_kmeans(X, k_range=range(2, 6), random_state=11)

    assert result.random_state == 11


def test_feature_means_have_one_row_per_cluster() -> None:
    """Per-cluster mean vectors live in the modelling space."""
    X = _three_blob_matrix(per_blob=40)

    result = fit_kmeans(X, k_range=range(2, 6), random_state=0)

    assert result.feature_means_per_cluster.shape == (result.k, len(FEATURE_COLUMNS))


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #


def test_k_below_two_raises() -> None:
    """K-Means with ``k < 2`` is rejected (nothing to cluster)."""
    X = _three_blob_matrix(per_blob=20)

    with pytest.raises(ValueError, match="k"):
        fit_kmeans(X, k_range=range(1, 2), random_state=0)


def test_k_range_max_above_n_samples_raises_clear_error() -> None:
    """KMeans would otherwise fail with a misleading "n_samples >="
    error from sklearn. The wrapper must name ``k_range``."""
    X = _three_blob_matrix(per_blob=3, seed=0).head(7)

    with pytest.raises(ValueError, match=r"k_range.*7"):
        fit_kmeans(X, k_range=range(2, 13), random_state=0)


def test_empty_matrix_raises_value_error() -> None:
    """Empty input is rejected."""
    empty = pd.DataFrame(columns=list(FEATURE_COLUMNS))

    with pytest.raises(ValueError):
        fit_kmeans(empty, k_range=range(2, 4), random_state=0)
