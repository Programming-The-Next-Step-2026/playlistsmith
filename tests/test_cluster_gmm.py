"""Tests for :func:`playlistsmith.cluster.algorithms.fit_gmm`.

The GMM fitter is the project-default clusterer (per ``CLAUDE.md``). It
sweeps ``k`` over a user-supplied range on a transformed-and-scaled
matrix, picks ``k`` by BIC with an ICL cross-check, refits, and packages
labels, posteriors and quality metrics into a
:class:`~playlistsmith.cluster.algorithms.ClusteringResult` dataclass.

These tests exercise the public contract against synthetic data where
the right answer is known by construction (three well-separated
Gaussian blobs in 9-D), plus a few invariants the result must always
satisfy (posteriors sum to 1, labels match ``argmax`` of posteriors,
small libraries fall back to diag covariance, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from playlistsmith.cluster.algorithms import (
    ClusteringResult,
    fit_gmm,
)
from playlistsmith.features.reccobeats import FEATURE_COLUMNS


def _three_blob_matrix(
    per_blob: int = 60, seed: int = 0
) -> pd.DataFrame:
    """Three well-separated Gaussian blobs in the nine-feature space.

    Centres are placed far enough apart (`±5σ`) that any reasonable
    clusterer should recover ``k = 3``. Returned as a DataFrame with
    the canonical :data:`FEATURE_COLUMNS` so it matches the modelling
    matrix that :func:`prepare_matrix` would produce.
    """
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
# Result shape                                                                #
# --------------------------------------------------------------------------- #


def test_fit_gmm_returns_clustering_result_dataclass() -> None:
    """The return type is the documented :class:`ClusteringResult`."""
    X = _three_blob_matrix()

    result = fit_gmm(X, k_range=range(2, 6), random_state=0)

    assert isinstance(result, ClusteringResult)


def test_clustering_result_fields_match_plan_contract() -> None:
    """Every field listed in plan Section 1 is present and populated."""
    X = _three_blob_matrix()

    result = fit_gmm(X, k_range=range(2, 6), random_state=0)

    # Required fields per plan/clustering_analysis_revised.md §1.
    assert isinstance(result.labels, np.ndarray)
    assert isinstance(result.posteriors, np.ndarray)
    assert isinstance(result.k, int)
    assert isinstance(result.bic_curve, dict)
    assert isinstance(result.icl_curve, dict)
    assert isinstance(result.silhouette, float)
    assert isinstance(result.cohesion, float)
    assert isinstance(result.feature_means_per_cluster, pd.DataFrame)
    assert isinstance(result.stability_ari, float)
    assert isinstance(result.random_state, int)
    # Added for small-library-fallback transparency (Section 1 step 5).
    assert isinstance(result.covariance_type, str)


def test_labels_shape_matches_n_rows_of_input() -> None:
    """One label per input row."""
    X = _three_blob_matrix(per_blob=40)

    result = fit_gmm(X, k_range=range(2, 6), random_state=0)

    assert result.labels.shape == (len(X),)


def test_posteriors_shape_is_n_by_k_and_rows_sum_to_one() -> None:
    """Soft assignments form a valid posterior matrix."""
    X = _three_blob_matrix(per_blob=40)

    result = fit_gmm(X, k_range=range(2, 6), random_state=0)

    assert result.posteriors.shape == (len(X), result.k)
    np.testing.assert_allclose(
        result.posteriors.sum(axis=1), np.ones(len(X)), atol=1e-6
    )


def test_labels_equal_argmax_of_posteriors() -> None:
    """Hard labels are derived from the posterior matrix."""
    X = _three_blob_matrix(per_blob=40)

    result = fit_gmm(X, k_range=range(2, 6), random_state=0)

    np.testing.assert_array_equal(
        result.labels, np.argmax(result.posteriors, axis=1)
    )


def test_feature_means_have_one_row_per_cluster_with_feature_columns() -> None:
    """Per-cluster mean vectors live in the transformed-scaled space and
    are indexed by the canonical feature columns."""
    X = _three_blob_matrix(per_blob=40)

    result = fit_gmm(X, k_range=range(2, 6), random_state=0)

    assert result.feature_means_per_cluster.shape == (result.k, len(FEATURE_COLUMNS))
    assert list(result.feature_means_per_cluster.columns) == list(FEATURE_COLUMNS)


def test_bic_and_icl_curves_have_one_entry_per_k_in_range() -> None:
    """Every ``k`` in the sweep contributes one BIC and one ICL point so
    the diagnostic plot is complete."""
    X = _three_blob_matrix(per_blob=40)
    k_range = range(2, 7)

    result = fit_gmm(X, k_range=k_range, random_state=0)

    assert set(result.bic_curve.keys()) == set(k_range)
    assert set(result.icl_curve.keys()) == set(k_range)


# --------------------------------------------------------------------------- #
# Behaviour on synthetic data                                                 #
# --------------------------------------------------------------------------- #


def test_bic_picks_three_blobs_on_well_separated_data() -> None:
    """On three well-separated Gaussian blobs, the chosen ``k`` is 3.

    The plan allows ``± 1`` for borderline cases; we keep the strict
    expectation here because the blobs are deliberately far apart.
    """
    X = _three_blob_matrix(per_blob=60, seed=0)

    result = fit_gmm(X, k_range=range(2, 8), random_state=0)

    assert result.k == 3


def test_silhouette_is_high_on_well_separated_blobs() -> None:
    """Silhouette score is bounded in [-1, 1] and should be clearly
    positive on data with obvious structure."""
    X = _three_blob_matrix(per_blob=60, seed=0)

    result = fit_gmm(X, k_range=range(2, 8), random_state=0)

    assert -1.0 <= result.silhouette <= 1.0
    assert result.silhouette > 0.5  # Comfortably above noise.


def test_cohesion_is_non_negative() -> None:
    """Intra-cluster cohesion (mean within-cluster squared distance to
    centroid) cannot be negative."""
    X = _three_blob_matrix(per_blob=40)

    result = fit_gmm(X, k_range=range(2, 6), random_state=0)

    assert result.cohesion >= 0.0


def test_stability_ari_is_high_on_well_separated_blobs() -> None:
    """Re-fitting with a second seed and computing adjusted Rand index
    should be near 1 on well-separated data."""
    X = _three_blob_matrix(per_blob=60, seed=0)

    result = fit_gmm(X, k_range=range(2, 8), random_state=0)

    assert -1.0 <= result.stability_ari <= 1.0
    assert result.stability_ari > 0.7  # Plan's stability threshold.


def test_fit_is_reproducible_for_fixed_random_state() -> None:
    """Two fits with the same ``random_state`` produce identical labels."""
    X = _three_blob_matrix(per_blob=40, seed=0)

    r1 = fit_gmm(X, k_range=range(2, 6), random_state=42)
    r2 = fit_gmm(X, k_range=range(2, 6), random_state=42)

    assert r1.k == r2.k
    np.testing.assert_array_equal(r1.labels, r2.labels)
    np.testing.assert_allclose(r1.posteriors, r2.posteriors, atol=1e-10)


def test_random_state_is_recorded_on_result() -> None:
    """The seed used is part of the result so users can re-derive it."""
    X = _three_blob_matrix(per_blob=40)

    result = fit_gmm(X, k_range=range(2, 6), random_state=7)

    assert result.random_state == 7


# --------------------------------------------------------------------------- #
# Small-library fallback                                                      #
# --------------------------------------------------------------------------- #


def test_small_library_falls_back_to_diag_covariance() -> None:
    """Below ~50 tracks, full covariance is under-determined; the fitter
    auto-falls back to ``covariance_type='diag'`` (plan Section 1 step 5)."""
    X = _three_blob_matrix(per_blob=10, seed=0)  # 30 rows total
    assert len(X) < 50

    result = fit_gmm(X, k_range=range(2, 5), random_state=0)

    assert result.covariance_type == "diag"


def test_large_library_uses_full_covariance() -> None:
    """At or above the threshold, ``covariance_type='full'`` is used."""
    X = _three_blob_matrix(per_blob=30, seed=0)  # 90 rows total
    assert len(X) >= 50

    result = fit_gmm(X, k_range=range(2, 5), random_state=0)

    assert result.covariance_type == "full"


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #


def test_k_range_with_only_k_equal_one_raises() -> None:
    """GMM ``k`` must be at least 2 — otherwise there is nothing to
    cluster."""
    X = _three_blob_matrix(per_blob=20)

    with pytest.raises(ValueError, match="k"):
        fit_gmm(X, k_range=range(1, 2), random_state=0)


def test_k_range_max_above_n_samples_raises_clear_error() -> None:
    """sklearn raises a cryptic ``n_components > n_samples`` error in
    this case; we must catch it and name ``k_range`` and the track
    count so the user knows what to fix."""
    # 7 tracks, default k_range tops out at 12.
    X = _three_blob_matrix(per_blob=3, seed=0).head(7)

    with pytest.raises(ValueError, match=r"k_range.*7"):
        fit_gmm(X, k_range=range(2, 13), random_state=0)


def test_empty_matrix_raises_value_error() -> None:
    """Empty input is rejected with a user-facing error."""
    empty = pd.DataFrame(columns=list(FEATURE_COLUMNS))

    with pytest.raises(ValueError):
        fit_gmm(empty, k_range=range(2, 4), random_state=0)
