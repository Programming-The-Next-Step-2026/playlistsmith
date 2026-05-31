"""Tests for :mod:`playlistsmith.cluster.preprocess`.

The preprocessing pipeline turns a unified feature DataFrame (identity
columns + nine ReccoBeats audio features) into a model-ready matrix.

Prepocessing steps are:
- bounded ``[0, 1]`` features are logit-transformed before scaling,
- ``tempo`` is log-transformed,
- ``loudness`` is z-scored as-is (already in dB),
- per-feature missing cells are filled with the global median,
- rows missing more than 3 of 9 features are dropped, and
- the resulting columns have ~zero mean and unit variance.

Every transformation decision is recorded on the returned
:class:`TransformLog` so the coverage report can surface it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from playlistsmith.cluster.preprocess import (
    BOUNDED_FEATURES,
    LOG_FEATURES,
    TransformLog,
    prepare_matrix,
)
from playlistsmith.features.reccobeats import FEATURE_COLUMNS
from playlistsmith.io.csv_loader import ARTIST, SPOTIFY_ID, TITLE


def _synthetic_features(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Build a deterministic features-shaped frame with ``n`` tracks.

    Bounded features are drawn from ``Beta(2, 5)`` so values stay strictly
    inside ``(0, 1)`` (logit-safe), ``tempo`` from a log-normal centred
    near 110 BPM, and ``loudness`` from a normal centred at -8 dB.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for i in range(n):
        row: dict[str, object] = {
            SPOTIFY_ID: f"sp{i:03d}",
            TITLE: f"Track {i}",
            ARTIST: f"Artist {i}",
        }
        for col in BOUNDED_FEATURES:
            row[col] = float(rng.beta(2.0, 5.0))
        row["tempo"] = float(np.exp(rng.normal(np.log(110.0), 0.2)))
        row["loudness"] = float(rng.normal(-8.0, 3.0))
        rows.append(row)
    return pd.DataFrame(rows, columns=[SPOTIFY_ID, TITLE, ARTIST, *FEATURE_COLUMNS])


# --------------------------------------------------------------------------- #
# Module-level constants                                                      #
# --------------------------------------------------------------------------- #


def test_bounded_features_constant_lists_the_seven_unit_interval_columns() -> None:
    """The bounded-feature list is the seven ``[0, 1]``-ranged columns."""
    assert set(BOUNDED_FEATURES) == {
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "speechiness",
        "valence",
    }


def test_log_features_constant_lists_only_tempo() -> None:
    """Only ``tempo`` is log-transformed; ``loudness`` is already in dB."""
    assert list(LOG_FEATURES) == ["tempo"]


# --------------------------------------------------------------------------- #
# Return shape                                                                #
# --------------------------------------------------------------------------- #


def test_prepare_matrix_returns_four_tuple_with_expected_types() -> None:
    """Return shape is ``(X, index, scaler, transform_log)`` with the documented
    types, per the plan."""
    features = _synthetic_features()

    X, index, scaler, log = prepare_matrix(features)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(index, pd.DataFrame)
    assert isinstance(scaler, StandardScaler)
    assert isinstance(log, TransformLog)


def test_x_has_all_nine_feature_columns_and_no_identity_columns() -> None:
    """``X`` is the modelling matrix: nine numeric columns, no identity."""
    features = _synthetic_features()

    X, _, _, _ = prepare_matrix(features)

    assert list(X.columns) == FEATURE_COLUMNS
    assert SPOTIFY_ID not in X.columns
    assert TITLE not in X.columns
    assert ARTIST not in X.columns


def test_identity_columns_survive_alongside_in_index_frame() -> None:
    """The identity columns are carried in ``index`` so labels can be
    re-attached later."""
    features = _synthetic_features(n=5)

    X, index, _, _ = prepare_matrix(features)

    assert list(index.columns) == [SPOTIFY_ID, TITLE, ARTIST]
    assert len(index) == len(X)
    # Row order is preserved so labels can be zipped back on by position.
    pd.testing.assert_series_equal(
        index[SPOTIFY_ID].reset_index(drop=True),
        features[SPOTIFY_ID].reset_index(drop=True),
        check_names=False,
    )


# --------------------------------------------------------------------------- #
# Transforms (logit, log, scale)                                              #
# --------------------------------------------------------------------------- #


def test_bounded_columns_are_logit_transformed_before_scaling() -> None:
    """Logit on each ``[0, 1]`` feature, then z-score. We can recover the
    raw cell value by inverting the scaler and then the logit."""
    features = _synthetic_features(n=50)

    X, _, scaler, _ = prepare_matrix(features)

    # Inverse-z the scaled matrix back to the logit-space values.
    bounded_idx = [FEATURE_COLUMNS.index(c) for c in BOUNDED_FEATURES]
    unscaled = scaler.inverse_transform(X.to_numpy())
    for col in BOUNDED_FEATURES:
        j = FEATURE_COLUMNS.index(col)
        logit_vals = unscaled[:, j]
        # Invert the logit and check it matches the original column values.
        recovered = 1.0 / (1.0 + np.exp(-logit_vals))
        np.testing.assert_allclose(
            recovered, features[col].to_numpy(), atol=1e-6
        )
    # Sanity: the bounded indices are exactly the columns we just covered.
    assert len(bounded_idx) == len(BOUNDED_FEATURES)


def test_logit_clips_values_at_the_unit_boundaries() -> None:
    """Raw 0 / 1 values would blow up under logit; preprocess must clip
    them to a small epsilon first."""
    features = _synthetic_features(n=5)
    # Force boundary values into one bounded column.
    features.loc[0, "acousticness"] = 0.0
    features.loc[1, "acousticness"] = 1.0

    X, _, _, _ = prepare_matrix(features)

    # Neither infinity nor NaN should appear after the transform.
    assert np.isfinite(X["acousticness"].to_numpy()).all()


def test_tempo_is_log_transformed() -> None:
    """``tempo`` goes through ``log`` before scaling."""
    features = _synthetic_features(n=50)

    X, _, scaler, _ = prepare_matrix(features)

    j = FEATURE_COLUMNS.index("tempo")
    unscaled = scaler.inverse_transform(X.to_numpy())
    recovered = np.exp(unscaled[:, j])
    np.testing.assert_allclose(recovered, features["tempo"].to_numpy(), atol=1e-6)


def test_loudness_is_neither_logit_nor_log_transformed() -> None:
    """``loudness`` is already in dB; we only z-score it. Recovering the
    raw value should be a plain inverse-z, no exp / inverse-logit."""
    features = _synthetic_features(n=50)

    X, _, scaler, _ = prepare_matrix(features)

    j = FEATURE_COLUMNS.index("loudness")
    unscaled = scaler.inverse_transform(X.to_numpy())
    np.testing.assert_allclose(unscaled[:, j], features["loudness"].to_numpy(), atol=1e-6)


def test_scaled_columns_have_zero_mean_and_unit_variance() -> None:
    """After scaling, every modelling column has ~mean 0 and ~SD 1."""
    features = _synthetic_features(n=200)

    X, _, _, _ = prepare_matrix(features)

    means = X.mean(axis=0).to_numpy()
    stds = X.std(axis=0, ddof=0).to_numpy()
    np.testing.assert_allclose(means, np.zeros_like(means), atol=1e-9)
    np.testing.assert_allclose(stds, np.ones_like(stds), atol=1e-9)


# --------------------------------------------------------------------------- #
# Missing-value handling                                                      #
# --------------------------------------------------------------------------- #


def test_median_imputation_fills_a_missing_cell_with_the_column_median() -> None:
    """A single missing cell in a column is replaced with that column's
    median (computed from the non-missing values)."""
    features = _synthetic_features(n=20)
    features.loc[3, "danceability"] = np.nan
    # Median is computed from the non-missing values (pandas skips NaN).
    expected_median = features["danceability"].median()

    X, _, scaler, log = prepare_matrix(features)

    j = FEATURE_COLUMNS.index("danceability")
    unscaled = scaler.inverse_transform(X.to_numpy())
    recovered = 1.0 / (1.0 + np.exp(-unscaled[:, j]))
    assert math.isclose(recovered[3], expected_median, abs_tol=1e-6)
    assert log.imputed_counts.get("danceability", 0) == 1


def test_row_with_more_than_three_missing_features_is_dropped() -> None:
    """If >3 of the 9 features are missing on a row, it is dropped from
    the modelling matrix and surfaced on the transform log."""
    features = _synthetic_features(n=10)
    # Knock out four features on row 2 (>3 of 9).
    for col in ["acousticness", "danceability", "energy", "instrumentalness"]:
        features.loc[2, col] = np.nan

    X, index, _, log = prepare_matrix(features)

    assert len(X) == 9
    assert len(index) == 9
    assert "sp002" not in index[SPOTIFY_ID].tolist()
    assert log.dropped_tracks[SPOTIFY_ID].tolist() == ["sp002"]


def test_row_with_exactly_three_missing_features_is_kept_and_imputed() -> None:
    """The threshold is *strictly* greater than 3, so 3-of-9 still imputes."""
    features = _synthetic_features(n=10)
    for col in ["acousticness", "danceability", "energy"]:
        features.loc[4, col] = np.nan

    X, index, _, log = prepare_matrix(features)

    assert len(X) == 10
    assert "sp004" in index[SPOTIFY_ID].tolist()
    assert log.dropped_tracks.empty
    # Three imputations were recorded, one per knocked-out column.
    assert sum(log.imputed_counts.values()) == 3


def test_transform_log_records_logit_and_log_columns() -> None:
    """The log lists which columns went through which transform so the
    coverage report can explain it."""
    features = _synthetic_features(n=5)

    _, _, _, log = prepare_matrix(features)

    assert set(log.logit_columns) == set(BOUNDED_FEATURES)
    assert list(log.log_columns) == ["tempo"]


def test_empty_features_frame_raises_value_error() -> None:
    """An empty modelling matrix is a user-facing error, not a silent
    success — there is nothing to cluster."""
    empty = pd.DataFrame(columns=[SPOTIFY_ID, TITLE, ARTIST, *FEATURE_COLUMNS])

    with pytest.raises(ValueError, match="empty"):
        prepare_matrix(empty)
