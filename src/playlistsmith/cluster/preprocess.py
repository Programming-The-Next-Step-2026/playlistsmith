"""Shared preprocessing for the clustering pipeline.

This module is the single entry point that every clustering algorithm
goes through, so they all see identical input. The contract is described
in ``plans/clustering_analysis_revised.md`` Section 0; in short:

- The seven bounded ``[0, 1]`` features (:data:`BOUNDED_FEATURES`) are
  logit-transformed before scaling, with a small epsilon clip so that
  raw ``0`` / ``1`` values do not blow up.
- ``tempo`` is log-transformed (BPM is heavy-tailed and perceived
  multiplicatively).
- ``loudness`` is already in dB; we only z-score it.
- Per-feature missing cells are filled with that feature's median.
- Rows missing more than 3 of the 9 features are dropped and surfaced
  on the returned :class:`TransformLog` so the coverage report can name
  them.

The output of :func:`prepare_matrix` is consumed by
:mod:`playlistsmith.cluster.algorithms`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from playlistsmith.features.reccobeats import FEATURE_COLUMNS
from playlistsmith.io.csv_loader import ARTIST, SPOTIFY_ID, TITLE

__all__ = [
    "BOUNDED_FEATURES",
    "LOG_FEATURES",
    "TransformLog",
    "prepare_matrix",
]

#: Features bounded to ``[0, 1]`` (logit-transformed before scaling).
BOUNDED_FEATURES: tuple[str, ...] = (
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "speechiness",
    "valence",
)

#: Features that are log-transformed before scaling. ``loudness`` is
#: already in dB and is *not* in this list.
LOG_FEATURES: tuple[str, ...] = ("tempo",)

#: Epsilon used to clip bounded features off the ``{0, 1}`` boundary
#: before logit so the transform stays finite.
_LOGIT_EPSILON: float = 1e-3

#: Identity columns carried alongside the modelling matrix.
_IDENTITY_COLUMNS: tuple[str, str, str] = (SPOTIFY_ID, TITLE, ARTIST)

#: Maximum number of missing features a row can have and still be kept.
#: A row with strictly more than this many NaNs is dropped.
_MAX_MISSING_PER_ROW: int = 3


@dataclass
class TransformLog:
    """Record of what :func:`prepare_matrix` did to the input frame.

    The clustering subpackage hands this to the coverage report so users
    can see how much imputation and dropping happened, and which columns
    were logit/log transformed.

    Attributes:
        logit_columns: Columns logit-transformed before scaling.
        log_columns: Columns log-transformed before scaling.
        imputed_counts: ``{column: n_cells_imputed}`` for any feature
            that had missing values filled with the global median.
            Columns with zero imputations are omitted.
        dropped_tracks: Identity frame (``spotify_id``, ``title``,
            ``artist``) for rows dropped because more than 3 of 9
            features were missing. Excluded from equality/repr.

    Examples:
        Returned as the fourth element of :func:`prepare_matrix` (see its
        example for how ``features`` is built):

        >>> X, index, scaler, log = prepare_matrix(features)
        >>> log.log_columns
        ('tempo',)
        >>> log.imputed_counts  # nothing missing in this synthetic frame
        {}
        >>> log.dropped_tracks.empty
        True
    """

    logit_columns: tuple[str, ...]
    log_columns: tuple[str, ...]
    imputed_counts: dict[str, int]
    dropped_tracks: pd.DataFrame = field(compare=False, repr=False)


def prepare_matrix(
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, TransformLog]:
    """Turn a unified features frame into a model-ready matrix.

    Steps, in order: drop rows with >3 missing features, median-impute
    the remaining missing cells, logit-transform :data:`BOUNDED_FEATURES`
    (with boundary clipping), log-transform :data:`LOG_FEATURES`, then
    z-score every column with a fitted ``StandardScaler``.

    Identity columns (``spotify_id``, ``title``, ``artist``) do not
    enter the modelling matrix; they are returned in a side frame so
    cluster labels can be re-attached by row position.

    Args:
        features_df: A unified features frame as produced by
            :func:`playlistsmith.features.extract`. Must contain the
            three identity columns and all nine
            :data:`~playlistsmith.features.reccobeats.FEATURE_COLUMNS`.

    Returns:
        A ``(X, index, scaler, transform_log)`` tuple. ``X`` is the
        transformed-and-scaled modelling matrix as a DataFrame whose
        columns are :data:`~playlistsmith.features.reccobeats.FEATURE_COLUMNS`
        in canonical order. ``index`` is the row-aligned identity frame.
        ``scaler`` is the fitted :class:`~sklearn.preprocessing.StandardScaler`.
        ``transform_log`` records what was imputed, dropped, and which
        non-linear transforms were applied.

    Raises:
        ValueError: If ``features_df`` is empty, or if any of the nine
            feature columns are missing from the input.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from playlistsmith.cluster import prepare_matrix
        >>> cols = ["acousticness", "danceability", "energy", "instrumentalness",
        ...         "liveness", "loudness", "speechiness", "tempo", "valence"]
        >>> sd = [0.02, 0.02, 0.02, 0.02, 0.02, 1.0, 0.02, 3.0, 0.02]
        >>> mellow = [0.88, 0.30, 0.20, 0.40, 0.15, -17.0, 0.05, 85.0, 0.25]
        >>> upbeat = [0.12, 0.80, 0.90, 0.40, 0.15, -6.0, 0.05, 128.0, 0.75]
        >>> rng = np.random.default_rng(5)
        >>> features = pd.DataFrame(
        ...     np.vstack([rng.normal(mellow, sd, (15, 9)),
        ...                rng.normal(upbeat, sd, (15, 9))]), columns=cols)
        >>> features.insert(0, "artist", "Various")
        >>> features.insert(0, "title", [f"Track {i}" for i in range(30)])
        >>> features.insert(0, "spotify_id", [f"id{i:02d}" for i in range(30)])
        >>> X, index, scaler, log = prepare_matrix(features)
        >>> X.shape  # (n_tracks, n_features); identity columns split out
        (30, 9)
        >>> index.columns.tolist()
        ['spotify_id', 'title', 'artist']
        >>> log.logit_columns[:3]
        ('acousticness', 'danceability', 'energy')
    """
    if features_df.empty:
        raise ValueError("Cannot prepare an empty features frame for clustering.")

    missing_columns = [c for c in FEATURE_COLUMNS if c not in features_df.columns]
    if missing_columns:
        raise ValueError(
            f"features_df is missing required columns: {missing_columns!r}"
        )

    # Coerce features to float so NaN-handling is well-defined.
    feats = features_df[list(FEATURE_COLUMNS)].astype(float).reset_index(drop=True)
    identity = (
        features_df[list(_IDENTITY_COLUMNS)].reset_index(drop=True).copy()
    )

    # Drop rows with too many missing features and remember which ones.
    missing_per_row = feats.isna().sum(axis=1)
    keep_mask = missing_per_row <= _MAX_MISSING_PER_ROW
    dropped_tracks = identity.loc[~keep_mask].reset_index(drop=True)
    feats = feats.loc[keep_mask].reset_index(drop=True)
    index = identity.loc[keep_mask].reset_index(drop=True)

    if feats.empty:
        raise ValueError(
            "All tracks were dropped during preprocessing (>3 missing "
            "features per row). Nothing left to cluster."
        )

    # Median-impute the remaining cells, recording counts per column.
    imputed_counts: dict[str, int] = {}
    for col in FEATURE_COLUMNS:
        missing = feats[col].isna()
        n_missing = int(missing.sum())
        if n_missing > 0:
            imputed_counts[col] = n_missing
            feats.loc[missing, col] = feats[col].median()

    # Apply non-linear transforms before scaling.
    for col in BOUNDED_FEATURES:
        clipped = feats[col].clip(lower=_LOGIT_EPSILON, upper=1.0 - _LOGIT_EPSILON)
        feats[col] = np.log(clipped / (1.0 - clipped))
    for col in LOG_FEATURES:
        feats[col] = np.log(feats[col])

    # Z-score everything. StandardScaler fitted here is part of the
    # public return so callers (e.g. diagnostics) can invert it.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats.to_numpy())
    X = pd.DataFrame(scaled, columns=list(FEATURE_COLUMNS))

    transform_log = TransformLog(
        logit_columns=BOUNDED_FEATURES,
        log_columns=LOG_FEATURES,
        imputed_counts=imputed_counts,
        dropped_tracks=dropped_tracks,
    )
    return X, index, scaler, transform_log
