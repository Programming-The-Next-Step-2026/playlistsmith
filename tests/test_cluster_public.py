"""Tests for the public :func:`playlistsmith.cluster.cluster` entry point.

This is the function downstream code (playlist export, viz) actually
calls. It wraps the whole pipeline:

- :func:`prepare_matrix` (preprocessing + dropped/imputed accounting),
- :func:`fit_gmm` (or another method when the user asks),
- canonical cluster ordering (descending size, ties broken by PC1
  of the cluster mean) so cluster ``0`` means roughly the same thing
  across re-runs,
- playlist-shape post-processing: small clusters collapse into an
  ``Unclassified`` bucket (cluster id ``-1``); over-large clusters
  emit a warning,
- :func:`describe_clusters` for human-readable labels.

The tests construct deterministic synthetic features (no API calls) so
the right answer is known by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from playlistsmith.cluster import cluster
from playlistsmith.cluster.algorithms import ClusteringResult
from playlistsmith.cluster.preprocess import TransformLog
from playlistsmith.cluster.public import ClusterPipelineResult
from playlistsmith.features.reccobeats import FEATURE_COLUMNS
from playlistsmith.io.csv_loader import ARTIST, SPOTIFY_ID, TITLE


def _synthetic_features(
    cluster_sizes: tuple[int, ...] = (50, 50, 50), seed: int = 0
) -> pd.DataFrame:
    """A features frame with controllable blob sizes.

    Each blob has a distinctive centre across the bounded features so
    GMM cleanly recovers the right ``k``. Identity columns are filled
    with deterministic placeholder strings.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for blob_idx, n in enumerate(cluster_sizes):
        # Bounded-feature centre and tempo/loudness shift unique to this blob.
        bounded_mean = 0.1 + 0.2 * blob_idx
        tempo_mean = 80.0 + 30.0 * blob_idx
        loudness_mean = -12.0 + 4.0 * blob_idx
        for _ in range(n):
            row: dict[str, object] = {
                SPOTIFY_ID: f"sp{len(rows):04d}",
                TITLE: f"Track {len(rows)}",
                ARTIST: f"Artist {len(rows) % 7}",
            }
            for col in (
                "acousticness",
                "danceability",
                "energy",
                "instrumentalness",
                "liveness",
                "speechiness",
                "valence",
            ):
                # Beta-ish centred near `bounded_mean`, clipped strictly inside (0, 1).
                v = float(np.clip(rng.normal(bounded_mean, 0.05), 0.01, 0.99))
                row[col] = v
            row["tempo"] = float(np.clip(rng.normal(tempo_mean, 3.0), 40.0, 200.0))
            row["loudness"] = float(rng.normal(loudness_mean, 1.0))
            rows.append(row)
    return pd.DataFrame(
        rows, columns=[SPOTIFY_ID, TITLE, ARTIST, *FEATURE_COLUMNS]
    )


# --------------------------------------------------------------------------- #
# Return shape                                                                #
# --------------------------------------------------------------------------- #


def test_returns_cluster_pipeline_result() -> None:
    """The public entry point returns the documented dataclass."""
    features = _synthetic_features()

    out = cluster(features, random_state=0)

    assert isinstance(out, ClusterPipelineResult)


def test_pipeline_result_has_all_required_fields() -> None:
    """Attributes of the result dataclass are of the correct type."""
    features = _synthetic_features()

    out = cluster(features, random_state=0)

    assert isinstance(out.tracks, pd.DataFrame)
    assert isinstance(out.descriptions, pd.DataFrame)
    assert isinstance(out.clustering, ClusteringResult)
    assert isinstance(out.transform_log, TransformLog)
    assert isinstance(out.warnings, list)


def test_tracks_frame_has_identity_plus_cluster_plus_summary() -> None:
    """One row per surviving track, with the identity columns and the
    final cluster id + summary."""
    features = _synthetic_features()

    out = cluster(features, random_state=0)

    assert list(out.tracks.columns) == [
        SPOTIFY_ID,
        TITLE,
        ARTIST,
        "cluster",
        "cluster_summary",
    ]


def test_track_count_equals_preprocessing_survivor_count() -> None:
    """Tracks frame has one row per track that made it through
    preprocessing (i.e. total minus rows dropped for >3 missing)."""
    features = _synthetic_features(cluster_sizes=(50, 50, 50))

    out = cluster(features, random_state=0)

    assert len(out.tracks) == len(features) - len(out.transform_log.dropped_tracks)


# --------------------------------------------------------------------------- #
# Method dispatch                                                             #
# --------------------------------------------------------------------------- #


def test_default_method_is_gmm() -> None:
    """Default ``method`` must be GMM, and the raw clustering result reflects that."""
    features = _synthetic_features()

    out = cluster(features, random_state=0)

    # GMM is the only method that produces posteriors.
    assert out.clustering.posteriors is not None


def test_unknown_method_raises_value_error() -> None:
    """An unrecognised ``method`` is a user-facing error."""
    features = _synthetic_features()

    with pytest.raises(ValueError, match="method"):
        cluster(features, method="not-a-real-method", random_state=0)


def test_kmeans_method_dispatches_to_fit_kmeans() -> None:
    """``method='kmeans'`` runs through K-Means; ``posteriors`` is
    ``None`` and ``covariance_type`` reflects the variant."""
    features = _synthetic_features()

    out = cluster(features, method="kmeans", random_state=0, k_range=range(2, 5))

    assert out.clustering.posteriors is None
    assert out.clustering.covariance_type == "kmeans"


def test_hdbscan_method_dispatches_to_fit_hdbscan() -> None:
    """``method='hdbscan'`` runs through HDBSCAN; the noise-rate field
    is populated and the variant marker reflects it."""
    features = _synthetic_features()

    out = cluster(features, method="hdbscan", random_state=0)

    assert out.clustering.covariance_type == "hdbscan"
    assert 0.0 <= out.clustering.noise_rate <= 1.0


# --------------------------------------------------------------------------- #
# Canonical cluster ordering                                                  #
# --------------------------------------------------------------------------- #


def test_clusters_are_relabelled_by_descending_size() -> None:
    """Cluster ``0`` is the biggest playlist."""
    features = _synthetic_features(cluster_sizes=(80, 40, 20))

    # Constrain k to keep this test focused on the relabelling step,
    # not on BIC over-segmenting the tight synthetic blobs.
    out = cluster(features, random_state=0, k_range=range(2, 5))

    sizes = out.tracks["cluster"].value_counts().sort_index()
    # Drop the Unclassified bucket if present so we compare real clusters.
    sizes = sizes[sizes.index >= 0]
    # Non-increasing by cluster id.
    sizes_list = sizes.tolist()
    assert sizes_list == sorted(sizes_list, reverse=True)


def test_descriptions_index_aligns_with_relabelled_clusters() -> None:
    """``descriptions`` references the same cluster ids that appear in
    the ``tracks`` frame."""
    features = _synthetic_features()

    out = cluster(features, random_state=0)

    track_ids = set(out.tracks["cluster"].unique())
    desc_ids = set(out.descriptions["cluster"].unique())
    assert track_ids == desc_ids


# --------------------------------------------------------------------------- #
# Playlist-shape post-processing                                              #
# --------------------------------------------------------------------------- #


def test_small_cluster_is_collapsed_into_unclassified_bucket() -> None:
    """Clusters below ``min_playlist_size`` move to cluster id ``-1``
    rather than producing a useless 2-track playlist."""
    # Two large blobs and a tiny one. Place the tiny blob at extreme
    # feature values so GMM gives it its own component, then test that
    # the collapse step moves it into Unclassified.
    features = _synthetic_features(cluster_sizes=(60, 60, 3))
    # Push the 3-track blob to an extreme corner of feature space.
    tiny_idx = features.index[-3:]
    for col in (
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "speechiness",
        "valence",
    ):
        features.loc[tiny_idx, col] = 0.95
    features.loc[tiny_idx, "tempo"] = 190.0
    features.loc[tiny_idx, "loudness"] = 0.0

    out = cluster(
        features, random_state=0, min_playlist_size=5, k_range=[3]
    )

    # The unclassified bucket exists in tracks and descriptions.
    assert -1 in out.tracks["cluster"].tolist()
    unclassified_size = int((out.tracks["cluster"] == -1).sum())
    assert unclassified_size >= 3
    assert -1 in out.descriptions["cluster"].tolist()
    unclassified_summary = out.descriptions.set_index("cluster").loc[-1][
        "cluster_summary" if "cluster_summary" in out.descriptions.columns
        else "summary"
    ]
    assert "unclassified" in str(unclassified_summary).lower()


def test_no_unclassified_bucket_when_all_clusters_meet_minimum() -> None:
    """If every cluster meets ``min_playlist_size``, no ``-1`` rows are
    introduced."""
    features = _synthetic_features(cluster_sizes=(40, 40, 40))

    # Force k=3 so model selection cannot over-segment and create a
    # tiny sub-cluster; this test is about the collapse step's no-op
    # behaviour when every cluster is large enough.
    out = cluster(features, random_state=0, min_playlist_size=5, k_range=[3])

    assert -1 not in out.tracks["cluster"].tolist()


def test_all_tracks_unclassified_emits_warning() -> None:
    """If post-processing collapses *every* track into the Unclassified
    bucket, the result is useless — surface a warning so the user
    knows to lower ``min_playlist_size`` or change method."""
    # 30 tracks; require playlists of at least 200. Everything collapses.
    features = _synthetic_features(cluster_sizes=(10, 10, 10))

    out = cluster(
        features, random_state=0, min_playlist_size=200, k_range=range(2, 5)
    )

    assert all(c == -1 for c in out.tracks["cluster"].tolist())
    assert any(
        "unclassified" in w.lower() or "no playlists" in w.lower()
        for w in out.warnings
    )


def test_dominant_cluster_triggers_warning_but_is_not_split() -> None:
    """A cluster holding more than ``max_playlist_share`` of the library
    is flagged in ``warnings`` (plan: warn, don't auto-resplit)."""
    # 200 tracks in one blob, 20 each in two others → biggest cluster ≈ 83%.
    features = _synthetic_features(cluster_sizes=(200, 20, 20))

    out = cluster(features, random_state=0, max_playlist_share=0.5)

    assert any("share" in w.lower() or "dominan" in w.lower() for w in out.warnings)
    # The dominant cluster was not split — we still have at most as many
    # clusters as we asked for.
    assert out.clustering.k <= 12


# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #


def test_same_random_state_produces_same_pipeline_output() -> None:
    """Threading ``random_state`` through the pipeline must be enough to
    reproduce the labels."""
    features = _synthetic_features()

    out1 = cluster(features, random_state=42)
    out2 = cluster(features, random_state=42)

    pd.testing.assert_series_equal(
        out1.tracks["cluster"].reset_index(drop=True),
        out2.tracks["cluster"].reset_index(drop=True),
        check_names=False,
    )
