"""Tests for the visualisation hooks emitted by the cluster pipeline.

Plan §6: the cluster step does not import any plotting library, but it
*does* hand the viz layer everything it needs to draw scatter plots,
heatmaps and selection diagnostics. The artefacts live on a
:class:`ClusterDiagnostics` bundle attached to
:class:`ClusterPipelineResult`:

- ``pca_components`` — deterministic numeric coordinates for each
  track in PCA space (cheap, used for the canonical cluster ordering
  and as a fallback for plotting);
- ``projection_2d`` — a 2-D embedding for scatter plots (t-SNE; UMAP
  can replace this later without changing the call sites);
- ``zprofile_heatmap`` — ``cluster × feature`` z-profile frame
  consumed by heatmap viz;
- BIC / ICL / silhouette / inertia / CH curves are already on the
  raw :class:`ClusteringResult` (Sections 1 and 2).

These tests pin shapes and row-alignment with the tracks frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from playlistsmith.cluster import cluster
from playlistsmith.cluster.public import ClusterDiagnostics
from playlistsmith.features.reccobeats import FEATURE_COLUMNS
from playlistsmith.io.csv_loader import ARTIST, SPOTIFY_ID, TITLE


def _synthetic_features(
    cluster_sizes: tuple[int, ...] = (50, 50, 50), seed: int = 0
) -> pd.DataFrame:
    """Same as the synthetic generator used in the public-cluster tests."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for blob_idx, n in enumerate(cluster_sizes):
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
                row[col] = float(np.clip(rng.normal(bounded_mean, 0.05), 0.01, 0.99))
            row["tempo"] = float(np.clip(rng.normal(tempo_mean, 3.0), 40.0, 200.0))
            row["loudness"] = float(rng.normal(loudness_mean, 1.0))
            rows.append(row)
    return pd.DataFrame(
        rows, columns=[SPOTIFY_ID, TITLE, ARTIST, *FEATURE_COLUMNS]
    )


# --------------------------------------------------------------------------- #
# Diagnostics bundle                                                          #
# --------------------------------------------------------------------------- #


def test_pipeline_result_carries_diagnostics_bundle() -> None:
    """``ClusterPipelineResult.diagnostics`` exists and is the right type."""
    features = _synthetic_features()

    out = cluster(features, random_state=0, k_range=range(2, 5))

    assert isinstance(out.diagnostics, ClusterDiagnostics)


# --------------------------------------------------------------------------- #
# PCA components                                                              #
# --------------------------------------------------------------------------- #


def test_pca_components_have_one_row_per_track() -> None:
    """PCA coordinates are row-aligned with ``tracks`` so the viz layer
    can colour by cluster id without re-indexing."""
    features = _synthetic_features()

    out = cluster(features, random_state=0, k_range=range(2, 5))

    assert len(out.diagnostics.pca_components) == len(out.tracks)


def test_pca_components_have_principal_component_columns() -> None:
    """Columns are ``pc1``, ``pc2``, ... so callers can pick the first
    few without needing to know how many we computed."""
    features = _synthetic_features()

    out = cluster(features, random_state=0, k_range=range(2, 5))

    cols = list(out.diagnostics.pca_components.columns)
    assert cols[0] == "pc1"
    assert cols[1] == "pc2"
    assert all(c.startswith("pc") for c in cols)


def test_pca_explained_variance_ratios_are_recorded() -> None:
    """The viz layer can label axes with explained-variance share."""
    features = _synthetic_features()

    out = cluster(features, random_state=0, k_range=range(2, 5))

    evrs = out.diagnostics.pca_explained_variance_ratio
    assert len(evrs) == out.diagnostics.pca_components.shape[1]
    assert all(0.0 <= r <= 1.0 for r in evrs)
    assert sum(evrs) <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# 2-D projection                                                              #
# --------------------------------------------------------------------------- #


def test_projection_2d_is_n_by_2() -> None:
    """A scatter plot needs exactly two coordinates per track."""
    features = _synthetic_features()

    out = cluster(features, random_state=0, k_range=range(2, 5))

    proj = out.diagnostics.projection_2d
    assert proj.shape == (len(out.tracks), 2)
    assert list(proj.columns) == ["dim1", "dim2"]


def test_projection_method_is_recorded() -> None:
    """Callers (and docs) can tell whether they got t-SNE or UMAP."""
    features = _synthetic_features()

    out = cluster(features, random_state=0, k_range=range(2, 5))

    assert out.diagnostics.projection_method in {"tsne", "umap"}


def test_projection_3d_is_n_by_3() -> None:
    """The 3-D scatter (plan §6.2) needs three PCA coordinates per track,
    row-aligned with the tracks frame."""
    features = _synthetic_features()

    out = cluster(features, random_state=0, k_range=range(2, 5))

    proj = out.diagnostics.projection_3d
    assert proj is not None
    assert proj.shape == (len(out.tracks), 3)
    assert list(proj.columns) == ["pc1", "pc2", "pc3"]


# --------------------------------------------------------------------------- #
# Heatmap frame                                                               #
# --------------------------------------------------------------------------- #


def test_zprofile_heatmap_is_k_by_nine_features() -> None:
    """Heatmap-ready frame: one row per cluster id appearing in tracks
    (including the Unclassified bucket if present), one column per
    feature."""
    features = _synthetic_features()

    out = cluster(features, random_state=0, k_range=range(2, 5))

    hm = out.diagnostics.zprofile_heatmap
    cluster_ids = sorted(set(out.tracks["cluster"].tolist()))
    assert list(hm.index) == cluster_ids
    assert list(hm.columns) == list(FEATURE_COLUMNS)
