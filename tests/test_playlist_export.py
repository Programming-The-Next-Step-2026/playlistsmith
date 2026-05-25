"""Tests for :mod:`playlistsmith.io.playlist_export`.

The export module's job is small but load-bearing: take a
``ClusterPipelineResult`` (or just its ``tracks`` frame) and write one
Exportify-shaped CSV per cluster so the user can re-import the playlist
into Spotify. The tests cover the contract that downstream consumers
(notebook, GUI) depend on: filenames, schema, Unclassified handling,
naming overrides, and the combined-CSV option.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from playlistsmith.io import playlist_export
from playlistsmith.io.playlist_export import (
    DEFAULT_COMBINED_FILENAME,
    EXPORTIFY_ARTIST,
    EXPORTIFY_TITLE,
    EXPORTIFY_URI,
    UNCLASSIFIED_FILENAME,
)


def _make_tracks(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a minimal ``result.tracks`` frame for the export tests."""
    return pd.DataFrame(
        rows,
        columns=["spotify_id", "title", "artist", "cluster", "cluster_summary"],
    )


@pytest.fixture
def tracks() -> pd.DataFrame:
    return _make_tracks(
        [
            {"spotify_id": "id000A1", "title": "Drift", "artist": "The Placeholders",
             "cluster": 0, "cluster_summary": "high danceability"},
            {"spotify_id": "id000A2", "title": "Quiet Tide", "artist": "Sample Collective",
             "cluster": 0, "cluster_summary": "high danceability"},
            {"spotify_id": "id000B1", "title": "Slow Rain", "artist": "Stub Choir",
             "cluster": 1, "cluster_summary": "low energy"},
            {"spotify_id": "id000X1", "title": "Null Wave", "artist": "Unknown Provider",
             "cluster": -1, "cluster_summary": "unclassified"},
        ]
    )


def test_writes_one_csv_per_real_cluster(tmp_path: Path, tracks: pd.DataFrame) -> None:
    paths = playlist_export.write_cluster_csvs(tracks, output_dir=tmp_path)

    # One file per real cluster (Unclassified excluded by default).
    assert len(paths) == 2
    assert all(p.parent == tmp_path for p in paths)
    assert all(p.suffix == ".csv" for p in paths)
    # Unclassified is not written by default.
    assert not (tmp_path / UNCLASSIFIED_FILENAME).exists()


def test_csv_schema_has_track_uri_first(tmp_path: Path, tracks: pd.DataFrame) -> None:
    paths = playlist_export.write_cluster_csvs(tracks, output_dir=tmp_path)
    df = pd.read_csv(paths[0])

    # Exportify-compatible: Track URI / Track Name / Artist Name(s) at the front.
    assert list(df.columns[:3]) == [EXPORTIFY_URI, EXPORTIFY_TITLE, EXPORTIFY_ARTIST]
    # URIs are full spotify:track:<id> form, not bare IDs.
    assert df[EXPORTIFY_URI].iloc[0].startswith("spotify:track:")
    # Cluster id + summary travel along so a human can read the file.
    assert "Cluster" in df.columns
    assert "Cluster Summary" in df.columns


def test_default_naming_uses_cluster_id(tmp_path: Path, tracks: pd.DataFrame) -> None:
    paths = playlist_export.write_cluster_csvs(tracks, output_dir=tmp_path)
    names = sorted(p.name for p in paths)
    assert names == ["cluster_0.csv", "cluster_1.csv"]


def test_naming_dict_overrides_default(tmp_path: Path, tracks: pd.DataFrame) -> None:
    paths = playlist_export.write_cluster_csvs(
        tracks,
        output_dir=tmp_path,
        naming={0: "Workout", 1: "Sunday Morning"},
    )
    names = sorted(p.name for p in paths)
    assert names == ["Sunday Morning.csv", "Workout.csv"]


def test_naming_callable_overrides_default(tmp_path: Path, tracks: pd.DataFrame) -> None:
    paths = playlist_export.write_cluster_csvs(
        tracks, output_dir=tmp_path, naming=lambda c: f"playlist-{c:02d}"
    )
    names = sorted(p.name for p in paths)
    assert names == ["playlist-00.csv", "playlist-01.csv"]


def test_partial_naming_falls_back_to_default(tmp_path: Path, tracks: pd.DataFrame) -> None:
    paths = playlist_export.write_cluster_csvs(
        tracks, output_dir=tmp_path, naming={0: "Workout"}
    )
    names = sorted(p.name for p in paths)
    assert names == ["Workout.csv", "cluster_1.csv"]


def test_unsafe_naming_raises(tmp_path: Path, tracks: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        playlist_export.write_cluster_csvs(
            tracks, output_dir=tmp_path, naming={0: "../escape"}
        )


def test_duplicate_naming_raises(tmp_path: Path, tracks: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unique"):
        playlist_export.write_cluster_csvs(
            tracks, output_dir=tmp_path, naming={0: "Same", 1: "Same"}
        )


def test_include_unclassified_writes_extra_file(tmp_path: Path, tracks: pd.DataFrame) -> None:
    paths = playlist_export.write_cluster_csvs(
        tracks, output_dir=tmp_path, include_unclassified=True
    )
    names = sorted(p.name for p in paths)
    assert UNCLASSIFIED_FILENAME in names
    assert len(paths) == 3


def test_features_merged_when_provided(tmp_path: Path, tracks: pd.DataFrame) -> None:
    features = pd.DataFrame(
        {
            "spotify_id": ["id000A1", "id000A2", "id000B1", "id000X1"],
            "title": ["Drift", "Quiet Tide", "Slow Rain", "Null Wave"],
            "artist": ["The Placeholders", "Sample Collective", "Stub Choir", "Unknown"],
            "acousticness": [0.8, 0.7, 0.2, 0.0],
            "danceability": [0.4, 0.5, 0.6, 0.0],
            "energy": [0.2, 0.3, 0.8, 0.0],
            "instrumentalness": [0.1, 0.1, 0.0, 0.0],
            "liveness": [0.1, 0.1, 0.2, 0.0],
            "loudness": [-18.0, -16.0, -6.0, 0.0],
            "speechiness": [0.05, 0.04, 0.06, 0.0],
            "tempo": [85.0, 88.0, 120.0, 0.0],
            "valence": [0.5, 0.6, 0.7, 0.0],
        }
    )
    paths = playlist_export.write_cluster_csvs(
        tracks, output_dir=tmp_path, features_df=features
    )
    df = pd.read_csv(paths[0])
    for feat in (
        "acousticness", "danceability", "energy", "instrumentalness",
        "liveness", "loudness", "speechiness", "tempo", "valence",
    ):
        assert feat in df.columns, f"missing feature column {feat}"


def test_combined_csv_includes_unclassified(tmp_path: Path, tracks: pd.DataFrame) -> None:
    paths = playlist_export.write_cluster_csvs(
        tracks, output_dir=tmp_path, write_combined=True
    )
    combined = tmp_path / DEFAULT_COMBINED_FILENAME
    assert combined.exists()
    df = pd.read_csv(combined)
    # Combined file contains every track including Unclassified.
    assert len(df) == len(tracks)
    assert (df["Cluster"] == -1).any()


def test_accepts_cluster_pipeline_result(tmp_path: Path, tracks: pd.DataFrame) -> None:
    # Lazy import: avoid pulling sklearn/umap if the test runner skips this file.
    from playlistsmith.cluster.public import ClusterPipelineResult

    # Build a minimal result whose only used attribute is .tracks. Other
    # fields stay None / empty — write_cluster_csvs must not touch them.
    result = ClusterPipelineResult.__new__(ClusterPipelineResult)
    result.tracks = tracks
    result.descriptions = pd.DataFrame()
    result.clustering = None  # type: ignore[assignment]
    result.transform_log = None  # type: ignore[assignment]
    result.diagnostics = None  # type: ignore[assignment]
    result.warnings = []

    paths = playlist_export.write_cluster_csvs(result, output_dir=tmp_path)
    assert len(paths) == 2


def test_creates_output_dir_if_missing(tmp_path: Path, tracks: pd.DataFrame) -> None:
    target = tmp_path / "fresh_subdir"
    paths = playlist_export.write_cluster_csvs(tracks, output_dir=target)
    assert target.is_dir()
    assert all(p.parent == target for p in paths)


def test_empty_tracks_returns_no_paths(tmp_path: Path) -> None:
    empty = _make_tracks([])
    paths = playlist_export.write_cluster_csvs(empty, output_dir=tmp_path)
    assert paths == []
