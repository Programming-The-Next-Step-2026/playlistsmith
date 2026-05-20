"""Tests for :class:`playlistsmith.io.csv_loader.TrackLibrary`."""

from __future__ import annotations

from pathlib import Path

import pytest

from playlistsmith.io import TrackLibrary
from playlistsmith.io.csv_loader import ARTIST, SPOTIFY_ID, TITLE
from tests import EXAMPLE_CSV, _row, _write_csv


def test_loads_example_csv() -> None:
    """The real example export parses into three tidy columns."""
    lib = TrackLibrary(EXAMPLE_CSV)
    df = lib.dataframe

    assert list(df.columns) == [TITLE, ARTIST, SPOTIFY_ID]
    assert len(lib) == 3
    assert df.loc[0, TITLE] == "Synthetic Sunrise"
    assert df.loc[0, SPOTIFY_ID] == "test0000000000000000001"


def test_multi_artist_string_is_preserved(tmp_path: Path) -> None:
    """Comma-separated artist names are kept verbatim, not split."""
    csv = _write_csv(
        tmp_path / "lib.csv",
        [_row("spotify:track:123", "Test Song", "Test, Artist")],
    )
    df = TrackLibrary(csv).dataframe

    assert df.loc[0, ARTIST] == "Test, Artist"
    assert df.loc[0, SPOTIFY_ID] == "123"


def test_whitespace_is_stripped(tmp_path: Path) -> None:
    """Surrounding whitespace in title/artist/URI is trimmed."""
    csv = _write_csv(
        tmp_path / "lib.csv",
        [_row("  spotify:track:zzz  ", "  Song  ", "  Artist  ")],
    )
    df = TrackLibrary(csv).dataframe

    assert df.loc[0, TITLE] == "Song"
    assert df.loc[0, ARTIST] == "Artist"
    assert df.loc[0, SPOTIFY_ID] == "zzz"


def test_rows_without_spotify_id_are_dropped(tmp_path: Path) -> None:
    """Rows whose Track URI lacks a Spotify track ID are removed."""
    csv = _write_csv(
        tmp_path / "lib.csv",
        [
            _row("spotify:track:keep", "Keep", "A"),
            _row("", "Local File", "B"),
        ],
    )
    lib = TrackLibrary(csv)

    assert len(lib) == 1
    assert lib.dataframe.loc[0, SPOTIFY_ID] == "keep"


def test_missing_file_raises_file_not_found() -> None:
    """A non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        TrackLibrary("does/not/exist.csv")


def test_missing_required_column_raises_value_error(tmp_path: Path) -> None:
    """A CSV without a required column raises ValueError."""
    csv = tmp_path / "bad.csv"
    csv.write_text(
        "Track URI;Artist Name(s)\nspotify:track:x;A\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing required column"):
        TrackLibrary(csv)


def test_no_usable_rows_raises_value_error(tmp_path: Path) -> None:
    """A CSV with no Spotify track IDs raises ValueError."""
    csv = _write_csv(tmp_path / "empty.csv", [_row("", "Local", "A")])
    with pytest.raises(ValueError, match="No tracks with a Spotify track ID"):
        TrackLibrary(csv)


def test_dataframe_property_returns_defensive_copy() -> None:
    """Mutating the returned frame does not affect the library."""
    lib = TrackLibrary(EXAMPLE_CSV)
    df = lib.dataframe
    df.loc[0, TITLE] = "MUTATED"

    assert lib.dataframe.loc[0, TITLE] != "MUTATED"


def test_display_prints_and_returns_none(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """display() prints every track and returns nothing (no double render)."""
    lib = TrackLibrary(EXAMPLE_CSV)
    returned = lib.display()

    out = capsys.readouterr().out
    assert "Synthetic Sunrise" in out
    assert "Sample Collective, Dummy Vox" in out
    assert returned is None


def test_display_truncates_when_over_max_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """display(max_rows=n) elides the middle when the library exceeds n."""
    rows = [
        _row(f"spotify:track:id{i:019d}", f"Song {i}", "Artist")
        for i in range(10)
    ]
    lib = TrackLibrary(_write_csv(tmp_path / "big.csv", rows))

    lib.display(max_rows=4)
    out = capsys.readouterr().out

    assert "..." in out  # middle rows elided
    assert "Song 0" in out  # head still shown
    assert "Song 9" in out  # tail still shown
    assert "Song 5" not in out  # an elided middle row is absent


def test_display_none_prints_all_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """display(max_rows=None) prints every row without eliding."""
    rows = [
        _row(f"spotify:track:id{i:019d}", f"Song {i}", "Artist")
        for i in range(10)
    ]
    lib = TrackLibrary(_write_csv(tmp_path / "big.csv", rows))

    lib.display(max_rows=None)
    out = capsys.readouterr().out

    assert "..." not in out
    assert all(f"Song {i}" in out for i in range(10))


def test_extract_features_delegates_to_features_extract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """extract_features() routes through features.extract (precomputed)."""
    import playlistsmith.features as features

    captured: dict[str, object] = {}
    sentinel = ("FRAME", "COVERAGE")

    def fake_extract(tracks: object, mode: str) -> tuple[str, str]:
        captured["columns"] = list(tracks.columns)  # type: ignore[attr-defined]
        captured["mode"] = mode
        return sentinel

    monkeypatch.setattr(features, "extract", fake_extract)

    lib = TrackLibrary(EXAMPLE_CSV)
    result = lib.extract_features()

    assert result == sentinel
    assert captured["mode"] == "precomputed"
    assert captured["columns"] == [TITLE, ARTIST, SPOTIFY_ID]


def test_repr_includes_source_and_count() -> None:
    """repr() exposes the source path and track count."""
    lib = TrackLibrary(EXAMPLE_CSV)
    text = repr(lib)

    assert "TrackLibrary" in text
    assert "tracks=3" in text
