"""Shared test fixtures and helpers for the playlistsmith test suite.

Centralises constants and CSV-builders used across test modules so individual
test files stay focused on assertions.
"""

from __future__ import annotations

from pathlib import Path

# Path to the synthetic example track list checked into the tests/ directory.
# Fully fabricated (Exportify-shaped) data — no real Spotify content.
EXAMPLE_CSV = Path(__file__).resolve().parent / "example_tracklist.csv"

# Minimal Exportify-shaped header. Only the three columns the loader cares
# about (Track URI, Track Name, Artist Name(s)) need realistic values.
_HEADER = (
    "Track URI;Track Name;Artist URI(s);Artist Name(s);Album URI;"
    "Album Name;Album Artist URI(s);Album Artist Name(s);"
    "Album Release Date;Album Image URL;Disc Number;Track Number;"
    "Track Duration (ms);Track Preview URL;Explicit;Popularity;ISRC;"
    "Added By;Added At"
)


def _row(track_uri: str, name: str, artists: str) -> str:
    """Build one semicolon-delimited Exportify data row.

    Args:
        track_uri: Value for the ``Track URI`` column.
        name: Value for the ``Track Name`` column.
        artists: Value for the ``Artist Name(s)`` column.

    Returns:
        A single CSV line matching :data:`_HEADER`'s column order.
    """
    return (
        f"{track_uri};{name};spotify:artist:x;{artists};spotify:album:y;"
        "Album;spotify:artist:x;AlbumArtist;2024-01-01;http://img;1;1;"
        "200000;http://prev;FALSE;50;ISRC123;;2026-01-01T00:00:00Z"
    )


def _write_csv(path: Path, rows: list[str], header: str = _HEADER) -> Path:
    """Write a CSV file with the given header and rows.

    Args:
        path: Destination file path.
        rows: Data lines (without trailing newline).
        header: Header line; defaults to a full Exportify header.

    Returns:
        The path that was written, for convenient chaining.
    """
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path
