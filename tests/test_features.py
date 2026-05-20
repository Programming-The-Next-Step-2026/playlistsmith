"""Tests for the public ``playlistsmith.features.extract`` entry point.

``extract`` is the single supported way to obtain features; it dispatches
on ``mode`` and otherwise stays out of the way. The ReccoBeats internals
are exercised separately in ``test_reccobeats.py`` and stubbed here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from playlistsmith import features
from playlistsmith.features import CoverageReport
from playlistsmith.io.csv_loader import ARTIST, SPOTIFY_ID, TITLE


def _tracks() -> pd.DataFrame:
    """A minimal TrackLibrary-shaped frame."""
    return pd.DataFrame([{SPOTIFY_ID: "sp1", TITLE: "S", ARTIST: "A"}])


def test_precomputed_delegates_to_reccobeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mode='precomputed' calls the ReccoBeats client and returns its result."""
    sentinel_df = pd.DataFrame()
    sentinel_cov = CoverageReport(
        total=1, resolved=0, dropped_tracks=_tracks().iloc[0:0]
    )
    seen: dict[str, object] = {}

    def fake_extract(tracks: pd.DataFrame):
        seen["tracks"] = tracks
        return sentinel_df, sentinel_cov

    monkeypatch.setattr(
        "playlistsmith.features.reccobeats.extract_precomputed",
        fake_extract,
    )

    tracks = _tracks()
    df, cov = features.extract(tracks, mode="precomputed")

    assert df is sentinel_df
    assert cov is sentinel_cov
    assert seen["tracks"] is tracks


def test_unknown_mode_raises_value_error() -> None:
    """An unsupported mode is rejected with a helpful ValueError."""
    with pytest.raises(ValueError, match="mode"):
        features.extract(_tracks(), mode="not-a-mode")


def test_coverage_report_renders_human_summary() -> None:
    """CoverageReport stringifies to a readable coverage summary."""
    cov = CoverageReport(
        total=3, resolved=2, dropped_tracks=_tracks()
    )
    text = str(cov)

    assert "3" in text and "2" in text
    assert "reccobeats" in text.lower() or "resolved" in text.lower()
