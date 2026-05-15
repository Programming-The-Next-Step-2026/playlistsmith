"""Tests for the internal ReccoBeats feature client.

The ``precomputed`` flow is two hops:
``spotify_id -> GET /v1/track -> reccobeats_id -> GET /v1/audio-features``.
All HTTP is mocked with a small fake server (``pytest-httpx`` callback);
no live API is touched. Per-batch failures must degrade into reported
misses, never abort the run.
"""

from __future__ import annotations

import time

import httpx
import pandas as pd
import pytest
from pytest_httpx import HTTPXMock

from playlistsmith.features import reccobeats
from playlistsmith.features.reccobeats import FEATURE_COLUMNS
from playlistsmith.io.csv_loader import ARTIST, SPOTIFY_ID, TITLE


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep retry backoff (in the shared HTTP client) instantaneous."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


def _tracks(*rows: tuple[str, str, str]) -> pd.DataFrame:
    """Build a TrackLibrary-shaped frame from ``(spotify_id, title, artist)``."""
    return pd.DataFrame(
        [{SPOTIFY_ID: s, TITLE: t, ARTIST: a} for s, t, a in rows]
    )


def _feat(value: float) -> dict[str, float]:
    """A full audio-feature payload with every field set to ``value``."""
    return {col: value for col in FEATURE_COLUMNS}


def _fake_server(
    track_map: dict[str, str],
    feature_map: dict[str, dict[str, float]],
    *,
    failing_ids: set[str] | None = None,
):
    """Return a request handler emulating the two ReccoBeats endpoints.

    Args:
        track_map: spotify_id -> reccobeats_id for resolvable tracks.
        feature_map: reccobeats_id -> feature payload for tracks that have
            precomputed features.
        failing_ids: ids whose presence in a batch makes that whole batch
            return HTTP 503 (to simulate a persistently failing batch).
    """
    failing = failing_ids or set()

    def handler(request: httpx.Request) -> httpx.Response:
        ids = request.url.params["ids"].split(",")
        if any(i in failing for i in ids):
            return httpx.Response(503)
        if request.url.path == "/v1/track":
            content = [
                {
                    "id": track_map[s],
                    "href": f"https://open.spotify.com/track/{s}",
                }
                for s in ids
                if s in track_map
            ]
            return httpx.Response(200, json={"content": content})
        if request.url.path == "/v1/audio-features":
            content = [
                {"id": rb, **feature_map[rb]}
                for rb in ids
                if rb in feature_map
            ]
            return httpx.Response(200, json={"content": content})
        return httpx.Response(404)

    return handler


def test_happy_path_builds_feature_frame(httpx_mock: HTTPXMock) -> None:
    """Both tracks resolve to a tidy frame with all feature columns."""
    httpx_mock.add_callback(
        _fake_server(
            {"sp1": "rb1", "sp2": "rb2"},
            {"rb1": _feat(0.1), "rb2": _feat(0.2)},
        ),
        is_reusable=True,
    )

    df, cov = reccobeats.extract_precomputed(
        _tracks(("sp1", "Song One", "A"), ("sp2", "Song Two", "B"))
    )

    assert list(df.columns) == [SPOTIFY_ID, TITLE, ARTIST, *FEATURE_COLUMNS]
    assert len(df) == 2
    assert df.loc[df[SPOTIFY_ID] == "sp1", "energy"].iloc[0] == 0.1
    assert cov.total == 2
    assert cov.resolved == 2
    assert len(cov.dropped_tracks) == 0


def test_unresolved_spotify_id_is_dropped(
    httpx_mock: HTTPXMock, capsys: pytest.CaptureFixture[str]
) -> None:
    """A track absent from /v1/track is dropped and reported to the user."""
    httpx_mock.add_callback(
        _fake_server({"sp1": "rb1"}, {"rb1": _feat(0.5)}),
        is_reusable=True,
    )

    df, cov = reccobeats.extract_precomputed(
        _tracks(("sp1", "Keep", "A"), ("sp_missing", "Gone", "B"))
    )

    assert list(df[SPOTIFY_ID]) == ["sp1"]
    assert cov.resolved == 1
    assert cov.total == 2
    assert set(cov.dropped_tracks[SPOTIFY_ID]) == {"sp_missing"}
    assert "Gone" in capsys.readouterr().out  # user is told what was removed


def test_missing_audio_features_is_dropped(httpx_mock: HTTPXMock) -> None:
    """A track that resolves but has no features is dropped."""
    httpx_mock.add_callback(
        _fake_server({"sp1": "rb1", "sp2": "rb2"}, {"rb1": _feat(0.3)}),
        is_reusable=True,
    )

    df, cov = reccobeats.extract_precomputed(
        _tracks(("sp1", "Has", "A"), ("sp2", "NoFeatures", "B"))
    )

    assert list(df[SPOTIFY_ID]) == ["sp1"]
    assert set(cov.dropped_tracks[SPOTIFY_ID]) == {"sp2"}


def test_requests_are_batched(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IDs are chunked at RECCOBEATS_BATCH_SIZE for both endpoints."""
    monkeypatch.setattr(reccobeats, "RECCOBEATS_BATCH_SIZE", 2)
    track_map = {f"sp{i}": f"rb{i}" for i in range(5)}
    feature_map = {f"rb{i}": _feat(float(i)) for i in range(5)}
    httpx_mock.add_callback(
        _fake_server(track_map, feature_map), is_reusable=True
    )

    df, cov = reccobeats.extract_precomputed(
        _tracks(*[(f"sp{i}", f"S{i}", "A") for i in range(5)])
    )

    requests = httpx_mock.get_requests()
    track_calls = [r for r in requests if r.url.path == "/v1/track"]
    feat_calls = [r for r in requests if r.url.path == "/v1/audio-features"]
    assert len(track_calls) == 3  # ceil(5 / 2)
    assert len(feat_calls) == 3
    assert cov.resolved == 5
    assert len(df) == 5


def test_failed_batch_becomes_misses_not_an_error(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persistently failing batch drops only its tracks; run continues."""
    monkeypatch.setattr(reccobeats, "RECCOBEATS_BATCH_SIZE", 2)
    track_map = {f"sp{i}": f"rb{i}" for i in range(4)}
    feature_map = {f"rb{i}": _feat(float(i)) for i in range(4)}
    # sp0 is in the first lookup batch -> that whole batch 503s forever.
    httpx_mock.add_callback(
        _fake_server(track_map, feature_map, failing_ids={"sp0"}),
        is_reusable=True,
    )

    df, cov = reccobeats.extract_precomputed(
        _tracks(*[(f"sp{i}", f"S{i}", "A") for i in range(4)])
    )

    assert cov.resolved == 2  # second batch (sp2, sp3)
    assert set(cov.dropped_tracks[SPOTIFY_ID]) == {"sp0", "sp1"}
    assert set(df[SPOTIFY_ID]) == {"sp2", "sp3"}


def test_no_tracks_resolved_returns_empty_frame(
    httpx_mock: HTTPXMock, capsys: pytest.CaptureFixture[str]
) -> None:
    """When nothing resolves, the frame is empty but well-formed."""
    httpx_mock.add_callback(
        _fake_server({}, {}), is_reusable=True
    )

    df, cov = reccobeats.extract_precomputed(
        _tracks(("spX", "Nope", "A"))
    )

    assert list(df.columns) == [SPOTIFY_ID, TITLE, ARTIST, *FEATURE_COLUMNS]
    assert len(df) == 0
    assert cov.resolved == 0
    assert cov.total == 1
    assert "Nope" in capsys.readouterr().out


def test_calls_reccobeats_endpoints_with_shared_client(
    httpx_mock: HTTPXMock,
) -> None:
    """Requests hit the documented hosts/paths via the shared client."""
    httpx_mock.add_callback(
        _fake_server({"sp1": "rb1"}, {"rb1": _feat(0.4)}), is_reusable=True
    )

    reccobeats.extract_precomputed(_tracks(("sp1", "S", "A")))

    requests = httpx_mock.get_requests()
    paths = {r.url.path for r in requests}
    assert paths == {"/v1/track", "/v1/audio-features"}
    assert all(
        r.url.host == "api.reccobeats.com" for r in requests
    )
    assert all(
        r.headers["User-Agent"].startswith("playlistsmith/")
        for r in requests
    )
