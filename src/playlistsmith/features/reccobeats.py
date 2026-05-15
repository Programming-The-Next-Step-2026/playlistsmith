"""ReccoBeats feature client (the ``precomputed`` source).

Internal to :mod:`playlistsmith.features`; do not import this module from
outside the package. Resolution is two hops, both batched and routed
through the shared HTTP client in :mod:`playlistsmith._http`:

1. ``GET /v1/track?ids=<spotify_ids>`` maps each Spotify track ID to a
   ReccoBeats internal ID (recovered from the returned ``href``).
2. ``GET /v1/audio-features?ids=<reccobeats_ids>`` returns the
   precomputed audio features for those ReccoBeats IDs.

Anything that does not resolve at either hop is treated as a miss: the
track is dropped, recorded in the :class:`~playlistsmith.features.CoverageReport`,
and reported to the user. A whole batch that keeps failing (after the
shared client's retry budget) is likewise downgraded to misses rather
than aborting the run.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pandas as pd

from playlistsmith import _http
from playlistsmith.features import CoverageReport
from playlistsmith.io.csv_loader import ARTIST, SPOTIFY_ID, TITLE

#: API host and endpoints (see https://reccobeats.com/docs).
RECCOBEATS_BASE_URL = "https://api.reccobeats.com"
_TRACK_ENDPOINT = f"{RECCOBEATS_BASE_URL}/v1/track"
_AUDIO_FEATURES_ENDPOINT = f"{RECCOBEATS_BASE_URL}/v1/audio-features"

#: Maximum IDs per request. Conservative; tune here if the API allows more.
RECCOBEATS_BATCH_SIZE = 40

#: The audio-feature fields ReccoBeats returns, in output column order.
FEATURE_COLUMNS = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
]


def _chunked(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    """Yield successive ``size``-length slices of ``items``.

    Args:
        items: The sequence to split.
        size: Maximum length of each yielded slice (must be positive).

    Yields:
        Consecutive, non-overlapping slices covering ``items`` in order.
    """
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _spotify_id_from_href(href: str | None) -> str | None:
    """Recover a bare Spotify track ID from a ReccoBeats ``href``.

    Args:
        href: A URL such as ``https://open.spotify.com/track/<id>``
            (optionally with a query string), or ``None``.

    Returns:
        The trailing path segment (the Spotify track ID), or ``None`` if
        ``href`` is empty or has no usable final segment.
    """
    if not href:
        return None
    path = href.split("?", 1)[0].rstrip("/")
    return path.rsplit("/", 1)[-1] or None


def _content(response_json: Any) -> list[dict[str, Any]]:
    """Return the ``content`` list from a ReccoBeats JSON body.

    Args:
        response_json: The parsed JSON returned by the API.

    Returns:
        The list under the ``content`` key, or an empty list if the body
        is not shaped as expected.
    """
    if isinstance(response_json, dict):
        content = response_json.get("content")
        if isinstance(content, list):
            return content
    return []


def _resolve_spotify_ids(spotify_ids: Sequence[str]) -> dict[str, str]:
    """Map Spotify track IDs to ReccoBeats internal IDs (batched).

    Each batch is requested independently; a batch that fails after the
    shared client's retries is skipped, so its IDs simply remain
    unresolved (and become misses downstream).

    Args:
        spotify_ids: The Spotify track IDs to resolve.

    Returns:
        A mapping ``spotify_id -> reccobeats_id`` containing only the IDs
        that resolved.
    """
    mapping: dict[str, str] = {}
    for batch in _chunked(spotify_ids, RECCOBEATS_BATCH_SIZE):
        try:
            response = _http.get(
                _TRACK_ENDPOINT, params={"ids": ",".join(batch)}
            )
        except _http.HTTPClientError:
            continue  # Whole batch unresolved -> misses.
        for item in _content(response.json()):
            spotify_id = _spotify_id_from_href(item.get("href"))
            reccobeats_id = item.get("id")
            if spotify_id and reccobeats_id:
                mapping[spotify_id] = str(reccobeats_id)
    return mapping


def _fetch_audio_features(
    reccobeats_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Fetch precomputed audio features by ReccoBeats ID (batched).

    Args:
        reccobeats_ids: Unique ReccoBeats internal IDs to look up.

    Returns:
        A mapping ``reccobeats_id -> {feature: value}`` for the IDs that
        returned features; missing IDs are simply absent.
    """
    features: dict[str, dict[str, Any]] = {}
    for batch in _chunked(reccobeats_ids, RECCOBEATS_BATCH_SIZE):
        try:
            response = _http.get(
                _AUDIO_FEATURES_ENDPOINT, params={"ids": ",".join(batch)}
            )
        except _http.HTTPClientError:
            continue  # Whole batch unresolved -> misses.
        for item in _content(response.json()):
            reccobeats_id = item.get("id")
            if reccobeats_id is None:
                continue
            features[str(reccobeats_id)] = {
                col: item.get(col) for col in FEATURE_COLUMNS
            }
    return features


def extract_precomputed(
    tracks: pd.DataFrame,
) -> tuple[pd.DataFrame, CoverageReport]:
    """Resolve precomputed ReccoBeats features for ``tracks``.

    Tracks are kept in their original order. Any track that cannot be
    resolved to ReccoBeats audio features (no match at ``/v1/track``, no
    features at ``/v1/audio-features``, or a persistently failing batch)
    is dropped, recorded in the coverage report, and named on stdout.

    Args:
        tracks: A TrackLibrary-shaped frame with ``spotify_id``,
            ``title`` and ``artist`` columns.

    Returns:
        A ``(features_df, coverage)`` tuple. ``features_df`` has columns
        ``[spotify_id, title, artist, *FEATURE_COLUMNS]`` with one row
        per resolved track (possibly empty). ``coverage`` reports the
        resolved/dropped counts and the dropped tracks.
    """
    records = tracks[[SPOTIFY_ID, TITLE, ARTIST]].to_dict("records")
    spotify_ids = [str(rec[SPOTIFY_ID]) for rec in records]

    spotify_to_reccobeats = _resolve_spotify_ids(spotify_ids)
    unique_reccobeats_ids = list(dict.fromkeys(spotify_to_reccobeats.values()))
    features_by_reccobeats_id = _fetch_audio_features(unique_reccobeats_ids)

    resolved_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    for rec in records:
        spotify_id = str(rec[SPOTIFY_ID])
        reccobeats_id = spotify_to_reccobeats.get(spotify_id)
        feats = (
            features_by_reccobeats_id.get(reccobeats_id)
            if reccobeats_id is not None
            else None
        )
        identity = {
            SPOTIFY_ID: spotify_id,
            TITLE: rec[TITLE],
            ARTIST: rec[ARTIST],
        }
        if feats is not None:
            resolved_rows.append({**identity, **feats})
        else:
            dropped_rows.append(identity)

    features_df = pd.DataFrame(
        resolved_rows,
        columns=[SPOTIFY_ID, TITLE, ARTIST, *FEATURE_COLUMNS],
    )
    dropped_df = pd.DataFrame(
        dropped_rows, columns=[SPOTIFY_ID, TITLE, ARTIST]
    )

    if dropped_rows:
        print(
            f"[playlistsmith] Dropped {len(dropped_rows)} track(s) with no "
            "ReccoBeats precomputed features:"
        )
        for row in dropped_rows:
            print(f"  - {row[TITLE]} — {row[ARTIST]}")

    coverage = CoverageReport(
        total=len(records),
        resolved=len(features_df),
        dropped_tracks=dropped_df,
    )
    return features_df, coverage
