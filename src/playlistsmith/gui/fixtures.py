"""Offline ReccoBeats mock transport for the GUI's demo mode.

Reuses the same synthetic-feature generator the vignette mocks
ReccoBeats with: every track's Spotify ID is a ``syn<letter><18 digits>``
string whose fourth character encodes a target cluster (``A``, ``B`` or
``C``); the mock returns deterministic synthetic audio features drawn
from per-cluster centres plus reproducible per-ID jitter. IDs that
don't match (e.g. ``synX...``) are reported as misses so the
``CoverageReport.dropped_tracks`` path is exercised end-to-end.

The whole point of doing it this way (rather than checked-in JSON
fixtures) is that the same fixture also drives the vignette and the
GUI smoke test, so all three stay in sync with zero manual upkeep.
"""

from __future__ import annotations

import hashlib

import httpx
import numpy as np

from playlistsmith import _http
from playlistsmith.features.reccobeats import FEATURE_COLUMNS

__all__ = [
    "CLUSTER_CENTRES",
    "install_mock_transport",
    "synthetic_features",
]


#: Cluster centres in raw ReccoBeats feature space. Loudness is in dB;
#: tempo in BPM; every other feature lives on [0, 1]. Picked so the
#: three clusters are clearly separable but not trivially so.
CLUSTER_CENTRES: dict[str, dict[str, float]] = {
    "A": {  # chill / acoustic
        "acousticness": 0.80, "danceability": 0.40, "energy": 0.20,
        "instrumentalness": 0.10, "liveness": 0.10, "loudness": -18.0,
        "speechiness": 0.05, "tempo": 85.0, "valence": 0.55,
    },
    "B": {  # upbeat / dance
        "acousticness": 0.15, "danceability": 0.80, "energy": 0.75,
        "instrumentalness": 0.05, "liveness": 0.15, "loudness": -6.0,
        "speechiness": 0.08, "tempo": 120.0, "valence": 0.75,
    },
    "C": {  # intense / electronic
        "acousticness": 0.05, "danceability": 0.55, "energy": 0.92,
        "instrumentalness": 0.50, "liveness": 0.20, "loudness": -4.0,
        "speechiness": 0.06, "tempo": 150.0, "valence": 0.30,
    },
}

_JITTER: dict[str, float] = {
    "acousticness": 0.05, "danceability": 0.05, "energy": 0.05,
    "instrumentalness": 0.05, "liveness": 0.04, "loudness": 1.5,
    "speechiness": 0.02, "tempo": 4.0, "valence": 0.05,
}

_UNIT_INTERVAL_FEATURES = frozenset(
    {
        "acousticness", "danceability", "energy", "instrumentalness",
        "liveness", "speechiness", "valence",
    }
)


def synthetic_features(spotify_id: str) -> dict[str, float] | None:
    """Return deterministic synthetic features for one Spotify ID.

    Args:
        spotify_id: The ``syn<letter><digits>`` ID encoding a cluster.

    Returns:
        A ``{feature: value}`` dict for known cluster letters, or
        ``None`` for IDs the mock pretends not to know.
    """
    if len(spotify_id) < 4 or not spotify_id.startswith("syn"):
        return None
    cluster_letter = spotify_id[3]
    if cluster_letter not in CLUSTER_CENTRES:
        return None
    centre = CLUSTER_CENTRES[cluster_letter]
    seed = int(hashlib.sha1(spotify_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        val = float(centre[col] + rng.normal(0.0, _JITTER[col]))
        if col in _UNIT_INTERVAL_FEATURES:
            val = float(np.clip(val, 0.0, 1.0))
        out[col] = val
    return out


def _handle(request: httpx.Request) -> httpx.Response:
    """``httpx.MockTransport`` handler for ReccoBeats endpoints."""
    ids = [s for s in request.url.params.get("ids", "").split(",") if s]
    if request.url.path == "/v1/track":
        content = [
            {"id": f"rb-{sid}", "href": f"https://open.spotify.com/track/{sid}"}
            for sid in ids
            if synthetic_features(sid) is not None
        ]
        return httpx.Response(200, json={"content": content})
    if request.url.path == "/v1/audio-features":
        content = []
        for rb_id in ids:
            if not rb_id.startswith("rb-"):
                continue
            sid = rb_id[3:]
            feats = synthetic_features(sid)
            if feats is None:
                continue
            content.append({"id": rb_id, **feats})
        return httpx.Response(200, json={"content": content})
    return httpx.Response(404, json={"content": []})


def install_mock_transport() -> httpx.Client:
    """Install a mock ``httpx.Client`` on the shared HTTP singleton.

    After this returns, every call routed through
    :mod:`playlistsmith._http` is served by the in-process synthetic
    ReccoBeats handler — no network access required. Idempotent: calling
    it twice simply re-installs a fresh mock client.

    Returns:
        The installed mock client (also stored on
        ``playlistsmith._http._client``).
    """
    client = httpx.Client(
        transport=httpx.MockTransport(_handle),
        headers={"User-Agent": "playlistsmith-gui/demo"},
    )
    _http.set_client(client)
    return client
