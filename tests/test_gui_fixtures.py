"""Tests for the demo-mode mock transport.

The mock transport is the swap point that makes demo mode and the
end-to-end GUI smoke test possible. The tests check that
:func:`install_mock_transport` reassigns the shared HTTP client to one
backed by an :class:`httpx.MockTransport` and that the routing handler
returns the expected JSON for known synthetic IDs.
"""

from __future__ import annotations

import httpx
import pytest

from playlistsmith import _http
from playlistsmith.gui import fixtures


@pytest.fixture(autouse=True)
def _reset_http_client():
    """Save and restore the shared client around each test."""
    saved = _http._client
    yield
    _http.set_client(saved)


def test_install_mock_transport_returns_mock_client() -> None:
    client = fixtures.install_mock_transport()
    assert isinstance(client._transport, httpx.MockTransport)
    # And it's the new singleton.
    assert _http.get_client() is client


def test_mock_track_endpoint_returns_known_ids() -> None:
    fixtures.install_mock_transport()
    response = _http.get(
        "https://api.reccobeats.com/v1/track",
        params={"ids": "synA000000000000000001,synX000000000000000001"},
    )
    assert response.status_code == 200
    content = response.json()["content"]
    # synA... is known; synX... is in the "pretend not to know" bucket.
    hrefs = [item["href"] for item in content]
    assert any("synA000000000000000001" in h for h in hrefs)
    assert all("synX000000000000000001" not in h for h in hrefs)


def test_mock_audio_features_returns_synthetic_features() -> None:
    fixtures.install_mock_transport()
    response = _http.get(
        "https://api.reccobeats.com/v1/audio-features",
        params={"ids": "rb-synB000000000000000001"},
    )
    assert response.status_code == 200
    content = response.json()["content"]
    assert len(content) == 1
    feats = content[0]
    # Cluster-B centre is the upbeat/dance one; danceability should be
    # closer to 0.8 than to the chill cluster's 0.4.
    assert feats["danceability"] > 0.5


def test_unknown_endpoint_returns_404() -> None:
    fixtures.install_mock_transport()
    with pytest.raises(_http.HTTPClientError) as excinfo:
        _http.get("https://api.reccobeats.com/v1/nope", params={"ids": "x"})
    assert "404" in str(excinfo.value)
