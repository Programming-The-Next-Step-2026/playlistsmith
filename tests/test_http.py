"""Tests for the shared HTTP client in :mod:`playlistsmith._http`.

All HTTP is mocked with ``pytest-httpx``; these tests never touch the
network. They pin the behaviour every external API client relies on:
a single configured client, a sane User-Agent, and consistent
retry/backoff on transient failures.
"""

from __future__ import annotations

import time

import httpx
import pytest
from pytest_httpx import HTTPXMock

from playlistsmith import _http


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make retry backoff instantaneous unless a test opts in."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


def test_get_client_is_singleton() -> None:
    """The shared client is created once and reused."""
    assert _http.get_client() is _http.get_client()


def test_client_advertises_user_agent() -> None:
    """The client sets a playlistsmith User-Agent by default."""
    user_agent = _http.get_client().headers["User-Agent"]
    assert user_agent.startswith("playlistsmith/")


def test_client_timeout_is_configured() -> None:
    """The client uses the package default timeout, not httpx's."""
    timeout = _http.get_client().timeout
    assert timeout.connect == _http.DEFAULT_TIMEOUT.connect
    assert timeout.read == _http.DEFAULT_TIMEOUT.read


def test_get_success_returns_response(httpx_mock: HTTPXMock) -> None:
    """A 2xx response is returned to the caller unchanged."""
    httpx_mock.add_response(json={"ok": True})

    resp = _http.get("https://api.example.com/ping")

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_request_sends_user_agent(httpx_mock: HTTPXMock) -> None:
    """Outbound requests carry the shared User-Agent header."""
    httpx_mock.add_response(json={})

    _http.get("https://api.example.com/ping")

    sent = httpx_mock.get_requests()[0]
    assert sent.headers["User-Agent"].startswith("playlistsmith/")


def test_retries_on_429_then_succeeds(httpx_mock: HTTPXMock) -> None:
    """A 429 is retried; the eventual 200 is returned."""
    httpx_mock.add_response(
        status_code=429, headers={"Retry-After": "0"}
    )
    httpx_mock.add_response(status_code=200, json={"ok": True})

    resp = _http.get("https://api.example.com/x")

    assert resp.status_code == 200
    assert len(httpx_mock.get_requests()) == 2


def test_retries_on_500_then_succeeds(httpx_mock: HTTPXMock) -> None:
    """A 5xx is retried; the eventual 200 is returned."""
    httpx_mock.add_response(status_code=500)
    httpx_mock.add_response(status_code=200, json={"ok": True})

    resp = _http.get("https://api.example.com/x")

    assert resp.status_code == 200
    assert len(httpx_mock.get_requests()) == 2


def test_raises_after_exhausting_retries(httpx_mock: HTTPXMock) -> None:
    """Persistent 5xx raises HTTPClientError after max_retries attempts."""
    httpx_mock.add_response(status_code=503, is_reusable=True)

    with pytest.raises(_http.HTTPClientError):
        _http.get("https://api.example.com/x", max_retries=2)

    # initial attempt + 2 retries
    assert len(httpx_mock.get_requests()) == 3


def test_honors_retry_after_header(
    monkeypatch: pytest.MonkeyPatch, httpx_mock: HTTPXMock
) -> None:
    """The Retry-After delay is respected before retrying."""
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    httpx_mock.add_response(
        status_code=429, headers={"Retry-After": "2"}
    )
    httpx_mock.add_response(status_code=200, json={})

    _http.get("https://api.example.com/x")

    assert slept
    assert slept[0] == pytest.approx(2.0)


def test_non_retryable_4xx_raises_immediately(httpx_mock: HTTPXMock) -> None:
    """A 404 is not retried and raises HTTPClientError at once."""
    httpx_mock.add_response(status_code=404)

    with pytest.raises(_http.HTTPClientError):
        _http.get("https://api.example.com/missing")

    assert len(httpx_mock.get_requests()) == 1
