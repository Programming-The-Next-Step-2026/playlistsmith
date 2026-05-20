"""Shared HTTP client for every external API playlistsmith talks to.

All five external services (ReccoBeats and friends) are reached through
the single :class:`httpx.Client` returned by :func:`get_client`, and every
request goes through :func:`request`/:func:`get`. Centralising this keeps
timeout, retry, backoff and User-Agent behaviour identical across APIs:
feature modules must never construct their own ``httpx.Client`` or call
``httpx.get`` directly.

The retry policy is deliberately conservative because the free APIs we use
do not publish their rate limits: transient failures (HTTP 429 and 5xx,
plus connection errors) are retried with backoff, honouring a server-sent
``Retry-After`` header when present. Any non-2xx response that survives the
retry budget raises :class:`HTTPClientError`; callers (e.g. the ReccoBeats
client) decide whether to treat that as a batch of misses.
"""

from __future__ import annotations

import time
from importlib import metadata

import httpx

#: Per-request timeout used by every API client. A short connect timeout
#: fails fast on network issues; a longer read timeout tolerates slow but
#: alive endpoints.
DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)

#: Number of *retries* (beyond the initial attempt) for transient failures.
DEFAULT_MAX_RETRIES = 3

#: Base delay (seconds) for exponential backoff when the server does not
#: send a ``Retry-After`` header.
_BACKOFF_BASE_SECONDS = 0.5

#: Upper bound (seconds) on any single backoff sleep.
_BACKOFF_MAX_SECONDS = 30.0


def _package_version() -> str:
    """Return the installed package version, or a fallback if unknown.

    Returns:
        The distribution version of ``playlistsmith`` if it is installed,
        otherwise ``"0.0.0"`` (the package is often run straight from the
        source tree during development).
    """
    try:
        return metadata.version("playlistsmith")
    except metadata.PackageNotFoundError:
        return "0.0.0"


#: User-Agent advertised on every outbound request, so the APIs we depend
#: on can identify (and, if needed, contact) this client.
USER_AGENT = (
    f"playlistsmith/{_package_version()} "
    "(+https://github.com/Programming-The-Next-Step-2026/playlistsmith)"
)


class HTTPClientError(RuntimeError):
    """Raised when a request ultimately fails after the retry budget.

    This wraps both exhausted-retry outcomes (persistent 429/5xx or
    connection errors) and non-retryable responses (other 4xx). Callers
    that can degrade gracefully should catch this and convert the affected
    work into reported misses rather than aborting the whole pipeline.

    Attributes:
        request: The final :class:`httpx.Request` that was attempted.
        response: The final :class:`httpx.Response`, or ``None`` if the
            failure was a transport/connection error with no response.
        attempts: Total number of attempts made (initial plus retries).
    """

    def __init__(
        self,
        message: str,
        *,
        request: httpx.Request,
        response: httpx.Response | None,
        attempts: int,
    ) -> None:
        """Initialise the error with the failing request/response context.

        Args:
            message: Human-readable description of the failure.
            request: The final request that was attempted.
            response: The final response, or ``None`` for transport errors.
            attempts: Total number of attempts made.
        """
        super().__init__(message)
        self.request = request
        self.response = response
        self.attempts = attempts


_client: httpx.Client | None = None


def get_client() -> httpx.Client:
    """Return the process-wide shared HTTP client, creating it on first use.

    The client is configured once with :data:`DEFAULT_TIMEOUT`, the shared
    :data:`USER_AGENT`, redirect following, and a transport that retries
    low-level connection errors. It is intentionally a singleton so that
    connection pooling and behaviour are shared across all API clients.

    Returns:
        The shared :class:`httpx.Client` instance.
    """
    global _client
    if _client is None:
        _client = httpx.Client(
            timeout=DEFAULT_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            transport=httpx.HTTPTransport(retries=2),
        )
    return _client


def _is_retryable_status(status_code: int) -> bool:
    """Return whether a status code represents a transient failure.

    Args:
        status_code: The HTTP status code of the response.

    Returns:
        ``True`` for 429 (rate limited) and any 5xx (server error);
        ``False`` otherwise.
    """
    return status_code == httpx.codes.TOO_MANY_REQUESTS or (
        500 <= status_code < 600
    )


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    """Compute how long to wait before the next retry.

    A server-sent ``Retry-After`` header (integer seconds) takes priority;
    otherwise an exponential backoff is used, capped at
    :data:`_BACKOFF_MAX_SECONDS`.

    Args:
        response: The failed response, or ``None`` for a connection error.
        attempt: Zero-based index of the attempt that just failed.

    Returns:
        The number of seconds to sleep before retrying.
    """
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(0.0, float(int(retry_after)))
            except ValueError:
                pass  # Not an integer (e.g. HTTP-date); fall back to backoff.
    return min(_BACKOFF_BASE_SECONDS * (2**attempt), _BACKOFF_MAX_SECONDS)


def request(
    method: str,
    url: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    **kwargs: object,
) -> httpx.Response:
    """Send a request through the shared client with retry and backoff.

    Transient failures (HTTP 429, 5xx, and connection errors) are retried
    up to ``max_retries`` times, sleeping between attempts per
    :func:`_retry_delay`. A 2xx response is returned as-is. Any other
    outcome that exhausts the budget, and any non-retryable non-2xx
    response, raises :class:`HTTPClientError`.

    Args:
        method: HTTP method, e.g. ``"GET"``.
        url: Absolute request URL.
        max_retries: Maximum number of retries beyond the initial attempt.
        **kwargs: Forwarded verbatim to :meth:`httpx.Client.request`
            (e.g. ``params``, ``json``, ``headers``).

    Returns:
        The successful :class:`httpx.Response` (status in the 2xx range).

    Raises:
        HTTPClientError: If the request fails non-retryably or keeps
            failing after ``max_retries`` retries.
    """
    client = get_client()
    attempt = 0
    while True:
        response: httpx.Response | None = None
        try:
            response = client.request(method, url, **kwargs)  # type: ignore[arg-type]
        except httpx.RequestError as exc:
            if attempt >= max_retries:
                raise HTTPClientError(
                    f"{method} {url} failed after {attempt + 1} attempt(s): "
                    f"{exc!r}",
                    request=exc.request,
                    response=None,
                    attempts=attempt + 1,
                ) from exc
            time.sleep(_retry_delay(None, attempt))
            attempt += 1
            continue

        if response.is_success:
            return response

        if not _is_retryable_status(response.status_code) or (
            attempt >= max_retries
        ):
            raise HTTPClientError(
                f"{method} {url} returned HTTP {response.status_code} "
                f"after {attempt + 1} attempt(s)",
                request=response.request,
                response=response,
                attempts=attempt + 1,
            )

        time.sleep(_retry_delay(response, attempt))
        attempt += 1


def get(
    url: str, *, max_retries: int = DEFAULT_MAX_RETRIES, **kwargs: object
) -> httpx.Response:
    """Send a GET request through the shared client.

    Thin convenience wrapper around :func:`request` with
    ``method="GET"``; see that function for retry semantics.

    Args:
        url: Absolute request URL.
        max_retries: Maximum number of retries beyond the initial attempt.
        **kwargs: Forwarded to :func:`request`.

    Returns:
        The successful :class:`httpx.Response`.

    Raises:
        HTTPClientError: If the request ultimately fails.
    """
    return request("GET", url, max_retries=max_retries, **kwargs)
