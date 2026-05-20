# Playlistsmith

Content-based clustering and recommendation for music libraries. Input is a CSV of tracks (artist, title, ISRC). The pipeline computes audio features from external sources, clusters tracks and outputs CSVs with playlists for the clusters.

## Critical constraint: Spotify data is NOT in the ML pipeline

The package operates on `(artist, title)` strings from a CSV. Spotify is *only* an optional input source, via for example:

* The user running web Exportify (https://exportify.app) and providing the resulting CSV; a synthetic example .csv (Exportify-shaped, no real Spotify content) can be found at './tests/example_tracklist.csv'

No Spotify-derived features ever enter clustering or recommendation.ReccoBeats. Do not suggest calling Spotify's `/audio-features` or `/recommendations` endpoints — they were deprecated in November 2024 for new applications, and Spotify's developer terms forbid ingesting their content into ML pipelines.

## Feature extraction mode

- **`precomputed`** — lookup-only. ReccoBeats Spotify-ID lookup. Tracks with no precomputed features are dropped from the output and reported in the coverage summary.

The mode is followed by a downstream pipeline (clustering, recommendation, viz).

## Tech stack

- Python 3.12 (pinned in `pyproject.toml`: `requires-python = ">=3.12,<3.13"`)
- Core: numpy, pandas, scikit-learn, scipy
- Tests: pytest

## External APIs (all free, all with quirks)

- **ReccoBeats `/v1/track`** — lookup by Spotify ID (when present in the CSV). Used in `precomputed`.

The pipeline must handle misses gracefully and always emit a coverage summary listing how many tracks were resolved by each source. Songs that were not found in an API should be removed with a message that tells the user about this.

## Pipeline (data flow)

CSV → `io.csv_loader` → list of `Track` records → `features.extract(mode=...)` → unified feature DataFrame + coverage report → `cluster.algorithms` → `viz.*` and `io.playlist_export`.

The `features` subpackage exposes a single `extract(tracks, mode)` entry point. Internal modules (`acousticbrainz`, `reccobeats`, `previews`, `librosa_features`, `align`) are implementation details; do not call them directly from outside `features`.

## Conventions
- Document every function/method with docstrings.
- When implementing classes/functions write the test first, ask for user feedback, then write the function and test thoroughly.
- Type hints everywhere; `mypy src/` must pass.
- Follow Google coding guidelines (https://google.github.io/styleguide/pyguide.html) 
- Functions return dataclasses or DataFrames, not raw dicts — except at API boundaries where the dict comes straight from JSON.
- All outbound HTTP requests use the shared client instance in `playlistsmith/_http.py`, which is preconfigured with timeouts, retries, and a proper User-Agent. Feature modules import this client; they never create their own `httpx.Client()` or call `httpx.get()` directly. This keeps timeout/retry/rate-limit behavior consistent across all five external APIs.
- Statistical defaults: prefer `GaussianMixture` over `KMeans` for soft cluster assignment; use BIC for model selection; report silhouette and intra-cluster cohesion alongside any clustering result.
- Tests for API clients must mock HTTP with `pytest-httpx`; never hit live APIs in CI.


## Out of scope (do not suggest)

- Training any model on Spotify data. The recommender uses pretrained embeddings for inference only, plus nearest-neighbor lookup.
- Persistent storage of Spotify metadata beyond the cache TTL.
- YouTube as an audio source (legal gray area).
- Calling Spotify's deprecated `/audio-features` or `/recommendations` endpoints, even as "fallback."
- Bypassing the `features.extract(mode=...)` entry point. New feature sources go inside `features/`, not as parallel pipelines.