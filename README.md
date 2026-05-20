# Playlistsmith

Content-based clustering and recommendation for music libraries. You hand
Playlistsmith a CSV track list; it computes audio features for those tracks,
clusters them by sound, and writes one playlist CSV per cluster.

## Motivation

This package grew out of my own habit of dumping every song I ever liked into a
single, ever-growing playlist. Splitting that by hand is hopeless. Playlistsmith
is meant to make "turn this pile of tracks into a set of coherent playlists"
automatic — for me, and for anyone else with the same problem.

## How it works

### Input: a preformatted CSV

Playlistsmith does **not** talk to Spotify, scrape it, or pull anything from
your account. It works purely from a CSV you provide. The expected layout is
the one produced by [Exportify](https://exportify.app) — a free, browser-based
tool that exports a Spotify playlist to a `;`-delimited CSV. You run Exportify
yourself and feed the resulting file in.

Only three fields are used: the track title, the artist name(s), and the bare
Spotify track ID (parsed out of the `spotify:track:<id>` URI). Every other
column in the export is ignored. Rows without a Spotify track ID are dropped on
load. A synthetic, Exportify-shaped fixture (no real Spotify content) lives at
[tests/example_tracklist.csv](tests/example_tracklist.csv).

### Pipeline

```
CSV → io.csv_loader.TrackLibrary → features.extract(mode=...) → (features_df, CoverageReport) → [clustering] → [playlist export]
```

- **`io.csv_loader.TrackLibrary`** — loads and validates the Exportify CSV into
  a tidy `(title, artist, spotify_id)` DataFrame.
- **`features.extract(tracks, mode)`** — the single supported entry point for
  feature extraction. The only mode today is `"precomputed"`: a lookup-only
  resolution against [ReccoBeats](https://reccobeats.com/) keyed by Spotify
  track ID (`/v1/track` → ReccoBeats ID → `/v1/audio-features`). Tracks with no
  precomputed features are dropped, named on stdout, and counted in the
  returned `CoverageReport`. Internal feature modules (`reccobeats`) are
  implementation details — always go through `extract`.
- **`_http`** — a single shared, retry/backoff/User-Agent–configured HTTP
  client that every external API call routes through.

Clustering and playlist export are the next pipeline stages and are not yet
implemented.

## Current module structure

```
src/playlistsmith/
├── __init__.py            # re-exports the public surface (TrackLibrary)
├── _http.py               # shared HTTP client (timeouts, retries, backoff)
├── io/
│   ├── __init__.py        # exposes TrackLibrary
│   └── csv_loader.py      # TrackLibrary: Exportify CSV → tidy DataFrame
└── features/
    ├── __init__.py        # extract() entry point + CoverageReport
    └── reccobeats.py      # internal: ReccoBeats "precomputed" client
```

## Usage

```python
import playlistsmith as ps

# Load an Exportify-format CSV.
library = ps.TrackLibrary("./tests/example_tracklist.csv")
print(library)                       # TrackLibrary(source_path='...', tracks=3)
library.display()                    # pretty-print the parsed tracks

# Resolve precomputed audio features via ReccoBeats.
features, coverage = library.extract_features(mode="precomputed")
print(coverage)                      # "Feature coverage: N/M track(s) resolved ..."
print(coverage.dropped_tracks)       # tracks with no precomputed features
```

## Installation (local / development)

Playlistsmith targets **Python 3.12**. From the repository root, install the
package in editable mode together with the development extras (test and type
tooling):

```bash
pip install -e ".[dev]"
```

The `[dev]` extra pulls in `pytest`, `pytest-httpx`, `pytest-cov`, `mypy`, and
`pandas-stubs` on top of the runtime dependencies (`pandas`, `httpx`).

## Running the tests

The test suite is configured via `pyproject.toml` (`pythonpath = ["src"]`,
`testpaths = ["tests"]`), so a bare `pytest` from the repo root finds and runs
everything:

```bash
pytest
```

All external HTTP is mocked with `pytest-httpx`; the tests never touch a live API.

To type-check the package:

```bash
mypy src/
```

## Test coverage diagnostics

Coverage is measured with `pytest-cov`. For a terminal report that also lists the specific lines not covered:

```bash
pytest --cov=playlistsmith --cov-report=term-missing
```

## Disclaimer

This project is not affiliated with, sponsored by, or endorsed by Exportify,
ReccoBeats, or Spotify. These names are used only to describe compatible input
formats and data sources. No Spotify-derived audio features ever enter the
clustering or recommendation pipeline.
