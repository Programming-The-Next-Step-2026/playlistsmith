# Playlistsmith

Content-based clustering for music libraries (recommendation planned). You hand
Playlistsmith a CSV track list; it computes audio features for those tracks,
clusters them by sound, and writes one playlist CSV per cluster.

> **New here? Start with the tutorial.** The annotated vignette in
> [docs/vignette.ipynb](docs/vignette.ipynb) walks through the whole pipeline
> end to end — from an Exportify CSV to a set of exported playlists — with
> screenshots, a worked rock-vs-classical example, and both the GUI and Python
> API. Run it from the `docs/` directory so the relative paths resolve.

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
- **`cluster`** — preprocessing, model fitting (K-means / GMM / HDBSCAN with
  BIC- and silhouette-based selection), and cluster interpretation.
- **`io.playlist_export`** — writes one CSV per cluster, preserving the
  original Exportify columns so the output is round-trippable.

## Current module structure

```
src/playlistsmith/
├── __init__.py            # re-exports the public surface (TrackLibrary)
├── _http.py               # shared HTTP client (timeouts, retries, backoff)
├── io/
│   ├── __init__.py        # exposes TrackLibrary
│   ├── csv_loader.py      # TrackLibrary: Exportify CSV → tidy DataFrame
│   └── playlist_export.py # write one CSV per cluster
├── features/
│   ├── __init__.py        # extract() entry point + CoverageReport
│   └── reccobeats.py      # internal: ReccoBeats "precomputed" client
├── cluster/
│   ├── __init__.py        # public cluster API
│   ├── preprocess.py      # feature scaling / preparation
│   ├── algorithms.py      # KMeans / GMM / HDBSCAN fitting + selection
│   ├── interpret.py       # per-cluster summaries
│   └── public.py          # high-level cluster(...) entry point
└── gui/
    ├── __init__.py        # exposes the playlistsmith-gui entry point
    ├── app.py             # Streamlit single-page app
    ├── cli.py             # console-script launcher (--demo flag)
    ├── state.py           # session-state keys + reset helpers
    ├── fixtures.py        # offline ReccoBeats mock transport
    └── widgets/           # upload / extract / cluster / viz / export panels
```

## Usage

### Python API

The example below runs the full pipeline offline against the bundled
synthetic Exportify CSV at [tests/example_synthetic.csv](tests/example_synthetic.csv).
Installing the GUI's mock transport reroutes every ReccoBeats call to a
deterministic in-process handler, so no network access is required:

```python
import playlistsmith as ps
from playlistsmith.gui import fixtures
from playlistsmith.io import playlist_export

# 0. Install the offline ReccoBeats mock. Every call routed through
#    playlistsmith._http now returns deterministic synthetic features
#    for `syn<letter><digits>` IDs (no network).
fixtures.install_mock_transport()

# 1. Load the bundled synthetic Exportify-format CSV.
library = ps.TrackLibrary("./tests/example_synthetic.csv")
print(library)                       # TrackLibrary(source_path='...', tracks=...)
library.display()                    # pretty-print the parsed tracks

# 2. Resolve precomputed audio features (served by the mock).
features, coverage = library.extract_features(mode="precomputed")
print(coverage)                      # "Feature coverage: N/M track(s) resolved ..."
print(coverage.dropped_tracks)       # tracks with no precomputed features

# 3. Cluster. Defaults: GMM with BIC-based k selection, canonical
#    cluster ordering, and small clusters collapsed into an
#    Unclassified bucket (cluster id -1).
result = ps.cluster(features, method="gmm", random_state=0)
print(result.tracks.head())          # per-track: spotify_id, title, artist, cluster, cluster_summary
print(result.descriptions)           # per-cluster: size, top_features, cluster_summary
for w in result.warnings:            # e.g. dominant-cluster notices
    print(w)

# 4. Export one Exportify-shaped CSV per cluster (round-trippable
#    back into Spotify via any Exportify-compatible import flow).
paths = playlist_export.write_cluster_csvs(
    result,
    output_dir="./playlists",
    features_df=features,            # optional: merge audio features into each CSV
)
print(paths)                         # [PosixPath('playlists/cluster_0.csv'), ...]
```

To run against real data instead, drop the `fixtures.install_mock_transport()`
call and point `TrackLibrary` at your own Exportify CSV — every other
step is identical.

### GUI

A Streamlit front-end walks through the same pipeline interactively
(Upload → Extract → Cluster → Visualize → Export). It is a thin shell —
every action maps to one public call into the package.

Install the GUI extras and launch the console script:

```bash
pip install -e ".[gui]"
playlistsmith-gui              # live mode (real ReccoBeats lookups)
playlistsmith-gui --demo       # offline mode (recorded ReccoBeats fixture, no network)
```

Use `--demo` to try the app without an internet connection or while
iterating on the UI; it installs a mock HTTP transport so the entire
pipeline runs against a deterministic synthetic dataset. Any extra
arguments after the flag are forwarded to Streamlit (e.g.
`playlistsmith-gui -- --server.port 8502`).

## Installation

Playlistsmith targets **Python 3.12**.

### From GitHub (remote)

To use the package without cloning it, install the latest `main` straight from
GitHub with `pip`:

```bash
pip install "git+https://github.com/Programming-The-Next-Step-2026/playlistsmith.git"
```

To pull in the Streamlit GUI at the same time (see [Usage → GUI](#gui)), request
the `[gui]` extra:

```bash
pip install "playlistsmith[gui] @ git+https://github.com/Programming-The-Next-Step-2026/playlistsmith.git"
```

Pin a specific commit or tag by appending `@<ref>` to the URL, e.g.
`...playlistsmith.git@main`.

### Local / development

From a clone of the repository root, install the package in editable mode
together with the development extras (test and type tooling):

```bash
pip install -e ".[dev]"
```

The `[dev]` extra pulls in `pytest`, `pytest-httpx`, `pytest-cov`, `mypy`, and
`pandas-stubs` on top of the runtime dependencies (`pandas`, `httpx`).

To also install the Streamlit GUI (see [Usage → GUI](#gui)):

```bash
pip install -e ".[dev,gui]"
```

The `[gui]` extra adds `streamlit`, `plotly`, and `matplotlib`.

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
clustering pipeline.
