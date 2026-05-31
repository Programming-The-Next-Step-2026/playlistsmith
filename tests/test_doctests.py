"""Execute the deterministic, network-free docstring examples.

Many functions carry runnable ``>>>`` examples. The ones that synthesise
their own data — the clustering-pipeline examples in ``cluster.*`` and
``io.playlist_export`` — are fully deterministic and need no network, so
we run them here to keep the documented output honest. Without this, the
examples (several of which assert exact numbers, e.g. a silhouette of
``0.567`` or a full BIC curve) can silently drift out of date as the
libraries underneath them change.

The ``features`` and ``TrackLibrary.extract_features`` examples are
deliberately *excluded*: they hit the live ReccoBeats API. CI never
touches live APIs, and their output depends on a third party, so they
are not reproducible offline.

Some examples reference a pre-built ``features`` frame or modelling
matrix ``X`` whose construction lives in a sibling docstring (the
:func:`~playlistsmith.cluster.fit_gmm` example). doctest gives each
docstring its own namespace, so we inject those two names via
``extraglobs``, built exactly as that example builds them.
"""

from __future__ import annotations

import contextlib
import doctest
import importlib
import io

import numpy as np
import pandas as pd
import pytest

from playlistsmith.cluster import prepare_matrix

# Modules whose docstring examples are deterministic and network-free.
_DOCTEST_MODULES = [
    "playlistsmith.cluster.algorithms",
    "playlistsmith.cluster.interpret",
    "playlistsmith.cluster.preprocess",
    "playlistsmith.cluster.public",
    "playlistsmith.io.playlist_export",
]

#: Canonical ReccoBeats feature columns, in the order the examples use.
_FEATURE_COLUMNS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence",
]


def _synthetic_features() -> pd.DataFrame:
    """Build the two-group synthetic feature frame the examples assume.

    This mirrors the :func:`~playlistsmith.cluster.fit_gmm` docstring
    example byte-for-byte (same seed, same means/spreads), so any example
    that consumes a ready-made ``features`` frame sees the data its
    documented output was written against.

    Returns:
        A feature DataFrame with ``spotify_id``, ``title`` and ``artist``
        identity columns followed by the nine ReccoBeats feature columns:
        30 tracks split into a mellow and an upbeat group.
    """
    sd = [0.02, 0.02, 0.02, 0.02, 0.02, 1.0, 0.02, 3.0, 0.02]
    mellow = [0.88, 0.30, 0.20, 0.40, 0.15, -17.0, 0.05, 85.0, 0.25]
    upbeat = [0.12, 0.80, 0.90, 0.40, 0.15, -6.0, 0.05, 128.0, 0.75]
    rng = np.random.default_rng(5)
    values = np.vstack([rng.normal(mellow, sd, (15, 9)),
                        rng.normal(upbeat, sd, (15, 9))])
    features = pd.DataFrame(values, columns=_FEATURE_COLUMNS)
    features.insert(0, "artist", "Various")
    features.insert(0, "title", [f"Track {i}" for i in range(30)])
    features.insert(0, "spotify_id", [f"id{i:02d}" for i in range(30)])
    return features


@pytest.mark.parametrize("module_name", _DOCTEST_MODULES)
def test_module_doctests(module_name: str) -> None:
    """Run a module's docstring examples and assert none fail.

    Args:
        module_name: Dotted import path of the module to doctest.
    """
    module = importlib.import_module(module_name)
    features = _synthetic_features()
    extraglobs = {"features": features, "X": prepare_matrix(features)[0]}

    # Capture doctest's own diff output so a failure points straight at
    # the offending example and its mismatched line.
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        result = doctest.testmod(
            module, extraglobs=extraglobs, verbose=False, report=False
        )

    assert result.failed == 0, (
        f"{result.failed} of {result.attempted} doctest(s) failed in "
        f"{module_name}:\n{report.getvalue()}"
    )
