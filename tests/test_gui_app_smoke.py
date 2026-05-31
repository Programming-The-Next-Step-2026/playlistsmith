"""End-to-end smoke test for the Streamlit GUI.

Uses Streamlit's official ``AppTest`` harness to walk the page through
the four user-visible stages — upload, extract, cluster, export — with
the synthetic ReccoBeats mock transport from
:mod:`playlistsmith.gui.fixtures` in place of the live API. The
synthetic example CSV (``tests/example_synthetic.csv``) is the same
fixture the GUI demo button loads, so the GUI and this test stay in sync.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from playlistsmith import _http
from playlistsmith.gui import fixtures
from playlistsmith.gui.state import KEYS

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

_APP_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "playlistsmith"
    / "gui"
    / "app.py"
)


@pytest.fixture
def demo_app(monkeypatch, tmp_path):
    """Build an ``AppTest`` with demo mode + mock transport installed."""
    monkeypatch.setenv("PLAYLISTSMITH_GUI_DEMO", "1")
    saved_client = _http._client
    fixtures.install_mock_transport()
    monkeypatch.chdir(tmp_path)
    try:
        app = AppTest.from_file(str(_APP_PATH), default_timeout=30)
        yield app
    finally:
        _http.set_client(saved_client)


def test_app_renders_in_demo_mode(demo_app) -> None:
    demo_app.run()
    assert not demo_app.exception
    # Sidebar carries the demo badge as an st.warning.
    sidebar_warnings = [w.value for w in demo_app.sidebar.warning]
    assert any("Demo mode" in t for t in sidebar_warnings)


def _get(session_state, key):
    """AppTest's session_state intercepts `.get`; bracket lookup instead."""
    return session_state[key] if key in session_state else None


def test_reset_session_clears_loaded_library(demo_app) -> None:
    demo_app.run()

    example_btn = next(
        b for b in demo_app.button if b.label == "Load example tracklist"
    )
    example_btn.click().run()
    assert _get(demo_app.session_state, KEYS.library) is not None

    reset_btn = next(
        b for b in demo_app.button if b.label == "Reset session"
    )
    reset_btn.click().run()
    assert _get(demo_app.session_state, KEYS.library) is None
    # The loaded-file success banner is gone after reset.
    assert not any("Loaded" in s.value for s in demo_app.success)


def _run_to_clustered(app):
    """Drive the app through load → extract → cluster and return it."""
    app.run()
    next(
        b for b in app.button if b.label == "Load example tracklist"
    ).click().run()
    next(
        b for b in app.button if b.label == "Extract features"
    ).click().run()
    next(b for b in app.button if b.label == "Cluster").click().run()
    return app


def test_changing_hyperparam_warns_results_are_stale(demo_app) -> None:
    app = _run_to_clustered(demo_app)
    # Fresh result, knobs untouched: no stale-params warning yet.
    assert not any(
        "Press **Cluster** again" in w.value for w in app.warning
    )

    # Move a hyperparameter without re-running the clustering.
    size_slider = next(
        s for s in app.slider if s.label == "min_playlist_size"
    )
    size_slider.set_value(size_slider.value + 1).run()

    assert any(
        "Press **Cluster** again" in w.value for w in app.warning
    )
    # The previous result is still on screen (we warn, we don't drop it).
    assert _get(app.session_state, KEYS.cluster_result) is not None


def test_reextract_clears_stale_cluster_params(demo_app) -> None:
    """Re-extracting must drop the cluster snapshot, not just the result.

    The extract stage invalidates downstream slots when features are
    recomputed. ``cluster_params`` (the snapshot the stale-result warning
    compares against) is a downstream slot too, so it must be cleared
    alongside ``cluster_result`` — otherwise a stale snapshot lingers.
    """
    app = _run_to_clustered(demo_app)
    assert _get(app.session_state, KEYS.cluster_params) is not None

    next(
        b for b in app.button if b.label == "Re-extract features"
    ).click().run()

    assert _get(app.session_state, KEYS.cluster_result) is None
    assert _get(app.session_state, KEYS.cluster_params) is None


def test_reclustering_clears_stale_warning(demo_app) -> None:
    app = _run_to_clustered(demo_app)
    size_slider = next(
        s for s in app.slider if s.label == "min_playlist_size"
    )
    size_slider.set_value(size_slider.value + 1).run()
    assert any("Press **Cluster** again" in w.value for w in app.warning)

    # Re-running with the new value re-syncs the snapshot; warning clears.
    next(b for b in app.button if b.label == "Cluster").click().run()
    assert not any(
        "Press **Cluster** again" in w.value for w in app.warning
    )


def test_full_pipeline_via_example_button(demo_app) -> None:
    demo_app.run()

    # 1. Load example tracklist.
    example_btn = next(
        b for b in demo_app.button if b.label == "Load example tracklist"
    )
    example_btn.click().run()
    library = _get(demo_app.session_state, KEYS.library)
    assert library is not None
    assert len(library) > 0

    # 2. Extract features.
    extract_btn = next(
        b for b in demo_app.button if b.label == "Extract features"
    )
    extract_btn.click().run()
    coverage = _get(demo_app.session_state, KEYS.coverage)
    features_df = _get(demo_app.session_state, KEYS.features_df)
    assert coverage is not None and features_df is not None
    assert coverage.resolved > 0
    # The synthetic CSV deliberately includes some synX... IDs the mock
    # returns as unknown, so the dropped-tracks path is exercised too.
    assert coverage.dropped > 0

    # 3. Cluster.
    cluster_btn = next(b for b in demo_app.button if b.label == "Cluster")
    cluster_btn.click().run()
    result = _get(demo_app.session_state, KEYS.cluster_result)
    assert result is not None
    real_clusters = sorted(
        {int(c) for c in result.tracks["cluster"] if int(c) >= 0}
    )
    assert real_clusters, "expected at least one real cluster"

    # 4. Export (default names, no Unclassified).
    write_btn = next(b for b in demo_app.button if b.label == "Write CSVs")
    write_btn.click().run()
    paths = _get(demo_app.session_state, KEYS.export_paths)
    assert paths and all(Path(p).exists() for p in paths)
    # Unclassified is excluded by default.
    assert not any(Path(p).name.startswith("Unclassified") for p in paths)
    # One CSV per real cluster.
    assert len(paths) == len(real_clusters)
