"""Top-level Streamlit script for the playlistsmith GUI.

Run via the ``playlistsmith-gui`` console script (or
``streamlit run path/to/app.py``). Streamlit launches its own subprocess
when invoked via the CLI, so this module re-installs the demo mock
transport at import time if :func:`playlistsmith.gui.cli.is_demo_mode`
indicates ``--demo`` was passed.
"""

from __future__ import annotations

import streamlit as st

from playlistsmith.gui import cli, fixtures
from playlistsmith.gui.state import KEYS, reset_pipeline_state
from playlistsmith.gui.widgets import cluster, export, extract, upload, viz

st.set_page_config(
    page_title="playlistsmith",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="expanded",
)


def _bootstrap_demo_mode() -> None:
    """Install the mock transport once per Streamlit session if requested."""
    if cli.is_demo_mode():
        st.session_state[KEYS.demo_mode] = True
        if not st.session_state.get("__demo_mock_installed", False):
            fixtures.install_mock_transport()
            st.session_state["__demo_mock_installed"] = True
    else:
        st.session_state[KEYS.demo_mode] = False


def _sidebar() -> None:
    """Global sidebar: demo badge, mode selector, reset button."""
    with st.sidebar:
        st.title("playlistsmith")
        if st.session_state.get(KEYS.demo_mode):
            st.warning(
                "**Demo mode** — running offline against the synthetic "
                "ReccoBeats fixture."
            )
        else:
            st.caption("Live mode — feature lookups hit ReccoBeats over the network.")

        st.selectbox(
            "Feature extraction mode",
            options=["precomputed"],
            index=0,
            help="Only `precomputed` is supported today; the selector "
            "stays so future modes drop in without UI churn.",
        )

        if st.button("Reset session", width='stretch'):
            reset_pipeline_state(st.session_state, clear_upload=True)
            st.rerun()


def _step_indicator() -> None:
    """Top-of-page breadcrumb showing the current pipeline stage."""
    stages = [
        ("Upload", KEYS.library),
        ("Extract", KEYS.features_df),
        ("Cluster", KEYS.cluster_result),
        ("Export", KEYS.export_paths),
    ]
    current_idx = 0
    for i, (_, key) in enumerate(stages):
        if st.session_state.get(key) is not None:
            current_idx = i + 1
    crumb = "  →  ".join(
        f"**{name}**" if i == min(current_idx, len(stages) - 1)
        else (f"✓ {name}" if i < current_idx else name)
        for i, (name, _) in enumerate(stages)
    )
    st.caption(crumb)


def main() -> None:
    """Render the whole single-page app top to bottom."""
    _bootstrap_demo_mode()
    _sidebar()
    _step_indicator()
    upload.render()
    extract.render()
    cluster.render()
    viz.render()
    export.render()


main()
