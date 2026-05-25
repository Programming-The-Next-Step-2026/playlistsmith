"""Upload widget: CSV file_uploader + TrackLibrary preview.

In demo mode the widget surfaces a one-click "Load example tracklist"
button that points at ``docs/example_synthetic.csv`` (the synthetic file
the vignette uses), so a first-time user can see the full pipeline run
without leaving the GUI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from playlistsmith import TrackLibrary
from playlistsmith.gui.state import KEYS, UPLOAD_IDENTITY_KEY, reset_pipeline_state

#: Path to the synthetic example CSV (lives at ``docs/example_synthetic.csv``).
#: Resolved relative to the repo root, located by walking up from this file.
_REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_CSV_PATH = _REPO_ROOT / "docs" / "example_synthetic.csv"


def _load_library_from_path(path: Path, *, filename: str, identity: str) -> None:
    """Construct a :class:`TrackLibrary` and stash it in session state."""
    try:
        library = TrackLibrary(path)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Could not load CSV: {exc}")
        return
    reset_pipeline_state(st.session_state)
    st.session_state[KEYS.library] = library
    st.session_state[KEYS.library_filename] = filename
    st.session_state[UPLOAD_IDENTITY_KEY] = identity


def _upload_identity(uploaded) -> str:  # type: ignore[no-untyped-def]
    """A stable identity for an UploadedFile across reruns.

    Streamlit assigns each upload a ``file_id`` that persists for the
    lifetime of the upload widget's value. We fall back to ``name`` +
    ``size`` if a future Streamlit version drops ``file_id``.
    """
    file_id = getattr(uploaded, "file_id", None)
    if file_id is not None:
        return f"upload:{file_id}"
    return f"upload:{uploaded.name}:{uploaded.size}"


def render() -> None:
    """Render the upload stage; populates ``st.session_state[library]``."""
    st.header("1. Upload your tracklist")
    st.caption(
        "Export your Spotify playlists as CSV using "
        "[Exportify](https://exportify.app), then upload the file here."
    )

    uploaded = st.file_uploader("Exportify CSV", type=["csv"])
    demo_mode = bool(st.session_state.get(KEYS.demo_mode))

    if demo_mode and EXAMPLE_CSV_PATH.exists():
        st.caption(
            "**Demo mode:** click below to load the bundled synthetic "
            "tracklist — the same one the vignette uses."
        )
        if st.button("Load example tracklist", type="primary"):
            _load_library_from_path(
                EXAMPLE_CSV_PATH,
                filename=EXAMPLE_CSV_PATH.name,
                identity=f"example:{EXAMPLE_CSV_PATH}",
            )

    if uploaded is not None:
        identity = _upload_identity(uploaded)
        # Only reload when the upload actually changed. Without this, a
        # rerun triggered by clicking *any* downstream button (Extract,
        # Cluster, Write CSVs, …) would re-load the library and
        # reset_pipeline_state would silently wipe the features.
        if st.session_state.get(UPLOAD_IDENTITY_KEY) != identity:
            with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False
            ) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)
            _load_library_from_path(
                tmp_path, filename=uploaded.name, identity=identity
            )

    library: TrackLibrary | None = st.session_state.get(KEYS.library)
    if library is None:
        return

    st.success(
        f"Loaded **{st.session_state.get(KEYS.library_filename, '?')}** — "
        f"{len(library)} track(s) with a Spotify ID."
    )
    with st.expander("Preview tracklist (first 20 rows)", expanded=False):
        st.dataframe(library.dataframe.head(20), use_container_width=True)
