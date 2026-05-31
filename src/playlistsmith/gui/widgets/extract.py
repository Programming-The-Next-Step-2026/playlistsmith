"""Feature-extraction stage of the GUI.

One "Extract features" button. While running, an ``st.spinner`` covers
the wait; when done, the one-line :meth:`CoverageReport.render` summary
and an expandable dropped-tracks table are shown. The empty-result
edge case (zero resolved tracks) is surfaced as an ``st.info`` rather
than silently letting the clustering stage stay invisible.
"""

from __future__ import annotations

import streamlit as st

from playlistsmith import TrackLibrary
from playlistsmith.gui.state import KEYS


def render() -> None:
    """Render the extract stage if a library is loaded."""
    library: TrackLibrary | None = st.session_state.get(KEYS.library)
    if library is None:
        return

    st.header("2. Extract features")
    st.caption(
        "Look up audio features for each track via [ReccoBeats](https://reccobeats.com/) "
        "(`precomputed` mode). Tracks ReccoBeats does not know are "
        "dropped and listed below."
    )

    with st.expander("Which features are extracted?", expanded=False):
        st.markdown(
            "Each resolved track gets these nine precomputed audio features:\n"
            "\n"
            "- `acousticness`\n"
            "- `danceability`\n"
            "- `energy`\n"
            "- `instrumentalness`\n"
            "- `liveness`\n"
            "- `loudness`\n"
            "- `speechiness`\n"
            "- `tempo`\n"
            "- `valence`\n"
            "\n"
            "See the [ReccoBeats audio features documentation]"
            "(https://reccobeats.com/docs/documentation/Analysis/audio-features-extraction) "
            "for an explanation of these features."
        )

    already_extracted = st.session_state.get(KEYS.features_df) is not None
    button_label = "Re-extract features" if already_extracted else "Extract features"

    if st.button(button_label):
        with st.spinner("Extracting features…"):
            try:
                features_df, coverage = library.extract_features(
                    mode="precomputed"
                )
            except Exception as exc:  # noqa: BLE001 — surface every error.
                st.error(f"Feature extraction failed: {exc}")
                return
        st.session_state[KEYS.features_df] = features_df
        st.session_state[KEYS.coverage] = coverage
        # Invalidate downstream stages. ``cluster_params`` is the snapshot
        # the cluster stage compares against to flag stale results, so it
        # must be cleared with ``cluster_result`` — not left dangling.
        for key in (KEYS.cluster_result, KEYS.cluster_params, KEYS.export_paths):
            if key in st.session_state:
                del st.session_state[key]

    coverage = st.session_state.get(KEYS.coverage)
    features_df = st.session_state.get(KEYS.features_df)
    if coverage is None or features_df is None:
        return

    st.info(coverage.render())
    if bool(st.session_state.get(KEYS.demo_mode)):
        st.caption(
            "Running against the recorded ReccoBeats fixture; results "
            "reflect only the example library."
        )

    if not coverage.dropped_tracks.empty:
        with st.expander(
            f"Dropped tracks ({len(coverage.dropped_tracks)})", expanded=False
        ):
            st.dataframe(coverage.dropped_tracks, width='stretch')

    if len(features_df) == 0:
        st.info(
            f"0 / {coverage.total} tracks resolved — try a different CSV, "
            "or enable demo mode to explore with the example tracklist."
        )
