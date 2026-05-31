"""Clustering stage of the GUI.

Renders the method selector and method-specific hyperparameters, runs
:func:`playlistsmith.cluster` on click, and surfaces the quality panel
(BIC curve where applicable, silhouette/cohesion, cluster sizes,
z-profile heatmap, post-processing warnings).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import playlistsmith as ps
from playlistsmith.gui.state import KEYS

#: Maps the package-level method id to the display label the user sees.
_METHOD_LABELS: dict[str, str] = {
    "gmm": "GMM",
    "kmeans": "K-means",
    "hdbscan": "HDBSCAN",
}
_METHODS: tuple[str, ...] = tuple(_METHOD_LABELS.keys())

#: Tooltip text for the method selector itself.
_METHOD_HELP = (
    "**GMM** (default): soft, probabilistic assignments; BIC picks k. Best "
    "for libraries with overlapping content.\n\n"
    "**K-means**: hard partition; tight, equal-sized blobs. Faster, no "
    "posteriors. Good when clusters are well separated.\n\n"
    "**HDBSCAN**: density-based, no `k`. Discovers the number of clusters "
    "and routes outliers to Unclassified. Best when some tracks really "
    "shouldn't belong to any playlist."
)

#: Tooltip text for every hyperparameter widget.
_HELP_MIN_PLAYLIST_SIZE = (
    "Smallest size a final cluster is allowed to keep. Clusters smaller "
    "than this are collapsed into the **Unclassified** bucket (cluster "
    "-1) so you don't end up with two-track playlists. Raise it to force "
    "larger, fewer playlists; lower it to keep small niche groups."
)
_HELP_MAX_PLAYLIST_SHARE = (
    "Maximum share of the library any one cluster is allowed to hold "
    "before the pipeline warns you. The pipeline does **not** auto-split "
    "— it surfaces a warning so you can re-run with a larger k or switch "
    "to HDBSCAN. Lower it to be stricter about dominant clusters."
)
_HELP_K_RANGE = (
    "Range of candidate cluster counts to evaluate. GMM picks the best k "
    "by BIC inside this range; K-means picks by silhouette. Wider range "
    "= more candidates evaluated (slower); narrower = faster but you "
    "might miss the best k. " "One value can be selected by dragging the "
    "knobs on top of each other."
)
_HELP_HDBSCAN_MIN_CLUSTER_SIZE = (
    "Smallest group of tracks HDBSCAN is allowed to call a cluster. "
    "Raise it to get fewer, larger clusters and more Unclassified "
    "outliers; lower it to discover smaller niches at the cost of more "
    "fragmentation."
)
_HELP_HDBSCAN_MIN_SAMPLES = (
    "How conservative HDBSCAN is about declaring core points. Higher "
    "values produce more outliers (Unclassified) and more conservative "
    "clusters. 0 means \"use the library default\" (= min_cluster_size)."
)


def _format_method(method_id: str) -> str:
    """Return the human-facing label for a method id (used by selectbox)."""
    return _METHOD_LABELS.get(method_id, method_id)


def _reset_cluster_state() -> None:
    """Clear the cluster + export slots — e.g. after changing method."""
    for key in (KEYS.cluster_result, KEYS.cluster_params, KEYS.export_paths):
        if key in st.session_state:
            del st.session_state[key]


def _on_method_change() -> None:
    """Selectbox ``on_change`` callback: invalidate stale fit + export."""
    _reset_cluster_state()


def _render_method_params(method: str, n_tracks: int) -> dict[str, object]:
    """Render method-specific widgets and collect their values."""
    params: dict[str, object] = {}
    k_upper = max(3, min(12, n_tracks - 1))

    if method in ("gmm", "kmeans"):
        k_min, k_max = st.slider(
            "k range (candidate cluster counts)",
            min_value=2,
            max_value=max(3, k_upper),
            value=(2, min(8, k_upper)),
            step=1,
            help=_HELP_K_RANGE,
        )
        params["k_range"] = range(k_min, k_max + 1)
    elif method == "hdbscan":
        params["hdbscan_min_cluster_size"] = st.number_input(
            "hdbscan_min_cluster_size",
            min_value=2,
            max_value=max(2, n_tracks),
            value=max(2, min(5, n_tracks // 4)),
            step=1,
            help=_HELP_HDBSCAN_MIN_CLUSTER_SIZE,
        )
        params["hdbscan_min_samples"] = st.number_input(
            "hdbscan_min_samples (0 = use default)",
            min_value=0,
            max_value=max(1, n_tracks),
            value=0,
            step=1,
            help=_HELP_HDBSCAN_MIN_SAMPLES,
        ) or None
    return params


def _render_quality_panel(result: "ps.ClusterPipelineResult") -> None:
    """Render the BIC / silhouette / cluster sizes / heatmap block."""
    clustering = result.clustering
    cols = st.columns(3)
    cols[0].metric("k chosen", clustering.k)
    cols[1].metric("silhouette", f"{clustering.silhouette:.3f}")
    cols[2].metric("cohesion", f"{clustering.cohesion:.3f}")

    bic_curve = getattr(clustering, "bic_curve", None) or {}
    if bic_curve:
        bic_df = pd.DataFrame(
            {"k": list(bic_curve.keys()), "BIC": list(bic_curve.values())}
        ).set_index("k")
        st.line_chart(bic_df, height=200)

    sizes = (
        result.tracks.groupby("cluster").size().rename("size").to_frame()
    )
    st.bar_chart(sizes, height=200)

    heatmap = result.diagnostics.zprofile_heatmap
    # Streamlit's default ~5-row dataframe viewport hides clusters past the
    # third one; size the widget to fit every row of the heatmap plus the
    # header so the whole thing reads at a glance.
    st.markdown("**Per-cluster z-profile heatmap**")
    st.dataframe(
        heatmap.style.background_gradient(
            cmap="coolwarm", axis=None, vmin=-2, vmax=2
        ),
        width='stretch',
        height=int(38 * (len(heatmap) + 1) + 8),
    )

    for w in result.warnings:
        st.warning(w)


def render() -> None:
    """Render the cluster stage if features are present."""
    features_df = st.session_state.get(KEYS.features_df)
    if features_df is None or len(features_df) == 0:
        return

    st.header("3. Cluster")
    st.caption(
        "GMM is preferred (soft assignments, BIC for k); K-means and "
        "HDBSCAN are available for hard / density-based needs."
    )

    method = st.selectbox(
        "Method",
        _METHODS,
        index=0,
        format_func=_format_method,
        key="cluster_method",
        on_change=_on_method_change,
        help=_METHOD_HELP,
    )

    common_cols = st.columns(2)
    with common_cols[0]:
        min_playlist_size = st.slider(
            "min_playlist_size",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            help=_HELP_MIN_PLAYLIST_SIZE,
        )
    with common_cols[1]:
        max_playlist_share = st.slider(
            "max_playlist_share",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help=_HELP_MAX_PLAYLIST_SHARE,
        )

    params = _render_method_params(method, n_tracks=len(features_df))

    # Snapshot of every knob feeding the run. Stored alongside the result so
    # we can tell, on later reruns, whether the user has since moved a slider
    # and the displayed result is now stale. ``k_range`` is a ``range``, which
    # compares by value, so equality on this dict is exact.
    current_params = {
        "method": method,
        "min_playlist_size": min_playlist_size,
        "max_playlist_share": max_playlist_share,
        **params,
    }

    if st.button("Cluster", type="primary"):
        with st.spinner(f"Clustering with {_format_method(method)}…"):
            try:
                result = ps.cluster(
                    features_df,
                    method=method,
                    random_state=0,
                    min_playlist_size=min_playlist_size,
                    max_playlist_share=max_playlist_share,
                    **params,
                )
            except (ValueError, NotImplementedError) as exc:
                st.error(f"Clustering failed: {exc}")
                return
        st.session_state[KEYS.cluster_result] = result
        st.session_state[KEYS.cluster_params] = current_params
        if KEYS.export_paths in st.session_state:
            del st.session_state[KEYS.export_paths]

    result = st.session_state.get(KEYS.cluster_result)
    if result is None:
        return

    if st.session_state.get(KEYS.cluster_params) != current_params:
        st.warning(
            "⚠️ Hyperparameters have changed since the results were "
            "computed. Press **Cluster** again to update the results below."
        )

    _render_quality_panel(result)
