"""Visualisation stage of the GUI.

Primary view: UMAP-2D scatter (always available, from
``diagnostics.projection_2d``). Optional toggle: PCA-3D scatter when
the cluster module emits ``diagnostics.projection_3d`` (plan §6.2).
For very small libraries (fewer than three features or rows after
preprocessing) the 3-D projection is not produced and the toggle is
greyed out with an explanatory tooltip.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from playlistsmith.gui.state import KEYS

_UNCLASSIFIED_LABEL = -1
_UNCLASSIFIED_COLOR = "#999999"
_UNCLASSIFIED_LEGEND = "Unclassified"

#: Plain-language explanation of each projection, shown as the radio tooltip.
_PROJECTION_HELP = (
    "Both views squash the many audio features down to a few dimensions so "
    "the library fits on a chart — they don't change the clustering.\n\n"
    "**UMAP (2-D)**: lays tracks out so that songs that sound similar sit "
    "close together. Great for *seeing* the clusters, but distances and "
    "directions on the chart aren't literal.\n\n"
    "**PCA (3-D)**: rotates the data to the three axes that capture the most "
    "variation between tracks. More faithful to real distances, and each "
    "axis carries a measurable share of the variance."
)
#: Appended to the tooltip when the 3-D view is unavailable.
_PROJECTION_HELP_NO_3D = (
    "\n\n_PCA-3D needs at least three feature columns; this library is too "
    "small, so only UMAP is available._"
)


def _cluster_label(cluster_id: int) -> str:
    """Pretty cluster label used both in legend keys and color mapping."""
    if int(cluster_id) == _UNCLASSIFIED_LABEL:
        return _UNCLASSIFIED_LEGEND
    return f"Cluster {int(cluster_id)}"


def _hover_columns(result) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    """Row-aligned per-track columns that drive the hovertemplate."""
    tracks = result.tracks.reset_index(drop=True)
    return tracks[["title", "artist", "cluster", "cluster_summary"]]


def _add_cluster_label_column(df: pd.DataFrame) -> pd.DataFrame:
    df["cluster_label"] = df["cluster"].apply(_cluster_label)
    return df


def _render_2d(result) -> None:  # type: ignore[no-untyped-def]
    proj = result.diagnostics.projection_2d.reset_index(drop=True)
    df = _add_cluster_label_column(
        pd.concat([proj, _hover_columns(result)], axis=1)
    )
    fig = px.scatter(
        df,
        x="dim1",
        y="dim2",
        color="cluster_label",
        color_discrete_map={_UNCLASSIFIED_LEGEND: _UNCLASSIFIED_COLOR},
        custom_data=["title", "artist", "cluster", "cluster_summary"],
        height=520,
    )
    fig.update_traces(
        marker=dict(size=9, opacity=0.85, line=dict(width=0)),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]}<br>"
            "Cluster %{customdata[2]}: %{customdata[3]}<extra></extra>"
        ),
    )
    fig.update_layout(
        xaxis_title="UMAP dim 1",
        yaxis_title="UMAP dim 2",
        legend_title="",
    )
    st.plotly_chart(fig, width='stretch')


def _render_3d(result) -> None:  # type: ignore[no-untyped-def]
    proj = result.diagnostics.projection_3d.reset_index(drop=True)
    df = _add_cluster_label_column(
        pd.concat([proj, _hover_columns(result)], axis=1)
    )
    fig = px.scatter_3d(
        df,
        x="pc1",
        y="pc2",
        z="pc3",
        color="cluster_label",
        color_discrete_map={_UNCLASSIFIED_LEGEND: _UNCLASSIFIED_COLOR},
        custom_data=["title", "artist", "cluster", "cluster_summary"],
        height=620,
    )
    fig.update_traces(
        marker=dict(size=5, opacity=0.85, line=dict(width=0)),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]}<br>"
            "Cluster %{customdata[2]}: %{customdata[3]}<extra></extra>"
        ),
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            zaxis_title="PC 3",
        ),
        legend_title="",
    )
    st.plotly_chart(fig, width='stretch')


def render() -> None:
    """Render the visualisation toggle if a clustering result is available."""
    result = st.session_state.get(KEYS.cluster_result)
    if result is None:
        return

    st.header("4. Visualise")

    has_3d = result.diagnostics.projection_3d is not None
    view_options = ["UMAP (2-D)"]
    if has_3d:
        view_options.append("PCA (3-D)")

    view = st.radio(
        "Projection",
        view_options,
        index=0,
        horizontal=True,
        help=_PROJECTION_HELP if has_3d else _PROJECTION_HELP + _PROJECTION_HELP_NO_3D,
    )

    if view == "UMAP (2-D)":
        st.caption(
            f"UMAP projection ({result.diagnostics.projection_method.upper()}). "
            "Hover for track title, artist, and cluster summary."
        )
        _render_2d(result)
    else:
        ev = result.diagnostics.pca_explained_variance_ratio[:3]
        st.caption(
            "PCA projection on the z-scored modelling matrix "
            f"(first three components explain {sum(ev):.1%} of variance). "
            "Drag to rotate; hover for track title, artist, and cluster summary."
        )
        _render_3d(result)
