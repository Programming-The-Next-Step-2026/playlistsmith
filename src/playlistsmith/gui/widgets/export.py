"""Export stage of the GUI.

Lets the user rename clusters (one row per real cluster, Unclassified
shown read-only), validates names inline, and on click writes one CSV
per cluster via :func:`playlistsmith.io.playlist_export.write_cluster_csvs`.
Each written CSV is also offered as an ``st.download_button`` so users
on a remote Streamlit deployment can grab the files without shell
access.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from playlistsmith.io import playlist_export
from playlistsmith.io.playlist_export import validate_name
from playlistsmith.gui.state import KEYS

_UNCLASSIFIED_LABEL = -1


def _build_editor_frame(result) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    """Build the cluster / size / summary / name table for the editor."""
    sizes = result.tracks.groupby("cluster").size().rename("size")
    df = result.descriptions[["cluster", "cluster_summary"]].copy()
    df["size"] = df["cluster"].map(sizes).fillna(0).astype(int)
    df["name"] = df["cluster"].apply(
        lambda c: "Unclassified" if int(c) == _UNCLASSIFIED_LABEL
        else f"Cluster {int(c)}"
    )
    return df[["cluster", "size", "cluster_summary", "name"]].reset_index(drop=True)


def _validate_naming(edited: pd.DataFrame) -> tuple[dict[int, str], list[str]]:
    """Pull the user-edited names out and check them.

    Returns:
        A ``(naming, errors)`` tuple: ``naming`` is the validated
        ``{cluster_id: name}`` mapping for real clusters only; ``errors``
        is a list of human-readable error strings (empty when valid).
    """
    errors: list[str] = []
    naming: dict[int, str] = {}
    real_rows = edited[edited["cluster"] != _UNCLASSIFIED_LABEL]
    for _, row in real_rows.iterrows():
        cid = int(row["cluster"])
        name = str(row["name"])
        if not validate_name(name):
            errors.append(
                f"Cluster {cid}: {name!r} is not a valid filename — "
                "avoid /, \\, :, *, ?, \", <, >, |, and leading/trailing dots."
            )
        naming[cid] = name.strip()
    counts: dict[str, int] = {}
    for n in naming.values():
        counts[n] = counts.get(n, 0) + 1
    duplicates = [n for n, c in counts.items() if c > 1]
    for n in duplicates:
        errors.append(f"Playlist name {n!r} is used by more than one cluster.")
    return naming, errors


def render() -> None:
    """Render the export stage if a clustering result is present."""
    result = st.session_state.get(KEYS.cluster_result)
    if result is None:
        return

    st.header("5. Export playlists")
    st.caption(
        "Each cluster becomes one CSV with `Track URI`, `Track Name`, "
        "`Artist Name(s)` and the audio features. Rename the playlists "
        "below — the names become filenames."
    )

    base_dir_str = st.text_input(
        "Output directory (server-side path)",
        value=str(Path.cwd() / "playlistsmith_out"),
    )
    include_unclassified = st.checkbox(
        "Also write Unclassified.csv (cluster -1)", value=False
    )
    write_combined = st.checkbox(
        "Also write a combined playlists.csv (one file, all clusters)",
        value=False,
    )

    editor_frame = _build_editor_frame(result)
    edited = st.data_editor(
        editor_frame,
        key="export_name_editor",
        hide_index=True,
        disabled=("cluster", "size", "cluster_summary"),
        column_config={
            "cluster": st.column_config.NumberColumn("Cluster", width="small"),
            "size": st.column_config.NumberColumn("Size", width="small"),
            "cluster_summary": st.column_config.TextColumn("Auto description"),
            "name": st.column_config.TextColumn(
                "Playlist name (filename)", required=True
            ),
        },
        use_container_width=True,
    )

    naming, errors = _validate_naming(edited)
    for err in errors:
        st.error(err)

    disabled = bool(errors)
    if st.button("Write CSVs", type="primary", disabled=disabled):
        out_dir = Path(base_dir_str).expanduser()
        try:
            paths = playlist_export.write_cluster_csvs(
                result,
                output_dir=out_dir,
                naming=naming,
                features_df=st.session_state.get(KEYS.features_df),
                include_unclassified=include_unclassified,
                write_combined=write_combined,
            )
        except (ValueError, OSError) as exc:
            st.error(f"Could not write CSVs: {exc}")
            return
        st.session_state[KEYS.export_paths] = paths
        st.success(f"Wrote {len(paths)} file(s) to {out_dir}.")

    paths: list[Path] | None = st.session_state.get(KEYS.export_paths)
    if not paths:
        return

    st.subheader("Download")
    for path in paths:
        try:
            data = path.read_bytes()
        except OSError as exc:
            st.error(f"Could not read {path}: {exc}")
            continue
        st.download_button(
            label=f"Download {path.name}",
            data=data,
            file_name=path.name,
            mime="text/csv",
            key=f"dl_{path.name}",
        )
