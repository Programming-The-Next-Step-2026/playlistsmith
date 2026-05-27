"""Write per-cluster CSVs from a clustering result.

Takes the ``tracks`` frame produced by
:func:`playlistsmith.cluster.cluster` and slices it into one CSV per
cluster, named by the caller (the GUI lets the user rename clusters
before export). The output is Exportify-compatible so the user can
re-import each CSV into Spotify via a Spotify-import flow that accepts
Exportify-shaped CSVs: ``Track URI`` is the leading column, followed by
``Track Name`` and ``Artist Name(s)``.

Cluster ``-1`` ("Unclassified") is excluded from the per-cluster output
by default; it can be requested explicitly (``include_unclassified=True``)
or rolled into the optional combined CSV via ``write_combined=True``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from playlistsmith.io.csv_loader import ARTIST, SPOTIFY_ID, TITLE

__all__ = [
    "DEFAULT_COMBINED_FILENAME",
    "EXPORTIFY_ARTIST",
    "EXPORTIFY_TITLE",
    "EXPORTIFY_URI",
    "UNCLASSIFIED_FILENAME",
    "validate_name",
    "write_cluster_csvs",
]

#: Exportify column names. Re-exported so callers (e.g. the GUI export
#: widget) can reference the schema without hard-coding strings.
EXPORTIFY_URI = "Track URI"
EXPORTIFY_TITLE = "Track Name"
EXPORTIFY_ARTIST = "Artist Name(s)"

_SPOTIFY_TRACK_URI_PREFIX = "spotify:track:"

#: Filename for the Unclassified bucket when ``include_unclassified=True``.
UNCLASSIFIED_FILENAME = "Unclassified.csv"

#: Filename for the combined CSV when ``write_combined=True``.
DEFAULT_COMBINED_FILENAME = "playlists.csv"

#: Cluster id reserved for the Unclassified bucket. Kept local rather than
#: imported from cluster/public.py to avoid pulling sklearn/umap into a
#: pure-io module.
_UNCLASSIFIED_LABEL: int = -1

#: Characters that are unsafe in playlist filenames across the common
#: operating systems we care about (Windows, macOS, Linux). Path
#: separators are explicitly included so a user-supplied "name" cannot
#: navigate the filesystem.
_UNSAFE_NAME_CHARS = frozenset('/\\:*?"<>|')


def validate_name(name: str) -> bool:
    """Return whether ``name`` is safe to use as a playlist filename.

    Rules:

    - non-empty after stripping whitespace,
    - no path separators or filesystem-reserved characters,
    - no leading or trailing ``.``.

    Args:
        name: Candidate playlist filename (without extension).

    Returns:
        ``True`` if the name is safe to use, ``False`` otherwise.
    """
    if not isinstance(name, str):
        return False
    stripped = name.strip()
    if not stripped:
        return False
    if stripped.startswith(".") or stripped.endswith("."):
        return False
    return not any(ch in _UNSAFE_NAME_CHARS for ch in stripped)


def _tracks_frame(source: Any) -> pd.DataFrame:
    """Coerce ``source`` to a tracks DataFrame.

    Accepts either a :class:`~playlistsmith.cluster.ClusterPipelineResult`
    or a DataFrame directly. The pipeline result carries its tracks frame
    on the ``tracks`` attribute; we read it lazily so importing this
    module does not pull in the clustering machinery.
    """
    if isinstance(source, pd.DataFrame):
        return source
    tracks = getattr(source, "tracks", None)
    if isinstance(tracks, pd.DataFrame):
        return tracks
    raise TypeError(
        "write_cluster_csvs expects a DataFrame or an object with a "
        f"`.tracks` DataFrame attribute; got {type(source).__name__}."
    )


def _resolve_name(
    cluster_id: int,
    naming: Mapping[int, str] | Callable[[int], str] | None,
) -> str:
    """Look up the filename stem for ``cluster_id``.

    Falls back to ``cluster_<id>`` when ``naming`` is ``None`` or does
    not have an entry for this cluster.
    """
    default = f"cluster_{cluster_id}"
    if naming is None:
        return default
    if callable(naming):
        value = naming(cluster_id)
        return value if value else default
    if isinstance(naming, Mapping):
        return naming.get(cluster_id, default)
    raise TypeError(
        "`naming` must be None, a Mapping[int, str], or Callable[[int], "
        f"str]; got {type(naming).__name__}."
    )


def _build_export_frame(
    cluster_tracks: pd.DataFrame, features_df: pd.DataFrame | None
) -> pd.DataFrame:
    """Build the Exportify-shaped frame written to a single cluster CSV.

    Columns: ``Track URI``, ``Track Name``, ``Artist Name(s)``, ``Cluster``,
    ``Cluster Summary``, plus any audio-feature columns merged from
    ``features_df`` (keyed on ``spotify_id``).
    """
    bare_ids = cluster_tracks[SPOTIFY_ID].astype(str)
    out = pd.DataFrame(
        {
            EXPORTIFY_URI: _SPOTIFY_TRACK_URI_PREFIX + bare_ids,
            EXPORTIFY_TITLE: cluster_tracks[TITLE].astype(str),
            EXPORTIFY_ARTIST: cluster_tracks[ARTIST].astype(str),
            "Cluster": cluster_tracks["cluster"].astype(int),
            "Cluster Summary": cluster_tracks["cluster_summary"].astype(str),
        }
    )
    if features_df is not None:
        identity = {SPOTIFY_ID, TITLE, ARTIST}
        feature_cols = [c for c in features_df.columns if c not in identity]
        aligned = (
            features_df.set_index(SPOTIFY_ID)[feature_cols]
            .reindex(bare_ids.to_numpy())
            .reset_index(drop=True)
        )
        out = pd.concat([out.reset_index(drop=True), aligned], axis=1)
    return out


def _validate_names(names: dict[int, str]) -> None:
    """Raise ``ValueError`` if any value is unsafe or a duplicate."""
    for cluster_id, name in names.items():
        if not validate_name(name):
            raise ValueError(
                f"Playlist name for cluster {cluster_id} is unsafe or empty: "
                f"{name!r}. Avoid path separators, the characters /\\:*?\"<>|, "
                "and leading/trailing dots."
            )
    seen: dict[str, int] = {}
    for cluster_id, name in names.items():
        if name in seen:
            raise ValueError(
                f"Playlist names must be unique: clusters {seen[name]} and "
                f"{cluster_id} both map to {name!r}."
            )
        seen[name] = cluster_id


def write_cluster_csvs(
    result: Any,
    output_dir: Path | str,
    naming: Mapping[int, str] | Callable[[int], str] | None = None,
    *,
    features_df: pd.DataFrame | None = None,
    include_unclassified: bool = False,
    write_combined: bool = False,
    combined_filename: str = DEFAULT_COMBINED_FILENAME,
) -> list[Path]:
    """Write one Exportify-shaped CSV per cluster to ``output_dir``.

    The output schema is::

        Track URI, Track Name, Artist Name(s), Cluster, Cluster Summary,
        <audio features if features_df is given>

    so each CSV is symmetric with the Exportify-style CSV the pipeline
    accepts as input. ``Track URI`` is the full ``spotify:track:<id>``
    form (not a bare ID) so the file is ready for re-import via any
    Exportify-compatible Spotify import flow.

    Args:
        result: Either a :class:`~playlistsmith.cluster.ClusterPipelineResult`
            (its ``.tracks`` is used) or a DataFrame with the same shape.
        output_dir: Directory to write CSVs into. Created if missing.
        naming: Override for per-cluster filenames. May be a
            ``{cluster_id: name}`` mapping or a ``cluster_id -> name``
            callable. Missing entries fall back to ``cluster_<id>``.
            The ``.csv`` extension is added automatically. Names must
            be filesystem-safe (see :func:`validate_name`) and unique.
        features_df: Optional features DataFrame (as returned by
            :func:`playlistsmith.features.extract`). When provided, the
            audio-feature columns are merged into each per-cluster CSV
            via ``spotify_id``.
        include_unclassified: If ``True``, also write a CSV for the
            ``-1`` Unclassified bucket (filename
            ``Unclassified.csv``). Defaults to ``False`` because those
            tracks were *not* assigned to a real playlist.
        write_combined: If ``True``, additionally write a single CSV
            containing every track (including Unclassified) with the
            same schema. Useful for users who want one file to inspect.
        combined_filename: Filename used when ``write_combined`` is
            ``True``. Defaults to ``playlists.csv``.

    Returns:
        Absolute paths of every CSV written, in the order they were
        written (real clusters first, then Unclassified if requested,
        then the combined CSV if requested).

    Raises:
        ValueError: If any ``naming`` value is unsafe or duplicates
            another playlist name.
        TypeError: If ``result`` is neither a DataFrame nor an object
            with a ``.tracks`` DataFrame attribute.
    """
    tracks = _tracks_frame(result)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if tracks.empty:
        return []

    real_ids = sorted({int(c) for c in tracks["cluster"] if int(c) != _UNCLASSIFIED_LABEL})

    resolved_names: dict[int, str] = {
        cid: _resolve_name(cid, naming) for cid in real_ids
    }
    if include_unclassified and (tracks["cluster"] == _UNCLASSIFIED_LABEL).any():
        resolved_names[_UNCLASSIFIED_LABEL] = _resolve_name(
            _UNCLASSIFIED_LABEL, naming
        ) if (
            callable(naming) or (isinstance(naming, Mapping) and _UNCLASSIFIED_LABEL in naming)
        ) else UNCLASSIFIED_FILENAME[:-len(".csv")]
    _validate_names(resolved_names)

    written: list[Path] = []
    ids_to_write = list(real_ids)
    if include_unclassified and _UNCLASSIFIED_LABEL in resolved_names:
        ids_to_write.append(_UNCLASSIFIED_LABEL)

    for cluster_id in ids_to_write:
        sub = tracks.loc[tracks["cluster"] == cluster_id].reset_index(drop=True)
        if sub.empty:
            continue
        frame = _build_export_frame(sub, features_df)
        path = output_dir / f"{resolved_names[cluster_id]}.csv"
        frame.to_csv(path, index=False)
        written.append(path)

    if write_combined:
        combined = _build_export_frame(tracks.reset_index(drop=True), features_df)
        combined_path = output_dir / combined_filename
        combined.to_csv(combined_path, index=False)
        written.append(combined_path)

    return written
