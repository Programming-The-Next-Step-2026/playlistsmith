"""Public entry point for the clustering pipeline.

Downstream code (``viz``, ``io.playlist_export``) should import
:func:`~playlistsmith.cluster.cluster` from
:mod:`playlistsmith.cluster`. It wraps every step in the pipeline:

1. :func:`~playlistsmith.cluster.preprocess.prepare_matrix` — logit/log
   transforms, median imputation, scaling.
2. :func:`~playlistsmith.cluster.algorithms.fit_gmm` (or another method
   when explicitly requested).
3. Canonical cluster ordering: relabel clusters by descending size,
   ties broken by PC1 of the cluster mean, so cluster ``0`` means
   roughly the same thing across re-runs.
4. Playlist-shape post-processing: collapse clusters smaller than
   ``min_playlist_size`` into an "Unclassified" bucket (cluster id
   ``-1``); warn (but do not auto-split) when a cluster holds more
   than ``max_playlist_share`` of the library.
5. :func:`~playlistsmith.cluster.interpret.describe_clusters` — short
   human-readable summary per cluster.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA

from playlistsmith.cluster.algorithms import (
    ClusteringResult,
    fit_gmm,
    fit_hdbscan,
    fit_kmeans,
)
from playlistsmith.cluster.interpret import describe_clusters
from playlistsmith.cluster.preprocess import TransformLog, prepare_matrix
from playlistsmith.features.reccobeats import FEATURE_COLUMNS
from playlistsmith.io.csv_loader import ARTIST, SPOTIFY_ID, TITLE

__all__ = ["ClusterDiagnostics", "ClusterPipelineResult", "cluster"]

#: Serialises UMAP fits across threads. UMAP's layout optimisation runs
#: under numba, and the only threading layer available in many installs
#: (no ``tbb``/``omp``) is ``workqueue``, which is *not* threadsafe: two
#: concurrent parallel regions abort the whole process with "Numba
#: workqueue threading layer is terminating: Concurrent access has been
#: detected." The Streamlit GUI runs each session/rerun in its own
#: worker thread, so overlapping clustering runs would otherwise crash
#: the app. UMAP is the only numba-parallel code in the pipeline
#: (HDBSCAN here is scikit-learn's, pure C/Cython), so guarding the fit
#: with one process-wide lock is sufficient.
_UMAP_LOCK = threading.Lock()

#: Cluster id reserved for tracks in playlists that were too small to
#: be useful (and, in future, for HDBSCAN noise points).
UNCLASSIFIED_LABEL: int = -1

#: Human-readable summary string for the Unclassified bucket.
_UNCLASSIFIED_SUMMARY: str = "unclassified"

#: Methods recognised by :func:`cluster`.
_SUPPORTED_METHODS: tuple[str, ...] = ("gmm", "kmeans", "hdbscan")


@dataclass
class ClusterDiagnostics:
    """Visualisation-ready artefacts emitted by the cluster pipeline.

    The cluster step itself does not import any plotting library; this
    bundle contains everything the viz subpackage needs to draw scatter
    plots, heatmaps and selection diagnostics (plan §6).

    Attributes:
        pca_components: ``(n_tracks × n_components)`` of PCA coordinates
            in the transformed-scaled space. Row-aligned with the
            ``tracks`` frame. Columns are ``pc1``, ``pc2``, .... PCA on
            nine features is cheap and deterministic; it is also used
            for the canonical cluster ordering (see
            :func:`_canonical_order`).
        pca_explained_variance_ratio: Variance share of each PCA
            component, in component order — useful for axis labels.
        projection_2d: ``(n_tracks × 2)`` 2-D embedding for scatter
            plots. Row-aligned with ``tracks``; columns ``dim1``,
            ``dim2``. UMAP is the MIR standard (plan §6).
        projection_method: Which projection produced ``projection_2d``
            (``"umap"`` here; ``"tsne"`` if we ever fall back).
        projection_3d: ``(n_tracks × 3)`` PCA-3D embedding for the
            optional 3-D scatter (plan §6.2). Row-aligned with
            ``tracks``; columns ``pc1``, ``pc2``, ``pc3``. Fit on the
            same z-scored modelling matrix used for clustering, so it
            shares the axes the model actually saw. ``None`` when the
            modelling matrix has fewer than three columns or rows (the
            GUI greys the 3-D toggle out in that case).
        zprofile_heatmap: ``(n_clusters × n_features)`` z-profile
            frame indexed by cluster id (including ``-1`` for the
            Unclassified bucket if present). Columns are the canonical
            feature columns. Heatmap-ready.

    Examples:
        Reached through :attr:`ClusterPipelineResult.diagnostics` (see
        :func:`cluster` for how ``features`` is built):

        >>> diag = cluster(features, k_range=range(2, 6)).diagnostics
        >>> diag.projection_method
        'umap'
        >>> diag.projection_2d.columns.tolist()
        ['dim1', 'dim2']
        >>> [round(v, 2) for v in diag.pca_explained_variance_ratio[:3]]
        [0.67, 0.13, 0.11]
        >>> diag.zprofile_heatmap.shape  # (n_clusters, n_features)
        (2, 9)
    """

    pca_components: pd.DataFrame
    pca_explained_variance_ratio: list[float]
    projection_2d: pd.DataFrame
    projection_method: str
    zprofile_heatmap: pd.DataFrame
    projection_3d: pd.DataFrame | None = None


@dataclass
class ClusterPipelineResult:
    """Output of the full clustering pipeline.

    Attributes:
        tracks: One row per surviving track with the identity columns
            plus the final ``cluster`` id (``-1`` for the Unclassified
            bucket) and ``cluster_summary`` string. This is the frame
            that ``io.playlist_export`` consumes.
        descriptions: One row per final cluster id (including
            ``-1`` if present) with the columns produced by
            :func:`~playlistsmith.cluster.interpret.describe_clusters`,
            with ``summary`` renamed to ``cluster_summary``.
        clustering: The canonical-ordered raw fit. Kept for diagnostics
            and so callers can inspect posteriors / BIC curves /
            stability ARI.
        transform_log: Preprocessing record (logit/log columns, imputed
            cells, dropped tracks).
        warnings: Post-processing warnings (e.g. a dominant cluster
            that exceeded ``max_playlist_share``).
        diagnostics: Visualisation-ready artefacts (PCA coordinates,
            2-D projection, z-profile heatmap) — see
            :class:`ClusterDiagnostics`.

    Examples:
        Returned by :func:`cluster` (see its example for how ``features``
        is built):

        >>> result = cluster(features, k_range=range(2, 6))
        >>> result.tracks.columns.tolist()
        ['spotify_id', 'title', 'artist', 'cluster', 'cluster_summary']
        >>> result.descriptions.columns.tolist()
        ['cluster', 'size', 'top_features', 'z_profile', 'cluster_summary']
        >>> result.clustering.k
        2
        >>> result.warnings
        []
    """

    tracks: pd.DataFrame
    descriptions: pd.DataFrame
    clustering: ClusteringResult
    transform_log: TransformLog
    diagnostics: ClusterDiagnostics
    warnings: list[str] = field(default_factory=list)


def _canonical_order(
    result: ClusteringResult, X: pd.DataFrame
) -> ClusteringResult:
    """Relabel clusters by descending size, with PC1 tiebreak.

    Cluster ``0`` becomes the largest cluster after this transformation,
    so re-runs that recover the same partition produce stable cluster
    numbering.

    Args:
        result: The raw clustering result.
        X: The modelling matrix the clusterer saw, used to compute PC1
            of each cluster mean for tiebreaking.

    Returns:
        A new :class:`ClusteringResult` with labels, posteriors and
        cluster-mean rows reordered into canonical order.
    """
    # Native -1 (HDBSCAN noise) is left alone here; the canonical
    # ordering only re-numbers real clusters. Noise flows into the
    # Unclassified bucket together with anything the post-processing
    # step collapses.
    non_noise = result.labels[result.labels >= 0]
    sizes = (
        np.bincount(non_noise, minlength=result.k)
        if len(non_noise) > 0
        else np.zeros(result.k, dtype=int)
    )
    if result.k > 0:
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(
            result.feature_means_per_cluster.to_numpy()
        ).ravel()
        order = sorted(
            range(result.k), key=lambda c: (-int(sizes[c]), float(pc1[c]))
        )
    else:
        order = []
    old_to_new = {old: new for new, old in enumerate(order)}

    new_labels = np.array(
        [old_to_new[lbl] if lbl >= 0 else UNCLASSIFIED_LABEL for lbl in result.labels]
    )
    new_posteriors = (
        result.posteriors[:, order] if result.posteriors is not None else None
    )
    if order:
        new_means = result.feature_means_per_cluster.iloc[order].reset_index(drop=True)
    else:
        new_means = result.feature_means_per_cluster.iloc[0:0].reset_index(drop=True)

    return replace(
        result,
        labels=new_labels,
        posteriors=new_posteriors,
        feature_means_per_cluster=new_means,
    )


def _collapse_small_clusters(
    labels: np.ndarray, min_size: int
) -> np.ndarray:
    """Move tracks in clusters smaller than ``min_size`` to ``-1``.

    Because :func:`_canonical_order` puts the smallest clusters at the
    highest ids, collapsing them never leaves gaps in the surviving
    label range — kept clusters remain ``0..k'-1`` with no relabel.
    Already-``-1`` labels (HDBSCAN noise after canonical ordering)
    pass through unchanged.
    """
    non_noise = labels[labels >= 0]
    if len(non_noise) == 0:
        return labels
    sizes = np.bincount(non_noise)
    small = {c for c, n in enumerate(sizes) if n < min_size}
    if not small:
        return labels
    return np.array(
        [UNCLASSIFIED_LABEL if lbl in small or lbl < 0 else lbl for lbl in labels]
    )


def _post_processing_warnings(
    final_labels: np.ndarray, max_share: float, min_size: int
) -> list[str]:
    """Build post-processing warnings for the pipeline result.

    Two failure modes are surfaced:

    - **All tracks unclassified** — if the collapse step routed every
      track to ``-1``, no real playlists were produced. The user
      probably wants a lower ``min_playlist_size`` or a different
      ``method``.
    - **Dominant cluster** — any cluster holding more than ``max_share``
      of the library is flagged. The plan deliberately does not
      auto-split (that would re-introduce algorithm choice); we surface
      the problem and let the user re-run with a larger ``k`` or
      ``method='hdbscan'``.
    """
    total = len(final_labels)
    if total == 0:
        return []
    warnings: list[str] = []
    real_labels = final_labels[final_labels >= 0]
    if len(real_labels) == 0:
        warnings.append(
            f"All {total} tracks landed in the Unclassified bucket — no "
            "playlists were produced. Consider lowering min_playlist_size "
            f"(currently {min_size}), passing a smaller k_range, or "
            "trying method='hdbscan' with a smaller min_cluster_size."
        )
        return warnings
    for c in sorted(set(real_labels.tolist())):
        n = int((real_labels == c).sum())
        share = n / total
        if share > max_share:
            warnings.append(
                f"Cluster {c} is dominant: it holds {n}/{total} tracks "
                f"({share:.0%}) of the library, more than the "
                f"max_playlist_share={max_share:.0%} threshold. Consider "
                f"a larger k or method='hdbscan'."
            )
    return warnings


def _build_tracks_frame(
    index: pd.DataFrame,
    final_labels: np.ndarray,
    summaries: dict[int, str],
) -> pd.DataFrame:
    """Assemble the per-track output frame consumed by playlist export."""
    return pd.DataFrame(
        {
            SPOTIFY_ID: index[SPOTIFY_ID].to_numpy(),
            TITLE: index[TITLE].to_numpy(),
            ARTIST: index[ARTIST].to_numpy(),
            "cluster": final_labels,
            "cluster_summary": [summaries[int(c)] for c in final_labels],
        }
    )


def _build_descriptions_frame(
    raw_descriptions: pd.DataFrame,
    final_labels: np.ndarray,
) -> pd.DataFrame:
    """Restrict descriptions to surviving clusters, rename ``summary``
    to ``cluster_summary``, and append an Unclassified row if needed.

    Args:
        raw_descriptions: Output of :func:`describe_clusters` on the
            canonical-ordered fit (one row per *original* cluster).
        final_labels: Post-collapse labels (with possible ``-1`` entries).

    Returns:
        A frame with one row per cluster id appearing in
        ``final_labels``.
    """
    surviving = sorted(set(int(c) for c in final_labels if c >= 0))
    kept = raw_descriptions[raw_descriptions["cluster"].isin(surviving)].copy()
    kept = kept.rename(columns={"summary": "cluster_summary"})

    if UNCLASSIFIED_LABEL in final_labels:
        n_unclassified = int((final_labels == UNCLASSIFIED_LABEL).sum())
        unclassified_row = pd.DataFrame(
            [
                {
                    "cluster": UNCLASSIFIED_LABEL,
                    "size": n_unclassified,
                    "top_features": [],
                    "z_profile": {},
                    "cluster_summary": _UNCLASSIFIED_SUMMARY,
                }
            ]
        )
        kept = pd.concat([kept, unclassified_row], ignore_index=True)

    return kept.reset_index(drop=True)


def _build_diagnostics(
    X: pd.DataFrame,
    final_labels: np.ndarray,
    raw_descriptions: pd.DataFrame,
    random_state: int,
) -> ClusterDiagnostics:
    """Compute the visualisation artefacts from the modelling matrix.

    PCA is deterministic; UMAP is seeded with ``random_state`` for
    reproducibility. The heatmap is assembled from the cluster
    z-profiles already computed by
    :func:`~playlistsmith.cluster.interpret.describe_clusters`, with
    an all-zero row appended for the Unclassified bucket if one is
    present in ``final_labels``.
    """
    n, d = X.shape
    n_components = min(d, n)
    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(X.to_numpy())
    pca_df = pd.DataFrame(
        pca_coords,
        columns=[f"pc{i + 1}" for i in range(n_components)],
    )

    # UMAP needs n_neighbors < n_samples; clamp for small libraries.
    umap_neighbors = min(15, max(2, n - 1))
    # ``n_jobs=1`` is implied when ``random_state`` is set; passing it
    # explicitly silences a UserWarning from umap-learn.
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_neighbors,
        random_state=random_state,
        n_jobs=1,
    )
    # Serialise the numba-backed fit; see _UMAP_LOCK for why.
    with _UMAP_LOCK:
        proj_arr = reducer.fit_transform(X.to_numpy())
    projection_2d = pd.DataFrame(proj_arr, columns=["dim1", "dim2"])

    # Cluster × feature heatmap. Real clusters come from describe_clusters'
    # z_profile column; the Unclassified bucket (if present) gets a row
    # of zeros (no defining mean).
    rows: dict[int, dict[str, float]] = {}
    for _, row in raw_descriptions.iterrows():
        cluster_id = int(row["cluster"])
        rows[cluster_id] = dict(row["z_profile"])
    final_ids = sorted(set(int(c) for c in final_labels))
    surviving = {c for c in final_ids if c >= 0}
    heatmap_rows = []
    for cid in final_ids:
        if cid in surviving and cid in rows:
            heatmap_rows.append([rows[cid].get(col, 0.0) for col in FEATURE_COLUMNS])
        else:
            heatmap_rows.append([0.0] * len(FEATURE_COLUMNS))
    heatmap = pd.DataFrame(
        heatmap_rows, index=final_ids, columns=list(FEATURE_COLUMNS)
    )

    # 3-D PCA toggle (plan §6.2). Reuses the PCA fit above so the 3-D
    # coordinates share the axes the canonical-ordering tiebreak uses.
    projection_3d: pd.DataFrame | None
    if n_components >= 3:
        pca_3d = pca_df.iloc[:, :3].copy()
        pca_3d.columns = ["pc1", "pc2", "pc3"]
        projection_3d = pca_3d
    else:
        projection_3d = None

    return ClusterDiagnostics(
        pca_components=pca_df,
        pca_explained_variance_ratio=[float(x) for x in pca.explained_variance_ratio_],
        projection_2d=projection_2d,
        projection_method="umap",
        zprofile_heatmap=heatmap,
        projection_3d=projection_3d,
    )


def cluster(
    features_df: pd.DataFrame,
    method: str = "gmm",
    random_state: int = 0,
    min_playlist_size: int = 5,
    max_playlist_share: float = 0.5,
    k_range: Iterable[int] = range(2, 13),
    hdbscan_min_cluster_size: int | None = None,
    hdbscan_min_samples: int | None = None,
) -> ClusterPipelineResult:
    """Cluster ``features_df`` and produce playlist-ready output.

    Wraps preprocessing, GMM fitting, canonical cluster ordering, and
    playlist-shape post-processing. Soft-assignment posteriors and
    quality diagnostics travel along on the returned ``clustering``
    field; the ``tracks`` and ``descriptions`` frames are the
    consumer-facing artefacts.

    Args:
        features_df: A features frame as produced by
            :func:`playlistsmith.features.extract`.
        method: ``"gmm"`` (default), ``"kmeans"`` or ``"hdbscan"``. 
            Other strings raise ``ValueError``.
        random_state: Seed threaded through preprocessing-independent
            stochastic steps (the GMM fit and its stability re-fit).
        min_playlist_size: Post-processing floor applied to *every*
            method, *after* clustering finishes: any cluster with fewer
            than this many tracks is collapsed wholesale into the
            Unclassified bucket (cluster id ``-1``). It only thresholds
            and relabels — it never reshapes the clusters it keeps.
            Governs output usability ("how small a playlist is too small
            to bother exporting"), as opposed to
            ``hdbscan_min_cluster_size``, which governs cluster discovery
            inside the HDBSCAN fit. For an HDBSCAN run both apply in
            sequence, so the effective floor on a surviving playlist is
            the larger of the two. Defaults to ``5``.
        max_playlist_share: If any cluster holds more than this share
            of the library, a warning is emitted (no auto-split).
        k_range: Candidate ``k`` values for GMM model selection.
        hdbscan_min_cluster_size: Smallest group of tracks HDBSCAN is
            allowed to call a cluster (``method="hdbscan"`` only). Unlike
            ``min_playlist_size``, this acts *during* the fit and governs
            cluster discovery: it shapes which groups HDBSCAN forms in the
            first place (raising it changes which merges happen, so the
            surviving clusters can be reshaped, not just pruned). ``None``
            (the default) uses ``max(5, n_tracks // 50)``. Raise it for
            fewer, larger playlists and more Unclassified outliers; lower
            it to surface smaller niches. Ignored by the GMM and K-means
            methods, which take a cluster *count* via ``k_range`` instead.
        hdbscan_min_samples: HDBSCAN's "anti-noise shield" — the number
            of close neighbours a track must have before it counts as a
            *core point* that can seed a cluster (``method="hdbscan"``
            only). Think of it as how big a crowd a track needs around it
            to be allowed to start a playlist: higher values make the
            algorithm more conservative, so more loosely-grouped tracks are
            left as noise and routed to the Unclassified bucket; lower
            values are more permissive and produce fewer outliers. ``None``
            (the default) falls back to ``hdbscan_min_cluster_size``.
            Ignored by the GMM and K-means methods.

    Returns:
        A :class:`ClusterPipelineResult` with per-track labels, per-
        cluster summaries, the raw clustering, the preprocessing log
        and any post-processing warnings.

    Raises:
        ValueError: If ``method`` is not one of ``{"gmm", "kmeans",
            "hdbscan"}``.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from playlistsmith.cluster import cluster
        >>> # Synthesise two obvious groups so the example needs no network.
        >>> cols = ["acousticness", "danceability", "energy", "instrumentalness",
        ...         "liveness", "loudness", "speechiness", "tempo", "valence"]
        >>> sd = [0.02, 0.02, 0.02, 0.02, 0.02, 1.0, 0.02, 3.0, 0.02]
        >>> mellow = [0.88, 0.30, 0.20, 0.40, 0.15, -17.0, 0.05, 85.0, 0.25]
        >>> upbeat = [0.12, 0.80, 0.90, 0.40, 0.15, -6.0, 0.05, 128.0, 0.75]
        >>> rng = np.random.default_rng(5)
        >>> values = np.vstack([rng.normal(mellow, sd, (15, 9)),
        ...                     rng.normal(upbeat, sd, (15, 9))])
        >>> features = pd.DataFrame(values, columns=cols)
        >>> features.insert(0, "artist", "Various")
        >>> features.insert(0, "title", [f"Track {i}" for i in range(30)])
        >>> features.insert(0, "spotify_id", [f"id{i:02d}" for i in range(30)])
        >>> result = cluster(features, k_range=range(2, 6))
        >>> result.descriptions[["cluster", "size", "cluster_summary"]]
           cluster  size cluster_summary
        0        0    15     low valence
        1        1    15    high valence
        >>> result.tracks.head(3)
          spotify_id    title   artist  cluster cluster_summary
        0       id00  Track 0  Various        0     low valence
        1       id01  Track 1  Various        0     low valence
        2       id02  Track 2  Various        0     low valence
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown method {method!r}; supported methods are "
            f"{list(_SUPPORTED_METHODS)!r}."
        )

    X, index, _scaler, transform_log = prepare_matrix(features_df)
    if method == "gmm":
        raw = fit_gmm(X, k_range=k_range, random_state=random_state)
    elif method == "kmeans":
        raw = fit_kmeans(X, k_range=k_range, random_state=random_state)
    else:  # method == "hdbscan" — guarded above.
        mcs = (
            hdbscan_min_cluster_size
            if hdbscan_min_cluster_size is not None
            else max(5, len(X) // 50)  # plan §3 default
        )
        raw = fit_hdbscan(
            X, min_cluster_size=mcs, min_samples=hdbscan_min_samples
        )
    ordered = _canonical_order(raw, X)

    final_labels = _collapse_small_clusters(ordered.labels, min_playlist_size)
    warnings = _post_processing_warnings(
        final_labels, max_playlist_share, min_playlist_size
    )

    raw_descriptions = describe_clusters(ordered, X)
    summaries: dict[int, str] = dict(
        zip(
            raw_descriptions["cluster"].astype(int),
            raw_descriptions["summary"].astype(str),
        )
    )
    summaries[UNCLASSIFIED_LABEL] = _UNCLASSIFIED_SUMMARY

    tracks_frame = _build_tracks_frame(index, final_labels, summaries)
    descriptions_frame = _build_descriptions_frame(raw_descriptions, final_labels)
    diagnostics = _build_diagnostics(
        X, final_labels, raw_descriptions, random_state
    )

    return ClusterPipelineResult(
        tracks=tracks_frame,
        descriptions=descriptions_frame,
        clustering=ordered,
        transform_log=transform_log,
        diagnostics=diagnostics,
        warnings=warnings,
    )
