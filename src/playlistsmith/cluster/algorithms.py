"""Clustering algorithms for the playlistsmith pipeline.

The default clusterer is a Gaussian Mixture Model. K-Means and HDBSCAN are offered as
alternatives. All three are exposed in the GUI method selector.

Methods
-------
- **GMM** (:func:`fit_gmm`, default) — soft, probabilistic assignments;
  ``k`` is chosen by BIC with an ICL cross-check. Each track gets a
  posterior over clusters rather than a single hard label. Best for
  libraries with overlapping content.
- **K-Means** (:func:`fit_kmeans`) — hard partition into tight,
  roughly equal-sized blobs; ``k`` is chosen by silhouette–Calinski–
  Harabasz agreement. Faster than GMM and yields no posteriors. Good
  when clusters are well separated, and the small-library fallback when
  GMM covariance is under-determined.
- **HDBSCAN** (:func:`fit_hdbscan`) — density-based, takes no ``k``.
  Discovers the cluster count from the data and routes low-density
  outliers to noise (label ``-1``) instead of forcing them into a
  cluster. Best when some tracks genuinely shouldn't belong to any
  playlist.

References
----------
- Kass, R. E. & Raftery, A. E. (1995). *Bayes Factors*. Journal of
  the American Statistical Association, 90(430), 773–795. We use the
  Kass–Raftery "strong evidence" threshold of ``ΔBIC > 10`` when
  deciding whether a larger ``k`` is justified.
- Biernacki, C., Celeux, G. & Govaert, G. (2000). *Assessing a mixture
  model for clustering with the integrated completed likelihood*. IEEE
  TPAMI, 22(7), 719–725. We use their ICL criterion as a cross-check on
  BIC; it adds an entropy penalty so overlapping clusters score worse.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

__all__ = ["ClusteringResult", "fit_gmm", "fit_hdbscan", "fit_kmeans"]

#: Library size at and above which we trust ``covariance_type='full'``.
#: Below this, GMM full covariance is under-determined on nine features
_FULL_COVARIANCE_MIN_TRACKS: int = 50

#: Kass–Raftery (1995) strong-evidence threshold on ΔBIC: a larger ``k``
#: is only preferred if it improves BIC by more than this amount.
_DELTA_BIC_THRESHOLD: float = 10.0

#: Floor for posterior probabilities when computing ICL, to keep
#: ``log(p)`` finite at the boundary.
_POSTERIOR_LOG_FLOOR: float = 1e-12


@dataclass
class ClusteringResult:
    """Outcome of a clustering fit.

    Attributes:
        labels: Hard cluster assignment per input row, shape ``(n,)``.
        posteriors: Soft assignment matrix, shape ``(n, k)``; rows sum
            to 1. ``None`` for non-probabilistic clusterers (K-Means,
            HDBSCAN).
        k: The number of clusters in the final fit.
        silhouette: Silhouette score of the final labels (in ``[-1, 1]``).
        cohesion: Mean within-cluster squared distance to centroid in
            the transformed-scaled feature space.
        feature_means_per_cluster: ``(k × n_features)`` frame of cluster
            mean vectors in the *transformed-scaled* space; column names
            are the canonical feature column names.
        stability_ari: Adjusted Rand index between the chosen fit and a
            second fit with a different seed. A value much below 0.7
            indicates an unstable partition.
        random_state: The seed used for the primary fit.
        covariance_type: Model-variant marker. ``"full"`` / ``"diag"``
            for GMM (``"diag"`` indicates the small-library fallback
            fired); ``"kmeans"`` for K-Means; ``"hdbscan"`` for HDBSCAN.
        bic_curve: ``{k: BIC}`` over the swept range, lower is better
            (GMM only; empty otherwise).
        icl_curve: ``{k: ICL}`` over the swept range, lower is better
            (GMM only; empty otherwise).
        inertia_curve: ``{k: inertia}`` over the swept range (K-Means
            only). Inertia is non-increasing in ``k`` by construction;
            useful for elbow plotting.
        silhouette_curve: ``{k: silhouette}`` across the K-Means sweep,
            used (with ``calinski_harabasz_curve``) for selection.
        calinski_harabasz_curve: ``{k: CH-index}`` across the K-Means
            sweep.
        noise_rate: Share of input rows labelled ``-1`` (HDBSCAN noise).
            Zero for clusterers that assign every point.

    Examples:
        Produced by :func:`fit_gmm`, :func:`fit_kmeans` and
        :func:`fit_hdbscan` (see :func:`fit_gmm` for how ``X`` is built):

        >>> fit = fit_gmm(X, k_range=range(2, 6))
        >>> fit.k
        2
        >>> fit.labels.shape
        (30,)
        >>> fit.posteriors.shape  # (n_tracks, k); None for hard clusterers
        (30, 2)
        >>> fit.feature_means_per_cluster.shape  # (k, n_features)
        (2, 9)
    """

    labels: np.ndarray
    posteriors: np.ndarray | None
    k: int
    silhouette: float
    cohesion: float
    feature_means_per_cluster: pd.DataFrame = field(compare=False)
    stability_ari: float = 0.0
    random_state: int = 0
    covariance_type: str = ""
    bic_curve: dict[int, float] = field(default_factory=dict)
    icl_curve: dict[int, float] = field(default_factory=dict)
    inertia_curve: dict[int, float] = field(default_factory=dict)
    silhouette_curve: dict[int, float] = field(default_factory=dict)
    calinski_harabasz_curve: dict[int, float] = field(default_factory=dict)
    noise_rate: float = 0.0


def _icl(bic: float, posteriors: np.ndarray) -> float:
    """Integrated completed likelihood (ICL) of a fitted mixture.

    Following Biernacki, Celeux & Govaert (2000), ICL is BIC plus an
    entropy penalty on the posterior assignment matrix::

        ICL = BIC - 2 * Σᵢⱼ pᵢⱼ · log(pᵢⱼ)

    Each ``pᵢⱼ · log(pᵢⱼ)`` term is non-positive, so the subtraction
    contributes a non-negative entropy penalty: clusters that overlap
    (mushy posteriors) score higher (worse) than clusters that
    cleanly separate the data. Lower ICL is better, like lower BIC.

    Args:
        bic: BIC of the fitted model (lower is better).
        posteriors: Posterior matrix, shape ``(n, k)``, rows summing to 1.

    Returns:
        The ICL value.
    """
    clipped = np.clip(posteriors, _POSTERIOR_LOG_FLOOR, 1.0)
    entropy_term = float(np.sum(posteriors * np.log(clipped)))
    return bic - 2.0 * entropy_term


def _choose_k(
    bic_curve: dict[int, float], icl_curve: dict[int, float]
) -> int:
    """Pick the preferred ``k`` from the BIC and ICL curves.

    BIC is the primary selector. The candidate ``k*`` is the BIC
    minimum, walked down to the smallest ``k`` whose improvement over
    ``k - 1`` still exceeds the Kass–Raftery (1995) "strong evidence"
    threshold (``ΔBIC > 10``); otherwise the simpler model wins. ICL
    then acts as a cross-check (Biernacki et al. 2000): if it favours a
    different ``k`` *and* BIC does not have strong evidence for its own
    pick over the ICL pick (``ΔBIC ≤ 10`` between them), we defer to
    ICL, whose entropy penalty guards against spuriously overlapping
    clusters. When BIC strongly prefers its own pick, that pick stands.

    Args:
        bic_curve: ``{k: BIC}`` over the swept range.
        icl_curve: ``{k: ICL}`` over the swept range.

    Returns:
        The chosen ``k``.
    """
    ks_sorted = sorted(bic_curve)
    bic_best = min(ks_sorted, key=lambda k: bic_curve[k])

    # Walk down from the BIC argmin to the smallest k that still
    # beats k-1 by more than the threshold.
    bic_choice = bic_best
    while bic_choice - 1 in bic_curve:
        smaller = bic_choice - 1
        if bic_curve[smaller] - bic_curve[bic_choice] <= _DELTA_BIC_THRESHOLD:
            bic_choice = smaller
        else:
            break

    # ICL cross-check: override the BIC pick only when BIC's evidence
    # for it over the ICL pick is not "strong" (ΔBIC ≤ 10).
    icl_choice = min(ks_sorted, key=lambda k: icl_curve[k])
    if icl_choice != bic_choice:
        bic_gap = bic_curve[icl_choice] - bic_curve[bic_choice]
        if bic_gap <= _DELTA_BIC_THRESHOLD:
            return icl_choice
    return bic_choice


def _cohesion(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean within-cluster squared distance to the cluster centroid.

    Args:
        X: The transformed-scaled feature matrix, shape ``(n, d)``.
        labels: Hard cluster assignment per row, shape ``(n,)``.

    Returns:
        The mean squared distance from each point to its cluster's
        centroid, averaged over all points. Zero indicates perfect
        cohesion; larger values mean more diffuse clusters.
    """
    total = 0.0
    for label in np.unique(labels):
        members = X[labels == label]
        centroid = members.mean(axis=0)
        total += float(np.sum((members - centroid) ** 2))
    return total / len(X)


def fit_gmm(
    X: pd.DataFrame,
    k_range: Iterable[int] = range(2, 13),
    random_state: int = 0,
) -> ClusteringResult:
    """Fit a Gaussian Mixture Model and select ``k`` by BIC/ICL.

    Sweeps ``GaussianMixture`` over ``k_range`` with ``n_init=5`` and a
    fixed ``random_state``, records BIC and ICL at each ``k``, picks
    ``k`` per :func:`_choose_k` (Kass–Raftery ΔBIC > 10 with an ICL
    cross-check), refits the chosen model, and packages everything into
    a :class:`ClusteringResult`.

    Small-library fallback: if ``X`` has fewer than
    :data:`_FULL_COVARIANCE_MIN_TRACKS` rows, ``covariance_type`` drops
    from ``"full"`` to ``"diag"`` because full covariance on nine
    features is under-determined.

    Args:
        X: Transformed-and-scaled modelling matrix from
            :func:`~playlistsmith.cluster.preprocess.prepare_matrix`.
            Rows are tracks, columns are the nine ReccoBeats features.
        k_range: Candidate values of ``k`` to sweep. All values must be ``>= 2``.
        random_state: Seed threaded through ``GaussianMixture`` for reproducibility.

    Returns:
        A :class:`ClusteringResult` with hard labels, posteriors, the
        BIC/ICL curves, quality metrics, the per-cluster mean vectors
        in the transformed-scaled space, a two-seed stability ARI, the
        seed used, and the ``covariance_type`` actually used.

    Raises:
        ValueError: If ``X`` is empty or ``k_range`` contains values
            below 2.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from playlistsmith.cluster import fit_gmm, prepare_matrix
        >>> # Two obvious groups, then z-score them into a modelling matrix.
        >>> cols = ["acousticness", "danceability", "energy", "instrumentalness",
        ...         "liveness", "loudness", "speechiness", "tempo", "valence"]
        >>> sd = [0.02, 0.02, 0.02, 0.02, 0.02, 1.0, 0.02, 3.0, 0.02]
        >>> mellow = [0.88, 0.30, 0.20, 0.40, 0.15, -17.0, 0.05, 85.0, 0.25]
        >>> upbeat = [0.12, 0.80, 0.90, 0.40, 0.15, -6.0, 0.05, 128.0, 0.75]
        >>> rng = np.random.default_rng(5)
        >>> features = pd.DataFrame(
        ...     np.vstack([rng.normal(mellow, sd, (15, 9)),
        ...                rng.normal(upbeat, sd, (15, 9))]), columns=cols)
        >>> features.insert(0, "artist", "Various")
        >>> features.insert(0, "title", [f"Track {i}" for i in range(30)])
        >>> features.insert(0, "spotify_id", [f"id{i:02d}" for i in range(30)])
        >>> X, index, scaler, log = prepare_matrix(features)
        >>> fit = fit_gmm(X, k_range=range(2, 6))
        >>> fit.k  # BIC/ICL select two clusters
        2
        >>> fit.covariance_type  # 'diag' fallback below 50 tracks
        'diag'
        >>> round(fit.silhouette, 3)
        0.567
        >>> {k: round(v, 1) for k, v in fit.bic_curve.items()}
        {2: 87.6, 3: 108.3, 4: 127.6, 5: 146.6}
    """
    if X.empty:
        raise ValueError("Cannot fit a GMM on an empty matrix.")

    k_values = sorted(set(int(k) for k in k_range))
    if not k_values or min(k_values) < 2:
        raise ValueError(
            f"k_range must contain integers >= 2 (got {k_values!r})."
        )

    X_arr = X.to_numpy()
    n = len(X_arr)
    if max(k_values) > n:
        raise ValueError(
            f"k_range max ({max(k_values)}) exceeds the {n} track(s) "
            "available after preprocessing. Pass a smaller k_range "
            "(e.g. range(2, n + 1)) or add more tracks to the library."
        )
    covariance_type = "full" if n >= _FULL_COVARIANCE_MIN_TRACKS else "diag"

    bic_curve: dict[int, float] = {}
    icl_curve: dict[int, float] = {}
    fits: dict[int, GaussianMixture] = {}
    for k in k_values:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            n_init=5,
            random_state=random_state,
        )
        gmm.fit(X_arr)
        bic = float(gmm.bic(X_arr))
        bic_curve[k] = bic
        icl_curve[k] = _icl(bic, gmm.predict_proba(X_arr))
        fits[k] = gmm

    chosen_k = _choose_k(bic_curve, icl_curve)
    chosen = fits[chosen_k]
    posteriors = chosen.predict_proba(X_arr)
    labels = np.argmax(posteriors, axis=1)

    # Quality metrics.
    sil = float(silhouette_score(X_arr, labels)) if chosen_k > 1 else 0.0
    cohesion = _cohesion(X_arr, labels)

    # Per-cluster mean vectors in the modelling space.
    means = pd.DataFrame(
        np.array([X_arr[labels == c].mean(axis=0) for c in range(chosen_k)]),
        columns=list(X.columns),
    )

    # Stability: fit again with a different seed and compute ARI.
    stability_seed = random_state + 1
    stability_gmm = GaussianMixture(
        n_components=chosen_k,
        covariance_type=covariance_type,
        n_init=5,
        random_state=stability_seed,
    )
    stability_labels = stability_gmm.fit(X_arr).predict(X_arr)
    stability_ari = float(adjusted_rand_score(labels, stability_labels))

    return ClusteringResult(
        labels=labels,
        posteriors=posteriors,
        k=chosen_k,
        bic_curve=bic_curve,
        icl_curve=icl_curve,
        silhouette=sil,
        cohesion=cohesion,
        feature_means_per_cluster=means,
        stability_ari=stability_ari,
        random_state=random_state,
        covariance_type=covariance_type,
    )


def _kmeans_choose_k(
    silhouette_curve: dict[int, float],
    ch_curve: dict[int, float],
) -> int:
    """Pick ``k`` by silhouette-Calinski–Harabasz agreement.

    Silhouette systematically prefers low ``k`` and
    well-separated convex blobs, which under-segments real audio
    libraries. The Calinski–Harabasz index pulls in the other direction.
    We score each ``k`` by the sum of its ranks under both metrics
    (rank 1 = best per metric), and pick the smallest sum. Ties break
    toward larger ``k`` so the playlist count stays closer to user
    expectations.

    Args:
        silhouette_curve: ``{k: silhouette}`` across the sweep.
        ch_curve: ``{k: CH-index}`` across the sweep.

    Returns:
        The chosen ``k``.
    """
    ks = sorted(silhouette_curve)
    sil_rank = {
        k: r
        for r, k in enumerate(sorted(ks, key=lambda x: -silhouette_curve[x]))
    }
    ch_rank = {
        k: r for r, k in enumerate(sorted(ks, key=lambda x: -ch_curve[x]))
    }
    return min(ks, key=lambda k: (sil_rank[k] + ch_rank[k], -k))


def fit_kmeans(
    X: pd.DataFrame,
    k_range: Iterable[int] = range(2, 13),
    random_state: int = 0,
) -> ClusteringResult:
    """Fit K-Means and select ``k`` by silhouette-CH agreement.

    K-Means is the cheap, hard-assignment alternative to GMM.
    It is useful as a sanity check and as the
    small-library fallback below ~25 tracks where GMM covariance is
    too under-determined to be trusted.

    Selection works as follows: for each ``k`` we compute
    inertia, silhouette and Calinski–Harabasz; ``k`` is picked by
    sum-of-ranks across silhouette and CH (see :func:`_kmeans_choose_k`).

    Args:
        X: Transformed-and-scaled modelling matrix from
            :func:`~playlistsmith.cluster.preprocess.prepare_matrix`.
        k_range: Candidate values of ``k`` to sweep. All values must be
            ``>= 2``.
        random_state: Seed threaded through ``KMeans`` for
            reproducibility.

    Returns:
        A :class:`ClusteringResult` with hard labels, ``posteriors=None``,
        the inertia / silhouette / CH curves, the chosen-``k``
        silhouette and cohesion, a two-seed stability ARI, and
        ``covariance_type="kmeans"``.

    Raises:
        ValueError: If ``X`` is empty or ``k_range`` contains values
            below 2.

    Examples:
        Using the modelling matrix ``X`` from :func:`prepare_matrix`
        (built as in the :func:`fit_gmm` example):

        >>> from playlistsmith.cluster import fit_kmeans
        >>> fit = fit_kmeans(X, k_range=range(2, 6))
        >>> fit.k
        2
        >>> fit.posteriors is None  # K-Means is a hard clusterer
        True
        >>> round(fit.silhouette, 3)
        0.567
        >>> fit.covariance_type
        'kmeans'
    """
    if X.empty:
        raise ValueError("Cannot fit K-Means on an empty matrix.")

    k_values = sorted(set(int(k) for k in k_range))
    if not k_values or min(k_values) < 2:
        raise ValueError(
            f"k_range must contain integers >= 2 (got {k_values!r})."
        )

    X_arr = X.to_numpy()
    n = len(X_arr)
    if max(k_values) > n:
        raise ValueError(
            f"k_range max ({max(k_values)}) exceeds the {n} track(s) "
            "available after preprocessing. Pass a smaller k_range "
            "(e.g. range(2, n + 1)) or add more tracks to the library."
        )

    inertia_curve: dict[int, float] = {}
    silhouette_curve: dict[int, float] = {}
    ch_curve: dict[int, float] = {}
    fits: dict[int, KMeans] = {}
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X_arr)
        labels_k = km.labels_
        inertia_curve[k] = float(km.inertia_)
        silhouette_curve[k] = float(silhouette_score(X_arr, labels_k))
        ch_curve[k] = float(calinski_harabasz_score(X_arr, labels_k))
        fits[k] = km

    chosen_k = _kmeans_choose_k(silhouette_curve, ch_curve)
    chosen = fits[chosen_k]
    labels = chosen.labels_
    sil = silhouette_curve[chosen_k]
    cohesion = _cohesion(X_arr, labels)

    means = pd.DataFrame(
        np.array([X_arr[labels == c].mean(axis=0) for c in range(chosen_k)]),
        columns=list(X.columns),
    )

    stability_km = KMeans(
        n_clusters=chosen_k, n_init=10, random_state=random_state + 1
    )
    stability_labels = stability_km.fit(X_arr).labels_
    stability_ari = float(adjusted_rand_score(labels, stability_labels))

    return ClusteringResult(
        labels=labels,
        posteriors=None,
        k=chosen_k,
        silhouette=sil,
        cohesion=cohesion,
        feature_means_per_cluster=means,
        stability_ari=stability_ari,
        random_state=random_state,
        covariance_type="kmeans",
        inertia_curve=inertia_curve,
        silhouette_curve=silhouette_curve,
        calinski_harabasz_curve=ch_curve,
    )


def fit_hdbscan(
    X: pd.DataFrame,
    min_cluster_size: int,
    min_samples: int | None = None,
) -> ClusteringResult:
    """Fit HDBSCAN, surfacing low-density points as noise.

    HDBSCAN discovers the cluster count from the data and is
    honest about songs that genuinely do not fit any playlist —
    low-density points are labelled ``-1`` rather than being forced
    into a cluster. The result's ``noise_rate`` reports the share of
    such points so the coverage report can surface them.

    Args:
        X: Transformed-and-scaled modelling matrix from
            :func:`~playlistsmith.cluster.preprocess.prepare_matrix`.
        min_cluster_size: The smallest cluster HDBSCAN is allowed to
            return. The default is ``max(5, n_tracks // 50)``; the
            public :func:`~playlistsmith.cluster.cluster` entry point
            applies that default.
        min_samples: Noise sensitivity (lower → fewer noise points).
            Defaults to ``min_cluster_size``.

    Returns:
        A :class:`ClusteringResult` with hard labels (``-1`` for noise),
        ``posteriors=None``, ``covariance_type="hdbscan"``, the
        ``noise_rate``, a chosen-fit silhouette computed over non-noise
        points only, and per-cluster mean vectors for the discovered
        non-noise clusters.

    Raises:
        ValueError: If ``X`` is empty.

    Examples:
        Using the modelling matrix ``X`` from :func:`prepare_matrix`
        (built as in the :func:`fit_gmm` example):

        >>> from playlistsmith.cluster import fit_hdbscan
        >>> fit = fit_hdbscan(X, min_cluster_size=5)
        >>> fit.k
        2
        >>> round(fit.noise_rate, 3)  # share of tracks labelled -1
        0.0
        >>> sorted(set(fit.labels.tolist()))
        [0, 1]
        >>> fit.covariance_type
        'hdbscan'
    """
    if X.empty:
        raise ValueError("Cannot fit HDBSCAN on an empty matrix.")

    effective_min_samples = (
        min_samples if min_samples is not None else min_cluster_size
    )
    # ``copy=True`` is sklearn's announced default from 1.10; setting it
    # explicitly silences the FutureWarning on 1.8/1.9.
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=effective_min_samples,
        copy=True,
    )
    X_arr = X.to_numpy()
    labels = model.fit_predict(X_arr)
    non_noise_mask = labels >= 0
    n_clusters = int(labels.max() + 1) if non_noise_mask.any() else 0
    noise_rate = float((~non_noise_mask).mean())

    # Silhouette is only meaningful over the non-noise subset, and
    # requires at least two distinct labels.
    if non_noise_mask.sum() > 1 and n_clusters >= 2:
        sil = float(
            silhouette_score(X_arr[non_noise_mask], labels[non_noise_mask])
        )
    else:
        sil = 0.0

    if non_noise_mask.any():
        cohesion = _cohesion(X_arr[non_noise_mask], labels[non_noise_mask])
    else:
        cohesion = 0.0

    means = pd.DataFrame(
        np.array(
            [X_arr[labels == c].mean(axis=0) for c in range(n_clusters)]
        ).reshape(n_clusters, X_arr.shape[1]),
        columns=list(X.columns),
    )

    # HDBSCAN is deterministic, so a same-params re-fit always agrees;
    # stability_ari is not really informative here. We surface 1.0 with
    # this caveat documented in the dataclass.
    return ClusteringResult(
        labels=labels,
        posteriors=None,
        k=n_clusters,
        silhouette=sil,
        cohesion=cohesion,
        feature_means_per_cluster=means,
        stability_ari=1.0,
        random_state=0,
        covariance_type="hdbscan",
        noise_rate=noise_rate,
    )
