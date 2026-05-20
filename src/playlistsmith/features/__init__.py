"""Feature extraction subpackage.

This package exposes exactly one public entry point, :func:`extract`,
which is the only supported way to obtain audio features for a track
library. Internal modules (currently :mod:`.reccobeats`) are
implementation details and must not be called from outside this package.

The pipeline contract is::

    (features_df, coverage) = extract(tracks, mode="precomputed")

where ``tracks`` is a TrackLibrary-shaped frame (``title``, ``artist``,
``spotify_id``), ``features_df`` carries those identity columns plus the
audio-feature columns for every *resolved* track, and ``coverage`` is a
:class:`CoverageReport` describing what was kept and what was dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

__all__ = ["extract", "CoverageReport"]


@dataclass
class CoverageReport:
    """Summary of how many tracks were resolved to features.

    The pipeline always emits one of these so the user can see how much
    of their library survived feature extraction and exactly which tracks
    were dropped (e.g. because no precomputed features were available).

    Attributes:
        total: Number of tracks handed to :func:`extract`.
        resolved: Number of tracks for which features were obtained.
        dropped_tracks: A frame of the dropped tracks (at least the
            ``spotify_id``, ``title`` and ``artist`` columns), so callers
            can report or inspect them. Excluded from equality/repr.
    """

    total: int
    resolved: int
    dropped_tracks: pd.DataFrame = field(compare=False, repr=False)

    @property
    def dropped(self) -> int:
        """Number of tracks dropped during feature extraction."""
        return len(self.dropped_tracks)

    def render(self) -> str:
        """Return a one-line human-readable coverage summary.

        Returns:
            A summary such as
            ``"Feature coverage: 2/3 track(s) resolved via ReccoBeats; 1
            dropped."``.
        """
        return (
            f"Feature coverage: {self.resolved}/{self.total} track(s) "
            f"resolved via ReccoBeats; {self.dropped} dropped."
        )

    def __str__(self) -> str:
        """Human-readable coverage summary (see :meth:`render`)."""
        return self.render()


def extract(
    tracks: pd.DataFrame, mode: str
) -> tuple[pd.DataFrame, CoverageReport]:
    """Extract audio features for ``tracks`` using the given ``mode``.

    This is the single supported entry point for feature extraction.
    ``mode="precomputed"`` performs a lookup-only resolution against
    ReccoBeats (Spotify-ID based); tracks without precomputed features
    are dropped and reported in the returned :class:`CoverageReport`.

    Args:
        tracks: A TrackLibrary-shaped frame with ``title``, ``artist``
            and ``spotify_id`` columns.
        mode: The extraction strategy. Only ``"precomputed"`` is
            currently supported.

    Returns:
        A ``(features_df, coverage)`` tuple. ``features_df`` has the
        identity columns plus one column per audio feature, one row per
        resolved track. ``coverage`` summarises resolved/dropped counts.

    Raises:
        ValueError: If ``mode`` is not a supported extraction mode.
    """
    if mode == "precomputed":
        # Imported lazily so the heavy/HTTP module is only loaded when
        # actually used, and to keep the package import cycle-free.
        from playlistsmith.features import reccobeats

        return reccobeats.extract_precomputed(tracks)

    raise ValueError(
        f"Unsupported feature extraction mode: {mode!r}. "
        "Supported modes: 'precomputed'."
    )
