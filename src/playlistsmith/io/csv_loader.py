"""Load a track CSV into a tidy DataFrame.

The expected CSV layout is the one produced by Exportify
(https://exportify.app): a semicolon-delimited export of a Spotify playlist.
This project is not affiliated with or endorsed by Exportify or Spotify; the
name is used only to describe the compatible file format.

Only three fields matter to the playlistsmith pipeline: the track title, the
artist name(s), and the bare Spotify track ID (parsed out of the
``spotify:track:<id>`` URI). Everything else in the export is ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from playlistsmith.features import CoverageReport

# Source column headers as written by Exportify (semicolon-delimited CSV).
_SRC_TRACK_NAME = "Track Name"
_SRC_ARTIST_NAME = "Artist Name(s)"
_SRC_TRACK_URI = "Track URI"

# Clean column names used throughout the pipeline.
TITLE = "title"
ARTIST = "artist"
SPOTIFY_ID = "spotify_id"

_SPOTIFY_TRACK_URI_PREFIX = "spotify:track:"


class TrackLibrary:
    """A music library loaded from a track CSV.

    Accepts a CSV in the layout produced by Exportify (exportify.app): a
    semicolon-delimited Spotify playlist export. The parsed library is held
    as a pandas DataFrame with exactly three columns: ``title``, ``artist``
    and ``spotify_id``. The ``artist`` column keeps the raw, comma-separated
    artist string (e.g. ``"Sample Collective, Dummy Vox"``) so no naming
    information is lost; splitting is left to later pipeline stages if needed.

    Attributes:
        source_path: Path to the CSV file this library was loaded from.

    Examples:
        >>> import playlistsmith as ps
        >>> tlib = ps.TrackLibrary("./tests/example_tracklist.csv")
        >>> print(tlib)
        TrackLibrary(source_path='tests/example_tracklist.csv', tracks=3)

        >>> tlib.display()
                       title                        artist               spotify_id
        0  Synthetic Sunrise              The Placeholders  test0000000000000000001
        1        Mock Anthem  Sample Collective, Dummy Vox  test0000000000000000002
        2  Placeholder Pulse                   Test Signal  test0000000000000000003
    """

    def __init__(self, csv_path: str | Path) -> None:
        """Load and parse an Exportify-format track CSV.

        Args:
            csv_path: Path to the Exportify CSV export.

        Raises:
            FileNotFoundError: If ``csv_path`` does not exist.
            ValueError: If the file is missing a required column or contains
                no track rows with a Spotify track ID.
        """
        self.source_path = Path(csv_path)
        self._tracks = self._load(self.source_path)

    @staticmethod
    def _load(path: Path) -> pd.DataFrame:
        """Read the CSV and return a tidy ``(title, artist, spotify_id)`` frame.

        Args:
            path: Path to the Exportify CSV export.

        Returns:
            A DataFrame with columns ``title``, ``artist`` and ``spotify_id``,
            with rows lacking a Spotify track ID dropped.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If a required column is missing, or no rows remain
                after dropping entries without a Spotify track ID.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Exportify CSV not found: {path}")

        raw = pd.read_csv(path, sep=";", dtype=str)

        required = [_SRC_TRACK_NAME, _SRC_ARTIST_NAME, _SRC_TRACK_URI]
        missing = [col for col in required if col not in raw.columns]
        if missing:
            raise ValueError(
                f"CSV is missing required column(s): {', '.join(missing)}. "
                "Is this an Exportify export? Expected a ';'-delimited file."
            )

        tracks = pd.DataFrame(
            {
                TITLE: raw[_SRC_TRACK_NAME].str.strip(),
                ARTIST: raw[_SRC_ARTIST_NAME].str.strip(),
                SPOTIFY_ID: (
                    raw[_SRC_TRACK_URI]
                    .str.strip()
                    .str.removeprefix(_SPOTIFY_TRACK_URI_PREFIX)
                ),
            }
        )

        has_id = tracks[SPOTIFY_ID].notna() & (tracks[SPOTIFY_ID] != "")
        tracks = tracks.loc[has_id].reset_index(drop=True)

        if tracks.empty:
            raise ValueError(
                f"No tracks with a Spotify track ID found in {path}."
            )

        return tracks

    @property
    def dataframe(self) -> pd.DataFrame:
        """The parsed library as a DataFrame.

        Returns:
            A defensive copy with columns ``title``, ``artist`` and
            ``spotify_id``. Mutating it does not affect this library.
        """
        return self._tracks.copy()

    # TODO see if it might be nicer to just return head or tail
    def display(self, max_rows: int | None = 20) -> None:
        """Pretty-print the library to stdout.

        This is a presentation helper only; it returns nothing so it never
        double-renders in a REPL or notebook. Use the :attr:`dataframe`
        property when you need the data as an object.

        Args:
            max_rows: Maximum number of rows to print. When the library has
                more rows than this, the middle is elided with ``...``.
                ``None`` prints every row.
        """
        print(self._tracks.to_string(max_rows=max_rows))

    def extract_features(self, mode: str = "precomputed") -> tuple[pd.DataFrame, CoverageReport]:
        """Compute audio features for the tracks in this library.

        Delegates to the :func:`playlistsmith.features.extract` entry point
        (the single supported way to obtain features) in ``precomputed``
        mode rather than calling internal feature modules directly. Tracks
        with no precomputed features are dropped and reported.
        
        Args:
            mode: The feature extraction mode to use. See
                :func:`playlistsmith.features.extract` for supported modes.
        
        Returns:
            A ``(features_df, coverage)`` tuple: a feature DataFrame with
            one row per resolved track, and a
            :class:`~playlistsmith.features.CoverageReport` describing what
            was resolved and what was dropped.
        
        Examples:
            >>> import playlistsmith as ps
            >>> tlib = ps.TrackLibrary("./tests/example_tracklist.csv")
            >>> features, coverage = tlib.extract_features(mode="precomputed")
            >>> print(coverage.dropped_tracks) # all dropped because of synthetic IDs
                            spotify_id              title                        artist
            0  test0000000000000000001  Synthetic Sunrise              The Placeholders
            1  test0000000000000000002        Mock Anthem  Sample Collective, Dummy Vox
            2  test0000000000000000003  Placeholder Pulse                   Test Signal
        """
        # Imported lazily to avoid an import cycle (the features package
        # reads this module's column names).
        import playlistsmith.features as features

        return features.extract(self.dataframe, mode=mode)

    def __len__(self) -> int:
        """Number of tracks in the library."""
        return len(self._tracks)

    def __repr__(self) -> str:
        """Concise representation including source path and track count."""
        return (
            f"{type(self).__name__}(source_path={str(self.source_path)!r}, "
            f"tracks={len(self)})"
        )
