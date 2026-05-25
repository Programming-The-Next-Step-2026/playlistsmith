"""I/O subpackage: CSV loading and playlist export.

Exposes :class:`~playlistsmith.io.csv_loader.TrackLibrary` for reading a
track CSV (Exportify-compatible layout) and
:mod:`~playlistsmith.io.playlist_export` for writing one CSV per
cluster after the pipeline runs.
"""

from playlistsmith.io import playlist_export
from playlistsmith.io.csv_loader import TrackLibrary

__all__ = ["TrackLibrary", "playlist_export"]
