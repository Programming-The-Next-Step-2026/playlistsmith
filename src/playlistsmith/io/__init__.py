"""I/O subpackage: CSV loading and playlist export.

Exposes :class:`~playlistsmith.io.csv_loader.TrackLibrary` for reading a
track CSV (Exportify-compatible layout) into a tidy DataFrame.
"""

from playlistsmith.io.csv_loader import TrackLibrary

__all__ = ["TrackLibrary"]
