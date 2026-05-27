"""Streamlit widget modules for the playlistsmith GUI.

Each module owns one stage of the vertical-flow page (upload, extract,
cluster, viz, export) and exposes a single ``render(...)`` function the
top-level :mod:`playlistsmith.gui.app` script calls in order.
"""
