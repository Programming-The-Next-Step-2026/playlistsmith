"""Unit tests for the GUI session-state helpers.

``st.session_state`` behaves like a plain mutable mapping for the slots
:func:`reset_pipeline_state` touches, so a dict stands in for it here.
"""

from __future__ import annotations

from playlistsmith.gui.state import (
    KEYS,
    UPLOAD_IDENTITY_KEY,
    UPLOAD_NONCE_KEY,
    reset_pipeline_state,
)


def _populated_state() -> dict:
    """A session-state dict with every pipeline slot filled in."""
    return {
        KEYS.library: object(),
        KEYS.library_filename: "tracks.csv",
        KEYS.features_df: object(),
        KEYS.coverage: object(),
        KEYS.cluster_result: object(),
        KEYS.export_paths: ["a.csv"],
        KEYS.demo_mode: True,
        UPLOAD_IDENTITY_KEY: "upload:abc",
    }


def test_reset_clears_pipeline_slots_but_keeps_demo_mode() -> None:
    state = _populated_state()
    reset_pipeline_state(state)
    for key in (
        KEYS.library,
        KEYS.library_filename,
        KEYS.features_df,
        KEYS.coverage,
        KEYS.cluster_result,
        KEYS.export_paths,
        UPLOAD_IDENTITY_KEY,
    ):
        assert key not in state
    # Demo mode survives a reset.
    assert state[KEYS.demo_mode] is True


def test_reset_without_clear_upload_leaves_nonce_untouched() -> None:
    # The new-upload path must not bump the nonce, or it would detach the
    # file the user just attached.
    state = _populated_state()
    reset_pipeline_state(state)
    assert UPLOAD_NONCE_KEY not in state


def test_clear_upload_bumps_nonce_from_zero() -> None:
    state = _populated_state()
    reset_pipeline_state(state, clear_upload=True)
    assert state[UPLOAD_NONCE_KEY] == 1


def test_clear_upload_increments_existing_nonce() -> None:
    state = _populated_state()
    state[UPLOAD_NONCE_KEY] = 3
    reset_pipeline_state(state, clear_upload=True)
    assert state[UPLOAD_NONCE_KEY] == 4
