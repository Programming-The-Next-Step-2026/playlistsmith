"""Session-state helpers for the Streamlit GUI.

Centralises the keys used in :data:`streamlit.session_state` so widgets
can read/write without typo'd string keys, and so "Reset session" knows
exactly what to clear.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Marker the upload widget uses to detect whether the currently-attached
#: file is one it has already loaded. Lives here (not in widgets/upload.py)
#: so :func:`reset_pipeline_state` can wipe it from a central place.
UPLOAD_IDENTITY_KEY = "_upload_identity"

#: Nonce folded into the file_uploader's widget key. Bumping it gives the
#: uploader a fresh key, so Streamlit re-creates it empty and forgets the
#: previously-attached file. Without this, "Reset session" clears the
#: library state but the still-attached file silently re-loads on rerun.
UPLOAD_NONCE_KEY = "_upload_nonce"


@dataclass(frozen=True)
class Keys:
    """The session-state slots used by the GUI."""

    library: str = "library"
    library_filename: str = "library_filename"
    features_df: str = "features_df"
    coverage: str = "coverage"
    cluster_result: str = "cluster_result"
    export_paths: str = "export_paths"
    demo_mode: str = "demo_mode"


KEYS = Keys()


def reset_pipeline_state(  # type: ignore[no-untyped-def]
    session_state, *, clear_upload: bool = False
) -> None:
    """Clear every pipeline slot in ``session_state`` except demo mode.

    Used by the "Reset session" button and whenever an upstream stage
    (e.g. a new CSV upload) invalidates the downstream ones. Also clears
    the upload-identity marker so a still-attached file in the uploader
    triggers a fresh load on the next rerun.

    Args:
        session_state: The :data:`streamlit.session_state` object.
        clear_upload: When ``True``, also detach any file currently held
            by the ``file_uploader`` widget by bumping
            :data:`UPLOAD_NONCE_KEY`. The "Reset session" button sets
            this; the new-upload path leaves it ``False`` so it does not
            wipe the file the user just attached.
    """
    for key in (
        KEYS.library,
        KEYS.library_filename,
        KEYS.features_df,
        KEYS.coverage,
        KEYS.cluster_result,
        KEYS.export_paths,
        UPLOAD_IDENTITY_KEY,
    ):
        if key in session_state:
            del session_state[key]
    if clear_upload:
        session_state[UPLOAD_NONCE_KEY] = (
            session_state.get(UPLOAD_NONCE_KEY, 0) + 1
        )
