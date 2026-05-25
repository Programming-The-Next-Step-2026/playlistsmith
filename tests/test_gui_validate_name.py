"""Unit tests for :func:`playlistsmith.io.playlist_export.validate_name`.

This is the filesystem-safety gate the export widget relies on. Keeping
the rules unit-tested means the GUI cannot accidentally drift away from
what the writer actually permits.
"""

from __future__ import annotations

import pytest

from playlistsmith.io.playlist_export import validate_name


@pytest.mark.parametrize(
    "name",
    [
        "Workout",
        "Sunday Morning",
        "Cluster 0",
        "Mix 2026-05",
        "after-hours (focus)",
        "café",  # non-ASCII letters are fine
    ],
)
def test_valid_names(name: str) -> None:
    assert validate_name(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
        "\t",
        "../escape",
        "with/slash",
        "with\\back",
        "colon:here",
        "star*",
        "q?",
        'quote"',
        "lt<",
        "gt>",
        "pipe|",
        ".leading-dot",
        "trailing-dot.",
    ],
)
def test_invalid_names(name: str) -> None:
    assert validate_name(name) is False


def test_non_string_input_is_invalid() -> None:
    assert validate_name(123) is False  # type: ignore[arg-type]
    assert validate_name(None) is False  # type: ignore[arg-type]
