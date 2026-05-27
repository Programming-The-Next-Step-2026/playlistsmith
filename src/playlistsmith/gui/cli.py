"""Entry point for the ``playlistsmith-gui`` console script.

Parses ``--demo`` (which installs the offline ReccoBeats mock transport
before any pipeline code runs) and then hands off to Streamlit's
in-process launcher pointed at :mod:`playlistsmith.gui.app`.
"""

from __future__ import annotations

import argparse
import importlib.resources
import os
import sys


_DEMO_ENV_VAR = "PLAYLISTSMITH_GUI_DEMO"


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args, install the demo mock if requested, run Streamlit.

    Args:
        argv: Optional argument list (defaults to :data:`sys.argv` minus
            the program name). The first ``--demo`` flag is consumed
            here; every other argument is forwarded to Streamlit.
    """
    parser = argparse.ArgumentParser(
        prog="playlistsmith-gui",
        description="Launch the playlistsmith Streamlit GUI.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run offline against the recorded ReccoBeats fixture. "
            "No network calls are made."
        ),
    )
    args, streamlit_argv = parser.parse_known_args(argv)

    if args.demo:
        # Set the env var so the Streamlit-spawned subprocess (which
        # re-imports this package) can detect demo mode and re-install
        # the mock transport in its own process.
        os.environ[_DEMO_ENV_VAR] = "1"
        from playlistsmith.gui import fixtures

        fixtures.install_mock_transport()

    app_path = importlib.resources.files("playlistsmith.gui") / "app.py"

    # Lazy import: streamlit is an optional dependency.
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(app_path), *streamlit_argv]
    sys.exit(stcli.main())


def is_demo_mode() -> bool:
    """Return whether demo mode was requested on the CLI."""
    return os.environ.get(_DEMO_ENV_VAR) == "1"
