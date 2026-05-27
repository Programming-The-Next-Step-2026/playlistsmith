"""Streamlit GUI for playlistsmith.

A thin shell over the package: every action it takes corresponds to one
public call into :mod:`playlistsmith`. Launch with the ``playlistsmith-gui``
console script (installed by the ``gui`` optional dependency group):

.. code-block:: bash

    pip install -e ".[gui]"
    playlistsmith-gui              # live mode (real ReccoBeats)
    playlistsmith-gui --demo       # offline mode (mock transport)
"""

from playlistsmith.gui.cli import main

__all__ = ["main"]
