"""Resolve the application root directory in both dev and frozen (Nuitka) runs.

In a normal source checkout this is the repo root (parent of this file's
package). In a Nuitka ``--standalone`` build the compiled module's ``__file__``
points *inside* ``app.dist/`` (the onedir), so ``Path(__file__).parents[N]`` no
longer reaches the deliverable root. For a frozen app the deliverable root is
the directory containing ``sys.executable`` (the exe sits at the onedir root).

Use ``app_root()`` everywhere the code needs to locate external assets
(``./models``, ``./data/avatars``, ``./web``, ``.env``, ``launch_config.json``
etc.) so the same path resolves correctly in both modes. The supervisor always
``Set-Location``s to the deliverable root too, so relative ``./`` paths and
``app_root()`` agree.
"""
import sys
from pathlib import Path


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]