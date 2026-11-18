"""Project roots, resolved from this file's location — never from the CWD.

Scripts and notebooks in this repo are launched from many different working
directories (repo root, ``notebooks/``, ``studies/*/``, an IDE's default, a
temp dir under pytest). Anything built on ``Path.cwd()`` or a bare relative
literal therefore resolves differently depending on where it was started, so
the CWD is deliberately not consulted here: ``PROJECT_ROOT`` is derived from
``__file__``, which is fixed by where the package is installed.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "cache"
CONFIG_DIR = PROJECT_ROOT / "config"


def cache_dir(*parts: str) -> Path:
    """Directory under CACHE_DIR, created if missing."""
    p = CACHE_DIR.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
