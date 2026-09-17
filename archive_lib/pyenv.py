"""Re-exec fallback for scripts that need numpy/sentence_transformers.

The plain `python3` on PATH resolves to whatever the shell picks up (Homebrew's
arm64 interpreter, externally-managed, pip install blocked by PEP 668) and has
no reason to have these packages installed — they're pip-installed only into
this one pinned venv (see archive-lib/requirements.txt). If the running
process lacks the module, re-exec under this one instead of crashing — the
caller shouldn't have to remember a PYTHON= prefix or which interpreter has
the packages.

Native arm64 venv (~/.venvs/archive-semantic-arm64, python3.12) — migrated off
the x86_64/Rosetta python@3.10 interpreter on 2026-09-13 once numpy==1.26.4/
sentence-transformers==2.7.0 confirmed to have arm64 wheels. Keep this venv
separate from requirements-diarization.txt's (numpy==2.4.6) — mixing the two
numpy versions in one env previously broke semantic search project-wide.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SEMANTIC_PYTHON = str(Path.home() / ".venvs/archive-semantic-arm64/bin/python3")


def ensure_semantic_capable() -> None:
    """If numpy/sentence_transformers aren't importable under sys.executable,
    re-exec this process under SEMANTIC_PYTHON. No-op if already capable, or
    if the fallback interpreter is unavailable/is the one already running (in
    which case the ModuleNotFoundError surfaces normally from the caller)."""
    try:
        import numpy  # noqa: F401
        import sentence_transformers  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    fallback = Path(SEMANTIC_PYTHON)
    if not fallback.exists():
        return
    current = Path(sys.executable)
    if current.exists() and current.samefile(fallback):
        return  # already running under it — the ModuleNotFoundError is real
    os.execv(str(fallback), [str(fallback)] + sys.argv)
