"""
Static registry of all known archive projects.

This is intentionally the ONE place that knows about all sibling
projects — no project's own config.py lists its siblings, which would mean
adding a new project requires editing N existing configs.

The actual name -> transcripts.db path mapping is machine-specific (real
project names and filesystem paths), so it's kept out of this repo: define it
in registry_local.py (gitignored) — see registry_local.example.py for the
format. Without a registry_local.py, KNOWN_ARCHIVES is just empty and
multi-archive search (--archives) / ai_time.py's all-projects default have
nothing to iterate over.

Used by query.py's --archives multi-archive search and by ai_time.py's
default (all-projects) scope.
"""

from pathlib import Path

try:
    from .registry_local import KNOWN_ARCHIVES  # type: ignore[import-not-found]
except ImportError:
    KNOWN_ARCHIVES: dict[str, Path] = {}
