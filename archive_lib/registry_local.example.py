"""
Template for registry_local.py (gitignored — not part of this repo).

Copy this file to registry_local.py and fill in your real project names and
transcripts.db paths. See registry.py for how it's consumed.
"""

from pathlib import Path

KNOWN_ARCHIVES: dict[str, Path] = {
    "project-a": Path("/absolute/path/to/project-a/archive/transcripts.db"),
    "project-b": Path("/absolute/path/to/project-b/archive/transcripts.db"),
}
