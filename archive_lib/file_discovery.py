"""Auto-discovery helpers for a project's FILE_REGISTRY.

A project's archive/scripts/config.py used to hand-list every markdown file
that should be searchable via RAG. That list drifts: a file created without
also editing config.py is invisible to index_files.py forever, silently —
there is no error, it just never gets indexed. This module lets a project
keep a short EXPLICIT list (files needing a specific weight/type, or files
outside project_root) and auto-discover everything else, so a new file is
covered the moment it exists.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Callable

from .config import FileRegistryEntry

DEFAULT_EXCLUDE_DIRS = frozenset({"archive", ".claude", ".git"})


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def discover_project_markdown(
    project_root: Path,
    full_scan_dirs: tuple[str, ...] = (),
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS,
) -> list[str]:
    """Root-level .md files, plus .md files directly inside `full_scan_dirs`
    (a real content folder with no README of its own — e.g. a folder of
    corporate instruction docs), plus just the top-level README.md of any
    other first-level subfolder.

    The README-only default is deliberate: a subfolder without one is
    treated as generated output or a raw data dump (script reports,
    screenshots, an audit's raw transcripts) — not itself a knowledge
    document — while a subfolder that does ship a README is a self-described
    bundle worth indexing by that description alone.

    Returned paths are NFC-normalized. macOS stores filenames NFD-decomposed
    on disk; a FILE_REGISTRY string literal is typically typed as NFC, and an
    exact-string match between the two forms silently fails.
    """
    found: list[str] = []

    for p in sorted(project_root.glob("*.md")):
        found.append(_nfc(p.name))

    for dname in full_scan_dirs:
        d = project_root / dname
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.md")):
            found.append(_nfc(f"{dname}/{p.name}"))

    for d in sorted(project_root.iterdir()):
        if not d.is_dir() or d.name in exclude_dirs or d.name in full_scan_dirs:
            continue
        readme = d / "README.md"
        if readme.exists():
            found.append(_nfc(f"{d.name}/README.md"))

    return found


def build_file_registry(
    project_root: Path,
    explicit: list[FileRegistryEntry],
    classify: Callable[[str], tuple[str, float]],
    full_scan_dirs: tuple[str, ...] = (),
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS,
) -> list[FileRegistryEntry]:
    """explicit + auto-discovered entries for everything explicit doesn't
    already cover. `explicit` wins on overlap — it's for files needing a
    specific (source_type, weight) or living outside project_root (root
    != None, or an absolute-path rel_path such as another repo's docs).
    `classify(rel_path)` returns (source_type, weight) for the rest.
    """
    explicit_local_paths = {_nfc(e[0]) for e in explicit if e[3] is None}

    discovered = discover_project_markdown(project_root, full_scan_dirs, exclude_dirs)
    auto_entries = [
        (rel_path, *classify(rel_path), None)
        for rel_path in discovered
        if rel_path not in explicit_local_paths
    ]

    return list(explicit) + auto_entries
