"""
Per-project configuration schema for archive-lib.

Each sibling project defines exactly one CONFIG = ProjectConfig(...) instance
in its own archive/scripts/config.py.
Everything in archive_lib is parameterized by this object — no project name,
path, or workspace ID is hardcoded anywhere in archive_lib itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# (path relative to its root, source_type, weight, root)
# root=None -> ProjectConfig.project_root ; root="memory" -> ProjectConfig.memory_root
FileRegistryEntry = tuple[str, str, float, Optional[str]]
# (query, expected_substring, description) — see eval.py
GoldenSetEntry = tuple[str, str, str]

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Pool names are fixed across all projects (query.py's semantic-search pooling
# always reasons about these three corpora), so every project declares all
# three keys even if one pool is currently empty for it (e.g. a project with no
# memory-tier files yet — see memory_root below).
DEFAULT_POOL_GUARANTEES = {"memory": 2, "project": 1, "transcript": 2}


@dataclass(frozen=True)
class ProjectConfig:
    name: str                                    # project slug, e.g. "project-a" — also the --archives label
    project_root: Path
    db_path: Path                                 # project_root / "archive/transcripts.db"
    transcripts_dir: Path                          # project_root / "archive/transcripts"
    files_dir: Path                                 # project_root / "archive/files"
    claude_source_dir: Path                          # ~/.claude/projects/<encoded-project-path>
    copilot_ws_id: Optional[str]                      # workspaceStorage UUID; None => skip Copilot ingest
    memory_root: Optional[Path]                        # project_root/.claude/memory, or None
    file_registry: list[FileRegistryEntry] = field(default_factory=list)
    # Additional ~/.claude/projects/-... dirs to ingest/search alongside claude_source_dir —
    # e.g. this project tracks work whose actual code lives in a separate repo with its
    # own Claude Code sessions. Empty for projects with a single source dir.
    extra_claude_source_dirs: list[Path] = field(default_factory=list)
    pool_guarantees: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_POOL_GUARANTEES))
    model_name: str = DEFAULT_MODEL_NAME
    golden_set: list[GoldenSetEntry] = field(default_factory=list)

    @property
    def scripts_dir(self) -> Path:
        return self.project_root / "archive" / "scripts"

    @property
    def copilot_chatsessions_dir(self) -> Optional[Path]:
        if not self.copilot_ws_id:
            return None
        return (
            Path.home()
            / "Library/Application Support/Code/User/workspaceStorage"
            / self.copilot_ws_id
            / "chatSessions"
        )
