#!/usr/bin/env python3
"""
Merges transcription_rules.md (tracked, generic) with altush_config.md
(gitignored, machine-local specifics) into TRANSCRIPTION.md (gitignored) —
the file projects actually read for the full transcription/diarization setup.

Run from the archive-lib root: python3 scripts/build_transcription_doc.py
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent
MARKER = "<!-- LOCAL_NOTES -->"


def main() -> None:
    rules = (ROOT / "transcription_rules.md").read_text()
    local_path = ROOT / "altush_config.md"
    local_notes = local_path.read_text().strip() if local_path.exists() else (
        "_No local notes configured — create altush_config.md to add them._"
    )
    merged = rules.replace(MARKER, local_notes)
    (ROOT / "TRANSCRIPTION.md").write_text(merged)
    print("Wrote TRANSCRIPTION.md" + (" (with local notes)" if local_path.exists() else " (no local notes found)"))


if __name__ == "__main__":
    main()
