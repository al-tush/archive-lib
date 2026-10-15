# archive-lib

**Turn your AI-assistant chat history and project notes into a private, searchable
memory — fully local, no cloud service.**

archive-lib ingests Claude Code (and GitHub Copilot) session transcripts, your
project's markdown files, and optionally transcribed call recordings into a
per-project SQLite database, then lets you search it by exact words, by meaning,
or both. It answers questions like "what did we decide about X last month?" or
"how many times did Y come up?" without re-reading hundreds of sessions by hand.

Why it is useful:

- **Nothing is lost.** Claude Code rotates its live session logs; archive-lib
  copies them into your repository incrementally (base64 attachments are extracted
  and deduplicated), so history outlives the source.
- **Three search modes in one CLI.** FTS5 keyword search (with automatic Cyrillic
  prefix expansion), semantic search (multilingual sentence embeddings), and a
  hybrid of both — with time-decay ranking and guaranteed slots for memory,
  project-file and transcript hits.
- **Answers for whole-corpus questions.** `full_scan.py` walks *all* candidate
  sessions chronologically in batches, for counting and "how did this change over
  time" questions that top-k search cannot answer.
- **One codebase, many projects.** Every project keeps only a small `config.py`
  and thin CLI wrappers; fix or improve the logic once and all projects get it.
  Multi-archive search spans several projects at once.
- **Call transcription pipeline.** Meeting recordings → text with speaker labels
  (whisper.cpp + sherpa-onnx diarization, voice enrollment, optional GigaAM
  cross-check and video-based speaker verification), which then flow into the
  same searchable archive.
- **Measurable quality.** A golden-set evaluator reports Recall@K / MRR, so
  ranking changes can be tuned against data rather than guessed.

## Capabilities

| Area | What it does | Entry point |
|---|---|---|
| Ingest | Claude Code JSONL → SQLite + FTS5; optional subagent transcripts; session-fork detection; per-source namespaces; attachment extraction | `archive_lib/ingest.py` |
| Copilot | VS Code Copilot chat sessions → same DB; reconstructs durable JSONL copies | `archive_lib/ingest_copilot.py` |
| Embeddings | Incremental embedding of new messages (`paraphrase-multilingual-MiniLM-L12-v2` by default) | `archive_lib/embed.py` |
| File index | Chunks project/memory markdown by `##` sections; incremental by mtime; auto-discovery of files beyond an explicit `FILE_REGISTRY` | `archive_lib/index_files.py`, `file_discovery.py` |
| Search | FTS / `--semantic` / `--hybrid`; role, date, session, source and corpus filters; artifact search; session exclusion; multi-archive (`--archives`) | `archive_lib/query.py` |
| Full scan | Exhaustive, chronologically batched candidate sessions for synthesis | `archive_lib/full_scan.py` |
| Restore | Rebuild a full session as readable markdown | `archive_lib/restore.py` |
| Pipeline | One-shot update: ingest → Copilot → embed → file index | `archive_lib/update.py` |
| Evaluation | Recall@K / MRR over a per-project golden set, plus full-scan mode | `archive_lib/eval.py` |
| Analytics | Time spent with AI per day across all projects (pauses ≥ 5 min split windows, overlaps merged) | `archive_lib/ai_time.py` |
| Text hygiene | Detects and repairs glued Cyrillic/Latin tokens that break FTS | `archive_lib/hybrid_words.py` |
| Call tooling | Transcription rules, low-confidence token flagging, speaker-ring detection on call video, doc builder | `scripts/`, `sherpa-models/` |
| Docs export | Pandoc reference template for `.docx` output | `templates/` |

Command usage: [CLI_REFERENCE.md](CLI_REFERENCE.md). Ranking/tuning methodology:
[TUNING.md](TUNING.md). Call pipeline: [transcription_rules.md](transcription_rules.md).

## How it is wired

archive-lib is a shared library referenced by several independent projects (each
its own git repository with no common parent) via
`sys.path.insert("<absolute-path-to-archive-lib>")` from each project's
`archive/scripts/config.py`.

Per-project differences (DB path, Copilot workspace ID, `FILE_REGISTRY`,
`GOLDEN_SET`, `pool_guarantees`) live only in that project's own `config.py` (a
`ProjectConfig` instance — see `archive_lib/config.py` for the schema). No project
name or path is hardcoded in `archive_lib` itself.

```
archive-lib/
├── archive_lib/          # all logic — see TUNING.md for the module map
├── scripts/              # call-transcription helpers, speaker detection, migrations
├── sherpa-models/        # diarization + voice-enrollment scripts (models gitignored)
├── templates/            # pandoc reference .docx
├── tests/                # unit tests, config-agnostic (query, ingest, hybrid_words)
├── CLI_REFERENCE.md      # command reference (usage)
├── TUNING.md             # RAG tuning methodology (POOL_GUARANTEES, decay, K)
├── transcription_rules.md# generic call transcription + diarization methodology
└── TRANSCRIPTION.md      # generated from the rules + local notes (gitignored)
```

Each project's `archive/scripts/*.py` is a ~6-line wrapper:

```python
import sys
from config import CONFIG
from archive_lib import query

if __name__ == "__main__":
    sys.exit(query.main(CONFIG, sys.argv[1:]))
```

Quick start, from a project's root:

```bash
python3 archive/scripts/'!update.py'                      # ingest + embed + index files
python3 archive/scripts/query.py "some words"             # keyword search
python3 archive/scripts/query.py "vague idea" --hybrid    # keyword + semantic
python3 archive/scripts/full_scan.py "topic"              # exhaustive, for counting/timelines
python3 archive/scripts/restore.py <session-id-prefix>    # full session as markdown
```

Verify parity across projects after any change here:

```bash
diff -rq <project-a>/archive/scripts <project-b>/archive/scripts
```

Differences should appear only in `config.py` and `README.md` — anything else
means logic leaked back into a per-project wrapper.

Run the unit tests (no database or model needed):

```bash
python3 tests/test_query.py -v
python3 tests/test_ingest.py -v
python3 tests/test_hybrid_words.py -v
```

## Requirements

- `pip install -r requirements.txt` — core ingest/search (`archive_lib`;
  `sentence-transformers`, `numpy`). Ingest, FTS search, restore and tests are
  stdlib-only; the packages are needed for `--semantic`/`--hybrid` and embedding.
  `query.py` re-executes itself under the interpreter that has them installed.
- `pip install -r requirements-diarization.txt` — call diarization / voice
  enrollment (`sherpa-models/`), **in a separate venv** — see
  [transcription_rules.md](transcription_rules.md) for why these two must not
  share an environment.
- Transcription itself uses `whisper-cli` (whisper.cpp) and `ffmpeg` as
  external command-line tools, not pip packages.

## License

MIT — see [LICENSE](LICENSE). `sherpa-models/offline-speaker-diarization-example.py`
is an unmodified upstream sherpa-onnx example (Apache License 2.0); see the
notice at the top of that file.

## Local-only files (gitignored)

A few files hold machine-specific data (real project names, paths, call
transcription notes) and are intentionally not part of this repo:

- `archive_lib/registry_local.py` — the real name → `transcripts.db` mapping
  for `archive_lib/registry.py`; copy `registry_local.example.py` to get started.
- `TRANSCRIPTION.md` / `altush_config.md` — generated from
  [transcription_rules.md](transcription_rules.md) plus local specifics via
  `python3 scripts/build_transcription_doc.py`; see
  [sample_config.md](sample_config.md) for the `altush_config.md` template.
- `call-logs/`, `.env`, model files and virtualenvs (`whisper-models/`,
  `sherpa-venv/`, `gigaam-venv/`, `pyannote-venv/`).

Русская версия: [README.ru.md](README.ru.md).
