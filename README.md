# archive-lib

Shared logic behind the transcript-archiving/RAG-search system used by several
independent projects (each its own git repository with no common parent). This
directory is a plain (non-git) shared library, referenced via
`sys.path.insert("<absolute-path-to-archive-lib>")` from each project's
`archive/scripts/config.py`.

Goal: one code change here applies to every sibling project. Per-project
differences (DB path, Copilot workspace ID, `FILE_REGISTRY`, `GOLDEN_SET`,
`pool_guarantees`) live only in each project's own `archive/scripts/config.py`
(a `ProjectConfig` instance — see `archive_lib/config.py` for the schema).

```
archive-lib/
├── archive_lib/        # all logic — see TUNING.md for the module map
├── tests/test_query.py # unit tests, config-agnostic
├── CLI_REFERENCE.md    # command reference (usage)
├── TUNING.md            # RAG tuning methodology (POOL_GUARANTEES, decay, K)
└── TRANSCRIPTION.md     # call transcription+diarization pipeline (whisper.cpp + sherpa-onnx)
```

Each project's `archive/scripts/*.py` is a ~6-line wrapper:

```python
import sys
from config import CONFIG
from archive_lib import query

if __name__ == "__main__":
    sys.exit(query.main(CONFIG, sys.argv[1:]))
```

Verify parity across projects after any change here:

```bash
diff -rq <project-a>/archive/scripts <project-b>/archive/scripts
```

Differences should appear only in `config.py` and `README.md` — anything else
means logic leaked back into a per-project wrapper.

## Requirements

- `pip install -r requirements.txt` — core ingest/search (`archive_lib`).
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
  [transcription_rules.md](transcription_rules.md) plus local specifics; see
  that file's header for how to (re)build it, and [sample_config.md](sample_config.md)
  for the `altush_config.md` template.

Русская версия: [README.ru.md](README.ru.md).
