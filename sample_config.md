# altush_config.md — template

Copy this to `altush_config.md` (gitignored) and fill in your own machine-local
specifics. `scripts/build_transcription_doc.py` merges it into
[transcription_rules.md](transcription_rules.md) to produce `TRANSCRIPTION.md` —
see that file's header for details. Everything below is a placeholder; nothing
here needs to be accurate or kept.

```markdown
## Личные заметки (voice-enrollment, звонки)

Собрана и откалибрована <дата> на звонках с <кем-то> (проект <project-name>).

Реальные пути:
- Модель whisper: <path-to-archive-lib>/whisper-models/ggml-large-v3.bin
- venv sherpa-onnx: <path-to-archive-lib>/sherpa-venv
- Модели/скрипты диаризации: <path-to-archive-lib>/sherpa-models/

Voice-профили (`voice_profiles.json`): профили «<имя-1>» и «<имя-2>»,
обучены на звонке <дата>, подтверждены на звонке <дата>
(score <x.xxx>/<x.xxx>, порог <x.x>).
```

If you don't need per-call voice notes at all, an empty (or missing)
`altush_config.md` is fine — `build_transcription_doc.py` falls back to a
placeholder line in that case.
