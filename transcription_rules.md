# Транскрипция и диаризация звонков — общая методология

Рабочая связка для расшифровки записей разговоров (`.cap`, `.m4a` и т.п.),
общая для всех проектов, использующих archive-lib.

Личные детали (реальные абсолютные пути, кто прошёл voice-enrollment и когда,
какие конкретно звонки калибровались) сюда не входят — они живут в
`altush_config.md` (gitignored, не часть этого репозитория). Итоговый
`TRANSCRIPTION.md`, который реально читают проекты, собирается из этого файла
плюс `altush_config.md`:

```bash
python3 scripts/build_transcription_doc.py
```

Без `altush_config.md` скрипт всё равно соберёт `TRANSCRIPTION.md` — просто
без секции с личными заметками.

## Транскрипция

`whisper-cli` (whisper.cpp, нативный arm64/Metal) + модель
`whisper-models/ggml-large-v3.bin` относительно корня archive-lib (гитигнорена).
Без ручной чистки текст обычно почти без пунктуации; иногда 1–2 места с
дублированием фразы подряд (артефакт скользящего окна) — не всегда, надо
проверять на глаз.

Аудио перед подачей — конвертировать в WAV 16kHz mono через ffmpeg.

## Диаризация

sherpa-onnx (arm64-native, ONNX Runtime, без PyTorch) — venv
`sherpa-venv/` относительно корня archive-lib (arm64 python3.11 из
`/opt/homebrew`, НЕ x86_64 `/usr/local` python3.10). Модели и скрипты —
в `sherpa-models/`:

- `run_diarization.py <wav> <out.json> <num_speakers>` — сегментация по спикерам
- `merge_transcript_diarization.py <srt> <diar.json> <out.txt>` — склейка транскрипта
  с диаризацией по таймкодам
- `enroll_voice.py add <имя> <wav> [start end]` — добавить голос в профиль
- `enroll_voice.py identify <wav> <diar.json>` — сверить кластеры спикеров с профилями
- `voice_profiles.json` (гитигнорен, там биометрия) — хранит именованные голосовые
  профили; кто именно в них обучен, когда и с каким score — см. `altush_config.md`.

## Почему не WhisperX/pyannote

Не пытаться прогонять WhisperX/pyannote.audio через `python3.10` из `/usr/local`
(x86_64-сборка, работает через Rosetta на M1 Pro) для диаризации.

**Why:** два прогона WhisperX (`--diarize`, `--compute_type int8 --device cpu`)
подряд крашились на нативном уровне — процесс зависал в состоянии
uninterruptible sleep (`UE` в `ps`), `kill -9` не действовал физически, ядро
блокировало убийство. Параллельно зависал `ReportCrash` (до 97% CPU по
несколько минут), пользователь видел зависший крашрепорт на экране. Похоже на
баг взаимодействия ctranslate2/MKL/multiprocessing под Rosetta-эмуляцией (в
логе — предупреждение о leaked semaphore перед зависанием), а не случайность —
воспроизвелось дважды подряд с разными файлами. Также при первой попытке
установка WhisperX подтянула numpy 2.x в ОБЩИЙ python3.10, сломав
`--semantic`/`--hybrid` в RAG-поиске archive-lib для всех проектов (пришлось
откатывать).

**How to apply:** если возникает похожая задача (диаризация, что-то на
PyTorch) на этой машине — сначала проверять, есть ли arm64-native альтернатива
(ONNX Runtime, ggml/whisper.cpp и т.п.), и ставить её в отдельный venv на arm64
python (`/opt/homebrew`), не в общий x86_64 python3.10. Тяжёлые ML-библиотеки —
всегда в изолированный venv, никогда в общее окружение, которым пользуется
RAG-инструментарий archive-lib.

<!-- LOCAL_NOTES -->
