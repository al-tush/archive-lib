# archive-lib

*English version: [README.md](README.md).*

**Превращает историю чатов с ИИ-ассистентом и заметки проекта в приватную
поисковую память — полностью локально, без облачных сервисов.**

archive-lib загружает транскрипты сессий Claude Code (и GitHub Copilot),
markdown-файлы проекта и, опционально, расшифровки записей звонков в
SQLite-базу каждого проекта, после чего по ней можно искать по точным словам,
по смыслу или комбинированно. Это отвечает на вопросы вроде «что мы решили по X
в прошлом месяце?» или «сколько раз поднималась тема Y?» без ручного
перечитывания сотен сессий.

Чем полезно:

- **Ничего не теряется.** Claude Code ротирует живые логи сессий; archive-lib
  инкрементально копирует их в ваш репозиторий (base64-вложения извлекаются и
  дедуплицируются), так что история переживает источник.
- **Три режима поиска в одном CLI.** FTS5-поиск по ключевым словам (с
  автоматическим расширением кириллических префиксов), семантический поиск
  (мультиязычные эмбеддинги предложений) и гибрид обоих — с time-decay-ранжированием
  и гарантированными местами для хитов из памяти, файлов проекта и транскриптов.
- **Ответы на вопросы по всему корпусу.** `full_scan.py` последовательно
  обходит *все* сессии-кандидаты в хронологическом порядке батчами — для
  вопросов «сколько раз» и «как менялось со временем», на которые top-k поиск
  ответить не может.
- **Одна кодовая база — много проектов.** В каждом проекте остаются только
  небольшой `config.py` и тонкие CLI-обёртки; логика правится один раз, и
  изменение получают все проекты. Мульти-архивный поиск охватывает сразу
  несколько проектов.
- **Пайплайн транскрипции звонков.** Записи встреч → текст с метками
  говорящих (whisper.cpp + диаризация sherpa-onnx, voice enrollment,
  опциональная сверка GigaAM и видео-верификация говорящего), после чего
  попадают в тот же поисковый архив.
- **Измеримое качество.** Оценщик по golden set считает Recall@K / MRR, так что
  изменения ранжирования настраиваются по данным, а не на глаз.

## Возможности

| Область | Что делает | Точка входа |
|---|---|---|
| Ingest | Claude Code JSONL → SQLite + FTS5; опционально транскрипты сабагентов; детекция форков сессий; пространства имён источников; извлечение вложений | `archive_lib/ingest.py` |
| Copilot | Сессии VS Code Copilot → та же БД; реконструкция долговечных JSONL-копий | `archive_lib/ingest_copilot.py` |
| Эмбеддинги | Инкрементальное встраивание новых сообщений (по умолчанию `paraphrase-multilingual-MiniLM-L12-v2`) | `archive_lib/embed.py` |
| Индекс файлов | Нарезка markdown проекта/памяти по секциям `##`; инкрементально по mtime; автообнаружение файлов сверх явного `FILE_REGISTRY` | `archive_lib/index_files.py`, `file_discovery.py` |
| Поиск | FTS / `--semantic` / `--hybrid`; фильтры по роли, дате, сессии, источнику и корпусу; поиск артефактов; исключение сессий; мульти-архив (`--archives`) | `archive_lib/query.py` |
| Full scan | Исчерпывающий хронологический обход сессий-кандидатов батчами для синтеза | `archive_lib/full_scan.py` |
| Restore | Полное восстановление сессии в читаемый markdown | `archive_lib/restore.py` |
| Пайплайн | Единое обновление: ingest → Copilot → embed → индекс файлов | `archive_lib/update.py` |
| Оценка | Recall@K / MRR по golden set проекта, плюс режим full-scan | `archive_lib/eval.py` |
| Аналитика | Время с ИИ по дням по всем проектам (пауза ≥ 5 мин разрывает окно, перекрытия сливаются) | `archive_lib/ai_time.py` |
| Гигиена текста | Находит и чинит слипшиеся кириллица/латиница-токены, ломающие FTS | `archive_lib/hybrid_words.py` |
| Звонки | Правила транскрипции, флаг низкоуверенных токенов, детекция рамки говорящего на видео, сборка документа | `scripts/`, `sherpa-models/` |
| Экспорт в docx | Эталонный pandoc-шаблон для `.docx` | `templates/` |

Использование команд: [CLI_REFERENCE.md](CLI_REFERENCE.md). Методология
ранжирования/тюнинга: [TUNING.md](TUNING.md). Пайплайн звонков:
[transcription_rules.md](transcription_rules.md).

## Как это устроено

archive-lib — общая библиотека, на которую ссылаются несколько независимых
проектов (у каждого свой git-репозиторий, без общего родителя) через
`sys.path.insert("<абсолютный-путь-до-archive-lib>")` в `archive/scripts/config.py`
каждого проекта.

Специфика проекта (путь к БД, Copilot workspace ID, `FILE_REGISTRY`,
`GOLDEN_SET`, `pool_guarantees`) живёт только в его собственном `config.py`
(экземпляр `ProjectConfig` — схему см. в `archive_lib/config.py`). Ни имя
проекта, ни путь не зашиты в сам `archive_lib`.

```
archive-lib/
├── archive_lib/          # вся логика — карту модулей см. в TUNING.md
├── scripts/              # хелперы транскрипции звонков, детекция говорящего, миграции
├── sherpa-models/        # скрипты диаризации и voice-enrollment (модели в gitignore)
├── templates/            # эталонный pandoc .docx
├── tests/                # unit-тесты, не зависят от проекта (query, ingest, hybrid_words)
├── CLI_REFERENCE.md      # справочник команд (usage)
├── TUNING.md             # методология тюнинга RAG (POOL_GUARANTEES, decay, K)
├── transcription_rules.md# общая методология транскрипции + диаризации звонков
└── TRANSCRIPTION.md      # собирается из правил + локальных заметок (gitignored)
```

`archive/scripts/*.py` каждого проекта — это обёртка на ~6 строк:

```python
import sys
from config import CONFIG
from archive_lib import query

if __name__ == "__main__":
    sys.exit(query.main(CONFIG, sys.argv[1:]))
```

Быстрый старт, из корня проекта:

```bash
python3 archive/scripts/'!update.py'                      # ingest + embed + индекс файлов
python3 archive/scripts/query.py "слова"                  # поиск по ключевым словам
python3 archive/scripts/query.py "смутная идея" --hybrid  # ключевые слова + семантика
python3 archive/scripts/full_scan.py "тема"               # исчерпывающий обход: подсчёты/хронология
python3 archive/scripts/restore.py <префикс-id-сессии>    # вся сессия в markdown
```

После любого изменения здесь — проверить идентичность обёрток между проектами:

```bash
diff -rq <project-a>/archive/scripts <project-b>/archive/scripts
```

Различия должны быть только в `config.py` и `README.md` — что-либо ещё
означает, что логика просочилась обратно в обёртку конкретного проекта.

Запуск unit-тестов (БД и модель не нужны):

```bash
python3 tests/test_query.py -v
python3 tests/test_ingest.py -v
python3 tests/test_hybrid_words.py -v
```

## Зависимости

- `pip install -r requirements.txt` — основной ingest/поиск (`archive_lib`;
  `sentence-transformers`, `numpy`). Ingest, FTS-поиск, restore и тесты
  работают на одной stdlib; пакеты нужны для `--semantic`/`--hybrid` и
  эмбеддингов. `query.py` сам перезапускается под интерпретатором, где они
  установлены.
- `pip install -r requirements-diarization.txt` — диаризация звонков / voice
  enrollment (`sherpa-models/`), **в отдельном venv** — почему их нельзя
  ставить в одно окружение, см. [transcription_rules.md](transcription_rules.md).
- Сама транскрипция использует `whisper-cli` (whisper.cpp) и `ffmpeg` как
  внешние CLI-инструменты, не pip-пакеты.

## Лицензия

MIT — см. [LICENSE](LICENSE). `sherpa-models/offline-speaker-diarization-example.py`
— неизменённый пример из upstream sherpa-onnx (Apache License 2.0); см.
пометку в начале этого файла.

## Локальные файлы (gitignored)

Несколько файлов содержат данные конкретной машины (реальные имена проектов,
пути, заметки по транскрипции звонков) и намеренно не входят в репозиторий:

- `archive_lib/registry_local.py` — реальный маппинг имя → `transcripts.db` для
  `archive_lib/registry.py`; для старта скопируйте `registry_local.example.py`.
- `TRANSCRIPTION.md` / `altush_config.md` — собираются из
  [transcription_rules.md](transcription_rules.md) плюс локальные детали через
  `python3 scripts/build_transcription_doc.py`; шаблон `altush_config.md` —
  [sample_config.md](sample_config.md).
- `call-logs/`, `.env`, файлы моделей и виртуальные окружения
  (`whisper-models/`, `sherpa-venv/`, `gigaam-venv/`, `pyannote-venv/`).
