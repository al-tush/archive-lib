# archive-lib

*English version: [README.md](README.md).*

Общая логика системы архивации транскриптов / RAG-поиска, используемая
несколькими независимыми проектами (у каждого свой git-репозиторий, без
общего родителя). Эта директория — обычная (не-git) shared-библиотека, на
которую каждый проект ссылается через
`sys.path.insert("<абсолютный-путь-до-archive-lib>")` в своём
`archive/scripts/config.py`.

Цель: одно изменение кода здесь применяется сразу ко всем проектам.
Специфика конкретного проекта (путь к БД, Copilot workspace ID,
`FILE_REGISTRY`, `GOLDEN_SET`, `pool_guarantees`) живёт только в его
собственном `archive/scripts/config.py` (экземпляр `ProjectConfig` — схему
см. в `archive_lib/config.py`).

```
archive-lib/
├── archive_lib/        # вся логика — карту модулей см. в TUNING.md
├── tests/test_query.py # unit-тесты, не зависят от конкретного проекта
├── CLI_REFERENCE.md    # справочник команд (usage)
├── TUNING.md            # методология тюнинга RAG (POOL_GUARANTEES, decay, K)
└── TRANSCRIPTION.md     # пайплайн транскрипции+диаризации звонков (whisper.cpp + sherpa-onnx)
```

`archive/scripts/*.py` каждого проекта — это обёртка на ~6 строк:

```python
import sys
from config import CONFIG
from archive_lib import query

if __name__ == "__main__":
    sys.exit(query.main(CONFIG, sys.argv[1:]))
```

После любого изменения здесь — проверить идентичность обёрток между проектами:

```bash
diff -rq <project-a>/archive/scripts <project-b>/archive/scripts
```

Различия должны быть только в `config.py` и `README.md` — что-либо ещё
означает, что логика просочилась обратно в обёртку конкретного проекта.

## Локальные файлы (gitignored)

Несколько файлов хранят машинно-специфичные данные (реальные имена проектов,
пути, заметки по транскрипции звонков) и намеренно не входят в этот
репозиторий:

- `archive_lib/registry_local.py` — реальный маппинг имя → `transcripts.db`
  для `archive_lib/registry.py`; за основу взять `registry_local.example.py`.
- `TRANSCRIPTION.md` / `altush_config.md` — собираются из
  [transcription_rules.md](transcription_rules.md) плюс локальные детали; как
  (пере)собрать — см. заголовок этого файла, шаблон для `altush_config.md` —
  в [sample_config.md](sample_config.md).
