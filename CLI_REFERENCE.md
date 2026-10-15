# RAG — справочник команд

Общий для всех проектов, использующих archive-lib. CLI-поверхность одинакова
во всех — отличается только содержимое БД и `FILE_REGISTRY`/`GOLDEN_SET` в
`archive/scripts/config.py` каждого проекта.

Все команды запускаются из корня конкретного проекта, например:
`python3 archive/scripts/query.py ...` — работает одинаково в любом из них.

## FTS (дефолт)

```bash
# Кириллица авторасширяется (* добавляется сам)
python3 archive/scripts/query.py "слово"
python3 archive/scripts/query.py "слово AND другое"
python3 archive/scripts/query.py "слово" --all-roles
python3 archive/scripts/query.py "слово" --roles user,assistant
python3 archive/scripts/query.py "слово" --role assistant
python3 archive/scripts/query.py "слово" --from 2026-04-01
python3 archive/scripts/query.py "слово" --recent
python3 archive/scripts/query.py "слово" --limit 10 --full
python3 archive/scripts/query.py "слово" --preview 3
python3 archive/scripts/query.py --session e165b96a
python3 archive/scripts/query.py --artifacts "тема"
python3 archive/scripts/query.py --exclude-session e165b96a
python3 archive/scripts/query.py --list-excluded
```

FTS5 + кириллица: без стемминга, `слово*` добавляется автоматически (`--no-expand` отключает).
Ловушка раскладки: e ≠ е, o ≠ о — проверять.

## Semantic / Hybrid

```bash
python3 archive/scripts/query.py "запрос" --semantic
python3 archive/scripts/query.py "запрос" --hybrid
python3 archive/scripts/query.py "запрос" --semantic --corpus files
python3 archive/scripts/query.py "запрос" --semantic --corpus transcripts
```

Обычный `python3` работает: если под текущим интерпретатором нет
`sentence_transformers` (например, голый `python3` без указания на venv),
`query.py` сам перезапускается под интерпретатором, где модуль установлен
(`_ensure_semantic_capable` в `archive_lib/query.py`). Явный префикс
`PYTHON=.../python3.10 $PYTHON ...` больше не обязателен — можно указать
вручную только чтобы пропустить лишний re-exec.

`--hybrid` при нуле FTS-кандидатов теперь не возвращает пусто, а откатывается
на чистый `--semantic` (эссеистские/рефлексивные формулировки часто не
пересекаются по словам с индексированным текстом, но семантически близкое
содержимое есть).

## Мульти-архивный поиск

Ищет по нескольким проектам сразу — имена берутся из `archive_lib.registry.KNOWN_ARCHIVES`
(см. `archive_lib/registry_local.py`, gitignored — локальный маппинг имя → `transcripts.db`),
либо можно передать абсолютный путь к чужому `transcripts.db`.

```bash
python3 archive/scripts/query.py "запрос" --archives project-a,project-b
python3 archive/scripts/query.py "запрос" --archives project-a,project-b,project-c --semantic
```

## Restore

```bash
python3 archive/scripts/restore.py <uuid-prefix> --no-attachments
python3 archive/scripts/restore.py <uuid-prefix> --no-gaps    # без дельт времени и маркеров пауз
```

## Сабагенты (Task/Agent-вызовы: auditor и др.) — опционально

По умолчанию транскрипты сабагентов не индексируются и не ищутся — это внутренняя кухня (grep/bash-шум), а не диалог с пользователем.

```bash
# Разовая индексация (нужно перезапускать вручную — auto-update её не подхватывает)
python3 archive/scripts/ingest.py --include-subagents

# Поиск с подмешиванием сабагентов к обычному результату
python3 archive/scripts/query.py "запрос" --include-subagents

# Поиск только внутри сабагентов
python3 archive/scripts/query.py "запрос" --source subagent

# Полный лог конкретного сабагента (префикс с "agent-", не голый хэш!)
python3 archive/scripts/query.py --session agent-ae60c451

# restore.py матчит по подстроке — голый хэш работает и здесь
python3 archive/scripts/restore.py ae60c451
```

В выдаче сабагентские хиты помечены `[agent:<type> ← <parent_session>]` — короткий id родительской сессии, откуда сабагент был вызван (можно сразу `restore.py <parent_session>` за полным контекстом). `restore.py` на сабагенте дополнительно печатает задачу (`description` из вызова `Agent(...)`) и id родительского сообщения.

Важно: сабагентские транскрипты физически лежат в `<session-id>/subagents/agent-<id>.jsonl`
внутри live-директории `~/.claude/projects/...` — `ingest.py` без `--include-subagents`
туда не заходит вовсе (не «фильтрует», а не читает источник).

## Выбор режима

| Задача | Режим |
|---|---|
| Точные слова / даты / препараты | FTS (дефолт) |
| Размытые запросы, синонимы | `--semantic` |
| Состояние, паттерны, отношения | `--hybrid` |
| Найти файл из сессии | `--artifacts` |
| Последние 30 дней | `--recent` |
| Подсчёт/агрегация по корпусу, хронологический multi-hop синтез | `full_scan.py` (см. ниже) |

## Full-scan (альтернатива top-k)

Top-k (`--semantic`/`--hybrid`) ранжирует по релевантности *отдельного* сниппета
запросу — на вопросы вида «сколько раз поднимался X» или «как менялась позиция
по X со временем» это не отвечает в принципе (другая ось: не похожесть на
запрос, а что происходило системно/по времени), и увеличение `--limit` не
чинит это. `full_scan.py` — систематический обход ВСЕХ сессий-кандидатов
(без top-k cap), сгруппированных по сессии и хронологии, подготовленных
батчами для `restore.py` + синтеза. Сам синтез не делает — это механический
слой, решение/синтез остаётся за вызывающей Claude Code сессией.

```bash
python3 archive/scripts/full_scan.py "запрос"
python3 archive/scripts/full_scan.py "запрос" --batch-size 8
python3 archive/scripts/full_scan.py "запрос" --manifest          # принудительно JSON, даже если один батч
python3 archive/scripts/full_scan.py "запрос" --from 2026-08-01 --to 2026-09-01
python3 archive/scripts/full_scan.py "запрос" --hybrid            # семантический ранкинг кандидатов внутри
```

- Кандидатов ≤ `--batch-size` и без `--manifest` → печатает один
  конкатенированный текст сессий (restore-эквивалент), готовый к прямому
  чтению.
- Иначе → JSON-манифест батчей (`session_id`-списки по хронологии + диапазон
  дат на батч) — на каждый батч спавнить `Agent` с задачей синтеза, слить N
  отчётов самостоятельно.
- Не нужен (не даёт выигрыша) на точечный факт или редкий точный токен
  (ID карточки, кодовое слово) — там exact FTS top-k уже точен и дешевле.
- `query.py` сам подсказывает, когда стоит переключиться: если найденных
  кандидатов существенно больше, чем показано в top-k, печатает `[hint]` в
  stderr с готовой командой `full_scan.py`.

## Обслуживание

```bash
$PYTHON archive/scripts/'!update.py'              # ingest + copilot + embed + index_files
$PYTHON archive/scripts/'!update.py' --dry-run    # что изменится, без записи
$PYTHON archive/scripts/'!update.py' --no-copilot # пропустить Copilot
$PYTHON archive/scripts/eval.py                   # Recall@K / MRR по golden_set проекта
python3 archive-lib/tests/test_query.py -v        # unit-тесты (без БД/модели, общие для всех проектов)
python3 archive/scripts/ai_time.py                # время с ИИ по дням — по умолчанию все проекты сразу
```

Время жизни данных: `ingest.py` читает **только** живую директорию
`~/.claude/projects/<encoded-project-path>/` (см. `config.claude_source_dir`) — она со временем
чистится/ротируется Claude Code. `archive/transcripts/` (git-tracked копии) —
это цель записи `ingest.py`, а не источник для повторного чтения. Из-за этого
полная пересборка `transcripts.db` с нуля **теряет** сессии, чей live-JSONL
уже пропал, даже если архивная копия в `archive/transcripts/` жива. Не удалять
`transcripts.db` без крайней необходимости — обновление всегда инкрементально.

Copilot-сессии (`archive/transcripts/copilot/`): в отличие от Claude Code,
исходные chatSessions JSONL живут только в workspaceStorage VS Code — не
git-tracked, не бэкапятся вообще. `ingest_copilot.py` при каждом запуске
реконструирует Claude-Code-совместимый JSONL (по тексту реплик из БД, не
байт-в-байт оригинал) для любой сессии, у которой такой копии ещё нет на
диске, и кладёт в `archive/transcripts/copilot/`. Это единственная долговременная
копия Copilot-истории — обновляется автоматически при каждом `!update.py`.

## Архитектура

Вся логика — в `archive-lib/archive_lib/`. Каждый проект
держит в своём `archive/scripts/` только `config.py` (пути, workspace ID,
`FILE_REGISTRY`, `GOLDEN_SET`, `pool_guarantees`) и тонкие CLI-обёртки, которые
импортируют `archive_lib` и передают ему свой `CONFIG`. Подробнее про тюнинг
семантического поиска (POOL_GUARANTEES, time-decay, диагностика промахов) →
[TUNING.md](TUNING.md).
