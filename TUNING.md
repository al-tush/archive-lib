# RAG — архитектура, параметры, тюнинг

Общая методология для всех проектов, использующих archive-lib. При обычном
использовании (не тюнинге) она не нужна — см. [CLI_REFERENCE.md](CLI_REFERENCE.md).

Текущие метрики (Recall@K/MRR) и список нерешённых промахов golden-set —
**не здесь**: они быстро устаревают и специфичны для содержимого каждого
проекта. Смотри их в коротком блоке в `<project>/archive/scripts/README.md`,
обновляемом вручную после запуска `eval.py`.

## Быстрый старт при тюнинге

```bash
# 1. Unit-тесты (без модели, ~0.002 сек) — запускать после любого изменения archive_lib.query/index_files:
python3 archive-lib/tests/test_query.py -v

# 2. Полный eval для конкретного проекта (загружает модель):
python3 archive/scripts/eval.py               # Recall@K / MRR, K по умолчанию = сумма pool_guarantees
python3 archive/scripts/eval.py --verbose     # + топ-3 сниппета на каждый запрос
python3 archive/scripts/eval.py --k 6         # если тестируешь другой K

# 3. Диагностика конкретного промаха:
python3 archive/scripts/query.py "запрос" --semantic --corpus files --limit 10
python3 archive/scripts/query.py "запрос" --semantic --corpus transcripts --limit 10
python3 archive/scripts/query.py "запрос" --hybrid --limit 10
```

## Устаревание результатов по типу темы

Семантический и hybrid-режимы применяют time-decay: `score × exp(−ln2 × days / 180)`.
Практически это означает:

| Возраст сниппета | Коэффициент |
|---|---|
| 7 дней | ×0.97 (почти нет штрафа) |
| 30 дней | ×0.89 |
| 90 дней | ×0.71 |
| 180 дней | ×0.50 (half-life) |
| 360 дней | ×0.25 |

**FTS-режим (без `--semantic`/`--hybrid`) time-decay не применяет** — результаты ранжируются только по BM25.

### Правило по типу темы

| Тип темы | Устаревание | Что делать |
|---|---|---|
| Фарма, дозировки, назначения | Высокое (меняется за дни) | Использовать `--from <дата последней коррекции>` или hybrid |
| Острое состояние, симптомы | Высокое | Hybrid с `--from` последних 2–3 недель |
| Поведенческие паттерны | Очень низкое | Memory-файлы достаточны; decay несущественен |
| Принятые решения, история | Низкое | Memory или FTS без ограничений по дате |
| Текущие проекты / задачи | Среднее | Оперативные файлы проекта первичны |

**Итог:** для холистических вопросов про паттерны читай memory без фильтра; для фармы и острого — всегда уточняй дату последнего события и передавай `--from`.

## Параметр K: eval vs интерактив

| Режим | K | Обоснование |
|---|---|---|
| `eval.py` | сумма `config.pool_guarantees.values()` (проектная) | Жёсткая диагностическая метрика — по построению совпадает с суммой гарантий, не может рассинхронизироваться. |
| `--semantic`, `--hybrid` в query.py | **10** (`--limit`) | Claude 4.x читает весь контекст надёжно; "lost in the middle" на 10 чанках несущественен. |

## pool_guarantees — минимум слотов из каждого корпуса

Поле `config.py` → `ProjectConfig.pool_guarantees`, используется в
`archive_lib/query.py::_merge_pools()`. Всегда объявляет все три ключа
(`memory`, `project`, `transcript`), даже если один пул у проекта пока пуст —
пустой пул с ненулевой гарантией просто не вносит вклад, это безопасно и
позволяет включить пул позже без изменения кода.

**Инвариант:** сумма гарантий ≤ K. Если нарушен — Phase 1 создаёт >K элементов,
`result[:k]` молча обрезает наименее скоровый гарантированный элемент, сводя гарантию к нулю.
`eval.py`'s `--k` по умолчанию = сумма гарантий именно по этой причине — не может нарушить инвариант случайно.
Тест `test_sum_does_not_exceed_default_k` в `archive-lib/tests/test_query.py` документирует инвариант на фикстуре.

**Когда пересматривать (per-project, в `config.py`):**

| Симптом | Действие |
|---|---|
| ИИ не видит паттерны / правила из memory-файлов | `pool_guarantees["memory"]` +1 |
| ИИ не видит назначения / протоколы из project-файлов | `pool_guarantees["project"]` +1 |
| ИИ не видит реальные слова пользователя из сессий | `pool_guarantees["transcript"]` +1 |
| Увеличил гарантии — eval ухудшился | `eval.py`'s K уже подстроится сам (сумма гарантий) |

## Структура пулов

```
Запрос
  ├── memory pool    — file_chunks WHERE source_type = 'memory'   (project-specific набор, weight=1.0)
  ├── project pool   — file_chunks WHERE source_type != 'memory'  (project-specific набор, weight=0.5–1.0)
  └── transcript pool — embeddings JOIN messages WHERE role='user'

_merge_pools(pools, k, guarantees):
  Phase 1: гарантированные слоты (top-N из каждого пула, N = guarantees[pool])
  Phase 2: дозаполнение до K лучшими оставшимися
  Phase 3: сортировка по score, result[:k]
```

## Архитектура скриптов

Логика — в `archive-lib/archive_lib/`, per-project обёртки — в `<project>/archive/scripts/`.

| Модуль (`archive_lib/...`) | Назначение |
|---|---|
| `ingest.py` | Claude Code JSONL → SQLite (FTS5, artifacts, вложения, сабагенты — опционально) |
| `ingest_copilot.py` | GitHub Copilot chatSessions → та же БД, `source='copilot'` |
| `embed.py` | Векторизация транскриптов (инкрементально) |
| `index_files.py` | Векторизация project/memory файлов; список и веса — в `config.file_registry` |
| `query.py` | Поиск: FTS / semantic / hybrid / artifacts / multi-archive; `_merge_pools`, `_load_model` (с кэшем) |
| `restore.py` | Полное восстановление сессии в markdown, с gap/pause-маркерами |
| `eval.py` | Golden-set оценка: `--mode semantic` (default, `config.golden_set`) — Recall@K + MRR через реальный `search_semantic`; `--mode full-scan` (`config.full_scan_golden_set`) — baseline vs `full_scan.py` покрытие, без Recall@K (нет единственно верного чанка) |
| `full_scan.py` | Систематический обход кандидатов-сессий без top-k cap (альтернатива для подсчёта/агрегации и хронологического синтеза) — `enumerate_candidates`, `chronological_batches`, `restore_batch` |
| `ai_time.py` | Время с ИИ по дням, по всем проектам из `registry.KNOWN_ARCHIVES` |
| `update.py` | Полный цикл: ingest → copilot → embed → index_files |

`archive-lib/tests/test_query.py` — 32 unit-теста на синтетических фикстурах, без БД/модели, общие для всех проектов.
