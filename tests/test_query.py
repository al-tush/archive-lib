#!/usr/bin/env python3
"""
Unit tests for archive_lib.query / archive_lib.index_files logic.

Run: python3 tests/test_query.py [-v]   (from archive-lib/)

No DB or ML model needed — all tests use synthetic data. A subset
(TestGoldenSetRegressions, TestProjectPoolTradeoff, TestSanitizeFtsQuery,
TestLooksLikeRagEcho) reproduce real score/query shapes behind production
golden-set regressions, with content replaced by anonymized stand-ins that
preserve the same structural properties (see individual test docstrings).
TEST_GUARANTEES matches one project's tuned production values. Projects
supply their own pool_guarantees via config.py; this file tests the generic
_merge_pools algorithm, not any one project's specific configuration.

Covers three layers:
  1. _merge_pools — pool guarantee & Phase-2 fill logic  (query.py)
  2. Score helpers — _time_decay, _cosine               (query.py)
  3. normalize_frontmatter — YAML stripping              (index_files.py)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from archive_lib.query import (
    _merge_pools, _time_decay, _cosine,
    _sanitize_fts_query, _expand_cyrillic, _looks_like_rag_echo,
)
from archive_lib.index_files import normalize_frontmatter

# Matches one project's production pool_guarantees (sum=6) — the regression
# tests below encode real score profiles tuned against these exact numbers.
TEST_GUARANTEES = {"memory": 3, "project": 1, "transcript": 2}
DEFAULT_K = sum(TEST_GUARANTEES.values())  # 6


def _merge(pools, k):
    return _merge_pools(pools, k=k, guarantees=TEST_GUARANTEES)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _t(tid: str, score: float, text: str = "") -> dict:
    """Synthetic transcript item."""
    return {"message_id": tid, "text_content": text or f"transcript:{tid}", "path": "", "score": score}


def _f(cid: int, score: float, path: str = "", text: str = "") -> dict:
    """Synthetic file chunk item."""
    return {"chunk_id": cid, "text_content": text or f"chunk:{cid}", "path": path or f"file_{cid}.md", "score": score}


def _pools(mem_scores, proj_scores, tr_scores):
    """Build named pools from score lists."""
    return {
        "memory":     [_f(i,       s, path=f"mem_{i}.md")    for i, s in enumerate(mem_scores)],
        "project":    [_f(100 + i, s, path=f"proj_{i}.md")   for i, s in enumerate(proj_scores)],
        "transcript": [_t(f"t{i:03d}", s)                     for i, s in enumerate(tr_scores)],
    }


# ── Core _merge_pools behaviour ───────────────────────────────────────────────

class TestMergePoolsBasics(unittest.TestCase):

    def test_result_sorted_descending(self):
        """Output is always sorted highest-score first."""
        p = _pools([0.3, 0.1], [0.8, 0.5], [0.6, 0.4])
        result = _merge(p, k=5)
        scores = [r["score"] for r in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_result_length_capped_at_k(self):
        """Never returns more than K items."""
        p = _pools([0.9, 0.8, 0.7], [0.6, 0.5], [0.4, 0.3, 0.2])
        for k in (3, 5, 7):
            with self.subTest(k=k):
                self.assertLessEqual(len(_merge(p, k=k)), k)

    def test_k_larger_than_total_items(self):
        """When K exceeds total pool items, all items are returned."""
        p = _pools([0.9], [0.8], [0.7])
        self.assertEqual(len(_merge(p, k=20)), 3)

    def test_empty_pool_graceful(self):
        """Empty pools don't crash; other pools still contribute."""
        p = _pools([0.8, 0.7], [], [0.6, 0.5])
        result = _merge(p, k=5)
        self.assertGreater(len(result), 0)
        self.assertLessEqual(len(result), 5)

    def test_all_pools_empty(self):
        """All-empty pools return empty list."""
        self.assertEqual(_merge({"memory": [], "project": [], "transcript": []}, k=5), [])

    def test_deduplication_across_pools(self):
        """Same chunk_id in multiple pools appears only once."""
        p = {
            "memory":     [_f(42, 0.9)],
            "project":    [_f(42, 0.5)],   # same chunk_id as memory item
            "transcript": [_t("t000", 0.7)],
        }
        result = _merge(p, k=5)
        cids = [r["chunk_id"] for r in result if "chunk_id" in r]
        self.assertEqual(cids.count(42), 1)


# ── Guarantee semantics ───────────────────────────────────────────────────────

class TestGuaranteedSlots(unittest.TestCase):
    """
    A pool's guaranteed slots ensure items appear even if their score is low.
    This prevents the "diary mass effect": hundreds of diary chunks outscoring
    the handful of memory files that carry structured knowledge.
    """

    def test_low_score_memory_item_included(self):
        """
        memory[1] with score=0.10 appears in K=5 result even though
        all transcript/project items score much higher.
        """
        p = _pools(
            mem_scores=[0.95, 0.10],   # memory[1] is the low-scorer
            proj_scores=[0.80],
            tr_scores=[0.99, 0.90],
        )
        result = _merge(p, k=5)
        scores_in = {r["score"] for r in result}
        self.assertIn(0.10, scores_in,
                      "memory[1] must appear despite low score (guaranteed slot)")

    def test_phase2_fills_remaining_slots_by_score(self):
        """When guarantees < K, Phase 2 picks best remaining items regardless of pool."""
        # TEST_GUARANTEES = {memory:3, project:1, transcript:2} → sum=6
        # K=9 → Phase 2 adds 3 more from the leftover pool items
        p = _pools(
            mem_scores=[0.50, 0.40, 0.30, 0.01, 0.01],   # mem[3..4] are leftovers
            proj_scores=[0.60, 0.55, 0.01],                # proj[1..2] are leftovers
            tr_scores=[0.70, 0.65, 0.30, 0.20],           # tr[2..3] are leftovers
        )
        result = _merge(p, k=9)
        self.assertEqual(len(result), 9)
        scores = [r["score"] for r in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_project_second_slot_not_guaranteed(self):
        """
        project pool has 1 guaranteed slot (TEST_GUARANTEES["project"]=1).
        project[1] is NOT guaranteed and stays out when K=6 is already filled by Phase 1.
        """
        p = {
            "memory":     [_f(1, 0.90, path="mem0.md"), _f(2, 0.60, path="mem1.md"), _f(3, 0.22, path="mem2.md")],
            "project":    [_f(10, 0.80, path="proj0.md"), _f(11, 0.75, path="proj1.md")],
            "transcript": [_t("t0", 0.95), _t("t1", 0.85)],
        }
        result_k6 = _merge(p, k=6)
        result_k8 = _merge(p, k=8)

        proj1_in_k6 = any(r.get("chunk_id") == 11 for r in result_k6)
        proj1_in_k8 = any(r.get("chunk_id") == 11 for r in result_k8)
        mem2_in_k6  = any(r.get("chunk_id") ==  3 for r in result_k6)

        self.assertFalse(proj1_in_k6,
                         "proj[1] not guaranteed and Phase 2 full at K=6 → excluded")
        self.assertTrue(proj1_in_k8,
                        "proj[1] enters via Phase 2 when K=8 > sum(guarantees)=6")
        self.assertTrue(mem2_in_k6,
                        "mem[2] IS guaranteed (score=0.22) and must appear at K=6")


# ── Regression tests for specific golden-set failures ────────────────────────

class TestGoldenSetRegressions(unittest.TestCase):
    """
    Reproduce the exact score profiles that caused golden-set misses in
    production tuning of one project. Each test is named after the failing
    query description (content anonymized). These catch regressions when a
    project's pool_guarantees is changed.
    """

    def test_low_score_memory_note_hit_at_k5(self):
        """
        A low-scoring memory-pool note (score 0.422, memory[1]) that a real
        golden-set query relied on finding. With TEST_GUARANTEES["memory"]=3,
        it is guaranteed and reaches the result.
        """
        note = _f(99, 0.422, path="contact_broken_note.md", text="Contact with person A was broken off")
        p = {
            "memory":     [_f(1, 0.436, path="support_pillars.md"), note],
            "project":    [_f(10, 0.525, path="diary/2026_05.md")],
            "transcript": [_t("t0", 0.539), _t("t1", 0.539)],
        }
        result = _merge(p, k=5)
        found = any(r.get("chunk_id") == 99 for r in result)
        self.assertTrue(found, "contact_broken_note.md must be in top-5")

    def test_keyword_memory_note_hit_at_k5(self):
        """
        A memory-pool note whose distinguishing keyword only appears there
        (memory[0], score 0.497). With guaranteed memory slots, it appears
        even when transcripts score higher.
        """
        note = _f(5, 0.497, path="condition_x_notes.md", text="keywordx details")
        p = {
            "memory":     [note, _f(6, 0.350, path="other_mem.md")],
            "project":    [_f(10, 0.440, path="diary/2026_05.md")],
            "transcript": [_t("t0", 0.497), _t("t1", 0.497)],
        }
        result = _merge(p, k=5)
        haystack = " ".join(r.get("text_content", "") for r in result).lower()
        self.assertIn("keywordx", haystack, "condition_x_notes.md must appear in top-5")


# ── Configuration sanity ──────────────────────────────────────────────────────

class TestPoolGuaranteesConfig(unittest.TestCase):
    """
    Invariants every project's config.pool_guarantees must satisfy, checked
    here against the TEST_GUARANTEES fixture.
    """

    def test_sum_does_not_exceed_default_k(self):
        """
        If sum(guarantees) > K, Phase 1 produces more than K items.
        result[:k] then silently drops the lowest-scoring guaranteed item —
        defeating the guarantee. eval.py's --k defaults to
        sum(config.pool_guarantees.values()) specifically to keep this true
        by construction; this test documents the invariant for the fixture.
        """
        total = sum(TEST_GUARANTEES.values())
        self.assertLessEqual(
            total, DEFAULT_K,
            f"TEST_GUARANTEES sums to {total} > DEFAULT_K={DEFAULT_K}."
        )

    def test_all_known_pool_names_present(self):
        self.assertEqual(set(TEST_GUARANTEES.keys()), {"memory", "project", "transcript"})

    def test_all_guarantees_positive(self):
        for name, n in TEST_GUARANTEES.items():
            with self.subTest(pool=name):
                self.assertGreater(n, 0, f"Pool '{name}' guarantee must be ≥ 1")

    def test_memory_guarantee_gte_2(self):
        """
        Memory files are sparse but carry the highest-priority knowledge.
        They need ≥ 2 guaranteed slots so that the second-most-relevant
        memory file (e.g. contact_broken_note.md) survives.
        """
        self.assertGreaterEqual(
            TEST_GUARANTEES["memory"], 2,
            "Memory pool needs ≥ 2 guaranteed slots (sparse but high-value corpus)"
        )


# ── Score helpers ─────────────────────────────────────────────────────────────

class TestScoreHelpers(unittest.TestCase):
    """Tests for _cosine and _time_decay — the two score factors we tune."""

    def test_cosine_identical_normalized_vectors(self):
        import numpy as np
        v = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        self.assertAlmostEqual(_cosine(v, v), 1.0, places=5)

    def test_cosine_orthogonal(self):
        import numpy as np
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(_cosine(a, b), 0.0, places=5)

    def test_time_decay_no_timestamp(self):
        """Missing timestamp → no decay (factor = 1.0)."""
        self.assertEqual(_time_decay(None), 1.0)
        self.assertEqual(_time_decay(""), 1.0)

    def test_time_decay_today(self):
        """Today → almost no decay."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        self.assertAlmostEqual(_time_decay(now), 1.0, places=3)

    def test_time_decay_half_life(self):
        """180 days ago → factor ≈ 0.5 (TIME_DECAY_HALFLIFE = 180)."""
        from datetime import datetime, timezone, timedelta
        past = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
        self.assertAlmostEqual(_time_decay(past), 0.5, places=2)

    def test_time_decay_monotone(self):
        """Older timestamps always produce lower factors."""
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        d30  = (now - timedelta(days=30)).isoformat()
        d90  = (now - timedelta(days=90)).isoformat()
        d365 = (now - timedelta(days=365)).isoformat()
        self.assertGreater(_time_decay(d30), _time_decay(d90))
        self.assertGreater(_time_decay(d90), _time_decay(d365))

    def test_time_decay_bad_format_graceful(self):
        """Unparseable timestamp → no decay, no crash."""
        self.assertEqual(_time_decay("not-a-date"), 1.0)


# ── normalize_frontmatter ─────────────────────────────────────────────────────

class TestNormalizeFrontmatter(unittest.TestCase):
    """
    Tests for index_files.normalize_frontmatter.

    This function is critical: stripping frontmatter entirely (old approach)
    dropped the 'name' value used as matching text for memory queries, causing
    Recall@5 to fall from 66.67% → 60.78%. The normalize approach preserves
    name/description as plain text while removing structural YAML keys.
    """

    def test_no_frontmatter_unchanged(self):
        text = "# Header\n\nBody content."
        self.assertEqual(normalize_frontmatter(text), text)

    def test_name_preserved_as_plain_text(self):
        """'name:' value becomes a plain text line (no YAML key prefix)."""
        text = "---\nname: Не подсовывать мелкие правки в этот файл\n---\n\nBody."
        result = normalize_frontmatter(text)
        self.assertIn("Не подсовывать мелкие правки в этот файл", result)
        self.assertNotIn("name:", result)
        self.assertNotIn("---", result)

    def test_description_preserved(self):
        text = "---\ndescription: Правило про мелкие правки в PR\n---\n\nBody."
        result = normalize_frontmatter(text)
        self.assertIn("Правило про мелкие правки в PR", result)

    def test_structural_keys_removed(self):
        """metadata:, type:, node_type: are structural — must be stripped."""
        text = "---\nname: Test\nmetadata:\n  type: feedback\n---\n\nBody."
        result = normalize_frontmatter(text)
        self.assertNotIn("metadata", result)
        self.assertNotIn("type:", result)
        self.assertIn("Test", result)

    def test_body_always_preserved(self):
        text = "---\nname: X\n---\n\nBody content here."
        result = normalize_frontmatter(text)
        self.assertIn("Body content here", result)

    def test_empty_value_line_skipped(self):
        """A key with no value (e.g. 'name:') should not produce an empty line."""
        text = "---\nname:\ndescription: Real\n---\n\nBody."
        result = normalize_frontmatter(text)
        self.assertIn("Real", result)
        lines = [l for l in result.splitlines() if l.strip()]
        self.assertTrue(all(l.strip() for l in lines))

    def test_unclosed_frontmatter_unchanged(self):
        """File that starts with --- but has no closing --- is returned as-is."""
        text = "---\nname: Test\n\nBody without closing dashes."
        self.assertEqual(normalize_frontmatter(text), text)

    def test_melkie_zadachi_survives(self):
        """
        Regression: 'мелкие' was only in the frontmatter name of
        small_edit_note.md. Stripping frontmatter caused a golden-set query
        containing that word to MISS. normalize_frontmatter must preserve it.
        """
        text = (
            "---\n"
            "name: Не предлагать мелкие правки без запроса\n"
            "description: Правило про формат PR\n"
            "metadata:\n"
            "  type: feedback\n"
            "---\n\n"
            "Body about PR conventions."
        )
        result = normalize_frontmatter(text)
        self.assertIn("мелкие", result, "'мелкие' must survive from frontmatter name")


# ── Project pool trade-off (prescriptions-note regression) ────────────────────

class TestProjectPoolTradeoff(unittest.TestCase):
    """
    Documents the prescriptions_note.md regression when project guarantee
    changed 2→1. Both states are tested explicitly so the trade-off is visible
    and any future change to a project's pool_guarantees is caught before
    running the full eval.
    """

    def _nazn_profile(self):
        """Score profile for a golden-set query about a medication side effect."""
        return {
            "memory": [
                _f(1, 0.543, path="user_episode.md"),
                _f(2, 0.513, path="project_note_a.md"),
                _f(3, 0.480, path="other_mem.md"),
            ],
            "project": [
                _f(10, 0.620, path="diary/2026_05.md"),
                _f(11, 0.556, path="prescriptions_note.md",
                   text="prescription: drug_x 50 mg"),
            ],
            "transcript": [_t("t0", 0.624), _t("t1", 0.616)],
        }

    def test_nazn_excluded_with_current_project_1(self):
        """
        Current setting: project=1. Phase 1 fills K=6 exactly
        (mem×3 + proj×1 + tr×2), leaving prescriptions_note.md (proj[1]) out.
        """
        self.assertEqual(TEST_GUARANTEES["project"], 1,
                         "This test assumes current project guarantee = 1")
        result = _merge(self._nazn_profile(), k=6)
        self.assertFalse(any(r.get("chunk_id") == 11 for r in result),
                         "prescriptions_note.md should be excluded with project=1, K=6")

    def test_nazn_recovers_with_k7(self):
        """
        If K=7, Phase 2 adds one more slot. prescriptions_note.md (score=0.556)
        wins the Phase-2 slot over remaining items.
        """
        result = _merge(self._nazn_profile(), k=7)
        self.assertTrue(any(r.get("chunk_id") == 11 for r in result),
                        "prescriptions_note.md should recover at K=7 via Phase 2")


class TestSanitizeFtsQuery(unittest.TestCase):
    """Regression tests for the 2026-07-10 FTS5 syntax-error crashes:
    a query containing a dotted latin project name plus digits, and a query
    containing a bare slash, both raised sqlite3.OperationalError: fts5:
    syntax error before the fix."""

    def test_strips_bare_slash(self):
        out = _sanitize_fts_query("какие-то планы после отказа / другие планы")
        self.assertNotIn("/", out)

    def test_strips_dot_in_latin_word(self):
        out = _sanitize_fts_query("Foo.bar устроился новая работа 2023")
        self.assertNotIn(".", out)
        # words on either side of the dot survive as separate bare terms
        self.assertIn("Foo", out)
        self.assertIn("bar", out)

    def test_leaves_words_and_digits_untouched(self):
        out = _sanitize_fts_query("тревога 2013 Иванов")
        self.assertEqual(out, "тревога 2013 Иванов")

    def test_leaves_quoted_phrase_untouched(self):
        query = 'что думал про "Foo.bar / релиз" вчера'
        out = _sanitize_fts_query(query)
        self.assertIn('"Foo.bar / релиз"', out)

    def test_preserves_star_and_parens(self):
        out = _sanitize_fts_query("погода* (дождь OR снег)")
        self.assertEqual(out, "погода* (дождь OR снег)")

    def test_sanitized_then_expanded_has_no_bare_punctuation_for_cyrillic(self):
        # Sanitize must run before _expand_cyrillic in the real call sites;
        # confirm the composition doesn't reintroduce the crash chars.
        sanitized = _sanitize_fts_query("отказа / другие планы на позицию")
        expanded = _expand_cyrillic(sanitized)
        self.assertNotIn("/", expanded)


class TestLooksLikeRagEcho(unittest.TestCase):
    """Regression tests for the 2026-07-10 self-referential noise: query.py
    finding its own prior invocations/output banners as search results."""

    def test_detects_expand_banner(self):
        text = "[expand] погода* дождь* завтра* (no results)"
        self.assertTrue(_looks_like_rag_echo(text))

    def test_detects_query_py_invocation(self):
        text = 'python3 archive/scripts/query.py "простуда вирус болею" --hybrid --limit 10'
        self.assertTrue(_looks_like_rag_echo(text))

    def test_detects_restore_py_invocation(self):
        text = "python3 archive/scripts/restore.py df71fdf8 2>&1 | head -200"
        self.assertTrue(_looks_like_rag_echo(text))

    def test_detects_no_results_footer(self):
        self.assertTrue(_looks_like_rag_echo("(no results)"))

    def test_detects_auto_update_banner(self):
        text = "[auto-update] last indexed 4215m ago — updating FTS index...\n[auto-update] done."
        self.assertTrue(_looks_like_rag_echo(text))

    def test_detects_multi_archive_banner(self):
        # A whole query.py --archives dump pasted verbatim into a message
        # (the 641bbff1 case: found as a top hit in an unrelated session).
        text = (
            "════════════════════════════════════════════════════════════\n"
            "  Archive: project-a  (/home/user/project-a/archive/transcripts.db)\n"
            "════════════════════════════════════════════════════════════\n"
            "  1. [2026-07-01 23:51 MSK] e56c2987 user  score=0.611\n"
        )
        self.assertTrue(_looks_like_rag_echo(text))

    def test_does_not_flag_ordinary_user_text(self):
        text = "Заказал доставку продуктов на завтра, обещали привезти утром."
        self.assertFalse(_looks_like_rag_echo(text))

    def test_does_not_flag_biographical_file_content(self):
        text = "Переезд в другой город в 2013 году стал поворотным моментом в карьере."
        self.assertFalse(_looks_like_rag_echo(text))

    def test_empty_text_is_false(self):
        self.assertFalse(_looks_like_rag_echo(""))

    def test_detects_eval_py_full_run(self):
        # eval.py's first print, captured whole (no tail truncation).
        text = "loading model paraphrase-multilingual-MiniLM-L12-v2 …\nmodel loaded\n\n  HIT@1     результат A — статус [K]\n"
        self.assertTrue(_looks_like_rag_echo(text))

    def test_detects_eval_py_tail_truncated_hit_lines(self):
        # 2026-07-12 case: `eval.py ... | tail -N` drops the "loading model"
        # header, leaving only HIT@/MISS status lines — head-prefix check
        # alone would miss this.
        text = "  HIT@4     тема A — через семантику [S→M]\n  MISS      тема B и тема C [S]\n"
        self.assertTrue(_looks_like_rag_echo(text))

    def test_detects_eval_py_summary_line(self):
        text = "──────────────────────────────────────────────────\nRecall@6: 82.35%  (42/51 queries)\nMRR@6:    0.604"
        self.assertTrue(_looks_like_rag_echo(text))

    def test_detects_embed_py_progress(self):
        text = "loading model …\nmodel loaded\n  64/241 (27%)\r  128/241 (53%)\r\ndone: 241 messages embedded"
        self.assertTrue(_looks_like_rag_echo(text))

    def test_does_not_flag_text_mentioning_eval_results_in_prose(self):
        # Meta-discussion about RAG quality (this very session's topic) must
        # stay searchable — only the tool's own literal report format is
        # filtered, not someone talking about recall/misses in prose.
        text = "Прогнал eval и получил довольно низкий результат, часть запросов не находится вообще."
        self.assertFalse(_looks_like_rag_echo(text))


if __name__ == "__main__":
    unittest.main(verbosity=2)
