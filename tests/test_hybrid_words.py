#!/usr/bin/env python3
"""
Unit tests for archive_lib.hybrid_words.

Run: python3 tests/test_hybrid_words.py [-v]   (from archive-lib/)

Two-tier design, driven by a full-corpus dry run against apteki's archive
(08.08.2026, see sect-lost/ИИ_самосовершенствование_проекта.md entry
2026-08-08) that found the naive "any bare script boundary" detector fires
on three unrelated things:

  1. find_hybrid_tokens() — broad detection, used for diagnostics/logging.
     Excludes two structural false positives: literal backslash-escape
     tails ("\\nДата" reading as token "nДата") and markdown anchor slugs
     (auditor'у -> auditorу once the apostrophe is stripped for the URL).
  2. normalize_hybrid_words() — narrow correction. Only inserts a space for
     a known Russian prefix (не/за/пере/...) glued onto a Latin root, which
     is the one shape where the fix reliably restores both halves as
     separate, searchable words. Homoglyph typos (бash, генеалogii,
     статeless — a wrong-script letter at a random position, not tied to a
     prefix; confirmed at least once to come from AI-authored analytical
     prose, not human typing) have the same "bare script boundary" shape but
     a random split point — a space-insert doesn't recover the intended
     word for those, so they're only ever logged, never rewritten.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from archive_lib.hybrid_words import (
    find_hybrid_tokens, find_prefix_glue_tokens, normalize_hybrid_words,
)


class TestFindHybridTokens(unittest.TestCase):
    def test_finds_cyrillic_prefix_glued_to_latin_root(self):
        text = "исключение потеряется в неawait-нутом async-колбэке"
        self.assertEqual(find_hybrid_tokens(text), ["неawait"])

    def test_finds_latin_word_embedded_without_translation(self):
        text = "неformalный контакт"
        self.assertEqual(find_hybrid_tokens(text), ["неformalный"])

    def test_finds_homoglyph_typo_too_broad_detector_still_sees_it(self):
        # Real corpus example: "генеалогии" with a wrong-script letter
        # mid-word (AI-authored prose, not human typing — see module
        # docstring). Detection sees it (useful for logging) even though
        # normalize_hybrid_words() below must NOT touch it.
        text = "глубокое расследование архитектурной генеалogii бага"
        self.assertEqual(find_hybrid_tokens(text), ["генеалogii"])

    def test_ignores_apostrophe_separated_slang(self):
        text = "передаче auditor'у скрыло доказательство, в React'е тоже"
        self.assertEqual(find_hybrid_tokens(text), [])

    def test_ignores_hyphen_separated_compound(self):
        text = "async-колбэк и bundleRelease-задача — оба чистые"
        self.assertEqual(find_hybrid_tokens(text), [])

    def test_ignores_pure_cyrillic_and_pure_latin(self):
        text = "обычный русский текст с английским словом await рядом"
        self.assertEqual(find_hybrid_tokens(text), [])

    def test_ignores_escaped_newline_tail_before_cyrillic_word(self):
        # Real corpus example: literal two-char "\n" (not an actual
        # newline) immediately followed by a Cyrillic word reads as a
        # bogus token "nДата" unless the backslash tail is excluded.
        text = "сильных/слабых сторон\\n\\nДата начала: 2026-07-18"
        self.assertEqual(find_hybrid_tokens(text), [])

    def test_ignores_escaped_tab_and_cr_tails_too(self):
        self.assertEqual(find_hybrid_tokens("итог\\tТаблица готова"), [])
        self.assertEqual(find_hybrid_tokens("вывод\\rПерезапись строки"), [])

    def test_ignores_markdown_anchor_slug_duplicate(self):
        # Real corpus example: heading "auditor'у скрыло..." generates an
        # anchor "#...-auditorу-скрыло-..." with the apostrophe stripped.
        text = ("исходного утверждения (ложноположительная находка "
                "«фабрикация»)](#2026-07-12--передаче-auditorу-скрыло-доказательство)")
        self.assertEqual(find_hybrid_tokens(text), [])

    def test_dedupes_repeated_token_keeps_first_seen_order(self):
        text = "неawait тут и неawait там, а ещё недоformalный где-то"
        self.assertEqual(find_hybrid_tokens(text), ["неawait", "недоformalный"])

    def test_empty_text(self):
        self.assertEqual(find_hybrid_tokens(""), [])


class TestFindPrefixGlueTokens(unittest.TestCase):
    def test_matches_known_prefix_plus_latin_root(self):
        text = "в неawait-нутом async-колбэке и неfavor'нутую задачу"
        self.assertEqual(find_prefix_glue_tokens(text), ["неawait", "неfavor"])

    def test_does_not_match_unlisted_single_letter_prefix(self):
        # Real corpus example: "мattermost" is a homoglyph typo of
        # "Mattermost", not a Russian prefix "м" + root — "м" isn't in the
        # known-prefix list, so this must NOT be treated as fixable.
        text = "можешь ей написать в мattermost"
        self.assertEqual(find_prefix_glue_tokens(text), [])

    def test_does_not_match_root_shorter_than_min_latin_root(self):
        # A listed prefix ("до") glued to a 2-letter Latin root — below
        # MIN_LATIN_ROOT (3), so the length gate alone should reject it
        # even though the prefix itself is on the allowlist.
        self.assertEqual(find_prefix_glue_tokens("доOK контекст"), [])


class TestNormalizeHybridWords(unittest.TestCase):
    def test_inserts_space_at_known_prefix_boundary(self):
        text = "неawait-нутом async-колбэке"
        self.assertEqual(
            normalize_hybrid_words(text), "не await-нутом async-колбэке"
        )

    def test_no_change_when_nothing_glued(self):
        text = "передаче auditor'у скрыло доказательство"
        self.assertEqual(normalize_hybrid_words(text), text)

    def test_confirmed_instance_favor(self):
        text = "Gradle запускает неfavor'нутую задачу bundleRelease"
        self.assertEqual(
            normalize_hybrid_words(text),
            "Gradle запускает не favor'нутую задачу bundleRelease",
        )

    def test_does_not_touch_homoglyph_typo(self):
        # The core regression this rewrite exists for: a mechanical
        # space-insert doesn't recover "bash" from "бash", so it must be
        # left untouched rather than "fixed" into "б ash".
        text = "# бash-скрипт эту функцию не вызывает"
        self.assertEqual(normalize_hybrid_words(text), text)

    def test_does_not_touch_mattermost_typo(self):
        text = "можешь ей написать в мattermost, представься"
        self.assertEqual(normalize_hybrid_words(text), text)

    def test_does_not_touch_escaped_newline_tail(self):
        text = "сильных/слабых сторон\\n\\nДата начала: 2026-07-18"
        self.assertEqual(normalize_hybrid_words(text), text)

    def test_does_not_touch_markdown_anchor_slug(self):
        text = "находка «фабрикация»)](#2026-07-12--передаче-auditorу-скрыло)"
        self.assertEqual(normalize_hybrid_words(text), text)

    def test_empty_and_none_safe(self):
        self.assertEqual(normalize_hybrid_words(""), "")
        self.assertEqual(normalize_hybrid_words(None), None)

    def test_warn_context_prints_both_categories_without_raising(self):
        text = "неawait тест и бash отдельно"
        try:
            out = normalize_hybrid_words(text, warn_context="test:0000")
        except Exception as e:  # pragma: no cover
            self.fail(f"normalize_hybrid_words raised with warn_context: {e}")
        self.assertEqual(out, "не await тест и бash отдельно")


if __name__ == "__main__":
    unittest.main(verbosity=2)
