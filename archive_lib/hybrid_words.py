"""Detect Cyrillic/Latin script collisions glued into a single token, and
mechanically repair the narrow subset that's safe to touch without guessing.

Originally scoped for one observed AI-generation artifact — a Russian prefix
written directly onto an untransliterated English word with no separator
(e.g. "неawait", "неformalный"). The FTS5 tokenizer this project uses
(`unicode61`) treats the whole glued run as ONE token, so the English root
becomes unsearchable on a Russian-language query and vice versa.

A full-corpus dry run against apteki's archive (08.08.2026, see
sect-lost/ИИ_самосовершенствование_проекта.md entry 2026-08-08) found the
naive "any bare script boundary" detector fires on three distinct things,
only one of which this module should touch:

  1. The real target: a short Russian prefix (не/за/пере/...) glued straight
     onto an English word — "неawait", "неfavor". Rare (a handful of
     instances), and inserting a space at the boundary genuinely restores
     both halves as separate FTS tokens.
  2. Homoglyph letter-substitution at a random position in the word —
     "генеалogii", "статeless", "мattermost", "докe" — same surface shape (a
     bare script boundary) but NOT a prefix construction, and a random
     position means a space-insert doesn't recover the intended word ("б
     ash" doesn't help you find "bash"). Initially misattributed to human
     keyboard-layout typing; confirmed on at least one instance (fич/докe in
     apteki_plus_mobile_обзор_проекта.md) to sit inside dense AI-authored
     analytical prose, not a chat quote — same generation-time phenomenon as
     category 1, just scattered instead of a fixed prefix shape. Cause not
     otherwise verified. Logged, never auto-corrected here — a per-character
     corrector is a different, not-yet-built tool.
  3. Pure false positives with no actual language mixing:
       - a literal backslash-escape tail ("\\n", "\\t", "\\r" as two literal
         characters, not a real newline/tab) immediately followed by a
         Cyrillic word — "\\nДата" reads as token "nДата".
       - markdown anchor slugs auto-generated from a heading, which strip
         apostrophes from tech-slang case endings — "auditor'у" in prose
         becomes "auditorу" inside `](#...-auditorу-...)`.

This module only touches the copy of text handed to the indexer — callers
apply it while building `text_content` / chunk text for the DB. It never
edits source files or raw JSONL transcripts, so `restore.py` still shows the
verbatim original.

Code-only, no LLM/heuristic guessing of intended words:
  - Detection (`find_hybrid_tokens`) is broad, minus the two structural false
    positives above (escape tails, anchor slugs) — useful as a diagnostic
    log even for cases we don't auto-fix.
  - Correction (`normalize_hybrid_words`) only inserts a space for tokens
    that structurally match "known short Russian prefix + Latin root of at
    least MIN_LATIN_ROOT letters" (category 1). Everything else found by
    detection is logged, not modified — a wrong "fix" that doesn't recover
    the word is worse than leaving it alone.
"""

from __future__ import annotations

import re
import sys

# Zero-width match at a bare Cyrillic->Latin or Latin->Cyrillic boundary
# (lookaround only, nothing consumed — lets us insert a space in place).
_SCRIPT_BOUNDARY_RE = re.compile(
    r"(?<=[а-яёА-ЯЁ])(?=[a-zA-Z])|(?<=[a-zA-Z])(?=[а-яёА-ЯЁ])"
)

# A maximal run of letters (either script), EXCLUDING runs that start right
# after a literal backslash — that's always either an escape-sequence tail
# ("\n", "\t", "\r" as literal text) or code/regex/LaTeX syntax ("\bfoo",
# "\alpha"), never a real word boundary worth flagging.
_LETTER_RUN_RE = re.compile(r"(?<!\\)[а-яёА-ЯЁa-zA-Z]+")

# Markdown link targets ("](#some-anchor-slug)") — headings with an
# apostrophe (auditor'у) get it stripped when the anchor slug is generated,
# so the same word appears twice in the source: once correct in prose, once
# apostrophe-less in the slug. The slug copy is a rendering artifact, not a
# new instance of the anomaly.
_MD_ANCHOR_RE = re.compile(r"\]\(#[^)\s]*\)")

# Known Russian prefix morphemes eligible for the auto-fix. Deliberately a
# closed list, not "any short Cyrillic run" — that's what keeps the fix from
# firing on homoglyph typos that happen to share the same shape (a single
# wrong-script letter: "мattermost", "бash" — "м"/"б" are not real prefixes).
_KNOWN_PREFIXES = (
    "не", "за", "по", "до", "от", "об", "во", "вы", "из", "на", "над",
    "недо", "низ", "пере", "под", "пред", "при", "про", "раз", "рас",
    "воз", "вос", "со",
)
# Longest-first, so e.g. "недо" wins over its own prefix "не" at the same
# start position — alternation takes the first alternative that matches,
# not the longest, so ordering here is load-bearing.
_PREFIX_GLUE_RE = re.compile(
    r"\b(" + "|".join(sorted(_KNOWN_PREFIXES, key=len, reverse=True)) + r")([a-zA-Z]{3,})\b",
    re.IGNORECASE,
)
MIN_LATIN_ROOT = 3  # matches the `{3,}` above; named for readability at call sites

# Instances confirmed by manual review when this module was added
# (journal entry 2026-08-08) — kept as a record of what was actually seen,
# not consulted by the code below.
CONFIRMED_INSTANCES = (
    "неawait-нутом",   # 2026-07-16 session afe639c7 — auto-fixed
    "неawait'нутых",   # 2026-07-22 session 05a63720 — auto-fixed
    "неfavor'нутую",   # 2026-07-31 session 393f968c — auto-fixed; root itself
                        # likely mistyped (favor vs flavor), not something
                        # this pass can fix
)


def _md_anchor_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _MD_ANCHOR_RE.finditer(text)]


def _in_any_span(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


def find_hybrid_tokens(text: str) -> list[str]:
    """Return distinct glued Cyrillic+Latin tokens in `text`, in first-seen
    order — minus escape-sequence tails and markdown-anchor-slug duplicates
    (see module docstring). Pure detection, no mutation; includes tokens
    this module will NOT auto-correct (homoglyph substitution typos)."""
    anchor_spans = _md_anchor_spans(text)
    tokens: list[str] = []
    seen: set[str] = set()
    for m in _LETTER_RUN_RE.finditer(text):
        if _in_any_span(m.start(), anchor_spans):
            continue
        tok = m.group(0)
        if tok not in seen and _SCRIPT_BOUNDARY_RE.search(tok):
            seen.add(tok)
            tokens.append(tok)
    return tokens


def find_prefix_glue_tokens(text: str) -> list[str]:
    """Subset of `find_hybrid_tokens` this module will actually auto-fix:
    a known Russian prefix glued straight onto a Latin root of at least
    MIN_LATIN_ROOT letters."""
    return [m.group(0) for m in _PREFIX_GLUE_RE.finditer(text)]


def normalize_hybrid_words(text: str, *, warn_context: str | None = None) -> str:
    """Insert a space after a known Russian prefix glued onto a Latin root
    (e.g. "неawait" -> "не await"), so the FTS5 `unicode61` tokenizer indexes
    the two halves as separate, searchable words. Returns `text` unchanged
    if there's nothing in that narrow category.

    Other hybrid tokens found by `find_hybrid_tokens` (homoglyph typos
    like "бash", "генеалogii") are logged when `warn_context` is given,
    but never modified — inserting a space doesn't recover the intended
    word for those, so leaving them as-is is the safer default.
    """
    if not text:
        return text

    if warn_context:
        fix_tokens = set(find_prefix_glue_tokens(text))
        for tok in find_hybrid_tokens(text):
            if tok in fix_tokens:
                continue
            print(f"  [hybrid-word: not auto-fixed] {warn_context}: {tok!r}",
                  file=sys.stderr)
        for tok in fix_tokens:
            fixed = _SCRIPT_BOUNDARY_RE.sub(" ", tok, count=1)
            print(f"  [hybrid-word: fixed] {warn_context}: {tok!r} -> {fixed!r}",
                  file=sys.stderr)

    return _PREFIX_GLUE_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}", text)
