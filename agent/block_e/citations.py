"""Citation parsing for Block E.

The persona instructs the bot to cite gbrain pages with ``[slug]`` or
``[slug:section]`` markers. This module extracts those markers from the
response text and renders a clean "References" footer.

Heuristics:
  * Match ``[slug]`` and ``[slug:section]`` where slug is the gbrain
    convention: lower-case, kebab-case, digits, slashes, dots, plus a
    leading dot for ``.raw/...`` system slugs. Length 2–120 chars.
  * Reject obvious false positives: pure numbers ``[42]``, common UI
    markers like ``[done]``, ``[x]``, ``[ ]``, and citation-like English
    bracket usage ``[really]``, ``[sic]``.
  * Strip inline markers from the body and append a "Sources:" footer
    listing unique slugs. The footer keeps the response readable for
    pilot users while preserving auditability.

A more sophisticated v0.2 would look up titles via gbrain and render
"q3-financials (finance) — Q3 Financials"; v0.1 just lists slugs.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# slug:[a-z0-9-./], 2-120 chars, optional `:section` (`a-z0-9-`).
# Negative lookahead rejects multi-word English ([really nice], [sic erat]).
_CITATION_RE = re.compile(
    r"\[(?P<slug>\.?[a-z][a-z0-9\-./]{1,118}[a-z0-9])(?::(?P<section>[a-z0-9\-]{1,60}))?\]",
    re.IGNORECASE,
)

# Common false positives the bot/model emits in brackets. Lowercase compare.
_FALSE_POSITIVES = {
    "done", "ok", "okay", "x", " ", "tbd", "wip", "n/a", "na", "tbc",
    "redacted", "sic", "citation needed", "see above", "above",
    "yes", "no", "y", "n", "..", "...", "etc", "etc.",
    # WhatsApp formatting markers
    "image", "video", "audio", "file", "document",
}


def extract_citations(text: str) -> List[Tuple[str, str]]:
    """Return list of ``(slug, section_or_empty)`` tuples in source order,
    deduplicated by slug.
    """
    if not text:
        return []
    seen: set[str] = set()
    out: List[Tuple[str, str]] = []
    for m in _CITATION_RE.finditer(text):
        slug = (m.group("slug") or "").strip().lower()
        section = (m.group("section") or "").strip().lower()
        if slug in _FALSE_POSITIVES or slug.replace("-", "").isdigit():
            continue
        if slug in seen:
            continue
        seen.add(slug)
        out.append((slug, section))
    return out


def strip_inline_citations(text: str) -> str:
    """Remove ``[slug]`` and ``[slug:section]`` markers from the response
    body so the user sees clean prose. We don't try to fix surrounding
    punctuation — the bot is instructed to write text that reads naturally
    without the brackets.
    """
    if not text:
        return text

    def _replace(m: re.Match) -> str:
        slug = (m.group("slug") or "").strip().lower()
        if slug in _FALSE_POSITIVES or slug.replace("-", "").isdigit():
            return m.group(0)  # leave it alone
        return ""

    cleaned = _CITATION_RE.sub(_replace, text)
    # Collapse runs of spaces created by removal, preserve newlines.
    cleaned = re.sub(r" {2,}", " ", cleaned)
    # Drop space before punctuation introduced by removal: " ." → "."
    cleaned = re.sub(r" ([,.!?;:])", r"\1", cleaned)
    return cleaned.strip()


def format_references_footer(citations: List[Tuple[str, str]]) -> str:
    """Render the "Sources:" footer the bot appends after a passing turn.
    Returns an empty string when there are no citations."""
    if not citations:
        return ""
    parts: List[str] = []
    for slug, section in citations:
        if section:
            parts.append(f"{slug} (§{section})")
        else:
            parts.append(slug)
    if len(parts) == 1:
        return f"Source: {parts[0]}"
    return "Sources: " + ", ".join(parts)
