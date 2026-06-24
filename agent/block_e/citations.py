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

v0.2 (2026-06-24): footer now prettifies known slug shapes (``attach-…``,
``email-…``, dated cohorts) into emoji-tagged human strings while keeping
the canonical slug as the audit anchor. No gbrain round-trip — pure
deterministic transforms — so it's safe to use on every turn.
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


_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)$")
_HEX_TAIL_RE = re.compile(r"-[0-9a-f]{4,12}$")
_VERSION_RE = re.compile(r"(?<![a-zA-Z])v(\d+)(?![a-zA-Z])")
_NUMERIC_ID_RE = re.compile(r"^\d{6,}-\d{6,}(?:v\d+)?-")


def prettify_slug(slug: str) -> str:
    """Render a gbrain slug as a human-friendly string while keeping the
    canonical slug retrievable. Pure deterministic — no gbrain calls.

    Examples
    --------
    >>> prettify_slug("attach-2026-06-17-13861000-505035058v9-project-mesec-buyer-b322db")
    '📎 2026-06-17 · project-mesec-buyer (v9)'
    >>> prettify_slug("email-2026-06-24-mesec-facilities-agreement-issues-list-417b53")
    '📧 2026-06-24 · mesec-facilities-agreement-issues-list'
    >>> prettify_slug("q3-financials")
    'q3-financials'
    """
    if not slug:
        return slug
    s = slug

    # Detect kind prefix → emoji
    emoji = ""
    if s.startswith("attach-"):
        emoji = "📎"
        s = s[len("attach-"):]
    elif s.startswith("email-"):
        emoji = "📧"
        s = s[len("email-"):]

    # Pull out leading YYYY-MM-DD date if present
    date_str = ""
    m = _DATE_RE.match(s)
    if m:
        date_str = m.group(1)
        s = m.group(2)

    # Strip random hex hash tails (4-12 hex chars at end). Common in
    # attach/email slugs as a short identifier.
    s = _HEX_TAIL_RE.sub("", s)

    # Strip leading numeric IDs like `13861000-505035058v9-` (mail-msg-id +
    # version markers) — they're noise to the human reader.
    version_match = _VERSION_RE.search(s)
    version_suffix = f" (v{version_match.group(1)})" if version_match else ""
    s = _NUMERIC_ID_RE.sub("", s)
    # Also drop a bare trailing version token if it's at the end.
    s = re.sub(r"-v\d+$", "", s)

    # Compose
    parts = []
    if emoji:
        parts.append(emoji)
    if date_str:
        parts.append(date_str + " ·")
    if s:
        parts.append(s + version_suffix)
    return " ".join(parts).strip() or slug


def format_references_footer(citations: List[Tuple[str, str]]) -> str:
    """Render the "Sources:" footer the bot appends after a passing turn.

    Each entry shows the prettified slug followed by the canonical slug in
    parens so the audit anchor stays intact. Returns an empty string when
    there are no citations.

    Example output (one citation):
        Source: 📎 2026-06-17 · project-mesec-buyer (v9) [attach-2026-06-17-…-b322db]
    """
    if not citations:
        return ""
    parts: List[str] = []
    for slug, section in citations:
        pretty = prettify_slug(slug)
        if pretty != slug:
            entry = f"{pretty} [{slug}]"
        else:
            entry = slug
        if section:
            entry = f"{entry} (§{section})"
        parts.append(entry)
    if len(parts) == 1:
        return f"Source: {parts[0]}"
    return "Sources:\n  - " + "\n  - ".join(parts)
