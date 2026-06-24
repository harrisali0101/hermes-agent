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

# Slug-shape citations — lowercase kebab-case, e.g.
# `[attach-2026-06-17-13861000-505035058v9-project-mesec-buyer-b322db]`.
# Kept for backward compat + system-emitted slugs from gbrain tools.
_SLUG_CITATION_RE = re.compile(
    r"\[(?P<slug>\.?[a-z][a-z0-9\-./]{1,118}[a-z0-9])(?::(?P<section>[a-z0-9\-]{1,60}))?\]",
    re.IGNORECASE,
)

# Human-readable TITLE citations — e.g.
# `[v9 Buyer Closing Checklist — 17 June]`,
# `[Mandate Letter execution version — 18 June 18:58 UTC]`.
# STATUS_QUERY.md instructs the model to cite by page title for chat-
# friendly output. To distinguish citations from incidental brackets
# (`[done]`, `[42]`, `[really nice]`) we REQUIRE at least one
# date/version/time anchor inside the title text.
_TITLE_CITATION_RE = re.compile(r"\[(?P<title>[^\[\]\n]{10,200})\]")
_TITLE_ANCHOR_RE = re.compile(
    # Month name (English short or long)
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b"
    # ISO date YYYY-MM-DD
    r"|\b\d{4}-\d{2}-\d{2}\b"
    # Version token (v9, v0.2.1, v2024-06)
    r"|\bv\d+(?:[\.\-]\d+)*\b"
    # Time HH:MM (with optional UTC/GMT/PKT/EST/PST/IST)
    r"|\b\d{1,2}:\d{2}\b",
    re.IGNORECASE,
)

# Common false positives in brackets. Lowercase compare.
_FALSE_POSITIVES = {
    "done", "ok", "okay", "x", " ", "tbd", "wip", "n/a", "na", "tbc",
    "redacted", "sic", "citation needed", "see above", "above",
    "yes", "no", "y", "n", "..", "...", "etc", "etc.",
    # WhatsApp formatting markers
    "image", "video", "audio", "file", "document",
}


def _is_slug_shape(text: str) -> bool:
    """True when the citation key looks like a technical slug (lowercase
    kebab-case with optional digits / dots / slashes)."""
    if not text:
        return False
    if any(ch.isspace() for ch in text):
        return False
    return bool(re.fullmatch(r"\.?[a-z][a-z0-9\-./]+", text))


def extract_citations(text: str) -> List[Tuple[str, str]]:
    """Return list of ``(citation_key, section_or_empty)`` tuples in
    source order, deduplicated. The citation key is either a slug
    (lowercase kebab-case) OR a human-readable title that contains
    a date/version/time anchor.
    """
    if not text:
        return []
    seen: set[str] = set()
    out: List[Tuple[str, str]] = []
    # Single pass: match every [...] block, classify each.
    for m in re.finditer(r"\[(?P<body>[^\[\]\n]{1,200})\](?::(?P<section>[a-z0-9\-]{1,60}))?", text, re.IGNORECASE):
        body = (m.group("body") or "").strip()
        section = (m.group("section") or "").strip().lower()
        if not body:
            continue
        key_lower = body.lower()
        if key_lower in _FALSE_POSITIVES:
            continue
        if body.replace("-", "").replace(" ", "").isdigit():
            continue
        # Accept if either: slug-shape OR title with a date/version anchor.
        is_slug = _is_slug_shape(body)
        has_anchor = bool(_TITLE_ANCHOR_RE.search(body))
        if not (is_slug or has_anchor):
            continue
        dedupe_key = key_lower if is_slug else body  # titles keep case for display
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        out.append((body if not is_slug else key_lower, section))
    return out


def strip_inline_citations(text: str) -> str:
    """Remove citation markers from the response body so the user sees
    clean prose. Recognises both slug-shape and title-shape citations
    (same rules as ``extract_citations``).
    """
    if not text:
        return text

    def _replace(m: re.Match) -> str:
        body = (m.group("body") or "").strip()
        if not body:
            return m.group(0)
        if body.lower() in _FALSE_POSITIVES:
            return m.group(0)
        if body.replace("-", "").replace(" ", "").isdigit():
            return m.group(0)
        is_slug = _is_slug_shape(body)
        has_anchor = bool(_TITLE_ANCHOR_RE.search(body))
        if not (is_slug or has_anchor):
            return m.group(0)
        return ""

    cleaned = re.sub(
        r"\[(?P<body>[^\[\]\n]{1,200})\](?::(?P<section>[a-z0-9\-]{1,60}))?",
        _replace,
        text,
        flags=re.IGNORECASE,
    )
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

    Two citation shapes are supported (see ``extract_citations``):
    - Technical slug (lowercase kebab-case) → run through ``prettify_slug``
      to produce a human label, then show the canonical slug in brackets
      as the audit anchor.
    - Human-readable title (already prettified by the model per STATUS_QUERY)
      → pass through as-is; no further prettification.

    Returns an empty string when there are no citations.
    """
    if not citations:
        return ""
    parts: List[str] = []
    for citation_key, section in citations:
        if _is_slug_shape(citation_key):
            pretty = prettify_slug(citation_key)
            entry = f"{pretty} [{citation_key}]" if pretty != citation_key else citation_key
        else:
            # Human-readable title — already in its display form.
            entry = citation_key
        if section:
            entry = f"{entry} (§{section})"
        parts.append(entry)
    if len(parts) == 1:
        return f"Source: {parts[0]}"
    return "Sources:\n  - " + "\n  - ".join(parts)
