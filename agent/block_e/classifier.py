"""Block E intent classifier.

Returns ``True`` when the gate should engage (require citations or a
clean refusal). Engagement is now narrowly scoped to **explicit brain
invocation by the user** — phrases like "search the brain", "look it up
in gbrain", "what's in the notes about X", "from the docs", "hermes,
find …", etc. Topic-based DIH detection has been removed: a question
that merely mentions internal entities does NOT engage the gate unless
the user asked the brain to be searched, OR the model itself called a
retrieval tool (the gate checks that separately via ``mcp_audit``).

Why the rewrite: the previous "is this question DIH-flavored?"
classifier produced false-positives — meta/status requests ("how are
you", "run a health check"), or chats merely mentioning "the ceo" or
"leadership" got refused because the bot had nothing to cite for what
isn't a knowledge question. The new model: brain is one tool among
many; engage citation enforcement only when retrieval is in play.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Operational/diagnostic-request patterns. Admin/save AND debug/diagnostic
# requests are never knowledge questions — the bot's reply is a tool-result
# confirmation or raw tool output. Refusing for lack of citations would
# break legitimate operator flows. Detected early so they short-circuit.
_OPERATIONAL_REQUEST_PATTERNS = [
    # Admin / save / onboarding tools
    r"\b(allowlist|whitelist|allow[- ]?list)\b",
    r"\b(approve|onboard)\s+(this|the|new|user|@?\d|@?[A-Za-z])",
    r"\b(add|remove)\s+(this|the|a|new)?\s*(user|phone|number|person|member)",
    r"\b(save|capture|put|remember|record)\s+(this|the|that|it|us|me|him|her|them)",
    r"\bset\s+\w+\s+(as|to)\s+(staff|finance_analyst|director|ceo|super_admin)",
    r"^\s*/(approve|allowlist|onboard|pending|save|deny)\b",
    r"\bpending\s+users?\b",
    r"\b(any|who|whose)\s+(new|pending|waiting)\s+(user|onboard)",
    r"\bhermes,?\s+(allowlist|approve|save|onboard|add|record|remove|deny)",
    r"\bphone\s+\d{6,}\b",
    # Diagnostic / introspection requests — user is asking the bot to RUN
    # a specific tool and report back, or to surface internal state. The
    # reply is raw tool output, not a DIH knowledge claim.
    r"\b(list_pages?|get_page|put_page|find_experts|list_skills|"
    r"list_pending_users|add_to_allowlist|approve_user|record_pending_user|"
    r"save_to_scope|capture\b)",
    r"\b(raw|verbatim)\s+(response|output|error|message|result)\b",
    r"\b(error\s+)?(message|output|response)\s+verbatim\b",
    r"\bwhich\s+(oauth\s*client|client|role|identity|scope|credentials?|bearer|source\s*id)\b",
    r"\bwhat\s+(oauth\s*client|role|identity|scope|credentials?|bearer|source\s*id)\s+(are|am)\b",
    r"\brun\s+(a\s+|the\s+|three\s+|some\s+|a\s+few\s+)?(check|checks|test|tests|diagnostic|diagnostics|debug|trace|probe)\b",
    r"\b(report|show|tell)\s+(me\s+)?(the\s+)?(raw\s+)?(response|output|error|result|exact)\b",
    r"\b(debug|doctor|trace|introspect)\b",
    r"\bsource\s*=\s*(general|leadership|finance|super_admin|ceos)\b",
    r"\bslug\s*=\s*\S+",
    r"\bexact[- ]slug\b",
]

# Explicit-brain-invocation patterns — the ONLY user-side trigger that
# engages the gate now. The user is asking the bot to consult the brain /
# gbrain / internal notes / docs / knowledge base. If none of these match
# AND the model didn't call a retrieval tool (checked in gate.py), the
# gate is a pass-through.
_BRAIN_TARGET = (
    r"(brain|gbrain|notes?|docs?|documents?|memory|knowledge\s*base|"
    r"data\s*store|repository|repo|archive|pages?|"
    r"(internal|company|dih)\s+(notes|docs|knowledge|files|content))"
)
_BRAIN_VERBS = (
    r"(search|searches|searching|look\s+up|looking\s+up|look\s+in|looking\s+in|"
    r"find|finds|finding|check|checks|checking|query|queries|querying|"
    r"retrieve|retrieves|retrieving|fetch|fetches|fetching|pull(?:\s+up)?|"
    r"dig(?:\s+up)?|tell\s+me|show\s+me|give\s+me|grab|get)"
)

_EXPLICIT_BRAIN_PATTERNS = [
    # "search/look up/find … (in|on|inside|from) the brain/notes/docs"
    rf"\b{_BRAIN_VERBS}\b[^.?!\n]{{0,80}}?\b(in|on|inside|from|out\s+of|within)\s+(the\s+|our\s+|gbrain|dih'?s?\s+)?{_BRAIN_TARGET}\b",
    # "search/check/look in the brain (for X)"
    rf"\b{_BRAIN_VERBS}\b[^.?!\n]{{0,80}}?\bthe\s+{_BRAIN_TARGET}\b",
    # "search memory" / "check brain" / "look up notes" — verb directly
    # followed by target, no article. Catches the common chat shorthand.
    rf"\b{_BRAIN_VERBS}\s+{_BRAIN_TARGET}\b",
    # "from/in (the|our|dih's) brain/memory/notes/docs" — standalone phrase.
    # Requires an article to disambiguate "from memory" (English idiom
    # for "from recollection") from "from the memory" (gbrain reference).
    rf"\b(from|in|on|inside|within)\s+(the|our|dih'?s?)\s+{_BRAIN_TARGET}\b",
    # Same standalone phrase, but for distinctive identifiers that don't
    # need an article ("from gbrain", "in the knowledge base", etc.).
    r"\b(from|in|on|inside|within)\s+(gbrain|the\s+knowledge\s*base|the\s+data\s*store)\b",
    # "hermes, search/find/look …" — addressing the bot with a retrieval verb
    r"\bhermes,?\s+(search|find|look(\s+up)?|check|query|fetch|pull(\s+up)?|retrieve|dig(\s+up)?)\b",
    # "is X in the brain", "do we have X (in the brain|stored|on file)"
    rf"\bis\s+(there|it|that|this)\s+(anything\s+)?(in|on)\s+(the\s+|our\s+|gbrain|dih'?s?\s+)?{_BRAIN_TARGET}\b",
    rf"\bdo\s+we\s+have\s+[^.?!\n]{{0,60}}?(in|on|stored\s+in|saved\s+in)\s+(the\s+|our\s+|gbrain|dih'?s?\s+)?{_BRAIN_TARGET}\b",
    # "what does the brain (have|say|know) about X"
    rf"\bwhat\s+(does|do)\s+(the\s+|our\s+|gbrain|dih'?s?\s+)?{_BRAIN_TARGET}\s+(have|say|know|contain|hold)\b",
    # Explicit slug references
    r"\b(slug|page|source)\s*[:=]\s*\S+",
    r"\bget(\s+me)?\s+(the\s+)?(page|slug|source)\s+\S+",
]


def _looks_operational_request(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t, re.IGNORECASE) for p in _OPERATIONAL_REQUEST_PATTERNS)


def _looks_explicit_brain_invocation(text: str) -> bool:
    """User is asking the bot to consult the brain / gbrain / internal notes.

    The gate engages on this signal so the bot must either return a cited
    answer or refuse honestly. Topic-based detection ("did they mention
    'ceo'?") is intentionally NOT used here — that's what produced the
    false-positives Block E v0.1 suffered from.
    """
    t = (text or "").lower()
    return any(re.search(p, t, re.IGNORECASE) for p in _EXPLICIT_BRAIN_PATTERNS)


def is_dih_question(text: str) -> bool:
    """Return True when the user explicitly asked the brain to be consulted.

    NOTE: kept under the old name so the gate import doesn't churn, but
    semantics changed in v0.2: this no longer detects "DIH-flavored
    topic" — it detects "explicit retrieval intent". See the module
    docstring.
    """
    if not text:
        return False
    text = text.strip()

    if _looks_operational_request(text):
        logger.info("block_e.classifier: operational request → pass-through")
        return False

    if _looks_explicit_brain_invocation(text):
        logger.info("block_e.classifier: explicit brain invocation → gate engaged")
        return True

    return False
