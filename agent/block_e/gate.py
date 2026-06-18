"""Block E gate — engage citation enforcement only on EXPLICIT user invocation.

Public entrypoint:
  ``run_gate(user_message, response_text, sender_lid, agent_name,
             lookback_seconds) -> str``

v0.3 decision tree (user-intent only — retrieval signal removed):

  1. If ``response_text`` is empty / pure metadata → pass-through.
  2. If the user message is an operational/diagnostic request
     ("list_pages", "approve user X", "run a health check",
     "raw response of get_page") → pass-through. Tool output is the
     expected reply; no citation required.
  3. If the user did NOT explicitly invoke the brain → pass-through.
     "Explicit invocation" means phrases like "search the brain",
     "look it up in the docs/notes", "from gbrain", "in the memory",
     etc. — the patterns in classifier._EXPLICIT_BRAIN_PATTERNS.
     General questions, DIH-flavored conversational questions,
     greetings, drafting, math, code, web — all pass through.
  4. The user DID explicitly invoke the brain — gate engages:
       a. If the bot's reply is an operational tool-result
          confirmation → pass-through (no citation expected).
       b. Extract ``[slug]`` citations from the response.
          - Citations present → strip inline markers, append a clean
            "Source(s):" footer.
          - No citations + self-refusal phrase ("not in the brain",
            "I don't have information") → pass-through (honest
            non-answer).
          - No citations and no self-refusal → REFUSE.

Why retrieval-call was removed as an engagement signal: even with the
operator persona rule "don't auto-retrieve", the model occasionally
drifts and calls gbrain on a DIH-flavored conversational question.
The user didn't ask for brain content, so refusing the answer for
lack of citations is wrong — it surprises the user and looks broken.
If the user wants brain-grounded answers they explicitly ask; if they
don't, they get an unmoderated answer. Simpler mental model.

Knobs:
  - ``HERMES_BLOCK_E_ENABLED`` (default on) — emergency off-switch.
  - ``HERMES_BLOCK_E_DRY_RUN`` (off by default) — log the decision but
    return the original response unchanged, useful for monitoring before
    enforcement.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from agent.block_e.citations import (
    extract_citations,
    format_references_footer,
    strip_inline_citations,
)
from agent.block_e.classifier import (
    _looks_explicit_brain_invocation,
    _looks_operational_request,
)

logger = logging.getLogger(__name__)

# The user-visible refusal message when the gate trips. Conservative wording
# — never claims the answer doesn't exist anywhere, just that the bot
# couldn't find a grounded answer.
REFUSAL_MESSAGE = (
    "I tried to check the brain and couldn't find anything I can cite on "
    "this. I'd rather say that than make something up. Want me to capture "
    "it as a new note, or search a different way?"
)

# Common self-refusal patterns the bot may emit unprompted. If we see one
# of these AND no citation, we still pass the response through unchanged —
# the bot is being honest about not knowing.
_SELF_REFUSAL_PATTERNS = [
    r"\bnot in the brain\b",
    r"\bdon'?t have (any )?information\b",
    r"\bdon'?t have anything\b",
    r"\bcouldn'?t find (anything|any|that)\b",
    r"\bnothing (in the brain|on this|relevant)\b",
    r"\bno (matching|relevant) (page|note|content)\b",
]

# Operational/diagnostic response shapes. The bot's reply is the result
# of calling a hermes-side admin tool (allowlist / approve / save) OR
# raw diagnostic output (list_pages / get_page / doctor / introspection).
# There's no DIH knowledge claim to cite, the tool execution IS the
# action. Skip the cite-or-refuse gate so the user sees the tool's
# output rather than the gate's "couldn't find anything to cite" refusal.
_OPERATIONAL_PATTERNS = [
    # Admin / save tool confirmations
    r"phone added to (?:gateway )?allowlist",
    r"phone\s+\d+\s+was already on the allowlist",
    r"\buser onboarded\b",
    r"\bis already in scopes\.yaml\b",
    r"\brecorded pending user\b",
    r"\bpending user .+ already queued\b",
    r"\d+\s+pending user\(s\)",
    r"\bno pending users\b",
    r"\bis restricted to super_admins\b",
    r"\bsaved to (the )?(general|leadership|finance|super_admin|ceos) scope\b",
    r"\bphone format invalid\b",
    r"\btarget_lid format invalid\b",
    r"\brole '\w+' is not defined\b",
    # Diagnostic / introspection output shapes — bot reporting raw tool
    # output to the operator, NOT making knowledge claims.
    r"\b(list_pages?|get_page|put_page|find_experts|list_skills)\s*\([^)]*\)\s*(returned|response|result|raw)",
    r"\b(list_pages?|get_page|put_page|find_experts|list_skills)\s*(returned|response|result|raw)",
    r"\b(raw|verbatim)\s+(response|output|error|result)\s*:",
    r"\b(tool\s+call|tool\s+output|tool\s+response|mcp\s+(call|output|response))\s*:",
    r"\boauth\s+client\s*:",
    r"\b(role|identity|scope|source\s*id)\s+(running|used)\s+(as|under|for)\b",
    r"\b(running|operating)\s+(as|under)\s+(staff|director|finance_analyst|ceo|super_admin)[- ]agent\b",
    r"\bpage\s+not\s+found\b",
    r"\berror\s*:\s*page\s+not\s+found\b",
]


def _looks_operational(text: str) -> bool:
    t = (text or "")
    return any(re.search(p, t, re.IGNORECASE) for p in _OPERATIONAL_PATTERNS)


def _looks_like_self_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in _SELF_REFUSAL_PATTERNS)


def _gate_enabled() -> bool:
    return os.environ.get("HERMES_BLOCK_E_ENABLED", "1") in ("1", "true", "yes")


def _dry_run() -> bool:
    return os.environ.get("HERMES_BLOCK_E_DRY_RUN", "0") in ("1", "true", "yes")


def run_gate(
    user_message: str,
    response_text: str,
    sender_lid: str,
    agent_name: Optional[str],
    lookback_seconds: int = 120,
) -> str:
    """Apply Block E. Return the possibly-rewritten response.

    The ``lookback_seconds`` window bounds the mcp_request_log query when
    checking whether the bot actually called a retrieval tool for this
    turn. 120s comfortably covers a long turn without leaking across user
    sessions (hermes processes turns serially per user, so cross-turn
    contamination would only matter if a user fired two questions within
    the window — for the pilot this is acceptable; revisit if false-pass
    rate becomes a problem).
    """
    if not _gate_enabled():
        return response_text or ""
    if not response_text or not response_text.strip():
        return response_text or ""

    # v0.3: pass-through for operational / diagnostic user requests.
    # The user asked the bot to run an admin or debug tool ("list_pages
    # with source=leadership", "approve user X", "raw response of
    # get_page"). The bot's reply IS tool output, not a knowledge claim.
    if _looks_operational_request(user_message or ""):
        logger.info(
            "block_e: pass-through (operational/diagnostic request) "
            "| sender=%s len=%d",
            sender_lid, len(response_text),
        )
        return response_text

    # v0.3: the gate engages ONLY on explicit user invocation of the
    # brain. Retrieval-call signal removed — if the model auto-retrieves
    # on a conversational question (persona drift), we don't punish the
    # user; we let the answer through uncited rather than refuse it.
    if not _looks_explicit_brain_invocation(user_message or ""):
        logger.info(
            "block_e: pass-through (user did not explicitly invoke brain) "
            "| sender=%s len=%d",
            sender_lid, len(response_text),
        )
        return response_text

    # From here, the user explicitly asked the brain. Gate is engaged —
    # the reply must either cite or honestly say "Not in the brain."

    if _looks_operational(response_text):
        logger.info(
            "block_e: pass-through (operational tool result) | sender=%s",
            sender_lid,
        )
        return response_text

    citations = extract_citations(response_text)

    if not citations and _looks_like_self_refusal(response_text):
        logger.info(
            "block_e: pass-through (self-refusal, no citation) | sender=%s",
            sender_lid,
        )
        return response_text

    if not citations:
        logger.warning(
            "block_e: REFUSE (explicit brain invocation, no citation) "
            "| sender=%s agent=%s",
            sender_lid, agent_name,
        )
        if _dry_run():
            return response_text + "\n\n[gate would refuse: no citation]"
        return REFUSAL_MESSAGE

    cleaned = strip_inline_citations(response_text)
    footer = format_references_footer(citations)
    logger.info(
        "block_e: PASS | sender=%s agent=%s citations=%d",
        sender_lid, agent_name, len(citations),
    )
    if footer:
        return cleaned.rstrip() + "\n\n" + footer
    return cleaned
