"""Block E gate — engage citation enforcement only when retrieval is in play.

Public entrypoint:
  ``run_gate(user_message, response_text, sender_lid, agent_name,
             lookback_seconds) -> str``

v0.2 decision tree (action-based, not topic-based):

  1. If ``response_text`` is empty / pure metadata → pass-through.
  2. Determine if THIS TURN involved retrieval:
       a. ``user_requested_brain`` — classifier matched an explicit
          brain-invocation pattern (the user said "search the brain",
          "look it up in the docs", etc.)
       b. ``retrieval_was_called`` — mcp_audit shows the bot actually
          called a gbrain retrieval tool (query / search / get_page / …)
       If neither is True → pass-through (this turn is general
       assistance — the gate is not relevant).
  3. If the bot's reply is an operational tool-result confirmation
     ("phone added to allowlist", etc.) → pass-through.
  4. Extract ``[slug]`` citations from the response.
       a. Citations present → strip inline markers, append a clean
          "Source(s):" footer.
          - If ``user_requested_brain`` but NO retrieval call was made,
            the citation is hallucinated → REFUSE.
       b. No citations + self-refusal phrase ("not in the brain",
          "I don't have information") → pass-through (honest non-answer).
       c. No citations and no self-refusal → REFUSE.

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
from agent.block_e.classifier import is_dih_question
from agent.block_e.mcp_audit import called_retrieval_tool, list_calls

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

    # v0.2: two signals can engage the gate.
    user_requested_brain = is_dih_question(user_message or "")
    retrieval_was_called = False
    if agent_name:
        retrieval_was_called = called_retrieval_tool(agent_name, lookback_seconds)

    if not user_requested_brain and not retrieval_was_called:
        logger.info(
            "block_e: pass-through (no brain invocation, no retrieval call) "
            "| sender=%s len=%d",
            sender_lid, len(response_text),
        )
        return response_text

    # Operational tool-result confirmation — pass-through regardless of
    # which engagement signal fired.
    if _looks_operational(response_text):
        logger.info(
            "block_e: pass-through (operational tool result) | sender=%s",
            sender_lid,
        )
        return response_text

    citations = extract_citations(response_text)

    # No citations + honest self-refusal → pass-through.
    if not citations and _looks_like_self_refusal(response_text):
        logger.info(
            "block_e: pass-through (self-refusal, no citation) | sender=%s "
            "user_brain=%s retrieval=%s",
            sender_lid, user_requested_brain, retrieval_was_called,
        )
        return response_text

    # No citations at all → REFUSE.
    if not citations:
        calls_made: list[str] = []
        if agent_name:
            calls_made = list_calls(agent_name, lookback_seconds)
        logger.warning(
            "block_e: REFUSE (no citation) | sender=%s agent=%s "
            "user_brain=%s retrieval=%s calls=%s",
            sender_lid, agent_name, user_requested_brain,
            retrieval_was_called, calls_made,
        )
        if _dry_run():
            return response_text + "\n\n[gate would refuse: no citation]"
        return REFUSAL_MESSAGE

    # Citations present. If the USER asked the brain but the model never
    # actually retrieved, citations are hallucinated → REFUSE. (If the
    # model retrieved without being asked, the citations are legitimate.)
    if user_requested_brain and not retrieval_was_called:
        logger.warning(
            "block_e: REFUSE (user asked brain, citations without retrieval) "
            "| sender=%s agent=%s citations=%d",
            sender_lid, agent_name, len(citations),
        )
        if _dry_run():
            return (
                response_text
                + "\n\n[gate would refuse: citations without retrieval call]"
            )
        return REFUSAL_MESSAGE

    # PASS — strip inline brackets, append References footer.
    cleaned = strip_inline_citations(response_text)
    footer = format_references_footer(citations)
    logger.info(
        "block_e: PASS | sender=%s agent=%s citations=%d "
        "user_brain=%s retrieval=%s",
        sender_lid, agent_name, len(citations),
        user_requested_brain, retrieval_was_called,
    )
    if footer:
        return cleaned.rstrip() + "\n\n" + footer
    return cleaned
