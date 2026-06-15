"""Block E gate — orchestrate classifier + citation extractor + MCP audit.

Public entrypoint:
  ``run_gate(user_message, response_text, sender_lid, agent_name,
             turn_started_at_iso) -> str``

Decision tree:

  1. If ``response_text`` is empty / pure metadata → pass-through.
  2. If the user's question is NOT DIH-flavored (classifier returns False)
     → pass-through.
  3. Extract ``[slug]`` citations from response.
     a. If citations exist → strip inline markers, append "Source(s):"
        footer, return cleaned text.
     b. If NO citations:
        - Check if the response already self-refuses ("not in the brain",
          "I don't have information", etc.). If yes → pass-through (bot
          refused honestly).
        - Else → REFUSE: replace response with the failure message.
  4. Belt + braces: even when citations exist, verify the bot actually
     called a retrieval MCP tool this turn. If citations exist but no
     retrieval call happened → REFUSE (the slug was hallucinated).

All decisions are logged at INFO so a single ``journalctl | grep block_e``
shows the gate's reasoning. The env var ``HERMES_BLOCK_E_ENABLED`` (default
on) lets us instantly disable the gate if anything misbehaves; the env var
``HERMES_BLOCK_E_DRY_RUN`` (off by default) logs decisions without changing
the response — useful for monitoring before enforcement.
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

# Operational-response patterns. The bot's reply is the result of calling
# a hermes-side admin tool (add_to_allowlist / approve_user /
# record_pending_user / list_pending_users / save_to_scope) — there's no
# DIH knowledge claim to cite, the tool execution IS the action. Skip the
# cite-or-refuse gate so the user sees the tool's confirmation text rather
# than the gate's "couldn't find anything to cite" refusal.
_OPERATIONAL_PATTERNS = [
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

    is_dih = is_dih_question(user_message or "")
    if not is_dih:
        logger.info(
            "block_e: pass-through (not DIH) | sender=%s len=%d",
            sender_lid, len(response_text),
        )
        return response_text

    # Operational response — bot just executed an admin/save tool call,
    # the response IS the tool's confirmation text. Nothing to cite, the
    # action is the value. Pass through before checking for citations.
    if _looks_operational(response_text):
        logger.info(
            "block_e: pass-through (operational tool result) | sender=%s",
            sender_lid,
        )
        return response_text

    citations = extract_citations(response_text)

    # If no citations and bot already self-refused, let it through.
    if not citations and _looks_like_self_refusal(response_text):
        logger.info(
            "block_e: pass-through (self-refusal, no citation) | sender=%s",
            sender_lid,
        )
        return response_text

    # If no citations at all → REFUSE.
    if not citations:
        calls_made: list[str] = []
        if agent_name:
            calls_made = list_calls(agent_name, lookback_seconds)
        logger.warning(
            "block_e: REFUSE (DIH question, no citation) | sender=%s "
            "agent=%s calls=%s",
            sender_lid, agent_name, calls_made,
        )
        if _dry_run():
            return response_text + "\n\n[gate would refuse: DIH question, no citation]"
        return REFUSAL_MESSAGE

    # We have citations. Verify retrieval call actually happened so we
    # catch the "bot hallucinated a plausible slug" case.
    retrieval_called = True  # fail open: if audit can't run, trust the citations
    if agent_name:
        retrieval_called = called_retrieval_tool(agent_name, lookback_seconds)
    if not retrieval_called:
        logger.warning(
            "block_e: REFUSE (citations present but no retrieval MCP call) | "
            "sender=%s agent=%s citations=%d",
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
        "block_e: PASS | sender=%s agent=%s citations=%d",
        sender_lid, agent_name, len(citations),
    )
    if footer:
        return cleaned.rstrip() + "\n\n" + footer
    return cleaned
