"""DIH-question classifier for Block E.

Two-stage:
  1. Keyword fast-path — cheap, covers obvious cases (mentions of DIH,
     internal projects, finance/leadership/super_admin/ceos scopes,
     possessive "our/we/the company" patterns).
  2. Sonnet 4.6 judge — invoked only when the keyword path is ambiguous,
     i.e., none of the strong-DIH keywords matched AND the question is
     long enough to plausibly be DIH-flavored (>= 6 words).

Returns ``True`` when the message should be treated as a DIH question
(triggers the cite-or-refuse gate), ``False`` otherwise (gate is a
pass-through; the bot can answer as a general assistant).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Strong indicators — if any match, classify as DIH without invoking the model.
_STRONG_DIH_PATTERNS = [
    r"\bdih\b",
    r"\bdigital innovation holdings\b",
    r"\bdigital infrastructure holdings\b",
    # Possessive "our X" / "our N" — covers the broad set of business-context
    # nouns the original short curated list missed (legal counsels, suppliers,
    # advisors, deal names, portfolio etc.). Catches "who are our X", "what's
    # our X", "where is our X" without needing every noun.
    r"\bour\s+(company|team|policy|policies|process|processes|finance|revenue|customers?|clients?|board|strategy|plan|legal|counsel|counsels|advisor|advisors|supplier|suppliers|vendor|vendors|deal|deals|investor|investors|lender|lenders|partner|partners|firm|firms|fund|funds|portfolio|tower|towers|capital|q[1-4]|quarter|pipeline|pipelines|risk|risks|milestone|milestones|brief|briefs)\b",
    r"\bthe company('s)?\b",
    # Scope names from scopes.yaml
    r"\b(super[_ -]?admin|leadership|ceos?)\b",
    # DIH-specific nouns
    r"\b(runbook|deployment|on[- ]?call|incident|sla)\b",
    r"\b(revenue|p&l|margin|opex|capex|burn|runway)\b",
    r"\b(roadmap|okr|kpi|hiring plan|all[- ]?hands)\b",
    # Named projects + people from the DIH ref docs (questionnaires + pipelines).
    r"\bproject\s+(mesec|ampere|heirloom|signal|ukraine)\b",
    r"\b(iyad|ghalia|tareq|will|basit|raja|philippe|omar)\b",
]

# Soft indicators — bump confidence but don't trigger alone.
_SOFT_DIH_PATTERNS = [
    r"\b(internal|in[- ]?house|proprietary|confidential)\b",
    r"\b(remember|recall|capture|note|brief)\b",
    r"\b(meeting|policy|process|decision|plan|review)\b",
    r"\bwhat (did|do) (we|i|the team|harris|shahzaib|the ceo)\b",
]


def _looks_strong_dih(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _STRONG_DIH_PATTERNS)


def _looks_soft_dih(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _SOFT_DIH_PATTERNS)


def _is_short_chitchat(text: str) -> bool:
    """Skip the classifier on short greetings / acknowledgements."""
    stripped = text.strip()
    if not stripped:
        return True
    words = stripped.split()
    if len(words) < 3:
        return True
    chitchat = {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "yes", "no", "sure", "got it"}
    return stripped.lower() in chitchat


# Sonnet 4.6 system prompt — terse binary classifier.
_SONNET_SYSTEM = (
    "You classify whether a user message to an internal assistant is asking "
    "about company-specific knowledge (people, projects, policies, finance, "
    "decisions, internal documents) versus general/public knowledge or "
    "small talk. Reply with EXACTLY one word: YES (company-specific) or NO "
    "(general/chitchat/implementation-question). No punctuation, no "
    "explanation."
)


def _ask_sonnet(text: str, timeout_s: float = 8.0) -> Optional[bool]:
    """Ask Sonnet 4.6 (via the Claude-CLI subscription path) to classify.
    Returns ``True``/``False`` on a confident yes/no; ``None`` on any error
    (caller treats None as conservative ``True`` — assume DIH, gate kicks
    in, refusal-on-fail is the safer default).
    """
    try:
        from agent.auxiliary_client import ClaudeCliAuxiliaryClient
    except Exception as exc:
        logger.warning("block_e.classifier: aux client import failed: %s", exc)
        return None

    try:
        client = ClaudeCliAuxiliaryClient(model="claude-sonnet-4-6")
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": _SONNET_SYSTEM},
                {"role": "user", "content": text.strip()[:1500]},
            ],
            max_tokens=4,
            temperature=0.0,
            timeout=timeout_s,
        )
        # OpenAI-shaped response from the shim
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return None
        content = getattr(choices[0].message, "content", "") or ""
        verdict = content.strip().upper().split(None, 1)[0] if content else ""
        if verdict.startswith("YES"):
            return True
        if verdict.startswith("NO"):
            return False
        logger.info("block_e.classifier: Sonnet returned ambiguous %r — fail open as DIH", verdict)
        return None
    except Exception as exc:
        logger.warning("block_e.classifier: Sonnet call failed: %s", exc)
        return None


def is_dih_question(text: str) -> bool:
    """Decide whether the gate should engage for this user message.

    Pipeline:
      1. Short chitchat → False (gate off, bot can chat freely)
      2. Strong keyword hit → True (gate on)
      3. No strong + no soft + short message → False
      4. Otherwise → ask Sonnet; treat ``None`` (model failure) as True
         (fail closed: when in doubt, require citations)
    """
    if not text:
        return False
    text = text.strip()

    if _is_short_chitchat(text):
        return False

    if _looks_strong_dih(text):
        logger.info("block_e.classifier: strong-keyword DIH match")
        return True

    # Short-circuit only the ultra-short cases (≤3 words after chitchat). Any
    # 4-word+ question goes to Sonnet — that's what it's for. The previous
    # `< 6` threshold skipped genuine DIH questions like "who are our legal
    # counsels?" (5 words) when the curated keyword list missed them.
    if not _looks_soft_dih(text) and len(text.split()) <= 3:
        return False

    # Allow disabling the Sonnet judge for cost/latency tuning via env var.
    if os.environ.get("HERMES_BLOCK_E_SONNET", "1") not in ("1", "true", "yes"):
        return _looks_soft_dih(text)

    verdict = _ask_sonnet(text)
    if verdict is None:
        # Fail closed — protect against soft fabrication when the classifier
        # itself is uncertain. Pilot users see one extra "Not in the brain"
        # rather than a hallucinated DIH claim.
        logger.info("block_e.classifier: Sonnet failure → fail closed to DIH=True")
        return True
    return verdict
