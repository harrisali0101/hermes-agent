"""Claude Code CLI runtime — drives a long-lived ``claude --print`` subprocess
session for each Hermes session.

Each Hermes turn:
  1. Reuses the per-session ``ClaudeCodeCliSession`` (spawning on first use).
  2. Flattens any system prompt + extracts the latest user message.
  3. Sends the user message over stdin.
  4. Drains stream-json events until ``result`` arrives.
  5. Returns the same result dict shape as ``run_codex_app_server_turn`` so
     ``conversation_loop.py`` can treat the two opt-in runtimes uniformly.

Auth: claude CLI reads ``~/.claude/.credentials.json`` itself; we explicitly
strip ``ANTHROPIC_API_KEY`` from the child env so the Max subscription base
allowance is spent rather than per-token API rates.

Tool calls + MCP: claude runs its native tools (Read/Write/Bash/Edit/Glob/Grep)
inside the subprocess with ``--permission-mode bypassPermissions``. gbrain (or
any other stdio MCP server) wires in via ``--mcp-config <file>``; the file
is staged at session-spawn time from hermes config.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from agent.transports.claude_code_cli_session import (
    ClaudeCodeCliSession,
    _find_claude_binary,
)

logger = logging.getLogger(__name__)

DEFAULT_TURN_TIMEOUT_SECONDS = 600


def _extract_system_prompt(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Concatenate any system messages into a single ``--system-prompt`` arg.

    Hermes typically sends one system message at index 0, but a few flows
    inject multiple (memory context block, soul, agents.md). We join them
    with blank lines so claude sees the full instruction stack.
    """
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") in {"text", "input_text"}:
                    txt = blk.get("text") or blk.get("content") or ""
                    if txt:
                        parts.append(str(txt))
        elif isinstance(content, str) and content.strip():
            parts.append(content)
    if not parts:
        return None
    return "\n\n".join(parts).strip() or None


def _latest_user_text(messages: List[Dict[str, Any]], fallback: str) -> str:
    """Reduce the most recent user message into plain text for the wire.

    The conversation history is maintained inside the long-lived claude
    subprocess across turns (same session_id), so we only need to send the
    NEWEST user turn — not the whole history. Hermes' messages list already
    has the previous turns appended by the projector.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for blk in content:
                if isinstance(blk, dict):
                    if blk.get("type") in {"text", "input_text"}:
                        parts.append(str(blk.get("text") or blk.get("content") or ""))
                    elif blk.get("type") in {"image", "image_url", "input_image"}:
                        parts.append("[image attached]")
                elif isinstance(blk, str):
                    parts.append(blk)
            text = "\n".join(p for p in parts if p).strip()
            if text:
                return text
    return fallback or ""


def _resolve_mcp_config_path() -> Optional[str]:
    """Return a filesystem path to a ``--mcp-config`` JSON file if one is
    configured, otherwise None.

    Resolution order:
      1. ``HERMES_CLAUDE_CODE_MCP_CONFIG`` env var (operator override).
      2. ``<HERMES_HOME>/claude-mcp.json`` if present.

    Returning None means we launch claude with no MCP servers; the pilot's
    gbrain client lives in hermes' own MCP layer and is reached via hermes'
    tool dispatcher, so this is safe.
    """
    explicit = os.environ.get("HERMES_CLAUDE_CODE_MCP_CONFIG", "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit
    try:
        from hermes_constants import get_hermes_home

        candidate = str(get_hermes_home() / "claude-mcp.json")
        if os.path.isfile(candidate):
            return candidate
    except Exception:
        pass
    return None


def _ensure_session(agent) -> ClaudeCodeCliSession:
    """Lazy-instantiate one ClaudeCodeCliSession per AIAgent instance.

    Reused across turns so the conversation accumulates inside the
    subprocess (cached prefix, persistent claude session_id) — matching the
    codex_app_server lifecycle.
    """
    existing = getattr(agent, "_claude_cli_session", None)
    if existing is not None:
        return existing

    binary = _find_claude_binary()
    if binary is None:
        raise FileNotFoundError(
            "claude CLI binary not found on PATH. Install it with "
            "`npm install -g @anthropic-ai/claude-code` (operator nodes) "
            "or seed `/etc/hermes/fetch-secrets.sh` to install on first boot."
        )

    cwd = getattr(agent, "session_cwd", None) or os.getcwd()
    model = getattr(agent, "model", None) or getattr(agent, "model_name", None)
    mcp_config_path = _resolve_mcp_config_path()

    session = ClaudeCodeCliSession(
        claude_bin=binary,
        model=model,
        # We don't pre-bake the system prompt here — hermes may rebuild it
        # per turn from memory/soul context. Each run_turn() will compute
        # the system prompt from the current messages and pass it via
        # the next subprocess incarnation; for the v1 single-session
        # pilot, we accept that swapping system prompts mid-session
        # requires restarting the subprocess. The system prompt for the
        # first turn is staged at spawn.
        system_prompt=None,
        mcp_config_path=mcp_config_path,
        permission_mode=os.environ.get(
            "HERMES_CLAUDE_CODE_PERMISSION_MODE", "bypassPermissions"
        ),
        cwd=cwd,
    )
    agent._claude_cli_session = session
    return session


def _retire_session(agent) -> None:
    sess = getattr(agent, "_claude_cli_session", None)
    if sess is not None:
        try:
            sess.close()
        except Exception:
            pass
    agent._claude_cli_session = None


def run_claude_code_cli_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """Run one turn through the claude CLI subprocess session.

    Returns the same dict shape as ``run_codex_app_server_turn``:

        {
          "final_response": str,
          "messages": list,
          "api_calls": int,
          "completed": bool,
          "partial": bool,
          "error": Optional[str],
          "claude_session_id": Optional[str],
          "claude_total_cost_usd": Optional[float],
          "claude_usage": Optional[dict],
        }
    """
    # The user message is already appended to ``messages`` by the standard
    # run_conversation() pre-loop — DO NOT append it again.
    try:
        session = _ensure_session(agent)
    except FileNotFoundError as exc:
        return {
            "final_response": str(exc),
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": "claude_binary_missing",
        }

    system_prompt = _extract_system_prompt(messages)
    # If the subprocess isn't running yet AND we have a system prompt, set it
    # on the session so the spawn picks it up. If the subprocess is already
    # running with a different system prompt, the change won't take effect
    # this turn — we accept that for v1 (system prompts are usually stable
    # across a hermes session).
    if system_prompt and session._client is None:
        session._system_prompt = system_prompt

    user_text = _latest_user_text(messages, user_message)
    if not user_text.strip():
        return {
            "final_response": "Empty user prompt; nothing to send to claude CLI.",
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": "empty_prompt",
        }

    turn_timeout = float(
        os.environ.get("HERMES_CLAUDE_CODE_TURN_TIMEOUT")
        or DEFAULT_TURN_TIMEOUT_SECONDS
    )

    try:
        turn = session.run_turn(user_input=user_text, turn_timeout=turn_timeout)
    except Exception as exc:
        logger.exception("claude-cli turn failed")
        _retire_session(agent)
        return {
            "final_response": f"claude CLI turn failed: {exc}",
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
        }

    if turn.should_retire:
        logger.warning(
            "claude-cli session retired (error=%s, interrupted=%s)",
            turn.error,
            turn.interrupted,
        )
        _retire_session(agent)

    # Splice projected messages into the conversation.
    if turn.projected_messages:
        messages.extend(turn.projected_messages)

    # Skill nudge counter — bump per tool iteration, mirroring codex.
    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )

    should_review_skills = False
    if (
        getattr(agent, "_skill_nudge_interval", 0) > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in getattr(agent, "valid_tool_names", ())
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    # External memory sync — same cadence as codex path.
    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

    # Background review fork.
    if (
        turn.final_text
        and not turn.interrupted
        and (should_review_memory or should_review_skills)
    ):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=should_review_memory,
                review_skills=should_review_skills,
            )
        except Exception:
            logger.debug("background review spawn raised", exc_info=True)

    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": 1,
        "completed": not turn.interrupted and turn.error is None,
        "partial": turn.interrupted or turn.error is not None,
        "error": turn.error,
        "claude_session_id": turn.session_id or session.session_id,
        "claude_total_cost_usd": turn.total_cost_usd,
        "claude_usage": turn.usage,
    }


__all__ = ["run_claude_code_cli_turn"]
