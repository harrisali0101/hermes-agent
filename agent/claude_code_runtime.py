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


# ── Reusable helpers (also imported by agent/auxiliary_client.py) ──────────


def _flatten_messages_to_prompt(messages: List[Dict[str, Any]]) -> tuple[str, str]:
    """Collapse OpenAI-shaped messages into (system_prompt, prompt_text).

    Tool messages fold in as plain text with a label so a one-shot subprocess
    can read the full prior trajectory. Used by both the main runtime's first-
    turn spawn and by the auxiliary client (one-shot per call).
    """
    system_parts: list[str] = []
    body_parts: list[str] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, list):
            collected: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") in {"text", "input_text"}:
                        text = part.get("text") or part.get("content") or ""
                        if text:
                            collected.append(str(text))
                    elif part.get("type") in {"image", "image_url", "input_image"}:
                        collected.append("[image attached]")
                elif isinstance(part, str):
                    collected.append(part)
            content_text = "\n".join(collected).strip()
        elif content is None:
            content_text = ""
        else:
            content_text = str(content)

        if role == "system":
            if content_text:
                system_parts.append(content_text)
            continue
        if role == "user":
            body_parts.append(f"User: {content_text}")
        elif role == "assistant":
            if not content_text and msg.get("tool_calls"):
                names = []
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        fn = (tc.get("function") or {}).get("name")
                        if fn:
                            names.append(str(fn))
                if names:
                    content_text = f"(called tools: {', '.join(names)})"
            body_parts.append(f"Assistant: {content_text}")
        elif role == "tool":
            tool_name = msg.get("name") or "tool"
            body_parts.append(f"Tool result ({tool_name}): {content_text}")
        else:
            body_parts.append(content_text)

    system_prompt = "\n\n".join(p for p in system_parts if p).strip()
    user_prompt = "\n\n".join(p for p in body_parts if p).strip()
    return system_prompt, user_prompt


def _parse_claude_json_output(stdout: str) -> Dict[str, Any]:
    """Parse the JSON object produced by ``claude -p --output-format json``.

    The binary emits exactly one JSON object on success; on error the object
    carries ``is_error: true`` and the failure text in ``result`` or
    ``error``. Trailing whitespace / stray non-JSON banner lines tolerated.
    """
    text = (stdout or "").strip()
    if not text:
        return {"is_error": True, "result": "", "raw": ""}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"is_error": True, "result": text[:2000], "raw": text}


def _extract_system_prompt(
    messages: List[Dict[str, Any]],
    *,
    cwd: Optional[str] = None,
) -> Optional[str]:
    """Build a single ``--system-prompt`` string from all available sources.

    Conversation_loop dispatches the claude_code_cli runtime BEFORE hermes'
    own ``_restore_or_build_system_prompt`` block runs, so any caller-staged
    system_message is in ``messages`` but the SOUL.md + AGENTS.md + cwd
    context-file injection happens later and we'd miss it. We mirror the
    same lookup hermes performs (``load_soul_md`` + ``build_context_files_prompt``)
    here so the persona reaches claude on the first spawn.

    Precedence: any ``role: system`` messages already in ``messages`` win
    (these are the caller's explicit overrides — gateway hooks, --system-message
    flag), followed by the SOUL.md identity slot and the project context
    files (AGENTS.md > CLAUDE.md > .cursorrules; first match) from ``cwd``.
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

    # Load SOUL.md + project context the same way hermes' own
    # build_system_prompt_parts does. Best-effort: if any of these helpers
    # fail (missing files, import problems) we degrade gracefully and
    # let claude run with whatever the caller put in messages.
    try:
        from agent.prompt_builder import (
            build_context_files_prompt,
            load_soul_md,
        )

        soul = load_soul_md()
        if soul:
            parts.append(soul.strip())

        project_ctx = build_context_files_prompt(cwd=cwd, skip_soul=True)
        if project_ctx:
            parts.append(project_ctx.strip())
    except Exception:
        logger.debug("claude-cli persona load fell back to messages-only", exc_info=True)

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


def _wrap_with_verified_sender(agent, user_text: str) -> str:
    """Prefix ``user_text`` with a structured ``<verified_sender/>`` marker
    when the gateway stamped sender info onto ``agent``.

    Hermes' gateway sets ``agent._user_id`` / ``agent.platform`` /
    ``agent._user_name`` from the verified transport identity (e.g. WhatsApp
    phone number). The claude CLI subprocess can't see those fields any other
    way — without this wrap, the model has nothing to anchor identity on and
    falls back to runtime/account metadata or message-text claims (the
    2026-06-15 leak). The persona stack (SOUL.md / AGENTS.md / ACCESS_POLICY.md)
    is taught to read this tag as the ONLY identity ground truth and map the
    id to a tier.

    Returns ``user_text`` unchanged when no verified sender is present (CLI
    runs, plugin tests, etc.) — the persona's "no verified sender = None tier"
    rule then applies.
    """
    user_id = str(getattr(agent, "_user_id", None) or "").strip()
    user_id_alt = str(getattr(agent, "_user_id_alt", None) or "").strip()
    platform = str(getattr(agent, "platform", None) or "").strip()
    user_name = str(getattr(agent, "_user_name", None) or "").strip()
    logger.info(
        "verified_sender wrap: platform=%r user_id=%r user_id_alt=%r name=%r",
        platform, user_id, user_id_alt, user_name,
    )
    if not user_id or not platform or platform == "cli":
        return user_text

    def _esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
        )

    attrs = f'id="{_esc(user_id)}" platform="{_esc(platform)}"'
    if user_name:
        attrs += f' name="{_esc(user_name)}"'
    return f"<verified_sender {attrs}/>\n{user_text}"


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


# ── Claude session_id persistence (restart-survival) ───────────────────────


def _claude_session_state_path(agent) -> Optional[str]:
    """Return the path to the on-disk file that holds claude's session_id for
    a given Hermes session, or None if we cannot determine one.

    Mapping: ``HERMES_HOME/claude-sessions/<hermes_session_id>.json``.

    Used so that when the hermes service restarts, the next claude subprocess
    spawn can call ``--resume <id>`` instead of starting from scratch and
    losing conversation context that hermes' own session DB still has.
    """
    sid = getattr(agent, "session_id", None)
    if not sid or not isinstance(sid, str):
        return None
    try:
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
    except Exception:
        return None
    # Sanitize the session id: only allow safe filename chars.
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in sid)[:128]
    if not safe:
        return None
    state_dir = home / "claude-sessions"
    return str(state_dir / f"{safe}.json")


def _load_resume_session_id(agent) -> Optional[str]:
    """Load a previously-saved claude session_id for this hermes session.

    Returns the session_id string to pass to ``claude --resume``, or None if
    we don't have one (fresh chat, missing state file, hermes session_id
    unknown).
    """
    path = _claude_session_state_path(agent)
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        val = data.get("claude_session_id")
        if isinstance(val, str) and val.strip():
            return val.strip()
    except Exception as exc:
        logger.debug("claude-cli resume state load failed: %s", exc)
    return None


def _save_claude_session_id(agent, claude_session_id: str) -> None:
    """Persist claude's session_id under HERMES_HOME so it survives restarts.

    Called after each turn's result frame arrives with a session_id. Cheap
    (small JSON write) and idempotent — claude keeps the same session_id for
    the lifetime of a subprocess, so subsequent saves are no-ops in terms of
    content but keep the mtime fresh.
    """
    if not claude_session_id or not isinstance(claude_session_id, str):
        return
    path = _claude_session_state_path(agent)
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Mode 700 dir so other VM users can't enumerate sessions.
        try:
            os.chmod(os.path.dirname(path), 0o700)
        except OSError:
            pass
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump({"claude_session_id": claude_session_id}, fh)
        os.replace(tmp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except Exception as exc:
        logger.debug("claude-cli resume state save failed: %s", exc)


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

    # If we have a previously-saved claude session_id for this Hermes
    # session, pass it via --resume so claude restores the conversation
    # state from its own local cache. This is the restart-survival path:
    # when hermes.service is restarted, the old subprocess dies but the
    # session_id file persists, so the next spawn picks up where we left
    # off instead of starting a brand-new claude conversation.
    resume_session_id = _load_resume_session_id(agent)
    if resume_session_id:
        logger.info(
            "claude-cli will resume claude session_id=%s for hermes session=%s",
            resume_session_id,
            getattr(agent, "session_id", "<unknown>"),
        )

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
        resume_session_id=resume_session_id,
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

    cwd = getattr(agent, "session_cwd", None) or os.getcwd()
    system_prompt = _extract_system_prompt(messages, cwd=cwd)
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
    # Prefix with gateway-verified sender so the model can anchor identity on
    # ACCESS_POLICY.md tiers instead of runtime/account metadata or message
    # text claims (see _wrap_with_verified_sender docstring).
    user_text = _wrap_with_verified_sender(agent, user_text)

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

    # Persist claude's session_id so the NEXT spawn (after a restart, crash,
    # or session-retire) can call --resume <id> and restore the conversation
    # from claude's own local state cache. Cheap idempotent write.
    if turn.session_id and not turn.interrupted and turn.error is None:
        _save_claude_session_id(agent, turn.session_id)

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
