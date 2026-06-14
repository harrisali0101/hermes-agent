"""Session adapter for the claude code CLI runtime.

Owns one long-lived ``claude --print --output-format stream-json`` subprocess
per Hermes session. Drives ``send_user`` per turn, consumes streaming events
via :class:`ClaudeCodeCliEventProjector`, and returns a clean per-turn result
that AIAgent.run_conversation() can splice into its ``messages`` list.

Lifecycle::

    session = ClaudeCodeCliSession(
        claude_bin="claude",
        model="claude-opus-4-8",
        system_prompt="...",
        mcp_config_path="/etc/hermes/claude-mcp.json",
    )
    session.ensure_started()
    result = session.run_turn(user_input="...")
    # result.final_text           → assistant text the user sees
    # result.projected_messages   → list of {role, content, ...} for messages list
    # result.tool_iterations      → count of tool_use blocks this turn
    # result.session_id           → claude's session id (also self.session_id)
    # result.total_cost_usd       → cost of this turn at API rates (informational)
    # result.should_retire        → True iff subprocess wedged / errored irrecoverably
    session.close()

The session is single-threaded from the caller's perspective. The underlying
:class:`ClaudeCodeCliClient` owns its own reader threads but exposes blocking
event queues that this adapter polls in a loop.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.transports.claude_code_cli_client import (
    ClaudeCliError,
    ClaudeCodeCliClient,
)
from agent.transports.claude_code_cli_event_projector import (
    ClaudeCodeCliEventProjector,
)

logger = logging.getLogger(__name__)


# How many tailing stderr lines to attach to a user-facing error when we don't
# have a more specific classification. Keeps the error legible.
_STDERR_TAIL_LINES = 12


_AUTH_FAILURE_HINTS = (
    "not logged in",
    "/login",
    "claude setup-token",
    "oauth",
    "unauthorized",
    "401 unauthorized",
    "invalid_grant",
    "invalid grant",
    "expired token",
    "token expired",
    "please log in",
    "please login",
    "credential",
)


def _classify_auth_failure(*parts: str) -> Optional[str]:
    """Return a user-friendly re-auth hint if any of the provided strings
    look like a claude CLI OAuth/token failure; otherwise None."""
    haystack = " ".join(p for p in parts if p).lower()
    if not haystack:
        return None
    for needle in _AUTH_FAILURE_HINTS:
        if needle in haystack:
            return (
                "Claude CLI authentication failed — your Claude Max login looks "
                "expired or missing. On the affected machine, run "
                "`claude /login` (interactive) or `claude setup-token` to mint a "
                "fresh credential and retry. Verify ANTHROPIC_API_KEY is NOT set "
                "in the environment (it overrides the Max subscription token "
                "and routes spend through API rates)."
            )
    return None


def _find_claude_binary() -> Optional[str]:
    """Locate the claude CLI binary on PATH and known install locations."""
    path = shutil.which("claude")
    if path:
        return path
    candidates = [
        "/usr/local/bin/claude",
        "/usr/bin/claude",
        os.path.expanduser("~/.npm-global/bin/claude"),
        os.path.expanduser("~/.local/bin/claude"),
    ]
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(os.path.join(appdata, "npm", "claude.cmd"))
    prog_files = os.environ.get("ProgramFiles")
    if prog_files:
        candidates.append(os.path.join(prog_files, "nodejs", "claude.cmd"))
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    return None


@dataclass
class ClaudeTurnResult:
    """Result of one user→assistant turn through the claude CLI session."""

    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None
    session_id: Optional[str] = None
    total_cost_usd: Optional[float] = None
    usage: Optional[dict] = None
    # Hint to the caller that the subprocess is wedged/errored — retire the
    # session so the next turn respawns from scratch.
    should_retire: bool = False


class ClaudeCodeCliSession:
    """One long-lived claude CLI subprocess per Hermes session.

    Not thread-safe — one caller drives it at a time, matching AIAgent's
    run_conversation() loop.
    """

    def __init__(
        self,
        *,
        claude_bin: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        mcp_config_path: Optional[str] = None,
        permission_mode: str = "bypassPermissions",
        extra_args: Optional[list[str]] = None,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        on_event: Optional[Callable[[dict], None]] = None,
        resume_session_id: Optional[str] = None,
    ) -> None:
        self._claude_bin = claude_bin or _find_claude_binary() or "claude"
        self._model = model
        self._system_prompt = system_prompt
        self._mcp_config_path = mcp_config_path
        self._permission_mode = permission_mode
        self._extra_args = list(extra_args or [])
        # If a prior claude session_id was persisted for this hermes session,
        # add --resume <id> so claude restores conversation state from its own
        # local cache. Keeps context continuity across hermes.service
        # restarts (the bug the pilot hit on 2026-06-10).
        if resume_session_id:
            self._extra_args.extend(["--resume", resume_session_id])
        self._cwd = cwd or os.getcwd()
        self._env = dict(env) if env else None
        self._on_event = on_event

        self._client: Optional[ClaudeCodeCliClient] = None
        self._projector = ClaudeCodeCliEventProjector()
        self._interrupt_event = threading.Event()
        # Pre-seed session_id from the resume hint so callers that read it
        # before the first run_turn() still see something sane.
        self.session_id: Optional[str] = resume_session_id
        self._closed = False

    # ---------- lifecycle ----------

    def ensure_started(self) -> None:
        """Spawn the subprocess. Idempotent — repeated calls are no-ops."""
        if self._client is not None and self._client.is_alive():
            return
        if self._closed:
            raise ClaudeCliError("session already closed; create a new one")
        # Fresh projector for a fresh subprocess so the tool_use_name map
        # doesn't leak between session retires.
        self._projector = ClaudeCodeCliEventProjector()
        try:
            self._client = ClaudeCodeCliClient(
                claude_bin=self._claude_bin,
                model=self._model,
                system_prompt=self._system_prompt,
                mcp_config_path=self._mcp_config_path,
                permission_mode=self._permission_mode,
                extra_args=self._extra_args,
                cwd=self._cwd,
                env=self._env,
            )
        except ClaudeCliError:
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def __enter__(self) -> "ClaudeCodeCliSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- interrupt ----------

    def request_interrupt(self) -> None:
        """Signal the active turn loop to issue an interrupt and unwind."""
        self._interrupt_event.set()

    # ---------- per-turn ----------

    def run_turn(
        self,
        user_input: str,
        *,
        turn_timeout: float = 600.0,
        poll_timeout: float = 0.25,
    ) -> ClaudeTurnResult:
        """Send a user message and block until ``result`` arrives.

        ``turn_timeout`` is the hard wall-clock ceiling per turn; on hit we
        close the subprocess and mark the session for retirement.
        """
        result = ClaudeTurnResult()
        try:
            self.ensure_started()
        except ClaudeCliError as exc:
            result.error = str(exc)
            result.should_retire = True
            return result
        assert self._client is not None

        self._interrupt_event.clear()

        try:
            self._client.send_user(user_input)
        except ClaudeCliError as exc:
            stderr_tail = "\n".join(self._client.stderr_tail(40))
            hint = _classify_auth_failure(str(exc), stderr_tail)
            result.error = hint or self._format_error_with_stderr(
                "claude CLI stdin write failed", exc
            )
            result.should_retire = True
            return result

        deadline = time.monotonic() + turn_timeout
        saw_terminal = False

        while time.monotonic() < deadline and not saw_terminal:
            if self._interrupt_event.is_set():
                self._client.send_interrupt()
                result.interrupted = True
                result.should_retire = True
                break

            if not self._client.is_alive():
                stderr_tail = "\n".join(self._client.stderr_tail(60))
                hint = _classify_auth_failure(stderr_tail)
                result.error = hint or self._format_error_with_stderr(
                    "claude CLI subprocess exited unexpectedly",
                )
                result.should_retire = True
                break

            event = self._client.take_event(timeout=poll_timeout)
            if event is None:
                continue
            if event.get("type") == "_eof":
                stderr_tail = "\n".join(self._client.stderr_tail(60))
                hint = _classify_auth_failure(stderr_tail)
                result.error = hint or self._format_error_with_stderr(
                    "claude CLI closed stdout before emitting a result"
                )
                result.should_retire = True
                break

            if self._on_event is not None:
                try:
                    self._on_event(event)
                except Exception:
                    logger.debug("claude-cli on_event callback raised", exc_info=True)

            projection = self._projector.project(event)
            if projection.messages:
                result.projected_messages.extend(projection.messages)
            if projection.is_tool_iteration:
                result.tool_iterations += 1
            if projection.final_text:
                result.final_text = projection.final_text
            if projection.session_id:
                result.session_id = projection.session_id
                self.session_id = projection.session_id
            if projection.total_cost_usd is not None:
                result.total_cost_usd = projection.total_cost_usd
            if projection.usage is not None:
                result.usage = projection.usage

            if event.get("type") == "result":
                saw_terminal = True
                if projection.is_error:
                    msg = projection.error_message or "claude CLI returned an error"
                    stderr_tail = "\n".join(self._client.stderr_tail(40))
                    hint = _classify_auth_failure(str(msg), stderr_tail)
                    result.error = hint or self._format_error_with_stderr(
                        f"claude result error: {msg}"
                    )
                break

        if not saw_terminal and not result.interrupted:
            self._client.send_interrupt()
            result.interrupted = True
            result.should_retire = True
            if not result.error:
                result.error = self._format_error_with_stderr(
                    f"claude CLI turn timed out after {turn_timeout:.0f}s"
                )

        return result

    # ---------- diagnostics ----------

    def _format_error_with_stderr(
        self,
        prefix: str,
        exc: Any = "",
        *,
        tail_lines: int = _STDERR_TAIL_LINES,
    ) -> str:
        exc_str = str(exc) if exc != "" and exc is not None else ""
        base = f"{prefix}: {exc_str}" if exc_str else prefix
        if self._client is None:
            return base
        try:
            tail = self._client.stderr_tail(tail_lines)
        except Exception:
            return base
        if not tail:
            return base
        joined = "\n".join(line.rstrip() for line in tail if line)
        if not joined.strip():
            return base
        return f"{base}\nclaude stderr (last {len(tail)} lines):\n{joined[:4000]}"


__all__ = [
    "ClaudeCodeCliSession",
    "ClaudeTurnResult",
    "_classify_auth_failure",
    "_find_claude_binary",
]
