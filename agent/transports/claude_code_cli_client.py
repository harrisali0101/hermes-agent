"""Long-lived claude CLI subprocess client speaking stream-json over stdio.

Wire protocol (Roasbeef Go SDK docs, validated by our probe):
  - Launch: ``claude --print --output-format stream-json --input-format
    stream-json --verbose [--model ... --system-prompt ... --mcp-config ...
    --permission-mode ...]``
  - Stdout: line-delimited JSON, one event per line. We discriminate by
    ``type``: ``system|assistant|user|result|stream|control_response|
    sdk_control_request|rate_limit_event``.
  - Stdin: line-delimited JSON. User prompts go as
    ``{"type":"user","message":{"role":"user","content":[{"type":"text",
    "text":"..."}]}}``.

This is the wire-level speaker only. Event projection into Hermes' messages
and per-turn driving live in sibling modules.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClaudeCliError(RuntimeError):
    """Raised on protocol or subprocess errors."""

    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class ClaudeCodeCliClient:
    """Spawns one ``claude --print --output-format stream-json`` subprocess
    and exposes a blocking-with-timeout event queue.

    Threading model:
      - One reader thread parses stdout, emits each JSON object to a queue.
      - One reader thread captures stderr for diagnostics.
      - The caller drives I/O synchronously: ``send_user(text)`` writes the
        prompt frame, ``take_event(timeout)`` blocks until the next event
        arrives. Mirrors the CodexAppServerClient style.
    """

    def __init__(
        self,
        *,
        claude_bin: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        mcp_config_path: Optional[str] = None,
        permission_mode: str = "bypassPermissions",
        extra_args: Optional[Iterable[str]] = None,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        argv: list[str] = [
            claude_bin,
            "--print",
            "--output-format", "stream-json",
            "--input-format", "stream-json",
            "--verbose",
            "--permission-mode", permission_mode,
        ]
        if model:
            argv.extend(["--model", model])
        if system_prompt:
            argv.extend(["--system-prompt", system_prompt])
        if mcp_config_path:
            argv.extend(["--mcp-config", mcp_config_path])
        if extra_args:
            argv.extend(list(extra_args))

        spawn_env = os.environ.copy()
        if env:
            spawn_env.update(env)
        # CRITICAL: ANTHROPIC_API_KEY takes precedence over CLAUDE_CODE_OAUTH_TOKEN.
        # If it's set in the inherited env, the CLI bills at API rates rather
        # than against the Max subscription base allowance, which is the entire
        # point of routing through the binary in the first place. Strip it
        # unconditionally — callers who genuinely want API billing should use
        # the in-process anthropic_adapter instead.
        spawn_env.pop("ANTHROPIC_API_KEY", None)

        logger.info(
            "claude-cli launch: bin=%s model=%s sys=%s mcp=%s perm=%s",
            claude_bin,
            model,
            bool(system_prompt),
            bool(mcp_config_path),
            permission_mode,
        )

        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered text mode
                env=spawn_env,
                cwd=cwd,
            )
        except (FileNotFoundError, OSError) as exc:
            raise ClaudeCliError(f"claude binary at {claude_bin!r} not launchable: {exc}") from exc

        self._argv = argv
        self._events: queue.Queue = queue.Queue()
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self._closed = False
        # Most-recent session id observed via system/init or result events —
        # callers can read it after run_turn() for trajectory metadata.
        self.last_session_id: Optional[str] = None

        self._stdout_reader = threading.Thread(
            target=self._read_stdout, daemon=True, name="claude-cli-stdout"
        )
        self._stderr_reader = threading.Thread(
            target=self._read_stderr, daemon=True, name="claude-cli-stderr"
        )
        self._stdout_reader.start()
        self._stderr_reader.start()

    # ---------- lifecycle ----------

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def close(self, timeout: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    self._proc.kill()
                except Exception:
                    pass

    def __enter__(self) -> "ClaudeCodeCliClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- send ----------

    def send_user(self, text: str) -> None:
        """Send a user turn over stdin."""
        if self._proc.stdin is None or self._proc.stdin.closed:
            raise ClaudeCliError("claude CLI stdin is closed")
        frame = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            },
        }
        line = json.dumps(frame, ensure_ascii=False) + "\n"
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ClaudeCliError(f"claude CLI stdin write failed: {exc}") from exc

    def send_interrupt(self) -> None:
        """Best-effort interrupt: close stdin so the CLI finishes the current
        turn and exits. The caller should then dispose of this client.

        Claude CLI doesn't currently expose a fine-grained interrupt frame
        on the wire protocol, unlike codex's ``turn/interrupt``. Closing
        stdin is the documented graceful-shutdown path.
        """
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except Exception:
            pass

    # ---------- receive ----------

    def take_event(self, timeout: float = 0.25) -> Optional[dict]:
        """Block up to *timeout* seconds for the next event. Returns None
        on timeout. Returns the parsed JSON dict, or ``{"type":"_eof"}``
        when stdout closes (the reader thread emits a sentinel)."""
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None

    def stderr_tail(self, n: int = 12) -> list[str]:
        with self._stderr_lock:
            return list(self._stderr_lines[-n:])

    # ---------- internals ----------

    def _read_stdout(self) -> None:
        try:
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Non-JSON stdout (banner, color codes if --verbose
                    # accidentally pipes them) — preserve as a synthetic
                    # event so the projector can dump it into the trace
                    # without parsing.
                    obj = {"type": "_nonjson", "raw": line[:2000]}
                # Opportunistically capture session_id so the runtime
                # has it even if the caller only reads the result event.
                sid = obj.get("session_id")
                if isinstance(sid, str) and sid:
                    self.last_session_id = sid
                self._events.put(obj)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("claude-cli stdout reader: %s", exc)
        finally:
            self._events.put({"type": "_eof"})

    def _read_stderr(self) -> None:
        try:
            assert self._proc.stderr is not None
            for line in self._proc.stderr:
                line = line.rstrip("\n")
                if not line:
                    continue
                with self._stderr_lock:
                    self._stderr_lines.append(line)
                    if len(self._stderr_lines) > 200:
                        # Cap memory in long-running sessions
                        del self._stderr_lines[: len(self._stderr_lines) - 200]
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("claude-cli stderr reader: %s", exc)


__all__ = ["ClaudeCodeCliClient", "ClaudeCliError"]
