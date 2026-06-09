"""Projects claude code CLI stream-json events into Hermes' messages list.

Claude CLI's stream-json protocol is documented (partially) here:
  https://github.com/Roasbeef/claude-agent-sdk-go/blob/main/docs/cli-protocol.md

Each event on stdout is a JSON object with a discriminator `type`:
  - system           — init / error / banner. Carries session_id, tool list.
  - rate_limit_event — usage warnings (display only).
  - assistant        — Claude's response. ``message.content`` is a list of
                       blocks: ``{type:"text", text}``, ``{type:"tool_use",
                       id, name, input}``, ``{type:"thinking", text}``.
  - user             — echo of input AND tool_result deliveries between
                       turns. The ``message.content`` may contain
                       ``{type:"tool_result", tool_use_id, content}``.
  - result           — terminal frame per turn. Carries final ``result``
                       string, ``session_id``, ``total_cost_usd``, ``usage``.
  - stream           — incremental deltas (only when --include-partial-messages).
  - control_response — replies to SDK control requests (we don't use these
                       at v1.0, only relevant if we register SDK MCP servers
                       via the control protocol — gbrain is wired via
                       --mcp-config stdio instead).
  - sdk_control_request — CLI asking the SDK for things (permission, MCP
                          call). At v1.0 we use --permission-mode
                          bypassPermissions and stdio MCP, so this should
                          not fire.

Each ``assistant`` event with one or more ``tool_use`` blocks produces:
  * one assistant message with ``tool_calls`` (and an empty text content),
  * for each subsequent ``user`` event with ``tool_result`` blocks, one
    ``tool`` role message keyed by ``tool_use_id``.

Tool execution is opaque from Hermes' point of view — claude runs its
native tools (Read/Bash/Edit/Glob/...) inside the subprocess and reports
back. We project the trajectory so memory + skill review can still see
what happened.

Counters tracked alongside projection:
  - tool_iterations: ticks once per assistant tool_use block. Used by
    AIAgent._iters_since_skill (skill nudge gate, default 10).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional


def _deterministic_call_id(tool_use_id: str, tool_name: str) -> str:
    """Stable id for tool_call message correlation.

    Claude emits its own tool_use ids (``toolu_*``); we wrap them with a
    prefix so traces are obviously claude-cli vs. codex/native. Falls back
    to a content hash if the upstream id is somehow missing.
    """
    if tool_use_id:
        return f"claude_{tool_use_id}"
    digest = hashlib.sha256(tool_name.encode()).hexdigest()[:16]
    return f"claude_anon_{tool_name}_{digest}"


def _format_tool_args(d: Any) -> str:
    """Format tool input as JSON the way Hermes' existing tool_calls path does."""
    if not isinstance(d, dict):
        d = {"arguments": d}
    return json.dumps(d, ensure_ascii=False, sort_keys=True)


def _truncate(s: str, limit: int = 4000) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"... [truncated {len(s) - limit} chars]"


def _extract_content_text(content: Any) -> str:
    """Reduce a content list (or string) to plain text.

    Claude's content arrays may contain mixed blocks; tool_result content
    in particular may be a list of ``{type:"text", text}`` or a list of
    image blocks. Drop non-text parts.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for blk in content:
        if isinstance(blk, dict):
            if blk.get("type") == "text":
                parts.append(str(blk.get("text") or ""))
            elif blk.get("type") in {"image", "input_image"}:
                parts.append("[image]")
            elif "text" in blk:
                parts.append(str(blk["text"]))
        elif isinstance(blk, str):
            parts.append(blk)
    return "\n".join(p for p in parts if p)


@dataclass
class ProjectionResult:
    """Output of projecting one stream-json event.

    `messages` is a list because one assistant event may produce one
    assistant message + zero-or-more correlated tool messages over the
    rest of the turn. Empty list = event ignored (system init, rate limit,
    stream deltas).
    """

    messages: list[dict] = field(default_factory=list)
    is_tool_iteration: bool = False
    final_text: Optional[str] = None
    # Set when the terminal result event arrives — captures session
    # metadata the runtime driver wants to surface to the caller.
    session_id: Optional[str] = None
    total_cost_usd: Optional[float] = None
    usage: Optional[dict] = None
    is_error: bool = False
    error_message: Optional[str] = None


class ClaudeCodeCliEventProjector:
    """Stateful projector consuming claude CLI events in arrival order.

    Owns the in-progress assistant text accumulator (text deltas across
    multiple assistant events in a single turn end up concatenated for the
    ``final_text`` reading at the result event).
    """

    def __init__(self) -> None:
        # Most recent assistant text seen this turn. Updated by each
        # assistant event whose content includes a text block.
        self._last_assistant_text: str = ""
        # Map of tool_use_id → tool_name so we can label tool results
        # consistently when the user event echoes them back.
        self._tool_use_names: dict[str, str] = {}

    def project(self, event: dict) -> ProjectionResult:
        """Project a single stream-json event."""
        if not isinstance(event, dict):
            return ProjectionResult()
        t = event.get("type") or ""
        if t == "system":
            return ProjectionResult()
        if t == "rate_limit_event":
            return ProjectionResult()
        if t == "stream":
            return ProjectionResult()
        if t == "assistant":
            return self._project_assistant(event)
        if t == "user":
            return self._project_user(event)
        if t == "result":
            return self._project_result(event)
        if t in {"control_response", "sdk_control_request"}:
            # Control plane — runtime handles these directly, not via
            # projection into the messages list.
            return ProjectionResult()
        # Unknown event — record opaquely so trajectory has something.
        try:
            payload = json.dumps(event, ensure_ascii=False)[:1000]
        except (TypeError, ValueError):
            payload = repr(event)[:1000]
        return ProjectionResult(
            messages=[
                {
                    "role": "assistant",
                    "content": f"[claude-cli unknown event type={t}] {payload}",
                }
            ]
        )

    # ---------- per-type projections ----------

    def _project_assistant(self, event: dict) -> ProjectionResult:
        message = event.get("message") or {}
        content_blocks = message.get("content") or []
        if not isinstance(content_blocks, list):
            content_blocks = [content_blocks]

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        thinking_parts: list[str] = []
        for blk in content_blocks:
            if not isinstance(blk, dict):
                continue
            btype = blk.get("type")
            if btype == "text":
                text = blk.get("text") or ""
                if text:
                    text_parts.append(text)
            elif btype == "thinking":
                thinking = blk.get("text") or blk.get("thinking") or ""
                if thinking:
                    thinking_parts.append(str(thinking))
            elif btype == "tool_use":
                tool_id = blk.get("id") or ""
                tool_name = blk.get("name") or "unknown"
                tool_input = blk.get("input") or {}
                self._tool_use_names[tool_id] = tool_name
                tool_calls.append(
                    {
                        "id": _deterministic_call_id(tool_id, tool_name),
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": _format_tool_args(tool_input),
                        },
                    }
                )

        text = "\n".join(p for p in text_parts if p)
        if text:
            self._last_assistant_text = text

        # Build the projected assistant message. If there are tool_calls,
        # content is None per the OpenAI convention; otherwise content is
        # the assembled text.
        messages: list[dict] = []
        if tool_calls or text:
            msg: dict[str, Any] = {"role": "assistant"}
            if tool_calls:
                msg["content"] = None
                msg["tool_calls"] = tool_calls
            else:
                msg["content"] = text
            if thinking_parts:
                msg["reasoning"] = "\n".join(thinking_parts)
            messages.append(msg)

        return ProjectionResult(
            messages=messages,
            is_tool_iteration=bool(tool_calls),
            final_text=text if text else None,
        )

    def _project_user(self, event: dict) -> ProjectionResult:
        """Project a user event — usually echoes of tool_result blocks
        claude is feeding back into its own conversation.

        We map tool_result blocks to Hermes' ``role: "tool"`` shape so the
        trajectory recorder sees a complete assistant→tool→assistant chain.
        Plain text user echoes (the original prompt) are dropped — Hermes
        already has the original user message in its own messages list.
        """
        message = event.get("message") or {}
        content_blocks = message.get("content") or []
        if not isinstance(content_blocks, list):
            return ProjectionResult()

        tool_msgs: list[dict] = []
        for blk in content_blocks:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") != "tool_result":
                continue
            tool_use_id = blk.get("tool_use_id") or ""
            raw_content = blk.get("content")
            text = _extract_content_text(raw_content)
            is_error = bool(blk.get("is_error"))
            content = _truncate(text or "")
            if is_error:
                content = f"[error] {content}"
            tool_msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": _deterministic_call_id(
                        tool_use_id, self._tool_use_names.get(tool_use_id, "tool")
                    ),
                    "content": content,
                }
            )
        return ProjectionResult(messages=tool_msgs)

    def _project_result(self, event: dict) -> ProjectionResult:
        """Project the terminal result event.

        Doesn't add to ``messages`` — the assistant text was already
        emitted by the preceding assistant events. We surface session
        metadata for the runtime driver to use.
        """
        is_error = bool(event.get("is_error"))
        final = event.get("result") or self._last_assistant_text or ""
        if is_error and not final:
            final = event.get("error") or "claude CLI returned an error"
        return ProjectionResult(
            final_text=final or None,
            session_id=event.get("session_id"),
            total_cost_usd=event.get("total_cost_usd"),
            usage=event.get("usage"),
            is_error=is_error,
            error_message=event.get("error") or event.get("api_error_status"),
        )


__all__ = [
    "ClaudeCodeCliEventProjector",
    "ProjectionResult",
    "_deterministic_call_id",
]
