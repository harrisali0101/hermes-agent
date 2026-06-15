"""Hermes-side MCP server: scoped writes to gbrain.

Exposes a single tool — `save_to_scope` — that takes (scope, title, body) and
internally picks the right OAuth client bearer to forward the put_page call
to gbrain. This lives in Hermes rather than gbrain because gbrain's OAuth
client hard-binds source_id at registration time; routing per-call requires
us to maintain N bearers (one per scope) and pick the right one based on
content classification (which the bot has just done in chat).

This module is launched by claude-code-cli as a stdio MCP subprocess; the
parent (hermes.service) writes a claude-mcp.json that lists this server
alongside the primary gbrain HTTP server.

Env vars (set by claude_code_runtime when configuring the per-session MCP
config; this module fails closed if any is missing):

  HERMES_SENDER_LID    — whatsapp lid of the verified sender, used for the
                         scopes.yaml role lookup + audit log
  HERMES_SCOPES_YAML   — path to scopes.yaml (authorization source of truth)
  HERMES_BEARERS_FILE  — path to a JSON file `{scope: bearer}` minted by
                         claude_code_runtime at session start; one bearer per
                         scope the sender is authorized to write to
  HERMES_GBRAIN_URL    — gbrain base url (e.g. http://127.0.0.1:7777)
  HERMES_GBRAIN_TIMEOUT — optional, seconds, default 30

Audit: every tool call writes one structured line to stderr so the parent
shell (hermes.service journal) captures it.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


def _slugify(text: str, max_len: int = 80) -> str:
    """Lowercase, alnum + dash, no leading/trailing dashes, capped length."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text).strip("-")
    return text[:max_len] or "untitled"

_PROTOCOL_VERSION = "2024-11-05"
_SERVER_NAME = "hermes-save"
_SERVER_VERSION = "0.1.0"


def _log(level: str, msg: str, **kwargs: Any) -> None:
    """One-line JSON to stderr. Parent process journal captures it."""
    record = {"ts": time.time(), "level": level, "msg": msg, **kwargs}
    try:
        sys.stderr.write(json.dumps(record) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name, default)
    if v is None or v == "":
        return None
    return v


def _load_yaml(path: str) -> Optional[Dict[str, Any]]:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        _log("error", "scopes.yaml load failed", path=path, error=str(exc))
        return None


def _load_bearers(path: str) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items() if v}
    except Exception as exc:
        _log("error", "bearers file load failed", path=path, error=str(exc))
        return {}


def _resolve_role(scopes_data: Dict[str, Any], sender_lid: str) -> Optional[str]:
    """Look up sender role: super_admin if in super_admins list, else from users list."""
    for entry in scopes_data.get("super_admins") or []:
        if str(entry.get("id", "")).strip() == sender_lid:
            return "super_admin"
    for entry in scopes_data.get("users") or []:
        if str(entry.get("id", "")).strip() == sender_lid:
            return str(entry.get("role", "")).strip() or None
    return None


def _can_write(scopes_data: Dict[str, Any], role: str, scope: str) -> bool:
    """super_admin role has god-mode writes. Other roles use roles.<role>.writes."""
    if role == "super_admin":
        return scope in (scopes_data.get("scopes") or {})
    role_def = (scopes_data.get("roles") or {}).get(role) or {}
    writes = role_def.get("writes") or []
    return scope in writes


def _scope_list(scopes_data: Dict[str, Any]) -> List[str]:
    return sorted(list((scopes_data.get("scopes") or {}).keys()))


def _parse_sse_envelope(raw: str) -> Dict[str, Any]:
    """gbrain's MCP HTTP transport always returns SSE-framed responses
    (event: message\\ndata: {...}). Parse out the first data line as JSON.
    Falls back to treating the body as plain JSON if no SSE frames detected.
    """
    json_payload: Optional[str] = None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("data:"):
            json_payload = stripped[len("data:"):].strip()
            break
    if json_payload is None:
        json_payload = raw.strip()
    return json.loads(json_payload)


def _gbrain_put_page(
    gbrain_url: str,
    bearer: str,
    title: str,
    body: str,
    slug: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    """POST to gbrain /mcp tools/call:put_page. Returns the parsed JSON-RPC envelope.

    gbrain requires `Accept: application/json, text/event-stream` (per the
    Streamable HTTP MCP spec) and always responds in SSE frames even for a
    single-response call. Sending Accept: application/json alone returns
    HTTP 406 "Not Acceptable: Client must accept both application/json and
    text/event-stream".
    """
    # gbrain put_page takes (slug, content). Title is read from the markdown
    # H1 in content. Build slug from title if not explicitly provided; prepend
    # `# <title>` to body so the rendered page has the right heading.
    resolved_slug = slug if slug else _slugify(title)
    if body.lstrip().startswith("#"):
        full_content = body
    else:
        full_content = f"# {title}\n\n{body}"
    arguments: Dict[str, Any] = {"slug": resolved_slug, "content": full_content}
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "put_page", "arguments": arguments},
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{gbrain_url.rstrip('/')}/mcp",
        data=body_bytes,
        method="POST",
        headers={
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return _parse_sse_envelope(raw)


# ─── MCP protocol handlers ───────────────────────────────────────────────


def _handle_initialize(req: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "protocolVersion": _PROTOCOL_VERSION,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": _SERVER_NAME, "version": _SERVER_VERSION},
    }


def _handle_tools_list(scopes_data: Dict[str, Any]) -> Dict[str, Any]:
    available_scopes = _scope_list(scopes_data)
    return {
        "tools": [
            {
                "name": "save_to_scope",
                "description": (
                    "Save a note to a specific knowledge scope in gbrain. "
                    "Pick the scope by inferring from content topic and "
                    "confirming with the user first. Available scopes: "
                    + ", ".join(available_scopes)
                    + ". Returns the gbrain put_page result on success, or "
                    "an error message if the sender is not authorized for "
                    "the requested scope."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "scope": {
                            "type": "string",
                            "enum": available_scopes,
                            "description": (
                                "Target scope. Must be one of the allowed "
                                "enum values; the sender's role must also "
                                "have write permission for it."
                            ),
                        },
                        "title": {
                            "type": "string",
                            "description": "Page title (becomes the page heading).",
                        },
                        "body": {
                            "type": "string",
                            "description": "Page body in markdown.",
                        },
                        "slug": {
                            "type": "string",
                            "description": (
                                "Optional URL-friendly slug; gbrain auto-"
                                "generates one from title if omitted."
                            ),
                        },
                    },
                    "required": ["scope", "title", "body"],
                },
            }
        ]
    }


def _handle_tools_call(
    req: Dict[str, Any],
    scopes_data: Dict[str, Any],
    bearers: Dict[str, str],
    sender_lid: str,
    role: Optional[str],
    gbrain_url: str,
    timeout: float,
) -> Dict[str, Any]:
    params = req.get("params") or {}
    name = str(params.get("name") or "")
    args = params.get("arguments") or {}
    if name != "save_to_scope":
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"unknown tool: {name}"}],
        }
    scope = str(args.get("scope") or "").strip()
    title = str(args.get("title") or "").strip()
    body = str(args.get("body") or "")
    slug = args.get("slug")
    if slug is not None:
        slug = str(slug).strip() or None

    # Validate inputs
    if not scope or not title or not body:
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": "save_to_scope requires scope, title, and body.",
            }],
        }
    available = set(_scope_list(scopes_data))
    if scope not in available:
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": f"unknown scope '{scope}'. Available: {sorted(available)}",
            }],
        }
    if not role:
        _log(
            "warn",
            "no role resolved for sender",
            sender=sender_lid,
            scope=scope,
        )
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": (
                    "Sender identity could not be resolved to a role. "
                    "Save aborted. Ask Harris to add this sender to scopes.yaml."
                ),
            }],
        }
    if not _can_write(scopes_data, role, scope):
        _log(
            "warn",
            "role not authorized for scope",
            sender=sender_lid,
            role=role,
            scope=scope,
        )
        role_def = (scopes_data.get("roles") or {}).get(role) or {}
        allowed = role_def.get("writes") or []
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": (
                    f"Role '{role}' is not authorized to write to scope "
                    f"'{scope}'. Allowed writes for this role: {sorted(allowed) or '(none)'}"
                ),
            }],
        }

    # Pick bearer via writer_role mapping
    scope_def = (scopes_data.get("scopes") or {}).get(scope) or {}
    writer_role = str(scope_def.get("writer_role") or "").strip()
    if not writer_role:
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": (
                    f"scopes.yaml missing `writer_role` field on scope "
                    f"'{scope}' — admin needs to fix the config."
                ),
            }],
        }
    bearer = bearers.get(writer_role)
    if not bearer:
        _log(
            "error",
            "no bearer available for writer_role",
            sender=sender_lid,
            role=role,
            scope=scope,
            writer_role=writer_role,
        )
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": (
                    f"No OAuth bearer minted for writer_role '{writer_role}' "
                    f"(scope '{scope}'). Hermes session needs to re-mint the "
                    f"bearer cache — restart this session or contact admin."
                ),
            }],
        }

    # Forward to gbrain
    _log(
        "info",
        "save_to_scope dispatch",
        sender=sender_lid,
        role=role,
        scope=scope,
        writer_role=writer_role,
        title=title,
        slug=slug,
        body_len=len(body),
    )
    try:
        result = _gbrain_put_page(gbrain_url, bearer, title, body, slug, timeout)
    except urllib.error.HTTPError as exc:
        snippet = ""
        try:
            snippet = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        _log("error", "gbrain HTTPError", code=exc.code, snippet=snippet)
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": f"gbrain put_page HTTP {exc.code}: {snippet}",
            }],
        }
    except Exception as exc:
        _log("error", "gbrain request failed", error=str(exc))
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"gbrain request failed: {exc}"}],
        }

    # Pass through gbrain's tool result if present; otherwise return the raw envelope
    inner = (result or {}).get("result")
    if isinstance(inner, dict) and ("content" in inner or "isError" in inner):
        _log("info", "save_to_scope success", sender=sender_lid, scope=scope)
        return inner
    _log("info", "save_to_scope returned raw envelope", sender=sender_lid, scope=scope)
    return {
        "content": [{
            "type": "text",
            "text": (
                f"Saved to scope '{scope}'. "
                f"gbrain response: {json.dumps(result)[:500]}"
            ),
        }],
    }


# ─── stdio JSON-RPC loop ─────────────────────────────────────────────────


def _read_request() -> Optional[Dict[str, Any]]:
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        _log("error", "bad json on stdin", error=str(exc), line=line[:200])
        return {"jsonrpc": "2.0", "id": None, "method": "_invalid_"}


def _write_response(req_id: Any, result: Any) -> None:
    sys.stdout.write(
        json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n"
    )
    sys.stdout.flush()


def _write_error(req_id: Any, code: int, message: str) -> None:
    sys.stdout.write(
        json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }) + "\n"
    )
    sys.stdout.flush()


def main() -> int:
    sender_lid = _env("HERMES_SENDER_LID") or ""
    scopes_yaml_path = _env("HERMES_SCOPES_YAML") or ""
    bearers_path = _env("HERMES_BEARERS_FILE") or ""
    gbrain_url = _env("HERMES_GBRAIN_URL") or ""
    timeout_str = _env("HERMES_GBRAIN_TIMEOUT") or "30"
    try:
        timeout = float(timeout_str)
    except ValueError:
        timeout = 30.0

    missing = [
        n for n, v in [
            ("HERMES_SENDER_LID", sender_lid),
            ("HERMES_SCOPES_YAML", scopes_yaml_path),
            ("HERMES_BEARERS_FILE", bearers_path),
            ("HERMES_GBRAIN_URL", gbrain_url),
        ] if not v
    ]
    if missing:
        _log("error", "required env vars missing — fail closed", missing=missing)
        # Don't exit — claude-code-cli will spawn us anyway; emit a clear
        # error on the first tool call so the bot surfaces it instead of
        # hanging.

    scopes_data = _load_yaml(scopes_yaml_path) or {}
    bearers = _load_bearers(bearers_path) if bearers_path else {}
    role = _resolve_role(scopes_data, sender_lid) if scopes_data and sender_lid else None
    _log(
        "info",
        "hermes-save MCP server started",
        sender=sender_lid,
        role=role,
        scopes=_scope_list(scopes_data),
        bearer_count=len(bearers),
    )

    while True:
        req = _read_request()
        if req is None:
            _log("info", "stdin EOF — exiting")
            return 0
        method = str(req.get("method") or "")
        req_id = req.get("id")
        try:
            if method == "initialize":
                _write_response(req_id, _handle_initialize(req))
            elif method == "initialized" or method == "notifications/initialized":
                # No response for notifications.
                continue
            elif method == "tools/list":
                _write_response(req_id, _handle_tools_list(scopes_data))
            elif method == "tools/call":
                _write_response(
                    req_id,
                    _handle_tools_call(
                        req, scopes_data, bearers, sender_lid, role,
                        gbrain_url, timeout,
                    ),
                )
            elif method == "ping":
                _write_response(req_id, {})
            elif method == "_invalid_":
                _write_error(req_id, -32700, "parse error")
            else:
                _write_error(req_id, -32601, f"method not found: {method}")
        except Exception as exc:
            _log("error", "handler raised", method=method, error=str(exc))
            _write_error(req_id, -32603, f"internal error: {exc}")


if __name__ == "__main__":
    sys.exit(main())
