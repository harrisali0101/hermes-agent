"""Hermes-side MCP server: per-subject READS via RFC 8693 token-exchange.

Companion to `hermes_save_mcp.py` — kept SEPARATE on purpose. The save
module owns writes + admin tools (save_to_scope, approve_user, etc.);
this module owns READS (query, search, get_page, list_pages). Two
concerns, two MCP servers, two `mcp_servers:` entries in config.yaml.

Why a Python proxy at all
-------------------------
hermes-agent registers MCP servers STATICALLY in config.yaml. Each
server entry has a static bearer. With a god-mode bearer, every
WhatsApp user gets the same RLS scope — staff can read CEO content
through the bot. The fix is per-turn token minting:

  1. Hold ONE delegator OAuth client credential on the gateway.
  2. On every read tool call, exchange the delegator's credentials
     for a short-lived token bound to the verified sender's subject
     identity via RFC 8693 OAuth Token Exchange.
  3. gbrain enforces RLS at the SQL layer using the subject's
     `allowed_sources` (from the `subjects` table), NOT the
     delegator's static `federated_read`.

The TokenCache that mints + caches per-subject tokens lives in
`gbrain_exchange.py` — pure library, no MCP protocol. This module
is the thin MCP-stdio wrapper that exposes the read tools.

Env vars
--------
  HERMES_GBRAIN_URL                    — gbrain base URL (e.g. http://127.0.0.1:7777)
  HERMES_GBRAIN_TIMEOUT                — per-call timeout in seconds (default 30)
  HERMES_GBRAIN_DELEGATOR_CLIENT_ID    — REQUIRED. The delegator OAuth client_id.
  HERMES_GBRAIN_DELEGATOR_CLIENT_SECRET — REQUIRED. The delegator OAuth client_secret.
  HERMES_GBRAIN_DELEGATOR_SCOPE        — Optional scope to request (default "read").
  HERMES_GBRAIN_DELEGATOR_RESOURCE     — Optional RFC 8707 audience URI; defaults
                                         to `<HERMES_GBRAIN_URL>/mcp`.
  HERMES_SCOPES_YAML                   — Optional. Used for audit-log role lookup
                                         only — gbrain itself enforces RLS.

If the delegator env vars are missing the server still starts but every
tool call fails closed with a clear "delegator not configured" message.
This matches the fail-closed posture of the save module.

Audit
-----
Every tool call writes one structured-JSON line to stderr (captured by
hermes.service journal). subject_id is logged verbatim — operators
already see it in the gateway's bridge logs, no incremental PII exposure
here. Errors are logged with HTTP status + first 300 chars of body.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


_PROTOCOL_VERSION = "2024-11-05"
_SERVER_NAME = "hermes-read"
_SERVER_VERSION = "0.1.0"

_LID_RE = re.compile(r"^\d{6,}@lid$")

# The sender_lid contract is IDENTICAL to hermes_save_mcp.py: every tool
# call carries the verified sender's WhatsApp lid as a per-call arg.
# Persona is responsible for sourcing it verbatim from the most recent
# <verified_sender id="..."/> marker on the user's message. Missing /
# malformed → server refuses the call, never falls back to a default.
_SENDER_LID_SCHEMA: Dict[str, Any] = {
    "type": "string",
    "pattern": r"^\d{6,}@lid$",
    "description": (
        "REQUIRED. The verified sender's WhatsApp lid (format "
        "`<digits>@lid`), copied verbatim from the most recent "
        "<verified_sender id=\"...\"/> marker on the user's message. "
        "Hermes mints a per-turn gbrain access token bound to this "
        "subject via RFC 8693 OAuth Token Exchange; gbrain enforces "
        "Row-Level Security at the SQL layer using the subject's row "
        "in its `subjects` table. Never invent or guess this — refuse "
        "at the persona layer if no verified_sender marker is present."
    ),
}

# These are the gbrain MCP tool names we proxy. Names match gbrain's
# exposed tools verbatim so the model's mental mapping stays simple.
_READ_TOOLS = ("query", "search", "get_page", "list_pages")


# ── Helpers ──────────────────────────────────────────────────────────────


def _log(level: str, msg: str, **kwargs: Any) -> None:
    """Structured JSON to stderr — captured by hermes.service journal."""
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


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load scopes.yaml. Returns an empty dict on any failure — this module
    only uses it for audit-log role lookup; gbrain itself is the
    authoritative RLS enforcement point."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        _log("warn", "scopes.yaml load failed (audit context unavailable)",
             path=path, error=str(exc))
        return {}


def _resolve_role(scopes_data: Dict[str, Any], sender_lid: str) -> Optional[str]:
    """Look up sender role for audit logging. Returns None if not found —
    that's allowed: gbrain still enforces RLS via the subjects table even
    when this side can't classify the role. We just log "role=unknown"."""
    for entry in scopes_data.get("super_admins") or []:
        if str(entry.get("id", "")).strip() == sender_lid:
            return "super_admin"
    for entry in scopes_data.get("users") or []:
        if str(entry.get("id", "")).strip() == sender_lid:
            return str(entry.get("role", "")).strip() or None
    return None


def _parse_sse_envelope(raw: str) -> Dict[str, Any]:
    """gbrain's MCP HTTP transport always returns SSE-framed responses
    (`event: message\\ndata: {...}`). Pull the first data line as JSON;
    fall back to whole-body JSON for non-SSE responses (defensive only —
    gbrain consistently uses SSE)."""
    json_payload: Optional[str] = None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("data:"):
            json_payload = stripped[len("data:"):].strip()
            break
    if json_payload is None:
        json_payload = raw.strip()
    return json.loads(json_payload)


def _gbrain_tool_call(
    gbrain_url: str,
    bearer: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    """POST gbrain `/mcp tools/call`. Streamable HTTP MCP requires
    `Accept: application/json, text/event-stream` (HTTP 406 otherwise)."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
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


# ── MCP protocol handlers ────────────────────────────────────────────────


def _handle_initialize(_req: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "protocolVersion": _PROTOCOL_VERSION,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": _SERVER_NAME, "version": _SERVER_VERSION},
    }


def _handle_tools_list() -> Dict[str, Any]:
    """The 4 per-subject read tools. gbrain enforces RLS — if the
    sender's subject row doesn't grant a source, no rows come back from
    Postgres regardless of how the tool is called."""
    return {
        "tools": [
            {
                "name": "query",
                "description": (
                    "Semantic search across the sender's authorized "
                    "gbrain sources. Use when the user asks a question "
                    "about company knowledge (meetings, policies, "
                    "documents). Returns ranked passages with page "
                    "slugs. RLS-enforced — results are filtered to "
                    "what the sender's role can read; forbidden "
                    "sources never appear in the result set."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sender_lid": _SENDER_LID_SCHEMA,
                        "q": {"type": "string", "description": "Natural-language query."},
                        "limit": {
                            "type": "integer",
                            "description": "Max passages (default 8).",
                            "minimum": 1, "maximum": 50,
                        },
                    },
                    "required": ["sender_lid", "q"],
                },
            },
            {
                "name": "search",
                "description": (
                    "Keyword (full-text) search across the sender's "
                    "authorized gbrain sources. Use for exact phrases "
                    "or known terms (proper nouns, codes, ticket "
                    "numbers) — complements `query` which is semantic. "
                    "RLS-enforced same as `query`."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sender_lid": _SENDER_LID_SCHEMA,
                        "q": {"type": "string", "description": "Exact keywords or phrase."},
                        "limit": {
                            "type": "integer",
                            "description": "Max hits (default 8).",
                            "minimum": 1, "maximum": 50,
                        },
                    },
                    "required": ["sender_lid", "q"],
                },
            },
            {
                "name": "get_page",
                "description": (
                    "Fetch the full text of a specific gbrain page by "
                    "slug. Use after `query`/`search` when the user "
                    "wants details on a specific result. Returns "
                    "`page_not_found` if the sender's role is not "
                    "authorized to read the page's source (RLS, "
                    "DB-enforced)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sender_lid": _SENDER_LID_SCHEMA,
                        "slug": {"type": "string", "description": "Page slug (kebab-case)."},
                    },
                    "required": ["sender_lid", "slug"],
                },
            },
            {
                "name": "list_pages",
                "description": (
                    "List pages in a specific gbrain source (or "
                    "across all the sender's authorized sources). "
                    "Use for browsing — 'what's in the finance "
                    "scope?'. allowed_sources is enforced at the SQL "
                    "layer — listing a source the sender cannot read "
                    "returns an empty list, not the pages."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sender_lid": _SENDER_LID_SCHEMA,
                        "limit": {
                            "type": "integer",
                            "description": "Max pages (default 25).",
                            "minimum": 1, "maximum": 200,
                        },
                    },
                    "required": ["sender_lid"],
                },
            },
        ]
    }


def _handle_tools_call(
    req: Dict[str, Any],
    scopes_data: Dict[str, Any],
    token_cache: Any,
    gbrain_url: str,
    timeout: float,
) -> Dict[str, Any]:
    """Dispatch a read tool call. Steps:
      1. Validate sender_lid (refuse if missing/malformed).
      2. Mint a per-subject access token via the gbrain_exchange.TokenCache.
      3. Forward the tool call to gbrain /mcp with that bearer.
      4. On 401 → invalidate cache so next call re-mints (handles the
         race where the subject was revoked between mint and use).
    """
    params = req.get("params") or {}
    name = str(params.get("name") or "")
    args = params.get("arguments") or {}

    sender_lid = str(args.get("sender_lid") or "").strip()
    if not _LID_RE.match(sender_lid):
        _log("warn", "tool call refused — sender_lid missing or malformed",
             tool=name, sender_lid_received=sender_lid or "(empty)")
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": (
                    "sender_lid required: every hermes_read:* call must "
                    "include the verified sender's lid (format "
                    "`<digits>@lid`) as the `sender_lid` argument. "
                    "Source it from the most recent <verified_sender "
                    "id=\"...\"/> marker on the user's message; never "
                    "invent it."
                ),
            }],
        }

    if name not in _READ_TOOLS:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"unknown tool: {name}"}],
        }

    if token_cache is None:
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": (
                    "Per-subject reads not available: the RFC 8693 "
                    "delegator is not configured. Set "
                    "HERMES_GBRAIN_DELEGATOR_CLIENT_ID and "
                    "HERMES_GBRAIN_DELEGATOR_CLIENT_SECRET on the "
                    "hermes process and restart."
                ),
            }],
        }

    # Tool-specific argument shaping — pass through what gbrain expects.
    if name in ("query", "search"):
        q = str(args.get("q") or "").strip()
        if not q:
            return {"isError": True, "content": [{"type": "text", "text": f"{name} requires q"}]}
        # gbrain's  and  operations both take the search
        # text under the canonical name  (not ). Our exposed
        # tool schema uses  for brevity at the chat-tool layer; we
        # translate here. See gbrain operations.ts: search/query both
        # declare .
        gbrain_args: Dict[str, Any] = {"query": q}
        if isinstance(args.get("limit"), int):
            gbrain_args["limit"] = args["limit"]
    elif name == "get_page":
        slug = str(args.get("slug") or "").strip()
        if not slug:
            return {"isError": True, "content": [{"type": "text", "text": "get_page requires slug"}]}
        gbrain_args = {"slug": slug}
    else:  # list_pages
        # gbrain list_pages has no per-call source filter — scope is
        # derived from the subject's allowed_sources. If the caller
        # passes  we drop it (legacy schema field), and warn
        # in the audit log so the model can learn to omit it.
        gbrain_args = {}
        if isinstance(args.get("limit"), int):
            gbrain_args["limit"] = args["limit"]
        # Sensible default: recency-ordered. Bot uses list_pages
        # as the discovery primitive — newest first is what it
        # actually wants 95% of the time.
        gbrain_args.setdefault("sort", "updated_desc")

    role = _resolve_role(scopes_data, sender_lid)
    try:
        bearer = token_cache.for_subject(sender_lid)
    except Exception as exc:
        _log("error", "token exchange failed", sender=sender_lid, tool=name, error=str(exc))
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": (
                    "Could not mint a per-user token for this read. "
                    "The sender may not be registered as a gbrain "
                    "subject yet. Ask Harris to run subjects-sync. "
                    f"Internal: {exc}"
                ),
            }],
        }

    _log("info", "subject read dispatch",
         sender=sender_lid, role=role or "unknown",
         tool=name, args_keys=sorted(gbrain_args.keys()))

    try:
        result = _gbrain_tool_call(gbrain_url, bearer, name, gbrain_args, timeout)
    except urllib.error.HTTPError as exc:
        snippet = ""
        try:
            snippet = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        # 401 → cached token is suddenly invalid (subject revoked, secret
        # rotated, etc.). Invalidate so the next call re-mints. Swallow
        # invalidate failures — the cache might already be in a clean
        # state, and we never want this branch to mask the original 401.
        if exc.code == 401:
            try:
                token_cache.invalidate(sender_lid)
            except Exception:
                pass
        _log("error", "gbrain read HTTPError",
             code=exc.code, tool=name, snippet=snippet)
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"gbrain {name} HTTP {exc.code}: {snippet}"}],
        }
    except Exception as exc:
        _log("error", "gbrain read failed", tool=name, error=str(exc))
        return {"isError": True, "content": [{"type": "text", "text": f"gbrain {name} failed: {exc}"}]}

    inner = (result or {}).get("result")
    if isinstance(inner, dict) and ("content" in inner or "isError" in inner):
        return inner
    # Defensive fallback: gbrain's MCP returns a JSON-RPC envelope whose
    # `result` is an object with `content` and/or `isError`. If we ever
    # see a different shape, we don't want to crash — JSON-stringify and
    # return. BUT: log a warn so silent protocol drift surfaces in the
    # journal. Without this, gbrain could change its response shape and
    # users would just see "weird text output" with no operator signal.
    _log("warn", "gbrain response shape unexpected — flattening via JSON-stringify",
         tool=name, result_keys=sorted(list((result or {}).keys())))
    return {"content": [{"type": "text", "text": json.dumps(result)[:2000]}]}


# ── stdio JSON-RPC loop ──────────────────────────────────────────────────


def _read_request() -> Optional[Dict[str, Any]]:
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return {"method": "_invalid_", "id": None}
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"method": "_invalid_", "id": None}


def _write_response(req_id: Any, result: Any) -> None:
    response = {"jsonrpc": "2.0", "id": req_id, "result": result}
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def _write_error(req_id: Any, code: int, message: str) -> None:
    response = {
        "jsonrpc": "2.0", "id": req_id,
        "error": {"code": code, "message": message},
    }
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def main() -> int:
    gbrain_url = _env("HERMES_GBRAIN_URL") or "http://127.0.0.1:7777"
    scopes_yaml_path = _env("HERMES_SCOPES_YAML") or ""
    timeout_str = _env("HERMES_GBRAIN_TIMEOUT") or "30"
    try:
        timeout = float(timeout_str)
    except ValueError:
        timeout = 30.0

    delegator_id = _env("HERMES_GBRAIN_DELEGATOR_CLIENT_ID")
    delegator_secret = _env("HERMES_GBRAIN_DELEGATOR_CLIENT_SECRET")
    delegator_scope = _env("HERMES_GBRAIN_DELEGATOR_SCOPE") or "read"
    delegator_resource = _env("HERMES_GBRAIN_DELEGATOR_RESOURCE") or f"{gbrain_url.rstrip('/')}/mcp"

    token_cache: Any = None
    if delegator_id and delegator_secret:
        try:
            from gbrain_exchange import TokenCache
        except ImportError:
            # Allow `python -m agent.hermes_read_mcp` execution layouts too.
            from agent.gbrain_exchange import TokenCache  # type: ignore
        token_cache = TokenCache(
            gbrain_url=gbrain_url,
            client_id=delegator_id,
            client_secret=delegator_secret,
            scope=delegator_scope,
            resource=delegator_resource,
            timeout=min(timeout, 10.0),
        )
        _log("info", "RFC 8693 delegator configured",
             client_id_prefix=delegator_id[:18] + "…",
             resource=delegator_resource,
             scope=delegator_scope)
    else:
        _log("warn",
             "RFC 8693 delegator NOT configured — read tools will refuse every call",
             missing=[n for n, v in [
                 ("HERMES_GBRAIN_DELEGATOR_CLIENT_ID", delegator_id),
                 ("HERMES_GBRAIN_DELEGATOR_CLIENT_SECRET", delegator_secret),
             ] if not v])

    scopes_data = _load_yaml(scopes_yaml_path) if scopes_yaml_path else {}
    _log("info", "hermes-read MCP server started",
         gbrain_url=gbrain_url,
         scopes_yaml=scopes_yaml_path or "(none — audit role lookup disabled)",
         delegator_configured=token_cache is not None)

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
            elif method in ("initialized", "notifications/initialized"):
                continue
            elif method == "tools/list":
                _write_response(req_id, _handle_tools_list())
            elif method == "tools/call":
                # Reload scopes.yaml per-call so a /approve doesn't require
                # a server restart for audit-log accuracy. Cheap on small YAML.
                if scopes_yaml_path:
                    scopes_data = _load_yaml(scopes_yaml_path) or scopes_data
                _write_response(req_id, _handle_tools_call(
                    req, scopes_data, token_cache, gbrain_url, timeout,
                ))
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
