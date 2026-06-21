"""Hermes-side MCP server: scoped writes + admin tools, with per-sender routing.

Exposes admin / write tools (`save_to_scope`, `add_to_allowlist`,
`approve_user`, `record_pending_user`, `list_pending_users`). Each tool
takes the verified sender's WhatsApp lid as a `sender_lid` argument; the
proxy looks the role up from scopes.yaml and picks the right per-role
OAuth bearer from a static bearers file before forwarding the call to
gbrain.

v0.2 (2026-06-18): per-call sender_lid arg instead of per-session env var.
Required because the new azure-foundry provider doesn't have the
per-session --mcp-config injection path the previous claude-code-cli
provider used to pass HERMES_SENDER_LID at startup. With the new design
the server is launched ONCE by hermes-agent (via config.yaml's
mcp_servers block) and stays up across all sessions; the persona is
responsible for adding the verified sender's lid to every tool call.

Env vars (read once at startup; no per-session env):

  HERMES_SCOPES_YAML   — path to scopes.yaml (authorization source of truth)
  HERMES_BEARERS_FILE  — path to a JSON file `{scope: bearer}` with one
                         bearer per scope. Bearers are fetched from Key
                         Vault by fetch-secrets.sh at hermes startup;
                         the file is shared across all sessions.
                         Defaults to /home/hermes-user/.hermes/role-bearers.json.
  HERMES_GBRAIN_URL    — gbrain base url (e.g. http://127.0.0.1:7777)
  HERMES_GBRAIN_TIMEOUT — optional, seconds, default 30
  HERMES_HERMES_ENV_PATH — optional, defaults to /home/hermes-user/.hermes/.env
  HERMES_PENDING_USERS_PATH — optional, defaults to the pilot path

Audit: every tool call writes one structured line to stderr so the parent
shell (hermes.service journal) captures it. The `sender_lid` from the
tool args is included in every audit record.
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
_SERVER_VERSION = "0.2.0"

# v0.2: every tool requires the verified sender's lid as a per-call arg
# (no longer a process-startup env var). The persona MUST include this
# in every hermes_save:* call, sourced from the most recent
# <verified_sender id="..."/> marker on the user's message. Without it
# the server fails closed — no role lookup, no bearer routing, no audit.
_SENDER_LID_SCHEMA: Dict[str, Any] = {
    "type": "string",
    "pattern": r"^\d{6,}@lid$",
    "description": (
        "REQUIRED. The verified sender's WhatsApp lid (format "
        "`<digits>@lid`), copied verbatim from the most recent "
        "<verified_sender id=\"...\"/> marker on the user's "
        "message. The server resolves the sender's role from "
        "scopes.yaml using this lid and picks the right per-role "
        "OAuth bearer to forward the gbrain call. Never invent or "
        "guess this — if no verified_sender marker is present "
        "the call must be refused at the persona layer before "
        "reaching this tool."
    ),
}


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


def _role_list(scopes_data: Dict[str, Any]) -> List[str]:
    return sorted(list((scopes_data.get("roles") or {}).keys()))


def _handle_tools_list(scopes_data: Dict[str, Any]) -> Dict[str, Any]:
    available_scopes = _scope_list(scopes_data)
    available_roles = _role_list(scopes_data)
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
                        "sender_lid": _SENDER_LID_SCHEMA,
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
                    "required": ["sender_lid", "scope", "title", "body"],
                },
            },
            {
                "name": "record_pending_user",
                "description": (
                    "Auto-called by the bot WHENEVER it detects a turn "
                    "from a sender who is allowlisted at the gateway but "
                    "NOT YET mapped to a role in scopes.yaml (the None "
                    "tier in ACCESS_POLICY.md). Records the new user's "
                    "lid so the admin can see who's waiting for /approve "
                    "the next time they call list_pending_users. Always "
                    "safe to call — idempotent: re-recording the same "
                    "lid is a no-op. Bot should also reply to the user "
                    "with the welcome-pending message and surface their "
                    "lid in the reply (per AGENTS.md). NOT super_admin "
                    "gated — this fires for every None-tier turn."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sender_lid": _SENDER_LID_SCHEMA,
                        "target_lid": {
                            "type": "string",
                            "description": (
                                "The new user's WhatsApp lid (the same "
                                "`<digits>@lid` string from the verified-"
                                "sender marker on their message)."
                            ),
                        },
                        "first_message_snippet": {
                            "type": "string",
                            "description": (
                                "Optional first ~120 chars of the user's "
                                "incoming message, so the admin has "
                                "context (e.g. 'hi i am tareq for dih')."
                            ),
                        },
                    },
                    "required": ["sender_lid", "target_lid"],
                },
            },
            {
                "name": "list_pending_users",
                "description": (
                    "Returns the current pending-users queue — every "
                    "allowlisted user who has messaged but doesn't yet "
                    "have a role assigned. Each entry shows lid, when "
                    "they first reached out, and (if recorded) the "
                    "snippet of their first message. Use this when the "
                    "Owner asks 'who's pending', 'any new users', 'show "
                    "the onboarding queue', etc. Super_admin only; the "
                    "tool re-checks authorization server-side."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"sender_lid": _SENDER_LID_SCHEMA},
                    "required": ["sender_lid"],
                },
            },
            {
                "name": "add_to_allowlist",
                "description": (
                    "STEP 1 of onboarding a new user: add their PHONE "
                    "NUMBER (digits only, no '+') to the gateway-level "
                    "allowlist (WHATSAPP_ALLOWED_USERS in ~/.hermes/.env). "
                    "Until this is done, messages from the new user are "
                    "DROPPED at the gateway before reaching the bot. The "
                    "lid (used for role assignment in step 2) is only "
                    "visible after they send their first message, so "
                    "allowlisting always comes first. Only super_admins "
                    "(Harris, Shahzaib today) may call this; the tool "
                    "re-checks server-side. Hermes reloads the .env per "
                    "turn, so the change takes effect on the next message."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sender_lid": _SENDER_LID_SCHEMA,
                        "phone": {
                            "type": "string",
                            "description": (
                                "Phone number, digits only, no '+'. "
                                "Example: 923333717117 for +92 333 3717117. "
                                "International format; country code first."
                            ),
                        },
                        "memo": {
                            "type": "string",
                            "description": (
                                "Optional one-line note for the audit "
                                "comment that the tool writes above the "
                                "key line (e.g. 'CEO Iyad — pending lid')."
                            ),
                        },
                    },
                    "required": ["sender_lid", "phone"],
                },
            },
            {
                "name": "approve_user",
                "description": (
                    "Onboard a new user to the DIH pilot by writing them "
                    "into scopes.yaml with a role. Only super_admins (CEOs "
                    "or dev-tier admins) may call this; the tool re-checks "
                    "authorization server-side and refuses if the caller "
                    "is not a super_admin. The new user's role determines "
                    "which scopes they can read and write. Available roles: "
                    + ", ".join(available_roles) + " (plus implicit "
                    "'super_admin' which is added via SSH only, never chat). "
                    "Cache is invalidated automatically on the next turn."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sender_lid": _SENDER_LID_SCHEMA,
                        "target_lid": {
                            "type": "string",
                            "description": (
                                "WhatsApp lid of the user to approve, "
                                "format `<digits>@lid` (e.g., "
                                "`87449845936164@lid`). The lid is the "
                                "privacy-mode identifier — NOT the phone "
                                "number. Capture it from the verified-sender "
                                "marker on a message the new user has "
                                "already sent."
                            ),
                        },
                        "name": {
                            "type": "string",
                            "description": (
                                "Human-readable name (e.g., 'Tareq', "
                                "'Iyad Mazhar'). Used in audit and persona "
                                "context; doesn't need to match anything "
                                "else exactly."
                            ),
                        },
                        "role": {
                            "type": "string",
                            "enum": available_roles + ["super_admin"],
                            "description": (
                                "One of the roles defined in scopes.yaml, OR "
                                "'super_admin' (high-privilege: reads every "
                                "scope + onboarding/offboarding tools). "
                                "Granting super_admin ALSO requires "
                                "confirm_super_admin=true and a WhatsApp-"
                                "resolved target lid. See AGENTS.md for the "
                                "role-to-scope mapping."
                            ),
                        },
                        "confirm_super_admin": {
                            "type": "boolean",
                            "description": (
                                "Set true ONLY when role='super_admin' — an "
                                "explicit acknowledgement you're minting a "
                                "high-privilege super_admin. Ignored for normal "
                                "roles. The server also requires the caller to "
                                "be a super_admin and the target lid to be "
                                "WhatsApp-resolved (no typos/guesses)."
                            ),
                        },
                        "platform": {
                            "type": "string",
                            "description": (
                                "Messaging platform. Defaults to 'whatsapp' "
                                "if omitted. Currently the only supported "
                                "value, but reserved for future Teams/SMS "
                                "integrations."
                            ),
                        },
                    },
                    "required": ["sender_lid", "target_lid", "name", "role"],
                },
            },
            {
                "name": "list_allowlist",
                "description": (
                    "Show every phone number currently on the gateway "
                    "allowlist (WHATSAPP_ALLOWED_USERS), with its audit memo "
                    "and whether it's been onboarded to a role yet. Use when "
                    "the Owner asks 'who's allowed', 'show the allowlist', "
                    "'who can message the bot'. Super_admin only."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"sender_lid": _SENDER_LID_SCHEMA},
                    "required": ["sender_lid"],
                },
            },
            {
                "name": "remove_from_allowlist",
                "description": (
                    "Remove a phone number from the gateway allowlist "
                    "(WHATSAPP_ALLOWED_USERS in ~/.hermes/.env). The gateway "
                    "loads the allowlist at startup, so a reload_gateway is "
                    "required afterwards for the removal to take effect — this "
                    "tool's result reminds you. Super_admin only."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sender_lid": _SENDER_LID_SCHEMA,
                        "phone": {
                            "type": "string",
                            "description": (
                                "Phone to remove, digits only, no '+' "
                                "(e.g. 923338888888)."
                            ),
                        },
                    },
                    "required": ["sender_lid", "phone"],
                },
            },
            {
                "name": "revoke_user",
                "description": (
                    "Remove a user from scopes.yaml by lid (off-boarding). "
                    "Takes effect on their next message (mtime reload, no "
                    "restart). Removing a super_admin ALSO requires "
                    "confirm_super_admin=true and is refused if it would leave "
                    "zero super_admins. Super_admin caller only."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sender_lid": _SENDER_LID_SCHEMA,
                        "target_lid": {
                            "type": "string",
                            "description": (
                                "The lid to remove, format `<digits>@lid`."
                            ),
                        },
                        "confirm_super_admin": {
                            "type": "boolean",
                            "description": (
                                "Set true ONLY when the target is a "
                                "super_admin — explicit acknowledgement of a "
                                "high-privilege removal. Ignored for normal users."
                            ),
                        },
                    },
                    "required": ["sender_lid", "target_lid"],
                },
            },
            {
                "name": "grant_scope_access",
                "description": (
                    "Grant a user read access to ONE additional scope "
                    "beyond their role's baseline. Super_admin only. "
                    "Adds a per-user `extra_reads` entry to scopes.yaml "
                    "and re-syncs the gbrain subjects table so RLS picks "
                    "up the new scope on the user's next message."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sender_lid": _SENDER_LID_SCHEMA,
                        "target_lid": {
                            "type": "string",
                            "description": "The user receiving the grant (`<digits>@lid`).",
                        },
                        "scope": {
                            "type": "string",
                            "description": (
                                "The scope id to grant (e.g. 'leadership', "
                                "'finance', 'project_mesec'). Must be one "
                                "of the scopes defined in scopes.yaml."
                            ),
                        },
                    },
                    "required": ["sender_lid", "target_lid", "scope"],
                },
            },
            {
                "name": "revoke_scope_access",
                "description": (
                    "Revoke a previously-granted extra scope from a user. "
                    "Super_admin only. Only removes from `extra_reads` — "
                    "if the scope is part of the user's role baseline, "
                    "the call refuses with instructions to change the "
                    "role instead (chat-driven role edits are out of "
                    "scope for v1)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sender_lid": _SENDER_LID_SCHEMA,
                        "target_lid": {
                            "type": "string",
                            "description": "The user losing the grant (`<digits>@lid`).",
                        },
                        "scope": {
                            "type": "string",
                            "description": "The scope id to revoke (must be in target's extra_reads).",
                        },
                    },
                    "required": ["sender_lid", "target_lid", "scope"],
                },
            },
            {
                "name": "reload_gateway",
                "description": (
                    "Restart the Hermes gateway so a just-changed allowlist "
                    "(add_to_allowlist / remove_from_allowlist) takes effect "
                    "— the allowlist is read only at gateway startup. Call "
                    "this AFTER the Owner confirms, since it briefly drops the "
                    "WhatsApp connection (~15-20s) and ends the current "
                    "session. NOT needed for approve_user/revoke_user (those "
                    "hot-reload via scopes.yaml mtime). Super_admin only."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"sender_lid": _SENDER_LID_SCHEMA},
                    "required": ["sender_lid"],
                },
            },
        ]
    }


_LID_RE = re.compile(r"^\d{6,}@lid$")
_PHONE_RE = re.compile(r"^\d{9,15}$")  # international digits-only, no '+'

# Pending-users queue path. Default lives under the hermes-user home so the
# stdio MCP subprocess can write to it without sudo. Override via
# HERMES_PENDING_USERS_PATH for non-standard deployments.
_DEFAULT_PENDING_PATH = "/home/hermes-user/.hermes/pending-users.jsonl"


def _load_pending(path: str) -> List[Dict[str, Any]]:
    """Read pending-users.jsonl. Each line is one JSON object. Tolerates an
    empty / missing file and skips unparseable lines."""
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        return []
    except Exception as exc:
        _log("error", "pending-users load failed", path=path, error=str(exc))
        return []
    return out


def _write_pending(path: str, entries: List[Dict[str, Any]]) -> None:
    """Rewrite the JSONL file atomically. Used by both record (idempotent
    append) and approve_user (remove-on-success)."""
    contents = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)
    _atomic_write(path, contents)


def _append_phone_to_allowlist(env_path: str, phone: str, memo: Optional[str]) -> bool:
    """Add ``phone`` to WHATSAPP_ALLOWED_USERS in ``env_path``. Returns
    True if a change was made, False if the phone was already present.

    Format: ``WHATSAPP_ALLOWED_USERS=<csv-of-phones>``. We do a TEXT edit
    so any neighbouring comments/keys survive untouched. The matching key
    line is rebuilt with the new CSV; an audit comment goes on the line
    above for forensics.
    """
    with open(env_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    lines = text.splitlines()
    key = "WHATSAPP_ALLOWED_USERS"
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        if not stripped.startswith(key + "="):
            continue
        # Parse current value (after '=', strip optional surrounding quotes)
        _, _, raw_val = ln.partition("=")
        raw_val = raw_val.strip().strip('"').strip("'")
        current = [v.strip() for v in raw_val.split(",") if v.strip()]
        if phone in current:
            return False
        new_csv = ",".join(current + [phone])
        audit = f"# {ts} added via /allowlist: {phone}"
        if memo:
            audit += f" — {memo}"
        # Insert audit ABOVE the key line; replace the key line with new CSV.
        lines[i] = f"{key}={new_csv}"
        lines.insert(i, audit)
        new_text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        _atomic_write(env_path, new_text)
        return True
    # Key line not found — append a new line at EOF.
    audit = f"# {ts} added via /allowlist (key was absent): {phone}"
    if memo:
        audit += f" — {memo}"
    new_text = text + ("" if text.endswith("\n") else "\n") + audit + "\n" + f"{key}={phone}\n"
    _atomic_write(env_path, new_text)
    return True


def _atomic_write(path: str, contents: str) -> None:
    """Atomic write via tempfile + rename (same dir → single-FS guarantee)."""
    import tempfile as _tempfile
    dir_ = os.path.dirname(path) or "."
    fd, tmp_path = _tempfile.mkstemp(prefix=".tmp-", suffix=".tmp", dir=dir_)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as wh:
            wh.write(contents)
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def _existing_lids(scopes_data: Dict[str, Any]) -> set[str]:
    """Collect every lid already declared in scopes.yaml (super_admins + users)
    so we don't double-add."""
    seen: set[str] = set()
    for e in scopes_data.get("super_admins") or []:
        v = str(e.get("id", "")).strip()
        if v:
            seen.add(v)
    for e in scopes_data.get("users") or []:
        v = str(e.get("id", "")).strip()
        if v:
            seen.add(v)
    return seen


def _is_super_admin(scopes_data: Dict[str, Any], sender_lid: str) -> bool:
    for e in scopes_data.get("super_admins") or []:
        if str(e.get("id", "")).strip() == sender_lid:
            return True
    return False


def _append_user_to_scopes_yaml(path: str, target_lid: str, name: str, role: str, platform: str) -> None:
    """Append a new user line to the END of scopes.yaml.

    The `users:` section in scopes.yaml is intentionally kept last so any
    appended line is unambiguously a user entry under that mapping. We
    preserve all comments + structure by editing the file as TEXT — PyYAML's
    safe_dump would strip the documentation comments that make scopes.yaml
    readable. Writes go to a temp file in the same directory then atomically
    rename to the target so a partial write can't corrupt the in-use config.

    The added timestamp comment doubles as an audit trail in the file
    itself: `# /approve by <sender> at <UTC>`.
    """
    safe_name = name.replace('"', "'")
    line = f'  - {{ id: "{target_lid}", name: "{safe_name}", role: {role}, platform: {platform} }}\n'
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    audit = f"  # added via /approve at {ts}\n"

    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    # Ensure the file ends with a newline so our append starts on its own line.
    if not text.endswith("\n"):
        text += "\n"
    new_text = text + audit + line
    _atomic_write(path, new_text)


def _append_super_admin_to_scopes_yaml(path: str, target_lid: str, name: str, sender_lid: str) -> None:
    """Insert an entry into the `super_admins:` block (right after the header —
    list order is irrelevant). Super_admins carry NO `role:` field. TEXT edit
    preserves comments; the audit comment records who granted it and when."""
    safe_name = name.replace('"', "'")
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    audit = f"  # SUPER_ADMIN_GRANT via /approve by {sender_lid} at {ts}"
    entry = f'  - {{ id: "{target_lid}", name: "{safe_name}", platform: whatsapp }}'
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    out: List[str] = []
    inserted = False
    for ln in lines:
        out.append(ln)
        if not inserted and ln.startswith("super_admins:"):
            out.append(audit)
            out.append(entry)
            inserted = True
    if not inserted:
        raise RuntimeError("super_admins: section not found in scopes.yaml")
    _atomic_write(path, "\n".join(out) + "\n")


def _remove_super_admin_from_scopes_yaml(path: str, target_lid: str) -> bool:
    """Remove a `super_admins:` entry by lid (a `- {` line with the id and NO
    `role:` field — that's what distinguishes super_admins from users). Drops a
    preceding SUPER_ADMIN_GRANT audit comment too. Returns True if removed."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    out: List[str] = []
    removed = False
    for ln in text.splitlines():
        if (f'id: "{target_lid}"' in ln and "- {" in ln and "role:" not in ln):
            removed = True
            if out and out[-1].strip().startswith("# SUPER_ADMIN_GRANT"):
                out.pop()
            continue
        out.append(ln)
    if not removed:
        return False
    _atomic_write(path, "\n".join(out) + ("\n" if text.endswith("\n") else ""))
    return True


def _lid_is_known(session_dir: str, lid: str) -> bool:
    """True if WhatsApp has resolved this lid (a lid-mapping file exists) — i.e.
    the lid came from a real contact/message, not a typo/guess. Guards
    super_admin grants against fabricated lids."""
    digits = lid.split("@", 1)[0]
    for suffix in ("_reverse.json", ".json"):
        if os.path.exists(os.path.join(session_dir, f"lid-mapping-{digits}{suffix}")):
            return True
    return False


def _count_super_admins(scopes_data: Dict[str, Any]) -> int:
    return len([e for e in (scopes_data.get("super_admins") or []) if str(e.get("id", "")).strip()])


def _read_allowlist(env_path: str) -> tuple[List[str], Dict[str, str]]:
    """Return (phones, memos) parsed from the .env. `memos` maps phone → the
    memo text from its `# … added via /allowlist: <phone> — <memo>` comment."""
    phones: List[str] = []
    memos: Dict[str, str] = {}
    try:
        with open(env_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception:
        return phones, memos
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("WHATSAPP_ALLOWED_USERS="):
            _, _, raw = ln.partition("=")
            raw = raw.strip().strip('"').strip("'")
            phones = [v.strip() for v in raw.split(",") if v.strip()]
        elif s.startswith("#") and "added via /allowlist" in s:
            after = s.split("added via /allowlist", 1)[1].lstrip(": ").strip()
            if after:
                head, _, memo = after.partition("—")
                ph = (head.strip().split() or [""])[0]
                if ph:
                    memos[ph] = memo.strip()
    return phones, memos


def _remove_phone_from_allowlist(env_path: str, phone: str) -> bool:
    """Remove `phone` from WHATSAPP_ALLOWED_USERS + drop its audit comment.
    Returns True if removed, False if it wasn't present."""
    with open(env_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    key = "WHATSAPP_ALLOWED_USERS"
    removed = False
    out: List[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if (s.startswith("#") and "added via /allowlist" in s
                and re.search(rf"(?<!\d){re.escape(phone)}(?!\d)", s)):
            continue  # drop this phone's audit comment
        if s.startswith(key + "="):
            _, _, raw = ln.partition("=")
            current = [v.strip() for v in raw.strip().strip('"').strip("'").split(",") if v.strip()]
            if phone in current:
                removed = True
                current = [p for p in current if p != phone]
            out.append(f"{key}={','.join(current)}")
            continue
        out.append(ln)
    if not removed:
        return False
    _atomic_write(env_path, "\n".join(out) + ("\n" if text.endswith("\n") else ""))
    return True


def _remove_user_from_scopes_yaml(path: str, target_lid: str) -> bool:
    """Remove the `users:` entry whose id == target_lid (TEXT edit; preserves
    comments). Drops a preceding `# added via /approve` audit line too. Only
    matches lines that carry a `role:` field, so super_admins entries (which
    have none) are never removed from chat. Returns True if removed."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    out: List[str] = []
    removed = False
    for ln in text.splitlines():
        if (f'id: "{target_lid}"' in ln and "role:" in ln and "- {" in ln):
            removed = True
            if out and out[-1].strip().startswith("# added via /approve"):
                out.pop()
            continue
        out.append(ln)
    if not removed:
        return False
    _atomic_write(path, "\n".join(out) + ("\n" if text.endswith("\n") else ""))
    return True


def _phone_to_lid(session_dir: str, phone: str) -> Optional[str]:
    """Resolve a phone → its WhatsApp lid via the bridge's reverse-mapping
    files (`lid-mapping-<lid>_reverse.json` contains the phone). The lid is in
    the filename. Returns `<digits>` or None."""
    try:
        for fn in os.listdir(session_dir):
            if fn.startswith("lid-mapping-") and fn.endswith("_reverse.json"):
                try:
                    with open(os.path.join(session_dir, fn), "r", encoding="utf-8") as fh:
                        if phone in fh.read():
                            return fn[len("lid-mapping-"):-len("_reverse.json")]
                except Exception:
                    continue
    except Exception:
        return None
    return None


def _compute_pending(scopes_data: Dict[str, Any], env_path: str, session_dir: str) -> List[Dict[str, Any]]:
    """Allowlisted phones whose resolved lid is NOT yet in scopes.yaml.
    Independent of record_pending_user — surfaces a user even if the agent
    never logged them. Returns [{phone, lid|None}]."""
    phones, _memos = _read_allowlist(env_path)
    known = _existing_lids(scopes_data)
    out: List[Dict[str, Any]] = []
    for ph in phones:
        if ph == "*":
            continue
        lid_digits = _phone_to_lid(session_dir, ph)
        lid = f"{lid_digits}@lid" if lid_digits else None
        if lid and lid in known:
            continue  # already onboarded
        out.append({"phone": ph, "lid": lid})
    return out


def _session_dir_for(env_path: str) -> str:
    """The bridge's WhatsApp session dir (holds the lid mappings), derived
    from the .env location: <…/.hermes>/whatsapp/session."""
    return os.path.join(os.path.dirname(env_path) or ".", "whatsapp", "session")


def _restart_gateway_detached() -> bool:
    """Restart hermes.service AFTER this response is flushed — detached + a
    short delay so the MCP reply reaches the user before the gateway drops.
    Uses passwordless `sudo -n systemctl restart hermes`."""
    import subprocess
    try:
        subprocess.Popen(
            ["bash", "-c", "sleep 3; sudo -n systemctl restart hermes"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception as exc:
        _log("error", "restart_gateway spawn failed", error=str(exc))
        return False




# ─── Per-user extra_reads (chat-driven scope grants) ───────────────────────
# Adds / removes the optional `extra_reads: [scope_id]` field on a user
# entry in scopes.yaml. We use ruamel.yaml round-trip mode so comments
# above + alongside user entries are preserved across the edit — the
# text-edit pattern the other helpers use can't safely insert into the
# middle of an inline-mapping line like `- { id: ..., name: ..., role: ... }`.
def _scopes_yaml_grant_extra_read(path: str, target_lid: str, scope_id: str) -> str:
    """Add `scope_id` to target_lid's extra_reads in scopes.yaml.

    Returns:
        'added'      — scope appended to extra_reads (or extra_reads created)
        'already'    — scope was already present in extra_reads
        'role_baseline' — scope is part of target's role.reads (no-op grant)
        'super_admin'   — target is super_admin (no-op grant, already god-mode)
        'not_found'  — target lid not in users[] or super_admins[]
        'unknown_scope' — scope_id not defined in scopes.yaml's scopes:
    """
    from ruamel.yaml import YAML
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.load(fh)
    scopes_map = (data.get("scopes") or {})
    if scope_id not in scopes_map:
        return "unknown_scope"
    # super_admin entries
    for entry in (data.get("super_admins") or []):
        if str(entry.get("id", "")).strip() == target_lid:
            return "super_admin"
    # user entries
    users = data.get("users") or []
    target_entry = None
    for entry in users:
        if str(entry.get("id", "")).strip() == target_lid:
            target_entry = entry
            break
    if target_entry is None:
        return "not_found"
    # Is the scope already in the role's baseline?
    role = str(target_entry.get("role", "")).strip()
    role_reads = (((data.get("roles") or {}).get(role) or {}).get("reads") or [])
    if "all" in role_reads or scope_id in role_reads:
        return "role_baseline"
    extra = target_entry.get("extra_reads")
    if extra is None:
        target_entry["extra_reads"] = [scope_id]
    elif scope_id in extra:
        return "already"
    else:
        extra.append(scope_id)
    tmp = path + ".tmp-grant"
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh)
    os.replace(tmp, path)
    return "added"


def _scopes_yaml_revoke_extra_read(path: str, target_lid: str, scope_id: str) -> str:
    """Remove `scope_id` from target_lid's extra_reads.

    Returns:
        'removed'     — scope dropped from extra_reads
        'not_in_extra' — scope wasn't in extra_reads (caller decides whether
                         that's an error: e.g. baseline-revoke isn't supported)
        'role_baseline' — scope is in role.reads, can't be revoked here
        'super_admin'   — target is super_admin (revoke from super_admin
                          status is a different op)
        'not_found'   — target lid missing from users[]
        'unknown_scope' — scope_id not defined
    """
    from ruamel.yaml import YAML
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.load(fh)
    scopes_map = (data.get("scopes") or {})
    if scope_id not in scopes_map:
        return "unknown_scope"
    for entry in (data.get("super_admins") or []):
        if str(entry.get("id", "")).strip() == target_lid:
            return "super_admin"
    users = data.get("users") or []
    target_entry = None
    for entry in users:
        if str(entry.get("id", "")).strip() == target_lid:
            target_entry = entry
            break
    if target_entry is None:
        return "not_found"
    role = str(target_entry.get("role", "")).strip()
    role_reads = (((data.get("roles") or {}).get(role) or {}).get("reads") or [])
    if scope_id in role_reads or "all" in role_reads:
        # In baseline; chat-driven revoke can't touch role config (that
        # would affect every other user with the same role). Caller
        # surfaces "change the role to revoke".
        return "role_baseline"
    extra = target_entry.get("extra_reads") or []
    if scope_id not in extra:
        return "not_in_extra"
    extra.remove(scope_id)
    # If extra_reads becomes empty, drop the key entirely to keep the
    # yaml clean (matches how the file looked pre-grant).
    if not extra:
        target_entry.pop("extra_reads", None)
    tmp = path + ".tmp-revoke"
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh)
    os.replace(tmp, path)
    return "removed"


# ─── Subjects-table sync (RFC 8693 subject registry, gbrain v117) ──────────
# Every chat-driven scopes.yaml edit (approve / revoke / grant / revoke
# scope) must also propagate to gbrain's `subjects` table so the next
# token-exchange call resolves the right `allowed_sources`. Without this
# the SQL-layer RLS keeps using the pre-edit scope until someone re-runs
# sync-subjects-to-gbrain.py by hand — the exact "scopes.yaml says one
# thing, gbrain serves another" drift class.
_SYNC_SUBJECTS_SCRIPT = "/datadrive/hermes/workspace/scripts/sync-subjects-to-gbrain.py"


def _sync_subjects_to_gbrain(scopes_yaml_path: str) -> Optional[str]:
    """Run the subjects sync. Returns None on success, or an error string.
    NEVER raises — a sync failure must not mask the scopes.yaml write."""
    import subprocess
    if not os.path.exists(_SYNC_SUBJECTS_SCRIPT):
        return (
            f"subjects-sync script not found at {_SYNC_SUBJECTS_SCRIPT} — "
            f"the gbrain subjects table was NOT updated. Run the sync "
            f"manually so per-user RLS reflects the new scope state."
        )
    try:
        proc = subprocess.run(
            ["python3", _SYNC_SUBJECTS_SCRIPT, "--scopes-yaml", scopes_yaml_path],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PATH": "/home/hermes-user/.bun/bin:" + os.environ.get("PATH", "")},
        )
    except subprocess.TimeoutExpired:
        return "subjects sync timed out (>30s) — re-run manually."
    except Exception as exc:
        return f"subjects sync raised: {exc!r}"
    if proc.returncode != 0:
        return (
            f"subjects sync exited {proc.returncode}: "
            f"stderr={proc.stderr.strip()[:300]!r}"
        )
    return None


def _remove_subject_from_gbrain(target_lid: str) -> Optional[str]:
    """Soft-delete the subject row for `target_lid` in gbrain.
    Returns None on success or an error string. NEVER raises."""
    import subprocess
    try:
        proc = subprocess.run(
            ["gbrain", "auth", "subjects", "remove", target_lid],
            capture_output=True, text=True, timeout=15,
            env={**os.environ, "PATH": "/home/hermes-user/.bun/bin:" + os.environ.get("PATH", "")},
        )
    except subprocess.TimeoutExpired:
        return "gbrain subjects remove timed out (>15s)"
    except FileNotFoundError:
        return "gbrain CLI not found on PATH — subject row NOT soft-deleted"
    except Exception as exc:
        return f"gbrain subjects remove raised: {exc!r}"
    if proc.returncode != 0:
        return (
            f"gbrain subjects remove exited {proc.returncode}: "
            f"stderr={proc.stderr.strip()[:300]!r}"
        )
    return None


def _handle_record_pending_user(
    args: Dict[str, Any],
    pending_path: str,
) -> Dict[str, Any]:
    """Append the new user's lid to the pending queue. No super_admin
    gate — the bot calls this for every None-tier turn."""
    target_lid = str(args.get("target_lid") or "").strip()
    snippet = str(args.get("first_message_snippet") or "").strip()[:200]

    if not target_lid:
        return {"isError": True, "content": [{"type": "text", "text": "target_lid is required."}]}
    if not _LID_RE.match(target_lid):
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                f"target_lid format invalid: '{target_lid}'. Expected "
                f"`<digits>@lid`."
            )}],
        }

    entries = _load_pending(pending_path)
    # Idempotent — don't duplicate.
    if any(e.get("lid") == target_lid for e in entries):
        return {"content": [{"type": "text", "text": (
            f"📋 Pending user {target_lid} already queued (no duplicate)."
        )}]}
    entry = {
        "lid": target_lid,
        "first_seen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "snippet": snippet,
    }
    entries.append(entry)
    try:
        _write_pending(pending_path, entries)
    except Exception as exc:
        _log("error", "record_pending_user write failed", lid=target_lid, error=str(exc))
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                f"Failed to record pending user: {exc}. Bot will still "
                f"reply welcome to the user, but admin won't see them in "
                f"the queue. Check file permissions."
            )}],
        }
    _log("info", "pending user recorded", lid=target_lid, snippet=snippet)
    return {"content": [{"type": "text", "text": (
        f"Recorded pending user {target_lid}. {len(entries)} total in queue."
    )}]}


def _handle_list_pending_users(
    scopes_data: Dict[str, Any],
    sender_lid: str,
    pending_path: str,
    env_path: str,
    session_dir: str,
) -> Dict[str, Any]:
    """Pending = the recorded queue UNION allowlisted numbers not yet in
    scopes.yaml (computed, lid-resolved). The computed half is the robust fix:
    it surfaces a user even if the agent never called record_pending_user.
    Super_admin only."""
    if not _is_super_admin(scopes_data, sender_lid):
        _log("warn", "list_pending_users denied", sender=sender_lid)
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                "Pending-users queue is restricted to super_admins."
            )}],
        }
    recorded = _load_pending(pending_path)
    computed = _compute_pending(scopes_data, env_path, session_dir)

    by_lid: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for e in recorded:
        lid = e.get("lid")
        if not lid:
            continue
        rec = by_lid.setdefault(lid, {"lid": lid})
        rec["first_seen_at"] = e.get("first_seen_at")
        rec["snippet"] = e.get("snippet")
        if lid not in order:
            order.append(lid)
    no_lid_yet: List[Dict[str, Any]] = []
    for c in computed:
        lid = c.get("lid")
        if lid:
            rec = by_lid.setdefault(lid, {"lid": lid})
            rec["phone"] = c.get("phone")
            if lid not in order:
                order.append(lid)
        else:
            no_lid_yet.append(c)

    if not order and not no_lid_yet:
        return {"content": [{"type": "text", "text": (
            "📭 No pending users — everyone on the allowlist already has a "
            "role in scopes.yaml, and nobody new is queued."
        )}]}

    lines = [f"📋 {len(order) + len(no_lid_yet)} pending:\n"]
    for lid in order:
        e = by_lid[lid]
        line = "- `%s`" % lid
        if e.get("phone"):
            line += " (phone %s)" % e["phone"]
        if e.get("first_seen_at"):
            line += " — first seen %s" % e["first_seen_at"]
        else:
            line += " — allowlisted, not yet roled"
        snip = e.get("snippet")
        if snip:
            line += ' — "%s"' % snip[:80]
        lines.append(line)
    for c in no_lid_yet:
        lines.append(
            "- phone %s — allowlisted but hasn't messaged yet "
            "(lid surfaces on first message)" % c["phone"]
        )
    lines.append("\nApprove each with: `/approve <lid> as <role> name <name>`")
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


def _remove_from_pending(pending_path: str, target_lid: str) -> bool:
    """Drop the matching lid from the queue. Returns True if anything
    was removed. Used by approve_user to clean up on success."""
    entries = _load_pending(pending_path)
    before = len(entries)
    entries = [e for e in entries if e.get("lid") != target_lid]
    if len(entries) == before:
        return False
    try:
        _write_pending(pending_path, entries)
    except Exception as exc:
        _log("warn", "remove_from_pending write failed", lid=target_lid, error=str(exc))
        return False
    return True


def _handle_add_to_allowlist(
    args: Dict[str, Any],
    scopes_data: Dict[str, Any],
    sender_lid: str,
    env_path: str,
) -> Dict[str, Any]:
    """Add a phone to WHATSAPP_ALLOWED_USERS. Super_admin only."""
    phone = str(args.get("phone") or "").strip().lstrip("+").replace(" ", "")
    memo = str(args.get("memo") or "").strip() or None

    if not _is_super_admin(scopes_data, sender_lid):
        _log("warn", "add_to_allowlist denied", sender=sender_lid, phone=phone)
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                "Allowlist is restricted to super_admins. Your sender "
                "identity does not have super_admin privileges."
            )}],
        }
    if not phone:
        return {"isError": True, "content": [{"type": "text", "text": "phone is required."}]}
    if not _PHONE_RE.match(phone):
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                f"phone format invalid: '{phone}'. Expected digits only, "
                f"9-15 chars, no '+' (e.g., 923333717117 for "
                f"+92 333 3717117)."
            )}],
        }
    if not env_path:
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                "Allowlist file path not configured (HERMES_HERMES_ENV_PATH). "
                "Ask Harris to fix the deployment."
            )}],
        }

    try:
        changed = _append_phone_to_allowlist(env_path, phone, memo)
    except Exception as exc:
        _log("error", "add_to_allowlist write failed", sender=sender_lid, phone=phone, error=str(exc))
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                f"Failed to write to .env: {exc}. Check file permissions."
            )}],
        }

    if not changed:
        return {"content": [{"type": "text", "text": (
            f"📋 Phone {phone} was already on the allowlist — no change."
        )}]}

    _log("info", "add_to_allowlist success", sender=sender_lid, phone=phone, memo=memo)
    return {"content": [{"type": "text", "text": (
        f"✅ Phone added to gateway allowlist.\n\n"
        f"- phone: {phone}\n"
        + (f"- memo: {memo}\n\n" if memo else "\n")
        + f"⚠️ IMPORTANT: the gateway only reads the allowlist at startup, so "
        f"this number is NOT live yet — their messages will still be rejected. "
        f"Confirm with the Owner, then call `reload_gateway` to restart the "
        f"gateway (~15-20s) so the new number is accepted.\n\n"
        f"After the reload: ask {phone} to send any WhatsApp message. Their "
        f"verified `lid` then surfaces (and they show in `list_pending_users`). "
        f"Run `approve_user` / `/approve <lid> as <role> name <name>` to assign "
        f"their role. Until then they're at None tier (general help only)."
    )}]}


def _handle_approve_user(
    args: Dict[str, Any],
    scopes_data: Dict[str, Any],
    sender_lid: str,
    role: Optional[str],
    scopes_yaml_path: str,
    session_dir: str,
) -> Dict[str, Any]:
    target_lid = str(args.get("target_lid") or "").strip()
    name = str(args.get("name") or "").strip()
    requested_role = str(args.get("role") or "").strip()
    platform = str(args.get("platform") or "whatsapp").strip().lower() or "whatsapp"

    # ── Authorization: only super_admins may onboard ───────────────────────
    if not _is_super_admin(scopes_data, sender_lid):
        _log("warn", "approve_user denied (not super_admin)", sender=sender_lid, target=target_lid)
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                "Onboarding is restricted to super_admins. Your sender "
                "identity does not have super_admin privileges, so I "
                "can't add a new user via chat. Ask Harris or Shahzaib "
                "(the dev super_admins) to run /approve, or have Iyad "
                "do it once the CEO super_admin is configured."
            )}],
        }

    # ── Validate inputs ────────────────────────────────────────────────────
    if not target_lid or not name or not requested_role:
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                "approve_user requires target_lid, name, and role."
            )}],
        }

    if not _LID_RE.match(target_lid):
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                f"target_lid format invalid: '{target_lid}'. Expected "
                f"`<digits>@lid` (the WhatsApp privacy-mode identifier). "
                f"Capture the lid from the verified-sender marker on a "
                f"message the new user already sent."
            )}],
        }

    # ── super_admin: guarded chat path (decision 2026-06-18) ───────────────
    # High-privilege. Caller is already super_admin (gated above). Extra rails:
    # explicit confirm flag, the target lid must be WhatsApp-resolved (not a
    # typo/guess), idempotency, and a distinct SUPER_ADMIN_GRANT audit line.
    if requested_role == "super_admin":
        if not bool(args.get("confirm_super_admin")):
            return {"isError": True, "content": [{"type": "text", "text": (
                "Granting super_admin is high-privilege — a super_admin reads "
                "every scope, runs on/off-boarding, and can mint other "
                "super_admins. Re-issue with confirm_super_admin=true to proceed."
            )}]}
        if not _lid_is_known(session_dir, target_lid):
            return {"isError": True, "content": [{"type": "text", "text": (
                f"Won't grant super_admin to {target_lid}: WhatsApp hasn't "
                f"resolved this lid (no mapping on file), so it may be a typo "
                f"or guess. Have them send one message first so the lid is "
                f"verified, then retry."
            )}]}
        if _is_super_admin(scopes_data, target_lid):
            return {"content": [{"type": "text", "text": (
                f"{target_lid} is already a super_admin — no change."
            )}]}
        try:
            _append_super_admin_to_scopes_yaml(scopes_yaml_path, target_lid, name, sender_lid)
        except Exception as exc:
            _log("error", "super_admin grant write failed", sender=sender_lid, target=target_lid, error=str(exc))
            return {"isError": True, "content": [{"type": "text", "text": (
                f"Failed to write scopes.yaml: {exc}. NOT granted."
            )}]}
        pending_path = os.environ.get("HERMES_PENDING_USERS_PATH") or _DEFAULT_PENDING_PATH
        _remove_from_pending(pending_path, target_lid)
        _log("warn", "SUPER_ADMIN_GRANT", sender=sender_lid, target_lid=target_lid, name=name)
        _sync_err = _sync_subjects_to_gbrain(scopes_yaml_path)
        _sync_note = f"\n\n⚠️ Subjects sync warning: {_sync_err}" if _sync_err else ""
        return {"content": [{"type": "text", "text": (
            f"✅ SUPER_ADMIN granted.{_sync_note}\n\n"
            f"- lid: `{target_lid}`\n- name: {name}\n- role: super_admin\n\n"
            f"Effective on their next message (scopes.yaml mtime reload, no "
            f"restart). Logged as SUPER_ADMIN_GRANT.\n\n"
            f"Reminder: `scp /opt/hermes/workspace/scopes.yaml ./azure/config/` "
            f"to keep the repo in sync."
        )}]}

    available_roles = _role_list(scopes_data)
    if requested_role not in available_roles:
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                f"role '{requested_role}' is not defined in scopes.yaml. "
                f"Allowed roles: {available_roles}. For super_admin, pass "
                f"role='super_admin' with confirm_super_admin=true."
            )}],
        }

    if platform not in {"whatsapp"}:
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                f"platform '{platform}' not supported yet (only 'whatsapp'). "
                f"Future Teams/SMS support will land in Phase 8."
            )}],
        }

    # ── Idempotency: don't double-add ──────────────────────────────────────
    existing = _existing_lids(scopes_data)
    if target_lid in existing:
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                f"User {target_lid} is already in scopes.yaml. To change "
                f"their role, SSH in and edit scopes.yaml — chat-driven "
                f"role changes are intentionally not supported in v0.1 "
                f"(avoid privilege-escalation paths through chat-only ops)."
            )}],
        }

    # ── Write ──────────────────────────────────────────────────────────────
    try:
        _append_user_to_scopes_yaml(scopes_yaml_path, target_lid, name, requested_role, platform)
    except Exception as exc:
        _log("error", "approve_user write failed", sender=sender_lid, target=target_lid, error=str(exc))
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                f"Failed to write to scopes.yaml: {exc}. The user was "
                f"NOT onboarded. Check file permissions on the VM."
            )}],
        }

    # Drop from the pending-users queue (no-op if they weren't queued).
    pending_path = os.environ.get("HERMES_PENDING_USERS_PATH") or _DEFAULT_PENDING_PATH
    _remove_from_pending(pending_path, target_lid)

    _sync_err = _sync_subjects_to_gbrain(scopes_yaml_path)
    _sync_note = f"\n\n⚠️ Subjects sync warning: {_sync_err}" if _sync_err else ""
    _log(
        "info",
        "approve_user success",
        sender=sender_lid,
        target_lid=target_lid,
        name=name,
        role=requested_role,
        platform=platform,
    )
    return {
        "content": [{"type": "text", "text": (
            f"✅ User onboarded.\n\n"
            f"- lid: `{target_lid}`\n"
            f"- name: {name}\n"
            f"- role: {requested_role}\n"
            f"- platform: {platform}{_sync_note}\n\n"
            f"The change takes effect immediately on their next message — "
            f"hermes reloads scopes.yaml when the file mtime advances. "
            f"No restart needed. Removed from pending queue if they were there.\n\n"
            f"Note: the VM's scopes.yaml has diverged from the repo. Harris "
            f"should `scp /opt/hermes/workspace/scopes.yaml ./azure/config/` "
            f"before the next deploy to keep the repo in sync."
        )}],
    }


def _handle_list_allowlist(
    scopes_data: Dict[str, Any],
    sender_lid: str,
    env_path: str,
    session_dir: str,
) -> Dict[str, Any]:
    """Show the gateway allowlist with memo + onboarding status. Super_admin only."""
    if not _is_super_admin(scopes_data, sender_lid):
        return {"isError": True, "content": [{"type": "text", "text": (
            "The allowlist is restricted to super_admins."
        )}]}
    phones, memos = _read_allowlist(env_path)
    known = _existing_lids(scopes_data)
    if not phones:
        return {"content": [{"type": "text", "text": "🔒 Allowlist is empty."}]}
    lines = ["🔒 %d number(s) on the allowlist:\n" % len(phones)]
    for ph in phones:
        if ph == "*":
            lines.append("- `*` — OPEN BOT (everyone allowed)")
            continue
        lid_digits = _phone_to_lid(session_dir, ph)
        lid = ("%s@lid" % lid_digits) if lid_digits else None
        if lid and lid in known:
            status = "✅ onboarded"
        elif lid:
            status = "⏳ messaged, no role yet"
        else:
            status = "• not messaged yet"
        line = "- `%s`" % ph
        if memos.get(ph):
            line += " — %s" % memos[ph]
        line += "  [%s]" % status
        lines.append(line)
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


def _handle_remove_from_allowlist(
    args: Dict[str, Any],
    scopes_data: Dict[str, Any],
    sender_lid: str,
    env_path: str,
) -> Dict[str, Any]:
    """Remove a phone from the allowlist. Super_admin only. Needs reload_gateway."""
    phone = str(args.get("phone") or "").strip().lstrip("+").replace(" ", "")
    if not _is_super_admin(scopes_data, sender_lid):
        return {"isError": True, "content": [{"type": "text", "text": (
            "The allowlist is restricted to super_admins."
        )}]}
    if not phone:
        return {"isError": True, "content": [{"type": "text", "text": "phone is required."}]}
    if not _PHONE_RE.match(phone):
        return {"isError": True, "content": [{"type": "text", "text": (
            "phone format invalid: '%s'. Digits only, no '+'." % phone
        )}]}
    try:
        removed = _remove_phone_from_allowlist(env_path, phone)
    except Exception as exc:
        _log("error", "remove_from_allowlist failed", sender=sender_lid, phone=phone, error=str(exc))
        return {"isError": True, "content": [{"type": "text", "text": (
            "Failed to edit .env: %s. Check file permissions." % exc
        )}]}
    if not removed:
        return {"content": [{"type": "text", "text": (
            "📋 %s was not on the allowlist — nothing to remove." % phone
        )}]}
    _log("info", "remove_from_allowlist success", sender=sender_lid, phone=phone)
    return {"content": [{"type": "text", "text": (
        "✅ Removed %s from the allowlist.\n\n"
        "⚠️ The gateway only reads the allowlist at startup, so this isn't "
        "live yet. Confirm with the Owner, then call `reload_gateway` to "
        "restart the gateway (~15-20s, ends this session) so the removal "
        "takes effect." % phone
    )}]}


def _handle_revoke_user(
    args: Dict[str, Any],
    scopes_data: Dict[str, Any],
    sender_lid: str,
    scopes_yaml_path: str,
    pending_path: str,
) -> Dict[str, Any]:
    """Remove a non-super_admin user from scopes.yaml. Super_admin only."""
    target_lid = str(args.get("target_lid") or "").strip()
    if not _is_super_admin(scopes_data, sender_lid):
        return {"isError": True, "content": [{"type": "text", "text": (
            "Off-boarding is restricted to super_admins."
        )}]}
    if not _LID_RE.match(target_lid):
        return {"isError": True, "content": [{"type": "text", "text": (
            "target_lid format invalid: '%s'. Expected `<digits>@lid`." % target_lid
        )}]}
    target_is_sa = _is_super_admin(scopes_data, target_lid)
    if target_is_sa:
        # Guarded super_admin removal (decision 2026-06-18).
        if not bool(args.get("confirm_super_admin")):
            return {"isError": True, "content": [{"type": "text", "text": (
                "%s is a super_admin. Removing a super_admin is high-privilege "
                "— re-issue with confirm_super_admin=true to proceed." % target_lid
            )}]}
        if _count_super_admins(scopes_data) <= 1:
            return {"isError": True, "content": [{"type": "text", "text": (
                "Refusing: %s is the LAST super_admin — removing it would lock "
                "everyone out of admin ops. Add another super_admin first." % target_lid
            )}]}
    try:
        if target_is_sa:
            removed = _remove_super_admin_from_scopes_yaml(scopes_yaml_path, target_lid)
        else:
            removed = _remove_user_from_scopes_yaml(scopes_yaml_path, target_lid)
    except Exception as exc:
        _log("error", "revoke_user failed", sender=sender_lid, target=target_lid, error=str(exc))
        return {"isError": True, "content": [{"type": "text", "text": (
            "Failed to edit scopes.yaml: %s. User NOT removed." % exc
        )}]}
    if not removed:
        return {"content": [{"type": "text", "text": (
            "📋 %s wasn't found in scopes.yaml — nothing to revoke." % target_lid
        )}]}
    _remove_from_pending(pending_path, target_lid)
    # Soft-delete the gbrain subject row so future token-exchange for this
    # lid returns invalid_grant (subject no longer resolvable). Existing
    # tokens already minted become invalid at next verify (gbrain joins
    # subjects with deleted_at IS NULL — see verifyAccessToken).
    _subject_err = _remove_subject_from_gbrain(target_lid)
    _subject_note = f"\n\n⚠️ Subject soft-delete warning: {_subject_err}" if _subject_err else ""
    _log(
        "warn" if target_is_sa else "info",
        "SUPER_ADMIN_REVOKE" if target_is_sa else "revoke_user success",
        sender=sender_lid, target=target_lid,
    )
    kind = "SUPER_ADMIN" if target_is_sa else "user"
    return {"content": [{"type": "text", "text": (
        "✅ Revoked %s `%s` — removed from scopes.yaml.%s Takes effect on their "
        "next message (mtime reload, no restart).%s\n\n"
        "Note: the VM's scopes.yaml has diverged from the repo — Harris should "
        "`scp /opt/hermes/workspace/scopes.yaml ./azure/config/` before the "
        "next deploy. (Their allowlist entry, if any, is separate — use "
        "`remove_from_allowlist` to drop that too.)" % (
            kind, target_lid, _subject_note,
            " Logged as SUPER_ADMIN_REVOKE." if target_is_sa else "",
        )
    )}]}



def _handle_grant_scope_access(
    args: Dict[str, Any],
    scopes_data: Dict[str, Any],
    sender_lid: str,
    scopes_yaml_path: str,
) -> Dict[str, Any]:
    """Grant a user read access to one additional scope. Super_admin only.

    Writes to scopes.yaml as a per-user `extra_reads` entry, then
    re-syncs the gbrain subjects table so the new scope shows up in
    the target's allowed_sources at the SQL-RLS layer on the next
    token-exchange call. Role baseline reads are unchanged — this only
    edits the per-user override list.
    """
    target_lid = str(args.get("target_lid") or "").strip()
    scope_id = str(args.get("scope") or "").strip()
    if not _is_super_admin(scopes_data, sender_lid):
        return {"isError": True, "content": [{"type": "text", "text": (
            "Scope grants are restricted to super_admins. Your sender "
            "identity does not have super_admin privileges."
        )}]}
    if not _LID_RE.match(target_lid):
        return {"isError": True, "content": [{"type": "text", "text": (
            f"target_lid format invalid: '{target_lid}'. Expected `<digits>@lid`."
        )}]}
    if not scope_id:
        return {"isError": True, "content": [{"type": "text", "text": (
            "scope is required (e.g. 'leadership', 'finance', 'project_mesec')."
        )}]}
    try:
        result = _scopes_yaml_grant_extra_read(scopes_yaml_path, target_lid, scope_id)
    except Exception as exc:
        _log("error", "grant_scope_access write failed",
             sender=sender_lid, target=target_lid, scope=scope_id, error=str(exc))
        return {"isError": True, "content": [{"type": "text", "text": (
            f"Failed to edit scopes.yaml: {exc}. No grant applied."
        )}]}

    if result == "unknown_scope":
        available = sorted((scopes_data.get("scopes") or {}).keys())
        return {"isError": True, "content": [{"type": "text", "text": (
            f"scope '{scope_id}' is not defined in scopes.yaml. "
            f"Available scopes: {available}."
        )}]}
    if result == "not_found":
        return {"isError": True, "content": [{"type": "text", "text": (
            f"User {target_lid} is not in scopes.yaml. Onboard them first "
            f"via approve_user, then re-issue the grant."
        )}]}
    if result == "super_admin":
        return {"content": [{"type": "text", "text": (
            f"{target_lid} is a super_admin and already has access to every scope. "
            f"No grant needed."
        )}]}
    if result == "role_baseline":
        role = ""
        for e in (scopes_data.get("users") or []):
            if str(e.get("id", "")).strip() == target_lid:
                role = str(e.get("role", "")).strip()
                break
        return {"content": [{"type": "text", "text": (
            f"{target_lid}'s role ('{role}') already includes '{scope_id}' in its "
            f"baseline reads. No grant needed."
        )}]}
    if result == "already":
        return {"content": [{"type": "text", "text": (
            f"{target_lid} already has '{scope_id}' in their extra_reads. No change."
        )}]}
    # result == "added"
    sync_err = _sync_subjects_to_gbrain(scopes_yaml_path)
    sync_note = (f"\n\n⚠️ Subjects sync warning: {sync_err}" if sync_err else "")
    _log("info", "grant_scope_access",
         sender=sender_lid, target=target_lid, scope=scope_id)
    return {"content": [{"type": "text", "text": (
        f"✅ Granted `{scope_id}` read access to `{target_lid}`.{sync_note}\n\n"
        f"Effective on their next message (gbrain RLS re-resolves on each "
        f"token-exchange). Stored as `extra_reads` in scopes.yaml so it "
        f"survives role changes.\n\n"
        f"Note: the VM's scopes.yaml has diverged from the repo — Harris "
        f"should `scp /opt/hermes/workspace/scopes.yaml ./azure/config/` "
        f"before the next deploy."
    )}]}


def _handle_revoke_scope_access(
    args: Dict[str, Any],
    scopes_data: Dict[str, Any],
    sender_lid: str,
    scopes_yaml_path: str,
) -> Dict[str, Any]:
    """Revoke one extra scope from a user. Super_admin only.

    Only removes from `extra_reads`. If the scope is part of the role's
    baseline reads, the request is refused with instructions to change
    the user's role instead — chat-driven role-config edits are out of
    scope for v1 (they'd ripple to every user with that role).
    """
    target_lid = str(args.get("target_lid") or "").strip()
    scope_id = str(args.get("scope") or "").strip()
    if not _is_super_admin(scopes_data, sender_lid):
        return {"isError": True, "content": [{"type": "text", "text": (
            "Scope revokes are restricted to super_admins."
        )}]}
    if not _LID_RE.match(target_lid):
        return {"isError": True, "content": [{"type": "text", "text": (
            f"target_lid format invalid: '{target_lid}'."
        )}]}
    if not scope_id:
        return {"isError": True, "content": [{"type": "text", "text": (
            "scope is required."
        )}]}
    try:
        result = _scopes_yaml_revoke_extra_read(scopes_yaml_path, target_lid, scope_id)
    except Exception as exc:
        _log("error", "revoke_scope_access write failed",
             sender=sender_lid, target=target_lid, scope=scope_id, error=str(exc))
        return {"isError": True, "content": [{"type": "text", "text": (
            f"Failed to edit scopes.yaml: {exc}. No revoke applied."
        )}]}

    if result == "unknown_scope":
        available = sorted((scopes_data.get("scopes") or {}).keys())
        return {"isError": True, "content": [{"type": "text", "text": (
            f"scope '{scope_id}' is not defined in scopes.yaml. Available: {available}."
        )}]}
    if result == "not_found":
        return {"isError": True, "content": [{"type": "text", "text": (
            f"User {target_lid} is not in scopes.yaml."
        )}]}
    if result == "super_admin":
        return {"isError": True, "content": [{"type": "text", "text": (
            f"{target_lid} is a super_admin — their scope access comes from "
            f"super_admin status, not extra_reads. To remove their access, use "
            f"revoke_user (will demote and trigger the soft-delete on their "
            f"subjects row)."
        )}]}
    if result == "role_baseline":
        role = ""
        for e in (scopes_data.get("users") or []):
            if str(e.get("id", "")).strip() == target_lid:
                role = str(e.get("role", "")).strip()
                break
        return {"isError": True, "content": [{"type": "text", "text": (
            f"`{scope_id}` is part of `{target_lid}`'s role ('{role}') baseline "
            f"reads — chat-driven revoke can't touch role definitions (would "
            f"ripple to every user with that role). To remove this access, "
            f"revoke_user and re-approve with a narrower role, or edit "
            f"scopes.yaml's roles: section by hand."
        )}]}
    if result == "not_in_extra":
        return {"content": [{"type": "text", "text": (
            f"`{target_lid}` doesn't have `{scope_id}` as an extra_read — "
            f"nothing to revoke."
        )}]}
    # result == "removed"
    sync_err = _sync_subjects_to_gbrain(scopes_yaml_path)
    sync_note = (f"\n\n⚠️ Subjects sync warning: {sync_err}" if sync_err else "")
    _log("info", "revoke_scope_access",
         sender=sender_lid, target=target_lid, scope=scope_id)
    return {"content": [{"type": "text", "text": (
        f"✅ Revoked `{scope_id}` extra read from `{target_lid}`.{sync_note}\n\n"
        f"Effective on their next message. Their role baseline reads are "
        f"unchanged.\n\n"
        f"Note: the VM's scopes.yaml has diverged from the repo — Harris should "
        f"`scp /opt/hermes/workspace/scopes.yaml ./azure/config/` before the "
        f"next deploy."
    )}]}


def _handle_reload_gateway(
    scopes_data: Dict[str, Any],
    sender_lid: str,
) -> Dict[str, Any]:
    """Restart hermes so allowlist changes load. Super_admin only."""
    if not _is_super_admin(scopes_data, sender_lid):
        return {"isError": True, "content": [{"type": "text", "text": (
            "Restarting the gateway is restricted to super_admins."
        )}]}
    ok = _restart_gateway_detached()
    if not ok:
        return {"isError": True, "content": [{"type": "text", "text": (
            "Couldn't trigger the restart. Restart manually via SSH: "
            "`sudo systemctl restart hermes`."
        )}]}
    _log("info", "reload_gateway triggered", sender=sender_lid)
    return {"content": [{"type": "text", "text": (
        "♻️ Restarting the gateway now — it reloads the allowlist on the way "
        "back up (~15-20s). This session ends; send a new message once it's "
        "reconnected and the latest allowlist will be in effect."
    )}]}


def _handle_tools_call(
    req: Dict[str, Any],
    scopes_data: Dict[str, Any],
    bearers: Dict[str, str],
    gbrain_url: str,
    timeout: float,
    scopes_yaml_path: str,
    env_path: str,
    pending_path: str,
) -> Dict[str, Any]:
    params = req.get("params") or {}
    name = str(params.get("name") or "")
    args = params.get("arguments") or {}

    # v0.2: sender_lid is a REQUIRED per-call arg now (not a startup env var).
    # Persona is responsible for passing the verified sender's lid here.
    sender_lid = str(args.get("sender_lid") or "").strip()
    if not _LID_RE.match(sender_lid):
        _log(
            "warn",
            "tool call refused — sender_lid missing or malformed",
            tool=name,
            sender_lid_received=sender_lid or "(empty)",
        )
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": (
                    "sender_lid required: every hermes_save:* call must "
                    "include the verified sender's lid (format "
                    "`<digits>@lid`) as the `sender_lid` argument. "
                    "Source it from the most recent <verified_sender "
                    "id=\"...\"/> marker on the user's message; "
                    "never invent it."
                ),
            }],
        }
    role = _resolve_role(scopes_data, sender_lid)

    session_dir = _session_dir_for(env_path)
    if name == "approve_user":
        return _handle_approve_user(
            args, scopes_data, sender_lid, role, scopes_yaml_path, session_dir,
        )
    if name == "add_to_allowlist":
        return _handle_add_to_allowlist(args, scopes_data, sender_lid, env_path)
    if name == "record_pending_user":
        return _handle_record_pending_user(args, pending_path)
    if name == "list_pending_users":
        return _handle_list_pending_users(
            scopes_data, sender_lid, pending_path, env_path, session_dir,
        )
    if name == "list_allowlist":
        return _handle_list_allowlist(scopes_data, sender_lid, env_path, session_dir)
    if name == "remove_from_allowlist":
        return _handle_remove_from_allowlist(args, scopes_data, sender_lid, env_path)
    if name == "revoke_user":
        return _handle_revoke_user(
            args, scopes_data, sender_lid, scopes_yaml_path, pending_path,
        )
    if name == "grant_scope_access":
        return _handle_grant_scope_access(args, scopes_data, sender_lid, scopes_yaml_path)
    if name == "revoke_scope_access":
        return _handle_revoke_scope_access(args, scopes_data, sender_lid, scopes_yaml_path)
    if name == "reload_gateway":
        return _handle_reload_gateway(scopes_data, sender_lid)
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
    # v0.2: no per-session env. The server is launched ONCE by
    # hermes-agent (via config.yaml mcp_servers) and stays up for the
    # lifetime of the gateway. Every tool call carries its own
    # sender_lid arg.
    scopes_yaml_path = _env("HERMES_SCOPES_YAML") or ""
    bearers_path = _env("HERMES_BEARERS_FILE") or "/home/hermes-user/.hermes/role-bearers.json"
    gbrain_url = _env("HERMES_GBRAIN_URL") or "http://127.0.0.1:7777"
    env_path = _env("HERMES_HERMES_ENV_PATH") or "/home/hermes-user/.hermes/.env"
    pending_path = _env("HERMES_PENDING_USERS_PATH") or _DEFAULT_PENDING_PATH
    timeout_str = _env("HERMES_GBRAIN_TIMEOUT") or "30"
    try:
        timeout = float(timeout_str)
    except ValueError:
        timeout = 30.0

    missing = [
        n for n, v in [
            ("HERMES_SCOPES_YAML", scopes_yaml_path),
            ("HERMES_BEARERS_FILE", bearers_path),
            ("HERMES_GBRAIN_URL", gbrain_url),
        ] if not v
    ]
    if missing:
        _log("error", "required env vars missing — fail closed", missing=missing)
        # Don't exit — hermes will spawn us anyway; emit a clear error on
        # the first tool call so the bot surfaces it instead of hanging.

    scopes_data = _load_yaml(scopes_yaml_path) or {}
    bearers = _load_bearers(bearers_path) if bearers_path else {}
    _log(
        "info",
        "hermes-save MCP server started",
        scopes=_scope_list(scopes_data),
        bearer_count=len(bearers),
        bearer_scopes=sorted(bearers.keys()),
        scopes_yaml=scopes_yaml_path,
        bearers_file=bearers_path,
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
                # Reload scopes.yaml per call so a prior approve_user is
                # reflected in the enum lists. Cheap (small YAML).
                scopes_data = _load_yaml(scopes_yaml_path) or scopes_data
                _write_response(req_id, _handle_tools_list(scopes_data))
            elif method == "tools/call":
                # Reload scopes.yaml per-call so a prior approve_user write
                # is visible to subsequent calls. Bearers file is also
                # reloaded so a fetch-secrets refresh takes effect without
                # a server restart.
                scopes_data = _load_yaml(scopes_yaml_path) or scopes_data
                bearers = _load_bearers(bearers_path) if bearers_path else bearers
                _write_response(
                    req_id,
                    _handle_tools_call(
                        req, scopes_data, bearers,
                        gbrain_url, timeout, scopes_yaml_path, env_path,
                        pending_path,
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
