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
                            "enum": available_roles,
                            "description": (
                                "One of the roles defined in scopes.yaml. "
                                "Determines which scopes the user can read "
                                "and write to. See AGENTS.md for the role-"
                                "to-scope mapping."
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
) -> Dict[str, Any]:
    """Return the pending-users queue. Super_admin only."""
    if not _is_super_admin(scopes_data, sender_lid):
        _log("warn", "list_pending_users denied", sender=sender_lid)
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                "Pending-users queue is restricted to super_admins."
            )}],
        }
    entries = _load_pending(pending_path)
    if not entries:
        return {"content": [{"type": "text", "text": (
            "📭 No pending users. Either nobody new has messaged after "
            "being allowlisted, or all pending users have been approved."
        )}]}
    lines = [f"📋 {len(entries)} pending user(s):\n"]
    for e in entries:
        lid = e.get("lid", "<unknown>")
        ts = e.get("first_seen_at", "<unknown>")
        snip = e.get("snippet", "")
        line = f"- `{lid}` — first seen {ts}"
        if snip:
            line += f' — "{snip[:80]}"'
        lines.append(line)
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
        + f"Next step: ask the new user to send any WhatsApp message to "
        f"the bot. Hermes will surface their verified-sender `lid` in "
        f"the journal. Run `/approve <lid> as <role> name <name>` to "
        f"assign their role. Until then they're at None tier (general "
        f"help only)."
    )}]}


def _handle_approve_user(
    args: Dict[str, Any],
    scopes_data: Dict[str, Any],
    sender_lid: str,
    role: Optional[str],
    scopes_yaml_path: str,
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

    available_roles = _role_list(scopes_data)
    if requested_role not in available_roles:
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                f"role '{requested_role}' is not defined in scopes.yaml. "
                f"Allowed roles: {available_roles}. Promotion to super_admin "
                f"is SSH-only — not via chat."
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
            f"- platform: {platform}\n\n"
            f"The change takes effect immediately on their next message — "
            f"hermes reloads scopes.yaml when the file mtime advances. "
            f"No restart needed. Removed from pending queue if they were there.\n\n"
            f"Note: the VM's scopes.yaml has diverged from the repo. Harris "
            f"should `scp /opt/hermes/workspace/scopes.yaml ./azure/config/` "
            f"before the next deploy to keep the repo in sync."
        )}],
    }


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

    if name == "approve_user":
        return _handle_approve_user(
            args, scopes_data, sender_lid, role, scopes_yaml_path,
        )
    if name == "add_to_allowlist":
        return _handle_add_to_allowlist(args, scopes_data, sender_lid, env_path)
    if name == "record_pending_user":
        return _handle_record_pending_user(args, pending_path)
    if name == "list_pending_users":
        return _handle_list_pending_users(scopes_data, sender_lid, pending_path)
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
