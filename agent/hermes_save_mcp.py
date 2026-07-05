"""Hermes-side MCP server: scoped writes + admin tools, with per-sender routing.

Exposes admin / write tools (`save_to_scope`, `add_to_allowlist`,
`approve_user`, `record_pending_user`, `list_pending_users`). Each tool
takes the verified sender's WhatsApp wa_id as a `sender_id` argument; the
proxy looks the role up from scopes.yaml and picks the right per-role
OAuth bearer from a static bearers file before forwarding the call to
gbrain.

v0.2 (2026-06-18): per-call sender_id arg instead of per-session env var.
Required because the new azure-foundry provider doesn't have the
per-session --mcp-config injection path the previous claude-code-cli
provider used to pass HERMES_SENDER_ID at startup. With the new design
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
  HERMES_SAVE_BEARER_REFRESH_CMD — OPTIONAL. Absolute path to a script that
                         prints a fresh access_token to stdout when invoked
                         as `<cmd> <role>`. When set, the server tries to
                         re-mint and retry once on a gbrain 401, instead of
                         surfacing the error. Mirrors the upstream gateway
                         fix (PR #52418). Static role-bearers.json tokens
                         have a finite TTL (~1h on client_credentials grants);
                         without this hook saves break the moment the file
                         goes stale and require an external `fetch-secrets`
                         re-run + service restart. Example value:
                         `/etc/hermes/refresh-mcp-bearer.sh`. Strictly opt-in
                         — unset → behavior identical to v0.2.

Audit: every tool call writes one structured line to stderr so the parent
shell (hermes.service journal) captures it. The `sender_id` from the
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
_SERVER_VERSION = "0.2.1"

# Bearer auto-refresh on 401. Mirrors the gateway-side fix in PR #52418 but
# adapted for this server's file-backed (not env-var) bearer model: on a 401
# from gbrain put_page, invoke the operator-configured refresh command for
# the writer_role, swap the in-memory bearer, atomically rewrite the
# role-bearers.json entry, and retry the call ONCE. Static client_credentials
# bearers have ~1h TTL; without this hook the next save after the file goes
# stale fails until fetch-secrets re-runs (typically a full hermes restart).
_BEARER_REFRESH_CMD_ENV = "HERMES_SAVE_BEARER_REFRESH_CMD"
_BEARER_REFRESH_TIMEOUT_S = 10.0   # kill the mint command past this
_BEARER_REFRESH_COOLDOWN_S = 60.0  # min seconds between refresh attempts per role
_bearer_refresh_last_attempt: Dict[str, float] = {}

# v0.2: every tool requires the verified sender's lid as a per-call arg
# (no longer a process-startup env var). The persona MUST include this
# in every hermes_save:* call, sourced from the most recent
# <verified_sender id="..."/> marker on the user's message. Without it
# the server fails closed — no role lookup, no bearer routing, no audit.
_SENDER_LID_SCHEMA: Dict[str, Any] = {
    "type": "string",
    "pattern": r"^\d{8,15}$",
    "description": (
        "REQUIRED. The verified sender's WhatsApp wa_id (format "
        "`<wa_id digits>`), copied verbatim from the most recent "
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


def _refresh_role_bearer(role: str, bearers_path: str) -> Optional[str]:
    """Re-mint a fresh OAuth bearer for `role` via the operator-configured
    refresh command. Returns the new bearer on success, or None when the
    feature is disabled, cooldown is active, the command fails, or output
    looks unhealthy. Also atomically rewrites the role's entry in
    `bearers_path` so the next file reload sees the fresh token.

    Discipline mirrors the upstream-gateway fix (PR #52418):
      * 10s timeout kills slow mints.
      * 60s cooldown per role prevents hammering on a stuck 401.
      * Output validated — empty / <16 chars / whitespace-containing /
        error-string-like outputs are rejected so a broken mint script
        can't poison the bearer cache.
      * One refresh attempt per call — caller is responsible for retry-once
        semantics; this helper never loops.
    """
    import subprocess

    cmd_path = _env(_BEARER_REFRESH_CMD_ENV)
    if not cmd_path:
        return None

    now = time.time()
    last = _bearer_refresh_last_attempt.get(role, 0.0)
    if now - last < _BEARER_REFRESH_COOLDOWN_S:
        _log(
            "info", "bearer refresh skipped — cooldown active",
            role=role,
            seconds_remaining=round(_BEARER_REFRESH_COOLDOWN_S - (now - last), 1),
        )
        return None
    _bearer_refresh_last_attempt[role] = now

    try:
        proc = subprocess.run(
            [cmd_path, role],
            capture_output=True, text=True,
            timeout=_BEARER_REFRESH_TIMEOUT_S, check=False,
        )
    except subprocess.TimeoutExpired:
        _log(
            "error", "bearer refresh timed out",
            role=role, timeout_s=_BEARER_REFRESH_TIMEOUT_S,
        )
        return None
    except FileNotFoundError:
        _log("error", "bearer refresh cmd not found", role=role, cmd=cmd_path)
        return None
    except Exception as exc:
        _log("error", "bearer refresh spawn failed", role=role, error=str(exc))
        return None

    if proc.returncode != 0:
        _log(
            "error", "bearer refresh non-zero exit",
            role=role, code=proc.returncode,
            stderr_snippet=(proc.stderr or "")[:200],
        )
        return None

    new_bearer = (proc.stdout or "").strip()
    if not new_bearer or len(new_bearer) < 16 or any(c.isspace() for c in new_bearer):
        _log(
            "error", "bearer refresh output rejected (empty/short/whitespace)",
            role=role, len=len(new_bearer),
        )
        return None
    lower = new_bearer.lower()
    if any(needle in lower for needle in ("error", "failed", "<html", '"error"', "{")):
        _log(
            "error", "bearer refresh output rejected (looks like an error string)",
            role=role,
        )
        return None

    # Persist to file: atomic tmp + rename. Failure here is non-fatal —
    # we still return the fresh bearer for the caller's immediate retry;
    # the in-memory copy carries it through this call even if disk write
    # is briefly unavailable. The next call will re-read the (now-fresh)
    # file on the normal _load_bearers path.
    if bearers_path:
        try:
            current = _load_bearers(bearers_path)
            current[role] = new_bearer
            tmp_path = bearers_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(current, fh)
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, bearers_path)
        except Exception as exc:
            _log(
                "warn",
                "bearer refresh disk-write failed (returning bearer anyway)",
                role=role, error=str(exc),
            )

    _log("info", "bearer refreshed", role=role)
    return new_bearer


def _resolve_role(scopes_data: Dict[str, Any], sender_id: str) -> Optional[str]:
    """Look up sender role: super_admin if in super_admins list, else from users list."""
    for entry in scopes_data.get("super_admins") or []:
        if str(entry.get("id", "")).strip() == sender_id:
            return "super_admin"
    for entry in scopes_data.get("users") or []:
        if str(entry.get("id", "")).strip() == sender_id:
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


# ─── Document upload guardrails (save_document_to_scope) ────────────────
# The document tool is called with a server-side path (typically the
# WhatsApp Cloud webhook's media cache entry), so path traversal +
# symlink escape + oversized-file DoS need to be handled here. Mime is
# allowlisted so the ingest pipeline (Track B) never has to guess.

_MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB — matches WhatsApp's Cloud API
                                       # media size cap; larger files need a
                                       # different pipeline (chunked upload
                                       # via SharePoint), not this MVP.

_ALLOWED_MIMES: frozenset = frozenset({
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "text/plain",
    "text/csv",
    "text/markdown",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/heic",
})

# Extension → mime for inference when the caller omits mime_type. Kept
# tight to the same set the allowlist accepts; anything not in this map
# falls through to `unsupported_mime` at the allowlist check.
_EXT_TO_MIME: Dict[str, str] = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".heic": "image/heic",
}


def _allowed_upload_roots() -> List[str]:
    """Real paths of directories that may hold a save_document_to_scope
    source file. Any `local_path` must resolve (post-realpath) to somewhere
    under one of these. Values are `realpath`d so symlink comparisons are
    honest — otherwise a rogue symlink outside the root but pointing in
    would look valid.
    """
    candidates = [
        os.path.expanduser("~/.hermes/platforms"),
        "/tmp",
    ]
    roots: List[str] = []
    for c in candidates:
        try:
            roots.append(os.path.realpath(c))
        except Exception:
            continue
    return roots


def _validate_upload_path(local_path: str) -> Optional[str]:
    """Return None if `local_path` is safe to open for a scoped ingest,
    otherwise a short error string suitable for the tool result.

    Guarded against:
      * empty / `..` components (classic traversal),
      * ANY symlink between the file and filesystem root (so a hostile
        symlink dropped inside the allowed root can't redirect us to
        /etc/shadow or similar),
      * paths that resolve outside `~/.hermes/platforms/*/media/` or
        `/tmp/`,
      * non-existent / non-regular / non-readable files.

    All negative cases are returned as strings; caller renders them in
    the tool envelope.
    """
    if not local_path:
        return "local_path is empty"
    norm = local_path.replace("\\", "/")
    if any(p == ".." for p in norm.split("/")):
        return "local_path contains '..'"

    abs_path = os.path.abspath(local_path)

    # Walk the path from the file up to root, checking each component
    # for a symlink. This catches BOTH a symlink at the leaf AND a
    # symlinked parent directory that would otherwise let a resolved
    # `real_path` land inside the allowed root while the pre-resolve
    # component pointed elsewhere.
    walk = abs_path
    while True:
        try:
            if os.path.islink(walk):
                return f"symlink in path: {walk}"
        except OSError:
            break
        parent = os.path.dirname(walk)
        if parent == walk:
            break
        walk = parent

    try:
        real = os.path.realpath(abs_path)
    except Exception as exc:
        return f"realpath failed: {exc}"

    roots = _allowed_upload_roots()
    if not any(real == r or real.startswith(r + os.sep) for r in roots):
        return (
            "local_path resolves outside allowed roots (must live under "
            "~/.hermes/platforms/*/media/ or /tmp/)"
        )

    if not os.path.exists(real):
        return "file does not exist"
    if not os.path.isfile(real):
        return "path is not a regular file"
    if not os.access(real, os.R_OK):
        return "file is not readable"
    return None


def _sha256_file(path: str) -> str:
    """SHA-256 hex digest of `path`, streamed in 1 MB chunks so a 20 MB
    PDF doesn't buffer entirely in memory before the hash is emitted."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(1024 * 1024)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _infer_mime_from_extension(local_path: str) -> Optional[str]:
    ext = os.path.splitext(local_path)[1].lower()
    return _EXT_TO_MIME.get(ext)


def _gbrain_find_page_by_content_hash(
    gbrain_url: str,
    bearer: str,
    source_id: str,
    content_hash: str,
    timeout: float,
) -> Optional[str]:
    """Best-effort dedup pre-check: ask gbrain whether a page with this
    content_hash already exists in `source_id`. Returns the existing slug
    on hit, None on miss OR on ANY error (network, unknown-method, malformed
    envelope, RLS refusal — the caller falls through to the ingest path
    which is itself idempotent on content_hash, so a false negative here
    only costs latency, never correctness).
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "find_page_by_content_hash",
            "arguments": {
                "source_id": source_id,
                "content_hash": content_hash,
            },
        },
    }
    try:
        req = urllib.request.Request(
            f"{gbrain_url.rstrip('/')}/mcp",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {bearer}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        envelope = _parse_sse_envelope(raw)
    except Exception:
        return None

    result = (envelope or {}).get("result") or {}
    if not isinstance(result, dict):
        return None
    # Accept either a top-level `slug` or a stringified JSON payload inside
    # content[0].text — gbrain tools historically use both shapes.
    slug = result.get("slug")
    if isinstance(slug, str) and slug:
        return slug
    content = result.get("content") or []
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, str):
                try:
                    parsed = json.loads(text)
                except Exception:
                    return None
                if isinstance(parsed, dict):
                    slug2 = parsed.get("slug")
                    if isinstance(slug2, str) and slug2:
                        return slug2
    return None


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
                        "sender_id": _SENDER_LID_SCHEMA,
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
                    "required": ["sender_id", "scope", "title", "body"],
                },
            },
            {
                "name": "save_document_to_scope",
                "description": (
                    "Persist an inbound document (from WhatsApp media, "
                    "SharePoint, etc.) as a scoped gbrain page. Runs OCR "
                    "(for scans/images) → chunks → embeds → upserts to "
                    "gbrain. Idempotent on content-hash: re-saving the "
                    "same file returns the existing slug. RLS-enforced: "
                    "sender must have write access to the target scope "
                    "per scopes.yaml roles.<role>.writes. Available "
                    "scopes: " + ", ".join(available_scopes) + ". Response "
                    "is JSON in content[0].text with status "
                    "'created_or_updated' or 'already_saved' on success."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sender_id": _SENDER_LID_SCHEMA,
                        "scope": {
                            "type": "string",
                            "enum": available_scopes,
                            "description": (
                                "Target scope. Must be one of the allowed "
                                "enum values; the sender's role must also "
                                "have write permission for it (same check "
                                "as save_to_scope)."
                            ),
                        },
                        "local_path": {
                            "type": "string",
                            "description": (
                                "Server-side path to the file, typically "
                                "`~/.hermes/platforms/whatsapp_cloud/"
                                "media/<media_id>.<ext>` from the inbound "
                                "media cache, or a staged path under "
                                "`/tmp/`. Paths outside those roots, paths "
                                "containing `..`, and paths that traverse "
                                "a symlink are refused."
                            ),
                        },
                        "title": {
                            "type": "string",
                            "description": (
                                "Human-readable title for the saved page. "
                                "Infer from the user's message and the "
                                "file's own metadata; used verbatim as the "
                                "page heading and as one of the inputs to "
                                "the derived slug."
                            ),
                        },
                        "mime_type": {
                            "type": "string",
                            "description": (
                                "MIME type of the file (e.g. "
                                "`application/pdf`, `image/jpeg`). Optional "
                                "— if omitted, the server infers it from "
                                "the file extension. Only PDF, Office "
                                "(doc/docx/xls/xlsx/ppt/pptx), text/CSV/"
                                "Markdown, and JPEG/PNG/TIFF/HEIC are "
                                "accepted; other types return "
                                "`unsupported_mime`."
                            ),
                        },
                    },
                    "required": ["sender_id", "scope", "local_path", "title"],
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
                        "sender_id": _SENDER_LID_SCHEMA,
                        "target_id": {
                            "type": "string",
                            "description": (
                                "The new user's WhatsApp wa_id (the same "
                                "`<wa_id digits>` string from the verified-"
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
                    "required": ["sender_id", "target_id"],
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
                    "properties": {"sender_id": _SENDER_LID_SCHEMA},
                    "required": ["sender_id"],
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
                        "sender_id": _SENDER_LID_SCHEMA,
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
                    "required": ["sender_id", "phone"],
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
                        "sender_id": _SENDER_LID_SCHEMA,
                        "target_id": {
                            "type": "string",
                            "description": (
                                "WhatsApp wa_id of the user to approve "
                                "(bare digits, e.g., `923333717117`). "
                                "Captured verbatim from the verified-sender "
                                "marker on a message the new user has "
                                "already sent — Meta's Cloud API delivers "
                                "the real E.164 number minus the `+` prefix."
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
                    "required": ["sender_id", "target_id", "name", "role"],
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
                    "properties": {"sender_id": _SENDER_LID_SCHEMA},
                    "required": ["sender_id"],
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
                        "sender_id": _SENDER_LID_SCHEMA,
                        "phone": {
                            "type": "string",
                            "description": (
                                "Phone to remove, digits only, no '+' "
                                "(e.g. 923338888888)."
                            ),
                        },
                    },
                    "required": ["sender_id", "phone"],
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
                        "sender_id": _SENDER_LID_SCHEMA,
                        "target_id": {
                            "type": "string",
                            "description": (
                                "The lid to remove, format `<wa_id digits>`."
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
                    "required": ["sender_id", "target_id"],
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
                        "sender_id": _SENDER_LID_SCHEMA,
                        "target_id": {
                            "type": "string",
                            "description": "The user receiving the grant (`<wa_id digits>`).",
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
                    "required": ["sender_id", "target_id", "scope"],
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
                        "sender_id": _SENDER_LID_SCHEMA,
                        "target_id": {
                            "type": "string",
                            "description": "The user losing the grant (`<wa_id digits>`).",
                        },
                        "scope": {
                            "type": "string",
                            "description": "The scope id to revoke (must be in target's extra_reads).",
                        },
                    },
                    "required": ["sender_id", "target_id", "scope"],
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
                    "properties": {"sender_id": _SENDER_LID_SCHEMA},
                    "required": ["sender_id"],
                },
            },
            {
                "name": "send_template_message",
                "description": (
                    "Send a Meta-approved WhatsApp template message to any "
                    "phone number. Templates are how you initiate a "
                    "conversation OUTSIDE the 24-hour customer service "
                    "window — required for cold outreach (e.g. an "
                    "onboarding welcome). Calls Meta's Graph API "
                    "/{phone_number_id}/messages with a template payload. "
                    "Use the default template 'hermes_onboarding_message' "
                    "to send the standard 'Yes, let's start' welcome card "
                    "as part of the onboarding flow "
                    "(add_to_allowlist → reload_gateway → send_template_message "
                    "→ wait for tap → approve_user). Super_admin only. The "
                    "recipient must be addressable via WhatsApp; Meta will "
                    "return an error if the number is invalid or unreachable."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sender_id": _SENDER_LID_SCHEMA,
                        "target_phone": {
                            "type": "string",
                            "description": (
                                "Recipient phone number, digits only, no '+'. "
                                "Example: 923333717117 for +92 333 3717117. "
                                "International format; country code first."
                            ),
                        },
                        "template_name": {
                            "type": "string",
                            "description": (
                                "Meta-approved template name. Defaults to "
                                "'hermes_onboarding_message' (the standard "
                                "DIH onboarding welcome). Must match a "
                                "template that's been APPROVED in your "
                                "Meta Business Manager — pending or rejected "
                                "templates return an error."
                            ),
                        },
                        "language_code": {
                            "type": "string",
                            "description": (
                                "BCP-47 language code matching the approved "
                                "template variant. Defaults to 'en'. Common "
                                "alternatives: 'en_US', 'es', 'ar'."
                            ),
                        },
                        "recipient_name": {
                            "type": "string",
                            "description": (
                                "Optional human-readable name for the "
                                "recipient — used only in the tool's "
                                "confirmation message back to the operator "
                                "(e.g. 'Template sent to Jane Doe at "
                                "923xxx'). NOT passed to Meta."
                            ),
                        },
                    },
                    "required": ["sender_id", "target_phone"],
                },
            },
        ]
    }


_SENDER_ID_RE = re.compile(r"^\d{8,15}$")
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


def _is_super_admin(scopes_data: Dict[str, Any], sender_id: str) -> bool:
    for e in scopes_data.get("super_admins") or []:
        if str(e.get("id", "")).strip() == sender_id:
            return True
    return False


def _append_user_to_scopes_yaml(path: str, target_id: str, name: str, role: str, platform: str) -> None:
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
    line = f'  - {{ id: "{target_id}", name: "{safe_name}", role: {role}, platform: {platform} }}\n'
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    audit = f"  # added via /approve at {ts}\n"

    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    # Ensure the file ends with a newline so our append starts on its own line.
    if not text.endswith("\n"):
        text += "\n"
    new_text = text + audit + line
    _atomic_write(path, new_text)


def _append_super_admin_to_scopes_yaml(path: str, target_id: str, name: str, sender_id: str) -> None:
    """Insert an entry into the `super_admins:` block (right after the header —
    list order is irrelevant). Super_admins carry NO `role:` field. TEXT edit
    preserves comments; the audit comment records who granted it and when."""
    safe_name = name.replace('"', "'")
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    audit = f"  # SUPER_ADMIN_GRANT via /approve by {sender_id} at {ts}"
    entry = f'  - {{ id: "{target_id}", name: "{safe_name}", platform: whatsapp }}'
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


def _remove_super_admin_from_scopes_yaml(path: str, target_id: str) -> bool:
    """Remove a `super_admins:` entry by lid (a `- {` line with the id and NO
    `role:` field — that's what distinguishes super_admins from users). Drops a
    preceding SUPER_ADMIN_GRANT audit comment too. Returns True if removed."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    out: List[str] = []
    removed = False
    for ln in text.splitlines():
        if (f'id: "{target_id}"' in ln and "- {" in ln and "role:" not in ln):
            removed = True
            if out and out[-1].strip().startswith("# SUPER_ADMIN_GRANT"):
                out.pop()
            continue
        out.append(ln)
    if not removed:
        return False
    _atomic_write(path, "\n".join(out) + ("\n" if text.endswith("\n") else ""))
    return True


def _wa_id_is_known(wa_id: str) -> bool:
    """True if the value looks like a verified WhatsApp Cloud wa_id (bare
    digits, 8–15 chars). Post-2026-06-24 cutover, identity verification
    is done by Meta's signed webhook before the value ever reaches us —
    so any value that arrives via a ``<verified_sender>`` marker is
    inherently real. The lingering Baileys-era ``_lid_is_known`` check
    (which validated against `lid-mapping-<digits>.json` files in the
    bridge session dir) is no longer applicable: those files don't
    exist on WA Cloud sessions. Reduced to a shape check that guards
    against operator typos / fabricated values."""
    s = str(wa_id or "").strip()
    return s.isdigit() and 8 <= len(s) <= 15


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


def _remove_user_from_scopes_yaml(path: str, target_id: str) -> bool:
    """Remove the `users:` entry whose id == target_id (TEXT edit; preserves
    comments). Drops a preceding `# added via /approve` audit line too. Only
    matches lines that carry a `role:` field, so super_admins entries (which
    have none) are never removed from chat. Returns True if removed."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    out: List[str] = []
    removed = False
    for ln in text.splitlines():
        if (f'id: "{target_id}"' in ln and "role:" in ln and "- {" in ln):
            removed = True
            if out and out[-1].strip().startswith("# added via /approve"):
                out.pop()
            continue
        out.append(ln)
    if not removed:
        return False
    _atomic_write(path, "\n".join(out) + ("\n" if text.endswith("\n") else ""))
    return True


def _compute_pending(scopes_data: Dict[str, Any], env_path: str) -> List[Dict[str, Any]]:
    """Allowlisted phones whose wa_id is NOT yet in scopes.yaml.

    Independent of record_pending_user — surfaces a user even if the agent
    never logged them. On WhatsApp Cloud the phone IS the wa_id (no @lid
    suffix), so this is a direct set-comparison against scopes.yaml `id`
    fields. Returns [{phone, wa_id}].
    """
    phones, _memos = _read_allowlist(env_path)
    known = _existing_lids(scopes_data)  # named for history; returns wa_ids post-cutover
    out: List[Dict[str, Any]] = []
    for ph in phones:
        if ph == "*":
            continue
        if ph in known:
            continue  # already onboarded as a wa_id-form identity
        out.append({"phone": ph, "wa_id": ph})
    return out


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
def _scopes_yaml_grant_extra_read(path: str, target_id: str, scope_id: str) -> str:
    """Add `scope_id` to target_id's extra_reads in scopes.yaml.

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
        if str(entry.get("id", "")).strip() == target_id:
            return "super_admin"
    # user entries
    users = data.get("users") or []
    target_entry = None
    for entry in users:
        if str(entry.get("id", "")).strip() == target_id:
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


def _scopes_yaml_revoke_extra_read(path: str, target_id: str, scope_id: str) -> str:
    """Remove `scope_id` from target_id's extra_reads.

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
        if str(entry.get("id", "")).strip() == target_id:
            return "super_admin"
    users = data.get("users") or []
    target_entry = None
    for entry in users:
        if str(entry.get("id", "")).strip() == target_id:
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


def _remove_subject_from_gbrain(target_id: str) -> Optional[str]:
    """Soft-delete the subject row for `target_id` in gbrain.
    Returns None on success or an error string. NEVER raises."""
    import subprocess
    try:
        proc = subprocess.run(
            ["gbrain", "auth", "subjects", "remove", target_id],
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
    target_id = str(args.get("target_id") or "").strip()
    snippet = str(args.get("first_message_snippet") or "").strip()[:200]

    if not target_id:
        return {"isError": True, "content": [{"type": "text", "text": "target_id is required."}]}
    if not _SENDER_ID_RE.match(target_id):
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                f"target_sender_id format invalid: '{target_id}'. Expected "
                f"`<wa_id digits>`."
            )}],
        }

    entries = _load_pending(pending_path)
    # Idempotent — don't duplicate.
    if any(e.get("lid") == target_id for e in entries):
        return {"content": [{"type": "text", "text": (
            f"📋 Pending user {target_id} already queued (no duplicate)."
        )}]}
    entry = {
        "lid": target_id,
        "first_seen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "snippet": snippet,
    }
    entries.append(entry)
    try:
        _write_pending(pending_path, entries)
    except Exception as exc:
        _log("error", "record_pending_user write failed", lid=target_id, error=str(exc))
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                f"Failed to record pending user: {exc}. Bot will still "
                f"reply welcome to the user, but admin won't see them in "
                f"the queue. Check file permissions."
            )}],
        }
    _log("info", "pending user recorded", lid=target_id, snippet=snippet)
    return {"content": [{"type": "text", "text": (
        f"Recorded pending user {target_id}. {len(entries)} total in queue."
    )}]}


def _handle_list_pending_users(
    scopes_data: Dict[str, Any],
    sender_id: str,
    pending_path: str,
    env_path: str,
) -> Dict[str, Any]:
    """Pending = the recorded queue UNION allowlisted numbers not yet in
    scopes.yaml (computed via wa_id set-compare against `id` fields).
    The computed half is the robust fix: it surfaces a user even if the
    agent never called record_pending_user. Super_admin only."""
    if not _is_super_admin(scopes_data, sender_id):
        _log("warn", "list_pending_users denied", sender=sender_id)
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                "Pending-users queue is restricted to super_admins."
            )}],
        }
    recorded = _load_pending(pending_path)
    computed = _compute_pending(scopes_data, env_path)

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


def _remove_from_pending(pending_path: str, target_id: str) -> bool:
    """Drop the matching lid from the queue. Returns True if anything
    was removed. Used by approve_user to clean up on success."""
    entries = _load_pending(pending_path)
    before = len(entries)
    entries = [e for e in entries if e.get("lid") != target_id]
    if len(entries) == before:
        return False
    try:
        _write_pending(pending_path, entries)
    except Exception as exc:
        _log("warn", "remove_from_pending write failed", lid=target_id, error=str(exc))
        return False
    return True


def _handle_add_to_allowlist(
    args: Dict[str, Any],
    scopes_data: Dict[str, Any],
    sender_id: str,
    env_path: str,
) -> Dict[str, Any]:
    """Add a phone to WHATSAPP_ALLOWED_USERS. Super_admin only."""
    phone = str(args.get("phone") or "").strip().lstrip("+").replace(" ", "")
    memo = str(args.get("memo") or "").strip() or None

    if not _is_super_admin(scopes_data, sender_id):
        _log("warn", "add_to_allowlist denied", sender=sender_id, phone=phone)
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
        _log("error", "add_to_allowlist write failed", sender=sender_id, phone=phone, error=str(exc))
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

    _log("info", "add_to_allowlist success", sender=sender_id, phone=phone, memo=memo)
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
    sender_id: str,
    role: Optional[str],
    scopes_yaml_path: str,
) -> Dict[str, Any]:
    target_id = str(args.get("target_id") or "").strip()
    name = str(args.get("name") or "").strip()
    requested_role = str(args.get("role") or "").strip()
    platform = str(args.get("platform") or "whatsapp_cloud").strip().lower() or "whatsapp_cloud"

    # ── Authorization: only super_admins may onboard ───────────────────────
    if not _is_super_admin(scopes_data, sender_id):
        _log("warn", "approve_user denied (not super_admin)", sender=sender_id, target=target_id)
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
    if not target_id or not name or not requested_role:
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                "approve_user requires target_id, name, and role."
            )}],
        }

    if not _SENDER_ID_RE.match(target_id):
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                f"target_id format invalid: '{target_id}'. Expected "
                f"`<wa_id digits>` (8-15 bare digits, no `+`, no `@<suffix>`). "
                f"Capture the wa_id from the verified-sender marker on a "
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
        if not _wa_id_is_known(target_id):
            return {"isError": True, "content": [{"type": "text", "text": (
                f"Won't grant super_admin to {target_id}: that value doesn't "
                f"look like a wa_id (bare digits, 8-15 chars). Capture the "
                f"target's wa_id from a verified-sender marker on a message "
                f"they sent, then retry."
            )}]}
        if _is_super_admin(scopes_data, target_id):
            return {"content": [{"type": "text", "text": (
                f"{target_id} is already a super_admin — no change."
            )}]}
        try:
            _append_super_admin_to_scopes_yaml(scopes_yaml_path, target_id, name, sender_id)
        except Exception as exc:
            _log("error", "super_admin grant write failed", sender=sender_id, target=target_id, error=str(exc))
            return {"isError": True, "content": [{"type": "text", "text": (
                f"Failed to write scopes.yaml: {exc}. NOT granted."
            )}]}
        pending_path = os.environ.get("HERMES_PENDING_USERS_PATH") or _DEFAULT_PENDING_PATH
        _remove_from_pending(pending_path, target_id)
        _log("warn", "SUPER_ADMIN_GRANT", sender=sender_id, target_id=target_id, name=name)
        _sync_err = _sync_subjects_to_gbrain(scopes_yaml_path)
        _sync_note = f"\n\n⚠️ Subjects sync warning: {_sync_err}" if _sync_err else ""
        return {"content": [{"type": "text", "text": (
            f"✅ SUPER_ADMIN granted.{_sync_note}\n\n"
            f"- lid: `{target_id}`\n- name: {name}\n- role: super_admin\n\n"
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

    # Accept both Baileys-era and Cloud-API platform names. The underlying
    # identity is the wa_id (Meta-side, platform-agnostic), so a single
    # scopes.yaml entry serves both transports. We normalise the stored
    # value to "whatsapp" right after the check so scopes.yaml stays
    # uniform — every existing entry uses `platform: whatsapp` and mixing
    # in `platform: whatsapp_cloud` would fragment the field for no gain.
    if platform not in {"whatsapp", "whatsapp_cloud"}:
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                f"platform '{platform}' not supported yet (whatsapp / "
                f"whatsapp_cloud only). Future Teams/SMS support will "
                f"land in Phase 8."
            )}],
        }
    if platform == "whatsapp_cloud":
        platform = "whatsapp"

    # ── Idempotency: don't double-add ──────────────────────────────────────
    existing = _existing_lids(scopes_data)
    if target_id in existing:
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                f"User {target_id} is already in scopes.yaml. To change "
                f"their role, SSH in and edit scopes.yaml — chat-driven "
                f"role changes are intentionally not supported in v0.1 "
                f"(avoid privilege-escalation paths through chat-only ops)."
            )}],
        }

    # ── Write ──────────────────────────────────────────────────────────────
    try:
        _append_user_to_scopes_yaml(scopes_yaml_path, target_id, name, requested_role, platform)
    except Exception as exc:
        _log("error", "approve_user write failed", sender=sender_id, target=target_id, error=str(exc))
        return {
            "isError": True,
            "content": [{"type": "text", "text": (
                f"Failed to write to scopes.yaml: {exc}. The user was "
                f"NOT onboarded. Check file permissions on the VM."
            )}],
        }

    # Drop from the pending-users queue (no-op if they weren't queued).
    pending_path = os.environ.get("HERMES_PENDING_USERS_PATH") or _DEFAULT_PENDING_PATH
    _remove_from_pending(pending_path, target_id)

    _sync_err = _sync_subjects_to_gbrain(scopes_yaml_path)
    _sync_note = f"\n\n⚠️ Subjects sync warning: {_sync_err}" if _sync_err else ""
    _log(
        "info",
        "approve_user success",
        sender=sender_id,
        target_id=target_id,
        name=name,
        role=requested_role,
        platform=platform,
    )
    return {
        "content": [{"type": "text", "text": (
            f"✅ User onboarded.\n\n"
            f"- lid: `{target_id}`\n"
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
    sender_id: str,
    env_path: str,
) -> Dict[str, Any]:
    """Show the gateway allowlist with memo + onboarding status. Super_admin only.

    Post-WhatsApp-Cloud-cutover the phone IS the wa_id (no @lid suffix), so
    onboarding status is a direct set-compare between the allowlist phones
    and the wa_ids registered in scopes.yaml.
    """
    if not _is_super_admin(scopes_data, sender_id):
        return {"isError": True, "content": [{"type": "text", "text": (
            "The allowlist is restricted to super_admins."
        )}]}
    phones, memos = _read_allowlist(env_path)
    known = _existing_lids(scopes_data)  # wa_ids post-cutover
    if not phones:
        return {"content": [{"type": "text", "text": "🔒 Allowlist is empty."}]}
    lines = ["🔒 %d number(s) on the allowlist:\n" % len(phones)]
    for ph in phones:
        if ph == "*":
            lines.append("- `*` — OPEN BOT (everyone allowed)")
            continue
        if ph in known:
            status = "✅ onboarded"
        else:
            status = "⏳ allowlisted, no role yet"
        line = "- `%s`" % ph
        if memos.get(ph):
            line += " — %s" % memos[ph]
        line += "  [%s]" % status
        lines.append(line)
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


def _handle_remove_from_allowlist(
    args: Dict[str, Any],
    scopes_data: Dict[str, Any],
    sender_id: str,
    env_path: str,
) -> Dict[str, Any]:
    """Remove a phone from the allowlist. Super_admin only. Needs reload_gateway."""
    phone = str(args.get("phone") or "").strip().lstrip("+").replace(" ", "")
    if not _is_super_admin(scopes_data, sender_id):
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
        _log("error", "remove_from_allowlist failed", sender=sender_id, phone=phone, error=str(exc))
        return {"isError": True, "content": [{"type": "text", "text": (
            "Failed to edit .env: %s. Check file permissions." % exc
        )}]}
    if not removed:
        return {"content": [{"type": "text", "text": (
            "📋 %s was not on the allowlist — nothing to remove." % phone
        )}]}
    _log("info", "remove_from_allowlist success", sender=sender_id, phone=phone)
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
    sender_id: str,
    scopes_yaml_path: str,
    pending_path: str,
) -> Dict[str, Any]:
    """Remove a non-super_admin user from scopes.yaml. Super_admin only."""
    target_id = str(args.get("target_id") or "").strip()
    if not _is_super_admin(scopes_data, sender_id):
        return {"isError": True, "content": [{"type": "text", "text": (
            "Off-boarding is restricted to super_admins."
        )}]}
    if not _SENDER_ID_RE.match(target_id):
        return {"isError": True, "content": [{"type": "text", "text": (
            "target_sender_id format invalid: '%s'. Expected `<wa_id digits>`." % target_id
        )}]}
    target_is_sa = _is_super_admin(scopes_data, target_id)
    if target_is_sa:
        # Guarded super_admin removal (decision 2026-06-18).
        if not bool(args.get("confirm_super_admin")):
            return {"isError": True, "content": [{"type": "text", "text": (
                "%s is a super_admin. Removing a super_admin is high-privilege "
                "— re-issue with confirm_super_admin=true to proceed." % target_id
            )}]}
        if _count_super_admins(scopes_data) <= 1:
            return {"isError": True, "content": [{"type": "text", "text": (
                "Refusing: %s is the LAST super_admin — removing it would lock "
                "everyone out of admin ops. Add another super_admin first." % target_id
            )}]}
    try:
        if target_is_sa:
            removed = _remove_super_admin_from_scopes_yaml(scopes_yaml_path, target_id)
        else:
            removed = _remove_user_from_scopes_yaml(scopes_yaml_path, target_id)
    except Exception as exc:
        _log("error", "revoke_user failed", sender=sender_id, target=target_id, error=str(exc))
        return {"isError": True, "content": [{"type": "text", "text": (
            "Failed to edit scopes.yaml: %s. User NOT removed." % exc
        )}]}
    if not removed:
        return {"content": [{"type": "text", "text": (
            "📋 %s wasn't found in scopes.yaml — nothing to revoke." % target_id
        )}]}
    _remove_from_pending(pending_path, target_id)
    # Soft-delete the gbrain subject row so future token-exchange for this
    # lid returns invalid_grant (subject no longer resolvable). Existing
    # tokens already minted become invalid at next verify (gbrain joins
    # subjects with deleted_at IS NULL — see verifyAccessToken).
    _subject_err = _remove_subject_from_gbrain(target_id)
    _subject_note = f"\n\n⚠️ Subject soft-delete warning: {_subject_err}" if _subject_err else ""
    _log(
        "warn" if target_is_sa else "info",
        "SUPER_ADMIN_REVOKE" if target_is_sa else "revoke_user success",
        sender=sender_id, target=target_id,
    )
    kind = "SUPER_ADMIN" if target_is_sa else "user"
    return {"content": [{"type": "text", "text": (
        "✅ Revoked %s `%s` — removed from scopes.yaml.%s Takes effect on their "
        "next message (mtime reload, no restart).%s\n\n"
        "Note: the VM's scopes.yaml has diverged from the repo — Harris should "
        "`scp /opt/hermes/workspace/scopes.yaml ./azure/config/` before the "
        "next deploy. (Their allowlist entry, if any, is separate — use "
        "`remove_from_allowlist` to drop that too.)" % (
            kind, target_id, _subject_note,
            " Logged as SUPER_ADMIN_REVOKE." if target_is_sa else "",
        )
    )}]}



def _handle_grant_scope_access(
    args: Dict[str, Any],
    scopes_data: Dict[str, Any],
    sender_id: str,
    scopes_yaml_path: str,
) -> Dict[str, Any]:
    """Grant a user read access to one additional scope. Super_admin only.

    Writes to scopes.yaml as a per-user `extra_reads` entry, then
    re-syncs the gbrain subjects table so the new scope shows up in
    the target's allowed_sources at the SQL-RLS layer on the next
    token-exchange call. Role baseline reads are unchanged — this only
    edits the per-user override list.
    """
    target_id = str(args.get("target_id") or "").strip()
    scope_id = str(args.get("scope") or "").strip()
    if not _is_super_admin(scopes_data, sender_id):
        return {"isError": True, "content": [{"type": "text", "text": (
            "Scope grants are restricted to super_admins. Your sender "
            "identity does not have super_admin privileges."
        )}]}
    if not _SENDER_ID_RE.match(target_id):
        return {"isError": True, "content": [{"type": "text", "text": (
            f"target_sender_id format invalid: '{target_id}'. Expected `<wa_id digits>`."
        )}]}
    if not scope_id:
        return {"isError": True, "content": [{"type": "text", "text": (
            "scope is required (e.g. 'leadership', 'finance', 'project_mesec')."
        )}]}
    try:
        result = _scopes_yaml_grant_extra_read(scopes_yaml_path, target_id, scope_id)
    except Exception as exc:
        _log("error", "grant_scope_access write failed",
             sender=sender_id, target=target_id, scope=scope_id, error=str(exc))
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
            f"User {target_id} is not in scopes.yaml. Onboard them first "
            f"via approve_user, then re-issue the grant."
        )}]}
    if result == "super_admin":
        return {"content": [{"type": "text", "text": (
            f"{target_id} is a super_admin and already has access to every scope. "
            f"No grant needed."
        )}]}
    if result == "role_baseline":
        role = ""
        for e in (scopes_data.get("users") or []):
            if str(e.get("id", "")).strip() == target_id:
                role = str(e.get("role", "")).strip()
                break
        return {"content": [{"type": "text", "text": (
            f"{target_id}'s role ('{role}') already includes '{scope_id}' in its "
            f"baseline reads. No grant needed."
        )}]}
    if result == "already":
        return {"content": [{"type": "text", "text": (
            f"{target_id} already has '{scope_id}' in their extra_reads. No change."
        )}]}
    # result == "added"
    sync_err = _sync_subjects_to_gbrain(scopes_yaml_path)
    sync_note = (f"\n\n⚠️ Subjects sync warning: {sync_err}" if sync_err else "")
    _log("info", "grant_scope_access",
         sender=sender_id, target=target_id, scope=scope_id)
    return {"content": [{"type": "text", "text": (
        f"✅ Granted `{scope_id}` read access to `{target_id}`.{sync_note}\n\n"
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
    sender_id: str,
    scopes_yaml_path: str,
) -> Dict[str, Any]:
    """Revoke one extra scope from a user. Super_admin only.

    Only removes from `extra_reads`. If the scope is part of the role's
    baseline reads, the request is refused with instructions to change
    the user's role instead — chat-driven role-config edits are out of
    scope for v1 (they'd ripple to every user with that role).
    """
    target_id = str(args.get("target_id") or "").strip()
    scope_id = str(args.get("scope") or "").strip()
    if not _is_super_admin(scopes_data, sender_id):
        return {"isError": True, "content": [{"type": "text", "text": (
            "Scope revokes are restricted to super_admins."
        )}]}
    if not _SENDER_ID_RE.match(target_id):
        return {"isError": True, "content": [{"type": "text", "text": (
            f"target_sender_id format invalid: '{target_id}'."
        )}]}
    if not scope_id:
        return {"isError": True, "content": [{"type": "text", "text": (
            "scope is required."
        )}]}
    try:
        result = _scopes_yaml_revoke_extra_read(scopes_yaml_path, target_id, scope_id)
    except Exception as exc:
        _log("error", "revoke_scope_access write failed",
             sender=sender_id, target=target_id, scope=scope_id, error=str(exc))
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
            f"User {target_id} is not in scopes.yaml."
        )}]}
    if result == "super_admin":
        return {"isError": True, "content": [{"type": "text", "text": (
            f"{target_id} is a super_admin — their scope access comes from "
            f"super_admin status, not extra_reads. To remove their access, use "
            f"revoke_user (will demote and trigger the soft-delete on their "
            f"subjects row)."
        )}]}
    if result == "role_baseline":
        role = ""
        for e in (scopes_data.get("users") or []):
            if str(e.get("id", "")).strip() == target_id:
                role = str(e.get("role", "")).strip()
                break
        return {"isError": True, "content": [{"type": "text", "text": (
            f"`{scope_id}` is part of `{target_id}`'s role ('{role}') baseline "
            f"reads — chat-driven revoke can't touch role definitions (would "
            f"ripple to every user with that role). To remove this access, "
            f"revoke_user and re-approve with a narrower role, or edit "
            f"scopes.yaml's roles: section by hand."
        )}]}
    if result == "not_in_extra":
        return {"content": [{"type": "text", "text": (
            f"`{target_id}` doesn't have `{scope_id}` as an extra_read — "
            f"nothing to revoke."
        )}]}
    # result == "removed"
    sync_err = _sync_subjects_to_gbrain(scopes_yaml_path)
    sync_note = (f"\n\n⚠️ Subjects sync warning: {sync_err}" if sync_err else "")
    _log("info", "revoke_scope_access",
         sender=sender_id, target=target_id, scope=scope_id)
    return {"content": [{"type": "text", "text": (
        f"✅ Revoked `{scope_id}` extra read from `{target_id}`.{sync_note}\n\n"
        f"Effective on their next message. Their role baseline reads are "
        f"unchanged.\n\n"
        f"Note: the VM's scopes.yaml has diverged from the repo — Harris should "
        f"`scp /opt/hermes/workspace/scopes.yaml ./azure/config/` before the "
        f"next deploy."
    )}]}


def _handle_reload_gateway(
    scopes_data: Dict[str, Any],
    sender_id: str,
) -> Dict[str, Any]:
    """Restart hermes so allowlist changes load. Super_admin only."""
    if not _is_super_admin(scopes_data, sender_id):
        return {"isError": True, "content": [{"type": "text", "text": (
            "Restarting the gateway is restricted to super_admins."
        )}]}
    ok = _restart_gateway_detached()
    if not ok:
        return {"isError": True, "content": [{"type": "text", "text": (
            "Couldn't trigger the restart. Restart manually via SSH: "
            "`sudo systemctl restart hermes`."
        )}]}
    _log("info", "reload_gateway triggered", sender=sender_id)
    return {"content": [{"type": "text", "text": (
        "♻️ Restarting the gateway now — it reloads the allowlist on the way "
        "back up (~15-20s). This session ends; send a new message once it's "
        "reconnected and the latest allowlist will be in effect."
    )}]}


def _handle_send_template_message(
    args: Dict[str, Any],
    scopes_data: Dict[str, Any],
    sender_id: str,
) -> Dict[str, Any]:
    """Send a Meta-approved WhatsApp template message via the Graph API.
    Super_admin only. Reads access_token + phone_number_id from the env
    (set by fetch-secrets.sh from KV)."""
    if not _is_super_admin(scopes_data, sender_id):
        _log("warn", "send_template_message denied", sender=sender_id)
        return {"isError": True, "content": [{"type": "text", "text": (
            "Sending template messages is restricted to super_admins."
        )}]}

    target_phone = str(args.get("target_phone") or "").strip().lstrip("+").replace(" ", "")
    template_name = str(args.get("template_name") or "hermes_onboarding_message").strip()
    language_code = str(args.get("language_code") or "en").strip()
    recipient_name = str(args.get("recipient_name") or "").strip()

    if not _PHONE_RE.match(target_phone):
        return {"isError": True, "content": [{"type": "text", "text": (
            f"target_phone format invalid: '{target_phone}'. Expected digits "
            f"only, 9-15 chars, no '+' (e.g., 923333717117 for "
            f"+92 333 3717117)."
        )}]}

    access_token = os.environ.get("WHATSAPP_CLOUD_ACCESS_TOKEN", "").strip()
    phone_number_id = os.environ.get("WHATSAPP_CLOUD_PHONE_NUMBER_ID", "").strip()
    api_version = os.environ.get("WHATSAPP_CLOUD_API_VERSION", "v20.0").strip()
    # Defense against the MCP-config ${VAR} passthrough leaking the literal
    # placeholder string when the parent process doesn't have the var set —
    # bit us at 2026-06-24 ~17:00 (Meta returned "Unknown path components"
    # because the URL was https://graph.facebook.com/${WHATSAPP_CLOUD_API_VERSION}/.../messages).
    # Any value containing `$` or `{` is not a real Graph version (e.g.
    # `v20.0`, `v21.0`); fall back to the safe default.
    if not api_version or "$" in api_version or "{" in api_version:
        api_version = "v20.0"

    if not access_token or not phone_number_id:
        return {"isError": True, "content": [{"type": "text", "text": (
            "Cloud API credentials missing from hermes env "
            "(WHATSAPP_CLOUD_ACCESS_TOKEN / WHATSAPP_CLOUD_PHONE_NUMBER_ID). "
            "Restart hermes via SSH so fetch-secrets.sh re-exports them from KV."
        )}]}

    url = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": target_phone,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": language_code},
        },
    }

    try:
        import urllib.request
        import urllib.error
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        _log(
            "error", "send_template_message graph api HTTPError",
            sender=sender_id, target=target_phone, template=template_name,
            status=e.code, body=err_body[:500],
        )
        return {"isError": True, "content": [{"type": "text", "text": (
            f"Graph API returned HTTP {e.code} for template '{template_name}' "
            f"to {target_phone}. Common causes: template not approved in "
            f"Meta Business Manager, invalid template name, language code "
            f"mismatch (need an approved variant), number not on WhatsApp, "
            f"or recipient outside our 24h window AND template marked "
            f"non-marketing. Meta response: {err_body[:500] or '(empty)'}"
        )}]}
    except Exception as e:
        _log(
            "error", "send_template_message network failure",
            sender=sender_id, target=target_phone, template=template_name,
            error=str(e),
        )
        return {"isError": True, "content": [{"type": "text", "text": (
            f"Couldn't reach the Graph API: {e}. Check the VM's outbound "
            f"network + the access token freshness."
        )}]}

    wamid = None
    try:
        wamid = data.get("messages", [{}])[0].get("id")
    except Exception:
        pass

    _log(
        "info", "send_template_message accepted",
        sender=sender_id, target=target_phone, template=template_name,
        wamid=wamid,
    )

    display_who = f"{recipient_name} at {target_phone}" if recipient_name else target_phone
    return {"content": [{"type": "text", "text": (
        f"📨 Template '{template_name}' ({language_code}) sent to "
        f"{display_who}. Meta returned message_status: accepted "
        f"(wamid: `{wamid or '?'}`).\n\n"
        f"Next: the recipient will receive the template card. Once they "
        f"tap a button or reply, the 24-hour customer service window "
        f"opens and you can run `approve_user` to assign their final role."
    )}]}


def _handle_save_document_to_scope(
    args: Dict[str, Any],
    scopes_data: Dict[str, Any],
    bearers: Dict[str, str],
    sender_id: str,
    role: Optional[str],
    gbrain_url: str,
    timeout: float,
) -> Dict[str, Any]:
    """Ingest an inbound document into a scoped gbrain page.

    Auth flow mirrors `save_to_scope`: sender_id is already validated
    by the outer dispatcher, `role` is resolved from scopes.yaml here,
    and the target scope is gated by `_can_write` (super_admin bypass
    via `_can_write` too). Additional document-only safety:
      * `_validate_upload_path` blocks traversal + symlink escape,
      * MIME allowlist blocks arbitrary/executable types,
      * 20 MB cap blocks resource-exhaustion via oversized attachments,
      * SHA-256 pre-check lets us short-circuit an already-saved file
        without invoking the (heavier) ingest pipeline.

    Response is JSON in content[0].text so both the agent and the
    audit log see a stable machine-parseable shape. Never raises — every
    exit returns an MCP envelope.
    """
    scope = str(args.get("scope") or "").strip()
    local_path = str(args.get("local_path") or "").strip()
    title = str(args.get("title") or "").strip()
    mime_type = args.get("mime_type")
    if mime_type is not None:
        mime_type = str(mime_type).strip() or None

    def _reply(is_error: bool, payload: Dict[str, Any]) -> Dict[str, Any]:
        env: Dict[str, Any] = {
            "content": [{
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=False),
            }],
        }
        if is_error:
            env["isError"] = True
        return env

    # ── Basic input validation ─────────────────────────────────────────
    if not scope:
        return _reply(True, {"status": "error", "reason": "missing_scope"})
    if not local_path:
        return _reply(True, {"status": "error", "reason": "missing_local_path"})
    if not title:
        return _reply(True, {"status": "error", "reason": "missing_title"})

    available = set(_scope_list(scopes_data))
    if scope not in available:
        return _reply(True, {
            "status": "error",
            "reason": "unknown_scope",
            "scope": scope,
            "available": sorted(available),
        })

    # ── Role + write authorization (mirrors save_to_scope) ─────────────
    if not role:
        _log(
            "warn", "save_document_to_scope: no role resolved for sender",
            sender=sender_id, scope=scope,
        )
        return _reply(True, {
            "status": "error", "reason": "sender_not_roled",
        })
    if not _can_write(scopes_data, role, scope):
        _log(
            "warn", "save_document_to_scope: role not authorized for scope",
            sender=sender_id, role=role, scope=scope,
        )
        role_def = (scopes_data.get("roles") or {}).get(role) or {}
        return _reply(True, {
            "status": "error",
            "reason": "unauthorized_scope",
            "role": role,
            "scope": scope,
            "allowed_writes": sorted(role_def.get("writes") or []),
        })

    # ── Path safety guard ──────────────────────────────────────────────
    path_err = _validate_upload_path(local_path)
    if path_err:
        _log(
            "warn", "save_document_to_scope: path rejected",
            sender=sender_id, local_path=local_path, reason=path_err,
        )
        return _reply(True, {
            "status": "error",
            "reason": "invalid_path",
            "detail": path_err,
        })
    real_path = os.path.realpath(local_path)

    # ── Size cap (fail-closed before we hash or ingest) ────────────────
    try:
        size_bytes = os.path.getsize(real_path)
    except OSError as exc:
        _log(
            "warn", "save_document_to_scope: stat failed",
            sender=sender_id, local_path=real_path, error=str(exc),
        )
        return _reply(True, {
            "status": "error", "reason": "stat_failed", "detail": str(exc),
        })
    if size_bytes > _MAX_UPLOAD_BYTES:
        _log(
            "warn", "save_document_to_scope: file too large",
            sender=sender_id, size_bytes=size_bytes, cap=_MAX_UPLOAD_BYTES,
        )
        return _reply(True, {
            "status": "error",
            "reason": "file_too_large",
            "size_bytes": size_bytes,
            "cap_bytes": _MAX_UPLOAD_BYTES,
        })

    # ── Mime allowlist ─────────────────────────────────────────────────
    # MIME type comparisons are case-insensitive per RFC 2045 §5.1
    # ("Matching of media type and subtype is ALWAYS case-insensitive").
    # Normalise the caller-supplied value to lowercase before checking
    # against _ALLOWED_MIMES so `TEXT/PLAIN` and `text/plain` both work.
    # Caught 2026-07-05 v2 verification battery.
    if not mime_type:
        mime_type = _infer_mime_from_extension(real_path)
    if mime_type:
        mime_type = mime_type.strip().lower()
    if not mime_type or mime_type not in _ALLOWED_MIMES:
        _log(
            "warn", "save_document_to_scope: unsupported mime",
            sender=sender_id, mime_type=mime_type or "(unknown)",
        )
        return _reply(True, {
            "status": "error",
            "reason": "unsupported_mime",
            "mime_type": mime_type,
            "allowed": sorted(_ALLOWED_MIMES),
        })

    # ── SHA-256 the file ───────────────────────────────────────────────
    try:
        content_hash = _sha256_file(real_path)
    except OSError as exc:
        _log(
            "error", "save_document_to_scope: hash read failed",
            sender=sender_id, local_path=real_path, error=str(exc),
        )
        return _reply(True, {
            "status": "error", "reason": "read_failed", "detail": str(exc),
        })

    # ── Writer bearer for target scope (drives dedup pre-check) ────────
    scope_def = (scopes_data.get("scopes") or {}).get(scope) or {}
    writer_role = str(scope_def.get("writer_role") or "").strip()
    if not writer_role:
        return _reply(True, {
            "status": "error",
            "reason": "scope_missing_writer_role",
            "scope": scope,
        })
    writer_bearer = bearers.get(writer_role) or ""
    # Missing bearer isn't fatal here — the ingest helper (Track B)
    # authenticates on its own; we just skip the fast-path dedup and
    # let the helper's own content_hash idempotency handle it.

    # ── Content-hash pre-check (fast-path dedup) ───────────────────────
    if writer_bearer:
        existing_slug = _gbrain_find_page_by_content_hash(
            gbrain_url, writer_bearer, scope, content_hash, timeout,
        )
        if existing_slug:
            _log(
                "info", "save_document_to_scope: dedup hit (pre-check)",
                sender=sender_id, scope=scope,
                slug=existing_slug, content_hash=content_hash,
            )
            return _reply(False, {
                "status": "already_saved",
                "slug": existing_slug,
                "content_hash": content_hash,
                "scope": scope,
            })

    # ── Slug derivation ────────────────────────────────────────────────
    ts = time.strftime("%Y-%m-%d", time.gmtime())
    derived_slug = (
        f"upload-{ts}-{_slugify(scope)}-{_slugify(title)}-{content_hash[:8]}"
    )

    # ── Dispatch to the Track B ingestion helper ───────────────────────
    # Kept as a late import so an operator can restart hermes with the new
    # tool schema BEFORE Track B has landed on the box — the tool then
    # fails cleanly with `ingest_helper_unavailable` instead of preventing
    # server startup.
    try:
        from gbrain_ingest_document import (  # type: ignore
            gbrain_ingest_document,
            IngestResult,  # noqa: F401 — re-exported for Track B contract clarity
        )
    except Exception as exc:
        _log(
            "error", "save_document_to_scope: ingest helper import failed",
            error=str(exc),
        )
        return _reply(True, {
            "status": "error",
            "reason": "ingest_helper_unavailable",
            "detail": str(exc),
        })

    # scopes.yaml `scopes.<scope>.gbrain_source` is the CANONICAL gbrain
    # source id (constrained to [a-z0-9-]{1,32} — hyphens only, no
    # underscores). Track A's scope names (from scopes.yaml keys) are
    # Python-identifier style with underscores (e.g. project_mesec, super_admin).
    # Look up the mapping; fall back to a hyphen-translation of the scope
    # name if the scope entry lacks a gbrain_source field. Existing
    # save_to_scope hits gbrain via the MCP HTTP API which doesn't enforce
    # the CLI regex; this handler uses `gbrain capture` under the hood, so
    # the mapping is mandatory here.
    scope_cfg = ((scopes_data or {}).get("scopes") or {}).get(scope) or {}
    gbrain_source_id = (
        str(scope_cfg.get("gbrain_source") or "").strip()
        or scope.replace("_", "-")
    )
    # Belt-and-braces: gbrain CLI enforces this at the receiver side too.
    if not re.match(r"^[a-z0-9-]{1,32}$", gbrain_source_id):
        _log(
            "error", "save_document_to_scope: derived gbrain_source_id invalid",
            sender=sender_id, scope=scope, gbrain_source=gbrain_source_id,
        )
        return _reply(True, {
            "status": "error",
            "reason": "invalid_gbrain_source",
            "detail": f"scopes.yaml scope '{scope}' maps to '{gbrain_source_id}' "
                      f"which does not match [a-z0-9-]{{1,32}}. Fix the "
                      f"gbrain_source field in scopes.yaml.",
        })

    _log(
        "info", "save_document_to_scope dispatch",
        sender=sender_id, role=role, scope=scope,
        gbrain_source=gbrain_source_id, slug=derived_slug,
        size_bytes=size_bytes, mime_type=mime_type,
        content_hash=content_hash,
    )

    try:
        result = gbrain_ingest_document(
            local_path=real_path,
            source_id=gbrain_source_id,
            slug=derived_slug,
            title=title,
            mime_type=mime_type,
        )
    except Exception as exc:
        _log(
            "error", "save_document_to_scope: ingest raised",
            sender=sender_id, scope=scope, slug=derived_slug, error=str(exc),
        )
        return _reply(True, {
            "status": "error",
            "reason": "ingest_failed",
            "detail": str(exc),
        })

    # IngestResult contract (Track B — see agent/gbrain_ingest_document.py):
    #   result.status         — str, one of {"created", "updated",
    #                           "already_ingested", "failed", "dry_run"}
    #   result.slug           — str, final slug (may differ from derived);
    #                           "" on failure
    #   result.page_id        — Optional[int], gbrain page id (None if not
    #                           returned by CLI)
    #   result.content_hash   — str, hex sha256 stored on the page
    #   result.chunks_created — int
    #   result.warnings       — list[str], non-fatal notices (OCR fallback,
    #                           embedding_failed, blob_mirror_failed, ...)
    #   result.error          — Optional[str], present when status=="failed"
    result_status = str(getattr(result, "status", "") or "").lower()
    slug_out = getattr(result, "slug", None) or derived_slug
    page_id = getattr(result, "page_id", None)
    hash_out = getattr(result, "content_hash", None) or content_hash
    warnings = list(getattr(result, "warnings", None) or [])

    # Propagate failure from Track B. Previously this handler unconditionally
    # returned "created_or_updated" regardless of result.status — a genuine
    # capture failure (missing gbrain binary, gbrain CLI returncode!=0,
    # blob mirror hard-fail, etc.) reached the caller as a success envelope.
    # Now: failed status → isError=True with the reason surfaced.
    if result_status == "failed":
        error_detail = getattr(result, "error", None) or "unknown_ingest_failure"
        _log(
            "error", "save_document_to_scope: ingest returned failed",
            sender=sender_id, scope=scope, slug=slug_out,
            error=error_detail, warnings=len(warnings),
        )
        return _reply(True, {
            "status": "error",
            "reason": "ingest_failed",
            "detail": error_detail,
            "slug": slug_out,
            "content_hash": hash_out,
            "scope": scope,
            "warnings": warnings,
        })

    if result_status == "already_ingested":
        envelope_status = "already_saved"
    elif result_status in ("created", "updated"):
        envelope_status = "created_or_updated"
    elif result_status == "dry_run":
        envelope_status = "dry_run"
    else:
        # Unknown status — surface it verbatim rather than lie.
        envelope_status = result_status or "unknown"
        warnings.append(f"unknown_ingest_status:{result_status}")

    _log(
        "info", "save_document_to_scope success",
        sender=sender_id, scope=scope, slug=slug_out,
        page_id=page_id, ingest_status=result_status,
        envelope_status=envelope_status, warnings=len(warnings),
    )
    return _reply(False, {
        "status": envelope_status,
        "slug": slug_out,
        "page_id": page_id,
        "content_hash": hash_out,
        "scope": scope,
        "warnings": warnings,
    })


def _handle_tools_call(
    req: Dict[str, Any],
    scopes_data: Dict[str, Any],
    bearers: Dict[str, str],
    gbrain_url: str,
    timeout: float,
    scopes_yaml_path: str,
    env_path: str,
    pending_path: str,
    bearers_path: str = "",
) -> Dict[str, Any]:
    params = req.get("params") or {}
    name = str(params.get("name") or "")
    args = params.get("arguments") or {}

    # v0.2: sender_id is a REQUIRED per-call arg now (not a startup env var).
    # Persona is responsible for passing the verified sender's lid here.
    sender_id = str(args.get("sender_id") or "").strip()
    if not _SENDER_ID_RE.match(sender_id):
        _log(
            "warn",
            "tool call refused — sender_id missing or malformed",
            tool=name,
            sender_id_received=sender_id or "(empty)",
        )
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": (
                    "sender_id required: every hermes_save:* call must "
                    "include the verified sender's lid (format "
                    "`<wa_id digits>`) as the `sender_id` argument. "
                    "Source it from the most recent <verified_sender "
                    "id=\"...\"/> marker on the user's message; "
                    "never invent it."
                ),
            }],
        }
    role = _resolve_role(scopes_data, sender_id)

    if name == "approve_user":
        return _handle_approve_user(
            args, scopes_data, sender_id, role, scopes_yaml_path,
        )
    if name == "add_to_allowlist":
        return _handle_add_to_allowlist(args, scopes_data, sender_id, env_path)
    if name == "record_pending_user":
        return _handle_record_pending_user(args, pending_path)
    if name == "list_pending_users":
        return _handle_list_pending_users(
            scopes_data, sender_id, pending_path, env_path,
        )
    if name == "list_allowlist":
        return _handle_list_allowlist(scopes_data, sender_id, env_path)
    if name == "remove_from_allowlist":
        return _handle_remove_from_allowlist(args, scopes_data, sender_id, env_path)
    if name == "revoke_user":
        return _handle_revoke_user(
            args, scopes_data, sender_id, scopes_yaml_path, pending_path,
        )
    if name == "grant_scope_access":
        return _handle_grant_scope_access(args, scopes_data, sender_id, scopes_yaml_path)
    if name == "revoke_scope_access":
        return _handle_revoke_scope_access(args, scopes_data, sender_id, scopes_yaml_path)
    if name == "reload_gateway":
        return _handle_reload_gateway(scopes_data, sender_id)
    if name == "send_template_message":
        return _handle_send_template_message(args, scopes_data, sender_id)
    if name == "save_document_to_scope":
        return _handle_save_document_to_scope(
            args, scopes_data, bearers, sender_id, role, gbrain_url, timeout,
        )
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
            sender=sender_id,
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
            sender=sender_id,
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
            sender=sender_id,
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
        sender=sender_id,
        role=role,
        scope=scope,
        writer_role=writer_role,
        title=title,
        slug=slug,
        body_len=len(body),
    )
    # Single attempt with optional one-shot bearer refresh on 401. When
    # HERMES_SAVE_BEARER_REFRESH_CMD is unset, _refresh_role_bearer returns
    # None and behavior matches v0.2 exactly (just log + surface the error).
    auth_retried = False
    while True:
        try:
            result = _gbrain_put_page(gbrain_url, bearer, title, body, slug, timeout)
            break
        except urllib.error.HTTPError as exc:
            snippet = ""
            try:
                snippet = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            if exc.code == 401 and not auth_retried:
                _log(
                    "warn", "gbrain put_page 401 — attempting bearer refresh",
                    sender=sender_id, role=role, writer_role=writer_role,
                    snippet=snippet,
                )
                new_bearer = _refresh_role_bearer(writer_role, bearers_path)
                if new_bearer:
                    bearer = new_bearer
                    bearers[writer_role] = new_bearer
                    auth_retried = True
                    continue
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
        _log("info", "save_to_scope success", sender=sender_id, scope=scope)
        return inner
    _log("info", "save_to_scope returned raw envelope", sender=sender_id, scope=scope)
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
    # sender_id arg.
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
                        pending_path, bearers_path,
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
