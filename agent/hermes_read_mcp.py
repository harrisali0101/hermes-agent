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

_SENDER_ID_RE = re.compile(r"^\d{8,15}$")

# The sender_id contract is IDENTICAL to hermes_save_mcp.py: every tool
# call carries the verified sender's WhatsApp wa_id as a per-call arg.
# Persona is responsible for sourcing it verbatim from the most recent
# <verified_sender id="..."/> marker on the user's message. Missing /
# malformed → server refuses the call, never falls back to a default.
_SENDER_LID_SCHEMA: Dict[str, Any] = {
    "type": "string",
    "pattern": r"^\d{8,15}$",
    "description": (
        "REQUIRED. The verified sender's WhatsApp wa_id (format "
        "`<wa_id digits>`), copied verbatim from the most recent "
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
# status_sweep is HERMES-side composite (not a gbrain tool) — it
# orchestrates query + get_page + sort into one atomic sweep so the
# model can't skip discovery steps on status questions.
_READ_TOOLS = ("query", "search", "get_page", "list_pages", "status_sweep")


# ── status_sweep helpers — content-date extraction + anchor pick ─────────

# Structured-doc title fingerprint — LIVE STATUS SNAPSHOTS only.
# Matches the anchor pattern from STATUS_QUERY.md: recurring, dated
# operational docs that summarize project state at a point in time.
#
# Excluded on purpose (2026-07-05 tune after WA E2E):
#   - `register` and `memo` — these are one-off compiled notes, not live
#     status snapshots. Including them let a `ceo-obligations-register-*`
#     page beat the actual `Buyer Internal Closing Checklist` as anchor.
#     The persona model correctly overrode this ("The July-5 CEO
#     obligations register is a compiled note, not a live status
#     snapshot. The real closing-status anchor is the 30 June Buyer
#     Internal Closing Checklist"), but the tool should produce the
#     right answer by default.
#
# Case-insensitive, word-boundaried so "status" doesn't match "statusquo".
_ANCHOR_TITLE_RE = re.compile(
    r"\b(checklist|dashboard|status|tracker|summary|briefing)\b",
    re.IGNORECASE,
)


def _content_date(page: Any) -> Optional[str]:
    """Extract the best content-date signal from a gbrain page dict.

    Fallback chain (STATUS_QUERY.md §4 discipline):
      1. frontmatter.date        — the AUTHORED date (email send, doc write)
      2. frontmatter.sent_at     — older email-record shape
      3. effective_date          — gbrain-computed top-level date signal
      4. frontmatter.captured_at — when the ingest captured it
      5. created_at              — first-ingest timestamp
      6. updated_at              — LAST-RESORT re-ingest timestamp (a stale
                                    email re-ingested today looks "newest"
                                    by updated_at — do NOT use this alone)

    Returns an ISO-8601 string or None. String comparison is safe on
    ISO-8601 (`2026-07-05T...` > `2026-06-30T...` lexically).
    """
    if not isinstance(page, dict):
        return None
    fm = page.get("frontmatter") if isinstance(page.get("frontmatter"), dict) else {}
    candidates = (
        fm.get("date"),
        fm.get("sent_at"),
        page.get("effective_date"),
        fm.get("captured_at"),
        page.get("created_at"),
        page.get("updated_at"),
    )
    for c in candidates:
        if c:
            return str(c)
    return None


def _pick_anchor(pages: list) -> Optional[Dict[str, Any]]:
    """From an enriched page list, pick the anchor: newest structured doc
    whose title matches _ANCHOR_TITLE_RE. Returns None if no page matches
    the regex — signals to the caller that no live status snapshot exists
    to anchor against, so all enriched items should be surfaced as deltas.
    Returns None on empty input too.

    2026-07-08 (see azure/design/status-sweep-sparse-flag-bug.md): the
    prior "newest by content_date" fallback caused the delta filter
    `content_date > anchor_date` to always return [] (nothing is strictly
    newer than the newest), which tripped sparse_flag=True on every run
    once the corpus stopped containing anchor-regex-matching titles.
    The caller in _handle_status_sweep_orchestration already has the
    correct `if anchor is None:` branch wired — that branch was dead
    code before this fix."""
    if not pages:
        return None
    matches = [p for p in pages if isinstance(p, dict) and _ANCHOR_TITLE_RE.search(str(p.get("title") or ""))]
    if not matches:
        return None
    matches.sort(key=lambda p: _content_date(p) or "", reverse=True)
    return matches[0]


def _shape_page_for_envelope(page: Dict[str, Any]) -> Dict[str, Any]:
    """Compact page shape for the status_sweep return envelope. Keeps
    body + metadata the model needs to compose an answer with citations;
    drops server-internal noise (content_hash, source_uri, ingested_at,
    timeline)."""
    if not isinstance(page, dict):
        return {}
    return {
        "slug": page.get("slug"),
        "title": page.get("title"),
        "type": page.get("type"),
        "content_date": _content_date(page),
        "source_id": page.get("source_id"),
        "body": page.get("compiled_truth") or page.get("body") or "",
        "tags": page.get("tags") or [],
    }


def _extract_gbrain_content(result: Any) -> Any:
    """Given gbrain's raw JSON-RPC result envelope, return the parsed
    inner content or None. gbrain always wraps the payload as
    ``{result: {content: [{type: 'text', text: '<json-string>'}]}}``.
    Failure modes handled: missing keys, isError set, non-JSON text.
    """
    if not isinstance(result, dict):
        return None
    inner = result.get("result")
    if not isinstance(inner, dict):
        return None
    if inner.get("isError"):
        return None
    content = inner.get("content")
    if not isinstance(content, list) or not content:
        return None
    first = content[0]
    if not isinstance(first, dict):
        return None
    text = first.get("text")
    if not isinstance(text, str) or not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _extract_query_results(result: Any) -> list:
    """query returns a list of ranked chunks. Some gbrain versions wrap
    in {results: [...]}; handle both shapes."""
    parsed = _extract_gbrain_content(result)
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for k in ("results", "chunks", "hits", "items"):
            v = parsed.get(k)
            if isinstance(v, list):
                return v
    return []


def _extract_list_pages_results(result: Any) -> list:
    """list_pages returns a list of page rows."""
    parsed = _extract_gbrain_content(result)
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("pages"), list):
        return parsed["pages"]
    return []


def _extract_get_page_result(result: Any) -> Optional[Dict[str, Any]]:
    """get_page returns a single page dict."""
    parsed = _extract_gbrain_content(result)
    if isinstance(parsed, dict):
        return parsed
    return None


def _handle_status_sweep_orchestration(
    args: Dict[str, Any],
    gbrain_url: str,
    bearer: str,
    timeout: float,
) -> Dict[str, Any]:
    """Composite status-sweep orchestrator (Option C from the design memo
    at azure/design/recency-first-status-query.md).

    Sequence — atomic, model cannot skip a step (v2 2026-07-05):
      1. gbrain.list_pages(sort='updated_desc', limit=100) — pure
         chronological discovery across the caller's authorized sources
         (RLS-enforced). No semantic filter → no under-recall on pages
         whose body doesn't semantically match the project name.
      2. Candidate build: substring-hit rows (slug/title contains the
         project name) FIRST, then top-15 chronological overall
         (belt-and-braces for pages whose slug lacks the project name,
         e.g. `ceo-obligations-register-*` inside project-mesec).
      3. FALLBACK — if list_pages returned nothing: gbrain.query with
         source_id + recency='strong'. Semantic-narrow risk vs primary,
         but any hit beats an empty envelope.
      4. Deduplicate by slug, preserving priority order.
      5. Enrich top 15 unique slugs via gbrain.get_page (full text +
         frontmatter for content-date extraction + source_id verify).
      6. Pick anchor: newest structured doc by _ANCHOR_TITLE_RE. Fall
         back to newest by content-date if no title match.
      7. Deltas: pages newer than anchor by content-date, sorted desc,
         capped at delta_limit.
      8. Envelope: {anchor, deltas, sparse_flag, sample_size, source_id,
                    warnings}.

    Fallback layers:
      - list_pages fails → query with source_id + recency='strong'.
      - Both fail → empty envelope with sparse_flag=True + warnings.
      - get_page fails on a candidate → skip it, log warning, continue.
      - get_page succeeds but source_id != project → drop (cross-scope
        leak from top-overall fetch).
      - No pages could be enriched → empty envelope + warnings.
      - anchor pick returns None → all enriched items become deltas
        (no anchor split).
      - date parse ambiguity → falls through the _content_date chain
        rather than crashing.

    The persona rule in STATUS_QUERY.md remains as a safety net for
    edge cases (drill-downs into a single delta, sparse-project handling).

    v1 (query-primary, shipped morning 2026-07-05) is retained as the
    fallback path. v2 promotes chronological to primary based on the
    5-message WhatsApp E2E test that revealed semantic under-recall
    (CEO Closing Checklist missed the first sweep because "mesec" isn't
    prominent in its body).
    """
    project = str(args.get("project") or "").strip()
    if not project:
        return {
            "isError": True,
            "content": [{"type": "text",
                         "text": "status_sweep requires project (e.g. 'project-mesec')"}],
        }

    # Derive a query hint. When the caller doesn't provide `topic`, we
    # derive from the project name — the persona is expected to pass the
    # user's actual question phrasing as `topic` (e.g. "what's the latest
    # on Mesec" → topic="latest Mesec"), which triggers gbrain's recency
    # intent classifier and pairs with source_id + recency='strong'.
    # Note: fallback to the project name alone can under-recall when
    # today's fresh content doesn't mention the project name in body
    # (e.g. a scope-tagged CEO register whose title is a legal term).
    # The `sparse_flag` in the envelope signals this cleanly so the
    # persona widens rather than hallucinates.
    topic_arg = str(args.get("topic") or "").strip()
    caller_provided_topic = bool(topic_arg)
    if topic_arg:
        topic = topic_arg
    else:
        stripped = project
        for prefix in ("project-", "projects/"):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                break
        topic = stripped.replace("-", " ").replace("_", " ").strip() or project

    try:
        delta_limit = int(args.get("delta_limit") or 5)
    except (TypeError, ValueError):
        delta_limit = 5
    delta_limit = max(1, min(delta_limit, 10))

    warnings: list = []
    candidates: list = []

    # Derive project needles for the substring pre-filter, used to
    # prioritise candidates whose slug or title mentions the project by
    # name (email-2026-06-19-mesec-*, etc.). Belt-and-braces catch for
    # pages whose slug doesn't carry the name (ceo-obligations-register-*)
    # is handled by also including the top-N chronological items overall.
    project_short = project
    for prefix in ("project-", "projects/"):
        if project_short.startswith(prefix):
            project_short = project_short[len(prefix):]
            break
    needles = {n for n in (project.lower(), project_short.lower()) if n}

    # Step 1 — Discovery. Two modes based on whether the caller narrowed
    # the topic:
    #
    #   NARROW MODE (caller-provided topic like "Mandate Letter"):
    #     query(query=topic, source_id=project, recency='strong', limit=30)
    #     is primary. Semantic filtering matters — the caller is asking
    #     about a SPECIFIC facet of the project, not "everything recent".
    #     list_pages fills gaps if query is empty (rare).
    #
    #   LATEST MODE (topic derived from project name, i.e. no narrowing):
    #     list_pages(sort='updated_desc', limit=100) is primary — pure
    #     chronological across the caller's allowed_sources. No semantic
    #     filter → no under-recall on pages whose body doesn't mention
    #     the project name (e.g. "CEO obligations register" filed under
    #     project-mesec but semantically about SPA/MSA legal terms).
    #     Substring pre-filter (project name in slug/title) backfills
    #     older in-project items beyond position 15.
    #
    # Both modes converge at Step 2 (dedup + enrich + anchor/deltas).
    if caller_provided_topic:
        # NARROW: query-primary path.
        try:
            q_result = _gbrain_tool_call(
                gbrain_url, bearer, "query",
                {"query": topic, "source_id": project, "recency": "strong", "limit": 30},
                timeout,
            )
            candidates = _extract_query_results(q_result)
            _log("info", "status_sweep NARROW: primary query returned",
                 project=project, topic=topic, count=len(candidates))
        except urllib.error.HTTPError as exc:
            _log("warn", "status_sweep NARROW: primary query HTTPError, will try list_pages fallback",
                 code=exc.code, project=project)
            warnings.append(f"NARROW primary query HTTP {exc.code} — using list_pages fallback")
        except Exception as exc:
            _log("warn", "status_sweep NARROW: primary query failed, will try list_pages fallback",
                 error=str(exc), project=project)
            warnings.append(f"NARROW primary query failed ({type(exc).__name__}) — using list_pages fallback")

        # Fallback for NARROW mode: list_pages + substring filter (same
        # heuristic as LATEST mode's substr_hits but also uses the topic
        # as a needle — catches project items with the topic keyword in
        # slug/title).
        if not candidates:
            try:
                lp_result = _gbrain_tool_call(
                    gbrain_url, bearer, "list_pages",
                    {"sort": "updated_desc", "limit": 100},
                    timeout,
                )
                all_meta = _extract_list_pages_results(lp_result)
                topic_needles = {n for n in (topic.lower(),
                                             topic.lower().replace(" ", "-")) if n}
                candidates = [
                    p for p in all_meta
                    if isinstance(p, dict) and any(
                        n in str(p.get("slug") or "").lower()
                        or n in str(p.get("title") or "").lower()
                        for n in (needles | topic_needles)
                    )
                ]
                _log("info", "status_sweep NARROW: list_pages fallback filtered",
                     project=project, topic=topic, count=len(candidates))
                if candidates:
                    warnings.append("NARROW: primary query returned nothing — used list_pages fallback with topic substring filter")
            except Exception as exc:
                _log("error", "status_sweep NARROW: list_pages fallback also failed",
                     error=str(exc), project=project)
                warnings.append(f"list_pages fallback failed: {type(exc).__name__}")

    else:
        # LATEST: list_pages-primary path.
        try:
            lp_result = _gbrain_tool_call(
                gbrain_url, bearer, "list_pages",
                {"sort": "updated_desc", "limit": 100},
                timeout,
            )
            all_meta = _extract_list_pages_results(lp_result)
            _log("info", "status_sweep LATEST: primary list_pages returned",
                 project=project, count=len(all_meta))
            # Build candidate ordering: TOP-15 chronological FIRST (they
            # are the newest across the caller's allowed_sources —
            # guaranteed to include any recently-touched page in the
            # project, whether or not its slug carries the project name
            # — e.g. a `ceo-obligations-register-*` under project-mesec).
            # Substring hits (project name in slug/title) BACKFILL beyond
            # position 15 to catch older in-project items that fell out
            # of the chronological top slice.
            #
            # Dedup below preserves this priority order (first-seen wins).
            top_overall = all_meta[:15]
            substr_hits = [
                p for p in all_meta[15:]
                if isinstance(p, dict) and any(
                    n in str(p.get("slug") or "").lower()
                    or n in str(p.get("title") or "").lower()
                    for n in needles
                )
            ]
            candidates = top_overall + substr_hits
        except urllib.error.HTTPError as exc:
            _log("warn", "status_sweep LATEST: primary list_pages HTTPError, will try query fallback",
                 code=exc.code, project=project)
            warnings.append(f"LATEST primary list_pages HTTP {exc.code} — using query fallback")
        except Exception as exc:
            _log("warn", "status_sweep LATEST: primary list_pages failed, will try query fallback",
                 error=str(exc), project=project)
            warnings.append(f"LATEST primary list_pages failed ({type(exc).__name__}) — using query fallback")

        # Fallback for LATEST mode: query with source_id + recency='strong'.
        # Fires only if list_pages returned nothing (rare — new senders
        # or HTTP fail).
        if not candidates:
            try:
                q_result = _gbrain_tool_call(
                    gbrain_url, bearer, "query",
                    {"query": topic, "source_id": project, "recency": "strong", "limit": 30},
                    timeout,
                )
                candidates = _extract_query_results(q_result)
                _log("info", "status_sweep LATEST: fallback query returned",
                     project=project, topic=topic, count=len(candidates))
                if candidates:
                    warnings.append("LATEST: primary list_pages returned nothing — used query fallback")
            except Exception as exc:
                _log("error", "status_sweep LATEST: fallback query also failed",
                     error=str(exc), project=project)
                warnings.append(f"query fallback failed: {type(exc).__name__}")

    # Deduplicate by slug, preserving discovery order (substring hits
    # first, then chronological, then query fallback if it fired).
    seen: set = set()
    unique: list = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        slug = str(c.get("slug") or "").strip()
        if slug and slug not in seen:
            seen.add(slug)
            unique.append(c)

    if not unique:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "anchor": None,
                    "deltas": [],
                    "sparse_flag": True,
                    "sample_size": 0,
                    "source_id": project,
                    "warnings": warnings + [
                        f"no candidates found for project={project} — "
                        "either the project has no content or the sender "
                        "lacks read access (RLS)."
                    ],
                })
            }]
        }

    # Step 3 — Enrich top-15 via get_page. Bumped from v1's 8 → 15 to
    # accommodate the chronological-primary path: substring hits pick
    # up in-project items across the recent tail; the top-overall block
    # backstops pages whose slug doesn't carry the project name. 15 keeps
    # latency ~4-5s in the worst case (15 sequential get_pages @ ~300ms
    # each). Downstream anchor+delta selection caps output at delta_limit.
    enrich_max = min(15, len(unique))
    top = unique[:enrich_max]

    enriched: list = []
    for c in top:
        slug = str(c.get("slug") or "").strip()
        if not slug:
            continue
        try:
            gp_result = _gbrain_tool_call(
                gbrain_url, bearer, "get_page", {"slug": slug}, timeout,
            )
            page = _extract_get_page_result(gp_result)
            if not page:
                warnings.append(f"get_page returned empty for {slug} — skipped")
                continue
            # Belt-and-braces: verify source_id matches project (relevant
            # for the list_pages fallback path where we filtered by
            # substring — a stray match might belong to another scope).
            page_source = str(page.get("source_id") or "").strip()
            if page_source and page_source != project:
                _log("info", "status_sweep: dropped cross-scope match",
                     slug=slug, page_source=page_source, expected=project)
                continue
            enriched.append(page)
        except urllib.error.HTTPError as exc:
            _log("warn", "status_sweep: get_page HTTPError, skipping candidate",
                 slug=slug, code=exc.code)
            warnings.append(f"get_page {slug}: HTTP {exc.code}, skipped")
        except Exception as exc:
            _log("warn", "status_sweep: get_page failed, skipping candidate",
                 slug=slug, error=str(exc))
            warnings.append(f"get_page {slug}: {type(exc).__name__}, skipped")

    if not enriched:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "anchor": None,
                    "deltas": [],
                    "sparse_flag": True,
                    "sample_size": 0,
                    "source_id": project,
                    "warnings": warnings + [
                        "no pages could be enriched — every candidate get_page failed"
                    ],
                })
            }]
        }

    # Step 3 — Anchor + deltas split by content date.
    anchor = _pick_anchor(enriched)
    if anchor is None:
        # Degenerate case: nothing matches the anchor regex AND enriched
        # is somehow non-empty (shouldn't happen since fallback is
        # "newest overall"). Return everything as deltas.
        deltas = sorted(enriched, key=lambda p: _content_date(p) or "", reverse=True)[:delta_limit]
        warnings.append("no anchor identified — returned all enriched items as deltas")
    else:
        anchor_date = _content_date(anchor) or ""
        deltas_pool = [
            p for p in enriched
            if p.get("slug") != anchor.get("slug")
            and _content_date(p)
            and (_content_date(p) or "") > anchor_date
        ]
        deltas_pool.sort(key=lambda p: _content_date(p) or "", reverse=True)
        deltas = deltas_pool[:delta_limit]

    sparse = len(deltas) < 3
    if sparse:
        warnings.append(
            f"sparse project: only {len(deltas)} delta(s) newer than anchor — "
            "persona should surface this explicitly in the reply"
        )

    envelope = {
        "anchor": _shape_page_for_envelope(anchor) if anchor else None,
        "deltas": [_shape_page_for_envelope(p) for p in deltas],
        "sparse_flag": sparse,
        "sample_size": len(deltas),
        "source_id": project,
        "warnings": warnings,
    }
    return {"content": [{"type": "text", "text": json.dumps(envelope)}]}


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


def _resolve_role(scopes_data: Dict[str, Any], sender_id: str) -> Optional[str]:
    """Look up sender role for audit logging. Returns None if not found —
    that's allowed: gbrain still enforces RLS via the subjects table even
    when this side can't classify the role. We just log "role=unknown"."""
    for entry in scopes_data.get("super_admins") or []:
        if str(entry.get("id", "")).strip() == sender_id:
            return "super_admin"
    for entry in scopes_data.get("users") or []:
        if str(entry.get("id", "")).strip() == sender_id:
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
                    "sources never appear in the result set. For "
                    "status/progress/latest-on questions, prefer the "
                    "composite `status_sweep` tool instead — it "
                    "enforces the anchor+deltas discipline atomically."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sender_id": _SENDER_LID_SCHEMA,
                        "q": {"type": "string", "description": "Natural-language query."},
                        "limit": {
                            "type": "integer",
                            "description": "Max passages (default 8).",
                            "minimum": 1, "maximum": 50,
                        },
                        "source_id": {
                            "type": "string",
                            "description": (
                                "Optional. Scope the query to a single "
                                "gbrain source (e.g. 'project-mesec', "
                                "'finance', 'leadership'). Default: all "
                                "the sender's authorized sources. Pass "
                                "'__all__' to explicitly span all "
                                "sources when a source is set at the "
                                "session level."
                            ),
                        },
                        "recency": {
                            "type": "string",
                            "enum": ["off", "on", "strong"],
                            "description": (
                                "Recency boost on the ranker. 'off' for "
                                "canonical-truth questions ('who is X'); "
                                "'on' for standard questions with mild "
                                "freshness tilt; 'strong' for status / "
                                "progress / 'latest' questions where "
                                "freshness dominates relevance. When "
                                "omitted, gbrain auto-detects from "
                                "phrasing — but the auto-detect is "
                                "weak for project-* scopes; be explicit."
                            ),
                        },
                        "since": {
                            "type": "string",
                            "description": (
                                "Optional. Filter to pages whose "
                                "effective_date is >= this. ISO-8601 "
                                "(YYYY-MM-DD or full timestamp) OR "
                                "relative shorthand ('7d', '2w', '1y'). "
                                "Combine with `until` for a range."
                            ),
                        },
                        "until": {
                            "type": "string",
                            "description": (
                                "Optional. Filter to effective_date <= "
                                "this. Same format as `since`. YYYY-MM-DD "
                                "lands at end-of-day."
                            ),
                        },
                    },
                    "required": ["sender_id", "q"],
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
                        "sender_id": _SENDER_LID_SCHEMA,
                        "q": {"type": "string", "description": "Exact keywords or phrase."},
                        "limit": {
                            "type": "integer",
                            "description": "Max hits (default 8).",
                            "minimum": 1, "maximum": 50,
                        },
                    },
                    "required": ["sender_id", "q"],
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
                        "sender_id": _SENDER_LID_SCHEMA,
                        "slug": {"type": "string", "description": "Page slug (kebab-case)."},
                    },
                    "required": ["sender_id", "slug"],
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
                        "sender_id": _SENDER_LID_SCHEMA,
                        "limit": {
                            "type": "integer",
                            "description": "Max pages (default 25).",
                            "minimum": 1, "maximum": 200,
                        },
                    },
                    "required": ["sender_id"],
                },
            },
            {
                "name": "status_sweep",
                "description": (
                    "STATUS PRIMITIVE — atomic recency-first sweep for "
                    "a project. USE FOR: 'what's the latest on X', "
                    "'what's outstanding on Y', 'update on Z', "
                    "'progress on <project>', 'any news on <project>'. "
                    "DO NOT USE FOR: definitional questions ('who is "
                    "X'), semantic search inside a project, or single-"
                    "page lookups — use `query` or `get_page` for "
                    "those. Returns (a) the newest structured anchor "
                    "doc (checklist/dashboard/status/tracker/summary/"
                    "memo/register/briefing) and (b) up to N items "
                    "dated newer than that anchor, each with body + "
                    "content_date + tags + source_id. Enforces the "
                    "two-source rule (STATUS_QUERY.md) at the tool "
                    "boundary so the model cannot skip discovery. "
                    "RLS-enforced same as query/get_page — the sender "
                    "must have read access to the project scope."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sender_id": _SENDER_LID_SCHEMA,
                        "project": {
                            "type": "string",
                            "description": (
                                "Project source ID (matches gbrain "
                                "slug namespace, e.g. 'project-mesec'). "
                                "Sender must have read access to this "
                                "scope via allowed_sources (RLS-"
                                "enforced at SQL layer)."
                            ),
                        },
                        "topic": {
                            "type": "string",
                            "description": (
                                "Optional query hint. Defaults to a "
                                "derived query from the project name. "
                                "Use when the user's status question "
                                "narrows the topic (e.g. 'latest on "
                                "the Mandate Letter for Mesec' → "
                                "topic='Mandate Letter')."
                            ),
                        },
                        "delta_limit": {
                            "type": "integer",
                            "minimum": 1, "maximum": 10,
                            "description": (
                                "Max delta items to return after the "
                                "anchor (default 5). Cap 10 to keep "
                                "the response bounded."
                            ),
                        },
                    },
                    "required": ["sender_id", "project"],
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
      1. Validate sender_id (refuse if missing/malformed).
      2. Mint a per-subject access token via the gbrain_exchange.TokenCache.
      3. Forward the tool call to gbrain /mcp with that bearer.
      4. On 401 → invalidate cache so next call re-mints (handles the
         race where the subject was revoked between mint and use).
    """
    params = req.get("params") or {}
    name = str(params.get("name") or "")
    args = params.get("arguments") or {}

    sender_id = str(args.get("sender_id") or "").strip()
    if not _SENDER_ID_RE.match(sender_id):
        _log("warn", "tool call refused — sender_id missing or malformed",
             tool=name, sender_id_received=sender_id or "(empty)")
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": (
                    "sender_id required: every hermes_read:* call must "
                    "include the verified sender's lid (format "
                    "`<wa_id digits>`) as the `sender_id` argument. "
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
    # status_sweep is a Hermes-side composite; it doesn't map 1:1 to a
    # gbrain tool. Dispatch to the orchestrator after bearer mint.
    if name == "status_sweep":
        gbrain_args = {}  # unused for status_sweep; orchestrator builds its own
    elif name in ("query", "search"):
        q = str(args.get("q") or "").strip()
        if not q:
            return {"isError": True, "content": [{"type": "text", "text": f"{name} requires q"}]}
        # gbrain's `query` and `search` operations both take the search
        # text under the canonical name `query` (not `q`). Our exposed
        # tool schema uses `q` for brevity at the chat-tool layer; we
        # translate here. See gbrain operations.ts: search/query both
        # declare `query`.
        gbrain_args: Dict[str, Any] = {"query": q}
        if isinstance(args.get("limit"), int):
            gbrain_args["limit"] = args["limit"]
        # Thread through the recency/source/date knobs when the model
        # sets them. Only present on `query` (not `search`) — search is
        # a plain keyword primitive on gbrain's side. Silently drop any
        # of these if the caller included them on `search`.
        if name == "query":
            source_id = str(args.get("source_id") or "").strip()
            if source_id:
                gbrain_args["source_id"] = source_id
            recency = str(args.get("recency") or "").strip().lower()
            if recency in ("off", "on", "strong"):
                gbrain_args["recency"] = recency
            since = str(args.get("since") or "").strip()
            if since:
                gbrain_args["since"] = since
            until = str(args.get("until") or "").strip()
            if until:
                gbrain_args["until"] = until
    elif name == "get_page":
        slug = str(args.get("slug") or "").strip()
        if not slug:
            return {"isError": True, "content": [{"type": "text", "text": "get_page requires slug"}]}
        gbrain_args = {"slug": slug}
    else:  # list_pages
        # gbrain list_pages has no per-call source filter — scope is
        # derived from the subject's allowed_sources. If the caller
        # passes source we drop it (legacy schema field), and warn
        # in the audit log so the model can learn to omit it.
        gbrain_args = {}
        if isinstance(args.get("limit"), int):
            gbrain_args["limit"] = args["limit"]
        # Sensible default: recency-ordered. Bot uses list_pages
        # as the discovery primitive — newest first is what it
        # actually wants 95% of the time.
        gbrain_args.setdefault("sort", "updated_desc")

    role = _resolve_role(scopes_data, sender_id)
    try:
        bearer = token_cache.for_subject(sender_id)
    except Exception as exc:
        _log("error", "token exchange failed", sender=sender_id, tool=name, error=str(exc))
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

    # status_sweep — composite orchestrator. Runs 1× query + up to 8×
    # get_page under the same bearer. Errors handled internally with
    # multi-layer fallbacks; envelope always returns.
    if name == "status_sweep":
        _log("info", "subject read dispatch",
             sender=sender_id, role=role or "unknown",
             tool=name, project=str(args.get("project") or ""),
             topic=str(args.get("topic") or "").strip() or "(derived)")
        try:
            return _handle_status_sweep_orchestration(args, gbrain_url, bearer, timeout)
        except urllib.error.HTTPError as exc:
            # Only reached if orchestration re-raises (it currently
            # catches HTTPError on each sub-call). Belt-and-braces.
            if exc.code == 401:
                try:
                    token_cache.invalidate(sender_id)
                except Exception:
                    pass
            _log("error", "status_sweep top-level HTTPError",
                 code=exc.code, sender=sender_id)
            return {
                "isError": True,
                "content": [{"type": "text",
                             "text": f"status_sweep HTTP {exc.code}"}],
            }
        except Exception as exc:
            _log("error", "status_sweep top-level exception",
                 error=str(exc), sender=sender_id)
            return {
                "isError": True,
                "content": [{"type": "text",
                             "text": f"status_sweep failed: {type(exc).__name__}: {exc}"}],
            }

    _log("info", "subject read dispatch",
         sender=sender_id, role=role or "unknown",
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
                token_cache.invalidate(sender_id)
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
