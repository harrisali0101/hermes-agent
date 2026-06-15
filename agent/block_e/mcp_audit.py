"""MCP-call audit for Block E.

Queries gbrain's ``mcp_request_log`` table to find what retrieval tools the
bot's OAuth client called during this turn. The gate uses this to detect a
specific failure mode the persona alone can't catch: a fabricated DIH
answer where the bot never actually queried the brain.

We shell out to ``psql`` instead of adding ``psycopg2`` as a hermes-agent
dependency — psql is on the VM, the query is read-only, and this matches
the pattern used elsewhere in hermes-agent for ad-hoc PG access.

Required retrieval operations (any one counts as "the bot tried"):
  ``query``, ``search``, ``search_by_image``, ``get_page``, ``recall``,
  ``find_experts``, ``find_orphans``, ``traverse_graph``, ``list_pages``,
  ``find_contradictions``, ``takes_search``, ``find_trajectory``.

Operations that do NOT count (writes / metadata / orchestration):
  ``put_page``, ``delete_page``, ``add_link``, ``submit_job``, ``get_stats``,
  ``list_skills``, ``whoami``, ``tools/list``, etc.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import List

logger = logging.getLogger(__name__)

# Read-only retrieval operations. Calling any of these proves the bot at
# least attempted to ground its answer in gbrain rather than hallucinate.
RETRIEVAL_OPS = frozenset({
    "query",
    "search",
    "search_by_image",
    "get_page",
    "recall",
    "find_experts",
    "find_orphans",
    "traverse_graph",
    "list_pages",
    "find_contradictions",
    "takes_search",
    "find_trajectory",
    "get_chunks",
    "resolve_slugs",
    "get_links",
    "get_backlinks",
    "list_link_sources",
    "get_timeline",
    "get_tags",
    "get_versions",
})

_PSQL_PATH = os.environ.get("HERMES_PSQL_PATH", "psql")
_GBRAIN_DB = os.environ.get("HERMES_GBRAIN_DB", "gbrain")
# On the VM hermes-user can't read gbrain DB directly; we use sudo -u postgres.
# Toggle via env in case a future deploy gives hermes-user a direct role.
_PSQL_USE_SUDO_POSTGRES = os.environ.get(
    "HERMES_PSQL_SUDO_POSTGRES", "1"
) in ("1", "true", "yes")


def _run_psql_query(sql: str, timeout_s: float = 8.0) -> List[str]:
    """Run a one-shot psql query, return non-empty stripped output lines."""
    cmd: List[str] = []
    if _PSQL_USE_SUDO_POSTGRES:
        cmd = ["sudo", "-u", "postgres", _PSQL_PATH, "-d", _GBRAIN_DB, "-At", "-c", sql]
    else:
        cmd = [_PSQL_PATH, "-d", _GBRAIN_DB, "-At", "-c", sql]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        logger.warning("block_e.mcp_audit: psql failed: %s", exc)
        return []
    if proc.returncode != 0:
        logger.warning(
            "block_e.mcp_audit: psql rc=%d stderr=%s",
            proc.returncode, (proc.stderr or "")[:300],
        )
        return []
    return [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]


def called_retrieval_tool(agent_name: str, lookback_seconds: int = 120) -> bool:
    """Return True iff the OAuth client made at least one retrieval-MCP
    call within the last ``lookback_seconds``. Default 120s safely covers
    typical turn durations without leaking across user sessions (hermes
    processes turns serially per user).
    """
    if not agent_name:
        return False
    sql = (
        "SELECT operation FROM mcp_request_log "
        f"WHERE agent_name = '{agent_name}' "
        f"AND created_at >= NOW() - INTERVAL '{int(lookback_seconds)} seconds'"
    )
    ops = _run_psql_query(sql)
    if not ops:
        return False
    for op in ops:
        # Some logs prefix with "tools/call:" — strip it.
        clean = op.split(":")[-1].strip()
        if clean in RETRIEVAL_OPS:
            return True
    return False


def list_calls(agent_name: str, lookback_seconds: int = 120) -> List[str]:
    """For debug/audit: list every operation the client called in the
    window. Returns operation names in chronological order.
    """
    if not agent_name:
        return []
    sql = (
        "SELECT operation FROM mcp_request_log "
        f"WHERE agent_name = '{agent_name}' "
        f"AND created_at >= NOW() - INTERVAL '{int(lookback_seconds)} seconds' "
        "ORDER BY created_at ASC"
    )
    return [op.split(":")[-1].strip() for op in _run_psql_query(sql)]
