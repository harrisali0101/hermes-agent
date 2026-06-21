"""RFC 8693 OAuth Token Exchange client for gbrain — per-subject token minting.

A trusted Hermes process (the WhatsApp gateway, the Slack bot, etc.) holds
ONE delegator OAuth client credential and uses it to mint short-lived
per-end-user access tokens via gbrain's `/token` endpoint with
`grant_type=urn:ietf:params:oauth:grant-type:token-exchange`. gbrain
resolves the end-user's RLS scope from its `subjects` table at every
verify, so a single delegator serves N humans at DB-enforced isolation.

This is the canonical "on-behalf-of" pattern: one process, many identities,
no god-mode credential, no confused-deputy anti-pattern.

Cache semantics
---------------
Exchanged access tokens are short-lived (~1h by default on gbrain) but
the exchange call itself costs a round-trip on every MCP invocation if
unmanaged. The `TokenCache` here caches each (subject_id → token) pair
in-process until 30 seconds before the gbrain-issued `expires_at`. A
hot path (multiple tool calls per WhatsApp turn) does ONE exchange per
subject per cache window, not one per tool call.

Cache is in-process only. Multiple Hermes worker processes will each
mint independently — that's fine: tokens are revocable individually
(gbrain `subjects remove` invalidates ALL outstanding tokens for that
subject regardless of which worker minted them).

Security posture
----------------
- The delegator's client_secret is the trust boundary. Anyone holding
  it can mint tokens for any subject in its allow-list. Treat it as
  you would a database password.
- This module makes NO assumption that subjects exist. If a subject
  hasn't been registered on gbrain via `gbrain auth subjects add`, the
  exchange call returns `invalid_grant` and the caller MUST handle it
  (typically: log + treat as "user not provisioned" → fall back to a
  bounce reply).
- The module never reads or writes subject metadata. Subject
  provisioning is the sysadmin's responsibility (or `subjects_sync.py`).
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional


SUBJECT_TOKEN_TYPE_SUBJECT_ID = "urn:gbrain:params:oauth:token-type:subject-id"
GRANT_TYPE_TOKEN_EXCHANGE = "urn:ietf:params:oauth:grant-type:token-exchange"


class TokenExchangeError(Exception):
    """Wraps gbrain's RFC 6749 / 8693 error envelope.

    `error` is the spec-defined code (e.g., `invalid_client`,
    `invalid_grant`). `description` is the server's human-readable
    message — used in logs, NEVER surfaced to end-users (it can leak
    delegator config). HTTP status is preserved for retry logic.
    """

    def __init__(self, status: int, error: str, description: str):
        super().__init__(f"{error}: {description} (HTTP {status})")
        self.status = status
        self.error = error
        self.description = description


@dataclass
class _CachedToken:
    access_token: str
    expires_at: float  # Unix seconds


class TokenCache:
    """Thread-safe per-subject cache of exchanged access tokens.

    Single-tenant: instantiate one per Hermes process. The delegator
    credentials are baked in at construction; callers reference
    subjects by opaque string ID.

    Concurrency: the lock is a single per-cache mutex, not per-subject.
    That means cold-cache turns for N different subjects serialize on
    the network round-trip. For the pilot (a few dozen WhatsApp users)
    this is the right trade — collapses thundering-herd to one exchange
    per fresh subject — but at higher scale (hundreds of concurrent
    subjects) replace with a per-subject lock or an asyncio queue. See
    docs/integrations/on-behalf-of.md "Wire-level flow" for the rationale.
    """

    def __init__(
        self,
        gbrain_url: str,
        client_id: str,
        client_secret: str,
        *,
        scope: Optional[str] = None,
        resource: Optional[str] = None,
        timeout: float = 10.0,
        safety_margin_seconds: int = 30,
    ) -> None:
        if not gbrain_url:
            raise ValueError("gbrain_url required")
        if not client_id or not client_secret:
            raise ValueError("client_id and client_secret required")
        self._url = gbrain_url.rstrip("/")
        self._client_id = client_id
        self._client_secret = client_secret
        self._scope = scope
        # RFC 8707 audience binding. When set, every exchange request
        # includes resource=<uri>; gbrain pins the issued token to that
        # audience (oauth_tokens.resource). Prevents cross-server token
        # replay if the same delegator client serves multiple upstreams.
        self._resource = resource
        self._timeout = timeout
        self._safety = safety_margin_seconds
        self._cache: Dict[str, _CachedToken] = {}
        self._lock = threading.Lock()

    def for_subject(self, subject_id: str) -> str:
        """Return a valid (cached or freshly-minted) access token for the subject.

        Idempotent and safe to call concurrently — the lock prevents a
        thundering-herd of exchange calls for the same subject when the
        cache is cold. The lock is per-cache (not per-subject) which
        means a single exchange call per request, not parallelized; for
        the WhatsApp-gateway pilot scale (<100 concurrent users) this is
        right. At higher scale switch to a per-subject lock.
        """
        if not subject_id:
            raise ValueError("subject_id required")
        now = time.time()
        with self._lock:
            cached = self._cache.get(subject_id)
            if cached and cached.expires_at > now + self._safety:
                return cached.access_token
            # Cold or expired — mint fresh. The exchange itself runs
            # OUTSIDE the lock would be cleaner for throughput, but
            # holding the lock here is a deliberate "no thundering herd"
            # — N parallel turns for the same fresh subject collapse
            # to one exchange call.
            token, expires_in = self._exchange(subject_id)
            self._cache[subject_id] = _CachedToken(
                access_token=token,
                expires_at=now + expires_in,
            )
            return token

    def invalidate(self, subject_id: str) -> None:
        """Drop a subject's cached token. Useful after a known-bad call
        (e.g. server returned 401 on a fresh-looking token — subject was
        revoked between mint and use)."""
        with self._lock:
            self._cache.pop(subject_id, None)

    def clear(self) -> None:
        """Drop every cached token. Useful in test teardown."""
        with self._lock:
            self._cache.clear()

    # ── exchange call ────────────────────────────────────────────────────

    def _exchange(self, subject_id: str) -> tuple[str, int]:
        body_params = {
            "grant_type": GRANT_TYPE_TOKEN_EXCHANGE,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "subject_token": subject_id,
            "subject_token_type": SUBJECT_TOKEN_TYPE_SUBJECT_ID,
        }
        if self._scope:
            body_params["scope"] = self._scope
        if self._resource:
            body_params["resource"] = self._resource
        body = urllib.parse.urlencode(body_params).encode("utf-8")
        req = urllib.request.Request(
            f"{self._url}/token",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = ""
            try:
                raw = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            try:
                envelope = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                envelope = {}
            raise TokenExchangeError(
                status=exc.code,
                error=str(envelope.get("error") or "http_error"),
                description=str(envelope.get("error_description") or raw[:300] or exc.reason),
            ) from exc
        except urllib.error.URLError as exc:
            raise TokenExchangeError(
                status=0,
                error="network_error",
                description=str(exc.reason),
            ) from exc

        access_token = payload.get("access_token")
        expires_in = payload.get("expires_in")
        if not isinstance(access_token, str) or not access_token:
            raise TokenExchangeError(
                status=200,
                error="invalid_response",
                description="gbrain returned no access_token",
            )
        if not isinstance(expires_in, (int, float)) or expires_in <= 0:
            # Conservative default — better short-cache than infinite.
            expires_in = 3600
        return access_token, int(expires_in)
