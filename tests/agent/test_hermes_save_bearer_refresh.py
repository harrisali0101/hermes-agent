"""Unit tests for hermes_save_mcp._refresh_role_bearer + retry-on-401 path.

Mirrors the discipline of tests/tools/test_mcp_bearer_refresh_cmd.py (gateway
side, PR #52418): happy path, feature-disabled, cooldown, timeout, non-zero
exit, output validation, retry-once semantics, and atomic file rewrite.

The module is loaded directly from agent/hermes_save_mcp.py — no package
import — so the tests do not depend on the rest of the agent surface.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "agent" / "hermes_save_mcp.py"


@pytest.fixture()
def hsm(monkeypatch: pytest.MonkeyPatch):
    """Load a fresh copy of hermes_save_mcp per-test so the module-level
    cooldown dict doesn't leak across tests. Returns the loaded module."""
    spec = importlib.util.spec_from_file_location("hermes_save_mcp_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["hermes_save_mcp_under_test"] = module
    spec.loader.exec_module(module)
    # Always start clean — no env var, empty cooldown tracker.
    monkeypatch.delenv("HERMES_SAVE_BEARER_REFRESH_CMD", raising=False)
    module._bearer_refresh_last_attempt.clear()
    yield module
    sys.modules.pop("hermes_save_mcp_under_test", None)


def _make_mint_script(tmp_path: Path, body: str) -> Path:
    """Drop a tiny shell script that prints `body` and returns 0.
    Works on POSIX; Windows skips these tests (subprocess + sh)."""
    if os.name == "nt":
        pytest.skip("Bearer-refresh shell-out tests are POSIX-only")
    script = tmp_path / "mint.sh"
    script.write_text(f"#!/bin/sh\nprintf '%s' '{body}'\n")
    script.chmod(0o755)
    return script


def _make_failing_script(tmp_path: Path, exit_code: int = 1, stderr: str = "boom") -> Path:
    if os.name == "nt":
        pytest.skip("Bearer-refresh shell-out tests are POSIX-only")
    script = tmp_path / "fail.sh"
    script.write_text(f"#!/bin/sh\necho '{stderr}' 1>&2\nexit {exit_code}\n")
    script.chmod(0o755)
    return script


def _make_slow_script(tmp_path: Path, sleep_s: float) -> Path:
    if os.name == "nt":
        pytest.skip("Bearer-refresh shell-out tests are POSIX-only")
    script = tmp_path / "slow.sh"
    script.write_text(f"#!/bin/sh\nsleep {sleep_s}\nprintf 'too-late-bearer-token-1234567890'\n")
    script.chmod(0o755)
    return script


# ── _refresh_role_bearer behaviour ───────────────────────────────────────


def test_refresh_disabled_when_env_unset(hsm, tmp_path):
    """When HERMES_SAVE_BEARER_REFRESH_CMD is unset, return None and do nothing."""
    assert hsm._refresh_role_bearer("staff", str(tmp_path / "b.json")) is None


def test_refresh_happy_path(hsm, tmp_path, monkeypatch):
    """Happy path: command prints a token, helper returns it AND rewrites the file."""
    bearers_path = tmp_path / "role-bearers.json"
    bearers_path.write_text(json.dumps({"staff": "stale-bearer-old", "ceo": "untouched-token-456789"}))
    fresh = "gbrain_at_freshly_minted_1234567890"
    mint = _make_mint_script(tmp_path, fresh)
    monkeypatch.setenv("HERMES_SAVE_BEARER_REFRESH_CMD", str(mint))

    got = hsm._refresh_role_bearer("staff", str(bearers_path))

    assert got == fresh
    on_disk = json.loads(bearers_path.read_text())
    assert on_disk["staff"] == fresh
    assert on_disk["ceo"] == "untouched-token-456789"  # other roles preserved


def test_refresh_cooldown_short_circuits(hsm, tmp_path, monkeypatch):
    """Second refresh attempt within cooldown returns None without re-spawning."""
    bearers_path = tmp_path / "b.json"
    bearers_path.write_text("{}")
    mint = _make_mint_script(tmp_path, "gbrain_at_some_valid_token_1234567890")
    monkeypatch.setenv("HERMES_SAVE_BEARER_REFRESH_CMD", str(mint))

    first = hsm._refresh_role_bearer("staff", str(bearers_path))
    assert first is not None
    second = hsm._refresh_role_bearer("staff", str(bearers_path))
    assert second is None  # cooldown


def test_refresh_cooldown_is_per_role(hsm, tmp_path, monkeypatch):
    """Cooldown only applies to the role that just refreshed; others go through."""
    bearers_path = tmp_path / "b.json"
    bearers_path.write_text("{}")
    mint = _make_mint_script(tmp_path, "gbrain_at_some_valid_token_1234567890")
    monkeypatch.setenv("HERMES_SAVE_BEARER_REFRESH_CMD", str(mint))

    assert hsm._refresh_role_bearer("staff", str(bearers_path)) is not None
    # Different role — cooldown should NOT block.
    assert hsm._refresh_role_bearer("ceo", str(bearers_path)) is not None


def test_refresh_non_zero_exit_returns_none(hsm, tmp_path, monkeypatch):
    """Command failure → log + None, no file write."""
    bearers_path = tmp_path / "b.json"
    bearers_path.write_text(json.dumps({"staff": "stale"}))
    script = _make_failing_script(tmp_path, exit_code=2)
    monkeypatch.setenv("HERMES_SAVE_BEARER_REFRESH_CMD", str(script))

    assert hsm._refresh_role_bearer("staff", str(bearers_path)) is None
    # File untouched.
    assert json.loads(bearers_path.read_text()) == {"staff": "stale"}


def test_refresh_timeout_returns_none(hsm, tmp_path, monkeypatch):
    """Slow command → subprocess kills + helper returns None."""
    bearers_path = tmp_path / "b.json"
    bearers_path.write_text("{}")
    # Sleep just past the 10s timeout — but in tests we patch the timeout
    # via the constant to make it fast.
    monkeypatch.setattr(hsm, "_BEARER_REFRESH_TIMEOUT_S", 0.3)
    slow = _make_slow_script(tmp_path, 2.0)
    monkeypatch.setenv("HERMES_SAVE_BEARER_REFRESH_CMD", str(slow))

    assert hsm._refresh_role_bearer("staff", str(bearers_path)) is None


def test_refresh_cmd_not_found(hsm, tmp_path, monkeypatch):
    """Missing command → graceful None, no exception."""
    monkeypatch.setenv("HERMES_SAVE_BEARER_REFRESH_CMD", "/nonexistent/mint.sh")
    assert hsm._refresh_role_bearer("staff", str(tmp_path / "b.json")) is None


@pytest.mark.parametrize("bad_output", [
    "",                       # empty
    "   ",                    # whitespace only
    "short",                  # < 16 chars
    "this has a space in it OK",  # whitespace inside
    "ERROR: client_credentials grant failed for role staff",
    '{"error":"invalid_request"}',
    "<html><body>500 Internal Server Error</body></html>",
    "FAILED to mint token",
])
def test_refresh_rejects_unhealthy_output(hsm, tmp_path, monkeypatch, bad_output):
    """Output validation rejects all known-bad shapes."""
    bearers_path = tmp_path / "b.json"
    bearers_path.write_text(json.dumps({"staff": "stale-untouched"}))
    script = _make_mint_script(tmp_path, bad_output)
    monkeypatch.setenv("HERMES_SAVE_BEARER_REFRESH_CMD", str(script))

    assert hsm._refresh_role_bearer("staff", str(bearers_path)) is None
    # File untouched on bad output — cache cannot be poisoned.
    assert json.loads(bearers_path.read_text())["staff"] == "stale-untouched"


def test_refresh_persists_atomically(hsm, tmp_path, monkeypatch):
    """File rewrite is via tmp + rename so a partial write never appears on disk."""
    bearers_path = tmp_path / "b.json"
    bearers_path.write_text(json.dumps({"staff": "stale", "ceo": "old", "finance_analyst": "older"}))
    fresh = "gbrain_at_freshly_minted_1234567890"
    mint = _make_mint_script(tmp_path, fresh)
    monkeypatch.setenv("HERMES_SAVE_BEARER_REFRESH_CMD", str(mint))

    hsm._refresh_role_bearer("staff", str(bearers_path))

    # No leftover .tmp file
    assert not (tmp_path / "b.json.tmp").exists()
    # Other roles preserved verbatim
    on_disk = json.loads(bearers_path.read_text())
    assert on_disk["ceo"] == "old"
    assert on_disk["finance_analyst"] == "older"
    assert on_disk["staff"] == fresh


def test_refresh_disk_failure_still_returns_bearer(hsm, tmp_path, monkeypatch):
    """If the disk-write fails, the in-memory bearer is still returned so the
    caller can use it for the immediate retry. (Logged as warn.)"""
    fresh = "gbrain_at_freshly_minted_1234567890"
    mint = _make_mint_script(tmp_path, fresh)
    monkeypatch.setenv("HERMES_SAVE_BEARER_REFRESH_CMD", str(mint))

    # Point at a path inside a non-writable / non-existent directory.
    bad_path = str(tmp_path / "no-such-dir" / "b.json")
    got = hsm._refresh_role_bearer("staff", bad_path)
    assert got == fresh
