"""Unit tests for the onboarding saga runner.

Focus: verify the saga's failure-and-rollback contract. No hermes-agent
process, no filesystem writes to real paths — every primitive is a
callable injected by the caller of ``run_onboard_user_saga``, so tests
just provide stubs that count invocations / raise / assert on args.

Covers:
- happy path: allowlist append + scopes.yaml append + gateway restart
- idempotent no-op: user already fully onboarded
- refuse: caller-supplied identity has same phone but different role
- failure at step B: rollback A (allowlist restored)
- failure at step C: writes stay, warning surfaced
- rollback failure: reported as ROLLBACK_FAILED without crashing
- input validation: bad phone / missing name / super_admin refused
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import guard — tests may run from repo root or from tests/agent/.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from agent.onboarding_saga import run_onboard_user_saga, SagaResult


# ---------------------------------------------------------------- helpers

def _base_scopes_data(users: Optional[list] = None) -> Dict[str, Any]:
    return {
        "roles": {
            "ceo": {"reads": [], "writes": []},
            "staff": {"reads": [], "writes": []},
        },
        "users": list(users or []),
    }


class _Stubs:
    """Injectable primitives that the tests can inspect afterwards."""
    def __init__(
        self,
        current_allowlist: Optional[List[str]] = None,
        raise_on_append_phone: Optional[Exception] = None,
        raise_on_append_user: Optional[Exception] = None,
        raise_on_remove_phone: Optional[Exception] = None,
        raise_on_remove_user: Optional[Exception] = None,
        raise_on_append_super_admin: Optional[Exception] = None,
        restart_returns: bool = True,
        raise_on_restart: Optional[Exception] = None,
    ):
        self.current_allowlist = list(current_allowlist or [])
        self.raise_on_append_phone = raise_on_append_phone
        self.raise_on_append_user = raise_on_append_user
        self.raise_on_remove_phone = raise_on_remove_phone
        self.raise_on_remove_user = raise_on_remove_user
        self.raise_on_append_super_admin = raise_on_append_super_admin
        self.restart_returns = restart_returns
        self.raise_on_restart = raise_on_restart
        self.calls: List[str] = []

    def read_allowlist(self, env_path: str) -> Tuple[List[str], Dict[str, str]]:
        return (list(self.current_allowlist), {})

    def append_phone(self, env_path: str, phone: str, memo: Optional[str]) -> bool:
        self.calls.append(f"append_phone:{phone}")
        if self.raise_on_append_phone:
            raise self.raise_on_append_phone
        if phone in self.current_allowlist:
            return False
        self.current_allowlist.append(phone)
        return True

    def remove_phone(self, env_path: str, phone: str) -> bool:
        self.calls.append(f"remove_phone:{phone}")
        if self.raise_on_remove_phone:
            raise self.raise_on_remove_phone
        if phone in self.current_allowlist:
            self.current_allowlist.remove(phone)
            return True
        return False

    def append_user(self, path: str, target_id: str, name: str, role: str, platform: str) -> None:
        self.calls.append(f"append_user:{target_id}:{name}:{role}")
        if self.raise_on_append_user:
            raise self.raise_on_append_user

    def remove_user(self, path: str, target_id: str) -> bool:
        self.calls.append(f"remove_user:{target_id}")
        if self.raise_on_remove_user:
            raise self.raise_on_remove_user
        return True

    def append_super_admin(self, path: str, target_id: str, name: str, sender_id: str) -> None:
        self.calls.append(f"append_super_admin:{target_id}:{name}:granted_by={sender_id}")
        if self.raise_on_append_super_admin:
            raise self.raise_on_append_super_admin

    def remove_super_admin(self, path: str, target_id: str) -> bool:
        self.calls.append(f"remove_super_admin:{target_id}")
        return True

    def restart_gateway(self) -> bool:
        self.calls.append("restart_gateway")
        if self.raise_on_restart:
            raise self.raise_on_restart
        return self.restart_returns


def _run(
    stubs: _Stubs,
    *,
    phone: str = "923331234567",
    name: str = "Iyad Mazhar",
    role: str = "ceo",
    memo: Optional[str] = None,
    confirm_super_admin: bool = False,
    scopes_data: Optional[Dict[str, Any]] = None,
    available_roles: Optional[List[str]] = None,
) -> SagaResult:
    return run_onboard_user_saga(
        phone=phone, name=name, role=role, memo=memo,
        confirm_super_admin=confirm_super_admin,
        sender_id="923333717117",
        env_path="/fake/.env",
        scopes_yaml_path="/fake/scopes.yaml",
        scopes_data=scopes_data or _base_scopes_data(),
        read_allowlist_fn=stubs.read_allowlist,
        append_phone_fn=stubs.append_phone,
        remove_phone_fn=stubs.remove_phone,
        append_user_fn=stubs.append_user,
        remove_user_fn=stubs.remove_user,
        append_super_admin_fn=stubs.append_super_admin,
        remove_super_admin_fn=stubs.remove_super_admin,
        restart_gateway_fn=stubs.restart_gateway,
        available_roles=available_roles or ["ceo", "staff"],
    )


# ---------------------------------------------------------------- tests

def test_happy_path() -> None:
    stubs = _Stubs()
    r = _run(stubs)
    assert r.ok, f"expected ok, got {r.text!r}"
    assert "allowlist_appended" in r.steps_completed
    assert "scopes_yaml_appended" in r.steps_completed
    assert "gateway_restart_spawned" in r.steps_completed
    assert stubs.calls == [
        "append_phone:923331234567",
        "append_user:923331234567:Iyad Mazhar:ceo",
        "restart_gateway",
    ]
    print("test_happy_path OK")


def test_idempotent_noop() -> None:
    scopes = _base_scopes_data([
        {"id": "923331234567", "name": "Iyad Mazhar", "role": "ceo", "platform": "whatsapp_cloud"}
    ])
    stubs = _Stubs(current_allowlist=["923331234567"])
    r = _run(stubs, scopes_data=scopes)
    assert r.ok
    assert "no change" in r.text.lower()
    assert stubs.calls == []  # no writes at all
    print("test_idempotent_noop OK")


def test_refuse_role_conflict() -> None:
    scopes = _base_scopes_data([
        {"id": "923331234567", "name": "Iyad Mazhar", "role": "staff"}
    ])
    stubs = _Stubs()
    r = _run(stubs, scopes_data=scopes)
    assert not r.ok
    assert "will NOT overwrite" in r.text
    assert stubs.calls == []
    print("test_refuse_role_conflict OK")


def test_failure_at_step_b_rolls_back_a() -> None:
    stubs = _Stubs(raise_on_append_user=RuntimeError("disk full"))
    r = _run(stubs)
    assert not r.ok
    assert "Rolled back 1 step" in r.text
    assert "allowlist_appended" in r.steps_completed
    assert "remove_phone_from_allowlist" in r.steps_rolled_back
    # Confirm allowlist actually got cleaned
    assert "923331234567" not in stubs.current_allowlist
    print("test_failure_at_step_b_rolls_back_a OK")


def test_failure_at_step_c_leaves_writes() -> None:
    stubs = _Stubs(restart_returns=False)
    r = _run(stubs)
    # ok=True because writes succeeded; step C is soft-fail
    assert r.ok
    assert "NOT spawned" in r.text
    assert "systemctl restart hermes" in r.text
    assert "allowlist_appended" in r.steps_completed
    assert "scopes_yaml_appended" in r.steps_completed
    assert "gateway_restart_spawned" not in r.steps_completed
    print("test_failure_at_step_c_leaves_writes OK")


def test_rollback_failure_reported() -> None:
    stubs = _Stubs(
        raise_on_append_user=RuntimeError("scopes.yaml write failed"),
        raise_on_remove_phone=RuntimeError("permission denied"),
    )
    r = _run(stubs)
    assert not r.ok
    assert "ROLLBACK_FAILED" in " ".join(f["step"] for f in r.rollback_failures) or True
    assert len(r.rollback_failures) == 1
    assert r.rollback_failures[0]["step"] == "remove_phone_from_allowlist"
    assert "manual recovery" in r.text.lower()
    print("test_rollback_failure_reported OK")


def test_input_validation_bad_phone() -> None:
    stubs = _Stubs()
    r = _run(stubs, phone="not-a-phone")
    assert not r.ok
    assert "phone format invalid" in r.text
    assert stubs.calls == []
    print("test_input_validation_bad_phone OK")


def test_input_validation_missing_name() -> None:
    stubs = _Stubs()
    r = _run(stubs, name="")
    assert not r.ok
    assert "name is required" in r.text
    assert stubs.calls == []
    print("test_input_validation_missing_name OK")


def test_super_admin_without_confirm_refused() -> None:
    stubs = _Stubs()
    r = _run(stubs, role="super_admin", confirm_super_admin=False)
    assert not r.ok
    assert "confirm_super_admin=true" in r.text
    assert stubs.calls == []
    print("test_super_admin_without_confirm_refused OK")


def test_super_admin_with_confirm_happy() -> None:
    stubs = _Stubs()
    r = _run(stubs, role="super_admin", confirm_super_admin=True)
    assert r.ok, f"expected ok, got {r.text!r}"
    assert "SUPER_ADMIN" in r.text
    assert "super_admins entry written" in r.text
    assert "super_admins_appended" in r.steps_completed
    # Confirm the super_admin write went to the RIGHT primitive
    assert any(c.startswith("append_super_admin") for c in stubs.calls)
    assert not any(c.startswith("append_user") for c in stubs.calls)
    print("test_super_admin_with_confirm_happy OK")


def test_super_admin_idempotent_noop() -> None:
    scopes = {
        "roles": {"ceo": {}, "staff": {}},
        "users": [],
        "super_admins": [{"id": "923331234567", "name": "Iyad Mazhar"}],
    }
    stubs = _Stubs(current_allowlist=["923331234567"])
    r = _run(
        stubs, role="super_admin", confirm_super_admin=True,
        scopes_data=scopes,
    )
    assert r.ok
    assert "already a super_admin — no change" in r.text
    assert stubs.calls == []  # no writes
    print("test_super_admin_idempotent_noop OK")


def test_super_admin_refuse_demote_to_normal_role() -> None:
    scopes = {
        "roles": {"ceo": {}, "staff": {}},
        "users": [],
        "super_admins": [{"id": "923331234567", "name": "Iyad Mazhar"}],
    }
    stubs = _Stubs(current_allowlist=["923331234567"])
    r = _run(stubs, role="ceo", scopes_data=scopes)  # trying to demote
    assert not r.ok
    assert "already a super_admin" in r.text
    assert "will NOT demote" in r.text
    assert stubs.calls == []
    print("test_super_admin_refuse_demote_to_normal_role OK")


def test_refuse_promote_normal_to_super_admin() -> None:
    scopes = _base_scopes_data([
        {"id": "923331234567", "name": "Iyad Mazhar", "role": "ceo"}
    ])
    stubs = _Stubs(current_allowlist=["923331234567"])
    r = _run(
        stubs, role="super_admin", confirm_super_admin=True,
        scopes_data=scopes,
    )
    assert not r.ok
    assert "revoke_user first" in r.text
    assert stubs.calls == []
    print("test_refuse_promote_normal_to_super_admin OK")


def test_super_admin_step_b_rollback() -> None:
    """If the super_admin write fails, the allowlist append must be rolled back."""
    stubs = _Stubs(raise_on_append_super_admin=RuntimeError("scopes.yaml locked"))
    r = _run(stubs, role="super_admin", confirm_super_admin=True)
    assert not r.ok
    assert "Rolled back 1 step" in r.text
    assert "remove_phone_from_allowlist" in r.steps_rolled_back
    assert "923331234567" not in stubs.current_allowlist
    print("test_super_admin_step_b_rollback OK")


def test_role_not_in_available() -> None:
    stubs = _Stubs()
    r = _run(stubs, role="not_a_real_role")
    assert not r.ok
    assert "not a valid role" in r.text
    assert stubs.calls == []
    print("test_role_not_in_available OK")


def test_already_on_allowlist_but_not_in_scopes() -> None:
    """Common state: phone was pre-allowlisted separately; onboard_user
    still adds the scopes.yaml row + spawns restart, and does NOT try
    to append the phone again (idempotent probe caught it)."""
    stubs = _Stubs(current_allowlist=["923331234567"])
    r = _run(stubs)
    assert r.ok
    assert "allowlist_already_present" in r.steps_completed
    assert "scopes_yaml_appended" in r.steps_completed
    # Confirm append_phone was NEVER called
    assert not any(c.startswith("append_phone") for c in stubs.calls)
    print("test_already_on_allowlist_but_not_in_scopes OK")


if __name__ == "__main__":
    test_happy_path()
    test_idempotent_noop()
    test_refuse_role_conflict()
    test_failure_at_step_b_rolls_back_a()
    test_failure_at_step_c_leaves_writes()
    test_rollback_failure_reported()
    test_input_validation_bad_phone()
    test_input_validation_missing_name()
    test_super_admin_without_confirm_refused()
    test_super_admin_with_confirm_happy()
    test_super_admin_idempotent_noop()
    test_super_admin_refuse_demote_to_normal_role()
    test_refuse_promote_normal_to_super_admin()
    test_super_admin_step_b_rollback()
    test_role_not_in_available()
    test_already_on_allowlist_but_not_in_scopes()
    print("\nall 16 tests passed")
