"""One-shot WhatsApp Cloud onboarding — saga runner with compensating rollbacks.

Consolidates the historical 5-step onboarding sequence (allowlist → reload →
wait-for-first-message → approve → mint bearer) into a single super_admin
call. Rationale:

Under WhatsApp Cloud API, ``wa_id`` equals the E.164 phone (digits only).
The "wait for first message to capture the lid" step from the Baileys era
is no longer needed — we already know the identity from the phone alone.

Design pattern: saga with compensating actions (Temporal-style).

  * Every write step registers a rollback thunk BEFORE it fires.
  * On failure at step N, rollbacks run for steps N-1 → 1 in reverse order.
  * Rollbacks are idempotent — a partial cleanup is safe to re-attempt.
  * If a rollback ITSELF fails, that's logged as ``ROLLBACK_FAILED`` with
    recovery instructions but does NOT crash the saga — the caller gets a
    partial-state error message and manual-recovery pointer instead of a
    silent half-onboarded user.

The saga is deliberately synchronous and short (2 writes + 1 detached
restart). Async / distributed patterns (Temporal, Cadence) are overkill
for a workflow this small.

Ships 2026-07-06 as follow-up to voice-lane + voice_config to consolidate
the DIH pilot onboarding UX. See ``azure/config/rules/ONBOARDING.md`` for
the persona-side contract.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PHONE_RE = re.compile(r"^\d{9,15}$")


@dataclass
class SagaResult:
    """Outcome of a saga run — success or a structured failure report."""
    ok: bool
    text: str
    steps_completed: List[str] = field(default_factory=list)
    steps_rolled_back: List[str] = field(default_factory=list)
    rollback_failures: List[Dict[str, str]] = field(default_factory=list)


def _log(level: str, msg: str, **kwargs: Any) -> None:
    fn = getattr(logger, level, logger.info)
    if kwargs:
        pairs = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
        fn(f"[onboarding_saga] {msg} {pairs}")
    else:
        fn(f"[onboarding_saga] {msg}")


def run_onboard_user_saga(
    *,
    phone: str,
    name: str,
    role: str,
    memo: Optional[str],
    sender_id: str,
    env_path: str,
    scopes_yaml_path: str,
    scopes_data: Dict[str, Any],
    # Injected primitives — passed in rather than imported to keep this
    # module unit-testable without a running hermes_save_mcp environment.
    read_allowlist_fn: Callable[[str], Tuple[List[str], Dict[str, str]]],
    append_phone_fn: Callable[[str, str, Optional[str]], bool],
    remove_phone_fn: Callable[[str, str], bool],
    append_user_fn: Callable[[str, str, str, str, str], None],
    remove_user_fn: Callable[[str, str], bool],
    append_super_admin_fn: Callable[[str, str, str, str], None],
    remove_super_admin_fn: Callable[[str, str], bool],
    restart_gateway_fn: Callable[[], bool],
    confirm_super_admin: bool = False,
    platform: str = "whatsapp_cloud",
    available_roles: Optional[List[str]] = None,
) -> SagaResult:
    """Run the atomic onboarding saga. Never raises — returns SagaResult.

    On success the tuple ``(allowlist appended, scopes.yaml appended,
    gateway restart spawned)`` all succeeded. On any failure everything
    that DID succeed gets rolled back before we return.
    """
    # ---------------------------------------------------------------- input validation
    try:
        phone_clean = str(phone or "").strip().lstrip("+").replace(" ", "")
        name_clean = str(name or "").strip()
        role_clean = str(role or "").strip()
    except Exception as exc:
        _log("error", "input normalization failed", error=str(exc))
        return SagaResult(
            ok=False,
            text=f"Failed to normalize inputs: {exc}",
        )

    if not phone_clean or not _PHONE_RE.match(phone_clean):
        return SagaResult(
            ok=False,
            text=(
                f"phone format invalid: '{phone_clean or phone}'. Expected "
                "digits only, 9-15 chars, no '+' (e.g., 923333717117)."
            ),
        )
    if not name_clean:
        return SagaResult(ok=False, text="name is required.")
    if not role_clean:
        return SagaResult(ok=False, text="role is required.")

    is_super_admin_grant = role_clean == "super_admin"

    if is_super_admin_grant and not confirm_super_admin:
        return SagaResult(
            ok=False,
            text=(
                "Granting super_admin is high-privilege — a super_admin reads "
                "every scope, runs on/off-boarding, and can mint other "
                "super_admins. Re-issue the call with confirm_super_admin=true "
                "to proceed."
            ),
        )

    # For non-super_admin roles, ensure the role is one of the configured
    # scopes.yaml roles. super_admin is always accepted (it's not a
    # roles.<role> entry — it's a top-level identity).
    if not is_super_admin_grant and available_roles and role_clean not in available_roles:
        return SagaResult(
            ok=False,
            text=(
                f"role '{role_clean}' is not a valid role for this pilot. "
                f"Available: {', '.join(available_roles)} (or 'super_admin' "
                "with confirm_super_admin=true)."
            ),
        )

    # ---------------------------------------------------------------- idempotency probe
    already_on_allowlist = False
    already_in_scopes = False
    try:
        current_phones, _memos = read_allowlist_fn(env_path)
        already_on_allowlist = phone_clean in current_phones
    except Exception as exc:
        _log("error", "allowlist read failed pre-saga", error=str(exc))
        return SagaResult(
            ok=False,
            text=f"Could not read allowlist ({exc}). Aborting before any writes.",
        )

    # Check super_admins block first: if phone_clean is already a
    # super_admin AND caller is requesting super_admin, that's an
    # idempotent no-op. If the phone is in super_admins but caller wants
    # a normal role, refuse (would need to demote via a separate tool).
    super_admins = (scopes_data.get("super_admins") or [])
    for e in super_admins:
        if str(e.get("id", "")).strip() == phone_clean:
            existing_name = str(e.get("name", "")).strip()
            if is_super_admin_grant:
                if existing_name == name_clean:
                    _log(
                        "info", "onboard idempotent no-op — already super_admin",
                        sender=sender_id, phone=phone_clean,
                    )
                    return SagaResult(
                        ok=True,
                        text=(
                            f"✅ {name_clean} ({phone_clean}) is already a "
                            f"super_admin — no change.\n\n"
                            + ("(Also on the allowlist ✓)" if already_on_allowlist
                               else "⚠️ super_admin entry exists but phone is NOT "
                                    "on the gateway allowlist. That is unusual — "
                                    "run onboard_user again to fix.")
                        ),
                        steps_completed=["idempotent_noop"],
                    )
                return SagaResult(
                    ok=False,
                    text=(
                        f"{phone_clean} is already a super_admin under name "
                        f"{existing_name!r}. onboard_user will NOT rename. "
                        "Edit scopes.yaml directly if the name needs to change."
                    ),
                )
            return SagaResult(
                ok=False,
                text=(
                    f"{phone_clean} is already a super_admin. onboard_user "
                    f"will NOT demote them to '{role_clean}'. Use revoke_user "
                    "first (with the super_admin caveat) to demote."
                ),
            )

    users = (scopes_data.get("users") or [])
    for u in users:
        if str(u.get("id", "")).strip() == phone_clean:
            already_in_scopes = True
            existing_role = str(u.get("role", "")).strip()
            existing_name = str(u.get("name", "")).strip()
            if is_super_admin_grant:
                # Promoting a normal user to super_admin — the current
                # design is to refuse and force explicit revoke + re-add,
                # so audit trails are unambiguous.
                return SagaResult(
                    ok=False,
                    text=(
                        f"{phone_clean} exists as a normal user "
                        f"({existing_name!r} / role={existing_role}). To "
                        "promote to super_admin, revoke_user first, then "
                        "re-run onboard_user with role='super_admin' + "
                        "confirm_super_admin=true."
                    ),
                )
            if existing_role == role_clean and existing_name == name_clean:
                _log(
                    "info", "onboard idempotent no-op",
                    sender=sender_id, phone=phone_clean, role=role_clean,
                )
                return SagaResult(
                    ok=True,
                    text=(
                        f"✅ {name_clean} ({phone_clean}) is already fully "
                        f"onboarded as {role_clean} — no change.\n\n"
                        + ("(Also on the allowlist ✓)" if already_on_allowlist
                           else "⚠️ scopes.yaml entry exists but phone is NOT "
                                "on the gateway allowlist. Run "
                                "onboard_user again to fix.")
                    ),
                    steps_completed=["idempotent_noop"],
                )
            # Same id, different role/name — refuse rather than silently mutate.
            return SagaResult(
                ok=False,
                text=(
                    f"{phone_clean} is already in scopes.yaml as "
                    f"{existing_name!r} / role={existing_role}. "
                    f"onboard_user will NOT overwrite. If you want to change "
                    f"the role, use revoke_user first, then onboard_user again."
                ),
            )

    _log(
        "info", "onboard saga start",
        sender=sender_id, phone=phone_clean, role=role_clean,
        memo=memo or "",
    )

    # ---------------------------------------------------------------- saga execution
    steps_completed: List[str] = []
    rollbacks: List[Tuple[str, Callable[[], None]]] = []

    # STEP A — append to allowlist
    if already_on_allowlist:
        # Nothing to do at step A but no rollback either. Log for audit.
        steps_completed.append("allowlist_already_present")
        _log("info", "step A skipped — phone already on allowlist", phone=phone_clean)
    else:
        try:
            appended = append_phone_fn(env_path, phone_clean, memo)
            if not appended:
                # Concurrent write raced us to it; treat as skipped.
                steps_completed.append("allowlist_raced_present")
                _log(
                    "warn", "step A no-op — phone appeared during saga",
                    phone=phone_clean,
                )
            else:
                steps_completed.append("allowlist_appended")
                rollbacks.append((
                    "remove_phone_from_allowlist",
                    lambda: (remove_phone_fn(env_path, phone_clean), None)[1],
                ))
                _log("info", "step A OK — phone appended", phone=phone_clean)
        except Exception as exc:
            _log(
                "error", "step A failed", phone=phone_clean, error=str(exc),
            )
            return SagaResult(
                ok=False,
                text=(
                    f"Failed to add {phone_clean} to the gateway allowlist: {exc}. "
                    "No changes made. Check .env write permissions."
                ),
                steps_completed=steps_completed,
            )

    # STEP B — write to scopes.yaml (super_admins block OR users block
    # depending on the role).
    #
    # If already_in_scopes was True we would have returned earlier — so at
    # this point the scope.yaml write always fires.
    try:
        if is_super_admin_grant:
            append_super_admin_fn(scopes_yaml_path, phone_clean, name_clean, sender_id)
            steps_completed.append("super_admins_appended")
            rollbacks.append((
                "remove_super_admin_from_scopes_yaml",
                lambda: (remove_super_admin_fn(scopes_yaml_path, phone_clean), None)[1],
            ))
            _log(
                "info", "step B OK — super_admin appended to scopes.yaml",
                phone=phone_clean, granted_by=sender_id,
            )
        else:
            append_user_fn(scopes_yaml_path, phone_clean, name_clean, role_clean, platform)
            steps_completed.append("scopes_yaml_appended")
            rollbacks.append((
                "remove_user_from_scopes_yaml",
                lambda: (remove_user_fn(scopes_yaml_path, phone_clean), None)[1],
            ))
            _log("info", "step B OK — user appended to scopes.yaml", phone=phone_clean)
    except Exception as exc:
        _log(
            "error", "step B failed — rolling back A",
            phone=phone_clean, error=str(exc),
        )
        rolled_back, rollback_failures = _run_rollbacks(rollbacks, sender_id, phone_clean)
        return SagaResult(
            ok=False,
            text=(
                f"Failed to add {phone_clean} to scopes.yaml: {exc}. "
                f"Rolled back {len(rolled_back)} step(s). "
                + (
                    f"⚠️ Rollback of {len(rollback_failures)} step(s) also failed — "
                    f"manual recovery needed. See gateway.log."
                    if rollback_failures else "State is clean."
                )
            ),
            steps_completed=steps_completed,
            steps_rolled_back=rolled_back,
            rollback_failures=rollback_failures,
        )

    # STEP C — spawn detached gateway restart
    #
    # This one is intentionally "soft-fail": if the restart-spawn fails we
    # keep the writes (they'll take effect on the NEXT restart anyway) and
    # tell the caller to run reload_gateway manually. Rolling back the two
    # writes just to force a re-do would be strictly worse UX.
    restart_spawned = False
    try:
        restart_spawned = bool(restart_gateway_fn())
        if restart_spawned:
            steps_completed.append("gateway_restart_spawned")
            _log("info", "step C OK — restart spawned", phone=phone_clean)
        else:
            _log(
                "warn", "step C soft-fail — restart_gateway_fn returned False",
                phone=phone_clean,
            )
    except Exception as exc:
        _log(
            "error", "step C failed — leaving writes in place",
            phone=phone_clean, error=str(exc),
        )

    # ---------------------------------------------------------------- audit trail
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _log(
        "info", "onboard saga complete",
        event="ONBOARD_SUPER_ADMIN_SUCCESS" if is_super_admin_grant else "ONBOARD_USER_SUCCESS",
        sender=sender_id,
        phone=phone_clean,
        name=name_clean,
        role=role_clean,
        platform=platform,
        restart_spawned=restart_spawned,
        ts=ts,
    )

    # ---------------------------------------------------------------- return text
    if restart_spawned:
        tail = (
            "\n\n♻️ Gateway restart spawned — the new user becomes active in "
            "~15-20s. Their FIRST message will trigger name/profile backfill "
            "automatically. No further steps required."
        )
    else:
        tail = (
            "\n\n⚠️ Allowlist + scopes.yaml written OK, but the automatic "
            "gateway restart could not be spawned. State is correct but "
            "the new user will remain rejected until the gateway restarts. "
            "Ask a super_admin with SSH access to run "
            "`sudo systemctl restart hermes.service` on the VM."
        )
    role_desc = "SUPER_ADMIN" if is_super_admin_grant else role_clean
    scopes_line = (
        "✓ super_admins entry written" if is_super_admin_grant
        else "✓ user row written"
    )
    return SagaResult(
        ok=True,
        text=(
            f"✅ Onboarded {name_clean} ({phone_clean}) as {role_desc}.\n\n"
            f"- allowlist: {'✓ added' if 'allowlist_appended' in steps_completed else '✓ already present'}\n"
            f"- scopes.yaml: {scopes_line}\n"
            f"- gateway restart: {'✓ spawned' if restart_spawned else '⚠️ NOT spawned'}"
            + tail
        ),
        steps_completed=steps_completed,
    )


def _run_rollbacks(
    rollbacks: List[Tuple[str, Callable[[], None]]],
    sender_id: str,
    phone: str,
) -> Tuple[List[str], List[Dict[str, str]]]:
    """Execute compensations in reverse. Each rollback is itself wrapped
    in try/except so a single failure doesn't abort the whole unwind.
    Returns (steps successfully rolled back, list of failures)."""
    rolled_back: List[str] = []
    failures: List[Dict[str, str]] = []
    for step_name, fn in reversed(rollbacks):
        try:
            fn()
            rolled_back.append(step_name)
            _log(
                "info", "rollback OK",
                step=step_name, sender=sender_id, phone=phone,
            )
        except Exception as exc:
            failures.append({"step": step_name, "error": str(exc)})
            _log(
                "error", "ROLLBACK_FAILED",
                step=step_name, sender=sender_id, phone=phone,
                error=str(exc),
            )
    return rolled_back, failures
