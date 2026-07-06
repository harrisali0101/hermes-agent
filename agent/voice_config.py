"""Per-subject + global-default voice preference for the WhatsApp voice-lane.

Voice-lane synthesis is Azure Speech (see :mod:`agent.tts_azure_speech`).
The active voice for a given reply is resolved as:

    per_subject[<wa_id>]  →  default_voice  →  AZURE_SPEECH_VOICE_ID env  →  hard fallback

Config file (JSON, atomic writes)::

    /home/hermes-user/.hermes/voice_config.json
    {
      "default_voice": "en-US-AvaMultilingualNeural",
      "per_subject":  { "923333717117": "en-US-AndrewMultilingualNeural" }
    }

Runtime users:

* ``resolve_voice(subject_id)`` — called by the gateway right before TTS.
* ``handle_voice_list / _get / _set`` — MCP tool handlers, imported by
  ``hermes_save_mcp`` and dispatched under names ``voice_list``,
  ``voice_get``, ``voice_set``.

RBAC:

* ``voice_list``, ``voice_get``  — anyone with a valid sender_id.
* ``voice_set``
    - ``target="self"`` (default) — always allowed.
    - ``target="default"`` — super_admin only (changes what NEW users get).
    - ``target=<wa_id>``       — super_admin only (unless target == sender).

Curated voice list — 8 entries with human labels. Rationale in
``azure/design/voice-round-trip.md``. To add a voice, append to
``_CURATED`` and redeploy. Non-curated Azure voice_ids are not accepted
by ``voice_set`` — keeps the shortlist honest.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_CONFIG_PATH_DEFAULT = "/home/hermes-user/.hermes/voice_config.json"

# Curated shortlist. Order matters for `voice_list` — top of list is the
# first choice users see. Each entry is (id, label, description).
_CURATED: List[Tuple[str, str, str]] = [
    (
        "en-US-AvaMultilingualNeural",
        "Ava",
        "US female, warm assistant tone — closest to Siri. HD Neural.",
    ),
    (
        "en-US-AndrewMultilingualNeural",
        "Andrew",
        "US male, warm/natural. HD Neural. Companion to Ava.",
    ),
    (
        "en-US-EmmaMultilingualNeural",
        "Emma",
        "US female, friendly and slightly younger sound. HD Neural.",
    ),
    (
        "en-US-BrianMultilingualNeural",
        "Brian",
        "US male, professional/business tone. HD Neural.",
    ),
    (
        "en-US-AriaNeural",
        "Aria",
        "US female, standard assistant tone (Cortana-style).",
    ),
    (
        "en-US-JennyMultilingualNeural",
        "Jenny",
        "US female, multilingual — the pilot default until 2026-07-06.",
    ),
    (
        "en-GB-SoniaNeural",
        "Sonia",
        "British female — for a UK accent.",
    ),
    (
        "en-GB-RyanNeural",
        "Ryan",
        "British male — for a UK accent.",
    ),
]

_VALID_IDS: frozenset = frozenset(v[0] for v in _CURATED)


def _default_config_path() -> str:
    return os.environ.get("HERMES_VOICE_CONFIG_PATH") or _CONFIG_PATH_DEFAULT


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Read the JSON config. Returns empty dict on missing/malformed file
    — callers must never crash the voice path over a bad config file.
    """
    path = path or _default_config_path()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_config(cfg: Dict[str, Any], path: Optional[str] = None) -> None:
    """Atomic write: temp file in the same dir, then rename."""
    path = path or _default_config_path()
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(parent),
        prefix=".voice_config.", suffix=".tmp", delete=False,
    ) as tmp:
        json.dump(cfg, tmp, indent=2, ensure_ascii=False, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def resolve_voice(
    subject_id: Optional[str],
    *,
    env_fallback: Optional[str] = None,
    hard_fallback: str = "en-US-JennyMultilingualNeural",
    path: Optional[str] = None,
) -> str:
    """Resolve the active voice_id for a subject.

    Order: per_subject → default_voice → env_fallback (typically
    ``AZURE_SPEECH_VOICE_ID``) → hard_fallback.
    """
    cfg = load_config(path)
    per = cfg.get("per_subject") or {}
    if isinstance(per, dict) and subject_id and per.get(subject_id):
        candidate = str(per.get(subject_id))
        if candidate:
            return candidate
    default_voice = cfg.get("default_voice")
    if isinstance(default_voice, str) and default_voice:
        return default_voice
    if env_fallback:
        return env_fallback
    return hard_fallback


def curated_voices() -> List[Dict[str, str]]:
    return [{"id": v[0], "label": v[1], "description": v[2]} for v in _CURATED]


def is_curated(voice_id: str) -> bool:
    return voice_id in _VALID_IDS


# ---------------------------------------------------------------------- MCP
# Handlers imported by hermes_save_mcp. Signature matches the pattern
# used for other tools there: return the JSON-RPC ``result`` shape (dict
# with ``content`` list, and optional ``isError``).

def _text_result(text: str, *, error: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if error:
        out["isError"] = True
    return out


def handle_voice_list(sender_id: str) -> Dict[str, Any]:
    voices = curated_voices()
    lines = ["Available voices (curated):"]
    for v in voices:
        lines.append(f"  - {v['label']}  ({v['id']}) — {v['description']}")
    lines.append("")
    lines.append(
        'To pick one for yourself: voice_set(voice_id="en-US-AvaMultilingualNeural") '
        '(or use the id from the list). To change the global default, '
        'super_admin only: voice_set(voice_id="…", target="default").'
    )
    return _text_result("\n".join(lines))


def handle_voice_get(sender_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
    target = str(args.get("target") or "self").strip() or "self"
    if target == "self":
        subject = sender_id
    elif target == "default":
        subject = None
    else:
        subject = target
    voice = resolve_voice(
        subject,
        env_fallback=os.environ.get("AZURE_SPEECH_VOICE_ID"),
    )
    label = next(
        (v["label"] for v in curated_voices() if v["id"] == voice),
        voice,
    )
    scope_desc = (
        "voice for you (self)" if target == "self"
        else "global default voice" if target == "default"
        else f"voice for subject {target}"
    )
    return _text_result(
        f"Active {scope_desc}: {label} ({voice}).\n"
        f"Change with voice_set(voice_id=\"<id>\""
        + (f", target=\"{target}\"" if target != "self" else "")
        + ")."
    )


def handle_voice_set(
    sender_id: str,
    role: Optional[str],
    args: Dict[str, Any],
) -> Dict[str, Any]:
    voice_id = str(args.get("voice_id") or "").strip()
    target = str(args.get("target") or "self").strip() or "self"

    if not voice_id:
        return _text_result(
            "voice_id required. Call voice_list to see options.", error=True,
        )
    if not is_curated(voice_id):
        return _text_result(
            f"voice_id '{voice_id}' is not in the curated shortlist. "
            "Call voice_list to see the accepted ids. To add a new voice, "
            "edit agent/voice_config.py._CURATED and redeploy.",
            error=True,
        )

    if target == "self":
        target_subject = sender_id
    elif target == "default":
        if role != "super_admin":
            return _text_result(
                "Changing the global default voice requires super_admin. "
                "Ask Harris or another super_admin to run this, or use "
                "target=\"self\" to change only your own voice.",
                error=True,
            )
        target_subject = None
    else:
        # target is a wa_id
        if target != sender_id and role != "super_admin":
            return _text_result(
                f"Changing another user's ({target}) voice requires "
                "super_admin. You can change your own voice with "
                "target=\"self\".",
                error=True,
            )
        target_subject = target

    cfg = load_config()
    if target_subject is None:
        cfg["default_voice"] = voice_id
    else:
        per = cfg.get("per_subject")
        if not isinstance(per, dict):
            per = {}
        per[target_subject] = voice_id
        cfg["per_subject"] = per
    save_config(cfg)

    label = next(
        (v[1] for v in _CURATED if v[0] == voice_id),
        voice_id,
    )
    scope_desc = (
        "your voice" if target == "self"
        else "the global default voice" if target == "default"
        else f"voice for {target}"
    )
    return _text_result(
        f"Set {scope_desc} to {label} ({voice_id}). "
        f"Takes effect on your next voice reply."
    )
