"""
Azure Speech Services TTS helper — DIH voice-lane

Gateway-facing helper that synthesises audio bytes for the WhatsApp
Cloud voice-lane feature (design memo:
``azure/design/voice-round-trip.md`` in hermes-personal-agent).

This module deliberately sits ALONGSIDE the upstream pluggable TTS
provider system (:mod:`agent.tts_provider`, :mod:`agent.tts_registry`,
:mod:`tools.tts_tool`) rather than plugging into it. Reasons:

1. The gateway's voice-lane selector runs *outside* the agent's tool
   loop — the reply text has already been composed by the model, and
   we need synchronous audio bytes to hand to
   :meth:`WhatsAppCloudAdapter._upload_media`. Going through the
   TTS tool dispatcher would require synthesising the round-trip as
   an in-line tool call, which does not fit the composed-reply
   send() path.
2. The DIH pilot targets a single voice/provider combination (Azure
   Neural, ``en-US-JennyMultilingualNeural``). The rich provider
   picker in ``tools/tts_tool.py`` earns its complexity when the user
   is choosing between backends per call — not helpful here.
3. Keeping this module standalone means upstream changes to the
   plugin ABC don't reach into the voice-lane path.

If the pilot upgrades to Track C (Azure STT + ElevenLabs TTS per
per-subject preference), the abstraction to introduce lives IN THIS
FILE — swap ``provider="azure"`` for a dispatch table. Do NOT
refactor into ``agent/tts_provider.py`` — that path collides with
the upstream ABC.

Env-var contract
================
The following env vars must be exported to the hermes process (via
``/etc/hermes/fetch-secrets.sh`` on the VM). See the
"AZURE_SPEECH_* env vars" block that fetch-secrets.sh needs to
add — keep in sync with this docstring:

``AZURE_SPEECH_KEY``
    Speech Services subscription key. On the DIH nonprod
    subscription this is the SAME value as
    ``hermes-foundry-api-key`` — the underlying resource
    ``dih-foundry-nonprod`` is an ``AIServices`` multi-service
    account that bundles Speech Services under the same key.
    (Verified 2026-07-05 via ``az cognitiveservices account show``
    — bundled ``Speech Services Text to Speech (Neural)`` endpoint
    at ``https://swedencentral.tts.speech.microsoft.com``.)

``AZURE_SPEECH_REGION``
    Speech Services region. For DIH nonprod this is
    ``swedencentral`` (matches the Foundry deployment region).

``AZURE_SPEECH_VOICE_ID``
    Optional. Default voice id. Falls back to
    ``en-US-JennyMultilingualNeural`` when unset.

fetch-secrets.sh block (append to
``azure/scripts/fetch-secrets.sh`` when deploying)::

    # ---- Azure Speech Services (voice-lane TTS) --------------------------
    # Reuses hermes-foundry-api-key (multi-service AIServices account
    # dih-foundry-nonprod bundles Speech Services under the same key).
    export AZURE_SPEECH_KEY="$AZURE_OPENAI_API_KEY"
    export AZURE_SPEECH_REGION="swedencentral"
    export AZURE_SPEECH_VOICE_ID="en-US-JennyMultilingualNeural"

Do not commit that block to fetch-secrets.sh yet — voice-lane is
staged for review, not deployed.

Response contract
=================
:func:`tts_bytes` returns raw audio bytes. On the happy path Azure
returns OGG Opus at 16 kHz — the format WhatsApp Cloud renders as a
native voice bubble without a ffmpeg re-encode. On unsupported
formats the caller re-encodes via ffmpeg (see
:meth:`WhatsAppCloudAdapter._convert_to_opus`).

Failures raise :class:`TTSError`. The gateway catches, logs, and
falls back to sending the text version — the reply is never
dropped.

Chunking
========
Azure Neural TTS caps each request at 10,000 chars. For longer
texts we split at sentence boundaries (`. ! ?`) and concatenate the
audio blobs. Concatenating OGG containers naively works because
WhatsApp/ffmpeg tolerate multi-page OGG streams; for cleaner audio
a future revision can decode/reencode via ffmpeg. The pilot's
voice replies target <150 words so chunking is a defensive rather
than common-path concern.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Iterable, Optional

try:
    import httpx  # type: ignore[import]

    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover — httpx is a hard dep of hermes
    _HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DEFAULT_VOICE_ID = "en-US-JennyMultilingualNeural"
DEFAULT_PROVIDER = "azure"

# Azure Neural TTS hard limit per request (SSML including markup). We stay
# well under the documented 10k ceiling to leave headroom for the
# ``<speak>`` envelope and voice tag we add per chunk.
_AZURE_TTS_MAX_CHARS_PER_REQUEST = 9000

# Output format request header value. Matches WhatsApp Cloud's native
# voice-bubble format (16 kHz OGG Opus) so the gateway does not need
# ffmpeg re-encode for the happy path.
_AZURE_TTS_OUTPUT_FORMAT = "ogg-16khz-16bit-mono-opus"

# Sentence-boundary split. Keeps the terminator glued to the preceding
# sentence so downstream synthesis prosody stays natural.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Chars considered structural markup that must be escaped inside SSML.
_SSML_ESCAPES = (
    ("&", "&amp;"),
    ("<", "&lt;"),
    (">", "&gt;"),
    ('"', "&quot;"),
    ("'", "&apos;"),
)


class TTSError(RuntimeError):
    """Raised when TTS synthesis fails. The gateway catches and falls
    back to the text send path — the reply is never dropped."""


def tts_bytes(
    text: str,
    voice: str = DEFAULT_VOICE_ID,
    provider: str = DEFAULT_PROVIDER,
) -> bytes:
    """Synthesize ``text`` to audio bytes.

    Empty / whitespace-only ``text`` returns ``b""`` without an HTTP
    call — cheap short-circuit for cases where the persona sends only
    the ``[VOICE_REPLY]`` marker with no body (the gateway strips the
    marker before this function is called, so a body of just the
    marker becomes empty text).

    ``voice`` is an Azure Neural voice id. Overridable via the
    ``AZURE_SPEECH_VOICE_ID`` env var, whose value wins over the
    argument default (``DEFAULT_VOICE_ID``) but NOT over a
    non-default argument (caller intent overrides env default).

    ``provider`` is a forward-compat hook. Only ``"azure"`` is
    implemented today; anything else raises. Track C (ElevenLabs)
    lands here, not in a separate file.
    """
    if not text or not text.strip():
        return b""

    if provider != "azure":
        raise TTSError(
            f"unknown tts provider: {provider!r} "
            "(only 'azure' is implemented; see module docstring for the "
            "Track C upgrade path)"
        )

    if not _HTTPX_AVAILABLE:  # pragma: no cover — hermes always ships httpx
        raise TTSError("httpx is not installed; tts_bytes cannot run")

    # Argument default falls back to env if the caller took the default.
    if voice == DEFAULT_VOICE_ID:
        voice = os.environ.get("AZURE_SPEECH_VOICE_ID", DEFAULT_VOICE_ID) or DEFAULT_VOICE_ID

    key = os.environ.get("AZURE_SPEECH_KEY", "").strip()
    region = os.environ.get("AZURE_SPEECH_REGION", "").strip()
    if not key or not region:
        raise TTSError(
            "AZURE_SPEECH_KEY / AZURE_SPEECH_REGION not exported to the "
            "hermes process — see agent/tts_azure_speech.py docstring for "
            "the fetch-secrets.sh block that wires them from KV"
        )

    chunks = list(_split_for_chunking(text, _AZURE_TTS_MAX_CHARS_PER_REQUEST))
    if not chunks:  # defensive — _split_for_chunking preserves non-empty input
        return b""

    endpoint = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": _AZURE_TTS_OUTPUT_FORMAT,
        "User-Agent": "hermes-voice-lane/1.0",
    }

    audio_parts: list[bytes] = []
    with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        for chunk in chunks:
            ssml = _build_ssml(chunk, voice=voice)
            try:
                resp = client.post(endpoint, headers=headers, content=ssml.encode("utf-8"))
            except httpx.HTTPError as exc:
                raise TTSError(f"azure speech HTTP request failed: {exc}") from exc

            if resp.status_code != 200:
                # Azure surfaces error details in the body — bounded to
                # keep the exception message readable in logs.
                snippet = resp.text[:300] if resp.text else "<empty>"
                raise TTSError(
                    f"azure speech returned HTTP {resp.status_code}: {snippet}"
                )

            if not resp.content:
                raise TTSError("azure speech returned HTTP 200 with empty body")

            audio_parts.append(resp.content)

    return b"".join(audio_parts)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _split_for_chunking(text: str, max_chars: int) -> Iterable[str]:
    """Yield chunks each <= ``max_chars`` characters, splitting on
    sentence boundaries when possible.

    Falls back to hard character splits when a single sentence exceeds
    the cap (rare — model replies are short and Azure Neural is generous
    at 10k). Preserves original text — join(chunks) reconstructs the
    input modulo whitespace between sentences.
    """
    text = text.strip()
    if not text:
        return

    if len(text) <= max_chars:
        yield text
        return

    sentences = _SENTENCE_SPLIT_RE.split(text)
    buf = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(sentence) > max_chars:
            # Emit any pending buffer, then hard-split the oversized
            # sentence into ``max_chars`` slices.
            if buf:
                yield buf
                buf = ""
            for start in range(0, len(sentence), max_chars):
                yield sentence[start : start + max_chars]
            continue
        # Would appending overflow? Emit buffer and start a new one.
        candidate = f"{buf} {sentence}".strip() if buf else sentence
        if len(candidate) > max_chars:
            if buf:
                yield buf
            buf = sentence
        else:
            buf = candidate
    if buf:
        yield buf


def _escape_ssml(text: str) -> str:
    """Escape text for embedding in an SSML ``<voice>`` body."""
    for src, dst in _SSML_ESCAPES:
        text = text.replace(src, dst)
    return text


def _build_ssml(text: str, *, voice: str) -> str:
    """Build the minimal SSML envelope Azure Speech expects.

    Uses ``xml:lang="en-US"`` at the ``<speak>`` level — the voice tag's
    own language wins for actual synthesis when the voice is
    multilingual (e.g. ``en-US-JennyMultilingualNeural`` picks the
    speaker's language from context per Azure's docs).
    """
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xml:lang="en-US">'
        f'<voice name="{voice}">{_escape_ssml(text)}</voice>'
        "</speak>"
    )


# Optional voice id used by the WhatsApp Cloud gateway when it wants the
# module-level default without having to import DEFAULT_VOICE_ID
# explicitly.
def default_voice() -> str:
    return os.environ.get("AZURE_SPEECH_VOICE_ID", DEFAULT_VOICE_ID) or DEFAULT_VOICE_ID
