"""Unit tests for the DIH voice-lane TTS helper.

Covers the four contract points the design memo calls out:

1. Happy path — mocked HTTP 200 with body bytes returns those bytes.
2. HTTP 500 — raises :class:`TTSError` with a bounded body snippet.
3. Chunking — text > ``_AZURE_TTS_MAX_CHARS_PER_REQUEST`` is split at
   sentence boundaries; each chunk gets its own POST and the resulting
   bytes are concatenated in order.
4. Empty text — returns ``b""`` without an HTTP call.

Uses monkeypatch to swap ``httpx.Client`` for a fake transport. This
keeps the test hermetic — no real network, no dependency on an Azure
subscription — while still exercising the SSML envelope + header
contract the module builds.
"""

from __future__ import annotations

import pytest

from agent import tts_azure_speech
from agent.tts_azure_speech import (
    DEFAULT_VOICE_ID,
    TTSError,
    _AZURE_TTS_MAX_CHARS_PER_REQUEST,
    tts_bytes,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Duck-typed stand-in for httpx.Response."""

    def __init__(self, status_code: int, content: bytes = b"", text: str = ""):
        self.status_code = status_code
        self.content = content
        self.text = text or content.decode("utf-8", errors="replace")


class _FakeClient:
    """Duck-typed stand-in for httpx.Client used as a context manager.

    Records every ``post`` call (headers + ssml body) so tests can
    assert on the request shape.
    """

    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, *, headers, content):
        self.calls.append(
            {"url": url, "headers": dict(headers), "content": content}
        )
        if not self._responses:
            raise AssertionError(
                "FakeClient ran out of prepared responses — test setup is off"
            )
        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def _speech_env(monkeypatch):
    """Every test in this module gets deterministic Speech Services env."""
    monkeypatch.setenv("AZURE_SPEECH_KEY", "test-key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "swedencentral")
    monkeypatch.delenv("AZURE_SPEECH_VOICE_ID", raising=False)


def _install_fake_client(monkeypatch, responses: list[_FakeResponse]) -> _FakeClient:
    """Point tts_azure_speech.httpx.Client at a scripted fake client."""
    fake = _FakeClient(responses)

    class _ClientFactory:
        def __call__(self, *args, **kwargs):
            # The module builds a Client via ``httpx.Client(timeout=...)``.
            # We ignore args/kwargs and return the shared fake so the
            # test can inspect calls after tts_bytes returns.
            return fake

    monkeypatch.setattr(tts_azure_speech.httpx, "Client", _ClientFactory())
    # Timeout constructor is called with httpx.Timeout(...) — swap for a
    # no-op so it doesn't need the real httpx package to accept args.
    monkeypatch.setattr(tts_azure_speech.httpx, "Timeout", lambda *a, **k: None)
    return fake


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tts_bytes_happy_path_returns_response_bytes(monkeypatch):
    """HTTP 200 with body bytes returns those bytes verbatim."""
    audio_payload = b"OggS\x00\x02" + b"\x00" * 40  # fake OGG header prefix
    fake = _install_fake_client(monkeypatch, [_FakeResponse(200, audio_payload)])

    result = tts_bytes("Hello, this is a test.")

    assert result == audio_payload
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["url"] == (
        "https://swedencentral.tts.speech.microsoft.com/cognitiveservices/v1"
    )
    assert call["headers"]["Ocp-Apim-Subscription-Key"] == "test-key"
    assert call["headers"]["Content-Type"] == "application/ssml+xml"
    assert call["headers"]["X-Microsoft-OutputFormat"] == (
        "ogg-16khz-16bit-mono-opus"
    )
    ssml = call["content"].decode("utf-8")
    assert '<voice name="' + DEFAULT_VOICE_ID + '">' in ssml
    assert "Hello, this is a test." in ssml


def test_tts_bytes_http_500_raises_tts_error(monkeypatch):
    """Non-200 responses raise TTSError with a bounded body snippet."""
    _install_fake_client(
        monkeypatch,
        [_FakeResponse(500, b"", text="Internal Server Error: something exploded")],
    )

    with pytest.raises(TTSError) as excinfo:
        tts_bytes("Hello.")

    msg = str(excinfo.value)
    assert "HTTP 500" in msg
    assert "Internal Server Error" in msg


def test_tts_bytes_chunks_long_text_into_multiple_requests(monkeypatch):
    """Text > the per-request cap splits into N POSTs and the bytes concatenate."""
    # Build a text that exceeds the cap at sentence boundaries.
    # Each sentence is ~4500 chars; two sentences ~= 9002 chars total which
    # exceeds the 9000 cap and forces a two-chunk split.
    sentence_a = "A" * 4500 + "."
    sentence_b = "B" * 4500 + "."
    long_text = sentence_a + " " + sentence_b
    assert len(long_text) > _AZURE_TTS_MAX_CHARS_PER_REQUEST

    fake = _install_fake_client(
        monkeypatch,
        [
            _FakeResponse(200, b"CHUNK-A-BYTES"),
            _FakeResponse(200, b"CHUNK-B-BYTES"),
        ],
    )

    result = tts_bytes(long_text)

    # Two POSTs, one per chunk; results concatenated in order.
    assert len(fake.calls) == 2
    assert result == b"CHUNK-A-BYTES" + b"CHUNK-B-BYTES"

    # First chunk carries sentence A only; second carries sentence B only.
    first_ssml = fake.calls[0]["content"].decode("utf-8")
    second_ssml = fake.calls[1]["content"].decode("utf-8")
    assert "A" * 100 in first_ssml
    assert "B" * 100 not in first_ssml
    assert "B" * 100 in second_ssml
    assert "A" * 100 not in second_ssml


def test_tts_bytes_empty_text_returns_empty_bytes_without_http(monkeypatch):
    """Empty / whitespace-only text short-circuits before the HTTP client."""
    fake = _install_fake_client(monkeypatch, [])  # zero prepared responses

    assert tts_bytes("") == b""
    assert tts_bytes("   \n\t  ") == b""
    assert fake.calls == []  # no HTTP calls at all


def test_tts_bytes_missing_env_raises_tts_error(monkeypatch):
    """Missing AZURE_SPEECH_KEY / REGION surfaces a clear error."""
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    monkeypatch.delenv("AZURE_SPEECH_REGION", raising=False)

    with pytest.raises(TTSError) as excinfo:
        tts_bytes("Hello.")
    assert "AZURE_SPEECH_KEY" in str(excinfo.value)


def test_tts_bytes_unknown_provider_raises_tts_error():
    """Only the ``azure`` provider is implemented today."""
    with pytest.raises(TTSError) as excinfo:
        tts_bytes("Hello.", provider="elevenlabs")
    assert "elevenlabs" in str(excinfo.value)
