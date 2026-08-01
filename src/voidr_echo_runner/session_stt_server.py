"""Small governed STT boundary for browser Session voice notes.

The Chrome extension never receives Deepgram credentials. It sends bounded
PCM16/16 kHz segments to voidr-service; the service authenticates here with a
runtime-projected shared secret. Transcript or audio bytes are never logged.
"""

from __future__ import annotations

import asyncio
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from typing import Any

from .voice_gateway import MAX_AUDIO_BYTES, VoiceGatewayAudioEngine, resolve_voice_config


def _shared_secret() -> str:
    value = os.environ.get("SESSION_STT_SHARED_SECRET", "").strip()
    if len(value) < 32:
        raise RuntimeError("SESSION_STT_SHARED_SECRET must contain at least 32 characters")
    return value


def authorized(header: str | None, expected: str) -> bool:
    if not header or not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(header.removeprefix("Bearer "), expected)


async def transcribe_pcm(pcm: bytes) -> dict[str, Any]:
    engine = VoiceGatewayAudioEngine(resolve_voice_config(), voice_id="session-notes")
    try:
        text = await engine.transcribe(pcm)
    finally:
        await engine.aclose()
    return {
        "text": text,
        "language": "pt-BR",
        "durationMs": round(len(pcm) / (16_000 * 2) * 1000),
        "provider": "deepgram",
        "model": "nova-2",
    }


class SessionSttHandler(BaseHTTPRequestHandler):
    server_version = "VoidrSessionSTT/1"

    def log_message(self, format: str, *args: object) -> None:
        # Request paths and user-controlled metadata do not belong in logs.
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        if self.path != "/healthz":
            self._json(404, {"error": "not_found"})
            return
        self._json(200, {"status": "ok", "service": "voidr-session-stt"})

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        if self.path != "/v1/transcribe":
            self._json(404, {"error": "not_found"})
            return
        try:
            expected = _shared_secret()
        except RuntimeError:
            self._json(503, {"error": "not_configured"})
            return
        if not authorized(self.headers.get("Authorization"), expected):
            self._json(401, {"error": "unauthorized"})
            return
        if self.headers.get("Content-Type") != "audio/pcm;rate=16000;channels=1":
            self._json(415, {"error": "unsupported_media_type"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length < 320 or length > MAX_AUDIO_BYTES:
            self._json(413, {"error": "audio_size_out_of_bounds"})
            return
        pcm = self.rfile.read(length)
        if len(pcm) != length or len(pcm) % 2:
            self._json(400, {"error": "invalid_pcm"})
            return
        try:
            result = asyncio.run(transcribe_pcm(pcm))
        except Exception:  # provider details and transcript never leave this boundary
            self._json(502, {"error": "transcription_failed"})
            return
        self._json(200, result)


def serve_session_stt(host: str, port: int) -> int:
    _shared_secret()
    server = ThreadingHTTPServer((host, port), SessionSttHandler)
    print(f"voidr-session-stt listening on {host}:{port}")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
