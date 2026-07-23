"""Pluggable call transports.

`LocalWebSocketTransport` speaks the vivo-autopilot-mock JSON protocol (text
and audio modes). `tel:` targets route to `TwilioMediaStreamTransport`
(twilio_transport.py): real PSTN calls via Twilio Media Streams, audio-only,
used behind `AudioTransportAdapter` (--mode audio).
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import websockets


class CallTransport(Protocol):
    async def connect(self) -> None: ...

    async def send_text(self, text: str) -> None: ...

    async def send_dtmf(self, digits: str) -> None: ...

    async def hangup(self) -> None: ...

    async def receive(self, timeout: float) -> dict[str, Any] | None:
        """Next protocol message, or None when the call/connection is over."""
        ...


class LocalWebSocketTransport:
    """Talks to vivo-autopilot-mock (or any target speaking the same JSON
    protocol) over a local WebSocket."""

    def __init__(self, url: str):
        self.url = url
        self._ws: websockets.ClientConnection | None = None

    async def connect(self) -> None:
        self._ws = await websockets.connect(self.url)

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            return  # call already torn down
        try:
            await self._ws.send(json.dumps(payload, ensure_ascii=False))
        except websockets.ConnectionClosed:
            pass  # call already ended on the far side; receive() will drain it

    async def send_text(self, text: str) -> None:
        await self._send({"type": "text", "text": text})

    async def send_dtmf(self, digits: str) -> None:
        await self._send({"type": "dtmf", "digits": digits})

    async def send_event(self, name: str, **fields: Any) -> None:
        await self._send({"type": "event", "name": name, **fields})

    async def send_audio(self, pcm: bytes, sample_rate: int = 16000) -> None:
        import base64

        await self._send(
            {
                "type": "audio",
                "encoding": "pcm_s16le",
                "sample_rate": sample_rate,
                "channels": 1,
                "data": base64.b64encode(pcm).decode("ascii"),
            }
        )

    async def hangup(self) -> None:
        await self._send({"type": "event", "name": "hangup"})
        await self.close()

    async def receive(self, timeout: float) -> dict[str, Any] | None:
        import asyncio

        if self._ws is None:
            return None
        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
        except websockets.ConnectionClosed:
            return None
        return json.loads(raw)

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


def build_transport(target: str, send_digits: str | None = None) -> CallTransport:
    if target.startswith(("ws://", "wss://")):
        return LocalWebSocketTransport(target)
    if target.startswith("tel:") or target.startswith("+"):
        from .twilio_transport import TwilioMediaStreamTransport

        number = target.removeprefix("tel:")
        return TwilioMediaStreamTransport(number, send_digits=send_digits)
    raise ValueError(
        f"unsupported target {target!r}: use ws://host:port/ws (local mock) "
        "or tel:+<E164> (Twilio, requires --mode audio)"
    )
