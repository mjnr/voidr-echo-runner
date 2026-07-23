"""Pluggable call transports.

v0: `LocalWebSocketTransport` speaks the vivo-autopilot-mock JSON protocol.
`TwilioTransport` is a structured stub gated behind TWILIO_* env vars; it will
carry the real PSTN path (Media Streams + calls.create with sendDigits).
"""

from __future__ import annotations

import json
import os
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


class TwilioTransport:
    """PSTN transport stub (phase with real Twilio credentials).

    TODO(echo/twilio): implement using the pattern validated in
    ARCHITECTURE.md section 3.2:
      1. twilio.rest.Client(...).calls.create(
             to=dial_plan.to, from_=TWILIO_FROM_NUMBER,
             send_digits="ww<code>ww<ani>#", record=True,
             twiml=<Connect><Stream url=wss://runner/media>)
      2. Serve the Media Streams WebSocket and bridge 8kHz mu-law frames into
         the Pipecat pipeline via pipecat's TwilioFrameSerializer (DTMF frames
         included).
      3. Prefer mid-call DTMF via POST /Calls/{sid}/Play.json with SendDigits
         when the IVR prompts in separate steps.
    """

    REQUIRED_ENV = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER")

    def __init__(self, to_number: str):
        missing = [name for name in self.REQUIRED_ENV if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                "TwilioTransport requires credentials that are not set: "
                + ", ".join(missing)
                + ". Use --target ws://... (LocalWebSocketTransport) for offline runs."
            )
        raise NotImplementedError(
            "TwilioTransport is a structured stub: credentials detected, but the "
            "Media Streams bridge is not implemented yet (see TODO in transport.py)."
        )


def build_transport(target: str) -> CallTransport:
    if target.startswith(("ws://", "wss://")):
        return LocalWebSocketTransport(target)
    if target.startswith("tel:") or target.startswith("+"):
        return TwilioTransport(target)  # type: ignore[return-value]
    raise ValueError(
        f"unsupported target {target!r}: use ws://host:port/ws (local mock) "
        "or tel:+<E164> (Twilio stub)"
    )
