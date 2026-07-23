"""Real PSTN transport over Twilio Programmable Voice + Media Streams.

Call flow:
  1. `calls.create` (outbound) with inline TwiML `<Connect><Stream url="wss://..."/>`
     pointing at this runner's local Media Streams WebSocket server (exposed via
     ngrok/cloudflared — see README). `send_digits` (with `w` pauses) handles
     IVR navigation at answer time; dual-channel recording is enabled.
  2. Twilio connects to the WebSocket and streams inbound audio (8 kHz mu-law).
     Pipecat's `TwilioFrameSerializer` decodes to PCM 16 kHz; an energy-based
     segmenter groups it into complete utterances, delivered as the same
     `{"type": "audio", ...}` dicts the local mock produces — so this transport
     plugs straight under `AudioTransportAdapter` (STT/TTS/brain unchanged).
  3. Mid-call DTMF: Twilio does not support outbound DTMF over bidirectional
     Media Streams, and the `/Calls/{sid}/Play.json` REST resource does not
     exist (404 code 20404 — verified on a live call). The supported
     workaround is updating the call TwiML with `<Play digits>` followed by
     re-`<Connect><Stream>`; the stream drops for ~1-2s and reconnects here.
     Prefer `send_digits` at `calls.create` time for IVR navigation.

This transport is audio-only: wrap it with `AudioTransportAdapter`
(`--mode audio`); `send_text` raises by design.

NEVER point this at Vivo/IBM numbers outside the contracted test windows.
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import json
import os
import time
from typing import Any

RECEIVE_POLL_S = 0.1
STREAM_START_TIMEOUT_S = 60.0
TWILIO_SAMPLE_RATE = 8000
PIPELINE_SAMPLE_RATE = 16000
OUT_CHUNK_MS = 20

# Energy-based utterance segmentation (PCM s16le @16kHz)
VAD_FRAME_MS = 20
VAD_RMS_THRESHOLD = 500
VAD_MIN_SPEECH_MS = 240
VAD_SILENCE_END_MS = 900
VAD_MAX_UTTERANCE_MS = 30_000


class UtteranceSegmenter:
    """Groups a continuous PCM stream into complete utterances by energy VAD."""

    def __init__(
        self,
        sample_rate: int = PIPELINE_SAMPLE_RATE,
        rms_threshold: int = VAD_RMS_THRESHOLD,
        min_speech_ms: int = VAD_MIN_SPEECH_MS,
        silence_end_ms: int = VAD_SILENCE_END_MS,
    ):
        self.sample_rate = sample_rate
        self.rms_threshold = rms_threshold
        self.min_speech_frames = max(1, min_speech_ms // VAD_FRAME_MS)
        self.silence_end_frames = max(1, silence_end_ms // VAD_FRAME_MS)
        self.max_frames = VAD_MAX_UTTERANCE_MS // VAD_FRAME_MS
        self._frame_bytes = int(sample_rate * VAD_FRAME_MS / 1000) * 2
        self._pending = bytearray()
        self._current = bytearray()
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False

    def feed(self, pcm: bytes) -> list[bytes]:
        """Feed PCM; returns zero or more complete utterances."""
        self._pending.extend(pcm)
        utterances: list[bytes] = []
        while len(self._pending) >= self._frame_bytes:
            frame = bytes(self._pending[: self._frame_bytes])
            del self._pending[: self._frame_bytes]
            if self._process_frame(frame):
                utterances.append(bytes(self._current))
                self._reset()
        return utterances

    def flush(self) -> bytes | None:
        """Return the in-progress utterance (e.g. when the stream stops)."""
        if self._in_speech and self._current:
            utterance = bytes(self._current)
            self._reset()
            return utterance
        return None

    def _process_frame(self, frame: bytes) -> bool:
        loud = audioop.rms(frame, 2) >= self.rms_threshold
        if not self._in_speech:
            if loud:
                self._speech_frames += 1
                self._current.extend(frame)
                if self._speech_frames >= self.min_speech_frames:
                    self._in_speech = True
                    self._silence_frames = 0
            else:
                self._speech_frames = 0
                self._current.clear()
            return False
        self._current.extend(frame)
        self._silence_frames = 0 if loud else self._silence_frames + 1
        return (
            self._silence_frames >= self.silence_end_frames
            or len(self._current) // self._frame_bytes >= self.max_frames
        )

    def _reset(self) -> None:
        self._current.clear()
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False


class TwilioMediaStreamTransport:
    """Audio-level CallTransport over Twilio Media Streams.

    Produces/consumes the same message dicts as the local WS audio mode, so it
    is used behind `AudioTransportAdapter` (which does STT/TTS + recording).
    """

    REQUIRED_ENV = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER")

    def __init__(
        self,
        to_number: str,
        *,
        public_url: str | None = None,
        listen_host: str = "0.0.0.0",
        listen_port: int = 8990,
        send_digits: str | None = None,
        record: bool = True,
        client: Any | None = None,
    ):
        missing = [name for name in self.REQUIRED_ENV if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                f"TwilioTransport requires env vars: {', '.join(missing)}. "
                "Set them (see .env.example) or use ws:// target for the local mock."
            )
        self.to_number = to_number
        self.from_number = os.environ["TWILIO_FROM_NUMBER"]
        self.account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        self.public_url = public_url or os.environ.get("TWILIO_STREAM_PUBLIC_URL", "")
        if not self.public_url:
            raise RuntimeError(
                "TwilioTransport requires TWILIO_STREAM_PUBLIC_URL (public wss:// "
                "endpoint of the Media Streams server — start ngrok/cloudflared "
                "against the runner port first, see README)."
            )
        self.public_url = self.public_url.replace("https://", "wss://")
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.send_digits = send_digits
        self.record = record
        if client is None:
            from twilio.rest import Client

            client = Client(self.account_sid, os.environ["TWILIO_AUTH_TOKEN"])
        self.client = client

        self.call_sid: str | None = None
        self.stream_sid: str | None = None
        self._serializer = None
        self._ws = None
        self._server = None
        self._segmenter = UtteranceSegmenter()
        self._inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._started = asyncio.Event()
        self._closed = False
        self._ended = False  # far side sent Media Streams `stop`
        self._expect_reconnect = False

    @property
    def url(self) -> str:
        return f"tel:{self.to_number} via {self.public_url}"

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        import websockets

        self._server = await websockets.serve(
            self._handle_media_ws, self.listen_host, self.listen_port
        )
        twiml = (
            "<Response><Connect>"
            f'<Stream url="{self.public_url}"/>'
            "</Connect></Response>"
        )
        kwargs: dict[str, Any] = {
            "to": self.to_number,
            "from_": self.from_number,
            "twiml": twiml,
        }
        if self.send_digits:
            kwargs["send_digits"] = self.send_digits
        if self.record:
            kwargs["record"] = True
            kwargs["recording_channels"] = "dual"
        call = await asyncio.to_thread(self.client.calls.create, **kwargs)
        self.call_sid = call.sid
        try:
            await asyncio.wait_for(self._started.wait(), timeout=STREAM_START_TIMEOUT_S)
        except (TimeoutError, asyncio.TimeoutError):
            await self.hangup()
            raise RuntimeError(
                f"Twilio call {self.call_sid} created but the Media Stream never "
                f"connected within {STREAM_START_TIMEOUT_S:.0f}s — check that "
                f"{self.public_url} tunnels to {self.listen_host}:{self.listen_port}"
            ) from None

    async def hangup(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.call_sid:
            try:
                await asyncio.to_thread(
                    self.client.calls(self.call_sid).update, status="completed"
                )
            except Exception:  # noqa: BLE001 — call may already be finished
                pass
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # -- media stream server -------------------------------------------------

    async def _handle_media_ws(self, ws) -> None:
        from pipecat.frames.frames import InputAudioRawFrame, StartFrame
        from pipecat.serializers.twilio import TwilioFrameSerializer

        async for raw in ws:
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = event.get("event")
            if kind == "start":
                self.stream_sid = event["start"]["streamSid"]
                self._serializer = TwilioFrameSerializer(
                    stream_sid=self.stream_sid,
                    params=TwilioFrameSerializer.InputParams(
                        twilio_sample_rate=TWILIO_SAMPLE_RATE,
                        sample_rate=PIPELINE_SAMPLE_RATE,
                        # hangup is handled by this transport via the REST API
                        auto_hang_up=False,
                    ),
                )
                await self._serializer.setup(
                    StartFrame(
                        audio_in_sample_rate=PIPELINE_SAMPLE_RATE,
                        audio_out_sample_rate=PIPELINE_SAMPLE_RATE,
                    )
                )
                self._ws = ws
                self._started.set()
            elif kind == "media" and self._serializer is not None:
                frame = await self._serializer.deserialize(raw)
                if isinstance(frame, InputAudioRawFrame):
                    for utterance in self._segmenter.feed(frame.audio):
                        self._push_utterance(utterance)
            elif kind == "stop":
                tail = self._segmenter.flush()
                if tail:
                    self._push_utterance(tail)
                if self._expect_reconnect:
                    # Old stream closing after a DTMF TwiML update; a new
                    # <Connect><Stream> connection is on its way.
                    self._expect_reconnect = False
                    break
                self._ended = True
                self._inbox.put_nowait(
                    {"type": "event", "name": "call_ended", "reason": "completed"}
                )
                break

    def _push_utterance(self, pcm: bytes) -> None:
        self._inbox.put_nowait(
            {
                "type": "audio",
                "encoding": "pcm_s16le",
                "sample_rate": PIPELINE_SAMPLE_RATE,
                "channels": 1,
                "data": base64.b64encode(pcm).decode("ascii"),
                "speaker": "agent",
            }
        )

    # -- CallTransport interface ----------------------------------------------

    async def receive(self, timeout: float) -> dict[str, Any] | None:
        try:
            return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError):
            raise asyncio.TimeoutError from None

    async def send_event(self, name: str, **fields: Any) -> None:
        """No-op by design: PSTN has no side channel for protocol events.

        `AudioTransportAdapter.connect()` sends `set_mode audio` for the local
        mock's handshake; over Twilio the transport is audio-only, so the mode
        is implicit and events are silently dropped."""

    async def send_audio(self, pcm: bytes, sample_rate: int = PIPELINE_SAMPLE_RATE) -> None:
        """Stream PCM to the call in ~20ms Media Stream frames (paced).

        Late sends are swallowed (same contract as LocalWebSocketTransport):
        when the far side hangs up right after a terminal turn, the remaining
        audio is dropped and `receive()` reports the call end — a completed
        journey must not turn into a transport_error."""
        import websockets

        from pipecat.frames.frames import OutputAudioRawFrame

        if self._ws is None or self._serializer is None:
            if self._closed or self._ended:
                return  # call already over; drop the late utterance
            raise RuntimeError("media stream is not connected")
        # Trailing silence flushes the serializer's stream resampler (which
        # buffers ~100ms) so the end of the utterance is not swallowed.
        pcm = pcm + b"\x00" * int(sample_rate * 0.24) * 2
        chunk_bytes = int(sample_rate * OUT_CHUNK_MS / 1000) * 2
        started = time.monotonic()
        sent_ms = 0.0
        for i in range(0, len(pcm), chunk_bytes):
            frame = OutputAudioRawFrame(
                audio=pcm[i : i + chunk_bytes],
                sample_rate=sample_rate,
                num_channels=1,
            )
            payload = await self._serializer.serialize(frame)
            if payload:
                try:
                    await self._ws.send(payload)
                except websockets.ConnectionClosed:
                    return  # line dropped mid-utterance; receive() drains the end
            sent_ms += OUT_CHUNK_MS
            # Pace slightly faster than real time; Twilio buffers a little.
            ahead_s = sent_ms / 1000 * 0.8 - (time.monotonic() - started)
            if ahead_s > 0:
                await asyncio.sleep(ahead_s)

    async def send_dtmf(self, digits: str) -> None:
        """Mid-call DTMF via TwiML update: `<Play digits>` + re-`<Connect><Stream>`.

        Twilio has no native outbound DTMF on bidirectional Media Streams (see
        module docstring), so the call TwiML is replaced; the stream drops and
        reconnects here within ~1-2s. Prefer `send_digits` at create time.
        """
        if not self.call_sid:
            raise RuntimeError("no active call")
        allowed = set("0123456789*#w")
        if not digits or not set(digits) <= allowed:
            raise ValueError(f"invalid DTMF digits {digits!r} (allowed: 0-9 * # w)")
        twiml = (
            f'<Response><Play digits="{digits}"/>'
            f'<Connect><Stream url="{self.public_url}"/></Connect></Response>'
        )
        self._expect_reconnect = True
        self._started.clear()
        await asyncio.to_thread(self.client.calls(self.call_sid).update, twiml=twiml)
        if self._server is not None:
            try:
                await asyncio.wait_for(self._started.wait(), timeout=30.0)
            except (TimeoutError, asyncio.TimeoutError):
                raise RuntimeError(
                    "media stream did not reconnect after mid-call DTMF TwiML update"
                ) from None

    async def send_text(self, text: str) -> None:
        raise RuntimeError(
            "TwilioMediaStreamTransport is audio-only — run with --mode audio "
            "so AudioTransportAdapter converts tester text to speech"
        )
