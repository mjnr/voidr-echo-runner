"""Live call events — "robô falando com robô" em tempo real na UI.

`LivePublisher` taps the CallRunner timeline/transcript hooks (and the
AudioTransportAdapter utterances) and POSTs batched events to the service:

    POST {VOIDR_API_URL}/v1/echo/live/{executionId}/events
    body: {"shardIndex": N, "events": [{"seq", "tsMs", "type", "data"}]}

Contract (fixed — the service/UI are built against these exact fields):
  phase            {phase: dialing|ura|agent|ended}
  turn             {speaker: tester|agent|ura, text, turnIndex}
  turn_audio       {speaker, turnIndex, format: "wav", sampleRate, audioB64}
  dtmf_sent        {digits}
  state_transition {state, turn}
  emotional_state  {emotion, intensity, action|null}
  call_ended       {reason, status}

Fire-and-forget by design: a network failure NEVER blocks or kills the call.
Batches are retried a couple of times and then dropped; a simple circuit
breaker disables the publisher when the endpoint 404s (not deployed yet) or
stays unreachable. Turn TEXT goes through the massa deny-list before leaving
the process; live audio is NOT redacted (see README — disable it with
ECHO_LIVE_AUDIO=0 on calls that use real sensitive massas).
"""

from __future__ import annotations

import asyncio
import base64
import io
import sys
import time
import wave
from typing import Any, Callable

import httpx

FLUSH_INTERVAL_S = 0.3
BATCH_MAX_EVENTS = 5
BATCH_MAX_AUDIO = 1
POST_TIMEOUT_S = 5.0
BATCH_MAX_RETRIES = 2  # attempts per batch before it is dropped
BREAKER_MAX_FAILURES = 5  # consecutive failed batches before giving up


def pcm_to_wav_b64(pcm: bytes, sample_rate: int) -> str:
    """PCM s16le mono -> in-memory WAV -> base64 (one utterance)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class LivePublisher:
    """Queue + async flush task emitting the live contract to the service."""

    def __init__(
        self,
        base_url: str,
        execution_id: str,
        shard_index: int = 1,
        *,
        token: str | None = None,
        redact: Callable[[str], str] | None = None,
        audio_enabled: bool = True,
        client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
    ):
        self.url = f"{base_url.rstrip('/')}/v1/echo/live/{execution_id}/events"
        self.shard_index = shard_index
        self.audio_enabled = audio_enabled
        self._redact = redact or (lambda text: text)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = client or httpx.AsyncClient(timeout=POST_TIMEOUT_S, headers=headers)
        self._owns_client = client is None
        self._sync_client = sync_client or httpx.Client(
            timeout=POST_TIMEOUT_S, headers=headers
        )
        self._owns_sync_client = sync_client is None

        self._queue: list[dict[str, Any]] = []
        self._seq = 0
        self._t0_ms: int | None = None
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._stopping = False
        self.disabled = False
        self._consecutive_failures = 0

        # phase machine (dialing -> ura -> agent -> ended); ura is optional.
        self._phase: str | None = None
        # turn/turn_audio pairing: audio may arrive before (agent, STT first)
        # or after (tester, TTS after the transcript record) the turn event.
        self._last_turn_index: dict[str, int] = {}
        self._audio_attached: set[int] = set()
        self._pending_audio: dict[str, tuple[bytes, int]] = {}

    # ── emission core ─────────────────────────────────────────────────────────

    def _now_ts_ms(self) -> int:
        if self._t0_ms is None:
            return 0
        return max(0, int(time.time() * 1000) - self._t0_ms)

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Queue one contract event (sync, callable from CallRunner hooks)."""
        if self.disabled:
            return
        self._seq += 1
        self._queue.append(
            {"seq": self._seq, "tsMs": self._now_ts_ms(), "type": event_type, "data": data}
        )
        if len(self._queue) >= BATCH_MAX_EVENTS:
            self._wake.set()

    def _emit_phase(self, phase: str) -> None:
        if self._phase == phase or self._phase == "ended":
            return
        self._phase = phase
        self.emit("phase", {"phase": phase})

    # ── taps (CallRunner + AudioTransportAdapter) ────────────────────────────

    def on_transcript(self, entry: dict[str, Any]) -> None:
        """Hook for CallRunner._record_transcript — emits the `turn` event."""
        speaker = entry["speaker"]
        if speaker == "ura":
            self._emit_phase("ura")
        elif speaker == "agent":
            self._emit_phase("agent")
        turn_index = entry["index"]
        self.emit(
            "turn",
            {"speaker": speaker, "text": self._redact(entry["text"]), "turnIndex": turn_index},
        )
        self._last_turn_index[speaker] = turn_index
        pending = self._pending_audio.pop(speaker, None)
        if pending is not None:
            self._emit_turn_audio(speaker, turn_index, *pending)

    def on_timeline_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Hook for CallRunner._record_event — maps internal types to the contract."""
        if event_type == "dtmf_sent":
            self._emit_phase("ura")  # dial plan without a text prompt (PSTN)
            self.emit("dtmf_sent", {"digits": self._redact(str(data.get("digits", "")))})
        elif event_type == "state_transition":
            self.emit("state_transition", {"state": data.get("state"), "turn": data.get("turn")})
        elif event_type == "emotional_state":
            self.emit(
                "emotional_state",
                {
                    "emotion": data.get("emotion"),
                    "intensity": data.get("intensity"),
                    "action": data.get("action") or None,
                },
            )
        elif event_type == "call_ended":
            self._emit_phase("ended")
        # other internal types (connected, tester_turn, remote_event, error,
        # hard_cap_reached…) have no live mapping — turn events come from the
        # transcript hook, which carries the text and the turnIndex.

    def add_turn_audio(self, speaker: str, pcm: bytes, sample_rate: int) -> None:
        """Hook for AudioTransportAdapter — one utterance of PCM s16le mono.

        Emitted together with the matching `turn`: tester TTS arrives after
        its turn was recorded (emit now); agent STT audio arrives before the
        turn exists (held until on_transcript pairs it)."""
        if self.disabled or not self.audio_enabled:
            return
        turn_index = self._last_turn_index.get(speaker)
        if turn_index is not None and turn_index not in self._audio_attached:
            self._emit_turn_audio(speaker, turn_index, pcm, sample_rate)
        else:
            self._pending_audio[speaker] = (pcm, sample_rate)

    def _emit_turn_audio(
        self, speaker: str, turn_index: int, pcm: bytes, sample_rate: int
    ) -> None:
        if not self.audio_enabled:
            return
        self._audio_attached.add(turn_index)
        self.emit(
            "turn_audio",
            {
                "speaker": speaker,
                "turnIndex": turn_index,
                "format": "wav",
                "sampleRate": sample_rate,
                "audioB64": pcm_to_wav_b64(pcm, sample_rate),
            },
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Begin the call: t0 for tsMs, phase `dialing`, spawn the flush task."""
        self._t0_ms = int(time.time() * 1000)
        self._emit_phase("dialing")
        self._task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Flush what's left and stop the task (call inside the event loop)."""
        self._stopping = True
        self._wake.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except (TimeoutError, asyncio.TimeoutError):
                self._task.cancel()
            self._task = None
        if self._owns_client:
            await self._client.aclose()

    def finish_sync(self, reason: str, status: str) -> None:
        """Final `phase ended` + `call_ended {reason, status}` after evaluation.

        The event loop is gone by now (status comes from the post-call
        evaluator), so this last small batch goes over a sync client."""
        was_disabled = self.disabled
        self.disabled = False  # allow the terminal events even after a trip…
        self._emit_phase("ended")
        self.emit("call_ended", {"reason": reason, "status": status})
        events, self._queue = self._queue, []
        self.disabled = was_disabled
        if self.disabled or not events:
            return
        try:
            self._sync_client.post(
                self.url, json={"shardIndex": self.shard_index, "events": events}
            )
        except httpx.HTTPError as exc:
            self._warn(f"final live batch dropped: {type(exc).__name__}: {exc}")
        finally:
            if self._owns_sync_client:
                self._sync_client.close()

    # ── flush task ────────────────────────────────────────────────────────────

    def _next_batch(self) -> list[dict[str, Any]]:
        """Up to BATCH_MAX_EVENTS events, at most BATCH_MAX_AUDIO audio each."""
        batch: list[dict[str, Any]] = []
        audio = 0
        while self._queue and len(batch) < BATCH_MAX_EVENTS:
            if self._queue[0]["type"] == "turn_audio":
                if audio >= BATCH_MAX_AUDIO:
                    break
                audio += 1
            batch.append(self._queue.pop(0))
        return batch

    async def _flush_loop(self) -> None:
        while True:
            if not self._queue:
                if self._stopping:
                    return
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=FLUSH_INTERVAL_S)
                except (TimeoutError, asyncio.TimeoutError):
                    pass
                self._wake.clear()
                continue
            batch = self._next_batch()
            if batch:
                await self._post_batch(batch)
            if self.disabled:
                self._queue.clear()
                return

    async def _post_batch(self, events: list[dict[str, Any]]) -> None:
        payload = {"shardIndex": self.shard_index, "events": events}
        for attempt in range(1, BATCH_MAX_RETRIES + 1):
            try:
                resp = await self._client.post(self.url, json=payload)
            except httpx.HTTPError as exc:
                if attempt == BATCH_MAX_RETRIES:
                    self._register_failure(f"{type(exc).__name__}: {exc}")
                    return
                await asyncio.sleep(0.2 * attempt)
                continue
            if resp.status_code == 404:
                # endpoint not deployed — no point retrying anything else
                self.disabled = True
                self._warn("live endpoint returned 404 — live events disabled for this call")
                return
            if resp.status_code >= 400:
                if attempt == BATCH_MAX_RETRIES:
                    self._register_failure(f"HTTP {resp.status_code}: {resp.text[:120]}")
                    return
                await asyncio.sleep(0.2 * attempt)
                continue
            self._consecutive_failures = 0
            return

    def _register_failure(self, detail: str) -> None:
        self._consecutive_failures += 1
        self._warn(f"live batch dropped ({detail})")
        if self._consecutive_failures >= BREAKER_MAX_FAILURES:
            self.disabled = True
            self._warn(
                f"{BREAKER_MAX_FAILURES} consecutive live batch failures — "
                "live events disabled for this call"
            )

    def _warn(self, message: str) -> None:
        print(f"  live: {message}", file=sys.stderr)
