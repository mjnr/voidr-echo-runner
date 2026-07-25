"""Audio conversation mode — real Pipecat pipelines.

STT: DeepgramSTTService (streaming websocket, language=pt-BR).
TTS: ElevenLabsHttpTTSService (eleven_flash_v2_5, PCM 16 kHz).

The conversation over the local WebSocket protocol is turn-based (one complete
utterance per `audio` message), so each turn runs a short-lived real Pipecat
pipeline: frames are queued into `Pipeline([service, collector])` driven by a
`PipelineWorker` + `WorkerRunner`. `AudioTransportAdapter` wraps any text-level
transport so `CallRunner` (and the serve-execution mode) stay unchanged.

This module imports pipecat at module level and must only be imported when
audio mode is requested (the CLI does exactly that).
"""

from __future__ import annotations

import asyncio
import base64
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService
from pipecat.transcriptions.language import Language
from pipecat.workers.runner import WorkerRunner

SAMPLE_RATE = 16000
STT_TIMEOUT_S = 25.0
STT_EXTRA_DRAIN_S = 1.0
STT_CONNECT_GRACE_S = 2.5  # Deepgram ws must be up before audio is queued
STT_TRAILING_SILENCE_S = 0.8  # helps endpointing close the utterance
AUDIO_CHUNK_BYTES = 8000  # 0.25s at 16kHz s16le mono

DEEPGRAM_MODEL = "nova-2"
ELEVENLABS_MODEL = "eleven_flash_v2_5"


@dataclass(frozen=True)
class AudioServices:
    stt_provider: str
    tts_provider: str


def resolve_audio_services() -> AudioServices:
    """Validate env-var gated providers, failing with actionable guidance."""
    if not os.environ.get("DEEPGRAM_API_KEY"):
        raise RuntimeError(
            "Audio mode requires DEEPGRAM_API_KEY (STT, pt-BR streaming). "
            "None is set — run with --mode text for offline execution."
        )
    if os.environ.get("ELEVENLABS_API_KEY"):
        return AudioServices(stt_provider="deepgram", tts_provider="elevenlabs")
    if os.environ.get("AZURE_SPEECH_KEY"):
        raise RuntimeError(
            "AZURE_SPEECH_KEY detected but the Azure TTS wiring is not implemented "
            "yet — set ELEVENLABS_API_KEY to use audio mode today."
        )
    raise RuntimeError(
        "Audio mode requires a TTS key: set ELEVENLABS_API_KEY (persona voices). "
        "None is set — run with --mode text for offline execution."
    )


class _Collector(FrameProcessor):
    """Pipeline sink that captures transcripts / TTS audio / errors."""

    def __init__(self):
        super().__init__()
        self.transcripts: list[str] = []
        self.audio = bytearray()
        self.errors: list[str] = []
        self.got_transcript = asyncio.Event()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            if frame.text.strip():
                self.transcripts.append(frame.text.strip())
                self.got_transcript.set()
        elif isinstance(frame, TTSAudioRawFrame):
            self.audio.extend(frame.audio)
        elif isinstance(frame, ErrorFrame):
            self.errors.append(str(frame.error))
            self.got_transcript.set()
        await self.push_frame(frame, direction)


class PipecatAudioEngine:
    """Per-call STT/TTS engine backed by real Pipecat service pipelines."""

    def __init__(self, voice_id: str, sample_rate: int = SAMPLE_RATE):
        resolve_audio_services()
        self.voice_id = voice_id
        self.sample_rate = sample_rate
        self._deepgram_key = os.environ["DEEPGRAM_API_KEY"]
        self._elevenlabs_key = os.environ["ELEVENLABS_API_KEY"]
        self._aiohttp_session: aiohttp.ClientSession | None = None

    async def _session(self) -> aiohttp.ClientSession:
        if self._aiohttp_session is None:
            self._aiohttp_session = aiohttp.ClientSession()
        return self._aiohttp_session

    async def aclose(self) -> None:
        if self._aiohttp_session is not None:
            await self._aiohttp_session.close()
            self._aiohttp_session = None

    async def synthesize(self, text: str) -> bytes:
        """Text -> PCM s16le mono @16kHz through an ElevenLabs Pipecat pipeline."""
        tts = ElevenLabsHttpTTSService(
            api_key=self._elevenlabs_key,
            aiohttp_session=await self._session(),
            sample_rate=self.sample_rate,
            settings=ElevenLabsHttpTTSService.Settings(
                voice=self.voice_id,
                model=ELEVENLABS_MODEL,
                language=Language.PT,
            ),
        )
        collector = _Collector()
        await self._run_pipeline([tts, collector], [TTSSpeakFrame(text)])
        if collector.errors:
            raise RuntimeError(f"ElevenLabs TTS error: {'; '.join(collector.errors)}")
        if not collector.audio:
            raise RuntimeError(
                f"ElevenLabs TTS returned no audio for voice {self.voice_id!r}"
            )
        return bytes(collector.audio)

    async def transcribe(self, pcm: bytes) -> str:
        """PCM s16le mono @16kHz -> text through a Deepgram Pipecat pipeline."""
        stt = DeepgramSTTService(
            api_key=self._deepgram_key,
            sample_rate=self.sample_rate,
            settings=DeepgramSTTService.Settings(
                model=DEEPGRAM_MODEL,
                language=Language.PT_BR,
                smart_format=True,
                punctuate=True,
                interim_results=True,
            ),
        )
        collector = _Collector()
        duration_s = len(pcm) / 2 / self.sample_rate
        pcm = pcm + b"\x00" * int(STT_TRAILING_SILENCE_S * self.sample_rate) * 2
        frames: list[Any] = [
            InputAudioRawFrame(
                audio=pcm[i : i + AUDIO_CHUNK_BYTES],
                sample_rate=self.sample_rate,
                num_channels=1,
            )
            for i in range(0, len(pcm), AUDIO_CHUNK_BYTES)
        ]
        await self._run_pipeline(
            [stt, collector],
            frames,
            wait_transcript=collector,
            # Audio is pushed faster than real time; give Deepgram time to chew
            # through it before forcing a finalize, or transcripts get truncated.
            finalize_delay_s=min(6.0, 0.5 + duration_s * 0.5),
        )
        if collector.errors:
            raise RuntimeError(f"Deepgram STT error: {'; '.join(collector.errors)}")
        return " ".join(collector.transcripts).strip()

    async def _run_pipeline(
        self,
        processors: list,
        frames: list,
        wait_transcript: _Collector | None = None,
        finalize_delay_s: float = 0.0,
    ) -> None:
        worker = PipelineWorker(
            Pipeline(processors),
            params=PipelineParams(
                audio_in_sample_rate=self.sample_rate,
                audio_out_sample_rate=self.sample_rate,
            ),
            idle_timeout_secs=None,
            enable_turn_tracking=False,
            enable_rtvi=False,
            check_dangling_tasks=False,
        )
        runner = WorkerRunner(handle_sigint=False)
        run_task = asyncio.create_task(runner.run(worker))
        try:
            if wait_transcript is not None:
                # Give the Deepgram websocket time to connect: audio frames
                # pushed before that are silently dropped by the service.
                await asyncio.sleep(STT_CONNECT_GRACE_S)
            await worker.queue_frames(frames)
            if wait_transcript is not None:
                await asyncio.sleep(finalize_delay_s)
                # Flush whatever is still buffered server-side.
                await worker.queue_frames([VADUserStoppedSpeakingFrame()])
                await asyncio.wait_for(
                    wait_transcript.got_transcript.wait(), timeout=STT_TIMEOUT_S
                )
                # Drain until no new final segment arrives within the window.
                seen = -1
                while seen != len(wait_transcript.transcripts):
                    seen = len(wait_transcript.transcripts)
                    await asyncio.sleep(STT_EXTRA_DRAIN_S)
            await worker.stop_when_done()
            await asyncio.wait_for(run_task, timeout=30.0)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            await worker.cancel()
            try:
                await run_task
            except BaseException:  # noqa: BLE001 — teardown best-effort
                pass
            raise RuntimeError(
                "Pipecat pipeline timed out (no final result from the speech service)"
            ) from exc


class StereoCallRecorder:
    """Dual-channel WAV of the call: L = tester (persona), R = agent under test.

    The local protocol is half-duplex (turn-based), so utterances are laid out
    sequentially with silence on the opposite channel.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._left = bytearray()
        self._right = bytearray()
        # (channel, start_sample, n_samples) per utterance — lets the PII
        # audio redactor map transcript spans back to WAV intervals.
        self.segments: list[tuple[str, int, int]] = []

    def add_silence(self, seconds: float, line_pcm: bytes | None = None) -> None:
        """Inter-turn gap on BOTH channels (humanized latency made audible).

        `line_pcm` optionally carries ambience-only audio for the tester side
        (the phone line is never perfectly dead)."""
        n_bytes = max(0, int(seconds * self.sample_rate)) * 2
        if n_bytes == 0:
            return
        left = bytearray(line_pcm[:n_bytes]) if line_pcm else bytearray(n_bytes)
        left.extend(b"\x00" * (n_bytes - len(left)))
        self._left.extend(left)
        self._right.extend(b"\x00" * n_bytes)

    def add(self, channel: str, pcm: bytes) -> int:
        """Append one utterance; returns its segment index."""
        start_sample = len(self._left) // 2
        if channel == "tester":
            self._left.extend(pcm)
            self._right.extend(b"\x00" * len(pcm))
        else:
            self._right.extend(pcm)
            self._left.extend(b"\x00" * len(pcm))
        self.segments.append((channel, start_sample, len(pcm) // 2))
        return len(self.segments) - 1

    def segment_pcm(self, index: int) -> bytes:
        channel, start, n = self.segments[index]
        buf = self._left if channel == "tester" else self._right
        return bytes(buf[start * 2 : (start + n) * 2])

    def channel_copies(self) -> tuple[bytearray, bytearray]:
        return bytearray(self._left), bytearray(self._right)

    @property
    def duration_ms(self) -> int:
        return int(len(self._left) / 2 / self.sample_rate * 1000)

    def save(self, path: Path) -> None:
        frames = bytearray()
        for i in range(0, len(self._left), 2):
            frames.extend(self._left[i : i + 2])
            frames.extend(self._right[i : i + 2])
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(bytes(frames))


class AudioTransportAdapter:
    """Wraps a text-level CallTransport, converting agent audio -> text (STT)
    and tester text -> audio (TTS). CallRunner stays media-agnostic."""

    def __init__(
        self,
        inner,
        engine: PipecatAudioEngine,
        recorder: StereoCallRecorder,
        channel_fx=None,
    ):
        self.inner = inner
        self.engine = engine
        self.recorder = recorder
        # TelephoneChannelFx (callfx.py): telephone band-pass + µ-law grit +
        # seeded ambience applied to the TESTER audio only, before send.
        self.channel_fx = channel_fx
        self.stt_turns = 0
        self.tts_turns = 0
        # (segment_index, speaker, text) — feeds the post-call audio redactor.
        self.utterances: list[tuple[int, str, str]] = []
        # Optional LivePublisher: per-utterance turn_audio for the live UI.
        self.live: Any | None = None

    @property
    def url(self) -> str:
        return getattr(self.inner, "url", "?")

    async def connect(self) -> None:
        await self.inner.connect()
        await self.inner.send_event("set_mode", mode="audio")

    def record_silence(self, seconds: float) -> None:
        """Humanized pre-reply gap, audible in the WAV (with line ambience)."""
        line = self.channel_fx.comfort_noise(seconds) if self.channel_fx else None
        self.recorder.add_silence(seconds, line_pcm=line)

    async def send_text(self, text: str) -> None:
        pcm = await self.engine.synthesize(text)
        if self.channel_fx is not None:
            pcm = self.channel_fx.process(pcm)
        segment = self.recorder.add("tester", pcm)
        self.utterances.append((segment, "tester", text))
        self.tts_turns += 1
        if self.live is not None:
            # the tester `turn` was already recorded by CallRunner — the
            # publisher pairs this audio with that turnIndex immediately
            self.live.add_turn_audio("tester", pcm, self.engine.sample_rate)
        await self.inner.send_audio(pcm, sample_rate=self.engine.sample_rate)

    async def send_dtmf(self, digits: str) -> None:
        await self.inner.send_dtmf(digits)

    async def hangup(self) -> None:
        await self.inner.hangup()

    async def receive(self, timeout: float) -> dict[str, Any] | None:
        while True:
            msg = await self.inner.receive(timeout)
            if msg is None:
                return None
            if msg.get("type") == "event" and msg.get("name") == "mode_set":
                continue  # handshake ack
            if msg.get("type") != "audio":
                return msg
            pcm = base64.b64decode(msg.get("data", ""))
            segment = self.recorder.add("agent", pcm)
            text = await self.engine.transcribe(pcm)
            self.utterances.append((segment, msg.get("speaker", "agent"), text))
            self.stt_turns += 1
            if self.live is not None:
                # the matching `turn` is only recorded when CallRunner sees
                # this message — the publisher holds the audio until then
                self.live.add_turn_audio(
                    msg.get("speaker", "agent"), pcm, self.engine.sample_rate
                )
            return {
                "type": "text",
                "speaker": msg.get("speaker", "agent"),
                "text": text,
                "source": "stt",
            }
