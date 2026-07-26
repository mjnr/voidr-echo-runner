"""Governed voice clients used by the runner; no provider credentials required."""

from __future__ import annotations

import asyncio
from array import array
from collections.abc import AsyncIterator
import io
import json
import math
import os
import uuid
import wave
from dataclasses import dataclass
from email.message import Message
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx
from .governed_url import validate_governed_url
import websockets

LOCAL_RUNTIMES = frozenset({"local", "dev", "development", "test"})
PRODUCTION_RUNTIMES = frozenset({"cloud", "prod", "production", "staging"})
MAX_AUDIO_BYTES = 16_000 * 2 * 120
MAX_AUDIO_CHUNKS = 2_048
TTS_DEADLINE_S = 45
STT_DEADLINE_S = 60
TTS_ALIAS_DEFAULT = "echo-tts-elevenlabs-flash-v2-5@id:2026-07-26"
STT_ALIAS_DEFAULT = "echo-stt-deepgram-nova-2@id:2026-07-26"
VAD_FRAME_MS = 20
VAD_SILENCE_MS = 320
VAD_OVERLAP_MS = 180
STT_MAX_CONCURRENCY = 3


def validate_pcm_16k(pcm: bytes, sample_rate: int) -> tuple[bool, float, float]:
    """Validate PCM shape, duration, energy and plausible zero-crossing frequency."""
    if sample_rate != 16_000 or len(pcm) < 320 or len(pcm) % 2:
        return False, 0.0, 0.0
    samples = array("h")
    samples.frombytes(pcm)
    duration_s = len(samples) / sample_rate
    rms = math.sqrt(sum(value * value for value in samples) / len(samples))
    crossings = sum(
        (left < 0 <= right) or (left >= 0 > right)
        for left, right in zip(samples, samples[1:])
    )
    crossing_hz = crossings * sample_rate / (2 * max(1, len(samples) - 1))
    valid = 0.01 <= duration_s <= 120 and rms >= 8 and 40 <= crossing_hz <= 6_000
    return valid, duration_s, crossing_hz


@dataclass(frozen=True)
class VoiceConfig:
    runtime: str
    litellm_url: str | None
    virtual_key: str | None
    tts_alias: str | None
    stt_alias: str | None
    direct: bool


def assert_direct_provider_access_allowed() -> None:
    runtime = os.environ.get("ECHO_RUNTIME_ENV", "").strip().lower()
    if (
        runtime not in LOCAL_RUNTIMES
        or os.environ.get("VOICE_ALLOW_DIRECT_PROVIDERS", "").strip() != "1"
    ):
        raise RuntimeError(
            "Direct voice provider access is disabled outside explicitly opted-in "
            "local/dev/test; use LiteLLM with an org-scoped virtual key"
        )


def resolve_voice_config() -> VoiceConfig:
    runtime = os.environ.get("ECHO_RUNTIME_ENV", "").strip().lower()
    direct = (
        runtime in LOCAL_RUNTIMES
        and os.environ.get("VOICE_ALLOW_DIRECT_PROVIDERS", "").strip() == "1"
    )
    litellm_url = os.environ.get("LITELLM_BASE_URL", "").strip() or None
    virtual_key = os.environ.get("LITELLM_API_KEY", "").strip() or None
    tts_alias = os.environ.get("LITELLM_TTS_MODEL", "").strip() or None
    stt_alias = os.environ.get("LITELLM_STT_MODEL", "").strip() or None

    if runtime not in LOCAL_RUNTIMES and runtime not in PRODUCTION_RUNTIMES:
        raise RuntimeError(
            "ECHO_RUNTIME_ENV must explicitly identify local/dev/test or cloud/production"
        )
    if runtime in PRODUCTION_RUNTIMES and any(
        os.environ.get(name) for name in ("DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY")
    ):
        raise RuntimeError(
            "Direct provider credentials must not be injected into cloud/production "
            "runners; provision only LITELLM_BASE_URL + LITELLM_API_KEY"
        )
    if direct:
        assert_direct_provider_access_allowed()
        return VoiceConfig(runtime, litellm_url, virtual_key, None, None, True)
    if not litellm_url or not virtual_key or not tts_alias or not stt_alias:
        raise RuntimeError(
            "LITELLM_BASE_URL, LITELLM_API_KEY, LITELLM_TTS_MODEL and "
            "LITELLM_STT_MODEL are required; direct provider access is allowed "
            "only in local/dev/test with VOICE_ALLOW_DIRECT_PROVIDERS=1"
        )
    litellm_url = validate_governed_url(litellm_url, name="LITELLM_BASE_URL")
    return VoiceConfig(runtime, litellm_url, virtual_key, tts_alias, stt_alias, False)


class VoiceGatewayAudioEngine:
    def __init__(
        self,
        config: VoiceConfig,
        voice_id: str,
        sample_rate: int = 16_000,
        *,
        http_client: httpx.AsyncClient | None = None,
    ):
        if (
            config.direct
            or not config.litellm_url
            or not config.virtual_key
            or not config.tts_alias
            or not config.stt_alias
        ):
            raise ValueError("VoiceGatewayAudioEngine requires governed LiteLLM config")
        if sample_rate != 16_000:
            raise ValueError("Echo audio invariant is PCM s16le mono at 16 kHz")
        self.config = config
        self.voice_id = voice_id
        self.sample_rate = sample_rate
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(STT_DEADLINE_S, connect=10)
        )
        self._owns_http = http_client is None

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    def _url(self, path: str) -> str:
        base = self.config.litellm_url.rstrip("/")
        if base.endswith("/v1") and path.startswith("/v1/"):
            path = path.removeprefix("/v1")
        return f"{base}{path}"

    def _headers(self, modality: str) -> dict[str, str]:
        tags = [
            f"org:{os.environ.get('VOIDR_ORGANIZATION_ID', 'unknown')}",
            f"execution:{os.environ.get('VOIDR_EXECUTION_ID', 'unknown')}",
            f"shard:{os.environ.get('SHARDS_CURRENT', 'unknown')}",
            f"modality:{modality}",
            "service:echo-runner",
        ]
        return {
            "Authorization": f"Bearer {self.config.virtual_key}",
            "X-Voice-Request-Id": str(uuid.uuid4()),
            "x-litellm-tags": ",".join(tags),
        }

    @staticmethod
    def _response_sample_rate(response: httpx.Response) -> int:
        candidates = (
            response.headers.get("x-audio-sample-rate"),
            response.headers.get("x-litellm-audio-sample-rate"),
        )
        content_type = Message()
        content_type["content-type"] = response.headers.get("content-type", "")
        candidates += (
            content_type.get_param("rate"),
            content_type.get_param("samplerate"),
        )
        for value in candidates:
            if value is None:
                continue
            try:
                rate = int(value)
            except (TypeError, ValueError):
                continue
            if 8_000 <= rate <= 192_000:
                return rate
        raise RuntimeError("LiteLLM TTS response omitted a valid PCM sample rate")

    async def synthesize_chunks(self, text: str) -> AsyncIterator[bytes]:
        if not text or len(text) > 5_000:
            raise ValueError("TTS text must contain 1..5000 characters")
        body = {
            "model": self.config.tts_alias,
            "input": text,
            "voice": self.voice_id,
            # LiteLLM passes this provider-native ElevenLabs parameter through.
            # `response_format=pcm` maps to pcm_44100 and `sample_rate` is not
            # the ElevenLabs selector, so neither proves a 16 kHz response.
            "output_format": "pcm_16000",
        }
        total = 0
        chunks = 0
        pending = b""
        async with asyncio.timeout(TTS_DEADLINE_S):
            async with self._http.stream(
                "POST",
                self._url("/v1/audio/speech"),
                headers=self._headers("tts"),
                json=body,
            ) as response:
                response.raise_for_status()
                source_rate = self._response_sample_rate(response)
                buffered = bytearray()
                async for message in response.aiter_bytes():
                    if not message:
                        continue
                    chunks += 1
                    total += len(message)
                    if chunks > MAX_AUDIO_CHUNKS or total > MAX_AUDIO_BYTES * 3:
                        raise RuntimeError("LiteLLM TTS response exceeded limits")
                    aligned = pending + message
                    pending = aligned[-1:] if len(aligned) % 2 else b""
                    aligned = aligned[:-1] if pending else aligned
                    if source_rate == self.sample_rate:
                        if aligned:
                            yield aligned
                    else:
                        buffered.extend(aligned)
                if pending or total == 0:
                    raise RuntimeError("LiteLLM returned invalid PCM")
                if buffered:
                    resampled = resample_pcm16(bytes(buffered), source_rate, self.sample_rate)
                    for offset in range(0, len(resampled), 8_000):
                        yield resampled[offset : offset + 8_000]

    async def synthesize(self, text: str) -> bytes:
        chunks = [chunk async for chunk in self.synthesize_chunks(text)]
        if not chunks:
            raise RuntimeError("LiteLLM returned no TTS audio")
        pcm = b"".join(chunks)
        valid, _, _ = validate_pcm_16k(pcm, self.sample_rate)
        if not valid:
            raise RuntimeError("LiteLLM TTS returned invalid PCM16k audio")
        return pcm

    async def transcribe(self, pcm: bytes) -> str:
        if not pcm or len(pcm) > MAX_AUDIO_BYTES or len(pcm) % 2:
            raise ValueError("STT PCM must be non-empty 16-bit audio within duration limits")
        segments = segment_utterance(pcm, self.sample_rate)
        semaphore = asyncio.Semaphore(STT_MAX_CONCURRENCY)

        async def run(index: int, segment: bytes) -> tuple[int, str]:
            async with semaphore:
                return index, await self._transcribe_segment(segment, index)

        async with asyncio.timeout(STT_DEADLINE_S):
            completed = await asyncio.gather(
                *(run(index, segment) for index, segment in enumerate(segments))
            )
        ordered = [text for _, text in sorted(completed)]
        return merge_overlapping_transcripts(ordered)

    async def transcribe_stream(
        self,
        chunks: AsyncIterator[bytes],
        *,
        on_interim=None,
    ) -> str:
        """Overlap capture and recognition through the governed LiteLLM WS."""
        configured = os.environ.get("LITELLM_STREAMING_STT_URL", "").strip()
        if configured:
            base = configured
        else:
            parsed = urlsplit(self.config.litellm_url)
            scheme = "wss" if parsed.scheme == "https" else "ws"
            path = parsed.path.rstrip("/")
            if path.endswith("/v1"):
                path += "/audio/transcriptions/stream"
            else:
                path += "/v1/audio/transcriptions/stream"
            base = urlunsplit((scheme, parsed.netloc, path, "", ""))
        query = urlencode(
            {
                "model": self.config.stt_alias,
                "language": "pt-BR",
                "encoding": "linear16",
                "sample_rate": str(self.sample_rate),
                "channels": "1",
                "interim_results": "true",
            }
        )
        finals: dict[int, str] = {}
        async with asyncio.timeout(STT_DEADLINE_S):
            async with websockets.connect(
                f"{base}?{query}",
                additional_headers=self._headers("stt"),
                max_queue=32,
                write_limit=64 * 1024,
            ) as ws:
                async def send_audio() -> None:
                    async for chunk in chunks:
                        if not chunk or len(chunk) % 2:
                            raise ValueError("streaming STT requires aligned PCM16 chunks")
                        await ws.send(chunk)
                    await ws.send(json.dumps({"type": "CloseStream"}))

                async def receive_results() -> None:
                    async for raw in ws:
                        if not isinstance(raw, str):
                            raise RuntimeError("streaming STT returned a binary control frame")
                        payload = json.loads(raw)
                        if payload.get("type") != "transcription":
                            continue
                        sequence = payload.get("sequence")
                        text = payload.get("text")
                        if not isinstance(sequence, int) or not isinstance(text, str):
                            raise RuntimeError("streaming STT returned an invalid result")
                        if payload.get("is_final"):
                            finals[sequence] = text.strip()
                        elif on_interim is not None:
                            on_interim(sequence, text)

                await asyncio.gather(send_audio(), receive_results())
        if not finals:
            raise RuntimeError("streaming STT returned no final transcript")
        return merge_overlapping_transcripts(
            [finals[index] for index in sorted(finals)]
        )

    async def _transcribe_segment(self, pcm: bytes, index: int) -> str:
        wav = pcm16_wav(pcm, self.sample_rate)
        response = await self._http.post(
            self._url("/v1/audio/transcriptions"),
            headers=self._headers("stt"),
            data={
                "model": self.config.stt_alias,
                "language": "pt",
                "response_format": "json",
            },
            files={"file": (f"utterance-{index}.wav", wav, "audio/wav")},
        )
        response.raise_for_status()
        payload = response.json()
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str):
            raise RuntimeError("LiteLLM transcription returned no text")
        return text.strip()


def pcm16_wav(pcm: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def resample_pcm16(pcm: bytes, source_rate: int, target_rate: int = 16_000) -> bytes:
    if source_rate <= 0 or target_rate <= 0 or len(pcm) % 2:
        raise ValueError("invalid PCM resampling parameters")
    if source_rate == target_rate:
        return pcm
    source = array("h")
    source.frombytes(pcm)
    if len(source) < 2:
        return pcm
    target_length = max(1, round(len(source) * target_rate / source_rate))
    result = array("h")
    scale = source_rate / target_rate
    for target_index in range(target_length):
        position = min(target_index * scale, len(source) - 1)
        left = int(position)
        right = min(left + 1, len(source) - 1)
        fraction = position - left
        result.append(round(source[left] + (source[right] - source[left]) * fraction))
    return result.tobytes()


def segment_utterance(pcm: bytes, sample_rate: int = 16_000) -> list[bytes]:
    frame_bytes = sample_rate * VAD_FRAME_MS // 1000 * 2
    silence_frames = max(1, VAD_SILENCE_MS // VAD_FRAME_MS)
    overlap_bytes = sample_rate * VAD_OVERLAP_MS // 1000 * 2
    frames = [pcm[i : i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]
    energies: list[float] = []
    for frame in frames:
        samples = array("h")
        samples.frombytes(frame[: len(frame) - len(frame) % 2])
        energies.append(
            math.sqrt(sum(value * value for value in samples) / max(1, len(samples)))
        )
    speech_threshold = max(120.0, (sorted(energies)[len(energies) // 4] * 3) if energies else 120.0)
    boundaries: list[int] = []
    silent = 0
    for index, energy in enumerate(energies):
        silent = silent + 1 if energy < speech_threshold else 0
        if silent == silence_frames and index + 1 < len(frames):
            boundaries.append((index + 1 - silence_frames // 2) * frame_bytes)
    if not boundaries:
        return [pcm]
    segments: list[bytes] = []
    start = 0
    for boundary in boundaries + [len(pcm)]:
        left = max(0, start - (overlap_bytes if segments else 0))
        right = min(len(pcm), boundary + overlap_bytes)
        segment = pcm[left:right]
        if len(segment) >= frame_bytes * 3:
            segments.append(segment)
        start = boundary
    return segments or [pcm]


def merge_overlapping_transcripts(transcripts: list[str]) -> str:
    merged: list[str] = []
    for transcript in transcripts:
        words = transcript.split()
        overlap = 0
        for size in range(min(8, len(merged), len(words)), 0, -1):
            if [word.casefold() for word in merged[-size:]] == [
                word.casefold() for word in words[:size]
            ]:
                overlap = size
                break
        merged.extend(words[overlap:])
    return " ".join(merged).strip()
