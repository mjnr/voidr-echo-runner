"""Provider-side transports. Credentials never cross the gateway boundary."""

from __future__ import annotations

import os
import io
import wave
from array import array
from collections.abc import AsyncIterator
from email.message import Message
from typing import Any
from urllib.parse import urlsplit

import httpx


class ProviderConfigurationError(RuntimeError):
    pass


def _pcm_sample_rate(headers: httpx.Headers) -> int:
    content_type = Message()
    content_type["content-type"] = headers.get("content-type", "")
    for value in (
        headers.get("x-audio-sample-rate"),
        headers.get("x-litellm-audio-sample-rate"),
        content_type.get_param("rate"),
        content_type.get_param("samplerate"),
    ):
        if value is None:
            continue
        try:
            rate = int(value)
        except (TypeError, ValueError):
            continue
        if 8_000 <= rate <= 192_000:
            return rate
    raise ProviderConfigurationError("litellm_pcm_sample_rate_missing")


def _resample_pcm16(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    if len(pcm) % 2:
        raise ProviderConfigurationError("litellm_invalid_pcm")
    if source_rate == target_rate:
        return pcm
    source = array("h")
    source.frombytes(pcm)
    if not source:
        return b""
    output_count = round(len(source) * target_rate / source_rate)
    output = array("h")
    for index in range(output_count):
        position = index * source_rate / target_rate
        left = min(int(position), len(source) - 1)
        right = min(left + 1, len(source) - 1)
        fraction = position - left
        output.append(round(source[left] * (1 - fraction) + source[right] * fraction))
    return output.tobytes()


class VoiceProviders:
    def __init__(
        self,
        *,
        elevenlabs_key: str | None = None,
        deepgram_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        require_tls: bool = False,
    ):
        self.elevenlabs_key = elevenlabs_key or os.environ.get("ELEVENLABS_API_KEY")
        self.deepgram_key = deepgram_key or os.environ.get("DEEPGRAM_API_KEY")
        self._http = http_client
        self.require_tls = require_tls

    def readiness(self, enabled: set[str]) -> tuple[bool, str | None]:
        """Validate only declared routes; credentials are never returned/logged."""
        if enabled != {"deepgram", "elevenlabs"}:
            return False, "unknown_provider_route"
        if not self.deepgram_key or not self.elevenlabs_key:
            return False, "provider_not_configured"
        return True, None

    async def elevenlabs(
        self,
        *,
        text: str,
        voice: str,
        model: str,
        output_format: str,
        tags: dict[str, str] | None = None,
    ) -> AsyncIterator[bytes]:
        if not self.elevenlabs_key:
            raise ProviderConfigurationError("elevenlabs_not_configured")
        if output_format != "pcm_16000":
            raise ProviderConfigurationError("elevenlabs_requires_pcm_16000")
        body = {
            "model_id": model,
            "text": text,
            "language_code": "pt",
        }
        owned = self._http is None
        client = self._http or httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10))
        try:
            async with client.stream(
                "POST",
                (
                    "https://api.elevenlabs.io/v1/text-to-speech/"
                    f"{voice}?output_format=pcm_16000"
                ),
                headers={
                    "xi-api-key": self.elevenlabs_key,
                    "Content-Type": "application/json",
                },
                json=body,
            ) as response:
                response.raise_for_status()
                source_rate = 16_000
                buffered = bytearray()
                async for chunk in response.aiter_bytes():
                    if source_rate == 16_000:
                        if len(chunk) % 2:
                            buffered.extend(chunk)
                        elif chunk:
                            yield chunk
                    else:
                        buffered.extend(chunk)
                if buffered:
                    normalized = _resample_pcm16(bytes(buffered), source_rate, 16_000)
                    for offset in range(0, len(normalized), 8_000):
                        yield normalized[offset : offset + 8_000]
        finally:
            if owned:
                await client.aclose()

    async def transcribe(
        self,
        *,
        pcm: bytes,
        model: str,
        sample_rate: int,
        language: str,
        tags: dict[str, str] | None = None,
    ) -> str:
        if not self.deepgram_key:
            raise ProviderConfigurationError("deepgram_not_configured")
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        owned = self._http is None
        client = self._http or httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10))
        try:
            response = await client.post(
                "https://api.deepgram.com/v1/listen",
                headers={
                    "Authorization": f"Token {self.deepgram_key}",
                    "Content-Type": "audio/wav",
                },
                params={
                    "model": model,
                    "language": language,
                    "smart_format": "true",
                    "punctuate": "true",
                },
                content=output.getvalue(),
            )
            response.raise_for_status()
            payload = response.json()
            alternatives = (
                payload.get("results", {}).get("channels", [{}])[0].get("alternatives", [])
                if isinstance(payload, dict)
                else []
            )
            text = alternatives[0].get("transcript") if alternatives else None
            if not isinstance(text, str):
                raise ProviderConfigurationError("deepgram_transcription_missing_text")
            return text.strip()
        finally:
            if owned:
                await client.aclose()

    async def _post_stream(self, url: str, **kwargs: Any) -> AsyncIterator[bytes]:
        owned = self._http is None
        client = self._http or httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10))
        try:
            async with client.stream("POST", url, **kwargs) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
        finally:
            if owned:
                await client.aclose()
