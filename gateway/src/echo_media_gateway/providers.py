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
        litellm_url: str | None = None,
        litellm_key: str | None = None,
        tts_alias: str | None = None,
        stt_alias: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        require_tls: bool = False,
    ):
        self.litellm_url = (litellm_url or os.environ.get("LITELLM_BASE_URL", "")).rstrip("/")
        self.litellm_key = litellm_key or os.environ.get("LITELLM_API_KEY")
        self.tts_alias = tts_alias or os.environ.get("LITELLM_TTS_MODEL")
        self.stt_alias = stt_alias or os.environ.get("LITELLM_STT_MODEL")
        self._http = http_client
        self.require_tls = require_tls
        if self.litellm_url:
            parsed = urlsplit(self.litellm_url)
            if (
                not parsed.hostname
                or parsed.username
                or parsed.password
                or (self.require_tls and parsed.scheme != "https")
                or (not self.require_tls and parsed.scheme not in {"http", "https"})
            ):
                raise ProviderConfigurationError("litellm_tls_required")
            rules = os.environ.get(
                "AI_EGRESS_HOST_ALLOWLIST",
                "llm.voidr.co,localhost,127.0.0.1,*.test",
            ).lower().split(",")
            host = parsed.hostname.lower()
            if not any(
                host == rule.strip()
                or (
                    rule.strip().startswith("*.")
                    and host.endswith(rule.strip()[1:])
                    and len(host) > len(rule.strip()[1:])
                )
                for rule in rules
            ):
                raise ProviderConfigurationError("litellm_host_not_allowed")

    def readiness(self, enabled: set[str]) -> tuple[bool, str | None]:
        """Validate only declared routes; credentials are never returned/logged."""
        if enabled != {"litellm"}:
            return False, "unknown_provider_route"
        if not all(
            (self.litellm_url, self.litellm_key, self.tts_alias, self.stt_alias)
        ):
            return False, "provider_not_configured"
        return True, None

    def _headers(self, tags: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.litellm_key}"}
        if tags:
            headers["x-litellm-tags"] = ",".join(
                f"{key}:{value}" for key, value in sorted(tags.items())
            )
        return headers

    async def litellm(
        self,
        *,
        text: str,
        voice: str,
        model: str,
        output_format: str,
        tags: dict[str, str] | None = None,
    ) -> AsyncIterator[bytes]:
        if not self.litellm_url:
            raise ProviderConfigurationError("litellm_not_configured")
        if output_format != "pcm_16000":
            raise ProviderConfigurationError("litellm_requires_pcm_16000")
        body = {
            "model": model,
            "input": text,
            "voice": voice,
            "output_format": "pcm_16000",
        }
        speech_path = (
            "/audio/speech"
            if self.litellm_url.endswith("/v1")
            else "/v1/audio/speech"
        )
        owned = self._http is None
        client = self._http or httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10))
        try:
            async with client.stream(
                "POST",
                f"{self.litellm_url}{speech_path}",
                headers=self._headers(tags),
                json=body,
            ) as response:
                response.raise_for_status()
                source_rate = _pcm_sample_rate(response.headers)
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
        if not self.litellm_url or not self.litellm_key:
            raise ProviderConfigurationError("litellm_not_configured")
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        path = (
            "/audio/transcriptions"
            if self.litellm_url.endswith("/v1")
            else "/v1/audio/transcriptions"
        )
        owned = self._http is None
        client = self._http or httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10))
        try:
            response = await client.post(
                f"{self.litellm_url}{path}",
                headers=self._headers(tags),
                data={"model": model, "language": language, "response_format": "json"},
                files={"file": ("utterance.wav", output.getvalue(), "audio/wav")},
            )
            response.raise_for_status()
            payload = response.json()
            text = payload.get("text") if isinstance(payload, dict) else None
            if not isinstance(text, str):
                raise ProviderConfigurationError("litellm_transcription_missing_text")
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
