"""Signed, scoped capabilities and abuse controls for voice provider access."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Callable
from uuid import uuid4

from .state import (
    MemoryVoiceStateBackend,
    StateBackendUnavailable,
    VoiceStateBackend,
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,127}$")
_PROVIDERS = frozenset({"litellm"})
_TTS_PROVIDERS = frozenset({"litellm"})


class CapabilityError(ValueError):
    """A capability is absent, malformed, expired, replayed, or out of scope."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class VoiceClaims:
    org: str
    execution: str
    shard: str
    providers: frozenset[str]
    models: dict[str, frozenset[str]]
    voices: dict[str, frozenset[str]]
    issued_at: int
    expires_at: int
    jti: str
    max_requests: int

    def allows(self, provider: str, model: str) -> bool:
        if provider not in self.providers:
            return False
        allowed_models = self.models.get(provider)
        return allowed_models is not None and model in allowed_models

    def allows_voice(self, provider: str, voice: str) -> bool:
        allowed_voices = self.voices.get(provider)
        return allowed_voices is not None and voice in allowed_voices

    def tags(self, provider: str, model: str) -> dict[str, str]:
        return {
            "org": self.org,
            "execution": self.execution,
            "shard": self.shard,
            "provider": provider,
            "model": model,
        }


def _b64decode(value: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except Exception as exc:  # noqa: BLE001 - deliberately normalize parser errors
        raise CapabilityError("malformed_token") from exc


def _valid_id(value: object) -> bool:
    return isinstance(value, str) and bool(_ID.fullmatch(value))


class ReplayGuard:
    """Bounds token uses and rejects duplicate request nonces until expiry."""

    def __init__(
        self,
        clock: Callable[[], float] = time.time,
        *,
        backend: VoiceStateBackend | None = None,
    ):
        self._backend = backend or MemoryVoiceStateBackend(clock)

    async def consume(self, claims: VoiceClaims, request_id: str) -> None:
        if not _valid_id(request_id):
            raise CapabilityError("missing_request_id")
        try:
            error = await self._backend.consume_capability(
                claims.jti, request_id, claims.max_requests, claims.expires_at
            )
        except StateBackendUnavailable as exc:
            raise CapabilityError("state_unavailable") from exc
        if error is not None:
            raise CapabilityError(error)


class CapabilityVerifier:
    """Verify compact HMAC-SHA256 capabilities without third-party JWT code."""

    def __init__(
        self,
        secret: str,
        *,
        clock: Callable[[], float] = time.time,
        max_ttl_seconds: int = 900,
        replay_guard: ReplayGuard | None = None,
    ):
        if len(secret.encode()) < 32:
            raise ValueError("VOICE_GATEWAY_SIGNING_SECRET must be at least 32 bytes")
        self._secret = secret.encode()
        self._clock = clock
        self._max_ttl = max_ttl_seconds
        self._replay = replay_guard or ReplayGuard(clock)

    async def verify(
        self, token: str, request_id: str, provider: str, model: str
    ) -> VoiceClaims:
        claims = self.decode(token)
        if not claims.allows(provider, model):
            raise CapabilityError("scope_denied")
        await self._replay.consume(claims, request_id)
        return claims

    def decode(self, token: str) -> VoiceClaims:
        try:
            header64, payload64, signature64 = token.split(".")
        except ValueError as exc:
            raise CapabilityError("malformed_token") from exc
        signed = f"{header64}.{payload64}".encode()
        expected = hmac.new(self._secret, signed, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64decode(signature64)):
            raise CapabilityError("bad_signature")
        try:
            header = json.loads(_b64decode(header64))
            payload = json.loads(_b64decode(payload64))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CapabilityError("malformed_token") from exc
        if header != {"alg": "HS256", "typ": "VOICE"} or not isinstance(payload, dict):
            raise CapabilityError("malformed_token")

        required_ids = ("org", "execution", "shard", "jti")
        if any(not _valid_id(payload.get(name)) for name in required_ids):
            raise CapabilityError("invalid_claims")
        providers_raw = payload.get("providers")
        if (
            not isinstance(providers_raw, list)
            or not providers_raw
            or any(p not in _PROVIDERS for p in providers_raw)
        ):
            raise CapabilityError("invalid_claims")
        try:
            issued_at = int(payload["iat"])
            expires_at = int(payload["exp"])
            max_requests = int(payload.get("max_requests", 1))
        except (KeyError, TypeError, ValueError) as exc:
            raise CapabilityError("invalid_claims") from exc
        now = int(self._clock())
        if issued_at > now + 30:
            raise CapabilityError("not_yet_valid")
        if expires_at <= now:
            raise CapabilityError("expired")
        if expires_at <= issued_at or expires_at - issued_at > self._max_ttl:
            raise CapabilityError("invalid_expiry")
        if not 1 <= max_requests <= 10_000:
            raise CapabilityError("invalid_claims")

        raw_models = payload.get("models", {})
        if not isinstance(raw_models, dict):
            raise CapabilityError("invalid_claims")
        models: dict[str, frozenset[str]] = {}
        if set(raw_models) != set(providers_raw):
            raise CapabilityError("invalid_claims")
        for provider in providers_raw:
            values = raw_models.get(provider)
            if not isinstance(values, list) or not values:
                raise CapabilityError("invalid_claims")
            if any(not _valid_id(model) for model in values):
                raise CapabilityError("invalid_claims")
            models[provider] = frozenset(values)
        raw_voices = payload.get("voices")
        if not isinstance(raw_voices, dict):
            raise CapabilityError("invalid_claims")
        required_voice_providers = set(providers_raw) & _TTS_PROVIDERS
        if set(raw_voices) != required_voice_providers:
            raise CapabilityError("invalid_claims")
        voices: dict[str, frozenset[str]] = {}
        for provider in required_voice_providers:
            values = raw_voices.get(provider)
            if not isinstance(values, list) or not values:
                raise CapabilityError("invalid_claims")
            if any(not _valid_id(voice) for voice in values):
                raise CapabilityError("invalid_claims")
            voices[provider] = frozenset(values)
        return VoiceClaims(
            org=payload["org"],
            execution=payload["execution"],
            shard=payload["shard"],
            providers=frozenset(providers_raw),
            models=models,
            voices=voices,
            issued_at=issued_at,
            expires_at=expires_at,
            jti=payload["jti"],
            max_requests=max_requests,
        )


class RateLimitExceeded(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class VoiceRateLimiter:
    """Per-org/provider starts-per-second and concurrent session limits."""

    def __init__(
        self,
        cps: int = 10,
        concurrent: int = 20,
        clock: Callable[[], float] = time.monotonic,
        backend: VoiceStateBackend | None = None,
        lease_ttl_seconds: int = 30,
    ):
        if cps < 1 or concurrent < 1 or lease_ttl_seconds < 3:
            raise ValueError("voice limits must be positive")
        self._cps = cps
        self._concurrent_limit = concurrent
        self._backend = backend or MemoryVoiceStateBackend(clock)
        self._lease_ttl_ms = lease_ttl_seconds * 1000

    @asynccontextmanager
    async def acquire(self, org: str, provider: str) -> AsyncIterator[None]:
        scope = f"{org}:{provider}"
        lease_id = uuid4().hex
        try:
            error = await self._backend.acquire_limit(
                scope,
                lease_id,
                self._cps,
                self._concurrent_limit,
                self._lease_ttl_ms,
            )
        except StateBackendUnavailable as exc:
            raise RateLimitExceeded("state_unavailable") from exc
        if error is not None:
            raise RateLimitExceeded(error)

        owner = asyncio.current_task()

        async def renew() -> None:
            while True:
                await asyncio.sleep(self._lease_ttl_ms / 3000)
                try:
                    renewed = await self._backend.renew_limit(
                        scope, lease_id, self._lease_ttl_ms
                    )
                except StateBackendUnavailable:
                    renewed = False
                if not renewed:
                    if owner is not None:
                        owner.cancel()
                    return

        renewal = asyncio.create_task(renew())
        try:
            yield
        finally:
            renewal.cancel()
            try:
                await renewal
            except asyncio.CancelledError:
                pass
            try:
                await self._backend.release_limit(scope, lease_id)
            except StateBackendUnavailable:
                pass
