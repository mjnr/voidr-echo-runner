import asyncio
import base64
import hashlib
import hmac
import json

import pytest

from echo_media_gateway.auth import (
    CapabilityError,
    CapabilityVerifier,
    RateLimitExceeded,
    ReplayGuard,
    VoiceRateLimiter,
)
from echo_media_gateway.state import MemoryVoiceStateBackend, StateBackendUnavailable

SECRET = "unit-test-signing-secret-at-least-32-bytes"
MODEL = "nova-2"


def capability(**overrides) -> str:
    payload = {
        "org": "org-test",
        "execution": "exec-123",
        "shard": "shard-0",
        "providers": ["deepgram"],
        "models": {"deepgram": [MODEL]},
        "voices": {},
        "iat": 1_000,
        "exp": 1_300,
        "jti": "token-123",
        "max_requests": 3,
        **overrides,
    }
    header = {"alg": "HS256", "typ": "VOICE"}

    def encode(value):
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    unsigned = f"{encode(header)}.{encode(payload)}"
    signature = hmac.new(SECRET.encode(), unsigned.encode(), hashlib.sha256).digest()
    return f"{unsigned}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"


async def test_scoped_capability_and_replay_protection():
    verifier = CapabilityVerifier(SECRET, clock=lambda: 1_100)
    claims = await verifier.verify(capability(), "request-1", "deepgram", MODEL)
    assert claims.org == "org-test"
    with pytest.raises(CapabilityError, match="replay"):
        await verifier.verify(capability(), "request-1", "deepgram", MODEL)
    with pytest.raises(CapabilityError, match="scope_denied"):
        await verifier.verify(capability(), "request-2", "deepgram", "unknown")


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"exp": 1_099}, "expired"),
        ({"exp": 2_000}, "invalid_expiry"),
        ({"providers": ["unknown"]}, "invalid_claims"),
        ({"org": "contains spaces"}, "invalid_claims"),
        ({"providers": ["deepgram"], "models": {}}, "invalid_claims"),
        ({"models": {"deepgram": []}}, "invalid_claims"),
        ({"voices": {"deepgram": ["not-allowed"]}}, "invalid_claims"),
    ],
)
async def test_capability_fails_closed(overrides, code):
    verifier = CapabilityVerifier(SECRET, clock=lambda: 1_100)
    with pytest.raises(CapabilityError, match=code):
        await verifier.verify(capability(**overrides), "request-1", "deepgram", MODEL)


async def test_token_request_limit_is_enforced():
    verifier = CapabilityVerifier(SECRET, clock=lambda: 1_100)
    token = capability(max_requests=1)
    await verifier.verify(token, "request-1", "deepgram", MODEL)
    with pytest.raises(CapabilityError, match="token_use_limit"):
        await verifier.verify(token, "request-2", "deepgram", MODEL)


async def test_cps_and_concurrency_limits():
    now = 10.0
    limiter = VoiceRateLimiter(cps=2, concurrent=1, clock=lambda: now)
    async with limiter.acquire("org", "deepgram"):
        with pytest.raises(RateLimitExceeded, match="concurrency_exceeded"):
            async with limiter.acquire("org", "deepgram"):
                pass
    async with limiter.acquire("org", "deepgram"):
        pass
    with pytest.raises(RateLimitExceeded, match="cps_exceeded"):
        async with limiter.acquire("org", "deepgram"):
            pass
    await asyncio.sleep(0)


async def test_replay_and_request_limit_are_shared_across_replicas():
    backend = MemoryVoiceStateBackend(clock=lambda: 1_100)
    first = CapabilityVerifier(
        SECRET, clock=lambda: 1_100, replay_guard=ReplayGuard(backend=backend)
    )
    second = CapabilityVerifier(
        SECRET, clock=lambda: 1_100, replay_guard=ReplayGuard(backend=backend)
    )
    token = capability(max_requests=2)
    await first.verify(token, "replica-a", "deepgram", MODEL)
    with pytest.raises(CapabilityError, match="replay"):
        await second.verify(token, "replica-a", "deepgram", MODEL)
    await second.verify(token, "replica-b", "deepgram", MODEL)
    with pytest.raises(CapabilityError, match="token_use_limit"):
        await first.verify(token, "replica-c", "deepgram", MODEL)


async def test_limits_are_shared_across_replicas():
    backend = MemoryVoiceStateBackend(clock=lambda: 10)
    first = VoiceRateLimiter(cps=2, concurrent=1, backend=backend)
    second = VoiceRateLimiter(cps=2, concurrent=1, backend=backend)
    async with first.acquire("org", "deepgram"):
        with pytest.raises(RateLimitExceeded, match="concurrency_exceeded"):
            async with second.acquire("org", "deepgram"):
                pass
    async with second.acquire("org", "deepgram"):
        pass
    with pytest.raises(RateLimitExceeded, match="cps_exceeded"):
        async with first.acquire("org", "deepgram"):
            pass


class FailingBackend(MemoryVoiceStateBackend):
    async def consume_capability(self, *args):
        raise StateBackendUnavailable("redis_unavailable")

    async def acquire_limit(self, *args):
        raise StateBackendUnavailable("redis_unavailable")


async def test_redis_failure_fails_closed():
    backend = FailingBackend()
    verifier = CapabilityVerifier(
        SECRET, clock=lambda: 1_100, replay_guard=ReplayGuard(backend=backend)
    )
    with pytest.raises(CapabilityError, match="state_unavailable"):
        await verifier.verify(capability(), "request-1", "deepgram", MODEL)
    limiter = VoiceRateLimiter(backend=backend)
    with pytest.raises(RateLimitExceeded, match="state_unavailable"):
        async with limiter.acquire("org", "deepgram"):
            pass
