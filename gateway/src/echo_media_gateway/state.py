"""Shared governance state for voice gateway replicas."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from typing import Protocol


class StateBackendUnavailable(RuntimeError):
    """The shared governance backend cannot safely answer."""


class VoiceStateBackend(Protocol):
    async def ping(self) -> None: ...

    async def close(self) -> None: ...

    async def consume_capability(
        self, jti: str, request_id: str, max_requests: int, expires_at: int
    ) -> str | None: ...

    async def acquire_limit(
        self,
        scope: str,
        lease_id: str,
        cps: int,
        concurrent: int,
        lease_ttl_ms: int,
    ) -> str | None: ...

    async def renew_limit(self, scope: str, lease_id: str, lease_ttl_ms: int) -> bool: ...

    async def release_limit(self, scope: str, lease_id: str) -> None: ...

    async def subscribe_relay(self, token: str, direction: str): ...

    async def claim_runner(self, token: str, owner: str, ttl_ms: int) -> bool: ...

    async def renew_runner(self, token: str, owner: str, ttl_ms: int) -> bool: ...

    async def release_runner(self, token: str, owner: str) -> None: ...

    async def runner_exists(self, token: str) -> bool: ...

    async def claim_twilio(self, token: str, owner: str, ttl_ms: int) -> None: ...

    async def renew_twilio(self, token: str, owner: str, ttl_ms: int) -> bool: ...

    async def release_twilio(self, token: str, owner: str) -> None: ...

    async def publish_relay(
        self, token: str, direction: str, message: str | bytes, owner: str | None = None
    ) -> bool: ...


def _pack_relay(message: str | bytes, owner: str = "") -> bytes:
    binary = isinstance(message, bytes)
    raw = message if binary else message.encode()
    return json.dumps(
        {
            "owner": owner,
            "binary": binary,
            "payload": base64.b64encode(raw).decode("ascii"),
        },
        separators=(",", ":"),
    ).encode()


def unpack_relay(payload: bytes) -> tuple[str, str | bytes]:
    owner = ""
    if payload.startswith(b"\x1e"):
        _, owner_raw, payload = payload.split(b"\x1e", 2)
        owner = owner_raw.decode()
    data = json.loads(payload)
    raw = base64.b64decode(data["payload"], validate=True)
    return owner or str(data.get("owner", "")), raw if data["binary"] else raw.decode()


class MemoryRelaySubscription:
    def __init__(self, backend: "MemoryVoiceStateBackend", channel: str):
        self._backend = backend
        self._channel = channel
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        backend._relay_subscribers.setdefault(channel, set()).add(self._queue)

    async def get(self) -> bytes:
        return await self._queue.get()

    async def close(self) -> None:
        subscribers = self._backend._relay_subscribers.get(self._channel)
        if subscribers is not None:
            subscribers.discard(self._queue)
            if not subscribers:
                self._backend._relay_subscribers.pop(self._channel, None)


@dataclass
class _MemoryLimit:
    tokens: float
    updated_ms: int
    active: dict[str, int] = field(default_factory=dict)


class MemoryVoiceStateBackend:
    """Explicit local/test double. Share one instance to simulate replicas."""

    def __init__(self, clock=time.time):
        self._clock = clock
        self._lock = asyncio.Lock()
        self._replay: dict[str, tuple[int, set[str]]] = {}
        self._limits: dict[str, _MemoryLimit] = {}
        self._runner_leases: dict[str, tuple[str, int]] = {}
        self._twilio_leases: dict[str, tuple[str, int]] = {}
        self._relay_subscribers: dict[str, set[asyncio.Queue[bytes]]] = {}

    async def ping(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def consume_capability(
        self, jti: str, request_id: str, max_requests: int, expires_at: int
    ) -> str | None:
        async with self._lock:
            now = int(self._clock())
            self._replay = {
                key: value for key, value in self._replay.items() if value[0] > now
            }
            _, requests = self._replay.setdefault(jti, (expires_at, set()))
            if request_id in requests:
                return "replay"
            if len(requests) >= max_requests:
                return "token_use_limit"
            requests.add(request_id)
            return None

    async def acquire_limit(
        self,
        scope: str,
        lease_id: str,
        cps: int,
        concurrent: int,
        lease_ttl_ms: int,
    ) -> str | None:
        async with self._lock:
            now_ms = int(self._clock() * 1000)
            state = self._limits.setdefault(scope, _MemoryLimit(float(cps), now_ms))
            state.active = {
                lease: expiry for lease, expiry in state.active.items() if expiry > now_ms
            }
            elapsed = max(0, now_ms - state.updated_ms)
            state.tokens = min(float(cps), state.tokens + elapsed * cps / 1000)
            state.updated_ms = now_ms
            if state.tokens < 1:
                return "cps_exceeded"
            if len(state.active) >= concurrent:
                return "concurrency_exceeded"
            state.tokens -= 1
            state.active[lease_id] = now_ms + lease_ttl_ms
            return None

    async def renew_limit(self, scope: str, lease_id: str, lease_ttl_ms: int) -> bool:
        async with self._lock:
            state = self._limits.get(scope)
            if state is None or lease_id not in state.active:
                return False
            state.active[lease_id] = int(self._clock() * 1000) + lease_ttl_ms
            return True

    async def release_limit(self, scope: str, lease_id: str) -> None:
        async with self._lock:
            state = self._limits.get(scope)
            if state is not None:
                state.active.pop(lease_id, None)

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    @staticmethod
    def _relay_channel(token: str, direction: str) -> str:
        return f"{token}:{direction}"

    async def subscribe_relay(self, token: str, direction: str) -> MemoryRelaySubscription:
        return MemoryRelaySubscription(self, self._relay_channel(token, direction))

    async def claim_runner(self, token: str, owner: str, ttl_ms: int) -> bool:
        async with self._lock:
            current = self._runner_leases.get(token)
            if current is not None and current[1] > self._now_ms():
                return False
            self._runner_leases[token] = (owner, self._now_ms() + ttl_ms)
            return True

    async def renew_runner(self, token: str, owner: str, ttl_ms: int) -> bool:
        async with self._lock:
            if self._runner_leases.get(token, ("", 0))[0] != owner:
                return False
            self._runner_leases[token] = (owner, self._now_ms() + ttl_ms)
            return True

    async def release_runner(self, token: str, owner: str) -> None:
        released = False
        async with self._lock:
            if self._runner_leases.get(token, ("", 0))[0] == owner:
                self._runner_leases.pop(token, None)
                released = True
        if released:
            await self._publish(self._relay_channel(token, "runner_to_twilio"), b"")

    async def runner_exists(self, token: str) -> bool:
        async with self._lock:
            current = self._runner_leases.get(token)
            return current is not None and current[1] > self._now_ms()

    async def claim_twilio(self, token: str, owner: str, ttl_ms: int) -> None:
        async with self._lock:
            self._twilio_leases[token] = (owner, self._now_ms() + ttl_ms)

    async def renew_twilio(self, token: str, owner: str, ttl_ms: int) -> bool:
        async with self._lock:
            if self._twilio_leases.get(token, ("", 0))[0] != owner:
                return False
            self._twilio_leases[token] = (owner, self._now_ms() + ttl_ms)
            return True

    async def release_twilio(self, token: str, owner: str) -> None:
        async with self._lock:
            if self._twilio_leases.get(token, ("", 0))[0] == owner:
                self._twilio_leases.pop(token, None)

    async def _publish(self, channel: str, payload: bytes) -> None:
        for queue in tuple(self._relay_subscribers.get(channel, ())):
            queue.put_nowait(payload)

    async def publish_relay(
        self, token: str, direction: str, message: str | bytes, owner: str | None = None
    ) -> bool:
        async with self._lock:
            target = ""
            if direction == "runner_to_twilio":
                lease = self._twilio_leases.get(token)
                if lease is None or lease[1] <= self._now_ms():
                    return True
                target = lease[0]
            elif self._twilio_leases.get(token, ("", 0))[0] != owner:
                return False
        await self._publish(self._relay_channel(token, direction), _pack_relay(message, target))
        return True


_REPLAY_SCRIPT = """
if redis.call('HEXISTS', KEYS[1], ARGV[1]) == 1 then return 1 end
if redis.call('HLEN', KEYS[1]) >= tonumber(ARGV[2]) then return 2 end
redis.call('HSET', KEYS[1], ARGV[1], '1')
redis.call('EXPIREAT', KEYS[1], ARGV[3])
return 0
"""

_ACQUIRE_SCRIPT = """
local now_parts = redis.call('TIME')
local now = tonumber(now_parts[1]) * 1000 + math.floor(tonumber(now_parts[2]) / 1000)
local cps = tonumber(ARGV[1])
local concurrent = tonumber(ARGV[2])
local lease_ttl = tonumber(ARGV[4])
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now)
local tokens = tonumber(redis.call('HGET', KEYS[1], 'tokens'))
local updated = tonumber(redis.call('HGET', KEYS[1], 'updated'))
if tokens == nil then tokens = cps end
if updated == nil then updated = now end
tokens = math.min(cps, tokens + math.max(0, now - updated) * cps / 1000)
redis.call('HSET', KEYS[1], 'tokens', tokens, 'updated', now)
redis.call('PEXPIRE', KEYS[1], lease_ttl * 2)
if tokens < 1 then return 1 end
if redis.call('ZCARD', KEYS[2]) >= concurrent then return 2 end
redis.call('HSET', KEYS[1], 'tokens', tokens - 1)
redis.call('ZADD', KEYS[2], now + lease_ttl, ARGV[3])
redis.call('PEXPIRE', KEYS[2], lease_ttl * 2)
return 0
"""

_RENEW_SCRIPT = """
if redis.call('ZSCORE', KEYS[1], ARGV[1]) == false then return 0 end
local now_parts = redis.call('TIME')
local now = tonumber(now_parts[1]) * 1000 + math.floor(tonumber(now_parts[2]) / 1000)
redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]), ARGV[1])
redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[2]) * 2)
return 1
"""

_RENEW_VALUE_SCRIPT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('PEXPIRE', KEYS[1], ARGV[2])
return 1
"""

_RELEASE_VALUE_SCRIPT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('DEL', KEYS[1])
return 1
"""

_PUBLISH_RUNNER_SCRIPT = """
local owner = redis.call('GET', KEYS[1])
if owner == false then return 0 end
redis.call('PUBLISH', KEYS[2], ARGV[1] .. owner .. ARGV[2])
return 1
"""

_PUBLISH_TWILIO_SCRIPT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('PUBLISH', KEYS[2], ARGV[2])
return 1
"""


class RedisRelaySubscription:
    def __init__(self, pubsub):
        self._pubsub = pubsub

    async def get(self) -> bytes:
        while True:
            message = await self._pubsub.get_message(
                ignore_subscribe_messages=True, timeout=None
            )
            if message is not None:
                return bytes(message["data"])

    async def close(self) -> None:
        await self._pubsub.aclose()


class RedisVoiceStateBackend:
    """Atomic Redis implementation using the platform/Hive Redis instance."""

    def __init__(self, client, *, prefix: str = "voidr:voice"):
        self._client = client
        self._prefix = prefix

    async def ping(self) -> None:
        try:
            await self._client.ping()
        except Exception as exc:  # redis errors vary by transport/version
            raise StateBackendUnavailable("redis_unavailable") from exc

    async def close(self) -> None:
        await self._client.aclose()

    async def consume_capability(
        self, jti: str, request_id: str, max_requests: int, expires_at: int
    ) -> str | None:
        try:
            result = await self._client.eval(
                _REPLAY_SCRIPT,
                1,
                f"{self._prefix}:replay:{jti}",
                request_id,
                max_requests,
                expires_at,
            )
        except Exception as exc:
            raise StateBackendUnavailable("redis_unavailable") from exc
        return {0: None, 1: "replay", 2: "token_use_limit"}[int(result)]

    @staticmethod
    def _limit_keys(prefix: str, scope: str) -> tuple[str, str]:
        slot = f"{{{scope}}}"
        return f"{prefix}:bucket:{slot}", f"{prefix}:active:{slot}"

    async def acquire_limit(
        self,
        scope: str,
        lease_id: str,
        cps: int,
        concurrent: int,
        lease_ttl_ms: int,
    ) -> str | None:
        bucket, active = self._limit_keys(self._prefix, scope)
        try:
            result = await self._client.eval(
                _ACQUIRE_SCRIPT,
                2,
                bucket,
                active,
                cps,
                concurrent,
                lease_id,
                lease_ttl_ms,
            )
        except Exception as exc:
            raise StateBackendUnavailable("redis_unavailable") from exc
        return {0: None, 1: "cps_exceeded", 2: "concurrency_exceeded"}[int(result)]

    async def renew_limit(self, scope: str, lease_id: str, lease_ttl_ms: int) -> bool:
        _, active = self._limit_keys(self._prefix, scope)
        try:
            return bool(
                await self._client.eval(
                    _RENEW_SCRIPT, 1, active, lease_id, lease_ttl_ms
                )
            )
        except Exception as exc:
            raise StateBackendUnavailable("redis_unavailable") from exc

    async def release_limit(self, scope: str, lease_id: str) -> None:
        _, active = self._limit_keys(self._prefix, scope)
        try:
            await self._client.zrem(active, lease_id)
        except Exception as exc:
            raise StateBackendUnavailable("redis_unavailable") from exc

    def _relay_key(self, token: str, suffix: str) -> str:
        return f"{self._prefix}:relay:{{{token}}}:{suffix}"

    async def subscribe_relay(self, token: str, direction: str) -> RedisRelaySubscription:
        pubsub = self._client.pubsub()
        try:
            await pubsub.subscribe(self._relay_key(token, direction))
        except Exception as exc:
            await pubsub.aclose()
            raise StateBackendUnavailable("redis_unavailable") from exc
        return RedisRelaySubscription(pubsub)

    async def claim_runner(self, token: str, owner: str, ttl_ms: int) -> bool:
        try:
            return bool(
                await self._client.set(
                    self._relay_key(token, "runner"), owner, nx=True, px=ttl_ms
                )
            )
        except Exception as exc:
            raise StateBackendUnavailable("redis_unavailable") from exc

    async def renew_runner(self, token: str, owner: str, ttl_ms: int) -> bool:
        return await self._renew_value(token, "runner", owner, ttl_ms)

    async def release_runner(self, token: str, owner: str) -> None:
        try:
            released = await self._client.eval(
                _RELEASE_VALUE_SCRIPT,
                1,
                self._relay_key(token, "runner"),
                owner,
            )
            if released:
                await self._client.publish(
                    self._relay_key(token, "runner_to_twilio"), b""
                )
        except Exception as exc:
            raise StateBackendUnavailable("redis_unavailable") from exc

    async def runner_exists(self, token: str) -> bool:
        try:
            return bool(await self._client.exists(self._relay_key(token, "runner")))
        except Exception as exc:
            raise StateBackendUnavailable("redis_unavailable") from exc

    async def claim_twilio(self, token: str, owner: str, ttl_ms: int) -> None:
        try:
            await self._client.set(
                self._relay_key(token, "twilio"), owner, px=ttl_ms
            )
        except Exception as exc:
            raise StateBackendUnavailable("redis_unavailable") from exc

    async def renew_twilio(self, token: str, owner: str, ttl_ms: int) -> bool:
        return await self._renew_value(token, "twilio", owner, ttl_ms)

    async def release_twilio(self, token: str, owner: str) -> None:
        try:
            await self._client.eval(
                _RELEASE_VALUE_SCRIPT,
                1,
                self._relay_key(token, "twilio"),
                owner,
            )
        except Exception as exc:
            raise StateBackendUnavailable("redis_unavailable") from exc

    async def _renew_value(
        self, token: str, suffix: str, owner: str, ttl_ms: int
    ) -> bool:
        try:
            return bool(
                await self._client.eval(
                    _RENEW_VALUE_SCRIPT,
                    1,
                    self._relay_key(token, suffix),
                    owner,
                    ttl_ms,
                )
            )
        except Exception as exc:
            raise StateBackendUnavailable("redis_unavailable") from exc

    async def publish_relay(
        self, token: str, direction: str, message: str | bytes, owner: str | None = None
    ) -> bool:
        try:
            channel = self._relay_key(token, direction)
            if direction == "runner_to_twilio":
                # The current Twilio attach is read and prefixed atomically so a
                # superseded replica can never receive frames for the new attach.
                marker = b"\x1e"
                packed = _pack_relay(message)
                return bool(
                    await self._client.eval(
                        _PUBLISH_RUNNER_SCRIPT,
                        2,
                        self._relay_key(token, "twilio"),
                        channel,
                        marker,
                        marker + packed,
                    )
                )
            if owner is None:
                return False
            return bool(
                await self._client.eval(
                    _PUBLISH_TWILIO_SCRIPT,
                    2,
                    self._relay_key(token, "twilio"),
                    channel,
                    owner,
                    _pack_relay(message),
                )
            )
        except Exception as exc:
            raise StateBackendUnavailable("redis_unavailable") from exc
