import asyncio
import uuid

import pytest
import redis.asyncio as redis

from echo_media_gateway.state import (
    RedisVoiceStateBackend,
    StateBackendUnavailable,
    unpack_relay,
)


async def _shared_redis():
    client = redis.Redis(
        host="127.0.0.1",
        port=6379,
        socket_connect_timeout=0.2,
        socket_timeout=0.2,
    )
    try:
        await client.ping()
    except Exception:
        await client.aclose()
        pytest.skip("shared local Redis is not available")
    return client


async def test_redis_atomic_state_is_shared_across_replicas():
    client = await _shared_redis()
    prefix = f"voidr:test:voice:{uuid.uuid4().hex}"
    first = RedisVoiceStateBackend(client, prefix=prefix)
    second = RedisVoiceStateBackend(client, prefix=prefix)
    try:
        replay_results = await asyncio.gather(
            first.consume_capability("jti", "same-request", 2, 4_102_444_800),
            second.consume_capability("jti", "same-request", 2, 4_102_444_800),
        )
        assert sorted(str(item) for item in replay_results) == ["None", "replay"]

        limit_results = await asyncio.gather(
            first.acquire_limit("org:deepgram", "lease-a", 10, 1, 30_000),
            second.acquire_limit("org:deepgram", "lease-b", 10, 1, 30_000),
        )
        assert sorted(str(item) for item in limit_results) == [
            "None",
            "concurrency_exceeded",
        ]
    finally:
        keys = [key async for key in client.scan_iter(match=f"{prefix}:*")]
        if keys:
            await client.delete(*keys)
        await client.aclose()


async def test_redis_relay_cross_replica_ordering_and_cleanup():
    client = await _shared_redis()
    prefix = f"voidr:test:relay:{uuid.uuid4().hex}"
    first = RedisVoiceStateBackend(client, prefix=prefix)
    second = RedisVoiceStateBackend(client, prefix=prefix)
    runner_sub = await first.subscribe_relay("call", "twilio_to_runner")
    twilio_sub = await second.subscribe_relay("call", "runner_to_twilio")
    try:
        assert await first.claim_runner("call", "runner-a", 30_000)
        assert await second.runner_exists("call")
        await second.claim_twilio("call", "twilio-b", 30_000)

        for index in range(20):
            assert await second.publish_relay(
                "call", "twilio_to_runner", f"in-{index}", "twilio-b"
            )
        assert [
            unpack_relay(await asyncio.wait_for(runner_sub.get(), 2))[1]
            for _ in range(20)
        ] == [f"in-{index}" for index in range(20)]

        for index in range(20):
            assert await first.publish_relay(
                "call", "runner_to_twilio", f"out-{index}"
            )
        received = [
            unpack_relay(await asyncio.wait_for(twilio_sub.get(), 2))
            for _ in range(20)
        ]
        assert [owner for owner, _ in received] == ["twilio-b"] * 20
        assert [message for _, message in received] == [
            f"out-{index}" for index in range(20)
        ]

        await first.release_runner("call", "runner-a")
        assert await asyncio.wait_for(twilio_sub.get(), 2) == b""
        assert not await second.runner_exists("call")
    finally:
        await runner_sub.close()
        await twilio_sub.close()
        keys = [key async for key in client.scan_iter(match=f"{prefix}:*")]
        if keys:
            await client.delete(*keys)
        await client.aclose()


class BrokenRedis:
    async def ping(self):
        raise ConnectionError("down")

    async def eval(self, *args):
        raise ConnectionError("down")

    async def aclose(self):
        return None


async def test_redis_unavailability_is_normalized_fail_closed():
    backend = RedisVoiceStateBackend(BrokenRedis())
    with pytest.raises(StateBackendUnavailable, match="redis_unavailable"):
        await backend.ping()
    with pytest.raises(StateBackendUnavailable, match="redis_unavailable"):
        await backend.consume_capability("jti", "request", 1, 4_102_444_800)
    with pytest.raises(StateBackendUnavailable, match="redis_unavailable"):
        await backend.acquire_limit("org:deepgram", "lease", 1, 1, 30_000)
