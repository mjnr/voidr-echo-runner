"""Proxy bidirecional, auth, reconexão e teardown do media gateway."""

import asyncio
import contextlib
import json

import httpx
import pytest
import websockets

from echo_media_gateway.server import ReadinessState, _redis_backend, main, run_gateway
from echo_media_gateway.state import MemoryVoiceStateBackend

AUTH = "test-gateway-secret"


@contextlib.asynccontextmanager
async def gateway(auth_token=AUTH):
    ready = asyncio.Event()
    task = asyncio.create_task(
        run_gateway(host="127.0.0.1", port=0, auth_token=auth_token, ready=ready)
    )
    # porta 0 → precisa descobrir a porta real; run_gateway não a expõe, então
    # os testes usam uma porta fixa alta por worker.
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    yield


PORT = 18991


@contextlib.asynccontextmanager
async def running_gateway(
    auth_token=AUTH,
    port=PORT,
    *,
    require_tls=False,
    allow_insecure_runner_auth=False,
    metrics_token=None,
    readiness=None,
    state_backend=None,
):
    ready = asyncio.Event()
    task = asyncio.create_task(
        run_gateway(
            host="127.0.0.1",
            port=port,
            auth_token=auth_token,
            ready=ready,
            require_tls=require_tls,
            allow_insecure_runner_auth=allow_insecure_runner_auth,
            metrics_token=metrics_token,
            readiness=readiness,
            state_backend=state_backend,
        )
    )
    await asyncio.wait_for(ready.wait(), timeout=5)
    try:
        yield f"ws://127.0.0.1:{port}"
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def runner_headers(token=AUTH):
    return {"Authorization": f"Bearer {token}"}


async def test_bidirectional_proxy():
    async with running_gateway(port=PORT) as base:
        runner = await websockets.connect(
            f"{base}/runner/call-1", additional_headers=runner_headers()
        )
        twilio = await websockets.connect(f"{base}/twilio/call-1")

        # Twilio → runner (frames start/media do Media Streams)
        start = json.dumps({"event": "start", "start": {"streamSid": "MZ123"}})
        await twilio.send(start)
        assert json.loads(await asyncio.wait_for(runner.recv(), 5))["event"] == "start"

        media = json.dumps({"event": "media", "media": {"payload": "AAAA"}})
        await twilio.send(media)
        assert json.loads(await asyncio.wait_for(runner.recv(), 5))["event"] == "media"

        # runner → Twilio (áudio da persona)
        out = json.dumps({"event": "media", "streamSid": "MZ123", "media": {"payload": "BBBB"}})
        await runner.send(out)
        received = json.loads(await asyncio.wait_for(twilio.recv(), 5))
        assert received["media"]["payload"] == "BBBB"

        await runner.close()
        await twilio.close()


async def test_cross_replica_proxy_orders_frames_and_cleans_up():
    shared = MemoryVoiceStateBackend()
    async with running_gateway(
        port=PORT + 20, state_backend=shared
    ) as runner_replica, running_gateway(
        port=PORT + 21, state_backend=shared
    ) as twilio_replica:
        runner = await websockets.connect(
            f"{runner_replica}/runner/cross-pod", additional_headers=runner_headers()
        )
        twilio = await websockets.connect(f"{twilio_replica}/twilio/cross-pod")

        inbound = [json.dumps({"event": "media", "sequenceNumber": index}) for index in range(25)]
        for frame in inbound:
            await twilio.send(frame)
        assert [
            json.loads(await asyncio.wait_for(runner.recv(), 5))["sequenceNumber"]
            for _ in inbound
        ] == list(range(25))

        outbound = [
            json.dumps({"event": "media", "media": {"payload": str(index)}})
            for index in range(25)
        ]
        for frame in outbound:
            await runner.send(frame)
        assert [
            json.loads(await asyncio.wait_for(twilio.recv(), 5))["media"]["payload"]
            for _ in outbound
        ] == [str(index) for index in range(25)]

        await runner.close()
        with pytest.raises(websockets.ConnectionClosed) as closed:
            await asyncio.wait_for(twilio.recv(), 5)
        assert closed.value.rcvd.code == 4410

        replacement = await websockets.connect(
            f"{twilio_replica}/twilio/cross-pod"
        )
        with pytest.raises(websockets.ConnectionClosed) as unknown:
            await asyncio.wait_for(replacement.recv(), 5)
        assert unknown.value.rcvd.code == 4404


async def test_runner_requires_auth_token():
    async with running_gateway(port=PORT + 1) as base:
        ws = await websockets.connect(
            f"{base}/runner/call-x", additional_headers=runner_headers("errado")
        )
        with pytest.raises(websockets.ConnectionClosed) as err:
            await asyncio.wait_for(ws.recv(), 5)
        assert err.value.rcvd.code == 4401


async def test_twilio_with_unknown_token_is_rejected():
    async with running_gateway(port=PORT + 2) as base:
        ws = await websockets.connect(f"{base}/twilio/nao-registrado")
        with pytest.raises(websockets.ConnectionClosed) as err:
            await asyncio.wait_for(ws.recv(), 5)
        assert err.value.rcvd.code == 4404


async def test_duplicate_runner_token_is_rejected():
    async with running_gateway(port=PORT + 3) as base:
        first = await websockets.connect(
            f"{base}/runner/dup", additional_headers=runner_headers()
        )
        second = await websockets.connect(
            f"{base}/runner/dup", additional_headers=runner_headers()
        )
        with pytest.raises(websockets.ConnectionClosed) as err:
            await asyncio.wait_for(second.recv(), 5)
        assert err.value.rcvd.code == 4409
        await first.close()


async def test_twilio_reconnect_replaces_slot_and_runner_survives():
    """DTMF mid-call: o lado Twilio cai e reconecta com o mesmo token; a
    conexão do runner permanece e os frames voltam a fluir."""
    async with running_gateway(port=PORT + 4) as base:
        runner = await websockets.connect(
            f"{base}/runner/call-dtmf", additional_headers=runner_headers()
        )
        twilio1 = await websockets.connect(f"{base}/twilio/call-dtmf")
        await twilio1.send(json.dumps({"event": "start"}))
        await asyncio.wait_for(runner.recv(), 5)
        await twilio1.close()

        twilio2 = await websockets.connect(f"{base}/twilio/call-dtmf")
        await twilio2.send(json.dumps({"event": "start", "attach": 2}))
        received = json.loads(await asyncio.wait_for(runner.recv(), 5))
        assert received["attach"] == 2

        await runner.send(json.dumps({"event": "media", "media": {"payload": "CC"}}))
        assert json.loads(await asyncio.wait_for(twilio2.recv(), 5))["media"]["payload"] == "CC"

        await runner.close()
        await twilio2.close()


async def test_runner_disconnect_tears_down_twilio_side():
    async with running_gateway(port=PORT + 5) as base:
        runner = await websockets.connect(
            f"{base}/runner/call-end", additional_headers=runner_headers()
        )
        twilio = await websockets.connect(f"{base}/twilio/call-end")
        await twilio.send(json.dumps({"event": "start"}))
        await asyncio.wait_for(runner.recv(), 5)

        await runner.close()
        with pytest.raises(websockets.ConnectionClosed) as err:
            await asyncio.wait_for(twilio.recv(), 5)
        assert err.value.rcvd.code == 4410

        # token liberado — um novo runner pode reusar (nova chamada)
        runner2 = await websockets.connect(
            f"{base}/runner/call-end", additional_headers=runner_headers()
        )
        await runner2.close()


async def test_gateway_without_auth_token_accepts_any_runner_dev_only():
    async with running_gateway(
        auth_token=None,
        port=PORT + 6,
        allow_insecure_runner_auth=True,
    ) as base:
        runner = await websockets.connect(f"{base}/runner/dev-call")
        twilio = await websockets.connect(f"{base}/twilio/dev-call")
        await twilio.send(json.dumps({"event": "start"}))
        assert json.loads(await asyncio.wait_for(runner.recv(), 5))["event"] == "start"
        await runner.close()
        await twilio.close()


async def test_empty_auth_never_opens_runner_without_explicit_dev_opt_in():
    async with running_gateway(auth_token=None, port=PORT + 7) as base:
        runner = await websockets.connect(f"{base}/runner/closed")
        with pytest.raises(websockets.ConnectionClosed) as err:
            await runner.recv()
        assert err.value.rcvd.code == 4401


async def test_production_tls_gate_rejects_ws_downgrade():
    async with running_gateway(
        port=PORT + 8,
        require_tls=True,
    ) as base:
        insecure = await websockets.connect(
            f"{base}/runner/downgrade", additional_headers=runner_headers()
        )
        with pytest.raises(websockets.ConnectionClosed) as err:
            await insecure.recv()
        assert err.value.rcvd.code == 4403

        secure = await websockets.connect(
            f"{base}/runner/forwarded",
            additional_headers={**runner_headers(), "X-Forwarded-Proto": "https"},
        )
        await secure.close()


async def test_metrics_require_bearer_token():
    token = "metrics-secret"
    async with running_gateway(
        port=PORT + 9, metrics_token=token
    ) as base:
        http_base = base.replace("ws://", "http://")
        async with httpx.AsyncClient() as client:
            hidden = await client.get(f"{http_base}/metrics")
            visible = await client.get(
                f"{http_base}/metrics",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert hidden.status_code == 404
        assert visible.status_code == 200
        assert "voice_gateway_requests_total" in visible.text


async def test_health_is_liveness_and_readyz_tracks_recent_redis_ping():
    readiness = ReadinessState(max_age_s=30)
    async with running_gateway(
        port=PORT + 10, readiness=readiness
    ) as base:
        http_base = base.replace("ws://", "http://")
        async with httpx.AsyncClient() as client:
            assert (await client.get(f"{http_base}/healthz")).status_code == 200
            assert (await client.get(f"{http_base}/readyz")).status_code == 503
            readiness.mark_success()
            assert (await client.get(f"{http_base}/readyz")).status_code == 200
            readiness.mark_failure(ConnectionError("redis down"))
            assert (await client.get(f"{http_base}/healthz")).status_code == 200
            assert (await client.get(f"{http_base}/readyz")).status_code == 503
            readiness.mark_success()
            readiness.set_provider_configuration(False, "provider_not_configured")
            assert (await client.get(f"{http_base}/readyz")).status_code == 503


@pytest.mark.parametrize(
    "redis_env",
    [
        {"REDIS_URL": "redis://redis.internal:6379/0"},
        {"REDIS_HOST": "redis.internal"},
    ],
)
def test_production_redis_rejects_plaintext(monkeypatch, redis_env):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_HOST", raising=False)
    for key, value in redis_env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(SystemExit, match="rediss://"):
        _redis_backend("production")


def test_production_rediss_enforces_certificate_and_hostname_validation(monkeypatch):
    import redis.asyncio as redis

    captured = {}
    monkeypatch.setenv("REDIS_URL", "rediss://redis.internal:6380/0")
    monkeypatch.setattr(
        redis,
        "from_url",
        lambda url, **kwargs: captured.update(url=url, **kwargs) or object(),
    )
    _redis_backend("production")
    assert captured["url"].startswith("rediss://")
    assert captured["ssl_cert_reqs"] == "required"
    assert captured["ssl_check_hostname"] is True


def test_production_startup_requires_signing_secret_and_runner_auth(monkeypatch):
    monkeypatch.setenv("ECHO_RUNTIME_ENV", "production")
    monkeypatch.delenv("VOICE_GATEWAY_SIGNING_SECRET", raising=False)
    monkeypatch.delenv("ECHO_MEDIA_GATEWAY_AUTH_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="VOICE_GATEWAY_SIGNING_SECRET"):
        main()

    monkeypatch.setenv(
        "VOICE_GATEWAY_SIGNING_SECRET", "unit-test-signing-secret-at-least-32-bytes"
    )
    with pytest.raises(SystemExit, match="ECHO_MEDIA_GATEWAY_AUTH_TOKEN"):
        main()
