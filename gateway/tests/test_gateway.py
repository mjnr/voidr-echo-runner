"""Proxy bidirecional, auth, reconexão e teardown do media gateway."""

import asyncio
import contextlib
import json

import pytest
import websockets

from echo_media_gateway.server import run_gateway

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
async def running_gateway(auth_token=AUTH, port=PORT):
    ready = asyncio.Event()
    task = asyncio.create_task(
        run_gateway(host="127.0.0.1", port=port, auth_token=auth_token, ready=ready)
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
    async with running_gateway(auth_token=None, port=PORT + 6) as base:
        runner = await websockets.connect(f"{base}/runner/dev-call")
        twilio = await websockets.connect(f"{base}/twilio/dev-call")
        await twilio.send(json.dumps({"event": "start"}))
        assert json.loads(await asyncio.wait_for(runner.recv(), 5))["event"] == "start"
        await runner.close()
        await twilio.close()
