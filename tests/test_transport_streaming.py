import asyncio
import base64
import json

import pytest

from voidr_echo_runner.transport import LocalWebSocketTransport


class CapturingWebSocket:
    def __init__(self):
        self.messages: list[str] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.messages.append(message)

    async def close(self) -> None:
        self.closed = True


def test_local_transport_buffers_provider_chunks_as_one_utterance():
    async def scenario():
        transport = LocalWebSocketTransport("ws://mock.internal")
        ws = CapturingWebSocket()
        transport._ws = ws

        await transport.send_audio_chunk(b"first-", sample_rate=16_000)
        await transport.send_audio_chunk(b"second", sample_rate=16_000)
        assert ws.messages == []

        await transport.finish_audio(sample_rate=16_000)
        assert len(ws.messages) == 1
        message = json.loads(ws.messages[0])
        assert message["type"] == "audio"
        assert base64.b64decode(message["data"]) == b"first-second"

        await transport.finish_audio(sample_rate=16_000)
        assert len(ws.messages) == 1

    asyncio.run(scenario())


def test_local_transport_rejects_sample_rate_change_mid_utterance():
    async def scenario():
        transport = LocalWebSocketTransport("ws://mock.internal")
        transport._ws = CapturingWebSocket()
        await transport.send_audio_chunk(b"first", sample_rate=16_000)
        with pytest.raises(ValueError, match="sample rate"):
            await transport.send_audio_chunk(b"second", sample_rate=8_000)

    asyncio.run(scenario())
