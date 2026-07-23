"""TwilioMediaStreamTransport tests — twilio REST mocked, Media Streams simulated."""

from __future__ import annotations

import asyncio
import audioop
import base64
import json
import math
import struct

import pytest

from voidr_echo_runner.twilio_transport import (
    PIPELINE_SAMPLE_RATE,
    TWILIO_SAMPLE_RATE,
    TwilioMediaStreamTransport,
    UtteranceSegmenter,
)

TEST_PORT = 8991


class FakeCall:
    sid = "CA_fake_123"


class FakeCallContext:
    def __init__(self, log: list):
        self._log = log

    def update(self, **kwargs):
        self._log.append(("update", kwargs))


class FakeCalls:
    def __init__(self, log: list):
        self._log = log

    def create(self, **kwargs):
        self._log.append(("create", kwargs))
        return FakeCall()

    def __call__(self, sid):
        self._log.append(("context", sid))
        return FakeCallContext(self._log)


class FakeResponse:
    status_code = 200
    text = ""


class FakeTwilioClient:
    def __init__(self):
        self.log: list = []
        self.calls = FakeCalls(self.log)

    def request(self, method, uri, params=None, data=None, **kwargs):
        self.log.append(("request", method, uri, data))
        return FakeResponse()


@pytest.fixture
def twilio_env(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC_test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token_test")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15550000001")
    monkeypatch.setenv("TWILIO_STREAM_PUBLIC_URL", "wss://example.ngrok.app")


def tone_pcm(sample_rate: int, ms: int, freq: int = 440, amp: int = 12000) -> bytes:
    n = int(sample_rate * ms / 1000)
    return b"".join(
        struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / sample_rate)))
        for i in range(n)
    )


# --- UtteranceSegmenter -------------------------------------------------------


def test_segmenter_ignores_silence():
    seg = UtteranceSegmenter()
    assert seg.feed(b"\x00" * 16000 * 2) == []  # 1s of silence
    assert seg.flush() is None


def test_segmenter_emits_utterance_after_speech_then_silence():
    seg = UtteranceSegmenter()
    speech = tone_pcm(PIPELINE_SAMPLE_RATE, 600)
    silence = b"\x00" * int(PIPELINE_SAMPLE_RATE * 1.2) * 2
    utterances = seg.feed(speech) + seg.feed(silence)
    assert len(utterances) == 1
    assert len(utterances[0]) >= len(speech)  # speech + trailing silence frames


def test_segmenter_two_bursts_two_utterances():
    seg = UtteranceSegmenter()
    speech = tone_pcm(PIPELINE_SAMPLE_RATE, 500)
    silence = b"\x00" * int(PIPELINE_SAMPLE_RATE * 1.2) * 2
    utterances = []
    for chunk in (speech, silence, speech, silence):
        utterances += seg.feed(chunk)
    assert len(utterances) == 2


# --- REST interactions (twilio lib mocked) ------------------------------------


def test_missing_env_raises(monkeypatch):
    for name in TwilioMediaStreamTransport.REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="TWILIO_ACCOUNT_SID"):
        TwilioMediaStreamTransport("+5511999999999", client=FakeTwilioClient())


def test_missing_public_url_raises(twilio_env, monkeypatch):
    monkeypatch.delenv("TWILIO_STREAM_PUBLIC_URL")
    with pytest.raises(RuntimeError, match="TWILIO_STREAM_PUBLIC_URL"):
        TwilioMediaStreamTransport("+5511999999999", client=FakeTwilioClient())


def test_send_text_is_rejected(twilio_env):
    transport = TwilioMediaStreamTransport("+5511999999999", client=FakeTwilioClient())
    with pytest.raises(RuntimeError, match="audio-only"):
        asyncio.run(transport.send_text("oi"))


def test_mid_call_dtmf_updates_twiml_and_reconnects_stream(twilio_env):
    client = FakeTwilioClient()
    transport = TwilioMediaStreamTransport("+5511999999999", client=client)
    transport.call_sid = "CA_fake_123"
    asyncio.run(transport.send_dtmf("ww919021552"))
    assert ("context", "CA_fake_123") in client.log
    kind, kwargs = client.log[-1]
    assert kind == "update"
    assert '<Play digits="ww919021552"/>' in kwargs["twiml"]
    assert '<Connect><Stream url="wss://example.ngrok.app"/></Connect>' in kwargs["twiml"]
    assert transport._expect_reconnect is True  # next stream stop is not call end


def test_mid_call_dtmf_rejects_bad_digits(twilio_env):
    transport = TwilioMediaStreamTransport("+5511999999999", client=FakeTwilioClient())
    transport.call_sid = "CA_fake_123"
    with pytest.raises(ValueError, match="invalid DTMF"):
        asyncio.run(transport.send_dtmf('"/><Hangup/>'))


# --- Full connect + media stream round trip ------------------------------------


def test_connect_media_roundtrip(twilio_env, monkeypatch):
    """calls.create params + start/media/stop handling + outbound serialization."""
    monkeypatch.setenv("TWILIO_STREAM_PUBLIC_URL", "https://example.ngrok.app")
    client = FakeTwilioClient()
    transport = TwilioMediaStreamTransport(
        "+5511999999999",
        client=client,
        listen_host="127.0.0.1",
        listen_port=TEST_PORT,
        send_digits="ww919021552ww11900000001#",
    )
    received_by_twilio: list[dict] = []

    async def fake_twilio_side():
        import websockets

        for _ in range(40):  # wait for the server to come up
            try:
                ws = await websockets.connect(f"ws://127.0.0.1:{TEST_PORT}")
                break
            except OSError:
                await asyncio.sleep(0.1)
        else:
            raise AssertionError("media server never came up")
        await ws.send(json.dumps({"event": "connected"}))
        await ws.send(
            json.dumps({"event": "start", "start": {"streamSid": "MZ_stream_1"}})
        )
        # ~600ms of 8kHz mu-law tone then 1.2s silence -> one utterance
        speech = audioop.lin2ulaw(tone_pcm(TWILIO_SAMPLE_RATE, 600), 2)
        silence = audioop.lin2ulaw(b"\x00" * int(TWILIO_SAMPLE_RATE * 1.2) * 2, 2)
        for blob in (speech, silence):
            for i in range(0, len(blob), 160):  # 20ms mu-law frames
                await ws.send(
                    json.dumps(
                        {
                            "event": "media",
                            "media": {
                                "payload": base64.b64encode(blob[i : i + 160]).decode()
                            },
                        }
                    )
                )
        # collect what the runner sends back for a short while
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.5)
                received_by_twilio.append(json.loads(raw))
        except (TimeoutError, asyncio.TimeoutError):
            pass
        await ws.send(json.dumps({"event": "stop"}))
        await ws.close()

    async def scenario():
        twilio_task = asyncio.create_task(fake_twilio_side())
        await transport.connect()
        msg = await transport.receive(timeout=10.0)
        assert msg["type"] == "audio"
        pcm = base64.b64decode(msg["data"])
        assert msg["sample_rate"] == PIPELINE_SAMPLE_RATE
        assert len(pcm) > TWILIO_SAMPLE_RATE  # >0.5s once upsampled to 16kHz
        await transport.send_audio(tone_pcm(PIPELINE_SAMPLE_RATE, 600))
        ended = await transport.receive(timeout=10.0)
        assert ended == {"type": "event", "name": "call_ended", "reason": "completed"}
        await twilio_task
        await transport.hangup()

    asyncio.run(scenario())

    kind, kwargs = client.log[0]
    assert kind == "create"
    assert kwargs["to"] == "+5511999999999"
    assert kwargs["from_"] == "+15550000001"
    assert kwargs["send_digits"] == "ww919021552ww11900000001#"
    assert kwargs["record"] is True and kwargs["recording_channels"] == "dual"
    assert '<Connect><Stream url="wss://example.ngrok.app"/></Connect>' in kwargs["twiml"]

    media_out = [m for m in received_by_twilio if m.get("event") == "media"]
    assert media_out, "runner sent no outbound media frames"
    assert all(m["streamSid"] == "MZ_stream_1" for m in media_out)
    # 600ms PCM @16kHz + ~240ms flush padding -> ~840ms mu-law @8kHz, minus
    # up to ~100ms retained inside the stream resampler.
    total = sum(len(base64.b64decode(m["media"]["payload"])) for m in media_out)
    assert 4800 <= total <= 7200

    assert ("update", {"status": "completed"}) in client.log
