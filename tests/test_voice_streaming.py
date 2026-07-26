import asyncio
import math
from array import array

import pytest

from voidr_echo_runner.audio import AudioTransportAdapter, StereoCallRecorder
from voidr_echo_runner.voice_gateway import validate_pcm_16k


class StreamingEngine:
    sample_rate = 16_000

    def __init__(self, events):
        self.events = events

    async def synthesize_chunks(self, text):
        self.events.append("produced-1")
        yield b"\x01\x00" * 160
        self.events.append("produced-2")
        yield b"\x02\x00" * 160


class StreamingTransport:
    def __init__(self, events):
        self.events = events
        self.audio = bytearray()

    async def send_audio_chunk(self, pcm, sample_rate):
        self.events.append(f"sent-{pcm[0]}")
        self.audio.extend(pcm)

    async def finish_audio(self, sample_rate):
        self.events.append("finished")


def test_runtime_forwards_tts_before_provider_stream_finishes():
    events = []
    engine = StreamingEngine(events)
    inner = StreamingTransport(events)
    adapter = AudioTransportAdapter(inner, engine, StereoCallRecorder())

    asyncio.run(adapter.send_text("olá"))

    assert events == ["produced-1", "sent-1", "produced-2", "sent-2", "finished"]
    assert bytes(inner.audio) == b"\x01\x00" * 160 + b"\x02\x00" * 160


def test_partial_stream_finishes_transport_without_masking_provider_error():
    events = []

    class FailingEngine(StreamingEngine):
        async def synthesize_chunks(self, text):
            yield b"\x01\x00" * 160
            raise ValueError("provider failed")

    class FailingCleanupTransport(StreamingTransport):
        async def finish_audio(self, sample_rate):
            self.events.append("finished")
            raise RuntimeError("cleanup failed")

    adapter = AudioTransportAdapter(
        FailingCleanupTransport(events),
        FailingEngine(events),
        StereoCallRecorder(),
    )
    with pytest.raises(ValueError, match="provider failed"):
        asyncio.run(adapter.send_text("olá"))
    assert events == ["sent-1", "finished"]


def test_runtime_barge_in_cancels_generator_clears_twilio_and_sends_no_late_chunks():
    events: list[str] = []

    class BargeEngine:
        sample_rate = 16_000

        async def synthesize_chunks(self, _text):
            try:
                events.append("generated-1")
                yield b"\x01\x00" * 160
                await asyncio.sleep(10)
                events.append("generated-late")
                yield b"\x02\x00" * 160
            finally:
                events.append("generator-cleanup")

    class BargeTransport(StreamingTransport):
        def __init__(self, log):
            super().__init__(log)
            self.inbox: asyncio.Queue = asyncio.Queue()

        async def receive(self, timeout):
            return await asyncio.wait_for(self.inbox.get(), timeout)

        async def send_audio_chunk(self, pcm, sample_rate):
            await super().send_audio_chunk(pcm, sample_rate)
            await self.inbox.put(
                {"type": "event", "name": "speech_started", "speaker": "agent"}
            )

        async def clear_audio(self):
            self.events.append("twilio-clear")

    inner = BargeTransport(events)
    adapter = AudioTransportAdapter(inner, BargeEngine(), StereoCallRecorder())

    asyncio.run(adapter.send_text("fala longa"))

    assert events == [
        "generated-1",
        "sent-1",
        "generator-cleanup",
        "twilio-clear",
    ]
    assert bytes(inner.audio) == b"\x01\x00" * 160
    assert adapter.barge_in_count == 1
    assert "finished" not in events
    assert "generated-late" not in events


def test_benchmark_validates_pcm_duration_and_frequency():
    rate = 16_000
    samples = array(
        "h",
        (
            int(12_000 * math.sin(2 * math.pi * 440 * index / rate))
            for index in range(rate)
        ),
    )

    valid, duration_s, crossing_hz = validate_pcm_16k(samples.tobytes(), rate)

    assert valid
    assert duration_s == 1
    assert 435 <= crossing_hz <= 445
