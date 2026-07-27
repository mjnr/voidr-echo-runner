import asyncio
import math
from array import array

import httpx
import pytest
import websockets

import voidr_echo_runner.voice_gateway as voice_module
from voidr_echo_runner.voice_gateway import (
    VoiceConfig,
    VoiceGatewayAudioEngine,
    merge_overlapping_transcripts,
    resample_pcm16,
    resolve_voice_config,
)


VOICE_ENV = (
    "ECHO_RUNTIME_ENV",
    "DEEPGRAM_API_KEY",
    "ELEVENLABS_API_KEY",
    "DEEPGRAM_STREAMING_URL",
)


def clear_voice_env(monkeypatch):
    for name in VOICE_ENV:
        monkeypatch.delenv(name, raising=False)


def direct_config() -> VoiceConfig:
    return VoiceConfig("test", "deepgram-key", "elevenlabs-key")


def test_all_runtimes_require_both_direct_projected_keys(monkeypatch):
    clear_voice_env(monkeypatch)
    monkeypatch.setenv("ECHO_RUNTIME_ENV", "production")
    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY and ELEVENLABS_API_KEY"):
        resolve_voice_config()
    monkeypatch.setenv("DEEPGRAM_API_KEY", "deepgram")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "elevenlabs")
    config = resolve_voice_config()
    assert config.deepgram_key == "deepgram"
    assert config.elevenlabs_key == "elevenlabs"


@pytest.mark.asyncio
async def test_tts_streams_direct_elevenlabs_pcm16k_without_litellm_headers():
    samples = array(
        "h",
        [round(4_000 * math.sin(2 * math.pi * 440 * index / 16_000)) for index in range(1600)],
    )
    captured = {}

    async def handler(request):
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = (await request.aread()).decode()
        return httpx.Response(200, content=samples.tobytes())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = VoiceGatewayAudioEngine(direct_config(), "voice-a", http_client=client)
        chunks = [chunk async for chunk in engine.synthesize_chunks("olá")]

    assert b"".join(chunks) == samples.tobytes()
    assert captured["url"].startswith(
        "https://api.elevenlabs.io/v1/text-to-speech/voice-a?output_format=pcm_16000"
    )
    assert captured["headers"]["xi-api-key"] == "elevenlabs-key"
    assert "authorization" not in captured["headers"]
    assert '"model_id":"eleven_flash_v2_5"' in captured["body"]


@pytest.mark.asyncio
async def test_tts_resamples_declared_pcm_before_exposing_16khz():
    source_rate = 24_000
    source = array(
        "h",
        [
            round(4_000 * math.sin(2 * math.pi * 440 * index / source_rate))
            for index in range(source_rate // 2)
        ],
    )

    async def handler(_request):
        return httpx.Response(
            200,
            headers={"x-audio-sample-rate": str(source_rate)},
            content=source.tobytes(),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = VoiceGatewayAudioEngine(direct_config(), "voice-a", http_client=client)
        pcm = b"".join([chunk async for chunk in engine.synthesize_chunks("olá")])
    assert len(pcm) // 2 == 8_000


@pytest.mark.asyncio
async def test_stt_direct_deepgram_preserves_segment_order(monkeypatch):
    engine = VoiceGatewayAudioEngine(
        direct_config(),
        "voice-a",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: None)),
    )
    monkeypatch.setattr(voice_module, "segment_utterance", lambda *_: [b"a" * 320, b"b" * 320])

    async def fake_transcribe(_segment, index):
        await asyncio.sleep(0.02 if index == 0 else 0)
        return ["saldo disponível", "disponível agora"][index]

    monkeypatch.setattr(engine, "_transcribe_segment", fake_transcribe)
    try:
        assert await engine.transcribe(b"\x01\x00" * 320) == "saldo disponível agora"
    finally:
        await engine._http.aclose()


def test_resampling_and_overlap_merge_are_deterministic():
    source = array("h", range(-1000, 1000)).tobytes()
    assert len(resample_pcm16(source, 8_000, 16_000)) == len(source) * 2
    assert (
        merge_overlapping_transcripts(["um dois três", "dois três quatro"])
        == "um dois três quatro"
    )


@pytest.mark.asyncio
async def test_streaming_stt_uses_deepgram_interim_and_final_protocol(monkeypatch):
    seen = {"chunks": []}
    audio_gate = asyncio.Event()
    interims = []

    async def upstream(ws):
        seen["authorization"] = ws.request.headers["authorization"]
        seen["path"] = ws.request.path
        seen["chunks"].append(await ws.recv())
        audio_gate.set()
        await ws.send(
            '{"type":"Results","start":0.7,"is_final":false,'
            '"channel":{"alternatives":[{"transcript":"par"}]}}'
        )
        seen["chunks"].append(await ws.recv())
        assert await ws.recv() == '{"type": "CloseStream"}'
        await ws.send(
            '{"type":"Results","start":1,"is_final":true,'
            '"channel":{"alternatives":[{"transcript":"disponível agora"}]}}'
        )
        await ws.send(
            '{"type":"Results","start":0,"is_final":true,'
            '"channel":{"alternatives":[{"transcript":"saldo disponível"}]}}'
        )

    async with websockets.serve(upstream, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv("DEEPGRAM_STREAMING_URL", f"ws://127.0.0.1:{port}/v1/listen")
        engine = VoiceGatewayAudioEngine(direct_config(), "voice-a")

        async def chunks():
            yield b"\x01\x00" * 160
            await asyncio.wait_for(audio_gate.wait(), 1)
            yield b"\x02\x00" * 160

        try:
            transcript = await engine.transcribe_stream(
                chunks(), on_interim=lambda sequence, text: interims.append((sequence, text))
            )
        finally:
            await engine.aclose()

    assert transcript == "saldo disponível agora"
    assert seen["authorization"] == "Token deepgram-key"
    assert len(seen["chunks"]) == 2
    assert "interim_results=true" in seen["path"]
    assert interims == [(700, "par")]
