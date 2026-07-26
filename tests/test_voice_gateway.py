import asyncio
import math
from array import array

import httpx
import pytest
import websockets

import voidr_echo_runner.voice_gateway as voice_module
from voidr_echo_runner.voice_gateway import (
    STT_ALIAS_DEFAULT,
    TTS_ALIAS_DEFAULT,
    VoiceConfig,
    VoiceGatewayAudioEngine,
    assert_direct_provider_access_allowed,
    merge_overlapping_transcripts,
    resample_pcm16,
    resolve_voice_config,
)


VOICE_ENV = (
    "ECHO_RUNTIME_ENV",
    "VOICE_ALLOW_DIRECT_PROVIDERS",
    "LITELLM_BASE_URL",
    "LITELLM_API_KEY",
    "LITELLM_TTS_MODEL",
    "LITELLM_STT_MODEL",
    "DEEPGRAM_API_KEY",
    "ELEVENLABS_API_KEY",
)


def clear_voice_env(monkeypatch):
    for name in VOICE_ENV:
        monkeypatch.delenv(name, raising=False)


def governed_config() -> VoiceConfig:
    return VoiceConfig(
        "test",
        "http://litellm.test",
        "org-virtual-key",
        TTS_ALIAS_DEFAULT,
        STT_ALIAS_DEFAULT,
        False,
    )


def test_production_requires_litellm_url_key_and_pinned_aliases(monkeypatch):
    clear_voice_env(monkeypatch)
    monkeypatch.setenv("ECHO_RUNTIME_ENV", "production")
    with pytest.raises(RuntimeError, match="LITELLM_BASE_URL"):
        resolve_voice_config()
    for name, value in {
        "LITELLM_BASE_URL": "https://llm.voidr.test",
        "LITELLM_API_KEY": "org-key",
        "LITELLM_TTS_MODEL": TTS_ALIAS_DEFAULT,
        "LITELLM_STT_MODEL": STT_ALIAS_DEFAULT,
    }.items():
        monkeypatch.setenv(name, value)
    config = resolve_voice_config()
    assert config.tts_alias == TTS_ALIAS_DEFAULT
    assert config.stt_alias == STT_ALIAS_DEFAULT


def test_production_rejects_direct_keys_and_tls_downgrade(monkeypatch):
    clear_voice_env(monkeypatch)
    monkeypatch.setenv("ECHO_RUNTIME_ENV", "production")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://llm.invalid")
    monkeypatch.setenv("LITELLM_API_KEY", "org-key")
    monkeypatch.setenv("LITELLM_TTS_MODEL", TTS_ALIAS_DEFAULT)
    monkeypatch.setenv("LITELLM_STT_MODEL", STT_ALIAS_DEFAULT)
    with pytest.raises(RuntimeError, match="must use https"):
        resolve_voice_config()
    monkeypatch.setenv("LITELLM_BASE_URL", "https://llm.invalid")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "forbidden")
    with pytest.raises(RuntimeError, match="must not be injected"):
        resolve_voice_config()


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@llm.voidr.co",
        "https://169.254.169.254/latest/meta-data",
        "https://llm.voidr.co.attacker.example",
    ],
)
def test_production_rejects_userinfo_and_ssrf_hosts(monkeypatch, url):
    clear_voice_env(monkeypatch)
    monkeypatch.setenv("ECHO_RUNTIME_ENV", "production")
    monkeypatch.setenv("LITELLM_BASE_URL", url)
    monkeypatch.setenv("LITELLM_API_KEY", "org-key")
    monkeypatch.setenv("LITELLM_TTS_MODEL", TTS_ALIAS_DEFAULT)
    monkeypatch.setenv("LITELLM_STT_MODEL", STT_ALIAS_DEFAULT)
    with pytest.raises(RuntimeError):
        resolve_voice_config()


def test_direct_provider_access_is_explicitly_test_only(monkeypatch):
    clear_voice_env(monkeypatch)
    monkeypatch.setenv("ECHO_RUNTIME_ENV", "local")
    with pytest.raises(RuntimeError, match="LITELLM"):
        resolve_voice_config()
    monkeypatch.setenv("VOICE_ALLOW_DIRECT_PROVIDERS", "1")
    assert resolve_voice_config().direct is True
    assert_direct_provider_access_allowed()


@pytest.mark.asyncio
async def test_tts_streams_litellm_pcm_and_sends_cost_tags(monkeypatch):
    samples = array(
        "h",
        [round(4_000 * math.sin(2 * math.pi * 440 * index / 16_000)) for index in range(1600)],
    )
    captured = {}

    async def handler(request):
        captured["authorization"] = request.headers["authorization"]
        captured["tags"] = request.headers["x-litellm-tags"]
        captured["body"] = (await request.aread()).decode()
        return httpx.Response(
            200,
            headers={"content-type": "audio/pcm; rate=16000"},
            content=samples.tobytes(),
        )

    monkeypatch.setenv("VOIDR_ORGANIZATION_ID", "org-a")
    monkeypatch.setenv("VOIDR_EXECUTION_ID", "exec-a")
    monkeypatch.setenv("SHARDS_CURRENT", "2")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = VoiceGatewayAudioEngine(governed_config(), "voice-a", http_client=client)
        chunks = [chunk async for chunk in engine.synthesize_chunks("olá")]

    assert b"".join(chunks) == samples.tobytes()
    assert captured["authorization"] == "Bearer org-virtual-key"
    assert "org:org-a" in captured["tags"]
    assert "modality:tts" in captured["tags"]
    assert TTS_ALIAS_DEFAULT in captured["body"]
    assert '"output_format":"pcm_16000"' in captured["body"]


@pytest.mark.asyncio
async def test_tts_rejects_pcm_without_real_rate_metadata():
    samples = array("h", [100] * 1600).tobytes()

    async def handler(_request):
        return httpx.Response(200, headers={"content-type": "audio/pcm"}, content=samples)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = VoiceGatewayAudioEngine(governed_config(), "voice-a", http_client=client)
        with pytest.raises(RuntimeError, match="omitted a valid PCM sample rate"):
            _ = [chunk async for chunk in engine.synthesize_chunks("olá")]


@pytest.mark.asyncio
async def test_tts_resamples_header_declared_pcm_before_exposing_16khz():
    source_rate = 24_000
    duration_s = 0.5
    source = array(
        "h",
        [
            round(4_000 * math.sin(2 * math.pi * 440 * index / source_rate))
            for index in range(round(source_rate * duration_s))
        ],
    )

    async def handler(_request):
        return httpx.Response(
            200,
            headers={"x-audio-sample-rate": str(source_rate), "content-type": "audio/pcm"},
            content=source.tobytes(),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = VoiceGatewayAudioEngine(governed_config(), "voice-a", http_client=client)
        pcm = b"".join([chunk async for chunk in engine.synthesize_chunks("olá")])

    assert len(pcm) // 2 == round(16_000 * duration_s)
    assert len(pcm) / 2 / engine.sample_rate == pytest.approx(duration_s, abs=1 / 16_000)


@pytest.mark.asyncio
async def test_tts_wire_fixture_validates_request_headers_frames_rate_and_duration():
    source_rate = 24_000
    duration_s = 0.25
    fixture = array(
        "h",
        [
            round(5_000 * math.sin(2 * math.pi * 523.25 * index / source_rate))
            for index in range(round(source_rate * duration_s))
        ],
    ).tobytes()
    seen: dict[str, str] = {}

    async def wire_server(reader, writer):
        headers = await reader.readuntil(b"\r\n\r\n")
        header_text = headers.decode()
        content_length = int(
            next(
                line.split(":", 1)[1].strip()
                for line in header_text.splitlines()
                if line.lower().startswith("content-length:")
            )
        )
        body = await reader.readexactly(content_length)
        seen["request"] = header_text
        seen["body"] = body.decode()
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: audio/pcm\r\n"
            b"X-Audio-Sample-Rate: 24000\r\n"
            + f"Content-Length: {len(fixture)}\r\n".encode()
            + b"Connection: close\r\n\r\n"
            + fixture
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(wire_server, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    config = VoiceConfig(
        "test",
        f"http://127.0.0.1:{port}",
        "org-virtual-key",
        TTS_ALIAS_DEFAULT,
        STT_ALIAS_DEFAULT,
        False,
    )
    engine = VoiceGatewayAudioEngine(config, "voice-a")
    try:
        pcm = b"".join([chunk async for chunk in engine.synthesize_chunks("olá")])
    finally:
        await engine.aclose()
        server.close()
        await server.wait_closed()

    assert "Authorization: Bearer org-virtual-key" in seen["request"]
    assert '"output_format":"pcm_16000"' in seen["body"]
    assert len(pcm) // 2 == 4_000
    assert len(pcm) / 2 / engine.sample_rate == pytest.approx(duration_s)


@pytest.mark.asyncio
async def test_stt_pipeline_preserves_order_with_overlapping_async_segments(monkeypatch):
    engine = VoiceGatewayAudioEngine(
        governed_config(),
        "voice-a",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: None)),
    )
    monkeypatch.setattr(voice_module, "segment_utterance", lambda *_: [b"a" * 320, b"b" * 320])

    async def fake_transcribe(segment, index):
        await asyncio.sleep(0.02 if index == 0 else 0)
        return ["saldo disponível", "disponível agora"][index]

    monkeypatch.setattr(engine, "_transcribe_segment", fake_transcribe)
    try:
        assert await engine.transcribe(b"\x01\x00" * 320) == "saldo disponível agora"
    finally:
        await engine._http.aclose()


def test_resampling_and_overlap_merge_are_deterministic():
    source = array("h", range(-1000, 1000)).tobytes()
    resampled = resample_pcm16(source, 8_000, 16_000)
    assert len(resampled) == len(source) * 2
    assert (
        merge_overlapping_transcripts(["um dois três", "dois três quatro"])
        == "um dois três quatro"
    )


@pytest.mark.asyncio
async def test_streaming_stt_wire_protocol_overlaps_audio_and_orders_finals(monkeypatch):
    seen: dict[str, object] = {"chunks": []}
    audio_gate = asyncio.Event()

    async def upstream(ws):
        seen["authorization"] = ws.request.headers["authorization"]
        seen["path"] = ws.request.path
        first = await ws.recv()
        assert isinstance(first, bytes)
        seen["chunks"].append(first)
        audio_gate.set()
        await ws.send(
            '{"type":"transcription","sequence":7,"is_final":false,"text":"par"}'
        )
        second = await ws.recv()
        assert isinstance(second, bytes)
        seen["chunks"].append(second)
        assert await ws.recv() == '{"type": "CloseStream"}'
        await ws.send(
            '{"type":"transcription","sequence":1,"is_final":true,"text":"disponível agora"}'
        )
        await ws.send(
            '{"type":"transcription","sequence":0,"is_final":true,"text":"saldo disponível"}'
        )

    async with websockets.serve(upstream, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv(
            "LITELLM_STREAMING_STT_URL",
            f"ws://127.0.0.1:{port}/v1/audio/transcriptions/stream",
        )
        engine = VoiceGatewayAudioEngine(governed_config(), "voice-a")

        async def chunks():
            yield b"\x01\x00" * 160
            await asyncio.wait_for(audio_gate.wait(), 1)
            yield b"\x02\x00" * 160

        try:
            transcript = await engine.transcribe_stream(chunks())
        finally:
            await engine.aclose()

    assert transcript == "saldo disponível agora"
    assert seen["authorization"] == "Bearer org-virtual-key"
    assert len(seen["chunks"]) == 2
    assert "interim_results=true" in str(seen["path"])
