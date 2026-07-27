import json

import httpx
import pytest

from echo_media_gateway.providers import ProviderConfigurationError, VoiceProviders


def test_provider_readiness_requires_both_direct_credentials():
    providers = VoiceProviders(elevenlabs_key="elevenlabs", deepgram_key="deepgram")
    assert providers.readiness({"deepgram", "elevenlabs"}) == (True, None)
    assert providers.readiness({"litellm"}) == (False, "unknown_provider_route")
    assert VoiceProviders(elevenlabs_key="elevenlabs").readiness(
        {"deepgram", "elevenlabs"}
    ) == (False, "provider_not_configured")


@pytest.mark.asyncio
async def test_elevenlabs_tts_negotiates_direct_pcm16k():
    captured = {}

    async def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads((await request.aread()).decode())
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, content=b"\x01\x00" * 160)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        providers = VoiceProviders(
            elevenlabs_key="elevenlabs",
            deepgram_key="deepgram",
            http_client=client,
        )
        audio = b"".join(
            [
                chunk
                async for chunk in providers.elevenlabs(
                    text="teste",
                    voice="voice",
                    model="eleven_flash_v2_5",
                    output_format="pcm_16000",
                )
            ]
        )

    assert audio == b"\x01\x00" * 160
    assert "api.elevenlabs.io/v1/text-to-speech/voice" in captured["url"]
    assert "output_format=pcm_16000" in captured["url"]
    assert captured["headers"]["xi-api-key"] == "elevenlabs"
    assert captured["body"]["model_id"] == "eleven_flash_v2_5"


@pytest.mark.asyncio
async def test_deepgram_transcription_uses_direct_listen_endpoint():
    captured = {}

    async def handler(request):
        captured["url"] = str(request.url)
        captured["content_type"] = request.headers["content-type"]
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = await request.aread()
        return httpx.Response(
            200,
            json={
                "results": {
                    "channels": [{"alternatives": [{"transcript": "texto sintético"}]}]
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        providers = VoiceProviders(
            elevenlabs_key="elevenlabs",
            deepgram_key="deepgram",
            http_client=client,
        )
        text = await providers.transcribe(
            pcm=b"\x01\x00" * 320,
            model="nova-2",
            sample_rate=16_000,
            language="pt-BR",
        )

    assert text == "texto sintético"
    assert captured["url"].startswith("https://api.deepgram.com/v1/listen?")
    assert captured["content_type"] == "audio/wav"
    assert captured["authorization"] == "Token deepgram"
    assert captured["body"].startswith(b"RIFF")


@pytest.mark.asyncio
async def test_elevenlabs_rejects_non_pcm16k_output():
    providers = VoiceProviders(elevenlabs_key="elevenlabs", deepgram_key="deepgram")
    with pytest.raises(ProviderConfigurationError, match="pcm_16000"):
        async for _ in providers.elevenlabs(
            text="teste",
            voice="voice",
            model="eleven_flash_v2_5",
            output_format="pcm",
        ):
            pass
