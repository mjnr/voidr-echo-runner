import json

import httpx
import pytest

from echo_media_gateway.providers import ProviderConfigurationError, VoiceProviders

TTS_ALIAS = "echo-tts-elevenlabs-flash-v2-5@id:2026-07-26"
STT_ALIAS = "echo-stt-deepgram-nova-2@id:2026-07-26"


@pytest.fixture(autouse=True)
def governed_test_hosts(monkeypatch):
    monkeypatch.setenv(
        "AI_EGRESS_HOST_ALLOWLIST",
        "litellm.internal,litellm.example,*.test,llm.voidr.co",
    )


def test_provider_readiness_accepts_only_fully_configured_litellm():
    providers = VoiceProviders(
        litellm_url="https://litellm.test",
        litellm_key="litellm",
        tts_alias=TTS_ALIAS,
        stt_alias=STT_ALIAS,
        require_tls=True,
    )
    assert providers.readiness({"litellm"}) == (True, None)
    assert providers.readiness({"deepgram"}) == (False, "unknown_provider_route")
    assert VoiceProviders(litellm_url="https://litellm.test").readiness(
        {"litellm"}
    ) == (False, "provider_not_configured")


@pytest.mark.parametrize(
    "url",
    [
        "http://litellm.internal",
        "https://user:secret@litellm.internal",
        "https://169.254.169.254/latest/meta-data",
    ],
)
def test_governed_gateway_rejects_tls_downgrade_userinfo_and_ssrf(url):
    with pytest.raises(ProviderConfigurationError):
        VoiceProviders(litellm_url=url, require_tls=True)


@pytest.mark.asyncio
async def test_litellm_tts_negotiates_pcm_16k_and_forwards_tags():
    captured = {}

    async def handler(request):
        captured["body"] = json.loads((await request.aread()).decode())
        captured["tags"] = request.headers["x-litellm-tags"]
        return httpx.Response(
            200,
            headers={"content-type": "audio/pcm; rate=16000"},
            content=b"\x01\x00" * 160,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        providers = VoiceProviders(
            litellm_url="https://litellm.test",
            litellm_key="virtual",
            tts_alias=TTS_ALIAS,
            stt_alias=STT_ALIAS,
            http_client=client,
            require_tls=True,
        )
        audio = b"".join(
            [
                chunk
                async for chunk in providers.litellm(
                    text="teste",
                    voice="voice",
                    model=TTS_ALIAS,
                    output_format="pcm_16000",
                    tags={"org": "org-a", "modality": "tts"},
                )
            ]
        )

    assert audio == b"\x01\x00" * 160
    assert captured["body"]["output_format"] == "pcm_16000"
    assert "response_format" not in captured["body"]
    assert "sample_rate" not in captured["body"]
    assert "org:org-a" in captured["tags"]


@pytest.mark.asyncio
async def test_litellm_transcription_uses_documented_multipart_endpoint():
    captured = {}

    async def handler(request):
        captured["path"] = request.url.path
        captured["content_type"] = request.headers["content-type"]
        captured["body"] = await request.aread()
        return httpx.Response(200, json={"text": "texto sintético"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        providers = VoiceProviders(
            litellm_url="https://litellm.test",
            litellm_key="virtual",
            tts_alias=TTS_ALIAS,
            stt_alias=STT_ALIAS,
            http_client=client,
        )
        text = await providers.transcribe(
            pcm=b"\x01\x00" * 320,
            model=STT_ALIAS,
            sample_rate=16_000,
            language="pt",
            tags={"modality": "stt"},
        )

    assert text == "texto sintético"
    assert captured["path"] == "/v1/audio/transcriptions"
    assert captured["content_type"].startswith("multipart/form-data; boundary=")
    assert STT_ALIAS.encode() in captured["body"]


@pytest.mark.asyncio
async def test_litellm_rejects_ambiguous_pcm_rate():
    providers = VoiceProviders(litellm_url="https://litellm.test")
    with pytest.raises(ProviderConfigurationError, match="pcm_16000"):
        async for _ in providers.litellm(
            text="teste",
            voice="voice",
            model=TTS_ALIAS,
            output_format="pcm",
        ):
            pass


@pytest.mark.asyncio
async def test_litellm_rejects_pcm_without_sample_rate_metadata():
    async def handler(_request):
        return httpx.Response(
            200,
            headers={"content-type": "audio/pcm"},
            content=b"\x01\x00" * 160,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        providers = VoiceProviders(
            litellm_url="https://litellm.test",
            litellm_key="virtual",
            tts_alias=TTS_ALIAS,
            stt_alias=STT_ALIAS,
            http_client=client,
        )
        with pytest.raises(ProviderConfigurationError, match="sample_rate_missing"):
            _ = [
                chunk
                async for chunk in providers.litellm(
                    text="teste",
                    voice="voice",
                    model=TTS_ALIAS,
                    output_format="pcm_16000",
                )
            ]
