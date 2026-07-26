import asyncio
import base64
import contextlib
import hashlib
import hmac
import json

import websockets

from echo_media_gateway.auth import CapabilityVerifier
from echo_media_gateway.observability import VoiceObservability
from echo_media_gateway.server import run_gateway

SECRET = "unit-test-signing-secret-at-least-32-bytes"
PORT = 19021
TTS_ALIAS = "echo-tts-elevenlabs-flash-v2-5@id:2026-07-26"
STT_ALIAS = "echo-stt-deepgram-nova-2@id:2026-07-26"


def capability(jti: str) -> str:
    payload = {
        "org": "org-a",
        "execution": "exec-a",
        "shard": "2",
        "providers": ["litellm"],
        "models": {"litellm": [TTS_ALIAS, STT_ALIAS]},
        "voices": {"litellm": ["voice-test"]},
        "iat": 1_000,
        "exp": 1_300,
        "jti": jti,
        "max_requests": 20,
    }

    def enc(value):
        return (
            base64.urlsafe_b64encode(
                json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
            )
            .rstrip(b"=")
            .decode()
        )

    unsigned = f'{enc({"alg": "HS256", "typ": "VOICE"})}.{enc(payload)}'
    signature = hmac.new(SECRET.encode(), unsigned.encode(), hashlib.sha256).digest()
    return f"{unsigned}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"


class MockProviders:
    tts_alias = TTS_ALIAS
    stt_alias = STT_ALIAS

    def __init__(self):
        self.requests = []

    async def litellm(self, **request):
        self.requests.append(("tts", request))
        yield b"\x01\x00" * 160
        yield b"\x02\x00" * 160

    async def transcribe(self, **request):
        self.requests.append(("stt", request))
        return "texto sintético"


@contextlib.asynccontextmanager
async def gateway():
    ready = asyncio.Event()
    providers = MockProviders()
    logs = []
    task = asyncio.create_task(
        run_gateway(
            host="127.0.0.1",
            port=PORT,
            verifier=CapabilityVerifier(SECRET, clock=lambda: 1_100),
            providers=providers,
            observability=VoiceObservability(logs.append),
            ready=ready,
        )
    )
    await asyncio.wait_for(ready.wait(), 5)
    try:
        yield f"ws://127.0.0.1:{PORT}", providers, logs
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def headers(request_id: str, jti: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {capability(jti)}",
        "X-Voice-Request-Id": request_id,
    }


def test_metrics_exclude_high_cardinality_scope_and_audio():
    logs = []
    observability = VoiceObservability(logs.append)
    observability.audit(
        "voice_stt",
        {
            "org": "org-a",
            "execution": "exec-a",
            "shard": "2",
            "provider": "litellm",
            "model": STT_ALIAS,
            "modality": "stt",
        },
        status="ok",
        duration_ms=12,
    )
    metrics = observability.prometheus()
    assert 'event="voice_stt",provider="litellm",status="ok"' in metrics
    assert "org-a" not in metrics
    assert "audio" not in logs[0]
    assert "transcript" not in logs[0]


async def test_stt_buffers_utterance_and_calls_only_litellm():
    async with gateway() as (base, providers, logs):
        async with websockets.connect(
            f"{base}/v1/stt/litellm?model={STT_ALIAS}",
            additional_headers=headers("stt-1", "stt-token"),
        ) as ws:
            await ws.send(b"\x01\x00" * 320)
            await ws.send(json.dumps({"type": "CloseStream"}))
            result = json.loads(await asyncio.wait_for(ws.recv(), 5))
        assert result == {
            "type": "transcription",
            "sequence": 0,
            "is_final": True,
            "text": "texto sintético",
        }
        assert providers.requests[0][0] == "stt"
        assert providers.requests[0][1]["model"] == STT_ALIAS
        assert "texto sintético" not in "\n".join(logs)


async def test_tts_streams_litellm_chunks_without_logging_text():
    async with gateway() as (base, providers, logs):
        async with websockets.connect(
            f"{base}/v1/tts/litellm?model={TTS_ALIAS}",
            additional_headers=headers("tts-1", "tts-token"),
        ) as ws:
            await ws.send(
                json.dumps(
                    {
                        "text": "sensitive phrase must never be logged",
                        "voice": "voice-test",
                        "output_format": "pcm_16000",
                    }
                )
            )
            messages = [message async for message in ws]
        assert len([item for item in messages if isinstance(item, bytes)]) == 2
        assert json.loads(messages[-1]) == {"type": "end", "chunks": 2}
        assert providers.requests[0][0] == "tts"
        assert providers.requests[0][1]["model"] == TTS_ALIAS
        assert "sensitive phrase" not in "\n".join(logs)
