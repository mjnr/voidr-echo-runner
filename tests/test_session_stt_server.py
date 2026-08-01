import http.client
from http.server import ThreadingHTTPServer
import json
import threading

import pytest

import voidr_echo_runner.session_stt_server as session_stt
from voidr_echo_runner.session_stt_server import authorized


@pytest.fixture
def stt_server(monkeypatch):
    secret = "s" * 32
    monkeypatch.setenv("SESSION_STT_SHARED_SECRET", secret)
    server = ThreadingHTTPServer(("127.0.0.1", 0), session_stt.SessionSttHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], secret
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(port: int, body: bytes, secret: str | None, content_type: str):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    headers = {"Content-Type": content_type}
    if secret is not None:
        headers["Authorization"] = f"Bearer {secret}"
    connection.request("POST", "/v1/transcribe", body=body, headers=headers)
    response = connection.getresponse()
    payload = json.loads(response.read())
    connection.close()
    return response.status, payload


def test_session_stt_shared_secret_is_exact_and_never_partial():
    secret = "s" * 32
    assert authorized(f"Bearer {secret}", secret)
    assert not authorized(f"Bearer {secret[:-1]}", secret)
    assert not authorized(secret, secret)
    assert not authorized(None, secret)


def test_session_stt_rejects_unauthorized_and_unbounded_audio(stt_server):
    port, secret = stt_server
    status, payload = request(port, b"\0" * 320, None, "audio/pcm;rate=16000;channels=1")
    assert (status, payload) == (401, {"error": "unauthorized"})

    status, payload = request(port, b"\0" * 318, secret, "audio/pcm;rate=16000;channels=1")
    assert (status, payload) == (413, {"error": "audio_size_out_of_bounds"})

    status, payload = request(port, b"\0" * 320, secret, "audio/wav")
    assert (status, payload) == (415, {"error": "unsupported_media_type"})


def test_session_stt_returns_only_governed_transcript_metadata(stt_server, monkeypatch):
    port, secret = stt_server

    async def fake_transcribe(pcm: bytes):
        assert pcm == b"\0" * 32_000
        return {
            "text": "O retry falhou uma vez.",
            "language": "pt-BR",
            "durationMs": 1_000,
            "provider": "deepgram",
            "model": "nova-2",
        }

    monkeypatch.setattr(session_stt, "transcribe_pcm", fake_transcribe)
    status, payload = request(
        port,
        b"\0" * 32_000,
        secret,
        "audio/pcm;rate=16000;channels=1",
    )
    assert status == 200
    assert payload == {
        "text": "O retry falhou uma vez.",
        "language": "pt-BR",
        "durationMs": 1_000,
        "provider": "deepgram",
        "model": "nova-2",
    }
