"""LivePublisher — contract, batching, pairing, deny-list and resilience."""

from __future__ import annotations

import asyncio
import base64
import io
import wave

import httpx
import pytest

from voidr_echo_runner.live_events import (
    BATCH_MAX_EVENTS,
    BREAKER_MAX_FAILURES,
    LivePublisher,
    pcm_to_wav_b64,
)
from voidr_echo_runner.redaction import RedactionSession

BASE = "http://svc.test"
EXEC_ID = "exec-42"


class FakeServer:
    """Captures every POSTed batch; scriptable status/errors per request."""

    def __init__(self):
        self.batches: list[dict] = []
        self.responses: list[int | Exception] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        outcome = self.responses.pop(0) if self.responses else 200
        if isinstance(outcome, Exception):
            raise outcome
        if outcome < 400:
            import json

            self.batches.append(json.loads(request.content))
        return httpx.Response(outcome, json={"success": outcome < 400})

    @property
    def events(self) -> list[dict]:
        return [e for b in self.batches for e in b["events"]]


def make_publisher(server: FakeServer, **kwargs) -> LivePublisher:
    transport = httpx.MockTransport(server.handler)
    return LivePublisher(
        BASE,
        EXEC_ID,
        1,
        client=httpx.AsyncClient(transport=transport),
        sync_client=httpx.Client(transport=transport),
        **kwargs,
    )


def run(coro):
    return asyncio.run(coro)


def drain(pub: LivePublisher):
    """start → (caller emitted) → stop, inside one loop."""

    async def _go():
        await pub.stop()

    return run(_go())


# ── contract URL and envelope ─────────────────────────────────────────────────


def test_url_and_envelope():
    server = FakeServer()
    pub = make_publisher(server)

    async def _go():
        await pub.start()
        pub.emit("state_transition", {"state": "saudacao", "turn": 1})
        await pub.stop()

    run(_go())
    assert pub.url == f"{BASE}/v1/echo/live/{EXEC_ID}/events"
    assert server.batches, "nothing was POSTed"
    body = server.batches[0]
    assert body["shardIndex"] == 1
    event = body["events"][-1]
    assert set(event) == {"seq", "tsMs", "type", "data"}
    assert isinstance(event["tsMs"], int) and event["tsMs"] >= 0


def test_batching_and_monotonic_seq():
    server = FakeServer()
    pub = make_publisher(server)

    async def _go():
        await pub.start()
        for i in range(12):
            pub.emit("state_transition", {"state": f"s{i}", "turn": i})
        await pub.stop()

    run(_go())
    assert all(len(b["events"]) <= BATCH_MAX_EVENTS for b in server.batches)
    seqs = [e["seq"] for e in server.events]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))
    # 12 + phase dialing = 13 events delivered
    assert len(seqs) == 13


def test_audio_events_capped_at_one_per_batch():
    server = FakeServer()
    pub = make_publisher(server)
    pcm = b"\x01\x00" * 160

    async def _go():
        await pub.start()
        for i in range(3):
            pub.on_transcript({"index": i, "speaker": "tester", "text": f"oi {i}", "ts": 0})
            pub.add_turn_audio("tester", pcm, 16000)
        await pub.stop()

    run(_go())
    for batch in server.batches:
        audio = [e for e in batch["events"] if e["type"] == "turn_audio"]
        assert len(audio) <= 1
    assert sum(1 for e in server.events if e["type"] == "turn_audio") == 3


# ── turn / turn_audio pairing ────────────────────────────────────────────────


def test_tester_audio_pairs_with_already_recorded_turn():
    server = FakeServer()
    pub = make_publisher(server)
    pcm = b"\x02\x00" * 320

    async def _go():
        await pub.start()
        # CallRunner records the tester turn BEFORE transport.send_text
        pub.on_transcript({"index": 3, "speaker": "tester", "text": "quero saldo", "ts": 0})
        pub.add_turn_audio("tester", pcm, 16000)
        await pub.stop()

    run(_go())
    audio = next(e for e in server.events if e["type"] == "turn_audio")
    assert audio["data"]["speaker"] == "tester"
    assert audio["data"]["turnIndex"] == 3
    assert audio["data"]["format"] == "wav"
    assert audio["data"]["sampleRate"] == 16000
    # valid mono 16k WAV with the exact PCM payload
    raw = base64.b64decode(audio["data"]["audioB64"])
    with wave.open(io.BytesIO(raw), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16000
        assert wav.getsampwidth() == 2
        assert wav.readframes(wav.getnframes()) == pcm


def test_agent_audio_waits_for_the_turn_event():
    server = FakeServer()
    pub = make_publisher(server)
    pcm = b"\x03\x00" * 160

    async def _go():
        await pub.start()
        # STT transcribes first; CallRunner records the agent turn after
        pub.add_turn_audio("agent", pcm, 16000)
        pub.on_transcript({"index": 5, "speaker": "agent", "text": "seu saldo é", "ts": 0})
        await pub.stop()

    run(_go())
    types = [e["type"] for e in server.events]
    turn_pos = types.index("turn")
    audio_pos = types.index("turn_audio")
    assert turn_pos < audio_pos, "turn must precede its turn_audio"
    audio = server.events[audio_pos]
    assert audio["data"]["turnIndex"] == 5
    assert audio["data"]["speaker"] == "agent"


def test_sensitive_audio_is_silenced_and_safe_speech_is_preserved():
    server = FakeServer()
    session = RedactionSession(deny={"CUSTOMER_CPF": "39053344705"})
    pub = make_publisher(
        server,
        redact=session.redact_deny,
        has_sensitive_data=lambda text: bool(session.find_spans(text)),
    )
    sensitive_pcm = b"\x11\x22" * 320
    safe_pcm = b"\x33\x44" * 320

    async def _go():
        await pub.start()
        pub.on_transcript(
            {
                "index": 1,
                "speaker": "tester",
                "text": "meu CPF é 390 533 447 05",
                "ts": 0,
            }
        )
        pub.add_turn_audio("tester", sensitive_pcm, 16000)
        pub.on_transcript(
            {"index": 2, "speaker": "tester", "text": "quero consultar o saldo", "ts": 0}
        )
        pub.add_turn_audio("tester", safe_pcm, 16000)
        await pub.stop()

    run(_go())
    audio_events = [e for e in server.events if e["type"] == "turn_audio"]

    def pcm(event):
        raw = base64.b64decode(event["data"]["audioB64"])
        with wave.open(io.BytesIO(raw), "rb") as wav:
            return wav.readframes(wav.getnframes())

    assert pcm(audio_events[0]) != sensitive_pcm
    assert pcm(audio_events[0]) == b"\x00" * len(sensitive_pcm)
    assert pcm(audio_events[1]) == safe_pcm


def test_audio_suppressed_when_disabled():
    server = FakeServer()
    pub = make_publisher(server, audio_enabled=False)

    async def _go():
        await pub.start()
        pub.on_transcript({"index": 0, "speaker": "tester", "text": "oi", "ts": 0})
        pub.add_turn_audio("tester", b"\x00\x00" * 100, 16000)
        await pub.stop()

    run(_go())
    assert not any(e["type"] == "turn_audio" for e in server.events)
    assert any(e["type"] == "turn" for e in server.events)


# ── deny-list on live text ────────────────────────────────────────────────────


def test_turn_text_passes_massa_deny_list():
    server = FakeServer()
    session = RedactionSession(deny={"MOCK_ACCESS_CODE": "919021552"})
    pub = make_publisher(server, redact=session.redact_deny)

    async def _go():
        await pub.start()
        pub.on_transcript(
            {"index": 0, "speaker": "tester", "text": "o código é 919021552, anota", "ts": 0}
        )
        await pub.stop()

    run(_go())
    turn = next(e for e in server.events if e["type"] == "turn")
    assert "919021552" not in turn["data"]["text"]
    assert "[MASSA_MOCK_ACCESS_CODE]" in turn["data"]["text"]


def test_turn_text_redacts_generic_cpf_email_phone_and_silences_audio():
    server = FakeServer()
    session = RedactionSession()
    pub = make_publisher(
        server,
        redact=session.redact,
        has_sensitive_data=lambda text: bool(session.find_spans(text)),
    )
    clear = (
        "CPF 390.533.447-05, email marcia.real@example.com, "
        "telefone (31) 98888-7777"
    )
    pcm = b"\x21\x43" * 160

    async def _go():
        await pub.start()
        pub.on_transcript({"index": 0, "speaker": "tester", "text": clear, "ts": 0})
        pub.add_turn_audio("tester", pcm, 16000)
        await pub.stop()

    run(_go())
    turn = next(e for e in server.events if e["type"] == "turn")
    for sensitive in ("390.533.447-05", "marcia.real@example.com", "(31) 98888-7777"):
        assert sensitive not in turn["data"]["text"]
    assert "[CPF_1]" in turn["data"]["text"]
    assert "[EMAIL_1]" in turn["data"]["text"]
    assert "[TELEFONE_1]" in turn["data"]["text"]

    audio = next(e for e in server.events if e["type"] == "turn_audio")
    raw = base64.b64decode(audio["data"]["audioB64"])
    with wave.open(io.BytesIO(raw), "rb") as wav:
        assert wav.readframes(wav.getnframes()) == b"\x00" * len(pcm)


def test_dtmf_digits_pass_deny_list():
    server = FakeServer()
    session = RedactionSession(deny={"MOCK_ACCESS_CODE": "919021552"})
    pub = make_publisher(server, redact=session.redact_deny)

    async def _go():
        await pub.start()
        pub.on_timeline_event("dtmf_sent", {"digits": "919021552"})
        await pub.stop()

    run(_go())
    dtmf = next(e for e in server.events if e["type"] == "dtmf_sent")
    assert "919021552" not in dtmf["data"]["digits"]


# ── event mapping and phases ─────────────────────────────────────────────────


def test_phase_sequence_and_call_ended():
    server = FakeServer()
    pub = make_publisher(server)

    async def _go():
        await pub.start()
        pub.on_transcript({"index": 0, "speaker": "ura", "text": "digite o código", "ts": 0})
        pub.on_timeline_event("dtmf_sent", {"digits": "123"})
        pub.on_transcript({"index": 1, "speaker": "agent", "text": "bom dia", "ts": 0})
        pub.on_transcript({"index": 2, "speaker": "agent", "text": "posso ajudar?", "ts": 0})
        await pub.stop()

    run(_go())
    pub.finish_sync("agent_hangup", "passed")
    phases = [e["data"]["phase"] for e in server.events if e["type"] == "phase"]
    assert phases == ["dialing", "ura", "agent", "ended"]
    ended = next(e for e in server.events if e["type"] == "call_ended")
    assert ended["data"] == {"reason": "agent_hangup", "status": "passed"}


def test_emotional_and_state_mapping():
    server = FakeServer()
    pub = make_publisher(server)

    async def _go():
        await pub.start()
        pub.on_timeline_event("state_transition", {"state": "identificacao", "turn": 2})
        pub.on_timeline_event(
            "emotional_state",
            {"turn": 2, "emotion": "ansioso", "intensity": 0.4, "events": ["x"]},
        )
        pub.on_timeline_event(
            "emotional_state",
            {"turn": 3, "emotion": "irritado", "intensity": 0.8, "events": [], "action": "pedir_humano"},
        )
        # internal-only types must not leak
        pub.on_timeline_event("connected", {"target": "ws://x"})
        pub.on_timeline_event("tester_turn", {"turn": 2})
        await pub.stop()

    run(_go())
    types = {e["type"] for e in server.events}
    assert types == {"phase", "state_transition", "emotional_state"}
    st = next(e for e in server.events if e["type"] == "state_transition")
    assert st["data"] == {"state": "identificacao", "turn": 2}
    emos = [e["data"] for e in server.events if e["type"] == "emotional_state"]
    assert emos[0] == {"emotion": "ansioso", "intensity": 0.4, "action": None}
    assert emos[1]["action"] == "pedir_humano"


# ── resilience ────────────────────────────────────────────────────────────────


def test_network_failures_never_raise_and_trip_the_breaker(capsys):
    server = FakeServer()
    # every attempt fails (2 retries per batch) — enough for the breaker
    server.responses = [httpx.ConnectError("boom")] * 100
    pub = make_publisher(server)

    async def _go():
        await pub.start()
        for i in range(BREAKER_MAX_FAILURES * BATCH_MAX_EVENTS + 5):
            pub.emit("state_transition", {"state": f"s{i}", "turn": i})
            await asyncio.sleep(0)  # let the flusher interleave
        await pub.stop()

    run(_go())  # must not raise
    assert pub.disabled
    assert "live batch dropped" in capsys.readouterr().err
    pub.emit("turn", {"speaker": "agent", "text": "late", "turnIndex": 9})
    assert not pub._queue  # emits after the trip are dropped silently


def test_404_disables_immediately(capsys):
    server = FakeServer()
    server.responses = [404]
    pub = make_publisher(server)

    async def _go():
        await pub.start()
        pub.emit("state_transition", {"state": "s", "turn": 1})
        await pub.stop()

    run(_go())
    assert pub.disabled
    assert "404" in capsys.readouterr().err
    # finish_sync respects the open breaker: no crash, nothing sent
    pub.finish_sync("connection_closed", "env_failure")
    assert server.batches == []


def test_finish_sync_swallows_network_errors():
    server = FakeServer()
    server.responses = [httpx.ConnectError("down")]
    pub = make_publisher(server)
    pub.finish_sync("agent_hangup", "passed")  # must not raise


# ── CallRunner integration (hooks fire through a real conversation) ──────────


def test_call_runner_hooks_feed_the_publisher(monkeypatch):
    monkeypatch.setenv("MOCK_ACCESS_CODE", "919021552")
    from pathlib import Path

    from voidr_echo_runner.flows import load_journey_flow
    from voidr_echo_runner.models import VoiceTestCase
    from voidr_echo_runner.runner import CallRunner

    repo = Path(__file__).resolve().parents[1]
    case = VoiceTestCase.load(repo / "cases" / "consulta-saldo-tc-001.yaml")
    flow = load_journey_flow(repo / "flows" / "consulta-saldo-v1.json")
    class FakeHiveBrain:
        def take_turn(self, text):
            return {
                "text": "Quero consultar meu saldo.",
                "source": "hive-llm",
                "turnId": "turn-test",
                "promptVersion": "test",
                "modelVersion": "test",
                "policyVersion": "test",
                "trace": {"promptHash": "test", "completionId": "test"},
            }

    brain = FakeHiveBrain()

    class ScriptedTransport:
        url = "ws://fake"

        def __init__(self):
            self.script = [
                {"type": "text", "speaker": "ura", "text": "digite o código de acesso"},
                {"type": "text", "speaker": "ura", "text": "digite o número da linha"},
                {"type": "text", "speaker": "agent", "text": "bom dia, aqui é da vivo"},
                {"type": "event", "name": "call_ended", "reason": "agent_hangup"},
            ]

        async def connect(self):
            pass

        async def send_text(self, text):
            pass

        async def send_dtmf(self, digits):
            pass

        async def hangup(self):
            pass

        async def receive(self, timeout):
            return self.script.pop(0) if self.script else None

    server = FakeServer()
    pub = make_publisher(server)

    async def _go():
        await pub.start()
        runner = CallRunner(case, flow, brain, ScriptedTransport(), live=pub)
        result = await runner.run()
        await pub.stop()
        return result

    run(_go())
    pub.finish_sync("agent_hangup", "passed")
    types = [e["type"] for e in server.events]
    assert "turn" in types and "dtmf_sent" in types and "call_ended" in types
    phases = [e["data"]["phase"] for e in server.events if e["type"] == "phase"]
    assert phases == ["dialing", "ura", "agent", "ended"]
    speakers = [e["data"]["speaker"] for e in server.events if e["type"] == "turn"]
    assert "ura" in speakers and "agent" in speakers and "tester" in speakers
    # Tester turns carry Hive provenance; remote turns keep the base shape.
    for e in server.events:
        if e["type"] == "turn":
            if e["data"]["speaker"] == "tester":
                assert e["data"]["source"] == "hive-llm"
                assert e["data"]["turnId"] == "turn-test"
            else:
                assert {"speaker", "text", "turnIndex"} <= set(e["data"])


def test_hive_failure_is_published_with_structured_outcome():
    server = FakeServer()
    pub = make_publisher(server)

    async def _go():
        await pub.start()
        pub.on_timeline_event(
            "hive_turn_failed",
            {
                "turnId": "turn-failed",
                "code": "UPSTREAM_UNAVAILABLE",
                "statusCode": 502,
                "outcome": "degraded",
            },
        )
        await pub.stop()

    run(_go())
    failure = next(e for e in server.events if e["type"] == "hive_generation_failed")
    assert failure["data"] == {
        "turnId": "turn-failed",
        "code": "UPSTREAM_UNAVAILABLE",
        "statusCode": 502,
        "outcome": "degraded",
    }
