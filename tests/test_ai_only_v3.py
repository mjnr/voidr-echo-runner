"""Contract proofs for the AI-only Hive persona-turn v3 runner."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from voidr_echo_runner.brain import HiveError, LLMBrain, build_brain
from voidr_echo_runner.flows import FlowState, JourneyFlow
from voidr_echo_runner.models import (
    GlossaryConfusedTerm,
    VoiceTestCase,
    load_persona_catalog,
)
from voidr_echo_runner.runner import CallRunner
from voidr_echo_runner.service_mode import (
    SessionVerdict,
    build_session_payload,
    classify_session,
)

ROOT = Path(__file__).resolve().parents[1]
CONVERSATION_ID = "d9267c63-0f0a-5a51-9b31-33fb85bbab03"
MODEL_REVISION = (
    "deepseek-v4-pro@sha256:"
    "59e858aa0bd9bdbc7524a5dd39d84904747dacd1f85d152d0c04bcc373db9a08"
)
MODEL_HASH = "bd00496aec074f5565909718c136fd1171e727b375b980798e0e4e00e8d67d5d"


@pytest.fixture
def hive_env(monkeypatch):
    monkeypatch.setenv("HIVE_URL", "http://hive.test:3001")
    monkeypatch.setenv("HIVE_GATEWAY_TOKEN", "test-token")
    monkeypatch.setenv("VOIDR_ORG_ID", "org-test")
    monkeypatch.setenv("HIVE_ECHO_PERSONA_V3_MODEL_REVISION", MODEL_REVISION)


@pytest.fixture
def persona():
    return load_persona_catalog(ROOT / "personas" / "catalog.yaml")[
        "dona-marcia-58-mineira"
    ].model_copy(deep=True)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _success(request: httpx.Request, text: str = "Quero consultar meu saldo.") -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "text": text,
            "provenance": {
                "source": "hive-llm",
                "contractVersion": "v3",
                "conversationId": body["conversationId"],
                "turnId": body["turnId"],
                "policyVersion": body["policyVersion"],
                "promptVersion": "echo-persona-system-v3.0.0",
                "promptHash": "sha256:prompt",
                "provider": "litellm",
                "modelAlias": "deepseek-v4-pro",
                "model": "deepseek-v4-pro",
                "modelResolved": MODEL_REVISION,
                "modelVersion": MODEL_REVISION,
                "deploymentPin": MODEL_REVISION,
                "deploymentId": MODEL_REVISION,
                "deploymentDigest": MODEL_REVISION.rsplit("@", 1)[1],
                "modelHash": MODEL_HASH,
                "completionId": "completion-1",
                "traceId": "trace-1",
                "generatedAt": "2026-07-26T00:00:00Z",
                "attempts": 1,
            },
            "usage": {"costUsd": 0.001},
        },
    )


def test_v3_success_is_pinned_stable_and_provenanced(hive_env, persona):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return _success(request)

    brain = LLMBrain(
        persona,
        "consultar saldo",
        seed=42,
        client=_client(handler),
        conversation_id=CONVERSATION_ID,
    )
    result = brain.take_turn("Como posso ajudar?")
    body = captured["body"]

    assert captured["url"].endswith("/echo/persona-turn/v3")
    assert body["conversationId"] == CONVERSATION_ID
    assert uuid.UUID(body["turnId"]).version == 5
    assert body["deadlineAt"].endswith("Z")
    assert body["policyVersion"] == "echo-persona-turn-v3.0.0"
    assert result["source"] == "hive-llm"
    assert result["turnId"] == body["turnId"]
    assert result["trace"]["promptHash"] == "sha256:prompt"
    assert set(result["trace"]) == {
        "contractVersion",
        "conversationId",
        "completionId",
        "traceId",
        "promptHash",
        "modelHash",
        "attempts",
        "durationMs",
    }

    twin = LLMBrain(
        persona,
        "consultar saldo",
        seed=42,
        client=_client(_success),
        conversation_id=CONVERSATION_ID,
    )
    assert twin.take_turn("Como posso ajudar?")["provenance"]["turnId"] == body["turnId"]


def test_ad_hoc_sessions_use_distinct_idempotency_namespaces(hive_env, persona):
    first = LLMBrain(persona, "consultar saldo", seed=42, client=_client(_success))
    second = LLMBrain(persona, "consultar saldo", seed=42, client=_client(_success))

    assert first.conversation_id != second.conversation_id


def test_v3_rejects_missing_model_version_provenance(hive_env, persona):
    def handler(request: httpx.Request) -> httpx.Response:
        response = _success(request)
        payload = response.json()
        del payload["provenance"]["modelVersion"]
        return httpx.Response(200, json=payload)

    brain = LLMBrain(
        persona,
        "consultar saldo",
        client=_client(handler),
        conversation_id=CONVERSATION_ID,
    )
    with pytest.raises(HiveError, match="modelVersion"):
        brain.take_turn("Como posso ajudar?")
    assert brain.history == []


def test_v3_redacts_pii_recursively_from_nested_persona_fields(hive_env, persona):
    persona.identity.facts["cpf"] = "390.533.447-05"
    persona.identity.backstory = "Contato antigo: marcia.real@example.com"
    persona.profile.context = "Telefone (31) 98888-7777"
    persona.speech.exemplars = ["Meu CPF é 390.533.447-05"]
    persona.vocabulary = ["marcia.real@example.com"]
    persona.glossaryVocabulary.confused.append(
        GlossaryConfusedTerm(term="franquia", trap="Ligue (31) 98888-7777")
    )
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["serialized"] = request.content.decode()
        return _success(request)

    brain = LLMBrain(
        persona,
        "consultar saldo",
        client=_client(handler),
        conversation_id=CONVERSATION_ID,
    )
    brain.take_turn("Como posso ajudar?")

    serialized = captured["serialized"]
    for cleartext in (
        "390.533.447-05",
        "marcia.real@example.com",
        "(31) 98888-7777",
    ):
        assert cleartext not in serialized
    assert "[CPF_1]" in serialized
    assert "[EMAIL_1]" in serialized
    assert "[TELEFONE_1]" in serialized


def test_missing_hive_is_a_hard_failure(monkeypatch, persona):
    for key in LLMBrain.REQUIRED_ENV:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="Hive env vars"):
        build_brain(persona, "consultar saldo", 42)


def test_mutable_model_alias_is_not_accepted_as_revision(hive_env, monkeypatch, persona):
    monkeypatch.setenv("HIVE_ECHO_PERSONA_V3_MODEL_REVISION", "deepseek-v4-pro")
    with pytest.raises(RuntimeError, match="immutable"):
        build_brain(persona, "consultar saldo", 42)


@pytest.mark.parametrize(
    "mutable_pin",
    [
        "deepseek-v4-pro@latest",
        "deepseek-v4-pro@prod",
        "deepseek-v4-pro@2026-07-25.1",
        "deepseek-v4-pro@stable",
    ],
)
def test_mutable_revision_labels_are_rejected(
    hive_env, monkeypatch, persona, mutable_pin
):
    monkeypatch.setenv("HIVE_ECHO_PERSONA_V3_MODEL_REVISION", mutable_pin)
    with pytest.raises(RuntimeError, match="immutable"):
        build_brain(persona, "consultar saldo", 42)


def test_duplicate_turn_id_is_not_downgraded_or_retried(hive_env, persona):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(
            409,
            json={
                "error": {
                    "code": "duplicate_turn_id",
                    "message": "turn already completed",
                }
            },
        )

    brain = LLMBrain(persona, "g", client=_client(handler), conversation_id=CONVERSATION_ID)
    turn_id = "8bbedf5a-e2c0-54f4-bb4d-99b7845a2f5a"
    with pytest.raises(HiveError, match="duplicate_turn_id") as caught:
        brain.take_turn("oi", turn_id=turn_id)
    assert caught.value.outcome == "failed"
    assert len(calls) == 1
    assert calls[0]["turnId"] == turn_id
    assert brain.history == []


@pytest.mark.parametrize(
    ("response", "outcome"),
    [
        (
            httpx.Response(
                429,
                json={
                    "error": {
                        "code": "upstream_rate_limited",
                        "message": "capacity exhausted",
                    }
                },
            ),
            "inconclusive",
        ),
        (
            httpx.Response(
                502,
                json={
                    "error": {
                        "code": "upstream_unavailable",
                        "message": "provider timeout",
                    }
                },
            ),
            "degraded",
        ),
    ],
)
def test_structured_hive_errors_are_single_attempt(hive_env, persona, response, outcome):
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return response

    brain = LLMBrain(persona, "g", client=_client(handler), conversation_id=CONVERSATION_ID)
    with pytest.raises(HiveError) as caught:
        brain.take_turn("oi")
    assert caught.value.outcome == outcome
    assert calls == 1
    assert brain.history == []


def test_timeout_is_single_attempt_and_degraded(hive_env, persona):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("deadline", request=request)

    brain = LLMBrain(persona, "g", client=_client(handler), conversation_id=CONVERSATION_ID)
    with pytest.raises(HiveError) as caught:
        brain.take_turn("oi")
    assert caught.value.outcome == "degraded"
    assert calls == 1


def test_hive_deadline_precedes_http_timeout_and_is_classified(
    hive_env, monkeypatch, persona
):
    captured = {}

    class RecordingClient:
        def __init__(self, *, timeout, headers):
            captured["timeout"] = timeout
            captured["headers"] = headers

        def post(self, url, *, json):
            captured["url"] = url
            captured["body"] = json
            return httpx.Response(
                504,
                json={
                    "error": {
                        "code": "DEADLINE_EXCEEDED",
                        "message": "persona turn exceeded its logical deadline",
                    }
                },
            )

    monkeypatch.setattr(httpx, "Client", RecordingClient)
    started_at = datetime.now(timezone.utc)
    brain = LLMBrain(persona, "g", conversation_id=CONVERSATION_ID)

    with pytest.raises(HiveError) as caught:
        brain.take_turn("oi")

    deadline_at = datetime.fromisoformat(
        captured["body"]["deadlineAt"].replace("Z", "+00:00")
    )
    logical_deadline_s = (deadline_at - started_at).total_seconds()
    assert LLMBrain.DEADLINE_S == 20.0
    assert 19.5 <= logical_deadline_s <= 20.5
    assert captured["timeout"] == LLMBrain.HTTP_TIMEOUT_S == 25.0
    assert LLMBrain.HTTP_TIMEOUT_S - LLMBrain.DEADLINE_S == 5.0
    assert caught.value.status_code == 504
    assert caught.value.code == "DEADLINE_EXCEEDED"
    assert caught.value.outcome == "inconclusive"


def test_timeout_produces_no_tester_turn_or_tts(hive_env, persona):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("deadline", request=request)

    brain = LLMBrain(
        persona,
        "g",
        client=_client(handler),
        conversation_id=CONVERSATION_ID,
    )
    transport = _Transport()
    case, flow = _case_and_flow()
    call = asyncio.run(CallRunner(case, flow, brain, transport).run())

    assert call.failure_status == "degraded"
    assert transport.sent_text == []
    assert not any(t["speaker"] == "tester" for t in call.transcript)
    assert not any(e["type"] == "tester_turn" for e in call.timeline)


class _Transport:
    def __init__(self):
        self.messages = [{"type": "text", "speaker": "agent", "text": "Como posso ajudar?"}]
        self.sent_text: list[str] = []
        self.hangups = 0

    async def connect(self):
        pass

    async def receive(self, timeout=None):
        return self.messages.pop(0) if self.messages else None

    async def send_text(self, text):
        self.sent_text.append(text)

    async def send_dtmf(self, digits):
        pass

    async def hangup(self):
        self.hangups += 1


def _case_and_flow():
    case = VoiceTestCase.model_validate(
        {
            "id": "ai-only",
            "persona": {"base": "p"},
            "journey_flow": "unused",
            "goal": "consultar saldo",
        }
    )
    flow = JourneyFlow(
        id="flow",
        source=None,
        states={"start": FlowState(name="start", expects=[], next=[], terminal=False)},
        deviation_rules=[],
    )
    return case, flow


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (
            429,
            {"error": {"code": "upstream_rate_limited", "message": "busy"}},
            "inconclusive",
        ),
        (
            502,
            {"error": {"code": "upstream_unavailable", "message": "down"}},
            "degraded",
        ),
        (
            409,
            {"error": {"code": "duplicate_turn_id", "message": "duplicate"}},
            "failed",
        ),
    ],
)
def test_hive_error_produces_no_tester_turn_or_tts(
    hive_env, persona, status, body, expected
):
    brain = LLMBrain(
        persona,
        "g",
        client=_client(lambda _: httpx.Response(status, json=body)),
        conversation_id=CONVERSATION_ID,
    )
    transport = _Transport()
    case, flow = _case_and_flow()
    call = asyncio.run(CallRunner(case, flow, brain, transport).run())

    assert transport.sent_text == []  # AudioTransportAdapter TTS starts only here.
    assert not any(t["speaker"] == "tester" for t in call.transcript)
    assert not any(e["type"] == "tester_turn" for e in call.timeline)
    assert any(e["type"] == "hive_generation_failed" for e in call.timeline)
    assert call.failure_status == expected
    verdict = classify_session(flow, case.assertion.flow, call)
    assert verdict.status == "env_failure"
    session = build_session_payload(
        "exec-1", 1, case.id, {}, call, verdict, None, brain="llm"
    )
    assert "aiOutcome" not in session
    failed = next(e for e in session["timeline"] if e["type"] == "hive_generation_failed")
    assert failed["data"]["outcome"] == expected
    assert session["status"] == "env_failure"
    assert session["runnerMode"] == "ai-only-v3"


def test_success_provenance_reaches_transcript_timeline_and_session(hive_env, persona):
    brain = LLMBrain(
        persona,
        "g",
        client=_client(_success),
        conversation_id=CONVERSATION_ID,
    )
    transport = _Transport()
    case, flow = _case_and_flow()
    call = asyncio.run(CallRunner(case, flow, brain, transport).run())
    tester = next(t for t in call.transcript if t["speaker"] == "tester")

    assert tester["source"] == "hive-llm"
    assert tester["turnId"]
    assert tester["promptVersion"] == "echo-persona-system-v3.0.0"
    assert tester["modelVersion"] == MODEL_REVISION
    assert tester["provenance"]["source"] == "hive-llm"
    assert tester["provenance"]["provider"] == "litellm"
    assert tester["provenance"]["model"] == "deepseek-v4-pro"
    assert tester["provenance"]["modelAlias"] == "deepseek-v4-pro"
    assert tester["provenance"]["deploymentId"] == MODEL_REVISION
    assert tester["provenance"]["traceId"] == "trace-1"
    assert tester["provenance"]["generatedAt"] == "2026-07-26T00:00:00Z"
    tester_event = next(e for e in call.timeline if e["type"] == "tester_turn")
    assert tester_event["trace"]["promptHash"] == "sha256:prompt"

    payload = build_session_payload(
        "exec-1",
        1,
        case.id,
        {},
        call,
        SessionVerdict(status="passed", deviations=[]),
        None,
        brain="llm",
    )
    persisted = next(t for t in payload["transcript"] if t["role"] == "tester")
    assert set(persisted) == {"role", "text", "tsMs", "provenance"}
    assert persisted["provenance"] == {
        "source": "hive-llm",
        "turnId": tester["provenance"]["turnId"],
        "promptVersion": "echo-persona-system-v3.0.0",
            "modelResolved": MODEL_REVISION,
        "modelVersion": MODEL_REVISION,
        "policyVersion": "echo-persona-turn-v3.0.0",
        "trace": tester["trace"],
    }
    # Contract proof against createVoiceSessionSchema:
    # provenance.trace is required and owns both correlation pins.
    assert persisted["provenance"]["trace"]["promptHash"] == "sha256:prompt"
    assert persisted["provenance"]["trace"]["traceId"] == "trace-1"
