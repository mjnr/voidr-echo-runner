"""LLMBrain: hive persona-turn client (mocked HTTP — no network, no LLM keys)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from voidr_echo_runner.brain import GENERIC_JOURNEY_STATE, HiveError, LLMBrain, build_brain
from voidr_echo_runner.models import load_persona_catalog

REPO_ROOT = Path(__file__).resolve().parents[1]
HIVE_ENV = {
    "HIVE_URL": "http://hive.test:3001",
    "HIVE_GATEWAY_TOKEN": "test-token",
    "VOIDR_ORG_ID": "org-vivo-staging",
}


@pytest.fixture
def persona():
    return load_persona_catalog(REPO_ROOT / "personas" / "catalog.yaml")[
        "dona-marcia-58-mineira"
    ]


@pytest.fixture
def hive_env(monkeypatch):
    for key, value in HIVE_ENV.items():
        monkeypatch.setenv(key, value)


def _client(handler) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": f"Bearer {HIVE_ENV['HIVE_GATEWAY_TOKEN']}"},
    )


def _ok_response(text="Uai, meu saldo sumiu, moço!", model="deepseek-v4-pro", escalated=False):
    return httpx.Response(
        200,
        json={
            "text": text,
            "model": model,
            "usage": {
                "inputTokens": 420,
                "outputTokens": 31,
                "costUsd": 0.0012,
                "escalated": escalated,
            },
        },
    )


def test_missing_env_fails_with_actionable_message(monkeypatch, persona):
    for key in HIVE_ENV:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="HIVE_URL, HIVE_GATEWAY_TOKEN, VOIDR_ORG_ID"):
        LLMBrain(persona, goal="consultar saldo", seed=42)


def test_build_brain_llm_returns_llm_brain(hive_env, persona):
    brain = build_brain("llm", persona, goal="consultar saldo", seed=7)
    assert isinstance(brain, LLMBrain)


def test_turn_payload_matches_contract(hive_env, persona):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return _ok_response()

    brain = LLMBrain(persona, goal="ver meu saldo", seed=42, client=_client(handler))
    text = brain.reply("Bom dia, aqui é da Vivo, como posso ajudar?")

    assert text == "Uai, meu saldo sumiu, moço!"
    assert captured["url"] == "http://hive.test:3001/echo/persona-turn"
    assert captured["auth"] == "Bearer test-token"

    body = captured["body"]
    assert body["organizationId"] == "org-vivo-staging"
    assert body["journeyState"] == GENERIC_JOURNEY_STATE
    assert body["options"] == {"seed": 42}
    assert body["history"] == [
        {"role": "agent", "text": "Bom dia, aqui é da Vivo, como posso ajudar?"}
    ]

    p = body["persona"]
    assert p["id"] == "dona-marcia-58-mineira"
    assert p["demographics"] == {"ageBand": "41-60", "region": "mineiro"}
    assert p["temperament"]["mood"] == "ansioso"
    assert p["temperament"]["patienceLevel"] == 2
    assert p["speech"] == {"disfluencyRate": 0.3}
    # goalTemplate arrives with {goal} already resolved
    assert "{goal}" not in p["goalTemplate"]
    assert "ver meu saldo" in p["goalTemplate"]
    assert p["vocabulary"] == ["uai", "trem", "ocê"]


def test_history_accumulates_and_cost_sums(hive_env, persona):
    brain = LLMBrain(persona, goal="g", client=_client(lambda _: _ok_response()))
    brain.reply("primeiro turno")
    brain.reply("segundo turno")
    roles = [t["role"] for t in brain.history]
    assert roles == ["agent", "persona", "agent", "persona"]
    assert brain.total_cost_usd == pytest.approx(0.0024)
    assert brain.last_model == "deepseek-v4-pro"


def test_escalate_per_turn_and_default(hive_env, persona):
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _ok_response(model="sonnet", escalated=True)

    brain = LLMBrain(persona, goal="g", client=_client(handler))
    brain.take_turn("turno normal")
    brain.take_turn("turno escalado", escalate=True)
    assert "options" not in bodies[0]
    assert bodies[1]["options"] == {"escalate": True}

    always = LLMBrain(persona, goal="g", escalate=True, client=_client(handler))
    always.take_turn("qualquer")
    assert bodies[2]["options"] == {"escalate": True}


def test_retries_once_on_502_then_succeeds(hive_env, persona):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(502, json={"error": "LLM gateway call failed"})
        return _ok_response()

    brain = LLMBrain(persona, goal="g", client=_client(handler))
    assert brain.reply("oi") == "Uai, meu saldo sumiu, moço!"
    assert calls["n"] == 2


def test_502_after_retry_raises_gateway_hint(hive_env, persona):
    handler = lambda _: httpx.Response(502, json={"error": "LLM gateway call failed"})  # noqa: E731
    brain = LLMBrain(persona, goal="g", client=_client(handler))
    with pytest.raises(HiveError, match="502.*gateway LLM.*LLM gateway call failed"):
        brain.reply("oi")


def test_422_pii_raises_clear_error_without_retry(hive_env, persona):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(422, json={"error": "history contém dados sensíveis"})

    brain = LLMBrain(persona, goal="g", client=_client(handler))
    with pytest.raises(HiveError, match="422.*PII"):
        brain.reply("meu cpf é tal")
    assert calls["n"] == 1  # 4xx must not be retried


def test_transport_error_retried_then_raises(hive_env, persona):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("connection refused")

    brain = LLMBrain(persona, goal="g", client=_client(handler))
    with pytest.raises(HiveError, match="hive unreachable"):
        brain.reply("oi")
    assert calls["n"] == 2
