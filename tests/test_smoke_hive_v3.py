from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from voidr_echo_runner.brain import HiveError, LLMBrain
from voidr_echo_runner.models import load_persona_catalog


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "smoke"))

from hive_v3_stub import success_response, validate_request  # noqa: E402


MODEL_REVISION = (
    "deepseek-v4-pro@sha256:"
    "59e858aa0bd9bdbc7524a5dd39d84904747dacd1f85d152d0c04bcc373db9a08"
)
CONVERSATION_ID = "d9267c63-0f0a-5a51-9b31-33fb85bbab03"


@pytest.fixture
def hive_env(monkeypatch):
    monkeypatch.setenv("HIVE_URL", "http://smoke-hive.test")
    monkeypatch.setenv("HIVE_GATEWAY_TOKEN", "smoke-test-token")
    monkeypatch.setenv("VOIDR_ORG_ID", "smoke-test-org")
    monkeypatch.setenv("HIVE_ECHO_PERSONA_V3_MODEL_REVISION", MODEL_REVISION)


@pytest.fixture
def persona():
    return load_persona_catalog(ROOT / "personas" / "catalog.yaml")[
        "dona-marcia-58-mineira"
    ].model_copy(deep=True)


def stub_client(*, malformed: bool = False) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/echo/persona-turn/v3"
        assert request.headers["Authorization"] == "Bearer smoke-test-token"
        body = validate_request(json.loads(request.content))
        response = success_response(body, MODEL_REVISION)
        if malformed:
            response["provenance"]["deploymentPin"] = "mutable-latest"
        return httpx.Response(200, json=response)

    return httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer smoke-test-token"},
    )


def test_smoke_stub_satisfies_strict_v3_contract(hive_env, persona):
    brain = LLMBrain(
        persona,
        "consultar saldo",
        seed=42,
        client=stub_client(),
        conversation_id=CONVERSATION_ID,
    )
    brain.journey_state = {
        "flowSlug": "consulta-saldo-v1",
        "currentState": "saudacao",
        "expects": ["apresentação do agente"],
    }

    turn = brain.take_turn("Como posso te ajudar?")

    assert turn["text"] == "consultar saldo"
    assert turn["provenance"]["source"] == "hive-llm"
    assert turn["provenance"]["contractVersion"] == "v3"
    assert turn["provenance"]["deploymentPin"] == MODEL_REVISION
    assert turn["usage"] == {"costUsd": 0.0}


def test_smoke_stub_malformed_provenance_fails_closed(hive_env, persona):
    brain = LLMBrain(
        persona,
        "consultar saldo",
        seed=42,
        client=stub_client(malformed=True),
        conversation_id=CONVERSATION_ID,
    )

    with pytest.raises(HiveError, match="deploymentPin"):
        brain.take_turn("Como posso te ajudar?")


def test_real_smoke_requires_hive_environment_without_printing_secrets():
    env = os.environ.copy()
    for name in (
        "HIVE_URL",
        "HIVE_GATEWAY_TOKEN",
        "VOIDR_ORG_ID",
        "HIVE_ECHO_PERSONA_V3_MODEL_REVISION",
    ):
        env.pop(name, None)
    env["SMOKE_HIVE_MODE"] = "real"

    result = subprocess.run(
        ["bash", str(ROOT / "smoke" / "run-smoke.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 2
    assert "SMOKE_HIVE_MODE=real exige as envs" in output
    assert "HIVE_GATEWAY_TOKEN" in output
    assert "smoke-hive-v3-token" not in output
