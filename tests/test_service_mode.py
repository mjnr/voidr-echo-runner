"""Canonical Journey ref (moduleSlug/testPlanId) — case schema + report payload."""

from pathlib import Path

from voidr_echo_runner.flows import FlowState, JourneyFlow
from voidr_echo_runner.models import VoiceTestCase
from voidr_echo_runner.runner import CallResult
from voidr_echo_runner.service_mode import (
    SessionVerdict,
    build_case,
    build_session_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = REPO_ROOT / "cases"


def _minimal_flow() -> JourneyFlow:
    return JourneyFlow(
        id="consulta-saldo-v1",
        source=None,
        states={"saudacao": FlowState(name="saudacao", expects=[], next=[], terminal=True)},
        deviation_rules=[],
    )


def _call_result() -> CallResult:
    return CallResult(started_at_ms=1000, ended_at_ms=2000, agent_turns=3)


# ── case YAML schema ─────────────────────────────────────────────────────────


def test_example_cases_carry_canonical_module_slug(monkeypatch):
    monkeypatch.setenv("MOCK_ACCESS_CODE", "000000000")
    case1 = VoiceTestCase.load(CASES_DIR / "consulta-saldo-tc-001.yaml")
    case2 = VoiceTestCase.load(CASES_DIR / "bloqueio-financeiro-tc-002.yaml")
    assert case1.module_slug == "jornada-consulta-de-saldo"
    assert case2.module_slug == "jornada-bloqueio-financeiro"
    # testPlanId is environment-specific — never hardcoded in the example YAMLs.
    assert case1.test_plan_id is None


def test_module_slug_is_optional_in_case_schema():
    case = VoiceTestCase.model_validate(
        {
            "id": "tc-x",
            "persona": {"base": "p"},
            "journey_flow": "flow.json",
            "goal": "g",
        }
    )
    assert case.module_slug is None
    assert case.test_plan_id is None


# ── serve-execution: build_case propagates the target's canonical ref ────────


def _target(module_slug: str = "jornada-consulta-de-saldo") -> dict:
    return {
        "moduleSlug": module_slug,
        "suiteSlug": "CONSU",
        "testCaseSlug": "JORNA-01",
    }


def _case_doc() -> dict:
    return {
        "name": "Consulta de saldo",
        "voice": {
            "channel": "voice",
            "dialPlan": {"to": "ws://localhost:8765/ws", "dtmfSteps": []},
            "goal": "saber o saldo",
            "seed": 42,
        },
    }


def test_build_case_sets_module_slug_and_plan_from_target():
    case, _ = build_case(
        _target(), _case_doc(), "dona-marcia-58-mineira", _minimal_flow(), {}, "plan-1"
    )
    assert case.module_slug == "jornada-consulta-de-saldo"
    assert case.test_plan_id == "plan-1"


def test_build_case_without_plan_id_keeps_fields_none():
    target = {"suiteSlug": "S", "testCaseSlug": "TC"}
    case, _ = build_case(target, _case_doc(), "p", _minimal_flow(), {})
    assert case.module_slug is None
    assert case.test_plan_id is None


# ── environment voice config (ECHO_* reserved keys in ENVIRONMENT_PARAMS) ────


def test_env_call_target_overrides_case_dial_plan_to():
    params = {"ECHO_CALL_TARGET": "tel:{{env.IBM_TEST_NUMBER}}", "IBM_TEST_NUMBER": "+16893996780"}
    case, call_target = build_case(
        _target(), _case_doc(), "dona-marcia-58-mineira", _minimal_flow(), params, "plan-1"
    )
    assert call_target == "tel:+16893996780"
    assert case.dial_plan.to == "tel:+16893996780"


def test_env_dial_plan_defaults_fill_missing_dtmf_steps():
    doc = _case_doc()
    doc["voice"]["dialPlan"] = {"to": "ws://localhost:8765/ws"}  # sem dtmfSteps
    params = {
        "ECHO_DIAL_ACCESS_CODE": "{{env.IBM_ACCESS_CODE}}",
        "ECHO_DIAL_ANI": "{{env.MASSA_ANI}}#",
        "IBM_ACCESS_CODE": "123456",
        "MASSA_ANI": "11900000001",
    }
    case, _ = build_case(_target(), doc, "p", _minimal_flow(), params)
    assert [s.send for s in case.dial_plan.dtmf_steps] == ["123456", "11900000001#"]


def test_case_dtmf_steps_win_over_env_defaults():
    params = {"ECHO_DIAL_ACCESS_CODE": "999999"}
    doc = _case_doc()
    doc["voice"]["dialPlan"]["dtmfSteps"] = [{"waitFor": "código", "send": "111111"}]
    case, _ = build_case(_target(), doc, "p", _minimal_flow(), params)
    assert [s.send for s in case.dial_plan.dtmf_steps] == ["111111"]


def test_session_payload_carries_audio_artifact():
    payload = build_session_payload(
        "exec-1",
        1,
        "JORNA-01",
        {"journeyFlowId": "flow-1"},
        _call_result(),
        SessionVerdict(status="passed", deviations=[]),
        "org/o/executions/e/shards/1/artifacts/transcript.json",
        module_slug="jornada-consulta-de-saldo",
        audio_path="org/o/executions/e/shards/1/artifacts/call.redacted.wav",
    )
    assert payload["artifacts"]["audioPath"].endswith("call.redacted.wav")
    assert payload["artifacts"]["transcriptPath"].endswith("transcript.json")


# ── report payload: moduleSlug present only when the case provides it ────────


def test_session_payload_includes_module_slug_when_present():
    payload = build_session_payload(
        "exec-1",
        1,
        "JORNA-01",
        {"journeyFlowId": "flow-1", "personaId": "p-1", "seed": 42},
        _call_result(),
        SessionVerdict(status="passed", deviations=[]),
        None,
        module_slug="jornada-consulta-de-saldo",
    )
    assert payload["moduleSlug"] == "jornada-consulta-de-saldo"
    assert payload["journeyFlowId"] == "flow-1"


def test_session_payload_omits_module_slug_when_absent():
    payload = build_session_payload(
        "exec-1",
        1,
        "JORNA-01",
        {},
        _call_result(),
        SessionVerdict(status="passed", deviations=[]),
        None,
    )
    assert "moduleSlug" not in payload


# ── report payload: brain fires the persona-fidelity judge on llm sessions ───


def test_session_payload_carries_brain_kind():
    payload = build_session_payload(
        "exec-1",
        1,
        "JORNA-01",
        {},
        _call_result(),
        SessionVerdict(status="passed", deviations=[]),
        None,
        brain="llm",
    )
    assert payload["brain"] == "llm"


def test_persona_from_service_maps_v2_blocks():
    from voidr_echo_runner.service_mode import persona_from_service

    persona = persona_from_service(
        {
            "_id": "6a0",
            "slug": "dona-marcia-58-mineira",
            "name": "Dona Márcia (58, mineira)",
            "identity": {"shortName": "Márcia", "facts": {"cidade": "Belo Horizonte"}},
            "psychometrics": {
                "openness": 35,
                "conscientiousness": 45,
                "extraversion": 55,
                "agreeableness": 62,
                "neuroticism": 75,
            },
            "behaviors": {"incompleteUtterance": 0.15},
            "emotionalModel": {
                "initialEmotion": "ansioso",
                "initialIntensity": 0.3,
                "triggers": [{"on": "repetiu_pergunta", "delta": 0.15}],
                "thresholds": {"pedirHumano": 0.74, "desligar": 0.89},
            },
            "demographics": {"ageBand": "41-60", "region": "mineiro"},
            "temperament": {
                "mood": "ansioso",
                "patienceLevel": 2,
                "techSavviness": "baixa",
                "verbosity": "normal",
                "intentNoise": "nenhum",
            },
            "speech": {"disfluencyRate": 0.3, "interruptionPolicy": "never", "fillers": []},
            "goalTemplate": "quero {goal}",
        }
    )
    assert persona.name == "Márcia"  # shortName wins over the display name
    assert persona.identity.facts["cidade"] == "Belo Horizonte"
    assert persona.psychometrics.neuroticism == 75
    assert persona.behaviors.incompleteUtterance == 0.15
    assert persona.emotionalModel.thresholds.pedirHumano == 0.74
    assert persona.emotionalModel.triggers[0].on == "repetiu_pergunta"


def test_session_payload_omits_brain_when_unset():
    payload = build_session_payload(
        "exec-1",
        1,
        "JORNA-01",
        {},
        _call_result(),
        SessionVerdict(status="passed", deviations=[]),
        None,
    )
    assert "brain" not in payload
