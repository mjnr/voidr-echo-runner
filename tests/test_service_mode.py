"""Canonical Journey ref (moduleSlug/testPlanId) — case schema + report payload."""

from pathlib import Path

from voidr_echo_runner.flows import FlowState, JourneyFlow
from voidr_echo_runner.models import VoiceTestCase
from voidr_echo_runner.runner import CallResult
from voidr_echo_runner.service_mode import (
    SessionVerdict,
    build_case,
    build_session_payload,
    promote_params_to_environ,
    resolve_persona_plan_entry,
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


# ── pré-âmbulo DTMF da jornada (flow.dialPlan) + {{massa.*}} nos sends ────────


def _flow_with_dial_plan() -> JourneyFlow:
    return JourneyFlow(
        id="jornada-info-de-planos",
        source=None,
        states={"saudacao": FlowState(name="saudacao", expects=[], next=[], terminal=True)},
        deviation_rules=[],
        dial_plan_steps=[
            {"waitFor": "código de acesso", "send": "{{massa.codigo_acesso}}"},
            {"waitFor": "número da linha", "send": "{{massa.telefone_ura}}#"},
        ],
    )


def _massa(values: dict[str, str]) -> "MassaFacts":
    from voidr_echo_runner.humanize import MassaFacts

    return MassaFacts(values=values, source="environment")


def test_flow_dial_plan_fills_case_without_dtmf_steps_resolving_massa():
    massa = _massa({"codigo_acesso": "919021050", "telefone_ura": "11900000001"})
    case, _ = build_case(_target(), _case_doc(), "p", _flow_with_dial_plan(), {}, massa=massa)
    assert [s.send for s in case.dial_plan.dtmf_steps] == ["919021050", "11900000001#"]
    assert [s.wait_for_prompt_matching for s in case.dial_plan.dtmf_steps] == [
        "código de acesso",
        "número da linha",
    ]


def test_massa_value_may_reference_env_secret():
    # Editor de massa: campo fixo apontando para um secret do environment —
    # duas passadas: {{massa.X}} → {{env.Y}} → valor real.
    massa = _massa({"codigo_acesso": "{{env.IBM_ACCESS_CODE}}", "telefone_ura": "11900000001"})
    params = {"IBM_ACCESS_CODE": "123456"}
    case, _ = build_case(_target(), _case_doc(), "p", _flow_with_dial_plan(), params, massa=massa)
    assert case.dial_plan.dtmf_steps[0].send == "123456"


def test_case_dtmf_steps_win_over_flow_dial_plan():
    doc = _case_doc()
    doc["voice"]["dialPlan"]["dtmfSteps"] = [{"waitFor": "código", "send": "111111"}]
    massa = _massa({"codigo_acesso": "919021050", "telefone_ura": "11900000001"})
    case, _ = build_case(_target(), doc, "p", _flow_with_dial_plan(), {}, massa=massa)
    assert [s.send for s in case.dial_plan.dtmf_steps] == ["111111"]


def test_echo_massa_from_params_resolves_env_refs_and_keeps_journey_keys():
    import json

    from voidr_echo_runner.humanize import MassaFacts

    params = {
        "ECHO_MASSA": json.dumps(
            {
                "codigo_acesso": "{{env.IBM_ACCESS_CODE}}",
                "telefone_ura": "11900000001",
                "quebrado": "{{env.NAO_EXISTE}}",
            }
        ),
        "IBM_ACCESS_CODE": "919021050",
    }
    massa = MassaFacts.from_params(params)
    assert massa is not None
    # Chave da jornada preservada + alias canônico do card pessoal.
    assert massa.values["codigo_acesso"] == "919021050"
    assert massa.values["accessCode"] == "919021050"
    assert massa.values["telefone_ura"] == "11900000001"
    # Placeholder irresolúvel continua fora do bag (guard de fala/discagem).
    assert "quebrado" not in massa.values


def test_unresolved_massa_placeholder_fails_loud():
    import pytest

    from voidr_echo_runner.service_mode import ServeExecutionError

    massa = _massa({"telefone_ura": "11900000001"})  # sem codigo_acesso
    with pytest.raises(ServeExecutionError, match="codigo_acesso"):
        build_case(_target(), _case_doc(), "p", _flow_with_dial_plan(), {}, massa=massa)


# ── ECHO_PERSONA_PLAN: persona/overrides por shard (multi-persona) ───────────


def test_persona_plan_entry_resolves_by_shard():
    import json as _json

    plan = [
        {"personaId": "dona-marcia-58-mineira", "overrides": {"patienceLevel": 1}},
        {"personaId": "carlos-34-paulista"},
    ]
    params = {"ECHO_PERSONA_PLAN": _json.dumps(plan)}
    assert resolve_persona_plan_entry(params, 1)["personaId"] == "dona-marcia-58-mineira"
    assert resolve_persona_plan_entry(params, 1)["overrides"] == {"patienceLevel": 1}
    assert resolve_persona_plan_entry(params, 2)["personaId"] == "carlos-34-paulista"


def test_persona_plan_absent_or_short_falls_back_to_case_persona():
    assert resolve_persona_plan_entry({}, 1) is None
    assert resolve_persona_plan_entry({"ECHO_PERSONA_PLAN": "[]"}, 1) is None
    assert resolve_persona_plan_entry({"ECHO_PERSONA_PLAN": '[{"personaId":"x"}]'}, 2) is None


def test_persona_plan_malformed_json_never_breaks_the_call(capsys):
    assert resolve_persona_plan_entry({"ECHO_PERSONA_PLAN": "{broken"}, 1) is None
    assert "ECHO_PERSONA_PLAN inválido" in capsys.readouterr().err


# ── serve-execution: promoção de secrets ENVIRONMENT_PARAMS → os.environ ─────


def test_promote_params_covers_media_hive_and_echo_prefixes():
    env: dict[str, str] = {}
    promoted = promote_params_to_environ(
        {
            "TWILIO_ACCOUNT_SID": "AC123",
            "TWILIO_AUTH_TOKEN": "tok",
            "DEEPGRAM_API_KEY": "dg",
            "ELEVENLABS_API_KEY": "el",
            "ECHO_CALL_MODE": "audio",
            "HIVE_URL": "http://hive:3001",
            "HIVE_GATEWAY_TOKEN": "gw",
            "BASE_URL": "https://app.example",  # não promovida (fora dos prefixos)
            "MOCK_ACCESS_CODE": "919021552",  # idem — resolve só via {{env.*}}
        },
        environ=env,
    )
    assert promoted == sorted(
        [
            "TWILIO_ACCOUNT_SID",
            "TWILIO_AUTH_TOKEN",
            "DEEPGRAM_API_KEY",
            "ELEVENLABS_API_KEY",
            "ECHO_CALL_MODE",
            "HIVE_URL",
            "HIVE_GATEWAY_TOKEN",
        ]
    )
    assert env["TWILIO_ACCOUNT_SID"] == "AC123"
    assert env["HIVE_GATEWAY_TOKEN"] == "gw"
    assert "BASE_URL" not in env
    assert "MOCK_ACCESS_CODE" not in env


def test_promote_params_overwrites_local_env_but_keeps_fallback():
    # ENVIRONMENT_PARAMS é a intenção do environment para ESTA execução —
    # ganha do os.environ herdado (ex.: .env do repo no dev local)...
    env = {"HIVE_URL": "http://localhost:3001", "DEEPGRAM_API_KEY": "local"}
    promote_params_to_environ({"HIVE_URL": "http://hive-prod:3001"}, environ=env)
    assert env["HIVE_URL"] == "http://hive-prod:3001"
    # ...mas chave ausente do bag preserva o fallback local (dev sem secret
    # cadastrado no environment continua funcionando via .env).
    assert env["DEEPGRAM_API_KEY"] == "local"


def test_promote_params_skips_empty_values():
    env = {"TWILIO_AUTH_TOKEN": "local"}
    promoted = promote_params_to_environ({"TWILIO_AUTH_TOKEN": ""}, environ=env)
    assert promoted == []
    assert env["TWILIO_AUTH_TOKEN"] == "local"


def test_session_payload_records_overrides_and_source():
    payload = build_session_payload(
        "exec-1",
        2,
        "JORNA-01",
        {"journeyFlowId": "flow-1", "personaId": "carlos-34-paulista"},
        _call_result(),
        SessionVerdict(status="passed", deviations=[]),
        None,
        brain="llm",
        persona_overrides={"initialEmotion": "irritado", "patienceLevel": 1},
        persona_source="execution",
    )
    assert payload["personaOverrides"] == {"initialEmotion": "irritado", "patienceLevel": 1}
    assert payload["personaSource"] == "execution"


def test_session_payload_omits_override_fields_when_unset():
    payload = build_session_payload(
        "exec-1",
        1,
        "JORNA-01",
        {"journeyFlowId": "flow-1"},
        _call_result(),
        SessionVerdict(status="passed", deviations=[]),
        None,
    )
    assert "personaOverrides" not in payload
    assert "personaSource" not in payload


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


def test_persona_from_service_maps_literacy_and_glossary_vocabulary():
    """E3: v2.1 axes (literacy + resolved glossary vocabulary) flow through."""
    from voidr_echo_runner.service_mode import persona_from_service

    persona = persona_from_service(
        {
            "_id": "6a2",
            "slug": "gen-s7-mineiro-rudimentar",
            "name": "Zé (61, mineiro)",
            "literacy": {"inafLevel": "rudimentar", "digitalFluency": "nenhuma", "numeracy": "baixa"},
            "glossaryVocabulary": {
                "masteryRate": 0.3,
                "band": "baixo",
                "knowledgeLevel": "baixa",
                "popularOnly": [{"termId": "t1", "term": "fatura", "synonym": "a conta"}],
                "unknown": [{"termId": "t2", "term": "roaming"}],
                "confused": [],
            },
            "demographics": {"ageBand": "60+", "region": "mineiro"},
            "temperament": {
                "mood": "confuso",
                "patienceLevel": 3,
                "techSavviness": "baixa",
                "verbosity": "normal",
                "intentNoise": "nenhum",
            },
            "speech": {"disfluencyRate": 0.3},
            "goalTemplate": "",
        }
    )
    assert persona.literacy.inafLevel == "rudimentar"
    assert persona.glossaryVocabulary.masteryRate == 0.3
    assert persona.glossaryVocabulary.popularOnly[0].synonym == "a conta"
    assert persona.glossaryVocabulary.unknown[0].term == "roaming"


def test_get_persona_forwards_knowledge_level():
    """E3 unification: the ephemeral 'conhecimento do assunto' override goes
    to the service as knowledgeLevel so the glossary partition follows it."""
    from voidr_echo_runner.service_mode import VoidrApi

    calls: list[str] = []

    api = VoidrApi.__new__(VoidrApi)
    api._request = lambda method, path, **kw: calls.append(path) or {}  # type: ignore[attr-defined]

    api.get_persona("dona-marcia")
    api.get_persona("dona-marcia", knowledge_level="baixa")

    assert calls[0] == "/echo/personas/dona-marcia?vocabulary=1"
    assert calls[1] == "/echo/personas/dona-marcia?vocabulary=1&knowledgeLevel=baixa"


def test_persona_from_service_accepts_free_traits_string():
    # The service stores profile.freeTraits as a single "a; b" string (hive
    # contract); the runner model must coerce it into a list.
    from voidr_echo_runner.service_mode import persona_from_service

    persona = persona_from_service(
        {
            "_id": "6a1",
            "slug": "seu-raimundo-62-cearense",
            "name": "Seu Raimundo",
            "profile": {
                "occupation": "comerciante",
                "context": "Fortaleza",
                "freeTraits": 'fala devagar; trata todo mundo por "meu rei"',
            },
            "demographics": {"ageBand": "60+", "region": "cearense"},
            "temperament": {
                "mood": "calmo",
                "patienceLevel": 4,
                "techSavviness": "baixa",
                "verbosity": "prolixo",
                "intentNoise": "nenhum",
            },
            "speech": {"disfluencyRate": 0.2},
            "goalTemplate": "quero {goal}",
        }
    )
    assert persona.profile.freeTraits == [
        "fala devagar",
        'trata todo mundo por "meu rei"',
    ]


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
