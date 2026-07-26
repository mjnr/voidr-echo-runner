"""Canonical Journey ref (moduleSlug/testPlanId) — case schema + report payload."""

from pathlib import Path

import httpx
import pytest

from voidr_echo_runner.flows import FlowState, JourneyFlow
from voidr_echo_runner.models import VoiceTestCase
from voidr_echo_runner.runner import CallResult
from voidr_echo_runner.service_mode import (
    ServeExecutionError,
    SessionVerdict,
    VoidrApi,
    build_case,
    build_session_payload,
    persist_voice_session,
    promote_params_to_environ,
    promote_trusted_envelope_to_environ,
    resolve_persona_plan_entry,
    serve_execution,
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


def test_serve_execution_rejects_raw_audio_before_api_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("VOIDR_API_URL", "http://service.test")
    monkeypatch.setenv("EXECUTION_ID", "exec-1")
    monkeypatch.setenv("VOIDR_ORG_ID", "org-1")
    monkeypatch.setenv("ENVIRONMENT_PARAMS", '{"ECHO_KEEP_RAW_AUDIO":"1"}')

    assert serve_execution(tmp_path) == 2


def test_voice_session_post_error_is_not_swallowed():
    class BrokenApi:
        def post_session(self, payload):
            raise RuntimeError("schema rejected")

    with pytest.raises(RuntimeError, match="schema rejected"):
        persist_voice_session(BrokenApi(), {"runnerMode": "ai-only-v3"})


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


# ── serve-execution: promoção de configuração não secreta ────────────────────


def test_promote_params_separates_customer_and_platform_configuration():
    env = {
        "VOICE_GATEWAY_URL": "wss://platform.voice.internal",
        "HIVE_URL": "https://platform.hive.internal",
        "ECHO_RUNTIME_ENV": "production",
    }
    promoted = promote_params_to_environ(
        {
            "TWILIO_ACCOUNT_SID": "AC123",
            "TWILIO_AUTH_TOKEN": "tok",
            "DEEPGRAM_API_KEY": "dg",
            "ELEVENLABS_API_KEY": "el",
            "ECHO_CALL_MODE": "audio",
            "VOICE_GATEWAY_URL": "wss://voice.example",
            "ECHO_RUNTIME_ENV": "local",
            "ECHO_MEDIA_GATEWAY_RUNNER_URL": "ws://attacker.invalid",
            "VOICE_TTS_ADAPTER": "direct",
            "HIVE_URL": "http://hive:3001",
            "HIVE_GATEWAY_TOKEN": "gw",
            "BASE_URL": "https://app.example",  # não promovida (fora dos prefixos)
            "MOCK_ACCESS_CODE": "919021552",  # idem — resolve só via {{env.*}}
        },
        environ=env,
    )
    assert promoted == ["ECHO_CALL_MODE"]
    assert env["VOICE_GATEWAY_URL"] == "wss://platform.voice.internal"
    assert env["HIVE_URL"] == "https://platform.hive.internal"
    assert env["ECHO_RUNTIME_ENV"] == "production"
    assert "TWILIO_ACCOUNT_SID" not in env
    assert "HIVE_GATEWAY_TOKEN" not in env
    assert "ECHO_MEDIA_GATEWAY_RUNNER_URL" not in env
    assert "VOICE_TTS_ADAPTER" not in env
    assert "DEEPGRAM_API_KEY" not in env
    assert "ELEVENLABS_API_KEY" not in env
    assert "BASE_URL" not in env
    assert "MOCK_ACCESS_CODE" not in env


def test_customer_params_cannot_overwrite_platform_fallbacks():
    env = {"HIVE_URL": "http://localhost:3001", "DEEPGRAM_API_KEY": "local"}
    promote_params_to_environ({"HIVE_URL": "http://attacker.invalid"}, environ=env)
    assert env["HIVE_URL"] == "http://localhost:3001"
    assert env["DEEPGRAM_API_KEY"] == "local"


def test_trusted_envelope_promotes_key_and_observability_scope():
    env = {}
    promoted = promote_trusted_envelope_to_environ(
        {
            "LITELLM_API_KEY": "platform-virtual-key",
            "VOIDR_ORGANIZATION_ID": "org-trusted",
            "VOIDR_EXECUTION_ID": "execution-trusted",
        },
        environ=env,
    )
    assert promoted == [
        "LITELLM_API_KEY",
        "VOIDR_EXECUTION_ID",
        "VOIDR_ORGANIZATION_ID",
    ]
    assert env == {
        "LITELLM_API_KEY": "platform-virtual-key",
        "VOIDR_ORGANIZATION_ID": "org-trusted",
        "VOIDR_EXECUTION_ID": "execution-trusted",
    }


def test_client_params_cannot_inject_governed_key_or_scope():
    env = {
        "LITELLM_API_KEY": "platform-virtual-key",
        "VOIDR_ORGANIZATION_ID": "org-trusted",
        "VOIDR_EXECUTION_ID": "execution-trusted",
    }
    promote_params_to_environ(
        {
            "LITELLM_API_KEY": "attacker-key",
            "VOIDR_ORGANIZATION_ID": "org-attacker",
            "VOIDR_EXECUTION_ID": "execution-attacker",
        },
        environ=env,
    )
    assert env["LITELLM_API_KEY"] == "platform-virtual-key"
    assert env["VOIDR_ORGANIZATION_ID"] == "org-trusted"
    assert env["VOIDR_EXECUTION_ID"] == "execution-trusted"


def test_consumed_envelope_rejects_client_key_tenant_and_execution_injection():
    api = object.__new__(VoidrApi)
    api._request = lambda *_args, **_kwargs: {
        "trustedEnv": {
            "LITELLM_API_KEY": "platform-key",
            "VOIDR_ORGANIZATION_ID": "org-trusted",
            "VOIDR_EXECUTION_ID": "execution-trusted",
        },
        "clientParams": {
            "LITELLM_API_KEY": "attacker-key",
            "VOIDR_ORGANIZATION_ID": "org-attacker",
            "VOIDR_EXECUTION_ID": "execution-attacker",
        },
    }
    with pytest.raises(ServeExecutionError, match="override governed fields"):
        api.consume_voice_envelope("execution-trusted", 1, "opaque-ref")


def test_promote_params_skips_empty_values():
    env = {"ECHO_CALL_MODE": "text"}
    promoted = promote_params_to_environ({"ECHO_CALL_MODE": ""}, environ=env)
    assert promoted == []
    assert env["ECHO_CALL_MODE"] == "text"


def test_customer_target_host_requires_service_allowlist():
    env = {"ECHO_ALLOWED_TARGET_HOSTS": "mock.internal,*.safe.example"}
    assert promote_params_to_environ(
        {"ECHO_CALL_TARGET": "wss://tenant.safe.example/call"}, environ=env
    ) == []
    with pytest.raises(ServeExecutionError, match="not allowed"):
        promote_params_to_environ(
            {"ECHO_CALL_TARGET": "wss://exfil.attacker.invalid/collect"},
            environ=env,
        )


def test_customer_cannot_inject_allowlist_to_enable_exfiltration():
    with pytest.raises(ServeExecutionError, match="not allowed"):
        promote_params_to_environ(
            {
                "ECHO_ALLOWED_TARGET_HOSTS": "attacker.invalid",
                "ECHO_CALL_TARGET": "wss://attacker.invalid/collect",
            },
            environ={},
        )


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


def test_service_errors_never_include_remote_body(monkeypatch):
    from voidr_echo_runner.service_mode import VoidrApi

    secret_body = "token=remote-secret cpf=123.456.789-09 user@example.com"
    api = VoidrApi("https://service.invalid", "access-secret")
    api._client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(400, text=secret_body)
        )
    )
    with pytest.raises(ServeExecutionError) as error:
        api.get_execution("execution")
    rendered = str(error.value)
    assert rendered == "service_http_400"
    assert "remote-secret" not in rendered
    assert "123.456.789-09" not in rendered
    assert "user@example.com" not in rendered


def test_put_shard_scrubs_environment_values_and_pii(monkeypatch):
    from voidr_echo_runner.service_mode import VoidrApi

    captured = {}
    monkeypatch.setenv(
        "ENVIRONMENT_PARAMS",
        '{"CUSTOMER_SECRET":"sensitive-value","PHONE":"+5511999999999"}',
    )
    api = VoidrApi.__new__(VoidrApi)
    api._request = (  # type: ignore[attr-defined]
        lambda _method, _path, **kwargs: captured.update(kwargs["json"]) or {}
    )
    api.put_shard(
        "execution",
        1,
        {
            "status": "FAILED",
            "errorMessage": (
                "sensitive-value +5511999999999 user@example.com "
                "Bearer abc.def.ghi"
            ),
        },
    )
    error = captured["errorMessage"]
    assert "sensitive-value" not in error
    assert "5511999999999" not in error
    assert "user@example.com" not in error
    assert "abc.def.ghi" not in error


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


def test_session_payload_is_always_marked_ai_only():
    payload = build_session_payload(
        "exec-1",
        1,
        "JORNA-01",
        {},
        _call_result(),
        SessionVerdict(status="passed", deviations=[]),
        None,
    )
    assert payload["brain"] == "llm"
    assert payload["runnerMode"] == "ai-only-v3"
