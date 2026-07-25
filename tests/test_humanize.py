"""EXEC-REALISM: memory imperfection, massa contract and humanized timing."""

from __future__ import annotations

from pathlib import Path

import pytest

from voidr_echo_runner.humanize import Humanizer, MassaFacts
from voidr_echo_runner.models import load_persona_catalog

REPO_ROOT = Path(__file__).resolve().parents[1]


def marcia():
    return load_persona_catalog(REPO_ROOT / "personas" / "catalog.yaml")[
        "dona-marcia-58-mineira"
    ]


def carlos():
    return load_persona_catalog(REPO_ROOT / "personas" / "catalog.yaml")[
        "carlos-34-paulista"
    ]


# --- MassaFacts: ECHO_MASSA contract ------------------------------------------


def test_massa_from_environment_params_json():
    massa = MassaFacts.from_params(
        {"ECHO_MASSA": '{"cpf": "390.533.447-05", "birthDate": "12/03/1966"}'}
    )
    assert massa is not None
    assert massa.source == "environment"
    assert massa.values == {"cpf": "390.533.447-05", "birthDate": "12/03/1966"}


def test_massa_accepts_portuguese_aliases():
    massa = MassaFacts.from_params(
        {"ECHO_MASSA": '{"data_nascimento": "12/03/1966", "nome": "Márcia F."}'}
    )
    assert massa is not None
    # Aliases canônicos (card pessoal) + chaves originais preservadas (dial
    # plan/steps referenciam {{massa.<chave da jornada>}}).
    assert massa.values == {
        "birthDate": "12/03/1966",
        "fullName": "Márcia F.",
        "data_nascimento": "12/03/1966",
        "nome": "Márcia F.",
    }


@pytest.mark.parametrize("raw", ["", "not-json", "[1,2]", '{"cpf": ""}', '{"cpf": "{{env.X}}"}'])
def test_massa_rejects_malformed_or_empty(raw):
    assert MassaFacts.from_params({"ECHO_MASSA": raw}) is None


def test_massa_fallback_to_persona_identity_facts():
    persona = marcia()
    persona.identity.facts["cpf"] = "390.533.447-05"
    massa = MassaFacts.resolve({}, persona)
    assert massa.source == "persona"
    assert massa.values["cpf"] == "390.533.447-05"


def test_massa_resolve_prefers_environment_over_persona():
    persona = marcia()
    persona.identity.facts["cpf"] = "111.111.111-11"
    massa = MassaFacts.resolve({"ECHO_MASSA": '{"cpf": "390.533.447-05"}'}, persona)
    assert massa.source == "environment"
    assert massa.values["cpf"] == "390.533.447-05"


def test_massa_resolve_none_when_nothing_available():
    massa = MassaFacts.resolve({}, marcia())
    assert massa.source == "none"
    assert not massa


def test_personal_data_lines_expose_placeholders_never_values():
    massa = MassaFacts.from_params({"ECHO_MASSA": '{"cpf": "390.533.447-05"}'})
    lines = massa.personal_data_lines()
    assert lines == [{"label": "CPF", "placeholder": "{{massa.cpf}}"}]
    assert "390" not in str(lines)


def test_resolve_placeholders_substitutes_and_reports_keys():
    massa = MassaFacts.from_params({"ECHO_MASSA": '{"cpf": "390.533.447-05"}'})
    text, used = massa.resolve_placeholders("peraí... é {{massa.cpf}}, isso.")
    assert text == "peraí... é 390.533.447-05, isso."
    assert used == ["cpf"]


def test_resolve_placeholders_handles_env_style_and_unknown_keys():
    massa = MassaFacts.from_params({"ECHO_MASSA": '{"cpf": "390.533.447-05"}'})
    text, used = massa.resolve_placeholders("{{env.MASSA_CPF}} e {{massa.rg}}")
    assert text == "390.533.447-05 e {{massa.rg}}"
    assert used == ["cpf"]


# --- Humanizer: memory lapses ---------------------------------------------------


def _massa_full() -> MassaFacts:
    return MassaFacts.from_params(
        {"ECHO_MASSA": '{"cpf": "390.533.447-05", "birthDate": "12/03/1966"}'}
    )


def test_lapse_probability_modulated_by_age_and_literacy():
    old = Humanizer(marcia(), 7, _massa_full())  # 58y
    young = Humanizer(carlos(), 7, _massa_full())  # 34y
    assert old.lapse_probability("cpf") > young.lapse_probability("cpf")
    assert old.lapse_probability("nome") == 0.0  # own name never lapses


def test_plan_turn_emits_directive_with_placeholder_on_data_request():
    hum = Humanizer(marcia(), 7, _massa_full())
    plan = hum.plan_turn("Para localizar seu cadastro, me informa o seu CPF?")
    assert plan.directives, "data request must produce a directive"
    assert "{{massa.cpf}}" in plan.directives[0]
    assert "390" not in " ".join(plan.directives)  # value never reaches the LLM


def test_plan_turn_deterministic_per_persona_and_seed():
    prompt = "Me informa o seu CPF, por favor?"
    a = Humanizer(marcia(), 42, _massa_full()).plan_turn(prompt)
    b = Humanizer(marcia(), 42, _massa_full()).plan_turn(prompt)
    assert (a.memory_lapse, a.directives) == (b.memory_lapse, b.directives)


def test_second_request_of_same_datum_never_lapses_again():
    # find a seed where the first CPF request lapses, then re-ask
    for seed in range(50):
        hum = Humanizer(marcia(), seed, _massa_full())
        first = hum.plan_turn("Me informa o seu CPF?")
        if first.memory_lapse:
            again = hum.plan_turn("Pode repetir o CPF, por gentileza?")
            assert not again.memory_lapse
            assert "em mãos" in again.directives[0]
            return
    pytest.fail("no seed in 0..49 produced a CPF lapse for Márcia (58y)")


def test_plan_turn_ignores_categories_without_massa():
    massa = MassaFacts.from_params({"ECHO_MASSA": '{"cpf": "390.533.447-05"}'})
    hum = Humanizer(marcia(), 7, massa)
    plan = hum.plan_turn("Qual a sua data de nascimento?")
    assert plan.directives == [] and not plan.memory_lapse


def test_finalize_reply_substitutes_massa():
    hum = Humanizer(marcia(), 7, _massa_full())
    out = hum.finalize_reply("ah achei... é {{massa.cpf}}, viu?")
    assert out == "ah achei... é 390.533.447-05, viu?"
    assert hum.substituted_keys == ["cpf"]


def test_scripted_prefix_only_on_lapse():
    hum = Humanizer(marcia(), 7, _massa_full())
    from voidr_echo_runner.humanize import TurnPlan

    assert hum.scripted_prefix(TurnPlan()) == ""
    prefix = hum.scripted_prefix(TurnPlan(memory_lapse=True))
    assert prefix.strip()


# --- Humanizer: timing ------------------------------------------------------------


def test_reply_delay_is_deterministic_and_bounded():
    a = Humanizer(marcia(), 7, _massa_full()).reply_delay_s("Oi, quero saber meu saldo.")
    b = Humanizer(marcia(), 7, _massa_full()).reply_delay_s("Oi, quero saber meu saldo.")
    assert a == b
    assert 0.35 <= a <= 9.0


def test_reply_delay_lapse_adds_seconds():
    hum1 = Humanizer(marcia(), 7, _massa_full())
    hum2 = Humanizer(marcia(), 7, _massa_full())
    from voidr_echo_runner.humanize import TurnPlan

    reply = "peraí... deixa eu pegar aqui... é 390.533.447-05"
    base = hum1.reply_delay_s(reply, TurnPlan())
    with_lapse = hum2.reply_delay_s(reply, TurnPlan(memory_lapse=True, extra_delay_s=2.5))
    assert with_lapse > base


def test_reply_delay_short_confirmations_faster_on_average():
    short = [Humanizer(carlos(), s).reply_delay_s("Isso.") for s in range(30)]
    long = [
        Humanizer(carlos(), s).reply_delay_s(
            "Então, deixa eu explicar direitinho o que aconteceu com a minha linha "
            "desde a semana passada, porque foram vários problemas seguidos e nada resolvido."
        )
        for s in range(30)
    ]
    assert sum(short) / len(short) < sum(long) / len(long)


def test_config_record_is_auditable():
    record = Humanizer(marcia(), 7, _massa_full()).config_record()
    assert record["massaSource"] == "environment"
    assert record["massaFields"] == ["birthDate", "cpf"]
    assert record["seed"] == 7 and record["timingEnabled"] is True


# --- CallRunner integration ---------------------------------------------------------


class _FakeTransport:
    def __init__(self, script):
        self._script = list(script)
        self.sent: list[str] = []
        self.silences: list[float] = []

    async def connect(self):
        pass

    async def receive(self, timeout=None):
        return self._script.pop(0) if self._script else None

    async def send_text(self, text):
        self.sent.append(text)

    async def send_dtmf(self, digits):
        pass

    async def hangup(self):
        pass

    def record_silence(self, seconds):
        self.silences.append(seconds)


class _EchoBrain:
    """Deterministic brain that mimics an LLM speaking the placeholder."""

    turn_directives: list[str] = []

    def reply(self, agent_text):
        if self.turn_directives:
            return "Ai peraí... deixa eu pegar aqui... pronto: {{massa.cpf}}."
        return "Oi, quero a segunda via da minha conta."


def test_runner_substitutes_massa_and_records_humanized_turn(monkeypatch):
    import asyncio

    from voidr_echo_runner.flows import load_journey_flow
    from voidr_echo_runner.models import VoiceTestCase
    from voidr_echo_runner.runner import CallRunner

    monkeypatch.setenv("MOCK_ACCESS_CODE", "000000000")
    case = VoiceTestCase.load(REPO_ROOT / "cases" / "consulta-saldo-tc-001.yaml")
    flow = load_journey_flow(REPO_ROOT / "flows" / "consulta-saldo-v1.json")
    transport = _FakeTransport(
        [
            {"type": "text", "speaker": "agent", "text": "Me informa o seu CPF do titular?"},
            {"type": "event", "name": "call_ended", "reason": "remote_hangup"},
        ]
    )
    hum = Humanizer(
        marcia(), 7, MassaFacts.from_params({"ECHO_MASSA": '{"cpf": "390.533.447-05"}'})
    )
    monkeypatch.setattr(hum, "reply_delay_s", lambda *a, **k: 0.01)
    runner = CallRunner(case, flow, _EchoBrain(), transport, humanizer=hum)
    result = asyncio.run(runner.run())

    assert result.transport_error is None
    tester_turns = [t for t in result.transcript if t["speaker"] == "tester"]
    assert "390.533.447-05" in tester_turns[0]["text"]
    assert "{{massa" not in transport.sent[0]
    humanized = [e for e in result.timeline if e["type"] == "humanized_turn"]
    assert humanized and humanized[0]["delayMs"] == 10
    assert transport.silences == [0.01]
