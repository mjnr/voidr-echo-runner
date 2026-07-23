"""Emotional appraisal state machine (PERSONAS-SOTA P0.3) — deterministic curves."""

from __future__ import annotations

from pathlib import Path

import pytest

from voidr_echo_runner.emotional import AppraisalDetector, EmotionalStateMachine
from voidr_echo_runner.models import (
    EmotionalModel,
    EmotionalThresholds,
    EmotionalTrigger,
    Persona,
    load_persona_catalog,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def marcia() -> Persona:
    return load_persona_catalog(REPO_ROOT / "personas" / "catalog.yaml")[
        "dona-marcia-58-mineira"
    ]


def carlos() -> Persona:
    return load_persona_catalog(REPO_ROOT / "personas" / "catalog.yaml")[
        "carlos-34-paulista"
    ]


# --- AppraisalDetector: individual triggers ------------------------------------


def test_detects_repeated_question():
    det = AppraisalDetector()
    assert "repetiu_pergunta" not in det.detect("O senhor pode confirmar o plano contratado da linha?")
    assert "repetiu_pergunta" in det.detect("Pode confirmar o plano contratado da linha, por favor?")


def test_different_questions_do_not_fire_repeat():
    det = AppraisalDetector()
    det.detect("Qual o motivo do seu contato hoje?")
    assert "repetiu_pergunta" not in det.detect("Qual a data da última recarga que a senhora fez?")


def test_detects_data_asked_again_and_suppresses_generic_repeat():
    det = AppraisalDetector()
    assert "pediu_dado_ja_informado" not in det.detect("Pode me informar o seu CPF?")
    events = det.detect("Pode me informar o seu CPF, por gentileza?")
    assert "pediu_dado_ja_informado" in events
    assert "repetiu_pergunta" not in events  # the stronger event wins, no double count


def test_data_categories_are_independent():
    det = AppraisalDetector()
    det.detect("Me informa o seu CPF?")
    assert "pediu_dado_ja_informado" not in det.detect("E qual o número do telefone?")
    assert "pediu_dado_ja_informado" in det.detect("Confirma o seu telefone de novo?")


@pytest.mark.parametrize(
    ("utterance", "event"),
    [
        ("Desculpa, não consegui entender, pode repetir?", "nao_entendeu_fala"),
        ("Aguarde um momento na linha, por favor.", "pediu_espera"),
        ("Vou transferir a senhora para o setor financeiro.", "transferiu"),
        ("Consta uma fatura em aberto na titularidade da linha.", "jargao_tecnico"),
        ("Sinto muito pelo transtorno, entendo sua situação.", "pediu_desculpa"),
    ],
)
def test_keyword_events(utterance, event):
    assert event in AppraisalDetector().detect(utterance)


def test_latency_and_state_change_signals():
    det = AppraisalDetector()
    events = det.detect("Certo.", state_changed=True, latency_s=5.0)
    assert "resolveu_etapa" in events
    assert "latencia_alta" in events
    assert "latencia_alta" not in det.detect("Certo.", latency_s=1.0)


# --- EmotionalStateMachine: curves, decay, thresholds ---------------------------


def test_default_neutral_stable_without_model():
    persona = marcia().model_copy(update={"emotionalModel": None})
    machine = EmotionalStateMachine.for_persona(persona, seed=42)
    for _ in range(6):
        rec = machine.update("Pode me informar o seu CPF, por favor?")
    assert machine.emotion == "calmo"
    assert machine.intensity == pytest.approx(0.2)  # no triggers, no decay: flat
    assert rec.action is None


def test_marcia_initial_state_from_catalog():
    machine = EmotionalStateMachine.for_persona(marcia(), seed=42)
    assert machine.emotion == "ansioso"
    assert machine.intensity == pytest.approx(0.30)
    assert machine.model.thresholds.pedirHumano == pytest.approx(0.75)


def test_decay_without_triggers():
    machine = EmotionalStateMachine.for_persona(marcia(), seed=42)
    machine.update("Certo, deixa eu verificar aqui para a senhora.")
    assert machine.intensity == pytest.approx(0.25)  # 0.30 - 0.05 decay


def test_repeated_cpf_request_escalates():
    machine = EmotionalStateMachine.for_persona(marcia(), seed=42)
    machine.update("Pode me informar o seu CPF?")  # first ask: decay
    rec = machine.update("Pode me informar o seu CPF, por gentileza?")
    assert "pediu_dado_ja_informado" in rec.events
    assert rec.delta == pytest.approx(0.20)
    assert machine.intensity == pytest.approx(0.45)  # 0.30 - 0.05 + 0.20


def test_emotion_switch_on_trigger():
    machine = EmotionalStateMachine.for_persona(marcia(), seed=42)
    rec = machine.update("Vou transferir a senhora para outro setor.")
    assert machine.emotion == "irritado"
    assert rec.delta == pytest.approx(0.25)


def test_apology_soothes():
    machine = EmotionalStateMachine.for_persona(marcia(), seed=42)
    rec = machine.update("Sinto muito pela demora, entendo sua preocupação.")
    assert rec.delta == pytest.approx(-0.10)
    assert machine.intensity == pytest.approx(0.20)


def test_thresholds_fire_pedir_humano_once_then_desligar():
    model = EmotionalModel(
        initialEmotion="ansioso",
        initialIntensity=0.6,
        decayPerTurn=-0.05,
        triggers=[EmotionalTrigger(on="pediu_dado_ja_informado", delta=0.2)],
        thresholds=EmotionalThresholds(pedirHumano=0.75, desligar=0.9),
    )
    machine = EmotionalStateMachine(model, seed=1)
    machine.update("Me informa o seu CPF?")  # 0.55 (decay)
    r2 = machine.update("Me informa o seu CPF de novo?")  # 0.75 -> pedir_humano
    assert r2.action == "pedir_humano"
    r3 = machine.update("Só mais uma vez o seu CPF, por favor?")  # 0.95 -> desligar
    assert r3.action == "desligar"
    # pedir_humano never fires twice
    assert [r.action for r in machine.history].count("pedir_humano") == 1


def test_intensity_clamped_to_unit_interval():
    model = EmotionalModel(
        initialIntensity=0.95,
        triggers=[EmotionalTrigger(on="pediu_dado_ja_informado", delta=0.5)],
    )
    machine = EmotionalStateMachine(model)
    machine.update("Seu CPF?")
    machine.update("Seu CPF de novo?")
    assert machine.intensity == 1.0


def test_same_conversation_same_curve_deterministic():
    turns = [
        "Bom dia, como posso ajudar?",
        "Pode me informar o seu CPF?",
        "Desculpa, pode me informar o seu CPF?",
        "Vou transferir a senhora para outro setor.",
    ]

    def run(seed):
        machine = EmotionalStateMachine.for_persona(marcia(), seed=seed)
        for text in turns:
            machine.update(text)
        return machine.curve()

    assert run(42) == run(42)
    # Rules are pure: the seed is recorded but does not perturb the curve.
    assert run(42) == run(7)


def test_curve_serializes_for_timeline():
    machine = EmotionalStateMachine.for_persona(carlos(), seed=42)
    machine.update("Aguarde um momento na linha.")
    curve = machine.curve()
    assert curve[0]["emotion"] == "apressado"
    assert curve[0]["intensity"] == pytest.approx(0.40)  # 0.25 + 0.15 pediu_espera
    assert "pediu_espera" in curve[0]["events"]


def test_prompt_block_reflects_state_and_threshold():
    model = EmotionalModel(
        initialEmotion="irritado",
        initialIntensity=0.7,
        triggers=[EmotionalTrigger(on="pediu_dado_ja_informado", delta=0.2)],
        thresholds=EmotionalThresholds(pedirHumano=0.75, desligar=0.95),
    )
    machine = EmotionalStateMachine(model)
    machine.update("Seu CPF?")
    machine.update("Seu CPF de novo?")  # crosses 0.75
    block = machine.prompt_block()
    assert "ESTADO EMOCIONAL" in block
    assert "irritado" in block
    assert "atendente humano" in block


def test_badge_format():
    machine = EmotionalStateMachine.for_persona(marcia(), seed=42)
    machine.update("Pode me informar seu CPF?")
    assert machine.badge() == "[ansioso 0.25 ↘]"


# --- LLMBrain integration -------------------------------------------------------


def test_llm_brain_sends_structured_emotional_state(monkeypatch):
    import httpx
    import json

    for key, value in {
        "HIVE_URL": "http://hive.test:3001",
        "HIVE_GATEWAY_TOKEN": "t",
        "VOIDR_ORG_ID": "org",
    }.items():
        monkeypatch.setenv(key, value)
    from voidr_echo_runner.brain import LLMBrain

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"text": "uai", "model": "m", "usage": {}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    brain = LLMBrain(marcia(), goal="ver saldo", client=client)
    brain.emotional = EmotionalStateMachine.for_persona(marcia(), seed=42)
    brain.emotional.update("Pode me informar o seu CPF?")
    brain.emotional.update("Pode me informar o seu CPF, por favor?")
    brain.reply("Pode me informar o seu CPF, por favor?")

    # Contract v2: structured field, no legacy block inside goalTemplate.
    state = captured["body"]["emotionalState"]
    assert state["emotion"] == "ansioso"
    assert state["intensity"] == pytest.approx(0.45)  # 0.30 - 0.05 + 0.20
    assert state["guidance"]
    assert "[ESTADO EMOCIONAL" not in captured["body"]["persona"]["goalTemplate"]


def test_llm_brain_falls_back_to_goal_template_on_400(monkeypatch):
    import httpx
    import json

    for key, value in {
        "HIVE_URL": "http://hive.test:3001",
        "HIVE_GATEWAY_TOKEN": "t",
        "VOIDR_ORG_ID": "org",
    }.items():
        monkeypatch.setenv(key, value)
    from voidr_echo_runner.brain import LLMBrain

    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "emotionalState" in body:  # old hive schema: rejects the new field
            return httpx.Response(400, json={"error": "Invalid persona-turn payload"})
        return httpx.Response(200, json={"text": "uai", "model": "m", "usage": {}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    brain = LLMBrain(marcia(), goal="ver saldo", client=client)
    brain.emotional = EmotionalStateMachine.for_persona(marcia(), seed=42)
    brain.emotional.update("Pode me informar o seu CPF?")

    assert brain.reply("Pode me informar o seu CPF?") == "uai"
    assert len(bodies) == 2  # structured attempt, then legacy fallback
    assert "emotionalState" not in bodies[1]
    assert "[ESTADO EMOCIONAL" in bodies[1]["persona"]["goalTemplate"]
    # Fallback is sticky: subsequent turns go straight to the legacy shape.
    brain.reply("Certo, anotei.")
    assert len(bodies) == 3
    assert "emotionalState" not in bodies[2]


def test_guidance_demands_human_at_threshold():
    model = EmotionalModel(
        initialEmotion="irritado",
        initialIntensity=0.7,
        triggers=[EmotionalTrigger(on="pediu_dado_ja_informado", delta=0.2)],
        thresholds=EmotionalThresholds(pedirHumano=0.75, desligar=0.95),
    )
    machine = EmotionalStateMachine(model)
    machine.update("Seu CPF?")
    machine.update("Seu CPF de novo?")  # crosses pedirHumano
    assert "DEVE exigir" in machine.guidance()
    assert "atendente humano" in machine.guidance()
