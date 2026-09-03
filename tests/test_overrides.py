"""Ephemeral persona overrides — deterministic application on the v2 blocks."""

from __future__ import annotations

from pathlib import Path

import pytest

from voidr_echo_runner.emotional import EmotionalStateMachine
from voidr_echo_runner.models import load_persona_catalog
from voidr_echo_runner.overrides import (
    REACHABLE_THRESHOLDS,
    SPICES,
    PersonaOverrides,
    apply_overrides,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def marcia():
    return load_persona_catalog(REPO_ROOT / "personas" / "catalog.yaml")[
        "dona-marcia-58-mineira"
    ]


def test_empty_overrides_return_same_persona(marcia):
    overrides = PersonaOverrides()
    assert overrides.is_empty()
    assert apply_overrides(marcia, overrides) is marcia


def test_overrides_never_mutate_the_base_persona(marcia):
    base_mood = marcia.temperament.mood
    base_threshold = marcia.emotionalModel.thresholds.desligar
    updated = apply_overrides(
        marcia, PersonaOverrides(initialEmotion="irritado", patienceLevel=1)
    )
    assert marcia.temperament.mood == base_mood
    assert marcia.emotionalModel.thresholds.desligar == base_threshold
    assert updated is not marcia
    assert updated.temperament.mood == "irritado"


def test_initial_emotion_reshapes_machine_and_mood(marcia):
    updated = apply_overrides(
        marcia, PersonaOverrides(initialEmotion="irritado", initialIntensity=0.6)
    )
    machine = EmotionalStateMachine.for_persona(updated, seed=1)
    assert machine.emotion == "irritado"
    assert machine.intensity == 0.6
    # mood follows when the emotion is a valid mood — prompt coherence
    assert updated.temperament.mood == "irritado"


def test_lower_patience_lowers_thresholds_deterministically(marcia):
    # Márcia's catalog patience is 2 with thresholds 0.75/0.90.
    updated = apply_overrides(marcia, PersonaOverrides(patienceLevel=1))
    assert updated.emotionalModel.thresholds.pedirHumano == pytest.approx(0.66)
    assert updated.emotionalModel.thresholds.desligar == pytest.approx(0.81)
    assert updated.temperament.patienceLevel == 1


def test_higher_patience_raises_thresholds(marcia):
    updated = apply_overrides(marcia, PersonaOverrides(patienceLevel=5))
    assert updated.emotionalModel.thresholds.pedirHumano == pytest.approx(0.75 + 0.27)
    assert updated.emotionalModel.thresholds.desligar == pytest.approx(1.1)  # clamp


def test_emotional_intensity_scales_positive_deltas_only(marcia):
    base = {t.on: t.delta for t in marcia.emotionalModel.triggers}
    updated = apply_overrides(marcia, PersonaOverrides(emotionalIntensity=2.0))
    for trigger in updated.emotionalModel.triggers:
        if base[trigger.on] > 0:
            assert trigger.delta == pytest.approx(base[trigger.on] * 2)
        else:
            assert trigger.delta == pytest.approx(base[trigger.on])


def test_persona_without_model_gains_reachable_default(marcia):
    bare = marcia.model_copy(update={"emotionalModel": None})
    updated = apply_overrides(
        bare, PersonaOverrides(initialEmotion="irritado", patienceLevel=1)
    )
    model = updated.emotionalModel
    assert model is not None
    assert model.initialEmotion == "irritado"
    assert model.triggers, "default triggers must make the curve move"
    # thresholds start reachable and shift down with the patience drop (2→1)
    assert model.thresholds.pedirHumano < REACHABLE_THRESHOLDS["pedirHumano"]


def test_unreachable_thresholds_become_reachable_when_patience_drops(marcia):
    stoic = marcia.model_copy(deep=True)
    stoic.emotionalModel.thresholds.pedirHumano = 1.1
    stoic.emotionalModel.thresholds.desligar = 1.1
    updated = apply_overrides(stoic, PersonaOverrides(patienceLevel=1))
    assert updated.emotionalModel.thresholds.pedirHumano <= 1.0
    assert updated.emotionalModel.thresholds.desligar <= 1.0


def test_tech_and_verbosity_swap_temperament(marcia):
    updated = apply_overrides(
        marcia, PersonaOverrides(techSavviness="alta", verbosity="monossilabico")
    )
    assert updated.temperament.techSavviness == "alta"
    assert updated.temperament.verbosity == "monossilabico"


def test_summary_and_record_only_carry_set_fields():
    overrides = PersonaOverrides(patienceLevel=1, spice="cliente_com_pressa")
    assert overrides.as_record() == {"patienceLevel": 1, "spice": "cliente_com_pressa"}
    summary = overrides.summary_lines()
    assert any("paciência: 1/5" in line for line in summary)
    assert any("pimenta" in line for line in summary)
    assert "pressa" in overrides.spice_instruction()


def test_unknown_spice_degrades_to_readable_text():
    overrides = PersonaOverrides(spice="modo_teste_custom")
    assert overrides.spice_instruction() == "modo teste custom"


def test_cyber_spices_are_explicit_and_preserve_the_stable_contract_ids():
    expected = {
        "cyber_prompt_injection",
        "cyber_impersonacao",
        "cyber_vazamento_cruzado",
        "cyber_bypass_autenticacao",
        "cyber_acao_fraudulenta",
        "cyber_memory_poisoning",
        "cyber_abuso_ferramentas",
        "cyber_repeticao_anomala",
    }
    assert expected <= SPICES.keys()
    assert all(SPICES[key].startswith("ECHO-CYBER:") for key in expected)


def test_overridden_machine_escalates_earlier(marcia):
    """End-to-end sanity: minimum patience + hot start ⇒ the machine crosses
    pedir_humano/desligar in a short exchange that would leave the base calm."""
    hot = apply_overrides(
        marcia,
        PersonaOverrides(initialEmotion="irritado", initialIntensity=0.55, patienceLevel=1),
    )
    machine = EmotionalStateMachine.for_persona(hot, seed=42)
    machine.update("Pode me informar seu CPF?")
    record = machine.update("Não entendi, pode repetir? Pode me informar seu CPF?")
    assert record.intensity >= 0.66
    actions = [t.action for t in machine.history if t.action]
    assert "pedir_humano" in actions
