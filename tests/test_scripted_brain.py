"""ScriptedBrain: fala em 1ª pessoa e reformula quando a URA pede para repetir.

Bug real (execução com a persona "Milson"): goalTemplate autorado em 3ª pessoa
("Ele quer falar sobre a conta que está com valores que ele não reconhece...")
era falado VERBATIM pela persona, e a MESMA frase era repetida quando a URA
não entendia — narração em vez de encarnação.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voidr_echo_runner.brain import ScriptedBrain
from voidr_echo_runner.models import load_persona_catalog
from voidr_echo_runner.textutil import to_first_person

REPO_ROOT = Path(__file__).resolve().parents[1]

THIRD_PERSON_GOAL = (
    "Ele quer falar sobre a conta que está com valores que ele não reconhece "
    "e ele já entrou em contato com suporte antes e mesmo assim não funciona."
)


@pytest.fixture
def persona():
    base = load_persona_catalog(REPO_ROOT / "personas" / "catalog.yaml")[
        "dona-marcia-58-mineira"
    ]
    # Reproduz o payload real do bug: template SEM {goal}, redigido em 3ª pessoa,
    # sem floreios estocásticos (disfluência/vocabulário) para asserção exata.
    return base.model_copy(
        update={
            "goalTemplate": THIRD_PERSON_GOAL,
            "vocabulary": [],
            "speech": base.speech.model_copy(update={"disfluencyRate": 0.0}),
        }
    )


class TestToFirstPerson:
    def test_converte_goal_do_bug_real_para_primeira_pessoa(self):
        converted = to_first_person(THIRD_PERSON_GOAL)
        assert converted == (
            "Eu quero falar sobre a conta que está com valores que eu não reconheço "
            "e eu já entrei em contato com suporte antes e mesmo assim não funciona."
        )

    def test_converte_sujeito_cliente(self):
        assert to_first_person("O cliente quer a segunda via da fatura") == (
            "Eu quero a segunda via da fatura"
        )

    def test_nao_mexe_em_texto_ja_em_primeira_pessoa(self):
        goal = "Eu quero consultar o meu saldo antes de viajar."
        assert to_first_person(goal) == goal

    def test_nao_conjuga_verbo_de_outro_sujeito(self):
        # "a conta que está com valores" — o sujeito é a conta, não a persona.
        assert "que está com valores" in to_first_person(THIRD_PERSON_GOAL)


class TestScriptedBrainFirstPerson:
    def test_fala_o_goal_em_primeira_pessoa(self, persona):
        brain = ScriptedBrain(persona, goal="", seed=7)
        reply = brain.reply("Oi, eu sou o assistente virtual da Vivo. Como posso te ajudar hoje?")
        assert "Eu quero falar sobre a conta" in reply
        assert "eu não reconheço" in reply
        assert "Ele quer" not in reply

    def test_reformula_em_vez_de_repetir_quando_a_ura_nao_entende(self, persona):
        brain = ScriptedBrain(persona, goal="", seed=7)
        first = brain.reply("Como posso te ajudar hoje?")
        second = brain.reply("Desculpe, não entendi. Pode me contar de novo?")
        assert first != second
        # A reformulação mantém o problema, com outra moldura.
        assert "eu não reconheço" in second

    def test_reformulacao_deterministica_por_seed(self, persona):
        one = ScriptedBrain(persona, goal="", seed=7)
        two = ScriptedBrain(persona, goal="", seed=7)
        for brain in (one, two):
            brain.reply("Como posso te ajudar hoje?")
        assert one.reply("Pode me contar de novo?") == two.reply("Pode me contar de novo?")
