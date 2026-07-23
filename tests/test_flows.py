from pathlib import Path

import pytest

from voidr_echo_runner.classifier import KeywordStateClassifier
from voidr_echo_runner.flows import load_journey_flow

FLOWS_DIR = Path(__file__).resolve().parents[1] / "flows"


def test_parses_consulta_saldo_flow():
    flow = load_journey_flow(FLOWS_DIR / "consulta-saldo-v1.json")
    assert flow.id == "consulta-saldo-v1"
    assert "saudacao" in flow.states
    assert flow.states["envio_deep_link"].terminal
    assert flow.states["envio_deep_link"].evidence == ["deep_link_sent"]
    assert "identificacao" in flow.states["saudacao"].next
    assert flow.terminal_states() >= {"envio_deep_link", "jornada_errada"}
    assert flow.deviation_rules


def test_rejects_unknown_next_state(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"id": "x", "states": {"a": {"next": ["missing"]}}}')
    with pytest.raises(ValueError, match="unknown next state"):
        load_journey_flow(bad)


def test_classifier_maps_agent_turns_to_states():
    flow = load_journey_flow(FLOWS_DIR / "consulta-saldo-v1.json")
    classifier = KeywordStateClassifier(flow)
    assert (
        classifier.classify("Oi! Eu sou o assistente virtual da Vivo. Como posso te ajudar hoje?")
        == "saudacao"
    )
    assert (
        classifier.classify("Antes, preciso confirmar os seus dados: você é o titular da linha?")
        == "identificacao"
    )
    assert (
        classifier.classify("O seu saldo atual é de vinte e cinco reais, válido até o fim do mês.")
        == "diagnostico_saldo"
    )
    assert classifier.classify("bom dia, tudo bem?") is None


def test_classifier_flags_wrong_journey_utterances():
    flow = load_journey_flow(FLOWS_DIR / "consulta-saldo-v1.json")
    classifier = KeywordStateClassifier(flow)
    wrong = (
        "Verifiquei aqui e a sua linha está com um bloqueio por débito: existe uma "
        "fatura em aberto no valor de oitenta e nove reais."
    )
    assert classifier.classify(wrong) == "jornada_errada"


def test_classifier_accent_insensitive():
    flow = load_journey_flow(FLOWS_DIR / "bloqueio-financeiro-v1.json")
    classifier = KeywordStateClassifier(flow)
    assert (
        classifier.classify("existe uma fatura em aberto e as ligações estão bloqueadas")
        == "diagnostico_bloqueio"
    )
