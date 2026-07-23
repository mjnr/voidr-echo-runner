"""PII redaction — text layer. All fixtures are SYNTHETIC (CPF/CNPJ check
digits computed by algorithm, phones in fake ranges, no real data)."""

from __future__ import annotations

import pytest

from voidr_echo_runner.redaction import (
    RedactionSession,
    build_session_for_case,
    cnpj_is_valid,
    cpf_is_valid,
    luhn_is_valid,
    redact_call_result,
    scan_number_runs,
)


def make_cpf(base9: str) -> str:
    """Generate a VALID synthetic CPF from 9 base digits (check digits by algorithm)."""
    digits = [int(d) for d in base9]
    for size in (9, 10):
        total = sum(d * (size + 1 - i) for i, d in enumerate(digits[:size]))
        digits.append((total * 10) % 11 % 10)
    return "".join(map(str, digits))


def make_cnpj(base12: str) -> str:
    digits = [int(d) for d in base12]
    weights_1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    weights_2 = [6] + weights_1
    for weights in (weights_1, weights_2):
        total = sum(d * w for d, w in zip(digits, weights))
        check = 11 - (total % 11)
        digits.append(0 if check >= 10 else check)
    return "".join(map(str, digits))


CPF = make_cpf("390533447")  # classic doc example base
CPF_MASKED = f"{CPF[:3]}.{CPF[3:6]}.{CPF[6:9]}-{CPF[9:]}"
CPF_SPOKEN = " ".join(
    {"0": "zero", "1": "um", "2": "dois", "3": "três", "4": "quatro",
     "5": "cinco", "6": "meia", "7": "sete", "8": "oito", "9": "nove"}[d]
    for d in CPF
)
CNPJ = make_cnpj("112223330001")


# ── validators ────────────────────────────────────────────────────────────────


def test_cpf_validator():
    assert cpf_is_valid(CPF)
    assert not cpf_is_valid(CPF[:-1] + str((int(CPF[-1]) + 1) % 10))  # wrong check digit
    assert not cpf_is_valid("11111111111")  # repdigit
    assert not cpf_is_valid("123")


def test_cnpj_validator():
    assert cnpj_is_valid(CNPJ)
    assert not cnpj_is_valid(CNPJ[:-1] + str((int(CNPJ[-1]) + 1) % 10))
    assert not cnpj_is_valid("11111111111111")


def test_luhn_validator():
    assert luhn_is_valid("4539148803436467")  # standard synthetic test number
    assert not luhn_is_valid("4539148803436468")


# ── generic detectors ─────────────────────────────────────────────────────────


@pytest.fixture
def session():
    return RedactionSession()


def test_cpf_masked_and_plain_share_placeholder(session):
    out1 = session.redact(f"meu CPF é {CPF_MASKED}")
    out2 = session.redact(f"anotei o documento {CPF}")
    assert out1 == "meu CPF é [CPF_1]"
    assert out2 == "anotei o documento [CPF_1]"  # same entity => same token


def test_cpf_spelled_out(session):
    out = session.redact(f"o cpf é {CPF_SPOKEN}, isso")
    assert "[CPF_1]" in out
    for word in ("zero", "nove", "meia"):
        assert word not in out.split("[CPF_1]")[1] if "[CPF_1]" in out else True
    assert CPF not in out


def test_invalid_cpf_still_redacted_as_numero(session):
    bad = CPF[:-1] + str((int(CPF[-1]) + 1) % 10)
    out = session.redact(f"documento {bad[:3]}.{bad[3:6]}.{bad[6:9]}-{bad[9:]}")
    assert bad not in out.replace(".", "").replace("-", "")
    assert "[NUMERO_1]" in out  # fail-closed: 11 digits, invalid check => generic


def test_cnpj(session):
    masked = f"{CNPJ[:2]}.{CNPJ[2:5]}.{CNPJ[5:8]}/{CNPJ[8:12]}-{CNPJ[12:]}"
    assert session.redact(f"empresa CNPJ {masked}") == "empresa CNPJ [CNPJ_1]"


@pytest.mark.parametrize(
    "text",
    [
        "liga no (31) 98888-7777",
        "meu número é +55 31 98888-7777",
        "anota aí: 31 98888 7777",
        "telefone três um nove oito oito oito oito sete sete sete sete",
    ],
)
def test_phone_variants(text, session):
    out = session.redact(text)
    assert "8888" not in out and "oito" not in out
    assert "[TELEFONE_1]" in out


def test_cep_and_email(session):
    out = session.redact("mora no CEP 30130-010, e-mail joao.teste@example.com")
    assert out == "mora no CEP [CEP_1], e-mail [EMAIL_1]"


def test_card_luhn(session):
    out = session.redact("cartão 4539 1488 0343 6467 final 67")
    assert "4539" not in out
    assert "[CARTAO_1]" in out


def test_birthdate_only_in_context(session):
    assert session.redact("nascida em 12/03/1968") == "nascida em [DATA_NASCIMENTO_1]"
    # a due date is not PII — must stay readable
    assert session.redact("a fatura vence em 15/08/2026") == "a fatura vence em 15/08/2026"


def test_long_digit_sequence_is_potential_ani(session):
    out = session.redact("digite 11900000001 depois da mensagem")
    assert "11900000001" not in out
    assert "[NUMERO_1]" in out or "[TELEFONE_1]" in out


def test_no_false_positives_on_ordinary_text(session):
    for text in (
        "seu saldo é de R$ 25,90, válido até 30/09",
        "aguarde 2 minutos na linha 1",
        "o plano custa 49,99 por mês",
        "protocolo curto 1234",
    ):
        assert session.redact(text) == text


# ── deny-list (massas) — false negatives are unacceptable ─────────────────────


@pytest.fixture
def deny_session():
    return RedactionSession(
        deny={"MOCK_ACCESS_CODE": "919021552", "ANI": "11900000001", "TITULAR": "Márcia Souza"}
    )


@pytest.mark.parametrize(
    "spoken",
    [
        "o código é 919021552",
        "o código é 919.021.552",
        "o código é 91 90 21 552",
        "nove um nove zero dois um cinco cinco dois",
        "nove um nove, zero dois um, cinco cinco dois",
        "9 1 9 0 2 1 5 5 2",
    ],
)
def test_deny_digits_all_spoken_forms(spoken, deny_session):
    out = deny_session.redact(spoken)
    assert "[MASSA_MOCK_ACCESS_CODE]" in out
    assert "919021552" not in out.replace(" ", "").replace(".", "").replace(",", "")
    assert "nove" not in out and "9" not in out


def test_deny_value_inside_longer_dictation(deny_session):
    # ANI dictated with a trailing '#' and spaced digits
    out = deny_session.redact("digitei 1 1 9 0 0 0 0 0 0 0 1 # na URA")
    assert "[MASSA_ANI]" in out
    assert "1 1 9" not in out


def test_deny_name_case_and_accent_insensitive(deny_session):
    out = deny_session.redact("a titular MARCIA SOUZA confirmou")
    assert "[MASSA_TITULAR]" in out
    assert "SOUZA" not in out


def test_deny_wins_over_generic_type(deny_session):
    # 11900000001 is phone-shaped, but it is massa => massa placeholder
    out = deny_session.redact("confirmando o 11900000001")
    assert out == "confirmando o [MASSA_ANI]"


def test_report_counts_distinct_entities(deny_session):
    deny_session.redact("código 919021552")
    deny_session.redact(f"cpf {CPF_MASKED} de novo o cpf {CPF}")
    report = deny_session.report()
    assert report["MASSA_MOCK_ACCESS_CODE"] == 1
    assert report["CPF"] == 1  # same CPF twice => one entity


# ── deep / call-level integration ─────────────────────────────────────────────


def test_redact_deep_nested(deny_session):
    obj = {
        "a": [f"cpf {CPF}", {"b": "code 919021552"}],
        "n": 42,
        "t": ("x", "ani 11900000001"),
    }
    out = deny_session.redact_deep(obj)
    assert out["a"][0] == "cpf [CPF_1]"
    assert out["a"][1]["b"] == "code [MASSA_MOCK_ACCESS_CODE]"
    assert out["n"] == 42
    assert out["t"][1] == "ani [MASSA_ANI]"


def test_build_session_for_case_and_redact_call_result(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCK_ACCESS_CODE", "919021552")
    from voidr_echo_runner.models import VoiceTestCase
    from voidr_echo_runner.runner import CallResult

    case_yaml = tmp_path / "case.yaml"
    case_yaml.write_text(
        """
id: tc-x
persona: {base: p, variant_seed: 1}
massa: {ani: "11900000001"}
dial_plan:
  dtmf_steps:
    - {send: "{{env.MOCK_ACCESS_CODE}}"}
    - {send: "11900000001#"}
journey_flow: f.json
goal: g
""",
        encoding="utf-8",
    )
    case = VoiceTestCase.load(case_yaml)
    assert case.resolved_secrets == {"MOCK_ACCESS_CODE": "919021552"}

    session = build_session_for_case(case)
    call = CallResult()
    call.transcript.append(
        {"index": 0, "speaker": "tester", "text": f"meu número é 11900000001 e cpf {CPF}", "ts": 1}
    )
    call.timeline.append({"ts": 1, "type": "dtmf_sent", "digits": "919021552"})
    redact_call_result(call, session)
    assert call.transcript[0]["text"] == "meu número é [MASSA_ANI] e cpf [CPF_1]"
    assert call.timeline[0]["digits"] == "[MASSA_MOCK_ACCESS_CODE]"


# ── scanner details ───────────────────────────────────────────────────────────


def test_scan_number_runs_groups_spoken_digits():
    runs = scan_number_runs("o código é nove um nove zero dois um cinco cinco dois, tá?")
    assert [r.digits for r in runs] == ["919021552"]


def test_scan_number_runs_breaks_on_non_digit_words():
    runs = scan_number_runs("tenho dois filhos e moro há cinco anos aqui")
    # "dois" / "cinco" become isolated 1-digit runs — never classified as PII
    assert all(len(r.digits) < 8 for r in runs)
