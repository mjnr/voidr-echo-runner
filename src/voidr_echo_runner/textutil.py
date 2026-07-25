"""Text normalization and {{env.*}} placeholder resolution."""

from __future__ import annotations

import os
import re
import unicodedata

ENV_PLACEHOLDER = re.compile(r"\{\{\s*env\.([A-Za-z0-9_]+)\s*\}\}")


def normalize(text: str) -> str:
    """Lowercase and strip accents ('não' -> 'nao') for keyword matching."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def keyword_matches(keyword: str, text: str) -> bool:
    """Multi-word keywords match as substring; single words on word boundary."""
    norm_kw = normalize(keyword)
    norm_text = normalize(text)
    if " " in norm_kw:
        return norm_kw in norm_text
    return re.search(rf"\b{re.escape(norm_kw)}\b", norm_text) is not None


# ── Conversão 3ª → 1ª pessoa (bug real: goalTemplate autorado em 3ª pessoa
#    — "Ele quer falar sobre a conta que está com valores que ele não
#    reconhece..." — era falado VERBATIM pelo ScriptedBrain, e a persona
#    narrava em vez de encarnar). A conversão é determinística e cobre os
#    padrões de autoria comuns: sujeito "ele/ela/o cliente" + verbos do
#    domínio na 3ª pessoa do singular. Verbos fora da tabela ficam como
#    estão (ex.: "a conta que está..." — sujeito é a conta, não a persona).

# Sujeito em 3ª pessoa: "ele/ela", "o cliente", "cliente de baixo letramento".
# Lookbehinds evitam converter "cliente" como complemento ("a conta do
# cliente", "ao cliente") — aí o dono é outro sintagma, não o sujeito.
_THIRD_PERSON_SUBJECT = re.compile(
    r"(?<![dD][oae]\s)(?<![aA]o\s)(?<!pel[oa]\s)"
    r"\b(?:(?:o|a)\s+)?cliente(?:\s+(?:de|do|da)\s+[\wáéíóúâêôãõç]+(?:\s+[\wáéíóúâêôãõç]+)?)?\b"
    r"|\bele\b|\bela\b",
    re.IGNORECASE,
)

_POSSESSIVE_1P = {"seu": "meu", "seus": "meus", "sua": "minha", "suas": "minhas"}

# 3ª pessoa do singular → 1ª pessoa do singular (verbos comuns de autoria de
# goals de atendimento). Só aplicado imediatamente após o sujeito "eu"
# (com adverbios intermediários tipo "não", "já", "nunca").
_VERB_1P: dict[str, str] = {
    "quer": "quero",
    "queria": "queria",
    "deseja": "desejo",
    "precisa": "preciso",
    "pede": "peço",
    "pediu": "pedi",
    "liga": "ligo",
    "ligou": "liguei",
    "consulta": "consulto",
    "consultou": "consultei",
    "reconhece": "reconheço",
    "reconheceu": "reconheci",
    "entrou": "entrei",
    "entra": "entro",
    "tem": "tenho",
    "tinha": "tinha",
    "teve": "tive",
    "recebeu": "recebi",
    "recebe": "recebo",
    "pagou": "paguei",
    "paga": "pago",
    "contratou": "contratei",
    "cancelou": "cancelei",
    "cancela": "cancelo",
    "usou": "usei",
    "usa": "uso",
    "sabe": "sei",
    "soube": "soube",
    "acha": "acho",
    "viu": "vi",
    "tentou": "tentei",
    "tenta": "tento",
    "foi": "fui",
    "fez": "fiz",
    "faz": "faço",
    "quis": "quis",
    "informou": "informei",
    "solicitou": "solicitei",
    "solicita": "solicito",
}

_ADVERBS = r"(?:n[aã]o|j[aá]|nunca|ainda|tamb[eé]m|sempre|s[oó]|mesmo\s+assim)"
_EU_VERB = re.compile(
    rf"\b([Ee]u)((?:\s+{_ADVERBS})*)\s+({'|'.join(_VERB_1P)})\b",
)


def to_first_person(text: str) -> str:
    """Reescreve um goal autorado em 3ª pessoa para a fala em 1ª pessoa.

    "Ele quer falar sobre a conta ... que ele não reconhece e ele já entrou
    em contato" → "Eu quero falar sobre a conta ... que eu não reconheço e
    eu já entrei em contato". Texto já em 1ª pessoa passa intocado.
    """

    def _subject(match: re.Match[str]) -> str:
        return "Eu" if match.group(0)[0].isupper() else "eu"

    converted, subjects_found = _THIRD_PERSON_SUBJECT.subn(_subject, text)
    if subjects_found:
        # Com sujeito em 3ª pessoa, "seu/sua" do goal se refere à própria
        # persona ("o cliente quer consultar seu saldo" → "meu saldo").
        def _possessive(match: re.Match[str]) -> str:
            replacement = _POSSESSIVE_1P[match.group(0).lower()]
            return replacement.capitalize() if match.group(0)[0].isupper() else replacement

        converted = re.sub(
            rf"\b(?:{'|'.join(_POSSESSIVE_1P)})\b", _possessive, converted, flags=re.IGNORECASE
        )

    def _verb(match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group(2)} {_VERB_1P[match.group(3).lower()]}"

    return _EU_VERB.sub(_verb, converted)


def resolve_env_placeholders(value: str, captured: dict[str, str] | None = None) -> str:
    """Resolve {{env.NAME}} placeholders from os.environ, failing loudly.

    When `captured` is given, every substituted (NAME, value) pair is recorded
    there — the PII redaction deny-list is built from these known values.
    """

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if resolved is None:
            raise KeyError(
                f"environment variable {name!r} required by placeholder "
                f"{match.group(0)!r} is not set. For local runs against the mock, "
                f"e.g.: export MOCK_ACCESS_CODE=919021552"
            )
        if captured is not None:
            captured[name] = resolved
        return resolved

    return ENV_PLACEHOLDER.sub(_sub, value)


def resolve_placeholders_deep(obj, captured: dict[str, str] | None = None):
    """Recursively resolve placeholders in a YAML-loaded structure."""
    if isinstance(obj, str):
        return resolve_env_placeholders(obj, captured)
    if isinstance(obj, list):
        return [resolve_placeholders_deep(v, captured) for v in obj]
    if isinstance(obj, dict):
        return {k: resolve_placeholders_deep(v, captured) for k, v in obj.items()}
    return obj
