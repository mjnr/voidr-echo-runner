"""PII redaction (ARCHITECTURE.md section 10) — text layer.

Detects and replaces Brazilian PII in transcripts/timeline/report BEFORE any
artifact is persisted or any payload leaves the runner (POST /echo/sessions,
hive persona-turn history). Two layers:

1. **Deny-list (the critical one)**: values injected via `{{env.X}}` and the
   case's `massa` are KNOWN at runtime — they are redacted by equality and by
   fuzzy digit matching (spoken digit-by-digit, with/without punctuation)
   even when no generic pattern would fire. Massa must NEVER appear in clear.
2. **Generic detectors**: CPF (masked/plain/spelled-out, check digits), CNPJ,
   BR phone (+55/DDD/spelled), CEP, e-mail, card (Luhn), birthdates in
   context, plus any dictated sequence of >= 8 digits (potential ANI — rule
   from section 10.2, fail-closed).

Replacement uses typed placeholders that are consistent within a session:
the same entity always maps to the same token (`[CPF_1]`, `[TELEFONE_1]`,
`[MASSA_MOCK_ACCESS_CODE]`), keeping transcripts readable and auditable
without the values.

Engine choice: pure regex + check-digit/Luhn validators + a spoken-digit
scanner — no heavy NLP dependency. Microsoft Presidio (+ spaCy pt) remains a
documented opt-in second pass for NER entities (names, addresses); see the
README section.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

# ── text folding (length-preserving, keeps spans valid) ─────────────────────


def fold(text: str) -> str:
    """Lowercase + strip accents WITHOUT changing string length."""
    out = []
    for ch in text.lower():
        decomposed = unicodedata.normalize("NFD", ch)
        base = decomposed[0] if decomposed and not unicodedata.combining(decomposed[0]) else ch
        out.append(base)
    return "".join(out)


# ── validators ───────────────────────────────────────────────────────────────


def cpf_is_valid(digits: str) -> bool:
    if len(digits) != 11 or not digits.isdigit() or digits == digits[0] * 11:
        return False
    for size in (9, 10):
        total = sum(int(d) * (size + 1 - i) for i, d in enumerate(digits[:size]))
        check = (total * 10) % 11 % 10
        if check != int(digits[size]):
            return False
    return True


def cnpj_is_valid(digits: str) -> bool:
    if len(digits) != 14 or not digits.isdigit() or digits == digits[0] * 14:
        return False
    weights_1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    weights_2 = [6] + weights_1
    for size, weights in ((12, weights_1), (13, weights_2)):
        total = sum(int(d) * w for d, w in zip(digits[:size], weights))
        check = 11 - (total % 11)
        if check >= 10:
            check = 0
        if check != int(digits[size]):
            return False
    return True


def luhn_is_valid(digits: str) -> bool:
    if not digits.isdigit() or len(digits) < 12:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# ── number-run scanner (numeric chars + spelled digits) ──────────────────────

# "meia" = 6 in dictated BR numbers (phone/CPF reading).
DIGIT_WORDS = {
    "zero": "0",
    "um": "1",
    "uma": "1",
    "dois": "2",
    "duas": "2",
    "tres": "3",
    "quatro": "4",
    "cinco": "5",
    "seis": "6",
    "meia": "6",
    "sete": "7",
    "oito": "8",
    "nove": "9",
}
# Tokens allowed INSIDE a dictated sequence without contributing digits.
CONNECTOR_WORDS = {"e", "ponto", "traco", "hifen", "barra", "ddd"}
MAX_TOKEN_GAP = 3  # chars of separator ('.', '-', '/', ' ', etc.) between tokens

_TOKEN = re.compile(r"\d+|[^\W\d_]+", re.UNICODE)


@dataclass
class NumberRun:
    digits: str
    start: int
    end: int  # exclusive, in the original text

    def raw(self, text: str) -> str:
        return text[self.start : self.end]


def scan_number_runs(text: str) -> list[NumberRun]:
    """Group contiguous digits / spelled digits into dictated number runs.

    '390.533.447-05', 'três nove zero cinco três três...' and '(31) 98888-7777'
    all become a single run with the concatenated digit string and full span.
    """
    folded = fold(text)
    runs: list[NumberRun] = []
    current: list[tuple[str, int, int]] = []  # (digits, start, end) per token
    last_end = -1

    def close() -> None:
        nonlocal current
        contributing = [t for t in current if t[0]]
        if contributing:
            digits = "".join(t[0] for t in contributing)
            runs.append(
                NumberRun(digits=digits, start=contributing[0][1], end=contributing[-1][2])
            )
        current = []

    for match in _TOKEN.finditer(folded):
        token = match.group(0)
        if current and match.start() - last_end > MAX_TOKEN_GAP:
            close()
        if token.isdigit():
            contribution = token
        elif token in DIGIT_WORDS:
            contribution = DIGIT_WORDS[token]
        elif token in CONNECTOR_WORDS and current:
            contribution = ""
        else:
            close()
            last_end = match.end()
            continue
        current.append((contribution, match.start(), match.end()))
        last_end = match.end()
    close()
    return runs


# ── generic detectors ─────────────────────────────────────────────────────────

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
CEP_MASKED_RE = re.compile(r"^\d{5}-\d{3}$")
DATE_RE = re.compile(r"^\d{1,2}[/.]\d{1,2}[/.]\d{4}$")
BIRTH_CONTEXT_RE = re.compile(r"nascimento|nascid[ao]|nasceu|nasci\b")
CEP_CONTEXT_RE = re.compile(r"\bcep\b")
CONTEXT_WINDOW = 60

# DDDs are 11-99 with second digit != 0; good-enough heuristic: 2 digits, first 1-9, second 1-9.
_DDD_RE = re.compile(r"[1-9][1-9]")


def _looks_like_br_phone(digits: str) -> bool:
    if len(digits) in (12, 13) and digits.startswith("55"):
        digits = digits[2:]
    if len(digits) == 11:
        return bool(_DDD_RE.fullmatch(digits[:2])) and digits[2] == "9"
    if len(digits) == 10:
        return bool(_DDD_RE.fullmatch(digits[:2])) and digits[2] in "2345"
    return False


def classify_run(run: NumberRun, text: str) -> str | None:
    """Map a number run to a PII type (deny-list is handled before this)."""
    digits, raw = run.digits, run.raw(text)
    n = len(digits)
    if n == 11 and cpf_is_valid(digits):
        return "CPF"
    if n == 14 and cnpj_is_valid(digits):
        return "CNPJ"
    if raw.lstrip().startswith("+") or _looks_like_br_phone(digits):
        return "TELEFONE"
    if 13 <= n <= 19 and luhn_is_valid(digits):
        return "CARTAO"
    context = fold(text[max(0, run.start - CONTEXT_WINDOW) : run.end + CONTEXT_WINDOW])
    if n == 8:
        if CEP_MASKED_RE.fullmatch(raw) or CEP_CONTEXT_RE.search(context):
            return "CEP"
        if DATE_RE.fullmatch(raw) and BIRTH_CONTEXT_RE.search(context):
            return "DATA_NASCIMENTO"
    if DATE_RE.fullmatch(raw):
        # Dates only count as PII in birth context; a due date is not an ANI.
        return "DATA_NASCIMENTO" if BIRTH_CONTEXT_RE.search(context) else None
    if n >= 8:
        # Section 10.2: any dictated sequence >= 8 digits is a potential ANI.
        return "NUMERO"
    return None


# ── session ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    type: str
    placeholder: str


@dataclass
class DenyEntry:
    name: str  # env var / massa key (NOT sensitive)
    value: str
    digits: str  # digits-only form ("" when the value is not digit-heavy)


_URL_RE = re.compile(r"^(wss?|https?)://", re.IGNORECASE)
MIN_DENY_LEN = 3
MIN_DENY_DIGITS = 4


class RedactionSession:
    """Stateful redactor: consistent placeholders for the whole call/session."""

    def __init__(self, deny: dict[str, str] | None = None):
        # (type, normalized value) -> placeholder
        self._placeholders: dict[tuple[str, str], str] = {}
        self._type_counts: dict[str, int] = {}
        self._deny: list[DenyEntry] = []
        for name, value in (deny or {}).items():
            self.add_deny_value(name, value)

    # -- deny-list ------------------------------------------------------------

    def add_deny_value(self, name: str, value: str) -> None:
        value = str(value).strip()
        if len(value) < MIN_DENY_LEN or _URL_RE.match(value):
            return
        digits = "".join(c for c in value if c.isdigit())
        if any(e.value == value or (digits and e.digits == digits) for e in self._deny):
            return  # same secret registered under another name (e.g. massa + dtmf)
        self._deny.append(
            DenyEntry(
                name=re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper(),
                value=value,
                digits=digits if len(digits) >= MIN_DENY_DIGITS else "",
            )
        )

    # -- placeholder bookkeeping ------------------------------------------------

    def _placeholder(self, pii_type: str, key: str) -> str:
        bucket = (pii_type, key)
        if bucket not in self._placeholders:
            if pii_type.startswith("MASSA_"):
                token = f"[{pii_type}]"
            else:
                self._type_counts[pii_type] = self._type_counts.get(pii_type, 0) + 1
                token = f"[{pii_type}_{self._type_counts[pii_type]}]"
            self._placeholders[bucket] = token
        return self._placeholders[bucket]

    # -- detection -------------------------------------------------------------

    def _deny_candidates(self, text: str) -> list[tuple[int, int, str, str]]:
        """Deny-list matches only: exact (fold-insensitive) + dictated digits."""
        candidates: list[tuple[int, int, str, str]] = []
        folded = fold(text)
        for entry in self._deny:
            needle = fold(entry.value)
            start = 0
            while (idx := folded.find(needle, start)) != -1:
                candidates.append((idx, idx + len(needle), f"MASSA_{entry.name}", entry.value))
                start = idx + len(needle)
        for run in scan_number_runs(text):
            deny_hit = next(
                (e for e in self._deny if e.digits and e.digits in run.digits), None
            )
            if deny_hit is not None:
                # Fail-closed: the whole dictated run is redacted, not just
                # the matched sub-sequence.
                candidates.append(
                    (run.start, run.end, f"MASSA_{deny_hit.name}", deny_hit.value)
                )
        return candidates

    def find_spans(self, text: str) -> list[Span]:
        """All PII spans in `text` (original offsets), deny-list first."""
        # (start, end, type, entity key) — placeholders are only allocated for
        # the spans that survive overlap resolution (keeps report() honest).
        candidates = self._deny_candidates(text)
        deny_ranges = {(start, end) for start, end, _t, _k in candidates}

        # 2. number runs: generic classifiers (deny hits already collected win
        # overlap resolution; skip them so counts stay identical to before).
        for run in scan_number_runs(text):
            if (run.start, run.end) in deny_ranges:
                continue
            pii_type = classify_run(run, text)
            if pii_type is not None:
                candidates.append((run.start, run.end, pii_type, run.digits))

        # 3. e-mail
        for match in EMAIL_RE.finditer(text):
            candidates.append((match.start(), match.end(), "EMAIL", match.group(0).lower()))

        return [
            Span(start, end, pii_type, self._placeholder(pii_type, key))
            for start, end, pii_type, key in _drop_overlaps(candidates)
        ]

    # -- application -------------------------------------------------------------

    def redact_with_spans(self, text: str) -> tuple[str, list[Span]]:
        spans = self.find_spans(text)
        redacted = text
        for span in sorted(spans, key=lambda s: s.start, reverse=True):
            redacted = redacted[: span.start] + span.placeholder + redacted[span.end :]
        return redacted, spans

    def redact(self, text: str) -> str:
        return self.redact_with_spans(text)[0]

    def redact_deny(self, text: str) -> str:
        """Deny-list-only redaction (no generic detectors) — used by the live
        event stream, where massa must never leave the process but generic
        false positives would garble the real-time transcript."""
        spans = [
            Span(start, end, pii_type, self._placeholder(pii_type, key))
            for start, end, pii_type, key in _drop_overlaps(self._deny_candidates(text))
        ]
        redacted = text
        for span in sorted(spans, key=lambda s: s.start, reverse=True):
            redacted = redacted[: span.start] + span.placeholder + redacted[span.end :]
        return redacted

    def redact_deep(self, obj: Any) -> Any:
        """Recursively redact every string in a JSON-like structure."""
        if isinstance(obj, str):
            return self.redact(obj)
        if isinstance(obj, list):
            return [self.redact_deep(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.redact_deep(v) for v in obj)
        if isinstance(obj, dict):
            return {k: self.redact_deep(v) for k, v in obj.items()}
        return obj

    # -- audit -------------------------------------------------------------------

    def report(self) -> dict[str, int]:
        """piiRedactionReport: distinct entities per type — never the values."""
        counts: dict[str, int] = {}
        for pii_type, _key in self._placeholders:
            counts[pii_type] = counts.get(pii_type, 0) + 1
        return dict(sorted(counts.items()))


def _drop_overlaps(
    candidates: list[tuple[int, int, str, str]],
) -> list[tuple[int, int, str, str]]:
    """Deny-list spans win over generic ones; earlier/longer wins otherwise."""

    def priority(span: tuple[int, int, str, str]) -> tuple[int, int, int]:
        start, end, pii_type, _key = span
        deny = 0 if pii_type.startswith("MASSA_") else 1
        return (deny, start, -(end - start))

    kept: list[tuple[int, int, str, str]] = []
    for span in sorted(candidates, key=priority):
        if all(span[1] <= k[0] or span[0] >= k[1] for k in kept):
            kept.append(span)
    return sorted(kept)


# ── call-level helpers ─────────────────────────────────────────────────────────


def build_session_for_case(case: Any) -> RedactionSession:
    """Deny-list from the case: {{env.*}} values captured at load/resolve time,
    `massa` values and dial-plan DTMF payloads (access codes, ANIs)."""
    session = RedactionSession()
    for name, value in (getattr(case, "resolved_secrets", None) or {}).items():
        session.add_deny_value(name, value)
    for key, value in _flatten(getattr(case, "massa", {}) or {}):
        session.add_deny_value(key, str(value))
    for step in case.dial_plan.dtmf_steps:
        digits = step.send.rstrip("#*")
        if len(digits) >= MIN_DENY_DIGITS and digits.isdigit():
            session.add_deny_value("DTMF", digits)
    return session


def _flatten(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            items.extend(_flatten(value, f"{prefix}_{key}" if prefix else str(key)))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            items.extend(_flatten(value, f"{prefix}_{i}"))
    elif obj is not None:
        items.append((prefix or "value", obj))
    return items


def redact_call_result(call: Any, session: RedactionSession) -> None:
    """In-place redaction of a CallResult: transcript, timeline, trajectory."""
    for entry in call.transcript:
        entry["text"] = session.redact(entry["text"])
    call.timeline[:] = [
        {k: (session.redact_deep(v) if k not in ("ts",) else v) for k, v in event.items()}
        for event in call.timeline
    ]
    for t in call.trajectory:
        t.utterance = session.redact(t.utterance)
    if call.transport_error:
        call.transport_error = session.redact(call.transport_error)
