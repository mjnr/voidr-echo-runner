"""Human realism engine (EXEC-REALISM): timing, memory imperfection and massa.

Research grounding (see mission report for full sources):

- Turn-taking latency: human floor-transfer offsets are unimodal and
  right-skewed with modal gaps of 100–300 ms in casual talk (Stivers et al.
  2009, PNAS; Levinson & Torreira 2015), but *task* answers on the phone —
  especially ones that require recall — routinely take 700–1900 ms, with a
  median around 1200 ms perceived as human-like for embodied agents (CHI'26
  "Quantifying Latencies"). Turn-taking speed is well modeled by a gamma
  distribution whose parameters shift with speaker traits (SIGDial 2025
  "Modeling Turn-Taking Speed and Speaker Characteristics").
- Memory imperfection: out-of-the-box LLM user simulators exhibit
  super-human recall; realistic simulation requires injecting forgetting
  behavior explicitly ("Simulating Human Memory with LLMs", 2026; τ-voice /
  non-collaborative user simulators, arXiv 2509.23124).
- Disfluencies while retrieving: fillers and silent pauses cluster where
  conceptual planning is hard — e.g. dictating a document number — and
  self-repair is expected (Interspeech 2015 "Micro-structure of
  disfluencies"; PMC4203439).

Everything here is DETERMINISTIC per (persona, seed): same persona + seed +
conversation ⇒ same lapses, same delays. No LLM calls; prompt-visible
behavior travels to the hive as per-turn directives (contract v2.2).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from random import Random
from typing import Any

from .emotional import _DATA_CATEGORIES
from .models import Persona
from .textutil import normalize

# ── massa (personal data) ─────────────────────────────────────────────────────

# Placeholder the persona speaks; the runner substitutes it OUTSIDE the LLM
# (the hive PII guard must never see the real value).
MASSA_PLACEHOLDER = re.compile(r"\{\{\s*(?:massa|env)\.([A-Za-z0-9_]+)\s*\}\}")

# {{env.X}} inside a massa VALUE: journey massa fields may point at an
# environment secret (massa editor pattern) — resolved against
# ENVIRONMENT_PARAMS when the bag is loaded.
_ENV_PLACEHOLDER = re.compile(r"\{\{\s*env\.([A-Za-z0-9_]+)\s*\}\}")

# Canonical massa keys and how they map to the agent's data-request
# categories (emotional._DATA_CATEGORIES keys) + spoken labels.
MASSA_FIELDS: dict[str, dict[str, str]] = {
    "cpf": {"category": "cpf", "label": "CPF"},
    "birthDate": {"category": "nascimento", "label": "data de nascimento"},
    "fullName": {"category": "nome", "label": "nome completo"},
    "phone": {"category": "telefone", "label": "número de telefone"},
    "accessCode": {"category": "codigo", "label": "código de acesso"},
    "address": {"category": "endereco", "label": "endereço"},
}

# Aliases accepted in ECHO_MASSA / identity.facts (case-insensitive, folded).
_KEY_ALIASES: dict[str, str] = {
    "cpf": "cpf",
    "birthdate": "birthDate",
    "datanascimento": "birthDate",
    "data_nascimento": "birthDate",
    "nascimento": "birthDate",
    "fullname": "fullName",
    "nome": "fullName",
    "nomecompleto": "fullName",
    "nome_completo": "fullName",
    "phone": "phone",
    "telefone": "phone",
    "celular": "phone",
    "ani": "phone",
    "accesscode": "accessCode",
    "codigo": "accessCode",
    "codigoacesso": "accessCode",
    "endereco": "address",
    "address": "address",
    "cep": "address",
}


def _canonical_key(raw: str) -> str | None:
    folded = re.sub(r"[^a-z]", "", raw.lower().replace("ç", "c"))
    return _KEY_ALIASES.get(folded) or _KEY_ALIASES.get(raw.strip())


@dataclass
class MassaFacts:
    """Personal test data of THIS call's persona (never sent to the LLM).

    Contract: `ENVIRONMENT_PARAMS.ECHO_MASSA` is a JSON object with fields
    like {"cpf": "...", "birthDate": "...", "fullName": "..."} — managed by
    the service (massa management is another agent's delivery). Fallback:
    values mined from the persona's `identity.facts` / `massaProfile`.
    """

    values: dict[str, str] = field(default_factory=dict)
    source: str = "none"  # environment | persona | none

    def __bool__(self) -> bool:
        return bool(self.values)

    def placeholder(self, key: str) -> str:
        return f"{{{{massa.{key}}}}}"

    def key_for_category(self, category: str) -> str | None:
        for key, meta in MASSA_FIELDS.items():
            if meta["category"] == category and key in self.values:
                return key
        return None

    def personal_data_lines(self) -> list[dict[str, str]]:
        """Payload block for the persona-turn prompt: label + placeholder,
        NEVER the value (the hive PII guard rejects clear PII by design)."""
        return [
            {"label": MASSA_FIELDS[key]["label"], "placeholder": self.placeholder(key)}
            for key in MASSA_FIELDS
            if key in self.values
        ]

    def resolve_placeholders(self, text: str) -> tuple[str, list[str]]:
        """Replace {{massa.X}} (and {{env.MASSA_X}}-style) with real values.

        Returns (resolved_text, substituted_keys). Unknown keys are left
        as-is — a loud artifact beats a silent wrong value.
        """
        used: list[str] = []

        def _sub(match: re.Match[str]) -> str:
            raw = match.group(1)
            key = raw if raw in self.values else None
            if key is None:
                candidate = _canonical_key(raw.removeprefix("MASSA_").removeprefix("massa_"))
                if candidate in self.values:
                    key = candidate
            if key is None:
                return match.group(0)
            used.append(key)
            return self.values[key]

        return MASSA_PLACEHOLDER.sub(_sub, text), used

    @classmethod
    def from_params(cls, params: dict[str, str]) -> "MassaFacts | None":
        raw = params.get("ECHO_MASSA")
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        values: dict[str, str] = {}
        for key, value in data.items():
            if value is None:
                continue
            text = str(value).strip()
            # Massa fields may reference environment secrets ({{env.X}}) —
            # resolve them against ENVIRONMENT_PARAMS here so both the dial
            # plan and the speech substitution consume the real value.
            if "{{" in text:
                text = _ENV_PLACEHOLDER.sub(
                    lambda m: params.get(m.group(1), m.group(0)), text
                ).strip()
            # "{{" guard: a STILL-unresolved placeholder is not a value —
            # speaking/dialing it literally is worse than having no massa.
            if not text or "{{" in text:
                continue
            canonical = _canonical_key(str(key)) or (
                str(key) if str(key) in MASSA_FIELDS else None
            )
            if canonical:
                values.setdefault(canonical, text)
            # The journey's OWN key is preserved too: dial plans and steps
            # reference {{massa.<key>}} with journey key names (e.g.
            # telefone_ura), not only the canonical personal-data card keys.
            values.setdefault(str(key), text)
        return cls(values=values, source="environment") if values else None

    @classmethod
    def from_persona(cls, persona: Persona) -> "MassaFacts | None":
        """Fallback: mine identity.facts for personal-data-shaped entries."""
        facts = dict(persona.identity.facts) if persona.identity else {}
        values: dict[str, str] = {}
        for key, value in facts.items():
            canonical = _canonical_key(key)
            if canonical and str(value).strip() and "{{" not in str(value):
                values[canonical] = str(value).strip()
        return cls(values=values, source="persona") if values else None

    @classmethod
    def resolve(cls, params: dict[str, str], persona: Persona) -> "MassaFacts":
        return (
            cls.from_params(params)
            or cls.from_persona(persona)
            or cls(values={}, source="none")
        )


# ── memory imperfection ───────────────────────────────────────────────────────

# Base probability that the persona does NOT remember the value on the spot
# and needs to "go get it" (wallet, drawer, phone notes). CPF and access
# codes are the classic lapses; one's own name never lapses.
_BASE_LAPSE_PROBABILITY: dict[str, float] = {
    "cpf": 0.45,
    "codigo": 0.65,
    "nascimento": 0.15,
    "telefone": 0.35,
    "endereco": 0.20,
    "nome": 0.0,
}

_LOW_LITERACY = ("analfabeto", "rudimentar")


def _age_from_persona(persona: Persona) -> int:
    if persona.age is not None:
        return persona.age
    band = persona.demographics.ageBand
    return {"18-25": 22, "26-40": 33, "41-60": 50, "60+": 68}.get(band, 40)


@dataclass
class TurnPlan:
    """What the humanizer decided for THIS persona turn."""

    directives: list[str] = field(default_factory=list)
    memory_lapse: bool = False
    lapse_category: str | None = None
    extra_delay_s: float = 0.0


class Humanizer:
    """Deterministic per-call human realism: memory lapses + response timing.

    The OWNER of the conversation loop (CallRunner) calls `plan_turn` on each
    agent turn (before the brain reply), `finalize_reply` on the persona text
    and `reply_delay_s` to obtain the humanized latency.
    """

    def __init__(
        self,
        persona: Persona,
        seed: int,
        massa: MassaFacts | None = None,
        *,
        timing_enabled: bool = True,
    ):
        self.persona = persona
        self.seed = seed
        self.massa = massa or MassaFacts()
        self.timing_enabled = timing_enabled
        self._rng = Random(f"{persona.id}#{seed}#humanize")
        self._age = _age_from_persona(persona)
        self._inaf = persona.literacy.inafLevel if persona.literacy else None
        self._retrieved: set[str] = set()
        self.substituted_keys: list[str] = []

    # -- memory ----------------------------------------------------------------

    def lapse_probability(self, category: str) -> float:
        prob = _BASE_LAPSE_PROBABILITY.get(category, 0.1)
        if prob <= 0.0:
            return 0.0
        if self._age >= 60:
            prob += 0.20
        elif self._age >= 45:
            prob += 0.10
        if self._inaf in _LOW_LITERACY:
            prob += 0.15
        conscientiousness = (
            self.persona.psychometrics.conscientiousness if self.persona.psychometrics else 50
        )
        if conscientiousness >= 70:
            prob -= 0.10
        elif conscientiousness <= 35:
            prob += 0.10
        return max(0.0, min(0.9, prob))

    def _requested_categories(self, agent_text: str) -> list[str]:
        norm = normalize(agent_text)
        return [
            category
            for category, keywords in _DATA_CATEGORIES.items()
            if any(kw in norm for kw in keywords)
        ]

    def plan_turn(self, agent_text: str) -> TurnPlan:
        plan = TurnPlan()
        for category in self._requested_categories(agent_text):
            key = self.massa.key_for_category(category)
            if key is None:
                continue
            label = MASSA_FIELDS[key]["label"]
            placeholder = self.massa.placeholder(key)
            if category in self._retrieved:
                # She has the document in hand now — repeat without drama.
                plan.directives.append(
                    f"Se pedirem seu {label} de novo, você JÁ está com ele em mãos: "
                    f"dite de novo, sem hesitar, usando exatamente {placeholder}."
                )
                continue
            self._retrieved.add(category)
            if self._rng.random() < self.lapse_probability(category):
                plan.memory_lapse = True
                plan.lapse_category = category
                plan.extra_delay_s += 1.8 + self._rng.random() * 2.4
                plan.directives.append(
                    f"O atendente pediu seu {label} e você NÃO lembra de cabeça: "
                    f'comece hesitando de verdade ("ai, peraí...", "deixa eu pegar aqui...", '
                    f'"onde foi que eu anotei..."), demore procurando (use reticências), '
                    f"pode se corrigir no meio, e SÓ ENTÃO dite o dado falando exatamente "
                    f"o placeholder {placeholder} — nunca invente números."
                )
            else:
                plan.directives.append(
                    f"Quando informar seu {label}, dite com naturalidade usando "
                    f"exatamente o placeholder {placeholder} — nunca invente números."
                )
        return plan

    # -- reply post-processing ---------------------------------------------------

    def finalize_reply(self, reply: str) -> str:
        """Substitute massa placeholders with the real values (outside the LLM)."""
        if not self.massa:
            return reply
        resolved, used = self.massa.resolve_placeholders(reply)
        self.substituted_keys.extend(used)
        return resolved

    def scripted_prefix(self, plan: TurnPlan) -> str:
        """Hesitation prefix for the deterministic ScriptedBrain (no LLM)."""
        if not plan.memory_lapse:
            return ""
        options = (
            "Ai, peraí... deixa eu pegar aqui... ",
            "Hã... peraí um instantinho, deixa eu procurar... ",
            "É... deixa eu ver onde eu anotei isso... ",
        )
        return options[self._rng.randrange(len(options))]

    # -- timing ------------------------------------------------------------------

    def reply_delay_s(
        self,
        reply: str,
        plan: TurnPlan | None = None,
        *,
        emotion_intensity: float | None = None,
    ) -> float:
        """Humanized response latency for this turn (seconds), deterministic.

        Gamma-distributed (SIGDial 2025) around a persona-conditioned base:
        older personas answer slower; agitated personas snap back faster;
        short confirmations come quicker; memory retrieval adds seconds.
        """
        base = 0.9
        if self._age >= 60:
            base *= 1.5
        elif self._age >= 45:
            base *= 1.2
        if self._inaf in _LOW_LITERACY:
            base *= 1.15
        stripped = reply.strip()
        if len(stripped) <= 25:
            base *= 0.6
        elif len(stripped) >= 160:
            base *= 1.25
        if emotion_intensity is not None and emotion_intensity >= 0.6:
            base *= 0.75
        # shape k=2 ⇒ right-skewed, mean == base, mode == base/2.
        sample = self._rng.gammavariate(2.0, base / 2.0)
        extra = plan.extra_delay_s if plan else 0.0
        return round(max(0.35, min(9.0, sample + extra)), 3)

    def config_record(self) -> dict[str, Any]:
        """Auditable snapshot for report meta / session payload."""
        return {
            "timingEnabled": self.timing_enabled,
            "massaSource": self.massa.source,
            "massaFields": sorted(self.massa.values.keys()),
            "seed": self.seed,
        }
