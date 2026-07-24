"""Persona brains.

`ScriptedBrain`: deterministic, seeded rule engine (default for tests) that
renders utterances from the persona's goalTemplate + vocabulary + the agent's
last utterance.

`LLMBrain`: LLM-driven persona via the hive's synchronous gateway
(`POST {HIVE_URL}/echo/persona-turn`). Per ARCHITECTURE.md section 8.5, ALL
LLM consumption is centralized in the hive — this runner holds no LLM keys;
model routing (DeepSeek v4 Pro -> Sonnet escalation) and billing live there.
"""

from __future__ import annotations

import os
import random
from typing import Any, Protocol

from .models import Persona
from .textutil import keyword_matches


class PersonaBrain(Protocol):
    def reply(self, agent_utterance: str) -> str:
        """Produce the persona's next utterance given the agent's last turn."""
        ...


# (trigger keywords on the agent utterance, canned persona intent) — first match wins
_RULES: list[tuple[list[str], str]] = [
    (
        ["titular", "confirmar os seus dados", "confirmar seus dados"],
        "confirm_identity",
    ),
    (
        ["como posso te ajudar", "como posso ajudar", "o que voce precisa", "me contar de novo"],
        "state_goal",
    ),
    (
        [
            "posso enviar",
            "posso te mandar",
            "posso mandar",
            "prefere o link",
            "quer aproveitar",
            "quer resolver",
            "quer fazer",
        ],
        "accept_offer",
    ),
]

_RESPONSES: dict[str, str] = {
    "confirm_identity": "Sim, sou eu, o titular da linha, pode confirmar.",
    "accept_offer": "Sim, quero sim, pode mandar, por favor.",
    "ack": "Entendi. Pode continuar, por favor.",
}


class ScriptedBrain:
    """Deterministic v0 brain. Same persona + same seed => same utterances."""

    def __init__(self, persona: Persona, goal: str, seed: int):
        self.persona = persona
        self.goal = goal
        self.rng = random.Random(seed)

    def reply(self, agent_utterance: str) -> str:
        intent = "ack"
        for triggers, rule_intent in _RULES:
            if any(keyword_matches(t, agent_utterance) for t in triggers):
                intent = rule_intent
                break
        if intent == "state_goal":
            base = self.persona.goalTemplate.format(goal=self.goal)
        else:
            base = _RESPONSES[intent]
        return self._stylize(base)

    def _stylize(self, utterance: str) -> str:
        """Seeded regional/disfluency flourishes; never changes the intent."""
        if self.rng.random() < self.persona.speech.disfluencyRate:
            utterance = "É... " + utterance[0].lower() + utterance[1:]
        if self.persona.vocabulary and self.rng.random() < 0.5:
            flourish = self.rng.choice(self.persona.vocabulary)
            utterance = f"{utterance.rstrip('.')}. {flourish.capitalize()}."
        return utterance


class HiveError(RuntimeError):
    """A persona-turn call to the hive failed (payload, PII guard or gateway)."""


GENERIC_JOURNEY_STATE: dict[str, Any] = {
    "flowSlug": "conversa-livre",
    "currentState": "conversa",
    "expects": [],
}


class LLMBrain:
    """Persona brain backed by the hive LLM gateway (no LLM keys here).

    Each turn POSTs the persona profile + journey state + conversation history
    to `{HIVE_URL}/echo/persona-turn` and returns the in-character reply.
    Mutate `journey_state` between turns to keep the persona contextualized
    (the chat mode does this with the keyword state classifier).
    """

    REQUIRED_ENV = ("HIVE_URL", "HIVE_GATEWAY_TOKEN", "VOIDR_ORG_ID")
    TIMEOUT_S = 20.0

    def __init__(
        self,
        persona: Persona,
        goal: str,
        seed: int | None = None,
        *,
        escalate: bool = False,
        client: Any | None = None,
    ):
        import httpx

        missing = [name for name in self.REQUIRED_ENV if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                f"LLMBrain requires env vars: {', '.join(missing)} "
                "(the hive is the LLM gateway — see .env.example). "
                "Use the default ScriptedBrain for offline runs (--brain scripted)."
            )
        self.persona = persona
        self.goal = goal
        self.seed = seed
        self.escalate_default = escalate
        self.base_url = os.environ["HIVE_URL"].rstrip("/")
        self.org_id = os.environ["VOIDR_ORG_ID"]
        self._client = client or httpx.Client(
            timeout=self.TIMEOUT_S,
            headers={"Authorization": f"Bearer {os.environ['HIVE_GATEWAY_TOKEN']}"},
        )
        self.history: list[dict[str, str]] = []
        self.journey_state: dict[str, Any] = dict(GENERIC_JOURNEY_STATE)
        # The hive rejects clear PII in history/goalTemplate (422) — every
        # outgoing payload is redacted (ARCHITECTURE.md section 10). Callers
        # with known massa replace this with the case-level session
        # (redaction.build_session_for_case) so deny-list values are covered.
        from .redaction import RedactionSession

        self.redaction = RedactionSession()
        # Optional deterministic emotional state machine (PERSONAS-SOTA P0.3).
        # The OWNER of the conversation loop (CallRunner/chat) updates it per
        # agent turn; this brain sends the current state as the structured
        # `emotionalState` field (contract v2). Against an older hive that
        # 400s on the field, we fall back once to the legacy workaround of
        # embedding the [ESTADO EMOCIONAL] block in goalTemplate.
        from .emotional import EmotionalStateMachine

        self.emotional: EmotionalStateMachine | None = None
        self._legacy_emotional = False
        # Ephemeral execution overrides (overrides.PersonaOverrides): the
        # already-applied persona carries the mechanical effects (thresholds,
        # temperament); this block tells the PROMPT the variation explicitly
        # so adherence is verifiable by the fidelity judge. Against an older
        # hive that 400s on the new fields we degrade once (contract v2.1).
        self.execution_overrides: Any | None = None
        self._legacy_contract = False
        self.total_cost_usd = 0.0
        self.last_model: str | None = None
        self.last_usage: dict[str, Any] | None = None

    def _persona_payload(self) -> dict[str, Any]:
        # Objective decoupling (mission delivery 4): the persona says WHO the
        # person is; the objective comes from the JOURNEY (case goal) and is
        # sent as the separate `journeyGoal` field. goalTemplate survives only
        # as legacy style/fallback: `{goal}` placeholders are still resolved,
        # and an empty template falls back to the journey goal so older hives
        # (min 1 char) keep working.
        goal_template = self.persona.goalTemplate
        if self.goal and "{goal}" in goal_template:
            goal_template = goal_template.format(goal=self.goal)
        if not goal_template.strip():
            goal_template = self.goal or "Resolver o objetivo desta ligação."
        if self._legacy_contract and self.goal:
            # Old hive without `journeyGoal`: make the journey objective win
            # by prepending it to the only field the old prompt reads.
            goal_template = (
                f"Objetivo desta ligação (definido pela jornada): {self.goal}\n\n"
                f"{goal_template}"
            )
        if self.emotional is not None and self._legacy_emotional:
            # Legacy workaround for hives without the structured field.
            goal_template = f"{goal_template}\n\n{self.emotional.prompt_block()}"
        # Identity fields are optional in the contract, but when `name` is set
        # the hive locks the persona identity in the prompt (no more LLM
        # inventing "sou o João" for a persona called Cida).
        identity: dict[str, Any] = {}
        if self.persona.name:
            identity["name"] = self.persona.name
        if self.persona.age is not None:
            identity["age"] = self.persona.age
        if self.persona.gender:
            identity["gender"] = self.persona.gender
        if self.persona.profile is not None:
            profile = {
                key: value
                for key, value in (
                    ("occupation", self.persona.profile.occupation),
                    ("context", self.persona.profile.context),
                    # the hive contract takes freeTraits as one string; the
                    # catalog keeps a YAML list for readability
                    ("freeTraits", "; ".join(self.persona.profile.freeTraits)),
                )
                if value
            }
            if profile:
                identity["profile"] = profile
        # v2 blocks (contract v2, all optional): identity.facts is the judge's
        # canonical truth; OCEAN and behaviors drive the prompt v2 blocks;
        # emotionalModel sends only the subset the hive accepts (triggers and
        # decay stay runner-side — the machine lives here).
        v2: dict[str, Any] = {}
        if self.persona.identity is not None:
            ident = self.persona.identity
            v2["identity"] = {
                key: value
                for key, value in (
                    ("fullName", ident.fullName),
                    ("shortName", ident.shortName),
                    ("backstory", ident.backstory),
                    ("facts", dict(ident.facts)),
                )
                if value
            }
        if self.persona.psychometrics is not None:
            v2["psychometrics"] = self.persona.psychometrics.model_dump()
        if self.persona.behaviors is not None:
            v2["behaviors"] = self.persona.behaviors.model_dump()
        if self.persona.emotionalModel is not None:
            model = self.persona.emotionalModel
            v2["emotionalModel"] = {
                "initialEmotion": model.initialEmotion,
                "initialIntensity": model.initialIntensity,
                "thresholds": {
                    "pedirHumano": model.thresholds.pedirHumano,
                    "desligar": model.thresholds.desligar,
                },
            }
        # Full v2 speech block: regional particles, hesitations, tu/você axis
        # and curated exemplars. Sending only disfluencyRate was the gap that
        # flattened every persona into the same generic caller.
        speech: dict[str, Any] = {"disfluencyRate": self.persona.speech.disfluencyRate}
        if self.persona.speech.interruptionPolicy:
            speech["interruptionPolicy"] = self.persona.speech.interruptionPolicy
        if self.persona.speech.discourseMarkers:
            speech["discourseMarkers"] = list(self.persona.speech.discourseMarkers)
        if self.persona.speech.fillerInventory:
            speech["fillerInventory"] = list(self.persona.speech.fillerInventory)
        if self.persona.speech.pronouns:
            speech["pronouns"] = self.persona.speech.pronouns
        if self.persona.speech.exemplars:
            speech["exemplars"] = list(self.persona.speech.exemplars)[:8]
        return {
            "id": self.persona.id,
            **identity,
            **v2,
            "demographics": {
                "ageBand": self.persona.demographics.ageBand,
                "region": self.persona.demographics.region,
            },
            "temperament": {
                "mood": self.persona.temperament.mood,
                "patienceLevel": self.persona.temperament.patienceLevel,
                "techSavviness": self.persona.temperament.techSavviness,
                "verbosity": self.persona.temperament.verbosity,
                "intentNoise": self.persona.temperament.intentNoise,
            },
            "speech": speech,
            "goalTemplate": goal_template,
            "vocabulary": list(self.persona.vocabulary),
        }

    def _build_payload(self, options: dict[str, Any]) -> dict[str, Any]:
        persona_payload = self._persona_payload()
        persona_payload["goalTemplate"] = self.redaction.redact(persona_payload["goalTemplate"])
        payload = {
            "organizationId": self.org_id,
            "persona": persona_payload,
            "journeyState": self.journey_state,
            "history": [
                {"role": turn["role"], "text": self.redaction.redact(turn["text"])}
                for turn in self.history
            ],
            **({"options": options} if options else {}),
        }
        if self.emotional is not None and not self._legacy_emotional:
            # Structured field (contract v2) — precedence over the legacy
            # goalTemplate block on the hive side.
            payload["emotionalState"] = {
                "emotion": self.emotional.emotion,
                "intensity": self.emotional.intensity,
                "guidance": self.emotional.guidance(),
            }
        if not self._legacy_contract:
            # Contract v2.1: objective decoupled from the persona + ephemeral
            # execution variation. Older hives 400 on unknown fields → the
            # caller degrades once via _legacy_contract.
            if self.goal:
                payload["journeyGoal"] = self.redaction.redact(self.goal)
            if self.execution_overrides is not None:
                summary = self.execution_overrides.summary_lines()
                spice = self.execution_overrides.spice_instruction()
                block: dict[str, Any] = {}
                if summary:
                    block["summary"] = summary
                if spice:
                    block["spice"] = spice
                if block:
                    payload["executionOverrides"] = block
        return payload

    def _post(self, url: str, payload: dict[str, Any]) -> Any:
        import httpx

        response = None
        last_exc: Exception | None = None
        for attempt in (1, 2):  # short timeout + 1 retry (transport/5xx)
            try:
                response = self._client.post(url, json=payload)
            except httpx.TransportError as exc:
                last_exc = exc
                continue
            if response.status_code < 500 or attempt == 2:
                break
        if response is None:
            raise HiveError(
                f"hive unreachable at {url}: {type(last_exc).__name__}: {last_exc}"
            ) from last_exc
        return response

    def take_turn(self, agent_utterance: str, *, escalate: bool | None = None) -> dict[str, Any]:
        """One persona turn; returns the full hive response (text/model/usage)."""
        self.history.append({"role": "agent", "text": agent_utterance})
        options: dict[str, Any] = {}
        effective_escalate = self.escalate_default if escalate is None else escalate
        if effective_escalate:
            options["escalate"] = True
        if self.seed is not None:
            options["seed"] = self.seed
        url = f"{self.base_url}/echo/persona-turn"

        payload = self._build_payload(options)
        response = self._post(url, payload)
        if response.status_code == 400 and "emotionalState" in payload:
            # Old hive schema without the structured field: fall back to the
            # legacy goalTemplate block for the rest of the session.
            self._legacy_emotional = True
            payload = self._build_payload(options)
            response = self._post(url, payload)
        if response.status_code == 400 and (
            "journeyGoal" in payload or "executionOverrides" in payload
        ):
            # Old hive schema without the v2.1 fields: embed the journey goal
            # into goalTemplate for the rest of the session.
            self._legacy_contract = True
            payload = self._build_payload(options)
            response = self._post(url, payload)
        if response.status_code != 200:
            detail = ""
            try:
                detail = response.json().get("error", "")
            except Exception:  # noqa: BLE001 — non-JSON error body
                detail = response.text[:200]
            hint = {
                400: "payload inválido para o contrato persona-turn",
                401: "HIVE_GATEWAY_TOKEN inválido",
                422: "PII em claro detectado — redija transcripts (<CPF>, <TELEFONE>)",
                502: "gateway LLM do hive indisponível (LiteLLM/chave não configurados?)",
            }.get(response.status_code, "")
            raise HiveError(
                f"persona-turn falhou ({response.status_code}"
                + (f" — {hint}" if hint else "")
                + (f"): {detail}" if detail else ")")
            )

        data = response.json()
        self.history.append({"role": "persona", "text": data["text"]})
        self.last_model = data.get("model")
        self.last_usage = data.get("usage") or {}
        self.total_cost_usd += float(self.last_usage.get("costUsd") or 0.0)
        return data

    def reply(self, agent_utterance: str) -> str:
        return self.take_turn(agent_utterance)["text"]


def build_brain(kind: str, persona: Persona, goal: str, seed: int) -> PersonaBrain:
    if kind == "scripted":
        return ScriptedBrain(persona, goal, seed)
    if kind == "llm":
        return LLMBrain(persona, goal, seed)
    raise ValueError(f"unknown brain kind {kind!r} (expected 'scripted' or 'llm')")
