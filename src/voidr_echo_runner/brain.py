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
        # agent turn; this brain only injects the current state in the prompt.
        # The hive persona-turn contract has no emotionalState field yet, so
        # the block rides inside goalTemplate (see README contract note).
        from .emotional import EmotionalStateMachine

        self.emotional: EmotionalStateMachine | None = None
        self.total_cost_usd = 0.0
        self.last_model: str | None = None
        self.last_usage: dict[str, Any] | None = None

    def _persona_payload(self) -> dict[str, Any]:
        goal_template = self.persona.goalTemplate
        if self.goal and "{goal}" in goal_template:
            goal_template = goal_template.format(goal=self.goal)
        if self.emotional is not None:
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
        return {
            "id": self.persona.id,
            **identity,
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
            "speech": {"disfluencyRate": self.persona.speech.disfluencyRate},
            "goalTemplate": goal_template,
            "vocabulary": list(self.persona.vocabulary),
        }

    def take_turn(self, agent_utterance: str, *, escalate: bool | None = None) -> dict[str, Any]:
        """One persona turn; returns the full hive response (text/model/usage)."""
        import httpx

        self.history.append({"role": "agent", "text": agent_utterance})
        options: dict[str, Any] = {}
        effective_escalate = self.escalate_default if escalate is None else escalate
        if effective_escalate:
            options["escalate"] = True
        if self.seed is not None:
            options["seed"] = self.seed
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
        url = f"{self.base_url}/echo/persona-turn"

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
