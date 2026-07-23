"""Persona brains.

v0 ships `ScriptedBrain`: a deterministic, seeded rule engine that renders
utterances from the persona's goalTemplate + vocabulary + the agent's last
utterance. `LLMBrain` is the structured stub for the LLM-driven persona
(phase 2), gated behind OPENAI_API_KEY / GEMINI_API_KEY.
"""

from __future__ import annotations

import os
import random
from typing import Protocol

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


class LLMBrain:
    """Stub for the LLM-driven persona brain (not implemented in v0).

    Planned wiring: pipecat LLM service (OpenAILLMService / GoogleLLMService)
    prompted with temperament + goalTemplate + vocabulary + journey state,
    low temperature; variation comes from the persona seed, not sampling.
    """

    def __init__(self, persona: Persona, goal: str, seed: int):
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLMBrain requires OPENAI_API_KEY or GEMINI_API_KEY. "
                "Use the default ScriptedBrain for offline runs (--brain scripted)."
            )
        raise NotImplementedError(
            "LLMBrain is a phase-2 feature; the interface is stable (PersonaBrain), "
            "only the implementation is pending."
        )

    def reply(self, agent_utterance: str) -> str:  # pragma: no cover
        raise NotImplementedError


def build_brain(kind: str, persona: Persona, goal: str, seed: int) -> PersonaBrain:
    if kind == "scripted":
        return ScriptedBrain(persona, goal, seed)
    if kind == "llm":
        return LLMBrain(persona, goal, seed)
    raise ValueError(f"unknown brain kind {kind!r} (expected 'scripted' or 'llm')")
