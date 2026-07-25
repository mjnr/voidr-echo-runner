"""Deterministic emotional appraisal state machine (PERSONAS-SOTA.md, P0.3).

Principle: evolving mood is NOT an adjective in the prompt — it is state
`{emotion, intensity}` kept by the RUNNER, updated by deterministic appraisal
rules over the Vivo agent's turns (CPM-MultiAgent/EmoLLM simplified to rules
for production), injected into the persona-turn prompt every turn and
recorded in the timeline as an auditable curve.

Determinism: every rule is pure (keyword/similarity matching over the turn +
history, plus runner-measured signals like latency and journey progress), so
the same conversation always yields the same curve. The `seed` is stored for
the record and reserved for future stochastic behaviors (P1.2 non-collaborative
probabilities); no RNG is consulted today.

No LLM calls happen here — appraisal must be free and per-turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import EmotionalModel, Persona
from .textutil import keyword_matches, normalize

HIGH_LATENCY_S = 3.5
# Jaccard similarity over normalized token sets for "agent repeated himself".
REPEAT_SIMILARITY = 0.7

# Data categories the agent may request; asking the same category twice fires
# `pediu_dado_ja_informado` ("eu JÁ falei isso, uai").
_DATA_CATEGORIES: dict[str, tuple[str, ...]] = {
    "cpf": ("cpf",),
    "telefone": ("numero da linha", "numero do telefone", "seu telefone", "seu celular", "seu numero"),
    "nome": ("nome completo", "seu nome"),
    "nascimento": ("data de nascimento", "quando voce nasceu"),
    "endereco": ("endereco", "seu cep"),
    "codigo": ("codigo de acesso", "codigo de verificacao"),
}

# (event, keywords over the normalized agent utterance) — substring match.
_KEYWORD_EVENTS: dict[str, tuple[str, ...]] = {
    "nao_entendeu_fala": (
        "nao entendi",
        "nao consegui entender",
        "pode repetir",
        "poderia repetir",
        "repete por favor",
    ),
    "pediu_espera": (
        "aguarde",
        "um momento",
        "momentinho",
        "so um instante",
        "permaneca na linha",
        "aguarda um",
    ),
    "transferiu": (
        "transferir",
        "vou te passar",
        "vou passar voce",
        "outro setor",
        "outra area",
        "departamento responsavel",
    ),
    "jargao_tecnico": (
        "protocolo",
        "titularidade",
        "fatura em aberto",
        "restricao cadastral",
        "regularizacao",
        "backoffice",
    ),
    "pediu_desculpa": (
        "desculpa",
        "desculpe",
        "perdao",
        "sinto muito",
        "lamento",
        "entendo sua",
        "entendo a sua",
        "imagino como",
    ),
}

_QUESTION_HINTS = ("?", "qual ", "quais ", "pode me informar", "pode confirmar", "me informa")


# Politeness/filler words ignored when comparing questions — "pode confirmar o
# plano?" and "pode confirmar o plano, por favor, senhor?" are the same ask.
_STOPWORDS = frozenset(
    "por favor gentileza senhor senhora que seu sua meu minha entao aqui".split()
)


def _tokens(text: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", normalize(text))
    return frozenset(w for w in words if len(w) > 2 and w not in _STOPWORDS)


def _looks_like_question(text: str) -> bool:
    norm = normalize(text)
    return any(hint in norm or (hint == "?" and "?" in text) for hint in _QUESTION_HINTS)


class AppraisalDetector:
    """Turns agent utterances + runner signals into appraisal events.

    Rules/keywords first (deterministic, zero cost); journey progress comes
    from the existing keyword state classifier via the `state_changed` flag.

    E4 (DEVIATIONS-METHODOLOGY §4.4): `jargon_terms` extends the static
    `jargao_tecnico` keyword list with the terms THIS persona does not know
    (the `unknown`/`confused` partitions of her resolved glossary vocabulary)
    — the agent saying "roaming" to a persona that doesn't know "roaming"
    fires the same appraisal event, which nudges her to ask for an
    explanation (the U1 probe the judge closes with glossaryTermId).
    """

    def __init__(self, jargon_terms: tuple[str, ...] = ()) -> None:
        self._questions: list[frozenset[str]] = []
        self._data_requested: set[str] = set()
        self._jargon_terms = tuple(t for t in jargon_terms if t and len(t.strip()) >= 3)

    def detect(
        self,
        agent_text: str,
        *,
        state_changed: bool = False,
        latency_s: float | None = None,
    ) -> list[str]:
        events: list[str] = []
        norm = normalize(agent_text)

        if state_changed:
            events.append("resolveu_etapa")
        if latency_s is not None and latency_s > HIGH_LATENCY_S:
            events.append("latencia_alta")

        # Data re-request beats generic question repeat (a verbatim second ask
        # for the CPF is both — count it once, as the stronger event).
        asked_again = False
        for category, keywords in _DATA_CATEGORIES.items():
            if any(kw in norm for kw in keywords):
                if category in self._data_requested:
                    asked_again = True
                self._data_requested.add(category)
        if asked_again:
            events.append("pediu_dado_ja_informado")

        if _looks_like_question(agent_text):
            tokens = _tokens(agent_text)
            if tokens:
                repeated = any(
                    len(tokens & prev) / len(tokens | prev) >= REPEAT_SIMILARITY
                    for prev in self._questions
                )
                if repeated and not asked_again:
                    events.append("repetiu_pergunta")
                self._questions.append(tokens)

        for event, keywords in _KEYWORD_EVENTS.items():
            if any(kw in norm for kw in keywords):
                events.append(event)

        # E4: persona-specific unknown/confused glossary terms — whole-word
        # match (keyword_matches) to avoid substring false positives.
        if "jargao_tecnico" not in events and any(
            keyword_matches(term, agent_text) for term in self._jargon_terms
        ):
            events.append("jargao_tecnico")
        return events


@dataclass
class EmotionalTurn:
    """One point of the emotional curve (goes to timeline.json/report)."""

    turn: int
    emotion: str
    intensity: float
    delta: float
    events: list[str]
    action: str | None = None  # "pedir_humano" | "desligar" on threshold crossing

    @property
    def direction(self) -> str:
        return "↗" if self.delta > 0 else ("↘" if self.delta < 0 else "→")

    def as_dict(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "turn": self.turn,
            "emotion": self.emotion,
            "intensity": self.intensity,
            "delta": self.delta,
            "events": list(self.events),
        }
        if self.action:
            entry["action"] = self.action
        return entry


class EmotionalStateMachine:
    """Appraisal reducer: persona's emotionalModel + detected events -> state.

    Sensitivity is fully persona-parametrized: patient personas ship smaller
    trigger deltas / higher thresholds in their catalog `emotionalModel`
    (the spec's Dona Márcia values already reflect patience 2). Personas
    without a model get the stable-neutral default (flat curve, thresholds
    unreachable).
    """

    def __init__(
        self,
        model: EmotionalModel,
        *,
        seed: int | None = None,
        jargon_terms: tuple[str, ...] = (),
    ):
        self.model = model
        self.seed = seed  # recorded for reproducibility; rules are pure (see module docstring)
        self.emotion = model.initialEmotion
        self.intensity = model.initialIntensity
        self.detector = AppraisalDetector(jargon_terms)
        self.history: list[EmotionalTurn] = []
        self._asked_human = False
        self._hung_up = False

    @classmethod
    def for_persona(cls, persona: Persona, *, seed: int | None = None) -> EmotionalStateMachine:
        model = persona.emotionalModel or EmotionalModel()
        # E4: the persona's unknown/confused glossary terms extend the
        # jargao_tecnico detector — deterministic, resolved by the service.
        vocab = persona.glossaryVocabulary
        jargon_terms = tuple(
            entry.term
            for entry in ((vocab.unknown if vocab else []) + (vocab.confused if vocab else []))
        )
        return cls(model, seed=seed, jargon_terms=jargon_terms)

    def update(
        self,
        agent_text: str,
        *,
        state_changed: bool = False,
        latency_s: float | None = None,
        extra_events: tuple[str, ...] = (),
    ) -> EmotionalTurn:
        """Appraise one agent turn and evolve the state (clamped to [0, 1])."""
        events = self.detector.detect(
            agent_text, state_changed=state_changed, latency_s=latency_s
        )
        events.extend(extra_events)

        delta = 0.0
        triggered = False
        for event in events:
            for trigger in self.model.triggers:
                if trigger.on == event:
                    triggered = True
                    delta += trigger.delta
                    if trigger.emotion:
                        self.emotion = trigger.emotion
        if not triggered:
            delta = self.model.decayPerTurn

        self.intensity = max(0.0, min(1.0, round(self.intensity + delta, 4)))

        action = None
        if not self._hung_up and self.intensity >= self.model.thresholds.desligar:
            action = "desligar"
            self._hung_up = True
        elif not self._asked_human and self.intensity >= self.model.thresholds.pedirHumano:
            action = "pedir_humano"
            self._asked_human = True

        record = EmotionalTurn(
            turn=len(self.history) + 1,
            emotion=self.emotion,
            intensity=self.intensity,
            delta=round(delta, 4),
            events=events,
            action=action,
        )
        self.history.append(record)
        return record

    def curve(self) -> list[dict[str, Any]]:
        return [turn.as_dict() for turn in self.history]

    def badge(self) -> str:
        """Compact display, e.g. `[ansioso 0.45 ↗]` (chat prompt/UI)."""
        direction = self.history[-1].direction if self.history else "→"
        return f"[{self.emotion} {self.intensity:.2f} {direction}]"

    def prompt_block(self) -> str:
        """Emotional-state block injected into the persona-turn prompt.

        Today it rides inside `goalTemplate` (the hive persona-turn contract
        has no emotionalState field yet — see README contract note); the hive
        prompt v2 (P0.2) will receive it as a structured block.
        """
        direction = {"↗": "subindo", "↘": "acalmando", "→": "estável"}[
            self.history[-1].direction if self.history else "→"
        ]
        lines = [
            "[ESTADO EMOCIONAL — mantido pelo sistema de teste; siga à risca]",
            f"Emoção atual: {self.emotion} | Intensidade: {self.intensity:.2f}/1.0 ({direction})",
            f"Manifestação: {self.guidance()}",
            "A mudança de humor é sempre gradual e motivada — nunca salte de calmo "
            "para furioso sem gatilho.",
        ]
        return "\n".join(lines)

    def guidance(self) -> str:
        """Behavioral instruction for the current state — goes to the hive's
        structured `emotionalState.guidance` (and to the legacy prompt block)."""
        last_action = self.history[-1].action if self.history else None
        if last_action == "desligar" or self._hung_up:
            return (
                "limite final ultrapassado: você DEVE se despedir de forma seca e "
                "ENCERRAR a ligação neste turno."
            )
        if last_action == "pedir_humano":
            return (
                "você acabou de cruzar seu limite: você DEVE exigir falar com um "
                "atendente humano agora, neste turno, uma única vez."
            )
        if self._asked_human:
            return (
                "você já pediu um atendente humano; está no limite da paciência — "
                "respostas curtas, tom impaciente, ameace desligar se piorar."
            )
        if self.intensity < 0.3:
            return f"leve: fala normal, sinais sutis de {self.emotion}."
        if self.intensity < 0.6:
            return (
                f"média: verbalize o {self.emotion} ('mas vai resolver hoje, né?'), "
                "peça confirmação, frases um pouco mais curtas."
            )
        return (
            f"alta: demonstre claramente o {self.emotion}, interrompa explicações "
            "longas, frases curtas, mencione que está perdendo a paciência."
        )
