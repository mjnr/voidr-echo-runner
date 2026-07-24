"""Pydantic schemas: voice test case (ARCHITECTURE.md section 4.4) and
persona (section 5.4)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from .textutil import resolve_placeholders_deep


class DtmfStep(BaseModel):
    wait_for: str | None = None
    wait_for_prompt_matching: str | None = None
    send: str


class DialPlan(BaseModel):
    to: str | None = None
    dtmf_steps: list[DtmfStep] = Field(default_factory=list)


class PersonaRef(BaseModel):
    base: str
    variant_seed: int = 0
    overrides: dict = Field(default_factory=dict)


class FlowAssert(BaseModel):
    must_visit: list[str] = Field(default_factory=list)
    must_not_visit: list[str] = Field(default_factory=list)
    max_turns: int = 20


class CaseAssert(BaseModel):
    flow: FlowAssert = Field(default_factory=FlowAssert)


class VoiceTestCase(BaseModel):
    id: str
    channel: str = "voice"
    persona: PersonaRef
    massa: dict = Field(default_factory=dict)
    dial_plan: DialPlan = Field(default_factory=DialPlan)
    journey_flow: str
    # Canonical Journey ref (ARCHITECTURE.md §8.1): the TestPlan ModuleItem this
    # case belongs to. Optional — when absent the runner omits `moduleSlug` from
    # the voice-session report and voidr-service derives it from the plan.
    module_slug: str | None = None
    # Plan id only makes sense in service mode (it is environment-specific);
    # in serve-execution it comes from the execution, not from YAML.
    test_plan_id: str | None = None
    goal: str
    arrange: str | None = None
    act: str | None = None
    assertion: CaseAssert = Field(default_factory=CaseAssert, alias="assert")

    model_config = {"populate_by_name": True}

    # {{env.*}} values substituted at load time — feeds the PII deny-list
    # (redaction.build_session_for_case). Never serialized.
    _resolved_secrets: dict[str, str] = PrivateAttr(default_factory=dict)

    @property
    def resolved_secrets(self) -> dict[str, str]:
        return self._resolved_secrets

    @classmethod
    def load(cls, path: Path) -> "VoiceTestCase":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        captured: dict[str, str] = {}
        data = resolve_placeholders_deep(data, captured)
        case = cls.model_validate(data)
        case._resolved_secrets = captured
        return case


class Demographics(BaseModel):
    ageBand: str
    region: str


class Temperament(BaseModel):
    mood: str
    patienceLevel: int
    techSavviness: str
    verbosity: str
    intentNoise: str = "nenhum"


class Speech(BaseModel):
    ttsProvider: str = "elevenlabs"
    voiceId: str = ""
    speakingRate: float = 1.0
    backgroundNoise: str | None = None
    disfluencyRate: float = 0.0
    interruptionPolicy: str | None = None
    # v2 regional register (PERSONAS-SOTA P0.1) — the hive persona-turn
    # contract accepts all of these; leaving them out of the payload was the
    # gap that made personas sound generic regardless of configuration.
    discourseMarkers: list[str] = Field(default_factory=list)
    fillerInventory: list[str] = Field(default_factory=list)
    pronouns: str | None = None
    exemplars: list[str] = Field(default_factory=list)


class PersonaProfile(BaseModel):
    occupation: str = ""
    context: str = ""
    freeTraits: list[str] = Field(default_factory=list)

    @field_validator("freeTraits", mode="before")
    @classmethod
    def _coerce_free_traits(cls, v: object) -> object:
        # The service stores freeTraits as a single "a; b; c" string (hive
        # contract); the local catalog YAML uses a list. Accept both.
        if isinstance(v, str):
            return [part.strip() for part in v.split(";") if part.strip()]
        return v


class PersonaIdentity(BaseModel):
    """v2 identity block (PERSONAS-SOTA P0.1): canonical, judge-verifiable
    facts — the antidote to persona drift. All values synthetic."""

    fullName: str = ""
    shortName: str = ""
    backstory: str = ""
    facts: dict[str, str] = Field(default_factory=dict)


class Psychometrics(BaseModel):
    """Big Five 0-100 (Big5-Scaler format)."""

    openness: int
    conscientiousness: int
    extraversion: int
    agreeableness: int
    neuroticism: int


class PersonaBehaviors(BaseModel):
    """Non-collaborative behavior probabilities per turn (arXiv 2509.23124)."""

    tangential: float = 0.05
    outOfScopeRequest: float = 0.05
    incompleteUtterance: float = 0.0
    selfCorrection: float = 0.0


class EmotionalTrigger(BaseModel):
    """Appraisal rule: event name -> intensity delta (optional emotion switch)."""

    on: str
    delta: float
    emotion: str | None = None


class EmotionalThresholds(BaseModel):
    # Defaults are unreachable: personas without explicit thresholds never
    # escalate to asking for a human / hanging up.
    pedirHumano: float = 1.1
    desligar: float = 1.1


class EmotionalModel(BaseModel):
    """PERSONAS-SOTA.md P0.3 — deterministic emotional state machine config.

    Absent from a persona => stable neutral default (flat curve). Trigger
    deltas/thresholds are authored per persona, already accounting for
    patience (patient personas escalate slower)."""

    initialEmotion: str = "calmo"
    initialIntensity: float = 0.2
    decayPerTurn: float = 0.0
    triggers: list[EmotionalTrigger] = Field(default_factory=list)
    thresholds: EmotionalThresholds = Field(default_factory=EmotionalThresholds)


class Persona(BaseModel):
    id: str
    kind: str = "curated"
    version: int = 1
    # Identity (optional, section 5.4): when `name` is set the hive persona-turn
    # prompt locks the identity so the LLM can't invent another one.
    name: str = ""
    age: int | None = None
    gender: str = ""
    profile: PersonaProfile | None = None
    identity: PersonaIdentity | None = None
    psychometrics: Psychometrics | None = None
    behaviors: PersonaBehaviors | None = None
    emotionalModel: EmotionalModel | None = None
    demographics: Demographics
    temperament: Temperament
    speech: Speech
    # DEPRECATED as an embedded problem: the persona defines WHO the person
    # is; the call OBJECTIVE comes from the journey (case goal). Kept optional
    # for legacy personas — when the journey provides a goal it wins.
    goalTemplate: str = ""
    vocabulary: list[str] = Field(default_factory=list)
    massaProfile: str = ""


def load_persona_catalog(path: Path) -> dict[str, Persona]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    personas = [Persona.model_validate(raw) for raw in data["personas"]]
    return {p.id: p for p in personas}
