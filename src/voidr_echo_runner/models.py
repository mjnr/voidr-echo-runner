"""Pydantic schemas: voice test case (ARCHITECTURE.md section 4.4) and
persona (section 5.4)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

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
    goal: str
    arrange: str | None = None
    act: str | None = None
    assertion: CaseAssert = Field(default_factory=CaseAssert, alias="assert")

    model_config = {"populate_by_name": True}

    @classmethod
    def load(cls, path: Path) -> "VoiceTestCase":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data = resolve_placeholders_deep(data)
        return cls.model_validate(data)


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


class Persona(BaseModel):
    id: str
    kind: str = "curated"
    version: int = 1
    demographics: Demographics
    temperament: Temperament
    speech: Speech
    goalTemplate: str
    vocabulary: list[str] = Field(default_factory=list)
    massaProfile: str = ""


def load_persona_catalog(path: Path) -> dict[str, Persona]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    personas = [Persona.model_validate(raw) for raw in data["personas"]]
    return {p.id: p for p in personas}
