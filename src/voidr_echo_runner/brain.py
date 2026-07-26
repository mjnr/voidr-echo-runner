"""AI-only persona turns through Hive's strict v3 contract."""

from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from .models import Persona
from .governed_url import validate_governed_url


class PersonaBrain(Protocol):
    def take_turn(
        self, agent_utterance: str, *, turn_id: str | None = None
    ) -> dict[str, Any]:
        """Produce one Hive-authored persona turn with provenance."""
        ...


class HiveError(RuntimeError):
    """A persona-turn call to the hive failed (payload, PII guard or gateway)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.turn_id = turn_id

    @property
    def outcome(self) -> str:
        code = (self.code or "").lower()
        if self.status_code in {429, 504} or code in {
            "deadline_exceeded",
            "upstream_rate_limited",
            "turn_in_progress",
        }:
            return "inconclusive"
        if self.status_code is None or self.status_code >= 500:
            return "degraded"
        return "failed"


GENERIC_JOURNEY_STATE: dict[str, Any] = {
    "flowSlug": "conversa-livre",
    "currentState": "conversa",
    "expects": [],
}


class LLMBrain:
    """Persona brain backed exclusively by Hive persona-turn v3."""

    REQUIRED_ENV = (
        "HIVE_URL",
        "HIVE_GATEWAY_TOKEN",
        "VOIDR_ORG_ID",
        "HIVE_ECHO_PERSONA_V3_MODEL_REVISION",
    )
    DEADLINE_S = 20.0
    # Leave transport headroom so Hive can return its structured
    # DEADLINE_EXCEEDED response before the runner aborts the HTTP request.
    HTTP_TIMEOUT_S = 25.0
    POLICY_VERSION = "echo-persona-turn-v3.0.0"
    CONTRACT_VERSION = "v3"
    PROMPT_VERSION = "echo-persona-system-v3.0.0"
    MODEL_ALIAS = "deepseek-v4-pro"
    MODEL_REVISION = "8b3fcb4e-61f2-4a76-9e0d-73e89bc3f1a2"

    def __init__(
        self,
        persona: Persona,
        goal: str,
        seed: int | None = None,
        *,
        client: Any | None = None,
        conversation_id: str | None = None,
    ):
        import httpx

        missing = [name for name in self.REQUIRED_ENV if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                f"LLMBrain requires Hive env vars: {', '.join(missing)}"
            )
        self.persona = persona
        self.goal = goal
        self.seed = seed
        self.base_url = validate_governed_url(
            os.environ["HIVE_URL"], name="HIVE_URL"
        )
        self.org_id = os.environ["VOIDR_ORG_ID"]
        self.model_revision = os.environ["HIVE_ECHO_PERSONA_V3_MODEL_REVISION"].strip()
        if not (
            re.search(r"@sha256:[a-f0-9]{64}$", self.model_revision)
            or re.fullmatch(
                r"(?:.*@id:)?[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
                r"[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                self.model_revision,
                re.IGNORECASE,
            )
        ):
            raise RuntimeError(
                "HIVE_ECHO_PERSONA_V3_MODEL_REVISION must be an immutable "
                "deployment/revision ID distinct from deepseek-v4-pro"
            )
        if self.model_revision != self.MODEL_REVISION:
            raise RuntimeError(
                "HIVE_ECHO_PERSONA_V3_MODEL_REVISION does not match this runner "
                "release's immutable LiteLLM deployment UUID"
            )
        # Separate calls must never share a Redis idempotency namespace merely
        # because persona/goal/seed match. Durable service executions pass an
        # execution+shard-derived ID; ad-hoc run/chat sessions get a fresh one.
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self._turn_number = 0
        self._client = client or httpx.Client(
            timeout=self.HTTP_TIMEOUT_S,
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
        # Deterministic state remains runner-side and is sent only as a Hive
        # directive. It never authors local speech.
        from .emotional import EmotionalStateMachine

        self.emotional: EmotionalStateMachine | None = None
        self.execution_overrides: Any | None = None
        self.turn_directives: list[str] = []
        self.personal_data: list[dict[str, str]] = []
        self.total_cost_usd = 0.0
        self.last_model: str | None = None
        self.last_usage: dict[str, Any] | None = None
        self.last_provenance: dict[str, Any] | None = None

    def _persona_payload(self) -> dict[str, Any]:
        # The persona says WHO the person is; the journey carries the goal.
        goal_template = self.persona.goalTemplate
        if self.goal and "{goal}" in goal_template:
            goal_template = goal_template.format(goal=self.goal)
        if not goal_template.strip():
            goal_template = self.goal or "Resolver o objetivo desta ligação."
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
        # v2.1 (E3): literacy axis + glossary vocabulary resolved by the
        # service — the hive renders the <letramento-e-vocabulario> block
        # (popular synonyms, unknown jargon, confused terms). Older hives
        # strip unknown keys (zod), so sending them is always safe.
        if self.persona.literacy is not None:
            v2["literacy"] = {
                key: value
                for key, value in self.persona.literacy.model_dump().items()
                if value is not None
            }
        if self.persona.glossaryVocabulary is not None:
            vocab = self.persona.glossaryVocabulary.model_dump(exclude_none=True)
            if vocab.get("popularOnly") or vocab.get("unknown") or vocab.get("confused"):
                v2["glossaryVocabulary"] = vocab
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

    def _build_payload(
        self,
        agent_utterance: str,
        turn_id: str,
        deadline_at: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        persona_payload = self._persona_payload()
        payload = {
            "organizationId": self.org_id,
            "conversationId": self.conversation_id,
            "turnId": turn_id,
            "policyVersion": self.POLICY_VERSION,
            "deadlineAt": deadline_at,
            "persona": persona_payload,
            "journeyState": self.journey_state,
            "history": [
                {"role": turn["role"], "text": self.redaction.redact(turn["text"])}
                for turn in self.history
            ]
            + [{"role": "agent", "text": self.redaction.redact(agent_utterance)}],
            **({"options": options} if options else {}),
        }
        if self.emotional is not None:
            payload["emotionalState"] = {
                "emotion": self.emotional.emotion,
                "intensity": self.emotional.intensity,
                "guidance": self.emotional.guidance(),
            }
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
        if self.personal_data:
            payload["personalData"] = [dict(item) for item in self.personal_data]
        if self.turn_directives:
            payload["turnDirectives"] = list(self.turn_directives)
        # Last-hop defense: recursively redact every textual field that can
        # reach the persona prompt, including nested identity facts,
        # profile/context, vocabulary, exemplars and glossary entries. Keep
        # correlation/deadline fields byte-stable for idempotency/provenance.
        for field in (
            "persona",
            "journeyState",
            "history",
            "emotionalState",
            "goalProgress",
            "journeyGoal",
            "executionOverrides",
            "personalData",
            "turnDirectives",
        ):
            if field in payload:
                payload[field] = self.redaction.redact_deep(payload[field])
        return payload

    def _post(self, url: str, payload: dict[str, Any], turn_id: str) -> Any:
        import httpx

        try:
            return self._client.post(url, json=payload)
        except httpx.TransportError as exc:
            raise HiveError(
                f"hive persona-turn v3 unreachable: {type(exc).__name__}: {exc}",
                code="transport_error",
                turn_id=turn_id,
            ) from exc

    @staticmethod
    def _required_string(data: dict[str, Any], key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing non-empty {key}")
        return value

    def _validate_response(self, data: Any, turn_id: str) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("response must be an object")
        text = self._required_string(data, "text")
        provenance = data.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("missing provenance object")
        if provenance.get("contractVersion") != self.CONTRACT_VERSION:
            raise ValueError("provenance.contractVersion must be v3")
        if provenance.get("source") != "hive-llm":
            raise ValueError("provenance.source must be hive-llm")
        response_turn_id = self._required_string(provenance, "turnId")
        if response_turn_id != turn_id:
            raise ValueError("provenance.turnId does not match request turnId")
        for key in (
            "conversationId",
            "policyVersion",
            "promptVersion",
            "promptHash",
            "provider",
            "modelAlias",
            "model",
            "modelResolved",
            "modelVersion",
            "deploymentPin",
            "completionId",
            "traceId",
            "generatedAt",
        ):
            self._required_string(provenance, key)
        if provenance["conversationId"] != self.conversation_id:
            raise ValueError("provenance.conversationId does not match request")
        if provenance["policyVersion"] != self.POLICY_VERSION:
            raise ValueError("provenance.policyVersion does not match the policy pin")
        if provenance["promptVersion"] != self.PROMPT_VERSION:
            raise ValueError("provenance.promptVersion does not match the prompt pin")
        if provenance["model"] != self.MODEL_ALIAS:
            raise ValueError("provenance.model does not match the requested model alias")
        if provenance["modelAlias"] != self.MODEL_ALIAS:
            raise ValueError("provenance.modelAlias does not match the requested model alias")
        if provenance["modelResolved"] != self.model_revision:
            raise ValueError("provenance.modelResolved does not match the immutable model pin")
        if provenance["modelVersion"] != self.model_revision:
            raise ValueError(
                "provenance.modelVersion does not match the immutable model revision pin"
            )
        if provenance["deploymentPin"] != self.model_revision:
            raise ValueError("provenance.deploymentPin does not match the immutable pin")
        if "@sha256:" in self.model_revision:
            expected_digest = self.model_revision.rsplit("@", 1)[1]
            if provenance.get("deploymentDigest") != expected_digest:
                raise ValueError("provenance.deploymentDigest does not match the immutable pin")
        else:
            expected_id = self.model_revision.rsplit("@id:", 1)[-1]
            if provenance.get("deploymentId") != expected_id:
                raise ValueError("provenance.deploymentId does not match the immutable pin")
        expected_model_hash = hashlib.sha256(self.model_revision.encode()).hexdigest()
        if self._required_string(provenance, "modelHash") != expected_model_hash:
            raise ValueError("provenance.modelHash does not match the model revision pin")
        return {**data, "text": text, "provenance": provenance}

    def take_turn(
        self,
        agent_utterance: str,
        *,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        """Call persona-turn v3 once; never retries or downgrades."""
        self._turn_number += 1
        stable_turn_id = turn_id or str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"voidr:echo:{self.conversation_id}:persona-turn:{self._turn_number}",
            )
        )
        options: dict[str, Any] = {}
        if self.seed is not None:
            options["seed"] = self.seed
        url = f"{self.base_url}/echo/persona-turn/v3"
        deadline_at = (
            datetime.now(timezone.utc) + timedelta(seconds=self.DEADLINE_S)
        ).isoformat().replace("+00:00", "Z")
        payload = self._build_payload(agent_utterance, stable_turn_id, deadline_at, options)
        started = time.monotonic()
        response = self._post(url, payload, stable_turn_id)
        if response.status_code != 200:
            detail = "Hive persona-turn v3 failed"
            code = None
            try:
                body = response.json()
                error = body.get("error") if isinstance(body, dict) else None
                if isinstance(error, dict):
                    code = error.get("code")
                    detail = str(error.get("message") or detail)
            except Exception:  # noqa: BLE001 — non-JSON error body
                pass
            raise HiveError(
                f"persona-turn v3 failed ({response.status_code}"
                + (f", {code}" if code else "")
                + (f"): {detail}" if detail else ")"),
                status_code=response.status_code,
                code=str(code) if code else None,
                turn_id=stable_turn_id,
            )

        try:
            data = self._validate_response(response.json(), stable_turn_id)
        except (ValueError, TypeError) as exc:
            raise HiveError(
                f"invalid persona-turn v3 response: {exc}",
                status_code=502,
                code="invalid_response",
                turn_id=stable_turn_id,
            ) from exc
        self.turn_directives = []  # directives are strictly per-turn
        self.history.append({"role": "agent", "text": agent_utterance})
        self.history.append({"role": "persona", "text": data["text"]})
        # Preserve the validated Hive DTO byte-for-byte at the object level.
        # Runner-local timing stays in a sibling trace and never replaces or
        # truncates provider/model/hash/correlation provenance fields.
        hive_provenance = dict(data["provenance"])
        trace = {
            "contractVersion": hive_provenance["contractVersion"],
            "conversationId": hive_provenance["conversationId"],
            "completionId": hive_provenance["completionId"],
            "traceId": hive_provenance["traceId"],
            "promptHash": hive_provenance["promptHash"],
            "modelHash": hive_provenance["modelHash"],
            "attempts": hive_provenance.get("attempts"),
            "durationMs": int((time.monotonic() - started) * 1000),
        }
        self.last_provenance = hive_provenance
        self.last_model = hive_provenance["model"]
        self.last_usage = data.get("usage") or {}
        self.total_cost_usd += float(self.last_usage.get("costUsd") or 0.0)
        return {**data, **hive_provenance, "provenance": hive_provenance, "trace": trace}

def build_brain(persona: Persona, goal: str, seed: int, **kwargs: Any) -> PersonaBrain:
    return LLMBrain(persona, goal, seed, **kwargs)
