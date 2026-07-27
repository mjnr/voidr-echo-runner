"""Service-integration mode (`echo-runner serve-execution`).

Runs ONE execution shard end-to-end against a voidr-service instance,
following the same HTTP contract as voidr-k6-runner/voidr-runner:

  1. auth: `VOIDR_ACCESS_TOKEN` (pre-minted by the dispatcher) or
     POST /v1/service-accounts/token with VOIDR_CLIENT_ID/SECRET.
  2. GET /v1/executions/:id  → planId + targets (VOICE: 1 call = 1 shard,
     shard N runs target N-1).
  3. GET /v1/test-plans/:planId → the case's `voice` subdocument
     (persona/journey-flow ids, dial plan, goal, flowAssert, seed).
  4. GET /v1/echo/journey-flows/:id + /v1/echo/personas/:id.
  5. Executes the call (CallRunner core), resolving `{{env.*}}` placeholders
     from ENVIRONMENT_PARAMS (injected by the service at dispatch).
  6. Uploads report/artifacts via POST /v1/file-storage/upload (signed URLs):
     `org/{orgId}/executions/{id}/shards/{i}/reporter/json/test-results.json`
     is what voice-report.parser.ts consumes at finalize time.
  7. PUT /v1/executions/:id/shards/:i  (RUNNING → FINISHED/FAILED with
     stats + results, `name` = test case slug).
  8. POST /v1/echo/sessions with transcript/trajectory/metrics/deviations.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from . import __version__
from .artifacts import write_artifacts
from .brain import build_brain
from .evaluator import EvaluationResult, evaluate_trajectory
from .flows import FlowState, JourneyFlow
from .humanize import MassaFacts
from .models import (
    CaseAssert,
    DialPlan,
    DtmfStep,
    FlowAssert,
    Persona,
    PersonaRef,
    VoiceTestCase,
)
from .redaction import build_session_for_case, redact_call_result
from .runner import CallResult, CallRunner
from .transport import build_transport

ENV_PLACEHOLDER = re.compile(r"\{\{\s*env\.([A-Za-z0-9_]+)\s*\}\}")
_SENSITIVE_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    re.compile(
        r"(?<!\d)(?:\+?\d{10,15}|(?:\+?55[\s.-]?)?\(?\d{2}\)?"
        r"[\s.-]?\d{4,5}[\s.-]?\d{4})(?!\d)"
    ),
    re.compile(r"(?i)\b(?:token|secret|password|authorization)\s*[:=]\s*\S+"),
)

class ServeExecutionError(RuntimeError):
    pass


def classify_operational_failure(exc: Exception) -> tuple[str, str, str]:
    """Return stable, non-sensitive failure telemetry for UI and persistence."""
    message = str(exc).lower()
    if "hive persona-turn" in message and "missing:" in message:
        return ("env_failure", "HIVE_CONFIG_MISSING", "Hive configuration is incomplete")
    if "service_http_403" in message:
        return ("env_failure", "SERVICE_AUTH_FORBIDDEN", "Runner access to a required API was denied")
    if "api_key" in message or "provider" in message:
        return ("provider_failure", "PROVIDER_AUTH_UNAVAILABLE", "Direct voice provider is unavailable")
    if any(part in message for part in ("connect", "target", "websocket", "transport")):
        return ("target_failure", "TARGET_UNAVAILABLE", "Voice target is unavailable")
    return ("runner_failure", "RUNNER_EXECUTION_FAILED", "Runner failed before completing the call")


def _environment_sensitive_values() -> tuple[str, ...]:
    try:
        params = json.loads(os.environ.get("ENVIRONMENT_PARAMS", "{}") or "{}")
    except (json.JSONDecodeError, TypeError):
        params = {}
    values = [str(value) for value in params.values() if value not in (None, "")]
    values.extend(
        os.environ.get(name, "")
        for name in (
            "VOIDR_ACCESS_TOKEN",
            "VOIDR_CLIENT_SECRET",
            "HIVE_GATEWAY_TOKEN",
            "VOICE_GATEWAY_TOKEN",
        )
    )
    return tuple(value for value in values if len(value) >= 3)


def scrub_sensitive(value: object) -> str:
    """Early-safe scrubber for every managed-mode log/error boundary."""
    text = str(value)
    for secret in sorted(_environment_sensitive_values(), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    for pattern in _SENSITIVE_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text[:500]


def _safe_log(message: object, *, stderr: bool = False) -> None:
    print(scrub_sensitive(message), file=sys.stderr if stderr else sys.stdout)


def _safe_payload(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_sensitive(value)
    if isinstance(value, dict):
        return {str(key): _safe_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_safe_payload(item) for item in value)
    return value


# ENVIRONMENT_PARAMS is customer-controlled execution data. Platform
# coordinates and credentials are injected as container environment variables
# by the service and must never be shadowed by this bag.
CLIENT_PROMOTED_ENV_KEYS = {
    "ECHO_CALL_MODE",
    "ECHO_HUMAN_REALISM",
    "ECHO_HUMAN_TIMING",
    "ECHO_CALL_AMBIENCE",
    "ECHO_LIVE",
    "ECHO_LIVE_AUDIO",
}
CLIENT_ENDPOINT_KEYS = {"ECHO_CALL_TARGET"}
PLATFORM_HOST_ALLOWLIST_ENV = "ECHO_ALLOWED_TARGET_HOSTS"
TRUSTED_ENVELOPE_ENV_KEYS = {
    "HIVE_URL",
    "HIVE_ECHO_PERSONA_V3_MODEL_REVISION",
    "VOIDR_ORGANIZATION_ID",
    "VOIDR_EXECUTION_ID",
}


def _host_allowed(host: str, allowlist: set[str]) -> bool:
    host = host.rstrip(".").lower()
    return any(
        host == entry
        or (entry.startswith("*.") and host.endswith(entry[1:]) and host != entry[2:])
        for entry in allowlist
    )


def _validate_client_endpoint(key: str, value: str, env: dict[str, str]) -> None:
    if key != "ECHO_CALL_TARGET" or value.startswith(("tel:", "+")):
        return
    parsed = urlsplit(value)
    if parsed.scheme not in {"ws", "wss", "http", "https"} or not parsed.hostname:
        raise ServeExecutionError(f"{key} must be an absolute ws(s)/http(s) URL or tel target")
    allowlist = {
        item.strip().rstrip(".").lower()
        for item in env.get(PLATFORM_HOST_ALLOWLIST_ENV, "").split(",")
        if item.strip()
    }
    if not allowlist or not _host_allowed(parsed.hostname, allowlist):
        raise ServeExecutionError(f"{key} host is not allowed by the service")


def promote_params_to_environ(
    params: dict[str, str], environ: dict[str, str] | None = None
) -> list[str]:
    """Promote the small customer-setting allowlist to the process environment.

    Returns the promoted key names (values never logged). Empty-string values
    are skipped — an empty secret must not shadow a locally exported one.
    """
    env = os.environ if environ is None else environ
    promoted: list[str] = []
    for key, value in params.items():
        if key in CLIENT_ENDPOINT_KEYS and value:
            _validate_client_endpoint(key, value, env)
        if key in CLIENT_PROMOTED_ENV_KEYS and value:
            env[key] = value
            promoted.append(key)
    return sorted(promoted)


def promote_trusted_envelope_to_environ(
    trusted: dict[str, str], environ: dict[str, str] | None = None
) -> list[str]:
    """Install service-authenticated credentials and scope into the engine.

    Unknown fields fail closed. Client params never pass through this function,
    so they cannot inject a virtual key or overwrite tenant/execution scope.
    """
    unknown = set(trusted) - TRUSTED_ENVELOPE_ENV_KEYS
    if unknown:
        raise ServeExecutionError("VOICE execution envelope returned unknown trusted fields")
    env = os.environ if environ is None else environ
    promoted: list[str] = []
    for key, value in trusted.items():
        if value:
            env[key] = value
            promoted.append(key)
    return sorted(promoted)

# Deviation flags — must stay aligned with DEVIATION_FLAGS in
# voidr-service/src/modules/echo/services/voice-eval.service.ts.
FLAG_LOOP = "flag:loop"
FLAG_ABANDONMENT = "flag:abandono"
FLAG_MUST_NOT_VISIT = "flag:must_not_visit"
FLAG_MISSING_MUST_VISIT = "flag:must_visit_missing"
FLAG_MAX_TURNS = "flag:max_turns_exceeded"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_params_placeholders(
    value: str, params: dict[str, str], captured: dict[str, str] | None = None
) -> str:
    """Resolve {{env.NAME}} from ENVIRONMENT_PARAMS (never os.environ here —
    the job env contract is that secrets travel in ENVIRONMENT_PARAMS).
    Substituted values are recorded in `captured` for the PII deny-list."""

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in params:
            raise ServeExecutionError(
                f"placeholder {match.group(0)!r} not found in ENVIRONMENT_PARAMS "
                f"(available keys: {', '.join(sorted(params)) or '(none)'})"
            )
        if captured is not None:
            captured[name] = params[name]
        return params[name]

    return ENV_PLACEHOLDER.sub(_sub, value)


class VoidrApi:
    """Minimal client for the voidr-service runner contract (httpx, sync)."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            timeout=30.0,
            headers={"Authorization": f"Bearer {token}"},
        )

    @classmethod
    def authenticate(cls, base_url: str, client_id: str, client_secret: str) -> str:
        try:
            resp = httpx.post(
                f"{base_url.rstrip('/')}/v1/service-accounts/token",
                json={
                    "grantType": "client_credentials",
                    "clientId": client_id,
                    "clientSecret": client_secret,
                },
                timeout=30.0,
            )
        except httpx.HTTPError:
            raise ServeExecutionError("service_auth_failed") from None
        if resp.status_code != 201 and resp.status_code != 200:
            raise ServeExecutionError(f"service_auth_http_{resp.status_code}")
        try:
            token = resp.json()["access_token"]
        except (json.JSONDecodeError, KeyError, TypeError):
            raise ServeExecutionError("service_auth_invalid_response") from None
        if not isinstance(token, str) or not token:
            raise ServeExecutionError("service_auth_invalid_response")
        return token

    def _request(self, method: str, path: str, retries: int = 3, **kwargs) -> Any:
        url = f"{self.base_url}/v1{path}"
        for attempt in range(retries):
            try:
                resp = self._client.request(method, url, **kwargs)
                if resp.status_code >= 500:
                    raise ServeExecutionError(f"service_http_{resp.status_code}")
                if resp.status_code >= 400:
                    raise ServeExecutionError(f"service_http_{resp.status_code}") from None
                body = resp.json()
                return body.get("data", body) if isinstance(body, dict) else body
            except ServeExecutionError as exc:
                if str(exc).startswith("service_http_4"):
                    raise
            except (httpx.TransportError, json.JSONDecodeError):
                pass
            time.sleep(1.5 * (attempt + 1))
        raise ServeExecutionError("service_request_failed") from None

    def get_execution(self, execution_id: str) -> dict:
        return self._request("GET", f"/executions/{execution_id}")

    def get_test_plan(self, plan_id: str) -> dict:
        return self._request("GET", f"/test-plans/{plan_id}")

    def get_journey_flow(self, flow_id: str) -> dict:
        return self._request("GET", f"/echo/journey-flows/{flow_id}")

    def get_persona(self, persona_id: str, knowledge_level: str | None = None) -> dict:
        """Persona + resolved glossary vocabulary (E3).

        `knowledge_level` is the ephemeral "conhecimento do assunto" override
        of this run — UNIFIED with glossary mastery: the service recomputes
        the deterministic partition with the adjusted rate (same seed).
        """
        query = "?vocabulary=1"
        if knowledge_level:
            query += f"&knowledgeLevel={knowledge_level}"
        return self._request("GET", f"/echo/personas/{persona_id}{query}")

    def put_shard(self, execution_id: str, index: int, payload: dict) -> dict:
        return self._request(
            "PUT",
            f"/executions/{execution_id}/shards/{index}",
            json=_safe_payload(payload),
        )

    def post_session(self, payload: dict) -> dict:
        return self._request("POST", "/echo/sessions", json=payload)

    def consume_voice_envelope(
        self, execution_id: str, shard_index: int, reference: str
    ) -> tuple[dict[str, str], dict[str, str]]:
        body = self._request(
            "POST",
            f"/executions/{execution_id}/shards/{shard_index}/voice-envelope/{reference}/consume",
        )
        trusted = body.get("trustedEnv") if isinstance(body, dict) else None
        client = body.get("clientParams") if isinstance(body, dict) else None
        valid = lambda value: isinstance(value, dict) and all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in value.items()
        )
        if not valid(trusted) or not valid(client):
            raise ServeExecutionError("VOICE execution envelope returned invalid fields")
        forbidden = set(client) & TRUSTED_ENVELOPE_ENV_KEYS
        if forbidden:
            raise ServeExecutionError(
                "VOICE client params attempted to override governed fields"
            )
        return trusted, client

    def upload_file(self, key: str, content: bytes, content_type: str) -> str:
        presigned = self._request(
            "POST",
            "/file-storage/upload",
            json={"key": key, "contentType": content_type, "expiresIn": 3600},
        )
        upload_url = presigned["uploadUrl"]
        headers = {"Content-Type": content_type, **(presigned.get("headers") or {})}
        try:
            put = httpx.put(upload_url, content=content, headers=headers, timeout=60.0)
        except httpx.HTTPError:
            raise ServeExecutionError("artifact_upload_failed") from None
        if put.status_code >= 300:
            raise ServeExecutionError(f"artifact_upload_http_{put.status_code}")
        return key


# ─────────────────────────────────────────────────────────────────────────────
# Service-document → runner-model mapping
# ─────────────────────────────────────────────────────────────────────────────


def flow_from_service(doc: dict) -> JourneyFlow:
    """Service journey flow (camelCase: maxTurns/deviationRules) → JourneyFlow."""
    states: dict[str, FlowState] = {}
    for name, raw in (doc.get("states") or {}).items():
        states[name] = FlowState(
            name=name,
            expects=list(raw.get("expects", [])),
            next=list(raw.get("next", [])),
            terminal=bool(raw.get("terminal", False)),
            classification=raw.get("classification"),
            max_turns=raw.get("maxTurns", raw.get("max_turns")),
            evidence=list(raw.get("evidence", [])),
            keywords=list(raw.get("keywords", [])),
        )
    rules = [
        {"if": r.get("if"), "then": r.get("then")}
        for r in (doc.get("deviationRules") or doc.get("deviation_rules") or [])
    ]
    return JourneyFlow(
        id=doc.get("slug") or str(doc.get("_id")),
        source=doc.get("source"),
        states=states,
        deviation_rules=rules,
        dial_plan_steps=list((doc.get("dialPlan") or {}).get("dtmfSteps") or []),
    )


def persona_from_service(doc: dict) -> Persona:
    identity = doc.get("identity") or {}
    return Persona.model_validate(
        {
            "id": doc.get("slug") or str(doc.get("_id")),
            "kind": doc.get("kind", "curated"),
            "version": doc.get("version", 1),
            # Display name may carry "(58, mineira)" — the spoken identity is
            # identity.shortName when the persona has the v2 block.
            "name": identity.get("shortName") or doc.get("name", ""),
            "age": doc.get("age"),
            "gender": doc.get("gender", ""),
            "profile": doc.get("profile"),
            "demographics": doc["demographics"],
            "temperament": doc["temperament"],
            "speech": doc["speech"],
            "goalTemplate": doc["goalTemplate"],
            "vocabulary": doc.get("vocabulary", []),
            "massaProfile": doc.get("massaProfile", ""),
            # v2 blocks (schemaVersion 2) — shape 1:1 with the runner models,
            # so the LLMBrain payload and the emotional machine use the same
            # calibration whether the persona comes from YAML or the service.
            "identity": doc.get("identity"),
            "psychometrics": doc.get("psychometrics"),
            "behaviors": doc.get("behaviors"),
            "emotionalModel": doc.get("emotionalModel"),
            # v2.1 (E3): literacy axis + glossary vocabulary resolved by the
            # service (texts, not ids) — flows into the persona-turn prompt.
            "literacy": doc.get("literacy"),
            "glossaryVocabulary": doc.get("glossaryVocabulary"),
        }
    )


def find_case(plan: dict, target: dict) -> dict:
    for module in plan.get("modules", []):
        if module.get("slug") != target["moduleSlug"]:
            continue
        for suite in module.get("suites", []):
            if suite.get("slug") != target["suiteSlug"]:
                continue
            for case in suite.get("cases", []):
                if case.get("slug") == target["testCaseSlug"]:
                    return case
    raise ServeExecutionError(
        f"test case {target['moduleSlug']}/{target['suiteSlug']}/{target['testCaseSlug']} "
        "not found in the test plan"
    )


def _resolve_dtmf_send(
    value: str,
    massa: "MassaFacts | None",
    params: dict[str, str],
    captured: dict[str, str],
) -> str:
    """Two-pass placeholder resolution for a DTMF send: {{massa.X}} from the
    ECHO_MASSA bag first (a massa value may itself be an {{env.Y}} secret
    reference), then {{env.Y}} from ENVIRONMENT_PARAMS. Unresolved massa keys
    fail LOUD — dialing a literal placeholder as digits is never right."""
    if massa is not None:
        value, _ = massa.resolve_placeholders(value)
    value = resolve_params_placeholders(value, params, captured)
    if "{{massa." in value:
        available = ", ".join(sorted(massa.values)) if massa else "(no ECHO_MASSA)"
        raise ServeExecutionError(
            f"dial-plan send {value!r} references massa keys missing from "
            f"ECHO_MASSA (available: {available}) — fix the journey massa or the dial plan"
        )
    return value


def build_case(
    target: dict,
    case_doc: dict,
    persona_slug: str,
    flow: JourneyFlow,
    params: dict[str, str],
    plan_id: str | None = None,
    massa: "MassaFacts | None" = None,
) -> tuple[VoiceTestCase, str]:
    """Builds the runner-side VoiceTestCase + the resolved call target URL."""
    voice = case_doc.get("voice") or {}
    dial_plan_doc = voice.get("dialPlan") or {}
    captured: dict[str, str] = {}
    # Environment-level target (ECHO_CALL_TARGET reserved key in
    # ENVIRONMENT_PARAMS, service §8) overrides the case dialPlan `to`: the
    # SAME cases run against the local ws:// mock or a real tel: number
    # depending only on which environment the execution was dispatched to.
    raw_target = params.get("ECHO_CALL_TARGET") or dial_plan_doc.get("to")
    if not raw_target:
        raise ServeExecutionError(
            f"case {target['testCaseSlug']} has no voice.dialPlan.to and the "
            "environment provides no ECHO_CALL_TARGET — cannot resolve the call target"
        )
    call_target = resolve_params_placeholders(str(raw_target), params, captured)

    # DTMF preamble precedence: case dtmfSteps → journey (flow.dialPlan,
    # editable in the journey editor) → environment dialPlanDefaults.
    steps_source = list(dial_plan_doc.get("dtmfSteps") or [])
    if not steps_source and flow.dial_plan_steps:
        steps_source = flow.dial_plan_steps
    steps = [
        DtmfStep(
            wait_for_prompt_matching=step.get("waitFor"),
            send=_resolve_dtmf_send(str(step["send"]), massa, params, captured),
        )
        for step in steps_source
    ]
    if not steps:
        # Environment dial-plan defaults (IVR access code + ANI) for cases
        # without their own DTMF steps — values may be {{env.X}} placeholders.
        defaults = [
            ("código de acesso", params.get("ECHO_DIAL_ACCESS_CODE")),
            ("número da linha", params.get("ECHO_DIAL_ANI")),
        ]
        steps = [
            DtmfStep(
                wait_for_prompt_matching=wait_for,
                send=resolve_params_placeholders(str(send), params, captured),
            )
            for wait_for, send in defaults
            if send
        ]

    flow_assert_doc = voice.get("flowAssert")
    if flow_assert_doc:
        flow_assert = FlowAssert(
            must_visit=list(flow_assert_doc.get("mustVisit", [])),
            must_not_visit=list(flow_assert_doc.get("mustNotVisit", [])),
            max_turns=flow_assert_doc.get("maxTurns") or 20,
        )
    else:
        # Fallback: graph knowledge only — 'desvio_critico' states are always
        # forbidden (same rule as voice-eval v0 in the service).
        flow_assert = FlowAssert(
            must_visit=[],
            must_not_visit=[
                name
                for name, st in flow.states.items()
                if st.classification == "desvio_critico"
            ],
            max_turns=20,
        )

    goal = voice.get("goal") or case_doc.get("objective") or ""
    if not goal:
        act = case_doc.get("act") or []
        goal = act[0] if act else f"resolver: {case_doc.get('name', target['testCaseSlug'])}"

    case = VoiceTestCase(
        id=target["testCaseSlug"],
        channel=voice.get("channel", "voice"),
        persona=PersonaRef(base=persona_slug, variant_seed=voice.get("seed") or 0),
        dial_plan=DialPlan(to=call_target, dtmf_steps=steps),
        journey_flow="(service)",
        # Canonical Journey ref: the execution target is authoritative here
        # (same {testPlanId, moduleSlug} the service resolved the case from).
        module_slug=target.get("moduleSlug"),
        test_plan_id=plan_id,
        goal=goal,
        assertion=CaseAssert(flow=flow_assert),
    )
    case._resolved_secrets = captured  # feeds the PII redaction deny-list
    return case, call_target


# ─────────────────────────────────────────────────────────────────────────────
# Session classification (mirrors voice-eval v0 precedence in the service)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SessionVerdict:
    status: str  # passed | deviation | escalation | abandoned | env_failure
    deviations: list[dict]


def classify_session(
    flow: JourneyFlow,
    flow_assert: FlowAssert,
    call: CallResult,
) -> SessionVerdict:
    trajectory = [t.state for t in call.trajectory]
    visited: list[str] = []
    for state in trajectory:
        if state not in visited:
            visited.append(state)
    deviations: list[dict] = []

    if call.failure_status:
        return SessionVerdict(
            # Keep the established service enum stable. The precise AI
            # classification remains observable in the failure timeline event.
            status="env_failure",
            deviations=[
                {
                    "rule": "hive_turn_failure",
                    "detail": call.transport_error or "Hive persona-turn failed",
                }
            ],
        )

    if call.transport_error and call.agent_turns == 0:
        return SessionVerdict(
            status="env_failure",
            deviations=[
                {
                    "rule": "env_failure",
                    "detail": f"Ambiente indisponível antes da conversa: {call.transport_error}",
                }
            ],
        )

    for state in flow_assert.must_not_visit:
        if state in visited:
            entry = next(t for t in call.trajectory if t.state == state)
            deviations.append(
                {
                    "stateId": state,
                    "rule": FLAG_MUST_NOT_VISIT,
                    "detail": (
                        f"Estado proibido '{state}' foi visitado no turno {entry.turn} "
                        f'(fala do agente: "{entry.utterance[:120]}")'
                    ),
                }
            )

    for state in flow_assert.must_visit:
        if state not in visited:
            deviations.append(
                {
                    "stateId": state,
                    "rule": FLAG_MISSING_MUST_VISIT,
                    "detail": f"Estado obrigatório '{state}' não foi visitado",
                }
            )

    if call.agent_turns > flow_assert.max_turns:
        deviations.append(
            {
                "rule": FLAG_MAX_TURNS,
                "detail": (
                    f"Conversa excedeu o orçamento de turnos: "
                    f"{call.agent_turns} > {flow_assert.max_turns}"
                ),
            }
        )

    # Loop: per-state maxTurns tolerates N consecutive repeats; N+1 flags.
    run_state: str | None = None
    run_length = 0
    looped: set[str] = set()
    for state in trajectory:
        if state == run_state:
            run_length += 1
        else:
            run_state = state
            run_length = 1
        limit = flow.states.get(state).max_turns if state in flow.states else None
        if limit is not None and run_length > limit and state not in looped:
            looped.add(state)
            deviations.append(
                {
                    "stateId": state,
                    "rule": FLAG_LOOP,
                    "detail": f"Estado '{state}' repetido {run_length}x (máximo {limit} turnos)",
                }
            )

    last_state = trajectory[-1] if trajectory else None
    terminal_state = (
        last_state
        if last_state is not None and flow.states.get(last_state, None) and flow.states[last_state].terminal
        else None
    )
    if terminal_state is None:
        deviations.append(
            {
                **({"stateId": last_state} if last_state else {}),
                "rule": FLAG_ABANDONMENT,
                "detail": (
                    f"Conversa encerrou em '{last_state}' sem alcançar um estado terminal"
                    if last_state
                    else "Conversa encerrou sem nenhum estado classificado"
                ),
            }
        )

    escalation = (
        terminal_state is not None
        and (flow.states[terminal_state].classification or "") in ("escalacao", "escalation")
    )
    has_non_abandonment = any(d["rule"] != FLAG_ABANDONMENT for d in deviations)

    if has_non_abandonment:
        status = "deviation"
    elif escalation:
        status = "escalation"
    elif terminal_state is None:
        status = "abandoned"
    else:
        status = "passed"
    return SessionVerdict(status=status, deviations=deviations)


def resolve_persona_plan_entry(
    params: dict[str, str], shard_index: int
) -> dict[str, Any] | None:
    """Per-shard persona assignment from ENVIRONMENT_PARAMS.ECHO_PERSONA_PLAN.

    The service serializes one JSON array aligned with the execution targets
    (shard N reads entry N-1): `[{"personaId": "...", "overrides": {...}}]`.
    Absent/short/broken plans mean "use the case's own persona" — a malformed
    plan must never fail the call, only fall back loudly.
    """
    raw = params.get("ECHO_PERSONA_PLAN")
    if not raw:
        return None
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        _safe_log("ECHO_PERSONA_PLAN inválido; ignorado", stderr=True)
        return None
    if not isinstance(plan, list) or shard_index < 1 or shard_index > len(plan):
        return None
    entry = plan[shard_index - 1]
    return entry if isinstance(entry, dict) else None


def build_session_payload(
    execution_id: str,
    shard_index: int,
    case_slug: str,
    voice_config: dict,
    call: CallResult,
    verdict: SessionVerdict,
    transcript_path: str | None,
    module_slug: str | None = None,
    audio_path: str | None = None,
    audio_status: str | None = None,
    audio_reason: str | None = None,
    brain: str | None = None,
    persona_overrides: dict[str, Any] | None = None,
    persona_source: str | None = None,
) -> dict:
    start = call.started_at_ms

    def rel(ts: int) -> int:
        return max(0, int(ts) - start)

    def persisted_provenance(entry: dict[str, Any]) -> dict[str, Any]:
        """Translate the validated Hive turn into the service session DTO."""
        provenance = entry.get("provenance")
        trace = provenance.get("trace") if isinstance(provenance, dict) else None
        if not isinstance(provenance, dict) or not isinstance(trace, dict):
            raise ValueError("tester turn is missing validated Hive provenance/trace")
        for key in ("traceId", "promptHash"):
            if not isinstance(trace.get(key), str) or not trace[key]:
                raise ValueError(f"tester turn provenance trace is missing {key}")
        return {
            "source": provenance["source"],
            "turnId": provenance["turnId"],
            "promptVersion": provenance["promptVersion"],
            "modelResolved": provenance["modelResolved"],
            "modelVersion": provenance["modelVersion"],
            "policyVersion": provenance["policyVersion"],
            "trace": dict(trace),
        }

    transcript = []
    for entry in call.transcript:
        item = {
            # Session schema is diarized tester|agent — URA prompts are far-side.
            "role": "tester" if entry["speaker"] == "tester" else "agent",
            "text": entry["text"],
            "tsMs": rel(entry["ts"]),
        }
        if "provenance" in entry:
            item["provenance"] = persisted_provenance(entry)
        transcript.append(item)
    timeline = [
        {
            "type": event["type"],
            "tsMs": rel(event["ts"]),
            "data": {k: v for k, v in event.items() if k not in ("ts", "type")},
        }
        for event in call.timeline
    ]
    payload: dict[str, Any] = {
        "executionId": execution_id,
        "shardIndex": shard_index,
        "caseSlug": case_slug,
        "runnerMode": "ai-only-v3",
        "brain": "llm",
        "channel": voice_config.get("channel", "voice"),
        "status": verdict.status,
        "trajectory": [t.state for t in call.trajectory],
        "transcript": transcript,
        "timeline": timeline,
        "metrics": {"turns": call.agent_turns, "durationMs": call.duration_ms},
        "deviations": verdict.deviations,
    }
    # The current service DTO has no aiOutcome field. Detailed Hive failure
    # classification remains in the accepted timeline event until the service
    # integration adds a first-class field; do not emit schema-incompatible
    # top-level states in the meantime.
    # `brain` is retained in the signature for report compatibility, but the
    # runtime is AI-only and every session must be judged as Hive-authored.
    # Canonical Journey (module) slug from the case — when absent the service
    # derives it from the execution's plan (fallback), so we simply omit it.
    if module_slug:
        payload["moduleSlug"] = module_slug
    if voice_config.get("journeyFlowId"):
        payload["journeyFlowId"] = voice_config["journeyFlowId"]
    if voice_config.get("personaId"):
        payload["personaId"] = voice_config["personaId"]
    # Ephemeral variation of this run (mission delivery 1): recorded verbatim
    # so the session detail shows the knobs and the fidelity judge audits them.
    if persona_overrides:
        payload["personaOverrides"] = persona_overrides
    if persona_source:
        payload["personaSource"] = persona_source
    if voice_config.get("seed") is not None:
        payload["seed"] = voice_config["seed"]
    resolved_audio_status = audio_status or ("available" if audio_path else "unavailable")
    resolved_audio_reason = (
        None
        if resolved_audio_status == "available"
        else (audio_reason or "audio_not_recorded")
    )
    payload["artifacts"] = {
        "audioStatus": resolved_audio_status,
        **(
            {"audioUnavailableReason": resolved_audio_reason}
            if resolved_audio_reason
            else {}
        ),
        **({"transcriptPath": transcript_path} if transcript_path else {}),
        **({"audioPath": audio_path} if audio_path else {}),
    }
    return payload


def persist_voice_session(api: VoidrApi, payload: dict[str, Any]) -> dict:
    """Persist the mandatory audit record; transport/schema errors propagate."""
    return api.post_session(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────


def serve_execution(out_dir: Path) -> int:
    api_url = os.environ.get("VOIDR_API_URL")
    execution_id = os.environ.get("EXECUTION_ID")
    org_id = os.environ.get("VOIDR_ORG_ID")
    if not api_url or not execution_id or not org_id:
        _safe_log(
            "echo-runner serve-execution: VOIDR_API_URL, EXECUTION_ID and VOIDR_ORG_ID are required",
            stderr=True,
        )
        return 2

    shard_index = int(os.environ.get("SHARDS_CURRENT", "1"))
    shard_total = int(os.environ.get("SHARDS_TOTAL", "1"))
    params: dict[str, str] = json.loads(os.environ.get("ENVIRONMENT_PARAMS", "{}") or "{}")

    token = os.environ.get("VOIDR_ACCESS_TOKEN")
    if not token:
        client_id = os.environ.get("VOIDR_CLIENT_ID")
        client_secret = os.environ.get("VOIDR_CLIENT_SECRET")
        if not client_id or not client_secret:
            _safe_log(
                "echo-runner serve-execution: set VOIDR_ACCESS_TOKEN or "
                "VOIDR_CLIENT_ID + VOIDR_CLIENT_SECRET",
                stderr=True,
            )
            return 2
        token = VoidrApi.authenticate(api_url, client_id, client_secret)
    api = VoidrApi(api_url, token)
    envelope_ref = os.environ.get("VOICE_EXECUTION_ENVELOPE_REF", "").strip()
    if not envelope_ref:
        _safe_log(
            "echo-runner serve-execution: VOICE_EXECUTION_ENVELOPE_REF is required",
            stderr=True,
        )
        return 2
    trusted_env, envelope_client_params = api.consume_voice_envelope(
        execution_id, shard_index, envelope_ref
    )
    # The authenticated envelope has two explicit trust domains. Governed
    # credentials/scope go directly to the engine environment; customer
    # settings remain params and cannot shadow any governed field.
    promoted_trusted = promote_trusted_envelope_to_environ(trusted_env)
    params.update(envelope_client_params)
    promoted = promote_params_to_environ(params)
    if promoted_trusted:
        _safe_log(f"managed credentials promoted: {len(promoted_trusted)}")
    if promoted:
        _safe_log(f"managed settings promoted: {len(promoted)}")
    if (
        params.get("ECHO_KEEP_RAW_AUDIO") == "1"
        or os.environ.get("ECHO_KEEP_RAW_AUDIO") == "1"
    ):
        _safe_log(
            "echo-runner serve-execution: ECHO_KEEP_RAW_AUDIO=1 is forbidden "
            "in managed execution mode",
            stderr=True,
        )
        return 2

    _safe_log(f"serve-execution started shard={shard_index}/{shard_total}")

    started_at = _iso_now()
    try:
        execution = api.get_execution(execution_id)
        targets = execution.get("targets") or []
        if shard_index < 1 or shard_index > len(targets):
            raise ServeExecutionError(
                f"shard {shard_index} has no target (execution has {len(targets)} targets; "
                "VOICE contract is 1 call = 1 shard = 1 target)"
            )
        target = targets[shard_index - 1]
        plan_id = str(execution.get("planId"))

        plan = api.get_test_plan(plan_id)
        case_doc = find_case(plan, target)
        voice_config = case_doc.get("voice") or {}
        if not voice_config:
            raise ServeExecutionError(
                f"case {target['testCaseSlug']} has no `voice` config — not a VOICE case"
            )

        journey_flow_id = voice_config.get("journeyFlowId")
        if not journey_flow_id:
            raise ServeExecutionError(f"case {target['testCaseSlug']} has no voice.journeyFlowId")
        flow = flow_from_service(api.get_journey_flow(journey_flow_id))

        # Persona per execution (mission delivery 1/5): the execution may pin
        # a DIFFERENT persona for this shard (multi-persona runs) and carry
        # ephemeral overrides. Falls back to the case's own persona.
        plan_entry = resolve_persona_plan_entry(params, shard_index) or {}
        persona_source = "flow"
        persona_id = voice_config.get("personaId")
        if plan_entry.get("personaId"):
            persona_id = str(plan_entry["personaId"])
            persona_source = "execution"
        if not persona_id:
            raise ServeExecutionError(f"case {target['testCaseSlug']} has no voice.personaId")

        from .overrides import PersonaOverrides, apply_overrides

        overrides = PersonaOverrides.model_validate(plan_entry.get("overrides") or {})
        # E3 unification: the "conhecimento do assunto" override (the same
        # techSavviness knob) also adjusts the glossary partition — the
        # service recomputes it with the adjusted mastery rate, SAME seed.
        persona = persona_from_service(
            api.get_persona(persona_id, knowledge_level=overrides.techSavviness)
        )
        persona_overrides_record = overrides.as_record()
        if persona_overrides_record:
            persona = apply_overrides(persona, overrides)
            persona_source = "execution"
        # Session convention is the persona SLUG (playground sessions, rollup
        # indexes and the fidelity judge all key on it) — the plan's voice
        # config stores the ObjectId, so report the resolved slug instead.
        voice_config = {**voice_config, "personaId": persona.id}

        # EXEC-REALISM: personal massa of this call — ECHO_MASSA (JSON in
        # ENVIRONMENT_PARAMS, managed service-side) with fallback to the
        # persona's own identity facts. Resolved BEFORE build_case: the dial
        # plan's {{massa.*}} sends need the bag; values feed the PII deny-list
        # via case.massa BEFORE the redaction session is built.
        from .humanize import Humanizer

        massa = MassaFacts.resolve(params, persona)
        case, call_target = build_case(
            target, case_doc, persona.id, flow, params, plan_id, massa=massa
        )
        _validate_client_endpoint("ECHO_CALL_TARGET", call_target, os.environ)
        seed = voice_config.get("seed") or 0
        if massa:
            case.massa = dict(massa.values)
        # PII redaction (ARCHITECTURE.md section 10) — ALWAYS on in service
        # mode; deny-list from the ENVIRONMENT_PARAMS values used by the case.
        redaction = build_session_for_case(case)

        api.put_shard(
            execution_id,
            shard_index,
            {"status": "RUNNING", "startedAt": started_at},
        )
        _safe_log(
            f"  case={case.id} persona={persona.id} seed={seed} flow={flow.id} "
            f"target={redaction.redact(call_target)}"
        )

        # Call mode comes from the environment (ECHO_CALL_MODE, default text).
        # tel:/PSTN targets are real calls and only exist in audio mode — same
        # rule the pure CLI enforces.
        mode = (params.get("ECHO_CALL_MODE") or "text").strip().lower()
        is_pstn = call_target.startswith(("tel:", "+"))
        if is_pstn and mode != "audio":
            raise ServeExecutionError(
                f"target {redaction.redact(call_target)} is a real phone number — "
                "the environment must set voice mode 'audio' (ECHO_CALL_MODE)"
            )

        from .emotional import EmotionalStateMachine

        emotional = EmotionalStateMachine.for_persona(persona, seed=seed)
        # AI-only v3: Hive is a hard dependency. Missing credentials fail the
        # shard before transport/audio creation, so no tester turn or TTS can
        # be produced accidentally.
        missing_hive = [
            name
            for name in ("HIVE_URL", "HIVE_GATEWAY_TOKEN", "VOIDR_ORG_ID")
            if not os.environ.get(name)
        ]
        if missing_hive:
            raise ServeExecutionError(
                f"Hive persona-turn v3 is required; missing: {', '.join(missing_hive)}"
            )
        # EXEC-REALISM: memory imperfection + humanized timing (default ON;
        # ECHO_HUMAN_REALISM=0 disables everything, ECHO_HUMAN_TIMING=0 keeps
        # lapses/massa but removes the latency).
        humanizer = None
        if (params.get("ECHO_HUMAN_REALISM") or "1") != "0":
            humanizer = Humanizer(
                persona,
                seed,
                massa,
                timing_enabled=(params.get("ECHO_HUMAN_TIMING") or "1") != "0",
            )
        conversation_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"voidr:echo:{execution_id}:shard:{shard_index}",
            )
        )
        brain = build_brain(
            persona,
            case.goal,
            seed,
            conversation_id=conversation_id,
        )
        brain.redaction = redaction  # deny-list of the case massa
        brain.emotional = emotional  # structured emotionalState per turn
        if massa:
            # Labels + placeholders only; clear values never reach Hive.
            brain.personal_data = massa.personal_data_lines()
        if persona_overrides_record:
            brain.execution_overrides = overrides
        send_digits = None
        if is_pstn and case.dial_plan.dtmf_steps:
            # IVR digits at answer time, with `w` (0.5s) pauses between steps.
            send_digits = "ww" + "ww".join(step.send for step in case.dial_plan.dtmf_steps)
        transport = build_transport(call_target, send_digits=send_digits)
        engine = recorder = channel_fx = None
        receive_timeout = 10.0
        if mode == "audio":
            os.environ.setdefault("LOGURU_LEVEL", "WARNING")  # quiet pipecat logs
            # Imports pipecat; validates DEEPGRAM/ELEVENLABS keys with clear errors.
            from .audio import AudioTransportAdapter, PipecatAudioEngine, StereoCallRecorder
            from .callfx import TelephoneChannelFx, parse_ambience

            engine = PipecatAudioEngine(voice_id=persona.speech.voiceId)
            recorder = StereoCallRecorder()
            # EXEC-REALISM: telephone channel on the persona audio — band-pass
            # 300–3400 Hz + µ-law grit + seeded ambience (ECHO_CALL_AMBIENCE,
            # default "quiet"; "none" disables). Deterministic per seed.
            ambience = parse_ambience(params.get("ECHO_CALL_AMBIENCE"))
            if ambience.enabled:
                channel_fx = TelephoneChannelFx(
                    ambience, seed=seed, sample_rate=engine.sample_rate
                )
            transport = AudioTransportAdapter(
                transport, engine, recorder, channel_fx=channel_fx
            )
            receive_timeout = 45.0  # remote STT+TTS per turn

        # Live events (plan Feature 1): default ON in serve-execution.
        # Text receives full PII + deny-list redaction; paired sensitive audio
        # is silenced before publication.
        live = None
        if (params.get("ECHO_LIVE") or os.environ.get("ECHO_LIVE") or "1") != "0":
            from .live_events import LivePublisher

            live = LivePublisher(
                api_url,
                execution_id,
                shard_index,
                token=token,
                redact=redaction.redact,
                has_sensitive_data=lambda text: bool(redaction.find_spans(text)),
                audio_enabled=(
                    params.get("ECHO_LIVE_AUDIO") or os.environ.get("ECHO_LIVE_AUDIO") or "1"
                )
                != "0",
            )
            if mode == "audio":
                transport.live = live

        runner = CallRunner(
            case,
            flow,
            brain,
            transport,
            receive_timeout=receive_timeout,
            emotional=emotional,
            live=live,
            humanizer=humanizer,
        )

        async def _run_call():
            if live is not None:
                await live.start()
            try:
                return await runner.run()
            finally:
                if live is not None:
                    await live.stop()
                if engine is not None:
                    await engine.aclose()

        call = asyncio.run(_run_call())
        evaluation = evaluate_trajectory(
            case.assertion.flow,
            call.trajectory,
            call.agent_turns,
            call.end_reason,
            transport_error=call.transport_error,
        )
        verdict = classify_session(flow, case.assertion.flow, call)
        if live is not None:
            # terminal live event carries the evaluated session status
            if verdict.status == "env_failure":
                if call.transport_error:
                    live.fail_sync(
                        "target_failure",
                        "TARGET_UNAVAILABLE",
                        "Voice target transport failed",
                    )
                else:
                    live.fail_sync(
                        "env_failure",
                        "CALL_ENV_FAILURE",
                        "Call ended before agent evaluation completed",
                    )
            else:
                live.finish_sync(call.end_reason or "unknown", verdict.status)

        # Everything persisted or POSTed from here on is redacted
        # (artifacts, session transcript/timeline, deviation details).
        redact_call_result(call, redaction)
        if evaluation.error_message:
            evaluation.error_message = redaction.redact(evaluation.error_message)
        verdict.deviations = redaction.redact_deep(verdict.deviations)

        run_id = f"{execution_id}-shard-{shard_index}"
        meta: dict[str, Any] = {
            "persona": {
                "id": persona.id,
                "version": persona.version,
                "variantSeed": seed,
                "source": persona_source,
                **({"overrides": persona_overrides_record} if persona_overrides_record else {}),
            },
            "journeyFlowId": flow.id,
            **({"moduleSlug": case.module_slug} if case.module_slug else {}),
            **({"testPlanId": case.test_plan_id} if case.test_plan_id else {}),
            "runnerVersion": __version__,
            "mode": mode,
            "brain": "hive-llm",
            "conversationId": conversation_id,
            "target": redaction.redact(call_target),
            "executionId": execution_id,
            "shard": {"index": shard_index, "total": shard_total},
        }
        if humanizer is not None:
            meta["humanize"] = humanizer.config_record()
        if emotional.history:
            meta["emotionalCurve"] = emotional.curve()
            meta["emotionalFinal"] = {
                "emotion": emotional.emotion,
                "intensity": emotional.intensity,
            }
        wav_path: Path | None = None
        if mode == "audio" and recorder is not None:
            meta["audio"] = {
                "sttProvider": "deepgram",
                "ttsProvider": "elevenlabs",
                "sttModel": "nova-2",
                "ttsModel": "eleven_flash_v2_5",
                "voiceId": persona.speech.voiceId,
                "sttTurns": transport.stt_turns,
                "ttsTurns": transport.tts_turns,
                "wavDurationMs": recorder.duration_ms,
                **({"channelFx": channel_fx.record()} if channel_fx is not None else {}),
            }
            # Audio PII redaction (§10): beeps over word-timestamps, writes
            # call.redacted.wav — the raw recording is never persisted.
            from .audio_redaction import redact_call_audio

            audio_dir = out_dir / run_id
            audio_dir.mkdir(parents=True, exist_ok=True)
            meta["audio"].update(
                redact_call_audio(recorder, transport.utterances, redaction, audio_dir)
            )
            meta["audio"]["wavFile"] = "call.redacted.wav"
            wav_path = audio_dir / "call.redacted.wav"
        meta["piiRedactionReport"] = redaction.report()
        report_path = write_artifacts(out_dir, run_id, case.id, call, evaluation, meta=meta)
        run_dir = report_path.parent

        # Upload artifacts. The reporter JSON path is the one the service's
        # finalizeReport reads for VOICE shards.
        shard_prefix = f"org/{org_id}/executions/{execution_id}/shards/{shard_index}"
        transcript_key: str | None = None
        audio_key: str | None = None
        audio_status = "unavailable"
        audio_reason = "audio_mode_disabled" if mode != "audio" else "audio_not_recorded"
        try:
            api.upload_file(
                f"{shard_prefix}/reporter/json/test-results.json",
                report_path.read_bytes(),
                "application/json",
            )
            transcript_key = api.upload_file(
                f"{shard_prefix}/artifacts/transcript.json",
                (run_dir / "transcript.json").read_bytes(),
                "application/json",
            )
            api.upload_file(
                f"{shard_prefix}/artifacts/timeline.json",
                (run_dir / "timeline.json").read_bytes(),
                "application/json",
            )
            _safe_log("non-audio artifacts uploaded")
        except Exception:  # noqa: BLE001 — report shard even if upload fails
            _safe_log("non_audio_artifact_upload_failed; continuing", stderr=True)

        # Audio has an independent outcome: JSON artifact failures must not
        # prevent its PUT, and the path is published only after a successful PUT.
        if wav_path is not None and wav_path.exists():
            try:
                audio_key = api.upload_file(
                    f"{shard_prefix}/artifacts/call.redacted.wav",
                    wav_path.read_bytes(),
                    "audio/wav",
                )
                audio_status = "available"
                audio_reason = None
                _safe_log("audio artifact uploaded")
            except Exception:  # noqa: BLE001 — persist only a stable safe reason
                audio_key = None
                audio_status = "unavailable"
                audio_reason = "audio_upload_failed"
                _safe_log(
                    "audio_upload_failed reason=audio_upload_failed; continuing",
                    stderr=True,
                )

        # Persist the voice session before the shard PUT that can trigger
        # execution finalization. A rejected/lost session is a shard failure,
        # never a silently successful execution without its audit record.
        persist_voice_session(
            api,
            build_session_payload(
                execution_id,
                shard_index,
                case.id,
                voice_config,
                call,
                verdict,
                transcript_key,
                module_slug=case.module_slug,
                audio_path=audio_key,
                audio_status=audio_status,
                audio_reason=audio_reason,
                brain="llm",
                persona_overrides=persona_overrides_record or None,
                persona_source=persona_source,
            )
        )
        _safe_log(f"voice session recorded status={verdict.status}")

        report = json.loads(report_path.read_text())
        api.put_shard(
            execution_id,
            shard_index,
            {
                "status": "FINISHED",
                "startedAt": started_at,
                "finishedAt": _iso_now(),
                "durationMs": call.duration_ms,
                **(
                    {"errorMessage": "env_failure:CALL_ENV_FAILURE"}
                    if verdict.status == "env_failure"
                    else {}
                ),
                **(
                    {"cloudJobId": os.environ["CLOUD_JOB_ID"]}
                    if os.environ.get("CLOUD_JOB_ID")
                    else {}
                ),
                "stats": report["stats"],
                "results": [
                    {
                        "name": r["name"],
                        "status": r["status"] if r["status"] in ("passed", "failed", "skipped") else "failed",
                        "durationMs": r.get("durationMs", 0),
                        **({"errorMessage": "test_failed"} if r.get("errorMessage") else {}),
                    }
                    for r in report["results"]
                ],
            },
        )
        _safe_log(f"trajectory states={len(call.trajectory)}")
        _safe_log(f"result={evaluation.status.upper()} session={verdict.status}")
        if evaluation.error_message:
            _safe_log("evaluation_failed")
        _safe_log(f"shard {shard_index}/{shard_total} FINISHED reported")
        return 0
    except Exception as exc:  # noqa: BLE001 — report infra failure to the service
        failure_category, failure_code, failure_reason = classify_operational_failure(exc)
        _safe_log(
            "echo-runner serve-execution: FAILED runner_execution_failed "
            f"category={failure_category} code={failure_code}",
            stderr=True,
        )
        try:
            from .live_events import LivePublisher

            failure_live = live if "live" in locals() and live is not None else LivePublisher(
                api_url,
                execution_id,
                shard_index,
                token=token,
                redact=redaction.redact if "redaction" in locals() else None,
                audio_enabled=False,
            )
            failure_live.fail_sync(failure_category, failure_code, failure_reason)
        except Exception:  # noqa: BLE001
            _safe_log("failed_to_publish_live_failure", stderr=True)
        try:
            api.put_shard(
                execution_id,
                shard_index,
                {
                    "status": "FAILED",
                    "startedAt": started_at,
                    "finishedAt": _iso_now(),
                    **(
                        {"cloudJobId": os.environ["CLOUD_JOB_ID"]}
                        if os.environ.get("CLOUD_JOB_ID")
                        else {}
                    ),
                    "errorMessage": f"{failure_category}:{failure_code}",
                },
            )
        except Exception:  # noqa: BLE001
            _safe_log("failed_to_report_shard", stderr=True)
        return 1
