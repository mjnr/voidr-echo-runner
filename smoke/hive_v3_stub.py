#!/usr/bin/env python3
"""Deterministic Hive persona-turn v3 boundary double for the official smoke."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


POLICY_VERSION = "echo-persona-turn-v3.0.0"
PROMPT_VERSION = "echo-persona-system-v3.0.0"
MODEL_ALIAS = "deepseek-v4-pro"
PIN_PATTERN = re.compile(
    r"^deepseek-v4-pro@(sha256:[0-9a-f]{64})$",
    re.IGNORECASE,
)


class InvalidRequest(ValueError):
    pass


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InvalidRequest(f"{key} must be a non-empty string")
    return value


def validate_request(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise InvalidRequest("body must be a JSON object")
    for key in ("organizationId", "conversationId", "turnId", "deadlineAt"):
        _required_string(data, key)
    for key in ("conversationId", "turnId"):
        try:
            uuid.UUID(data[key])
        except (ValueError, TypeError) as exc:
            raise InvalidRequest(f"{key} must be a UUID") from exc
    if data.get("policyVersion") != POLICY_VERSION:
        raise InvalidRequest(f"policyVersion must be {POLICY_VERSION}")
    try:
        datetime.fromisoformat(data["deadlineAt"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidRequest("deadlineAt must be ISO-8601") from exc

    persona = data.get("persona")
    if not isinstance(persona, dict):
        raise InvalidRequest("persona must be an object")
    _required_string(persona, "id")

    journey = data.get("journeyState")
    if not isinstance(journey, dict):
        raise InvalidRequest("journeyState must be an object")
    _required_string(journey, "flowSlug")
    _required_string(journey, "currentState")
    if not isinstance(journey.get("expects"), list):
        raise InvalidRequest("journeyState.expects must be an array")

    history = data.get("history")
    if not isinstance(history, list) or not history:
        raise InvalidRequest("history must be a non-empty array")
    last = history[-1]
    if not isinstance(last, dict) or last.get("role") != "agent":
        raise InvalidRequest("history must end with an agent turn")
    _required_string(last, "text")

    options = data.get("options", {})
    if not isinstance(options, dict):
        raise InvalidRequest("options must be an object")
    if "seed" in options and (
        not isinstance(options["seed"], int) or isinstance(options["seed"], bool)
    ):
        raise InvalidRequest("options.seed must be an integer")
    return data


def persona_text(request: dict[str, Any]) -> str:
    state = request["journeyState"]["currentState"]
    agent_text = request["history"][-1]["text"].lower()
    if state == "saudacao":
        goal = request.get("journeyGoal")
        if isinstance(goal, str) and goal.strip():
            # Keep the intended journey as the strongest match while giving
            # MOCK_DEVIATION=jornada_errada an unambiguous runner-visible
            # alternate for the financial-block smoke case.
            if "bloqueio-financeiro" in request["journeyState"]["flowSlug"]:
                return f"{goal}. Também queria consultar meu saldo."
            return goal
        return request["persona"].get("goalTemplate") or "Preciso de ajuda com a linha."
    if state == "identificacao":
        # A jornada errada usada pelo mock pode pedir dados antes de revelar
        # sua classificação. Estes valores são sintéticos e serão redigidos
        # pelo runner; o stub continua sendo apenas o peer HTTP do Hive.
        if "cpf" in agent_text:
            return "Meu CPF de teste é 390.533.447-05."
        if "nascimento" in agent_text:
            return "Nasci em 10 de janeiro de 1980."
        return "Sim, sou o titular da linha."
    if state in {"diagnostico_saldo", "diagnostico_bloqueio"}:
        return "Sim, quero resolver isso agora."
    if state in {"oferta_recarga", "oferta_pagamento"}:
        return "Sim, pode enviar o link por SMS, por favor."
    return "Entendi, obrigado."


def success_response(request: dict[str, Any], model_revision: str) -> dict[str, Any]:
    match = PIN_PATTERN.fullmatch(model_revision)
    if match is None:
        raise RuntimeError(
            "SMOKE_HIVE_MODEL_REVISION must be an immutable deepseek-v4-pro digest"
        )
    request_hash = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    model_hash = hashlib.sha256(model_revision.encode()).hexdigest()
    provenance = {
        "source": "hive-llm",
        "contractVersion": "v3",
        "conversationId": request["conversationId"],
        "turnId": request["turnId"],
        "policyVersion": POLICY_VERSION,
        "promptVersion": PROMPT_VERSION,
        "promptHash": f"sha256:{request_hash}",
        "provider": "smoke-boundary-stub",
        "modelAlias": MODEL_ALIAS,
        "model": MODEL_ALIAS,
        "modelResolved": model_revision,
        "modelVersion": model_revision,
        "deploymentPin": model_revision,
        "deploymentId": model_revision,
        "deploymentDigest": match.group(1),
        "modelHash": model_hash,
        "completionId": f"smoke-{request_hash[:20]}",
        "traceId": f"smoke-{request_hash[20:40]}",
        "generatedAt": "2026-01-01T00:00:00Z",
        "attempts": 1,
    }
    return {
        "text": persona_text(request),
        "provenance": provenance,
        "usage": {"costUsd": 0.0},
    }


class HiveV3Handler(BaseHTTPRequestHandler):
    server_version = "HiveV3SmokeStub/1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._json(200, {"status": "ok", "contractVersion": "v3"})
            return
        self._json(404, {"error": {"code": "NOT_FOUND", "message": "not found"}})

    def do_POST(self) -> None:
        if self.path != "/echo/persona-turn/v3":
            self._json(404, {"error": {"code": "NOT_FOUND", "message": "not found"}})
            return
        expected_token = self.server.gateway_token  # type: ignore[attr-defined]
        if self.headers.get("Authorization") != f"Bearer {expected_token}":
            self._json(
                401,
                {"error": {"code": "UNAUTHORIZED", "message": "invalid bearer token"}},
            )
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            request = validate_request(json.loads(self.rfile.read(size)))
            response = success_response(
                request,
                self.server.model_revision,  # type: ignore[attr-defined]
            )
        except (InvalidRequest, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._json(
                400,
                {"error": {"code": "INVALID_REQUEST", "message": str(exc)}},
            )
            return
        if self.server.response_mode == "malformed":  # type: ignore[attr-defined]
            response["provenance"]["deploymentPin"] = "mutable-latest"
        self._json(200, response)


def main() -> None:
    host = os.environ.get("SMOKE_HIVE_HOST", "127.0.0.1")
    port = int(os.environ.get("SMOKE_HIVE_PORT", "18765"))
    token = os.environ.get("SMOKE_HIVE_TOKEN", "smoke-hive-v3-token")
    revision = os.environ.get(
        "SMOKE_HIVE_MODEL_REVISION",
        (
            "deepseek-v4-pro@sha256:"
            "59e858aa0bd9bdbc7524a5dd39d84904747dacd1f85d152d0c04bcc373db9a08"
        ),
    )
    # Validate the pin before announcing readiness.
    if PIN_PATTERN.fullmatch(revision) is None:
        raise SystemExit(
            "SMOKE_HIVE_MODEL_REVISION must be an immutable "
            "deepseek-v4-pro digest"
        )
    server = ThreadingHTTPServer((host, port), HiveV3Handler)
    server.gateway_token = token  # type: ignore[attr-defined]
    server.model_revision = revision  # type: ignore[attr-defined]
    server.response_mode = os.environ.get("SMOKE_HIVE_STUB_RESPONSE", "valid")  # type: ignore[attr-defined]
    server.serve_forever()


if __name__ == "__main__":
    main()
