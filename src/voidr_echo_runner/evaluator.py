"""v0 local trajectory evaluation: must_visit / must_not_visit / max_turns.

Later phases move richer evaluation (LLM judge, rubrics) to voidr-service's
echo module; this local pass is what turns a call into passed/failed for the
runner report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import FlowAssert


@dataclass
class TrajectoryEntry:
    state: str
    turn: int
    utterance: str
    timestamp_ms: int


@dataclass
class EvaluationResult:
    status: str  # "passed" | "failed"
    error_message: str | None = None
    visited: list[str] = field(default_factory=list)


def evaluate_trajectory(
    flow_assert: FlowAssert,
    trajectory: list[TrajectoryEntry],
    agent_turns: int,
    end_reason: str | None,
    transport_error: str | None = None,
) -> EvaluationResult:
    visited_order = [t.state for t in trajectory]
    visited = set(visited_order)
    problems: list[str] = []

    if transport_error:
        problems.append(f"erro de transporte: {transport_error}")

    for state in flow_assert.must_not_visit:
        if state in visited:
            entry = next(t for t in trajectory if t.state == state)
            problems.append(
                f"must_not_visit violado: estado inesperado '{state}' visitado no "
                f"turno {entry.turn} (fala do agente: \"{entry.utterance[:120]}\")"
            )

    missing = [s for s in flow_assert.must_visit if s not in visited]
    if missing:
        problems.append(
            "must_visit não atingido: estados faltantes: " + ", ".join(missing)
        )

    if agent_turns > flow_assert.max_turns:
        problems.append(
            f"max_turns excedido: {agent_turns} turnos do agente > limite {flow_assert.max_turns}"
        )

    if end_reason not in (None, "completed") and not problems:
        problems.append(f"chamada encerrada sem estado terminal esperado (reason={end_reason})")

    if problems:
        return EvaluationResult(status="failed", error_message="; ".join(problems), visited=visited_order)
    return EvaluationResult(status="passed", visited=visited_order)
