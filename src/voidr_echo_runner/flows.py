"""Journey flow parser (state machine JSON, ARCHITECTURE.md section 6.1).

The runner-side flow carries an extra `keywords` list per state used by the
v0 keyword classifier to map agent turns to states.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FlowState:
    name: str
    expects: list[str] = field(default_factory=list)
    next: list[str] = field(default_factory=list)
    terminal: bool = False
    classification: str | None = None
    max_turns: int | None = None
    evidence: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class JourneyFlow:
    id: str
    source: str | None
    states: dict[str, FlowState]
    deviation_rules: list[dict]
    # Journey-level DTMF preamble (service flow.dialPlan.dtmfSteps, raw dicts
    # with waitFor/send). Precedence in build_case: case dtmfSteps → these →
    # environment dialPlanDefaults. Sends may carry {{massa.*}}/{{env.*}}.
    dial_plan_steps: list[dict] = field(default_factory=list)

    def terminal_states(self) -> set[str]:
        return {name for name, s in self.states.items() if s.terminal}


def load_journey_flow(path: Path) -> JourneyFlow:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "id" not in data or "states" not in data:
        raise ValueError(f"journey flow {path} must define 'id' and 'states'")
    states: dict[str, FlowState] = {}
    for name, raw in data["states"].items():
        states[name] = FlowState(
            name=name,
            expects=list(raw.get("expects", [])),
            next=list(raw.get("next", [])),
            terminal=bool(raw.get("terminal", False)),
            classification=raw.get("classification"),
            max_turns=raw.get("max_turns"),
            evidence=list(raw.get("evidence", [])),
            keywords=list(raw.get("keywords", [])),
        )
    for state in states.values():
        for target in state.next:
            if target not in states:
                raise ValueError(
                    f"journey flow {data['id']}: state {state.name!r} references "
                    f"unknown next state {target!r}"
                )
    return JourneyFlow(
        id=data["id"],
        source=data.get("source"),
        states=states,
        deviation_rules=list(data.get("deviation_rules", [])),
    )
