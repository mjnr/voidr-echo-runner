"""Call orchestrator: dial plan -> conversation loop -> trajectory tracking.

Produces the raw material for artifacts: diarized transcript, event timeline
and the classified state trajectory that the evaluator scores.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .brain import PersonaBrain
from .classifier import KeywordStateClassifier
from .evaluator import TrajectoryEntry
from .flows import JourneyFlow
from .models import VoiceTestCase
from .textutil import keyword_matches, normalize
from .transport import CallTransport

RECEIVE_TIMEOUT_S = 10.0
HARD_CAP_EXTRA_TURNS = 6


@dataclass
class CallResult:
    transcript: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    trajectory: list[TrajectoryEntry] = field(default_factory=list)
    agent_turns: int = 0
    end_reason: str | None = None
    transport_error: str | None = None
    started_at_ms: int = 0
    ended_at_ms: int = 0

    @property
    def duration_ms(self) -> int:
        return max(0, self.ended_at_ms - self.started_at_ms)


class CallRunner:
    def __init__(
        self,
        case: VoiceTestCase,
        flow: JourneyFlow,
        brain: PersonaBrain,
        transport: CallTransport,
    ):
        self.case = case
        self.flow = flow
        self.brain = brain
        self.transport = transport
        self.classifier = KeywordStateClassifier(flow)
        self.result = CallResult()
        self._pending_dtmf = list(case.dial_plan.dtmf_steps)
        self._turn = 0

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _record_event(self, event_type: str, **data: Any) -> None:
        self.result.timeline.append({"ts": self._now_ms(), "type": event_type, **data})

    def _record_transcript(self, speaker: str, text: str, state: str | None = None) -> None:
        entry: dict[str, Any] = {
            "index": len(self.result.transcript),
            "speaker": speaker,
            "text": text,
            "ts": self._now_ms(),
        }
        if state is not None:
            entry["state"] = state
        self.result.transcript.append(entry)

    async def run(self) -> CallResult:
        self.result.started_at_ms = self._now_ms()
        try:
            await self.transport.connect()
            self._record_event("connected", target=getattr(self.transport, "url", "?"))
            await self._loop()
        except Exception as exc:  # noqa: BLE001 — reported in the result, not raised
            self.result.transport_error = f"{type(exc).__name__}: {exc}"
            self._record_event("error", message=self.result.transport_error)
        finally:
            self.result.ended_at_ms = self._now_ms()
        return self.result

    async def _loop(self) -> None:
        import asyncio

        hard_cap = self.case.assertion.flow.max_turns + HARD_CAP_EXTRA_TURNS
        while True:
            try:
                msg = await self.transport.receive(timeout=RECEIVE_TIMEOUT_S)
            except (TimeoutError, asyncio.TimeoutError):
                self.result.transport_error = (
                    f"timeout de {RECEIVE_TIMEOUT_S:.0f}s aguardando resposta do agente"
                )
                self._record_event("error", message=self.result.transport_error)
                await self.transport.hangup()
                return
            if msg is None:
                if self.result.end_reason is None:
                    self.result.end_reason = "connection_closed"
                    self._record_event("call_ended", reason="connection_closed")
                return

            msg_type = msg.get("type")
            if msg_type == "event":
                if msg.get("name") == "call_ended":
                    self.result.end_reason = msg.get("reason", "unknown")
                    self._record_event("call_ended", reason=self.result.end_reason)
                    return
                self._record_event("remote_event", **{k: v for k, v in msg.items() if k != "type"})
                continue
            if msg_type == "error":
                self.result.transport_error = str(msg.get("message"))
                self._record_event("remote_error", message=self.result.transport_error)
                return
            if msg_type != "text":
                self._record_event("unexpected_message", raw=msg)
                continue

            speaker = msg.get("speaker", "agent")
            text = str(msg.get("text", ""))
            if speaker == "ura":
                await self._handle_ura(text)
                continue

            await self._handle_agent(text)
            if self.result.agent_turns >= hard_cap:
                self._record_event("hard_cap_reached", agent_turns=self.result.agent_turns)
                await self.transport.hangup()
                self.result.end_reason = "runner_hangup"
                return

    async def _handle_ura(self, text: str) -> None:
        self._record_transcript("ura", text)
        self._record_event("ura_prompt", text=text)
        if not self._pending_dtmf:
            raise RuntimeError(
                f"URA prompt received but the dial plan is exhausted: {text!r} "
                "(access denied or unexpected IVR loop?)"
            )
        step = self._pending_dtmf[0]
        matches = True
        if step.wait_for_prompt_matching:
            matches = keyword_matches(step.wait_for_prompt_matching, text) or (
                normalize(step.wait_for_prompt_matching) in normalize(text)
            )
        if not matches:
            raise RuntimeError(
                f"URA prompt {text!r} does not match expected dial-plan step "
                f"{step.wait_for_prompt_matching!r}"
            )
        self._pending_dtmf.pop(0)
        await self.transport.send_dtmf(step.send)
        self._record_event("dtmf_sent", digits=step.send)

    async def _handle_agent(self, text: str) -> None:
        """Record + classify an agent turn, then reply.

        Terminal turns are still replied to: the far side ends the call with a
        `call_ended` event and late sends are swallowed by the transport, while
        mid-call turns misclassified as terminal (deviations) keep flowing."""
        self._turn += 1
        self.result.agent_turns += 1
        state = self.classifier.classify(text)
        self._record_transcript("agent", text, state=state)
        self._record_event("agent_turn", turn=self._turn, state=state)

        if state is not None and (
            not self.result.trajectory or self.result.trajectory[-1].state != state
        ):
            self.result.trajectory.append(
                TrajectoryEntry(
                    state=state,
                    turn=self._turn,
                    utterance=text,
                    timestamp_ms=self._now_ms(),
                )
            )
            self._record_event("state_transition", state=state, turn=self._turn)

        reply = self.brain.reply(text)
        self._record_transcript("tester", reply)
        self._record_event("tester_turn", turn=self._turn)
        await self.transport.send_text(reply)
