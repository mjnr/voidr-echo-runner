"""Call orchestrator: dial plan -> conversation loop -> trajectory tracking.

Produces the raw material for artifacts: diarized transcript, event timeline
and the classified state trajectory that the evaluator scores.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .brain import HiveError, PersonaBrain
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
    failure_status: str | None = None
    ai_error: dict[str, Any] | None = None
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
        receive_timeout: float = RECEIVE_TIMEOUT_S,
        emotional: Any | None = None,
        live: Any | None = None,
        humanizer: Any | None = None,
    ):
        self.case = case
        self.flow = flow
        self.brain = brain
        self.transport = transport
        self.receive_timeout = receive_timeout
        self.classifier = KeywordStateClassifier(flow)
        self.result = CallResult()
        self._pending_dtmf = list(case.dial_plan.dtmf_steps)
        self._turn = 0
        # EmotionalStateMachine (emotional.py): updated per agent turn; the
        # per-turn state goes to the timeline as the auditable emotional curve.
        self.emotional = emotional
        # LivePublisher (live_events.py): fire-and-forget tap on the hooks
        # below — real-time UI feed, never blocks nor fails the call.
        self.live = live
        # Humanizer (humanize.py, EXEC-REALISM): plans memory lapses per agent
        # turn (prompt directives for the LLM brain), substitutes {{massa.*}}
        # placeholders in the reply and produces the humanized reply latency.
        self.humanizer = humanizer
        self._last_reply_monotonic: float | None = None

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _record_event(self, event_type: str, **data: Any) -> None:
        self.result.timeline.append({"ts": self._now_ms(), "type": event_type, **data})
        if self.live is not None:
            self.live.on_timeline_event(event_type, data)

    def _record_transcript(
        self,
        speaker: str,
        text: str,
        state: str | None = None,
        source: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "index": len(self.result.transcript),
            "speaker": speaker,
            "text": text,
            "ts": self._now_ms(),
        }
        if state is not None:
            entry["state"] = state
        if source is not None:
            entry["source"] = source
        if provenance is not None:
            entry.update(provenance)
            entry["provenance"] = dict(provenance)
        self.result.transcript.append(entry)
        if self.live is not None:
            self.live.on_transcript(entry)

    async def run(self) -> CallResult:
        self.result.started_at_ms = self._now_ms()
        try:
            await self.transport.connect()
            self._record_event("connected", target=getattr(self.transport, "url", "?"))
            await self._loop()
        except HiveError as exc:
            self.result.transport_error = f"HiveError: {exc}"
            self.result.failure_status = exc.outcome
            self.result.ai_error = {
                "component": "hive",
                "outcome": exc.outcome,
                "turnId": exc.turn_id,
                "statusCode": exc.status_code,
                "code": exc.code or "hive_error",
            }
            self.result.end_reason = "hive_error"
            self._record_event(
                "hive_generation_failed",
                **self.result.ai_error,
            )
        except Exception as exc:  # noqa: BLE001 — reported in the result, not raised
            self.result.transport_error = f"{type(exc).__name__}: {exc}"
            self._record_event("error", message=self.result.transport_error)
        finally:
            cleanup_error: Exception | None = None
            for name, kwargs in (
                ("finish_audio", {"sample_rate": 16_000}),
                ("hangup", {}),
                ("close", {}),
            ):
                action = getattr(self.transport, name, None)
                if action is None:
                    continue
                try:
                    await action(**kwargs)
                except Exception as exc:  # noqa: BLE001 - continue remaining cleanup
                    cleanup_error = cleanup_error or exc
            if cleanup_error is not None and self.result.transport_error is None:
                self.result.transport_error = (
                    f"cleanup {type(cleanup_error).__name__}: {cleanup_error}"
                )
                self._record_event("error", message=self.result.transport_error)
            self.result.ended_at_ms = self._now_ms()
        return self.result

    async def _loop(self) -> None:
        import asyncio

        hard_cap = self.case.assertion.flow.max_turns + HARD_CAP_EXTRA_TURNS
        while True:
            try:
                msg = await self.transport.receive(timeout=self.receive_timeout)
            except (TimeoutError, asyncio.TimeoutError):
                self.result.transport_error = (
                    f"timeout de {self.receive_timeout:.0f}s aguardando resposta do agente"
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

            await self._handle_agent(text, source=msg.get("source", "protocol"))
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

    async def _handle_agent(self, text: str, source: str = "protocol") -> None:
        """Record + classify an agent turn, then reply.

        Terminal turns are still replied to: the far side ends the call with a
        `call_ended` event and late sends are swallowed by the transport, while
        mid-call turns misclassified as terminal (deviations) keep flowing."""
        self._turn += 1
        self.result.agent_turns += 1
        state = self.classifier.classify(text)
        self._record_transcript("agent", text, state=state, source=source)
        self._record_event("agent_turn", turn=self._turn, state=state)

        state_changed = state is not None and (
            not self.result.trajectory or self.result.trajectory[-1].state != state
        )
        if state_changed:
            self.result.trajectory.append(
                TrajectoryEntry(
                    state=state,
                    turn=self._turn,
                    utterance=text,
                    timestamp_ms=self._now_ms(),
                )
            )
            self._record_event("state_transition", state=state, turn=self._turn)

        # Journey context for the LLM persona (mission delivery 2): keep the
        # brain's journeyState in sync with the classified flow state — same
        # wiring the chat mode does. Without this the persona-turn prompt ran
        # the whole call on the generic "conversa-livre" state.
        if state is not None and hasattr(self.brain, "journey_state"):
            state_def = self.flow.states.get(state)
            self.brain.journey_state = {
                "flowSlug": self.flow.id,
                "currentState": state,
                "expects": list(state_def.expects) if state_def else [],
            }

        if self.emotional is not None:
            latency_s = (
                time.monotonic() - self._last_reply_monotonic
                if self._last_reply_monotonic is not None
                else None
            )
            emo = self.emotional.update(
                text, state_changed=state_changed, latency_s=latency_s
            )
            self._record_event(
                "emotional_state",
                turn=self._turn,
                emotion=emo.emotion,
                intensity=emo.intensity,
                events=emo.events,
                **({"action": emo.action} if emo.action else {}),
            )

        plan = None
        if self.humanizer is not None:
            plan = self.humanizer.plan_turn(
                text,
                emotion=(self.emotional.emotion if self.emotional is not None else None),
                emotion_intensity=(
                    self.emotional.intensity if self.emotional is not None else None
                ),
            )
            if plan.directives and hasattr(self.brain, "turn_directives"):
                self.brain.turn_directives = list(plan.directives)

        turn = self.brain.take_turn(text)
        reply = turn["text"]
        # Real v3 responses carry the canonical nested DTO. The top-level
        # fallback keeps older test/custom PersonaBrain implementations
        # compatible without weakening LLMBrain's strict response validation.
        nested = turn.get("provenance")
        provenance = dict(nested) if isinstance(nested, dict) else {
            key: turn[key]
            for key in (
                "source",
                "turnId",
                "promptVersion",
                "modelVersion",
                "modelHash",
                "policyVersion",
            )
            if key in turn
        }
        trace = turn.get("trace")
        runner_trace = dict(trace) if isinstance(trace, dict) else {}
        if runner_trace:
            # The voice-session service contract persists validated trace
            # correlation inside provenance (not as a transcript sibling).
            provenance["trace"] = runner_trace

        if self.humanizer is not None and plan is not None:
            # {{massa.*}} placeholders become real values OUTSIDE the LLM;
            # persisted artifacts are redacted later (deny-list covers them).
            reply = self.humanizer.finalize_reply(reply)
            if self.humanizer.timing_enabled:
                import asyncio

                delay_s = self.humanizer.reply_delay_s(
                    reply,
                    plan,
                    emotion_intensity=(
                        self.emotional.intensity if self.emotional is not None else None
                    ),
                )
                self._record_event(
                    "humanized_turn",
                    turn=self._turn,
                    delayMs=int(delay_s * 1000),
                    memoryLapse=plan.memory_lapse,
                    **({"lapseCategory": plan.lapse_category} if plan.lapse_category else {}),
                )
                record_silence = getattr(self.transport, "record_silence", None)
                if record_silence is not None:
                    # the WAV shows the human gap (with ambience) on the line
                    record_silence(delay_s)
                await asyncio.sleep(delay_s)

        self._record_transcript(
            "tester",
            reply,
            source="hive-llm",
            provenance=provenance,
        )
        self._record_event(
            "tester_turn",
            turn=self._turn,
            source="hive-llm",
            turnId=provenance["turnId"],
            promptVersion=provenance["promptVersion"],
            modelVersion=provenance["modelVersion"],
            policyVersion=provenance["policyVersion"],
            trace=runner_trace,
        )
        await self.transport.send_text(reply)
        self._last_reply_monotonic = time.monotonic()
