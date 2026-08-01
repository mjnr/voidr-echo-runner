"""CLI: uv run echo-runner run --case cases/x.yaml --target ws://localhost:8765 --seed 42"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from . import __version__
from .artifacts import write_artifacts
from .brain import build_brain
from .evaluator import evaluate_trajectory
from .flows import load_journey_flow
from .models import VoiceTestCase, load_persona_catalog
from .projected_secrets import load_projected_secrets
from .runner import CallRunner
from .transport import build_transport

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    load_dotenv(REPO_ROOT / ".env")  # local credentials; never committed
    load_projected_secrets()
    parser = argparse.ArgumentParser(prog="echo-runner", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="execute one voice test case against a target agent")
    run.add_argument("--case", required=True, type=Path, help="voice test case YAML")
    run.add_argument("--target", required=True, help="ws://host:port/ws (mock) or tel:+E164 (Twilio stub)")
    run.add_argument("--seed", type=int, default=None, help="override persona variant_seed")
    run.add_argument("--mode", choices=("text", "audio"), default="text")
    run.add_argument("--personas", type=Path, default=REPO_ROOT / "personas" / "catalog.yaml")
    run.add_argument("--out", type=Path, default=Path("out"))
    run.add_argument("--run-id", default=None)
    run.add_argument(
        "--ambience",
        default=None,
        help=(
            "telephone-channel ambience for the persona audio "
            "(none|quiet|home|office|street[:level]); default quiet in audio mode. "
            "Also honors ECHO_CALL_AMBIENCE."
        ),
    )
    run.add_argument(
        "--no-humanize",
        action="store_true",
        help="disable human realism (memory lapses, humanized reply latency)",
    )
    run.add_argument(
        "--no-redaction",
        action="store_true",
        help="DEV ONLY: skip PII redaction of artifacts (never use with real massas)",
    )
    run.add_argument(
        "--live",
        action="store_true",
        help=(
            "emit live call events to the service (needs VOIDR_API_URL; "
            "executionId = EXECUTION_ID env or the run id)"
        ),
    )

    chat = sub.add_parser(
        "chat",
        help="persona playground: talk to a persona interactively (LLM via hive)",
    )
    chat.add_argument("--persona", required=True, help="persona id from the catalog")
    chat.add_argument("--personas", type=Path, default=REPO_ROOT / "personas" / "catalog.yaml")
    chat.add_argument("--journey", type=Path, default=None, help="journey flow JSON for state tracking")
    chat.add_argument("--voice", action="store_true", help="speak replies (ElevenLabs + afplay)")
    chat.add_argument("--seed", type=int, default=None, help="best-effort LLM seed")
    chat.add_argument("--goal", default=None, help="persona goal (fills goalTemplate's {goal})")

    serve = sub.add_parser(
        "serve-execution",
        help=(
            "run one voidr-service execution shard (env contract: VOIDR_API_URL, "
            "EXECUTION_ID, VOIDR_ORG_ID, VOIDR_CLIENT_ID/SECRET or VOIDR_ACCESS_TOKEN, "
            "SHARDS_CURRENT/TOTAL, ENVIRONMENT_PARAMS)"
        ),
    )
    serve.add_argument("--out", type=Path, default=Path("out"))

    session_stt = sub.add_parser(
        "serve-session-stt",
        help="serve the governed Deepgram boundary for browser Session voice notes",
    )
    session_stt.add_argument("--host", default="127.0.0.1")
    session_stt.add_argument("--port", type=int, default=3110)

    args = parser.parse_args(argv)
    if args.command == "serve-execution":
        from .service_mode import serve_execution

        sys.exit(serve_execution(args.out))
    if args.command == "serve-session-stt":
        from .session_stt_server import serve_session_stt

        sys.exit(serve_session_stt(args.host, args.port))
    if args.command == "chat":
        from .chat import run_chat

        sys.exit(run_chat(args))
    sys.exit(_run(args))


def _run(args: argparse.Namespace) -> int:
    try:
        case = VoiceTestCase.load(args.case)
        catalog = load_persona_catalog(args.personas)
        if case.persona.base not in catalog:
            raise KeyError(
                f"persona {case.persona.base!r} not found in {args.personas} "
                f"(available: {', '.join(sorted(catalog))})"
            )
        persona = catalog[case.persona.base]
        seed = args.seed if args.seed is not None else case.persona.variant_seed
        flow_path = (args.case.parent / case.journey_flow).resolve()
        if not flow_path.exists():
            flow_path = (REPO_ROOT / case.journey_flow).resolve()
        flow = load_journey_flow(flow_path)

        from .emotional import EmotionalStateMachine

        emotional = EmotionalStateMachine.for_persona(persona, seed=seed)
        # EXEC-REALISM: massa from ECHO_MASSA (env JSON) or the persona's own
        # identity facts; humanizer plans memory lapses + humanized latency.
        import os as _os

        from .humanize import Humanizer, MassaFacts

        massa = MassaFacts.resolve(
            {"ECHO_MASSA": _os.environ.get("ECHO_MASSA", "")}, persona
        )
        if massa:
            case.massa = {**case.massa, **massa.values}
        humanizer = None
        if not args.no_humanize:
            humanizer = Humanizer(persona, seed, massa)
        brain = build_brain(persona, case.goal, seed)
        brain.emotional = emotional  # current state injected per turn
        if massa:
            brain.personal_data = massa.personal_data_lines()
        # The history sent to the hive must carry massa as placeholders
        # (the gateway 422s on clear PII) — share the case deny-list.
        from .redaction import build_session_for_case

        brain.redaction = build_session_for_case(case)
        is_pstn = args.target.startswith(("tel:", "+"))
        if is_pstn and args.mode != "audio":
            raise RuntimeError("tel: targets are audio-only — run with --mode audio")
        send_digits = None
        if is_pstn and case.dial_plan.dtmf_steps:
            # IVR digits at answer time, with `w` (0.5s) pauses between steps.
            send_digits = "ww" + "ww".join(
                step.send for step in case.dial_plan.dtmf_steps
            )
        transport = build_transport(args.target, send_digits=send_digits)
        engine = recorder = channel_fx = None
        receive_timeout = 10.0
        if args.mode == "audio":
            import os

            os.environ.setdefault("LOGURU_LEVEL", "WARNING")  # quiet pipecat logs
            # Imports pipecat; validates DEEPGRAM/ELEVENLABS keys with clear errors.
            from .audio import AudioTransportAdapter, PipecatAudioEngine, StereoCallRecorder
            from .callfx import TelephoneChannelFx, parse_ambience

            engine = PipecatAudioEngine(voice_id=persona.speech.voiceId)
            recorder = StereoCallRecorder()
            ambience = parse_ambience(
                args.ambience or os.environ.get("ECHO_CALL_AMBIENCE")
            )
            if ambience.enabled:
                channel_fx = TelephoneChannelFx(
                    ambience, seed=seed, sample_rate=engine.sample_rate
                )
            transport = AudioTransportAdapter(
                transport, engine, recorder, channel_fx=channel_fx
            )
            receive_timeout = 45.0  # remote STT+TTS per turn
    except Exception as exc:  # noqa: BLE001 — setup errors are user-facing
        print(f"echo-runner: setup error: {exc}", file=sys.stderr)
        return 2

    run_id = args.run_id or f"{case.id}-{int(time.time())}"
    print(f"▶ case={case.id} persona={persona.id} seed={seed} target={args.target} mode={args.mode}")

    live = None
    if args.live:
        import os

        api_url = os.environ.get("VOIDR_API_URL")
        if not api_url:
            print("⚠ --live ignorado: VOIDR_API_URL não definido", file=sys.stderr)
        else:
            from .live_events import LivePublisher
            from .redaction import build_session_for_case as _build_deny

            live_redaction = _build_deny(case)
            live = LivePublisher(
                api_url,
                os.environ.get("EXECUTION_ID") or run_id,
                1,
                token=os.environ.get("VOIDR_ACCESS_TOKEN"),
                # Live output always receives full PII + massa redaction, even
                # when local persisted artifacts use --no-redaction.
                redact=live_redaction.redact,
                has_sensitive_data=lambda text: bool(live_redaction.find_spans(text)),
                audio_enabled=os.environ.get("ECHO_LIVE_AUDIO", "1") != "0",
            )
            if args.mode == "audio":
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
    if live is not None:
        live.finish_sync(call.end_reason or "unknown", evaluation.status)
    meta = {
        "persona": {"id": persona.id, "version": persona.version, "variantSeed": seed},
        "journeyFlowId": flow.id,
        **({"moduleSlug": case.module_slug} if case.module_slug else {}),
        **({"testPlanId": case.test_plan_id} if case.test_plan_id else {}),
        "runnerVersion": __version__,
        "mode": args.mode,
        "target": args.target,
        "brain": "hive-llm",
    }
    if humanizer is not None:
        meta["humanize"] = humanizer.config_record()
    if emotional.history:
        meta["emotionalCurve"] = emotional.curve()
        meta["emotionalFinal"] = {
            "emotion": emotional.emotion,
            "intensity": emotional.intensity,
        }
    if args.mode == "audio" and recorder is not None:
        meta["audio"] = {
            "sttProvider": "deepgram",
            "ttsProvider": "elevenlabs",
            "voiceId": persona.speech.voiceId,
            "sttTurns": transport.stt_turns,
            "ttsTurns": transport.tts_turns,
            "wavDurationMs": recorder.duration_ms,
            **({"channelFx": channel_fx.record()} if channel_fx is not None else {}),
        }

    wav_path = None
    if args.no_redaction:
        print(
            "⚠ --no-redaction: artifacts ficam com PII/massa EM CLARO — uso "
            "exclusivo de dev, nunca com massas reais",
            file=sys.stderr,
        )
        if args.mode == "audio" and recorder is not None:
            meta["audio"]["wavFile"] = "call.wav"
            wav_path = args.out / run_id / "call.wav"
            recorder.save(wav_path)
    else:
        from .redaction import build_session_for_case, redact_call_result

        session = build_session_for_case(case)
        if args.mode == "audio" and recorder is not None:
            from .audio_redaction import redact_call_audio

            meta["audio"].update(
                redact_call_audio(recorder, transport.utterances, session, args.out / run_id)
            )
            meta["audio"]["wavFile"] = "call.redacted.wav"
            wav_path = args.out / run_id / "call.redacted.wav"
        redact_call_result(call, session)
        if evaluation.error_message:
            evaluation.error_message = session.redact(evaluation.error_message)
        meta = session.redact_deep(meta)
        meta["piiRedactionReport"] = session.report()

    report_path = write_artifacts(args.out, run_id, case.id, call, evaluation, meta=meta)
    if wav_path is not None:
        print(f"  áudio: {wav_path} ({recorder.duration_ms} ms, estéreo tester/agente)")

    trajectory = " -> ".join(t.state for t in call.trajectory) or "(vazia)"
    print(f"  trajetória: {trajectory}")
    print(f"  turnos do agente: {call.agent_turns}  encerramento: {call.end_reason}")
    print(f"  resultado: {evaluation.status.upper()}")
    if evaluation.error_message:
        print(f"  motivo: {evaluation.error_message}")
    print(f"  report: {report_path}")
    print(json.dumps(json.loads(report_path.read_text())["stats"], ensure_ascii=False))
    return 0 if evaluation.status == "passed" else 1


if __name__ == "__main__":
    main()
