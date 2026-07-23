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
from .runner import CallRunner
from .transport import build_transport

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    load_dotenv(REPO_ROOT / ".env")  # local credentials; never committed
    parser = argparse.ArgumentParser(prog="echo-runner", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="execute one voice test case against a target agent")
    run.add_argument("--case", required=True, type=Path, help="voice test case YAML")
    run.add_argument("--target", required=True, help="ws://host:port/ws (mock) or tel:+E164 (Twilio stub)")
    run.add_argument("--seed", type=int, default=None, help="override persona variant_seed")
    run.add_argument("--mode", choices=("text", "audio"), default="text")
    run.add_argument("--brain", choices=("scripted", "llm"), default="scripted")
    run.add_argument("--personas", type=Path, default=REPO_ROOT / "personas" / "catalog.yaml")
    run.add_argument("--out", type=Path, default=Path("out"))
    run.add_argument("--run-id", default=None)

    chat = sub.add_parser(
        "chat",
        help="persona playground: talk to a persona interactively (LLM via hive)",
    )
    chat.add_argument("--persona", required=True, help="persona id from the catalog")
    chat.add_argument("--personas", type=Path, default=REPO_ROOT / "personas" / "catalog.yaml")
    chat.add_argument("--journey", type=Path, default=None, help="journey flow JSON for state tracking")
    chat.add_argument("--voice", action="store_true", help="speak replies (ElevenLabs + afplay)")
    chat.add_argument("--escalate", action="store_true", help="always route to the escalation model")
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

    args = parser.parse_args(argv)
    if args.command == "serve-execution":
        from .service_mode import serve_execution

        sys.exit(serve_execution(args.out))
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

        brain = build_brain(args.brain, persona, case.goal, seed)
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
        engine = recorder = None
        receive_timeout = 10.0
        if args.mode == "audio":
            import os

            os.environ.setdefault("LOGURU_LEVEL", "WARNING")  # quiet pipecat logs
            # Imports pipecat; validates DEEPGRAM/ELEVENLABS keys with clear errors.
            from .audio import AudioTransportAdapter, PipecatAudioEngine, StereoCallRecorder

            engine = PipecatAudioEngine(voice_id=persona.speech.voiceId)
            recorder = StereoCallRecorder()
            transport = AudioTransportAdapter(transport, engine, recorder)
            receive_timeout = 45.0  # remote STT+TTS per turn
    except Exception as exc:  # noqa: BLE001 — setup errors are user-facing
        print(f"echo-runner: setup error: {exc}", file=sys.stderr)
        return 2

    run_id = args.run_id or f"{case.id}-{int(time.time())}"
    print(f"▶ case={case.id} persona={persona.id} seed={seed} target={args.target} mode={args.mode}")

    runner = CallRunner(case, flow, brain, transport, receive_timeout=receive_timeout)

    async def _run_call():
        try:
            return await runner.run()
        finally:
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
    meta = {
        "persona": {"id": persona.id, "version": persona.version, "variantSeed": seed},
        "journeyFlowId": flow.id,
        "runnerVersion": __version__,
        "mode": args.mode,
        "target": args.target,
    }
    if args.mode == "audio" and recorder is not None:
        meta["audio"] = {
            "sttProvider": "deepgram",
            "ttsProvider": "elevenlabs",
            "voiceId": persona.speech.voiceId,
            "sttTurns": transport.stt_turns,
            "ttsTurns": transport.tts_turns,
            "wavDurationMs": recorder.duration_ms,
            "wavFile": "call.wav",
        }
    report_path = write_artifacts(args.out, run_id, case.id, call, evaluation, meta=meta)
    if args.mode == "audio" and recorder is not None:
        wav_path = args.out / run_id / "call.wav"
        recorder.save(wav_path)
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
