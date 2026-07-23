"""CLI: uv run echo-runner run --case cases/x.yaml --target ws://localhost:8765 --seed 42"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

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

        if args.mode == "audio":
            from .audio import resolve_audio_services

            resolve_audio_services()  # raises with a clear message when keys are absent
            raise NotImplementedError("audio mode pipeline is not wired yet (see audio.py)")

        brain = build_brain(args.brain, persona, case.goal, seed)
        transport = build_transport(args.target)
    except Exception as exc:  # noqa: BLE001 — setup errors are user-facing
        print(f"echo-runner: setup error: {exc}", file=sys.stderr)
        return 2

    run_id = args.run_id or f"{case.id}-{int(time.time())}"
    print(f"▶ case={case.id} persona={persona.id} seed={seed} target={args.target}")

    runner = CallRunner(case, flow, brain, transport)
    call = asyncio.run(runner.run())
    evaluation = evaluate_trajectory(
        case.assertion.flow,
        call.trajectory,
        call.agent_turns,
        call.end_reason,
        transport_error=call.transport_error,
    )
    report_path = write_artifacts(
        args.out,
        run_id,
        case.id,
        call,
        evaluation,
        meta={
            "persona": {"id": persona.id, "version": persona.version, "variantSeed": seed},
            "journeyFlowId": flow.id,
            "runnerVersion": __version__,
            "mode": args.mode,
            "target": args.target,
        },
    )

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
