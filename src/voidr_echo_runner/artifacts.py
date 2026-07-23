"""Artifact writer: ./out/<run-id>/{transcript,timeline,report}.json.

report.json follows the Voidr runner contract (same shape voidr-k6-runner
PATCHes to /v1/executions/:id/shards/:i):
  stats   {total, passed, failed, flaky, skipped, durationMs}
  results [{name, status, durationMs, errorMessage?}]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluator import EvaluationResult
from .runner import CallResult


def write_artifacts(
    out_dir: Path,
    run_id: str,
    case_id: str,
    call: CallResult,
    evaluation: EvaluationResult,
    meta: dict[str, Any],
) -> Path:
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "transcript.json", {"caseId": case_id, "entries": call.transcript})
    _write_json(
        run_dir / "timeline.json",
        {
            "caseId": case_id,
            "events": call.timeline,
            "trajectory": [
                {
                    "state": t.state,
                    "turn": t.turn,
                    "utterance": t.utterance,
                    "ts": t.timestamp_ms,
                }
                for t in call.trajectory
            ],
            "endReason": call.end_reason,
            "meta": meta,
        },
    )

    result: dict[str, Any] = {
        "name": case_id,
        "status": evaluation.status,
        "durationMs": call.duration_ms,
    }
    if evaluation.error_message:
        result["errorMessage"] = evaluation.error_message
    report = {
        "stats": {
            "total": 1,
            "passed": 1 if evaluation.status == "passed" else 0,
            "failed": 1 if evaluation.status == "failed" else 0,
            "flaky": 0,
            "skipped": 0,
            "durationMs": call.duration_ms,
        },
        "results": [result],
    }
    report_path = run_dir / "report.json"
    _write_json(report_path, report)
    return report_path


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
