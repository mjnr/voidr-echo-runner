"""Scoped metrics and metadata-only structured audit events."""

from __future__ import annotations

import json
import time
from collections import Counter
from threading import Lock
from typing import Callable

_ALLOWED_TAGS = {"org", "execution", "shard", "provider", "model", "modality"}
_METRIC_PROVIDERS = {"deepgram", "elevenlabs"}
_METRIC_STATUSES = {"ok", "rejected", "error"}


class VoiceObservability:
    def __init__(self, sink: Callable[[str], None] = print):
        self._sink = sink
        self._counts: Counter[tuple[str, ...]] = Counter()
        self._latency_sum: Counter[tuple[str, ...]] = Counter()
        self._lock = Lock()

    def audit(
        self,
        event: str,
        tags: dict[str, str],
        *,
        status: str,
        error_code: str | None = None,
        duration_ms: int | None = None,
        chunks: int | None = None,
    ) -> None:
        safe_tags = {key: tags[key] for key in _ALLOWED_TAGS if key in tags}
        record: dict[str, object] = {
            "event": event,
            "timestamp_ms": int(time.time() * 1000),
            **safe_tags,
            "status": status,
        }
        if error_code:
            record["error_code"] = error_code
        if duration_ms is not None:
            record["duration_ms"] = duration_ms
        if chunks is not None:
            record["chunks"] = chunks
        self._sink(json.dumps(record, separators=(",", ":"), sort_keys=True))
        metric_event = event if event in {"voice_stt", "voice_tts"} else "voice_other"
        metric_provider = safe_tags.get("provider", "unknown")
        if metric_provider not in _METRIC_PROVIDERS:
            metric_provider = "unknown"
        metric_status = status if status in _METRIC_STATUSES else "error"
        scope = (metric_event, metric_provider)
        with self._lock:
            self._counts[(*scope, metric_status)] += 1
            if duration_ms is not None:
                self._latency_sum[scope] += duration_ms

    def prometheus(self) -> str:
        lines = [
            "# HELP voice_gateway_requests_total Completed voice gateway requests.",
            "# TYPE voice_gateway_requests_total counter",
        ]
        with self._lock:
            for (event, provider, status), count in sorted(self._counts.items()):
                labels = (
                    f'event="{event}",provider="{provider}",status="{status}"'
                )
                lines.append(f"voice_gateway_requests_total{{{labels}}} {count}")
            lines += [
                "# HELP voice_gateway_latency_milliseconds_sum Total request latency.",
                "# TYPE voice_gateway_latency_milliseconds_sum counter",
            ]
            for (event, provider), value in sorted(self._latency_sum.items()):
                labels = f'event="{event}",provider="{provider}"'
                lines.append(
                    f"voice_gateway_latency_milliseconds_sum{{{labels}}} {value}"
                )
        return "\n".join(lines) + "\n"
