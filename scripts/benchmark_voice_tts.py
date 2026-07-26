#!/usr/bin/env python3
"""Mockable LiteLLM voice benchmark: TTS TTFB, STT, ordering and barge-in."""

from __future__ import annotations

import argparse
import asyncio
from array import array
import json
import math
from pathlib import Path
import statistics
import time

import httpx

import voidr_echo_runner.voice_gateway as voice_module
from voidr_echo_runner.voice_gateway import (
    STT_ALIAS_DEFAULT,
    TTS_ALIAS_DEFAULT,
    VoiceConfig,
    VoiceGatewayAudioEngine,
)


class TimedPcmStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], delay_s: float = 0.008):
        self.chunks = chunks
        self.delay_s = delay_s

    async def __aiter__(self):
        for chunk in self.chunks:
            await asyncio.sleep(self.delay_s)
            yield chunk


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def mock_pcm() -> bytes:
    return array(
        "h",
        [round(4_000 * math.sin(2 * math.pi * 440 * index / 16_000)) for index in range(3200)],
    ).tobytes()


async def mock_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/audio/speech"):
        pcm = mock_pcm()
        return httpx.Response(
            200,
            stream=TimedPcmStream([pcm[index : index + 1600] for index in range(0, len(pcm), 1600)]),
        )
    body = await request.aread()
    first = b'filename="utterance-0.wav"' in body
    await asyncio.sleep(0.018 if first else 0.003)
    return httpx.Response(200, json={"text": "saldo disponível" if first else "disponível agora"})


async def run_once(engine: VoiceGatewayAudioEngine) -> dict[str, float | bool | str | int]:
    started = time.perf_counter()
    first_at = None
    chunks = 0
    async for _chunk in engine.synthesize_chunks("benchmark sintético"):
        chunks += 1
        first_at = first_at or time.perf_counter()
    tts_end = time.perf_counter()

    iterator = engine.synthesize_chunks("interrupção sintética")
    barge_started = time.perf_counter()
    await anext(iterator)
    await iterator.aclose()
    barge_end = time.perf_counter()

    original_segmenter = voice_module.segment_utterance
    voice_module.segment_utterance = lambda *_: [b"\x01\x00" * 320, b"\x02\x00" * 320]
    try:
        stt_started = time.perf_counter()
        transcript = await engine.transcribe(b"\x01\x00" * 640)
        stt_end = time.perf_counter()
    finally:
        voice_module.segment_utterance = original_segmenter
    return {
        "tts_ttfb_ms": round((first_at - started) * 1000, 3) if first_at else -1,
        "tts_total_ms": round((tts_end - started) * 1000, 3),
        "tts_chunks": chunks,
        "stt_utterance_ms": round((stt_end - stt_started) * 1000, 3),
        "ordered": transcript == "saldo disponível agora",
        "barge_in_cancel_ms": round((barge_end - barge_started) * 1000, 3),
        "transcript_shape": "three_words" if len(transcript.split()) == 3 else "unexpected",
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--baseline-json", type=Path)
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2")
    baseline = json.loads(args.baseline_json.read_text()) if args.baseline_json else None
    config = VoiceConfig(
        "test",
        "http://litellm.local",
        "mock-virtual-key",
        TTS_ALIAS_DEFAULT,
        STT_ALIAS_DEFAULT,
        False,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)) as client:
        engine = VoiceGatewayAudioEngine(config, "mock-voice", http_client=client)
        samples = [await run_once(engine) for _ in range(args.runs)]
    summary = {
        "mode": "mock-litellm-local-no-provider",
        "runs": args.runs,
        "p50_tts_ttfb_ms": percentile([sample["tts_ttfb_ms"] for sample in samples], 0.5),
        "p95_tts_ttfb_ms": percentile([sample["tts_ttfb_ms"] for sample in samples], 0.95),
        "p95_stt_utterance_ms": percentile(
            [sample["stt_utterance_ms"] for sample in samples], 0.95
        ),
        "p95_barge_in_cancel_ms": percentile(
            [sample["barge_in_cancel_ms"] for sample in samples], 0.95
        ),
        "mean_tts_chunks": round(statistics.mean(sample["tts_chunks"] for sample in samples), 2),
        "ordering_pass_rate": statistics.mean(bool(sample["ordered"]) for sample in samples),
    }
    regression = None
    if baseline:
        regression = {
            key: round(summary[key] - baseline[key], 3)
            for key in (
                "p95_tts_ttfb_ms",
                "p95_stt_utterance_ms",
                "p95_barge_in_cancel_ms",
            )
        }
    print(json.dumps({"summary": summary, "regression_ms": regression, "samples": samples}, indent=2))
    return 0 if summary["ordering_pass_rate"] == 1 and summary["mean_tts_chunks"] > 1 else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
