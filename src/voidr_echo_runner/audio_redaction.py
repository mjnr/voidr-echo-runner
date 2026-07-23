"""PII redaction (ARCHITECTURE.md section 10) — audio layer.

Maps the PII text spans found by `redaction.RedactionSession` back to WAV
intervals and overwrites them with a 1 kHz beep, producing
`out/<run-id>/call.redacted.wav`. The raw `call.wav` is only kept when
`ECHO_KEEP_RAW_AUDIO=1` (default: discarded after redaction).

Word alignment: the call recording is turn-based (each utterance is a known
segment of the stereo WAV), so only segments whose text contains PII are
re-transcribed via the Deepgram prerecorded API (word-level timestamps come
in the response). This runs POST-call: zero latency added to the live
conversation, one REST call per PII-bearing segment.

Fail-closed by design: if word timestamps cannot be fetched or the PII cannot
be re-located in the batch transcript, the ENTIRE segment is beeped — leaking
massa is unacceptable, an over-long beep is not.
"""

from __future__ import annotations

import io
import math
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .redaction import RedactionSession

BEEP_FREQ_HZ = 1000
BEEP_AMPLITUDE = 8000  # ~25% full scale: audible, not aggressive
WORD_PAD_S = 0.12  # padding around matched words
DEEPGRAM_MODEL = "nova-2"


@dataclass(frozen=True)
class Word:
    text: str
    start: float  # seconds, relative to the segment
    end: float


@dataclass(frozen=True)
class Beep:
    channel: str  # tester | agent
    start_sample: int
    end_sample: int


class RecorderLike(Protocol):
    sample_rate: int
    segments: list[tuple[str, int, int]]

    def segment_pcm(self, index: int) -> bytes: ...
    def channel_copies(self) -> tuple[bytearray, bytearray]: ...


WordsFn = Callable[[bytes, int], list[Word]]


def deepgram_words(pcm: bytes, sample_rate: int) -> list[Word]:
    """Word-level timestamps from the Deepgram prerecorded API (pt-BR)."""
    import httpx

    response = httpx.post(
        "https://api.deepgram.com/v1/listen",
        params={
            "model": DEEPGRAM_MODEL,
            "language": "pt-BR",
            "smart_format": "true",
            "punctuate": "true",
        },
        headers={
            "Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}",
            "Content-Type": "audio/wav",
        },
        content=_pcm_to_wav(pcm, sample_rate),
        timeout=30.0,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Deepgram prerecorded failed ({response.status_code})")
    alternatives = response.json()["results"]["channels"][0]["alternatives"]
    return [
        Word(
            text=w.get("punctuated_word") or w["word"],
            start=float(w["start"]),
            end=float(w["end"]),
        )
        for w in (alternatives[0].get("words") or [])
    ]


def plan_beeps(
    recorder: RecorderLike,
    utterances: list[tuple[int, str, str]],
    session: RedactionSession,
    words_fn: WordsFn = deepgram_words,
) -> list[Beep]:
    """Compute beep intervals for every utterance whose text contains PII."""
    rate = recorder.sample_rate
    beeps: list[Beep] = []
    for segment_index, _speaker, text in utterances:
        if not session.find_spans(text):
            continue
        channel, seg_start, seg_samples = recorder.segments[segment_index]
        intervals = _pii_word_intervals(
            recorder.segment_pcm(segment_index), rate, session, words_fn
        )
        if intervals is None:
            # Fail-closed: no reliable word alignment -> beep the whole turn.
            beeps.append(Beep(channel, seg_start, seg_start + seg_samples))
            continue
        for start_s, end_s in intervals:
            beeps.append(
                Beep(
                    channel,
                    seg_start + max(0, int((start_s - WORD_PAD_S) * rate)),
                    seg_start + min(seg_samples, int((end_s + WORD_PAD_S) * rate)),
                )
            )
    return beeps


def _pii_word_intervals(
    pcm: bytes,
    sample_rate: int,
    session: RedactionSession,
    words_fn: WordsFn,
) -> list[tuple[float, float]] | None:
    """PII time intervals (seconds) inside one segment, or None if alignment
    is not trustworthy (caller falls back to whole-segment beep)."""
    try:
        words = words_fn(pcm, sample_rate)
    except Exception:  # noqa: BLE001 — any STT failure => fail-closed
        return None
    if not words:
        return None

    batch_text = " ".join(w.text for w in words)
    offsets: list[tuple[int, int]] = []  # char span of each word in batch_text
    cursor = 0
    for w in words:
        offsets.append((cursor, cursor + len(w.text)))
        cursor += len(w.text) + 1

    spans = session.find_spans(batch_text)
    if not spans:
        # PII was detected in the turn text but not re-located in the batch
        # transcript (different number formatting, STT drift, etc.).
        return None

    intervals: list[tuple[float, float]] = []
    for span in spans:
        hit = [
            words[i]
            for i, (w_start, w_end) in enumerate(offsets)
            if w_start < span.end and w_end > span.start
        ]
        if hit:
            intervals.append((hit[0].start, hit[-1].end))
    return _merge_intervals(intervals) if intervals else None


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + WORD_PAD_S:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def save_redacted_wav(recorder: RecorderLike, beeps: list[Beep], path: Path) -> None:
    """Write the stereo WAV with beeps applied (L=tester, R=agent)."""
    left, right = recorder.channel_copies()
    rate = recorder.sample_rate
    for beep in beeps:
        buf = left if beep.channel == "tester" else right
        _write_beep(buf, beep.start_sample, beep.end_sample, rate)
    frames = bytearray()
    for i in range(0, len(left), 2):
        frames.extend(left[i : i + 2])
        frames.extend(right[i : i + 2])
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(frames))


def _write_beep(buf: bytearray, start_sample: int, end_sample: int, rate: int) -> None:
    end_sample = min(end_sample, len(buf) // 2)
    for i in range(max(0, start_sample), end_sample):
        value = int(BEEP_AMPLITUDE * math.sin(2 * math.pi * BEEP_FREQ_HZ * i / rate))
        buf[i * 2 : i * 2 + 2] = value.to_bytes(2, "little", signed=True)


def redact_call_audio(
    recorder: RecorderLike,
    utterances: list[tuple[int, str, str]],
    session: RedactionSession,
    out_dir: Path,
    words_fn: WordsFn = deepgram_words,
) -> dict[str, Any]:
    """Full audio-redaction step. Returns metadata for the artifacts.

    Writes `call.redacted.wav`; writes the raw `call.wav` ONLY when
    ECHO_KEEP_RAW_AUDIO=1.
    """
    beeps = plan_beeps(recorder, utterances, session, words_fn)
    redacted_path = out_dir / "call.redacted.wav"
    save_redacted_wav(recorder, beeps, redacted_path)
    keep_raw = os.environ.get("ECHO_KEEP_RAW_AUDIO") == "1"
    if keep_raw:
        recorder.save(out_dir / "call.wav")  # type: ignore[attr-defined]
    return {
        "redactedWavFile": "call.redacted.wav",
        "rawWavKept": keep_raw,
        "beeps": len(beeps),
        "beepedMs": int(
            sum((b.end_sample - b.start_sample) for b in beeps) / recorder.sample_rate * 1000
        ),
    }


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()
