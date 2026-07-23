"""PII redaction — audio layer: beep placement from word timestamps (STT mocked)."""

from __future__ import annotations

import math
import wave

import pytest

from voidr_echo_runner.audio import StereoCallRecorder
from voidr_echo_runner.audio_redaction import (
    BEEP_FREQ_HZ,
    Word,
    plan_beeps,
    redact_call_audio,
    save_redacted_wav,
)
from voidr_echo_runner.redaction import RedactionSession

RATE = 16000
CPF = "390.533.447-05"  # synthetic (valid check digits, doc example base)


def tone(freq: float, seconds: float, amplitude: int = 6000) -> bytes:
    n = int(RATE * seconds)
    return b"".join(
        int(amplitude * math.sin(2 * math.pi * freq * i / RATE)).to_bytes(2, "little", signed=True)
        for i in range(n)
    )


@pytest.fixture
def recorder():
    rec = StereoCallRecorder(sample_rate=RATE)
    rec.add("agent", tone(300, 2.0))  # segment 0: agent, no PII
    rec.add("tester", tone(200, 4.0))  # segment 1: tester speaks the CPF
    return rec


def words_with_cpf(pcm: bytes, sample_rate: int) -> list[Word]:
    """Fake batch STT: 'meu CPF é 390.533.447-05 tá' with known timings."""
    return [
        Word("meu", 0.10, 0.30),
        Word("CPF", 0.35, 0.60),
        Word("é", 0.65, 0.75),
        Word(CPF, 1.00, 2.80),
        Word("tá", 3.10, 3.30),
    ]


def test_beep_aligned_to_word_timestamps(recorder):
    session = RedactionSession()
    utterances = [(0, "agent", "bom dia, como posso ajudar?"), (1, "tester", f"meu CPF é {CPF} tá")]
    beeps = plan_beeps(recorder, utterances, session, words_fn=words_with_cpf)

    assert len(beeps) == 1
    beep = beeps[0]
    assert beep.channel == "tester"
    seg_start = recorder.segments[1][1]
    # word at 1.00-2.80s with 0.12s pad => [0.88s, 2.92s] into the segment
    assert beep.start_sample == seg_start + int(0.88 * RATE)
    assert beep.end_sample == seg_start + int(2.92 * RATE)


def test_beep_written_only_in_interval(recorder, tmp_path):
    session = RedactionSession()
    utterances = [(1, "tester", f"meu CPF é {CPF} tá")]
    beeps = plan_beeps(recorder, utterances, session, words_fn=words_with_cpf)
    path = tmp_path / "call.redacted.wav"
    save_redacted_wav(recorder, beeps, path)

    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 2
        frames = wav.readframes(wav.getnframes())
    left = [
        int.from_bytes(frames[i * 4 : i * 4 + 2], "little", signed=True)
        for i in range(len(frames) // 4)
    ]
    seg_start = recorder.segments[1][1]

    def dominant_is_beep(window: list[int]) -> bool:
        # crude 1kHz check: count sign changes => freq ~= changes/2 per second
        changes = sum(1 for a, b in zip(window, window[1:]) if (a >= 0) != (b >= 0))
        freq = changes / 2 / (len(window) / RATE)
        return abs(freq - BEEP_FREQ_HZ) < 60

    inside = left[seg_start + int(1.5 * RATE) : seg_start + int(1.6 * RATE)]
    before = left[seg_start + int(0.3 * RATE) : seg_start + int(0.4 * RATE)]
    assert dominant_is_beep(inside)
    assert not dominant_is_beep(before)  # 200 Hz source untouched outside the span


def test_fail_closed_beeps_whole_segment_on_stt_failure(recorder):
    session = RedactionSession()
    utterances = [(1, "tester", f"meu CPF é {CPF} tá")]

    def broken_stt(pcm: bytes, sample_rate: int) -> list[Word]:
        raise RuntimeError("deepgram down")

    beeps = plan_beeps(recorder, utterances, session, words_fn=broken_stt)
    channel, seg_start, seg_samples = recorder.segments[1]
    assert beeps == [type(beeps[0])("tester", seg_start, seg_start + seg_samples)]


def test_fail_closed_when_pii_not_relocated_in_batch_text(recorder):
    session = RedactionSession()
    utterances = [(1, "tester", f"meu CPF é {CPF} tá")]

    def stt_without_pii(pcm: bytes, sample_rate: int) -> list[Word]:
        return [Word("fala", 0.1, 0.5), Word("qualquer", 0.6, 1.0)]

    beeps = plan_beeps(recorder, utterances, session, words_fn=stt_without_pii)
    channel, seg_start, seg_samples = recorder.segments[1]
    assert beeps[0].start_sample == seg_start
    assert beeps[0].end_sample == seg_start + seg_samples


def test_no_pii_no_beep_no_stt_call(recorder):
    session = RedactionSession()
    calls = {"n": 0}

    def counting_stt(pcm: bytes, sample_rate: int) -> list[Word]:
        calls["n"] += 1
        return []

    beeps = plan_beeps(
        recorder,
        [(0, "agent", "bom dia!"), (1, "tester", "quero saber meu saldo")],
        session,
        words_fn=counting_stt,
    )
    assert beeps == []
    assert calls["n"] == 0  # STT only runs for PII-bearing segments


def test_redact_call_audio_discards_raw_by_default(recorder, tmp_path, monkeypatch):
    monkeypatch.delenv("ECHO_KEEP_RAW_AUDIO", raising=False)
    session = RedactionSession()
    meta = redact_call_audio(
        recorder,
        [(1, "tester", f"meu CPF é {CPF} tá")],
        session,
        tmp_path,
        words_fn=words_with_cpf,
    )
    assert (tmp_path / "call.redacted.wav").exists()
    assert not (tmp_path / "call.wav").exists()
    assert meta["rawWavKept"] is False
    assert meta["beeps"] == 1
    assert meta["beepedMs"] == pytest.approx(2040, abs=5)  # 2.92s - 0.88s


def test_redact_call_audio_keeps_raw_when_env_set(recorder, tmp_path, monkeypatch):
    monkeypatch.setenv("ECHO_KEEP_RAW_AUDIO", "1")
    session = RedactionSession()
    meta = redact_call_audio(
        recorder, [(1, "tester", f"meu CPF é {CPF} tá")], session, tmp_path,
        words_fn=words_with_cpf,
    )
    assert (tmp_path / "call.wav").exists()
    assert meta["rawWavKept"] is True
