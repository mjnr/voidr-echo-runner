"""EXEC-REALISM: telephone-channel DSP (band-pass + µ-law + seeded ambience)."""

from __future__ import annotations

import numpy as np
import pytest

from voidr_echo_runner.callfx import (
    AmbienceConfig,
    TelephoneChannelFx,
    parse_ambience,
)

SAMPLE_RATE = 16000


def tone_pcm(freq_hz: float, seconds: float = 0.5, amplitude: float = 0.5) -> bytes:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    x = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return (x * 32767.0).astype(np.int16).tobytes()


def rms(pcm: bytes) -> float:
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64) / 32768.0
    return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0


# --- parse_ambience -------------------------------------------------------------


def test_parse_ambience_defaults_to_quiet():
    assert parse_ambience(None) == AmbienceConfig(preset="quiet", level=1.0)
    assert parse_ambience("") == AmbienceConfig(preset="quiet", level=1.0)


def test_parse_ambience_none_disables():
    cfg = parse_ambience("none")
    assert not cfg.enabled


def test_parse_ambience_with_level_and_json():
    assert parse_ambience("office:0.5") == AmbienceConfig(preset="office", level=0.5)
    assert parse_ambience('{"preset": "street", "level": 0.3}') == AmbienceConfig(
        preset="street", level=0.3
    )


@pytest.mark.parametrize("raw", ["voo-espacial", '{"preset": 1'])
def test_parse_ambience_malformed_falls_back(raw):
    cfg = parse_ambience(raw)
    assert cfg.preset == "quiet"  # a broken knob must never fail the call


def test_parse_ambience_bad_level_keeps_preset():
    assert parse_ambience("office:abc") == AmbienceConfig(preset="office", level=1.0)


# --- band-pass / channel --------------------------------------------------------


def _fx(preset: str = "quiet", seed: int = 42) -> TelephoneChannelFx:
    return TelephoneChannelFx(
        AmbienceConfig(preset=preset, level=1.0), seed=seed, sample_rate=SAMPLE_RATE
    )


def test_process_kills_out_of_band_energy_keeps_speech_band():
    in_band = _fx().process(tone_pcm(1000))
    below = _fx().process(tone_pcm(120))
    above = _fx().process(tone_pcm(6000))
    assert rms(in_band) > 0.2  # speech band survives
    assert rms(below) < 0.02  # below 300 Hz: gone (only ambience floor left)
    assert rms(above) < 0.02  # above 3400 Hz: gone


def test_process_deterministic_per_seed():
    a = _fx(seed=7).process(tone_pcm(1000))
    b = _fx(seed=7).process(tone_pcm(1000))
    c = _fx(seed=8).process(tone_pcm(1000))
    assert a == b
    assert a != c  # ambience noise differs per seed


@pytest.mark.parametrize(
    "partitions",
    [
        [1, 7, 319, 640, 5, 1024],
        [640, 640, 640],
        [13, 29, 47, 83, 131],
    ],
)
def test_streaming_is_bit_exact_across_chunk_boundaries(partitions):
    pcm = tone_pcm(997, seconds=0.37)
    expected = _fx("office", seed=91).process(pcm)
    fx = _fx("office", seed=91)
    chunks = []
    offset = 0
    for size in partitions:
        if offset >= len(pcm):
            break
        chunks.append(fx.process_stream(pcm[offset : offset + size]))
        offset += size
    chunks.append(fx.process_stream(pcm[offset:]))
    chunks.append(fx.process_stream(b"", final=True))
    assert b"".join(chunks) == expected
    assert len(expected) == len(pcm)


def test_process_disabled_is_passthrough():
    fx = TelephoneChannelFx(
        AmbienceConfig(preset="none", level=1.0), seed=1, sample_rate=SAMPLE_RATE
    )
    pcm = tone_pcm(1000)
    assert fx.process(pcm) == pcm
    assert fx.process(b"") == b""


def test_mu_law_adds_quantization_but_keeps_signal_close():
    clean = tone_pcm(1000)
    out = _fx().process(clean)
    x = np.frombuffer(clean, dtype=np.int16).astype(np.float64)
    y = np.frombuffer(out, dtype=np.int16).astype(np.float64)
    # correlated (same tone) but not identical (companding + noise)
    corr = float(np.corrcoef(x, y)[0, 1])
    assert corr > 0.95
    assert not np.array_equal(x, y)


def test_ambience_presets_have_increasing_noise_floor():
    silence = b"\x00\x00" * SAMPLE_RATE  # 1 s of digital silence
    floors = {p: rms(_fx(p).process(silence)) for p in ("quiet", "home", "office", "street")}
    assert floors["quiet"] < floors["office"] < floors["street"]
    assert all(f > 0.0 for f in floors.values())  # the line is never truly dead
    assert floors["street"] < 0.1  # still background, not foreground


def test_comfort_noise_matches_duration_and_is_quiet():
    fx = _fx("quiet")
    pcm = fx.comfort_noise(0.5)
    assert len(pcm) == int(0.5 * SAMPLE_RATE) * 2  # s16le mono
    assert 0.0 < rms(pcm) < 0.02


def test_comfort_noise_disabled_returns_pure_silence():
    fx = TelephoneChannelFx(
        AmbienceConfig(preset="none", level=1.0), seed=1, sample_rate=SAMPLE_RATE
    )
    pcm = fx.comfort_noise(0.25)
    assert pcm == b"\x00\x00" * int(0.25 * SAMPLE_RATE)


def test_ambience_is_continuous_across_calls():
    fx = _fx("home")
    first = fx.comfort_noise(0.2)
    second = fx.comfort_noise(0.2)
    assert first != second  # phase advances — noise "keeps running"


def test_record_is_auditable():
    assert _fx("office").record() == {"preset": "office", "level": 1.0}
