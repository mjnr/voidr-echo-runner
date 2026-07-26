"""Telephone-channel simulation for the persona's TTS audio (EXEC-REALISM).

Grounding: τ-voice (Sierra, 2026) applies G.711 µ-law companding at 8 kHz,
environmental noise mixing and channel degradation to the *user simulator*
audio; EVA (ServiceNow) ships a perturbation suite of background noises and
connection degradation; the idiap acoustic-simulator and ITU G.712 define the
classic 300–3400 Hz telephony band-pass.

This module reproduces the audible signature of a phone call on the tester
(persona) channel only:

1. causal stateful 300–3400 Hz band-pass (the narrowband "phone voice");
2. G.711-style µ-law companding round-trip at 8-bit (quantization grit);
3. seeded low-level background ambience (line hiss / home / office / street).

The agent's audio and the STT path are untouched. Deterministic: same seed +
same inputs ⇒ bit-identical output (fresh numpy Generator per call).

Config: `ECHO_CALL_AMBIENCE` = "none" | "<preset>" | "<preset>:<level 0..1>"
or JSON {"preset": "office", "level": 0.5}. Default in audio executions is
"quiet" (band-pass + µ-law + faint hiss) — subtle enough that Deepgram STT
on the far side keeps transcribing normally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

TELEPHONE_LOW_HZ = 300.0
TELEPHONE_HIGH_HZ = 3400.0
PCM_FRAME_MS = 20

# preset -> (noise dBFS at level=1.0, colors)
_PRESETS: dict[str, dict] = {
    # faint line hiss only — the "clean call" baseline
    "quiet": {"noise_dbfs": -48.0, "hum": 0.0, "lowband": 0.0},
    # residential: hiss + mains hum
    "home": {"noise_dbfs": -44.0, "hum": 1.0, "lowband": 0.3},
    # office: speech-band shaped noise (murmur-like)
    "office": {"noise_dbfs": -40.0, "hum": 0.3, "lowband": 0.6},
    # street: broadband, low-frequency heavy, slowly modulated
    "street": {"noise_dbfs": -36.0, "hum": 0.0, "lowband": 1.0},
}


@dataclass(frozen=True)
class AmbienceConfig:
    preset: str  # "none" disables everything
    level: float  # 0..1 scales the noise floor

    @property
    def enabled(self) -> bool:
        return self.preset != "none"


def parse_ambience(value: str | None, default: str = "quiet") -> AmbienceConfig:
    """Parse ECHO_CALL_AMBIENCE. Malformed values fall back to the default
    (a broken knob must never fail the call)."""
    raw = (value if value is not None else default).strip()
    if not raw:
        raw = default
    preset, level = raw, 1.0
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            preset = str(data.get("preset", default))
            level = float(data.get("level", 1.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            preset, level = default, 1.0
    elif ":" in raw:
        head, _, tail = raw.partition(":")
        preset = head
        try:
            level = float(tail)
        except ValueError:
            level = 1.0
    preset = preset.strip().lower()
    if preset not in _PRESETS and preset != "none":
        preset = default
    return AmbienceConfig(preset=preset, level=max(0.0, min(1.0, level)))


class _CausalBandpass:
    """Cascaded causal HP/LP with history, invariant to chunk boundaries."""

    def __init__(self, sample_rate: int):
        dt = 1.0 / sample_rate
        highpass_rc = 1.0 / (2.0 * np.pi * TELEPHONE_LOW_HZ)
        lowpass_rc = 1.0 / (2.0 * np.pi * TELEPHONE_HIGH_HZ)
        self._hp_alpha = highpass_rc / (highpass_rc + dt)
        self._lp_alpha = dt / (lowpass_rc + dt)
        self._x_prev = [0.0] * 4
        self._hp_prev = [0.0] * 4
        self._lp_prev = [0.0] * 4

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x, dtype=np.float64)
        for index, sample in enumerate(x):
            value = float(sample)
            for stage in range(4):
                stage_input = value
                value = self._hp_alpha * (
                    self._hp_prev[stage] + stage_input - self._x_prev[stage]
                )
                self._x_prev[stage] = stage_input
                self._hp_prev[stage] = value
            for stage in range(4):
                self._lp_prev[stage] += self._lp_alpha * (
                    value - self._lp_prev[stage]
                )
                value = self._lp_prev[stage]
            y[index] = value
        return y


class _CausalLowpass:
    def __init__(self, sample_rate: int, cutoff_hz: float):
        dt = 1.0 / sample_rate
        rc = 1.0 / (2.0 * np.pi * cutoff_hz)
        self._alpha = dt / (rc + dt)
        self._previous = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        result = np.empty_like(x, dtype=np.float64)
        previous = self._previous
        for index, sample in enumerate(x):
            previous += self._alpha * (float(sample) - previous)
            result[index] = previous
        self._previous = previous
        return result


def _mu_law_roundtrip(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """G.711-style µ-law compand → 8-bit quantize → expand (codec grit)."""
    compressed = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quantized = np.round(compressed * 127.0) / 127.0
    return np.sign(quantized) * (np.expm1(np.abs(quantized) * np.log1p(mu))) / mu


class TelephoneChannelFx:
    """Stateful per-call channel: seeded ambience is continuous across turns
    (the noise 'keeps running' between utterances, deterministically)."""

    def __init__(self, config: AmbienceConfig, seed: int, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        self._rng = np.random.default_rng(seed & 0xFFFFFFFF)
        self._phase = 0  # samples elapsed, keeps hum/modulation continuous
        self._signal_filter = _CausalBandpass(sample_rate)
        self._noise_filter = _CausalBandpass(sample_rate)
        self._rumble_filter = _CausalLowpass(sample_rate, 150.0)
        self._frame_samples = max(1, sample_rate * PCM_FRAME_MS // 1000)
        self._pcm_carry = b""
        self._sample_carry = np.empty(0, dtype=np.float64)

    def _noise(self, n: int) -> np.ndarray:
        preset = _PRESETS[self.config.preset]
        amplitude = (10.0 ** (preset["noise_dbfs"] / 20.0)) * self.config.level
        if amplitude <= 0.0 or n == 0:
            self._phase += n
            return np.zeros(n)
        white = self._rng.standard_normal(n)
        noise = self._noise_filter.process(white) * 1.8
        if preset["lowband"] > 0.0:
            low = self._rumble_filter.process(white)
            noise = noise * (1.0 - 0.5 * preset["lowband"]) + low * (2.5 * preset["lowband"])
        t = (np.arange(n) + self._phase) / self.sample_rate
        if preset["hum"] > 0.0:
            noise = noise + preset["hum"] * 0.35 * (
                np.sin(2 * np.pi * 60.0 * t) + 0.5 * np.sin(2 * np.pi * 120.0 * t)
            )
        if self.config.preset == "street":
            # slow amplitude modulation (traffic swell), period ~7 s
            noise = noise * (0.7 + 0.3 * np.sin(2 * np.pi * t / 7.0))
        self._phase += n
        return noise * amplitude

    def process(self, pcm: bytes) -> bytes:
        """Process one complete utterance, flushing its final partial frame."""
        return self.process_stream(pcm, final=True)

    def process_stream(self, pcm: bytes, *, final: bool = False) -> bytes:
        """Process fixed 20 ms PCM frames while carrying bytes/samples between chunks."""
        if not self.config.enabled:
            return pcm
        if not pcm and not final:
            return b""
        raw = self._pcm_carry + pcm
        aligned_size = len(raw) - (len(raw) % 2)
        self._pcm_carry = raw[aligned_size:]
        incoming = (
            np.frombuffer(raw[:aligned_size], dtype=np.int16).astype(np.float64)
            / 32768.0
        )
        samples = np.concatenate((self._sample_carry, incoming))
        process_count = (
            len(samples)
            if final
            else len(samples) - (len(samples) % self._frame_samples)
        )
        output: list[bytes] = []
        for offset in range(0, process_count, self._frame_samples):
            output.append(
                self._process_samples(samples[offset : min(offset + self._frame_samples, process_count)])
            )
        self._sample_carry = samples[process_count:].copy()
        if final:
            if self._pcm_carry:
                raise ValueError("PCM stream ended with a partial sample")
            self._sample_carry = np.empty(0, dtype=np.float64)
        return b"".join(output)

    def _process_samples(self, x: np.ndarray) -> bytes:
        y = self._signal_filter.process(x) * 1.35
        y = _mu_law_roundtrip(y)
        y = y * 0.97 + self._noise(len(y))
        y = np.clip(y, -1.0, 1.0)
        return (y * 32767.0).astype(np.int16).tobytes()

    def abort_stream(self) -> None:
        """Discard only un-emitted carry after a truncated provider stream."""
        self._pcm_carry = b""
        self._sample_carry = np.empty(0, dtype=np.float64)

    def comfort_noise(self, seconds: float) -> bytes:
        """Ambience-only PCM for silence gaps (the line is never truly dead)."""
        n = int(seconds * self.sample_rate)
        if not self.config.enabled or n <= 0:
            return b"\x00\x00" * max(0, n)
        y = np.clip(self._noise(n), -1.0, 1.0)
        return (y * 32767.0).astype(np.int16).tobytes()

    def record(self) -> dict:
        return {"preset": self.config.preset, "level": self.config.level}
