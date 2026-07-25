"""Telephone-channel simulation for the persona's TTS audio (EXEC-REALISM).

Grounding: τ-voice (Sierra, 2026) applies G.711 µ-law companding at 8 kHz,
environmental noise mixing and channel degradation to the *user simulator*
audio; EVA (ServiceNow) ships a perturbation suite of background noises and
connection degradation; the idiap acoustic-simulator and ITU G.712 define the
classic 300–3400 Hz telephony band-pass.

This module reproduces the audible signature of a phone call on the tester
(persona) channel only:

1. band-pass 300–3400 Hz (FFT brick-wall with raised-cosine skirts — the
   narrowband "phone voice");
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
TRANSITION_HZ = 60.0  # raised-cosine skirt width

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


def _bandpass(x: np.ndarray, sample_rate: int) -> np.ndarray:
    """300–3400 Hz FFT band-pass with raised-cosine transitions."""
    n = len(x)
    if n == 0:
        return x
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    gain = np.ones_like(freqs)
    # high-pass skirt at 300 Hz
    lo0, lo1 = TELEPHONE_LOW_HZ - TRANSITION_HZ, TELEPHONE_LOW_HZ
    gain = np.where(freqs < lo0, 0.0, gain)
    ramp = (freqs >= lo0) & (freqs < lo1)
    gain[ramp] = 0.5 - 0.5 * np.cos(np.pi * (freqs[ramp] - lo0) / TRANSITION_HZ)
    # low-pass skirt at 3400 Hz
    hi0, hi1 = TELEPHONE_HIGH_HZ, TELEPHONE_HIGH_HZ + TRANSITION_HZ
    ramp = (freqs > hi0) & (freqs <= hi1)
    gain[ramp] = 0.5 + 0.5 * np.cos(np.pi * (freqs[ramp] - hi0) / TRANSITION_HZ)
    gain = np.where(freqs > hi1, 0.0, gain)
    return np.fft.irfft(spectrum * gain, n=n)


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

    def _noise(self, n: int) -> np.ndarray:
        preset = _PRESETS[self.config.preset]
        amplitude = (10.0 ** (preset["noise_dbfs"] / 20.0)) * self.config.level
        if amplitude <= 0.0 or n == 0:
            self._phase += n
            return np.zeros(n)
        white = self._rng.standard_normal(n)
        # shape into the telephone band (noise outside it would be filtered
        # away by the far side anyway and just wastes headroom)
        noise = _bandpass(white, self.sample_rate)
        if preset["lowband"] > 0.0:
            # emphasize the low end (street rumble / office HVAC): 1/f-shaped
            # noise via FFT gain, fully vectorized and deterministic
            spectrum = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
            low = np.fft.irfft(spectrum / (1.0 + freqs / 150.0), n=n)
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
        rms = float(np.sqrt(np.mean(noise**2))) or 1.0
        return noise * (amplitude / rms)

    def process(self, pcm: bytes) -> bytes:
        """Apply the telephone channel to one PCM s16le mono utterance."""
        if not self.config.enabled or not pcm:
            return pcm
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64) / 32768.0
        y = _bandpass(x, self.sample_rate)
        y = _mu_law_roundtrip(y)
        y = y * 0.97 + self._noise(len(y))
        y = np.clip(y, -1.0, 1.0)
        return (y * 32767.0).astype(np.int16).tobytes()

    def comfort_noise(self, seconds: float) -> bytes:
        """Ambience-only PCM for silence gaps (the line is never truly dead)."""
        n = int(seconds * self.sample_rate)
        if not self.config.enabled or n <= 0:
            return b"\x00\x00" * max(0, n)
        y = np.clip(self._noise(n), -1.0, 1.0)
        return (y * 32767.0).astype(np.int16).tobytes()

    def record(self) -> dict:
        return {"preset": self.config.preset, "level": self.config.level}
