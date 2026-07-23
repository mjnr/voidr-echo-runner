"""Audio conversation mode — Pipecat STT<->TTS pipeline structure.

Not exercised by the offline smoke: without provider keys this module fails
fast with a clear message. The service wiring below mirrors ARCHITECTURE.md
section 4 (Deepgram STT, ElevenLabs/Azure TTS, pluggable behind env vars).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

STT_ENV = ("DEEPGRAM_API_KEY",)
TTS_ENV = ("ELEVENLABS_API_KEY", "AZURE_SPEECH_KEY")


@dataclass(frozen=True)
class AudioServices:
    stt_provider: str
    tts_provider: str


def resolve_audio_services() -> AudioServices:
    """Validate env-var gated providers, failing with actionable guidance."""
    if not os.environ.get("DEEPGRAM_API_KEY"):
        raise RuntimeError(
            "Audio mode requires DEEPGRAM_API_KEY (STT, pt-BR streaming). "
            "None is set — run with --mode text for offline execution."
        )
    if os.environ.get("ELEVENLABS_API_KEY"):
        tts = "elevenlabs"
    elif os.environ.get("AZURE_SPEECH_KEY"):
        tts = "azure"
    else:
        raise RuntimeError(
            "Audio mode requires a TTS key: set ELEVENLABS_API_KEY (persona "
            "voices) or AZURE_SPEECH_KEY (volume). None is set — run with "
            "--mode text for offline execution."
        )
    return AudioServices(stt_provider="deepgram", tts_provider=tts)


def build_audio_pipeline(services: AudioServices):  # pragma: no cover — needs keys
    """Assemble the Pipecat pipeline (transport <-> STT <-> brain <-> TTS).

    TODO(echo/audio): with keys available, wire:
      - pipecat.services.deepgram.stt.DeepgramSTTService(language="pt-BR")
      - pipecat.services.elevenlabs.tts.ElevenLabsTTSService(voice_id=persona.speech.voiceId)
        or pipecat.services.azure.tts.AzureTTSService
      - a FrameProcessor bridging PersonaBrain replies into TTS frames
      - pipecat.pipeline.pipeline.Pipeline + PipelineRunner over the
        WebSocket/Twilio transport (TwilioFrameSerializer handles DTMF).
    """
    raise NotImplementedError(
        "Audio pipeline assembly lands with the first keyed environment; "
        "structure documented in this module and in the README."
    )
