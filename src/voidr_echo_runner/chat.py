"""Persona playground: `echo-runner chat` — talk to a persona in the terminal.

You type as the Vivo agent; the persona answers in character via the hive LLM
gateway (LLMBrain — no LLM keys in this repo). With `--journey`, the keyword
state classifier tracks the journey state from what you type, keeping the
persona contextualized. With `--voice`, replies are synthesized with the
persona's ElevenLabs voice and played through the speaker (afplay).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from .brain import GENERIC_JOURNEY_STATE, HiveError, LLMBrain
from .models import Persona, load_persona_catalog

ELEVENLABS_MODEL = "eleven_flash_v2_5"
TTS_SAMPLE_RATE = 16000

# ANSI (degrade to plain when not a tty)
_TTY = sys.stdout.isatty()
BOLD = "\033[1m" if _TTY else ""
DIM = "\033[2m" if _TTY else ""
CYAN = "\033[36m" if _TTY else ""
YELLOW = "\033[33m" if _TTY else ""
GREEN = "\033[32m" if _TTY else ""
RED = "\033[31m" if _TTY else ""
RESET = "\033[0m" if _TTY else ""

HELP = """comandos:
  /escalate   força Sonnet no próximo turno
  /state      mostra o estado atual da jornada
  /help       esta ajuda
  /quit       encerra (Ctrl-D também)"""


class VoicePlayer:
    """ElevenLabs TTS + afplay. Direct REST on purpose: the playground favors
    snappiness; the full Pipecat pipeline lives in the tested audio mode."""

    def __init__(self, persona: Persona):
        self.voice_id = persona.speech.voiceId
        self.api_key = os.environ.get("ELEVENLABS_API_KEY", "")
        self.enabled = bool(self.api_key and self.voice_id)

    def speak(self, text: str) -> str | None:
        """Synthesize + play; returns a warning message on failure."""
        import httpx

        try:
            response = httpx.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}",
                params={"output_format": f"pcm_{TTS_SAMPLE_RATE}"},
                headers={"xi-api-key": self.api_key},
                json={"text": text, "model_id": ELEVENLABS_MODEL, "language_code": "pt"},
                timeout=30.0,
            )
        except httpx.TransportError as exc:
            return f"voz indisponível ({type(exc).__name__})"
        if response.status_code != 200:
            return f"voz indisponível (ElevenLabs {response.status_code})"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave.open(tmp, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(TTS_SAMPLE_RATE)
                wav.writeframes(response.content)
            path = tmp.name
        try:
            subprocess.run(["afplay", path], check=False)
        finally:
            os.unlink(path)
        return None


def _print_header(persona: Persona, journey_slug: str | None, voice_on: bool) -> None:
    d, t = persona.demographics, persona.temperament
    identity = " · ".join(
        part
        for part in (
            persona.name,
            f"{persona.age} anos" if persona.age is not None else f"{d.ageBand} anos",
            persona.gender,
        )
        if part
    )
    lines = [
        f"{BOLD}{persona.id}{RESET}  {DIM}v{persona.version} · {persona.kind}{RESET}",
        f"{identity} · {d.region} · humor {t.mood} · paciência {t.patienceLevel}/5 "
        f"· tech {t.techSavviness} · {t.verbosity}",
        f"jornada: {journey_slug or 'conversa livre'} · voz: "
        + (f"{GREEN}on{RESET} ({persona.speech.voiceId})" if voice_on else f"{DIM}off{RESET}"),
        f"{DIM}você é o agente da Vivo — digite e a persona responde. /help para comandos{RESET}",
    ]
    width = max(len(_strip_ansi(line)) for line in lines) + 2
    print("┌" + "─" * width + "┐")
    for line in lines:
        pad = width - len(_strip_ansi(line)) - 1
        print("│ " + line + " " * pad + "│")
    print("└" + "─" * width + "┘")


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\033\[[0-9;]*m", "", text)


def run_chat(args) -> int:
    catalog = load_persona_catalog(args.personas)
    if args.persona not in catalog:
        print(
            f"persona {args.persona!r} não encontrada em {args.personas} "
            f"(disponíveis: {', '.join(sorted(catalog))})",
            file=sys.stderr,
        )
        return 2
    persona = catalog[args.persona]

    flow = classifier = None
    if args.journey:
        from .classifier import KeywordStateClassifier
        from .flows import load_journey_flow

        flow = load_journey_flow(Path(args.journey))
        classifier = KeywordStateClassifier(flow)

    goal = args.goal or "resolver um problema na minha linha Vivo"
    try:
        brain = LLMBrain(persona, goal, args.seed, escalate=args.escalate)
    except RuntimeError as exc:
        print(f"echo-runner chat: {exc}", file=sys.stderr)
        return 2

    voice = None
    if args.voice:
        voice = VoicePlayer(persona)
        if not voice.enabled:
            print(
                f"{YELLOW}aviso: --voice sem ELEVENLABS_API_KEY (ou persona sem voiceId) "
                f"— seguindo só com texto{RESET}"
            )
            voice = None

    _print_header(persona, flow.id if flow else None, voice is not None)

    current_state = next(iter(flow.states)) if flow else "conversa"
    turns = 0
    force_escalate = False
    while True:
        try:
            agent_text = input(f"{CYAN}{BOLD}agente>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not agent_text:
            continue
        if agent_text in ("/quit", "/exit"):
            break
        if agent_text == "/help":
            print(DIM + HELP + RESET)
            continue
        if agent_text == "/state":
            print(f"{DIM}journeyState: {brain.journey_state}{RESET}")
            continue
        if agent_text == "/escalate":
            force_escalate = True
            print(f"{YELLOW}próximo turno será escalado (Sonnet){RESET}")
            continue

        if classifier is not None and flow is not None:
            state = classifier.classify(agent_text)
            if state is not None:
                current_state = state
            state_def = flow.states.get(current_state)
            brain.journey_state = {
                "flowSlug": flow.id,
                "currentState": current_state,
                "expects": list(state_def.expects) if state_def else [],
            }
        else:
            brain.journey_state = dict(GENERIC_JOURNEY_STATE)

        try:
            result = brain.take_turn(agent_text, escalate=force_escalate or None)
        except HiveError as exc:
            print(f"{RED}erro: {exc}{RESET}")
            continue
        finally:
            force_escalate = False
        turns += 1
        usage = result.get("usage") or {}
        model = result.get("model", "?")
        badge = f"{YELLOW}sonnet{RESET}" if usage.get("escalated") else f"{GREEN}{model}{RESET}"
        state_tag = f" · {brain.journey_state['currentState']}" if flow else ""
        print(f"{BOLD}{persona.id.split('-')[0]}>{RESET} {result['text']}")
        print(f"  {DIM}[{RESET}{badge}{DIM} · ${usage.get('costUsd', 0):.4f}{state_tag}]{RESET}")
        if voice is not None:
            warning = voice.speak(result["text"])
            if warning:
                print(f"  {YELLOW}{warning}{RESET}")

    print(
        f"\n{BOLD}fim da conversa{RESET} — {turns} turno(s), "
        f"custo acumulado {GREEN}${brain.total_cost_usd:.4f}{RESET}"
    )
    return 0
