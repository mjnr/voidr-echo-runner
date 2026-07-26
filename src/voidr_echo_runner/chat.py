"""Persona playground: `echo-runner chat` — talk to a persona in the terminal.

You type as the Vivo agent; the persona answers in character via the hive LLM
gateway (LLMBrain — no LLM keys in this repo). With `--journey`, the keyword
state classifier tracks the journey state from what you type, keeping the
persona contextualized. With `--voice`, replies are synthesized with the
persona's ElevenLabs voice and played through the speaker (afplay).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from .brain import GENERIC_JOURNEY_STATE, HiveError, LLMBrain
from .models import Persona, load_persona_catalog
from .voice_gateway import VoiceGatewayAudioEngine, resolve_voice_config

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
  /state      mostra o estado atual da jornada
  /emotion    curva emocional da conversa (estado por turno)
  /help       esta ajuda
  /quit       encerra (Ctrl-D também)"""


class VoicePlayer:
    """LiteLLM TTS + afplay for the local persona playground."""

    def __init__(self, persona: Persona):
        self.voice_id = persona.speech.voiceId
        try:
            self.config = resolve_voice_config()
        except RuntimeError:
            self.config = None
        self.enabled = bool(self.config and not self.config.direct and self.voice_id)

    def speak(self, text: str) -> str | None:
        """Synthesize + play; returns a warning message on failure."""
        try:
            async def synthesize() -> bytes:
                engine = VoiceGatewayAudioEngine(self.config, self.voice_id)
                try:
                    return await engine.synthesize(text)
                finally:
                    await engine.aclose()

            pcm = asyncio.run(synthesize())
        except Exception as exc:  # playground degrades to text
            return f"voz indisponível ({type(exc).__name__})"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave.open(tmp, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(TTS_SAMPLE_RATE)
                wav.writeframes(pcm)
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
        brain = LLMBrain(persona, goal, args.seed)
    except RuntimeError as exc:
        print(f"echo-runner chat: {exc}", file=sys.stderr)
        return 2

    from .emotional import EmotionalStateMachine

    emotional = EmotionalStateMachine.for_persona(persona, seed=args.seed)
    brain.emotional = emotional

    voice = None
    if args.voice:
        voice = VoicePlayer(persona)
        if not voice.enabled:
            print(
                f"{YELLOW}aviso: --voice sem LiteLLM configurado (ou persona sem voiceId) "
                f"— seguindo só com texto{RESET}"
            )
            voice = None

    _print_header(persona, flow.id if flow else None, voice is not None)

    current_state = next(iter(flow.states)) if flow else "conversa"
    turns = 0
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
        if agent_text == "/emotion":
            if not emotional.history:
                print(f"{DIM}curva emocional: (ainda sem turnos){RESET}")
                continue
            print(f"{DIM}curva emocional ({emotional.badge()}):{RESET}")
            for rec in emotional.history:
                events = ", ".join(rec.events) or "sem gatilho (decay)"
                action = f"  {RED}<< {rec.action}{RESET}" if rec.action else ""
                print(
                    f"  {DIM}t{rec.turn:02d}{RESET} {rec.emotion} "
                    f"{rec.intensity:.2f} {rec.direction} ({rec.delta:+.2f})"
                    f" {DIM}{events}{RESET}{action}"
                )
            continue
        state_changed = False
        if classifier is not None and flow is not None:
            state = classifier.classify(agent_text)
            state_changed = state is not None and state != current_state
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

        # Appraise BEFORE the turn so the persona replies from the new state
        # (same order as CallRunner).
        emo_rec = emotional.update(agent_text, state_changed=state_changed)

        try:
            result = brain.take_turn(agent_text)
        except HiveError as exc:
            print(f"{RED}erro: {exc}{RESET}")
            continue
        turns += 1
        usage = result.get("usage") or {}
        model = result["provenance"]["model"]
        badge = f"{GREEN}{model}{RESET}"
        state_tag = f" · {brain.journey_state['currentState']}" if flow else ""
        emo_color = RED if emo_rec.intensity >= 0.6 else (YELLOW if emo_rec.intensity >= 0.3 else GREEN)
        emo_tag = f" · {emo_color}{emotional.badge()}{RESET}{DIM}"
        if emo_rec.action:
            emo_tag += f" {RED}<< {emo_rec.action}{RESET}{DIM}"
        print(f"{BOLD}{persona.id.split('-')[0]}>{RESET} {result['text']}")
        print(
            f"  {DIM}[{RESET}{badge}{DIM} · ${usage.get('costUsd', 0):.4f}"
            f"{state_tag}{emo_tag}]{RESET}"
        )
        if voice is not None:
            warning = voice.speak(result["text"])
            if warning:
                print(f"  {YELLOW}{warning}{RESET}")

    print(
        f"\n{BOLD}fim da conversa{RESET} — {turns} turno(s), "
        f"custo acumulado {GREEN}${brain.total_cost_usd:.4f}{RESET}, "
        f"emoção final {emotional.badge()}"
    )
    return 0
