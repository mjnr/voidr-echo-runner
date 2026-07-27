"""Load runner-scoped credentials from read-only CSI files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping

PROJECTED_SECRET_NAMES = (
    "DEEPGRAM_API_KEY",
    "ELEVENLABS_API_KEY",
    "HIVE_GATEWAY_TOKEN",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "VOICE_GATEWAY_TOKEN",
    "ECHO_MEDIA_GATEWAY_TOKEN",
)
DEFAULT_SECRET_DIR = Path("/var/run/secrets/voidr")


def load_projected_secrets(
    environ: MutableMapping[str, str] | None = None,
    secret_dir: Path | None = None,
) -> list[str]:
    env = os.environ if environ is None else environ
    root = secret_dir or Path(env.get("VOIDR_SECRET_DIR", DEFAULT_SECRET_DIR))
    loaded: list[str] = []
    for name in PROJECTED_SECRET_NAMES:
        if env.get(name):
            continue
        path = Path(env.get(f"{name}_FILE", root / name))
        if not path.is_file():
            continue
        value = path.read_text(encoding="utf-8").strip()
        if value:
            env[name] = value
            loaded.append(name)
    return loaded
