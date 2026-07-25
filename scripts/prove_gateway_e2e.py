"""Prova E2E local do media gateway: Twilio simulado → gateway → runner em Docker.

    Twilio simulado (este script, host)          runner (imagem Docker)
        │  wss /twilio/{token}                       │  ws /runner/{token}
        └──────────────► gateway real ◄──────────────┘
                      (subprocess, host)

Passos:
  1. sobe o echo-media-gateway real (subprocess, porta 18991, auth ligada);
  2. roda a imagem voidr-echo-runner:dev com o script-prova montado
     (transporte Twilio REAL em modo gateway, REST fake) e lê o token de
     chamada do stdout;
  3. conecta o "Twilio" em /twilio/{token}, manda start + 600ms de tom
     mu-law 8k + 1.2s de silêncio (fecha uma utterance no segmenter do
     runner), coleta os frames de mídia que o runner responde e manda stop;
  4. valida: runner recebeu a fala, Twilio-sim recebeu áudio de volta,
     runner viu call_ended e saiu com exit 0.

Uso: uv run python scripts/prove_gateway_e2e.py
Pré-requisito: docker build -t voidr-echo-runner:dev .
"""

import asyncio
import audioop
import base64
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import httpx
import websockets

REPO = Path(__file__).resolve().parents[1]
GATEWAY_PORT = 18991
AUTH_TOKEN = "prova-gateway-secret"
IMAGE = os.environ.get("ECHO_RUNNER_IMAGE", "voidr-echo-runner:dev")


def tone_pcm(sample_rate: int, ms: int, freq: int = 440, amp: int = 12000) -> bytes:
    n = int(sample_rate * ms / 1000)
    return b"".join(
        struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / sample_rate)))
        for i in range(n)
    )


def wait_gateway_up() -> None:
    for _ in range(50):
        try:
            if httpx.get(f"http://127.0.0.1:{GATEWAY_PORT}/healthz", timeout=1).status_code == 200:
                return
        except httpx.TransportError:
            time.sleep(0.2)
    raise RuntimeError("gateway não subiu")


async def simulated_twilio(token: str, received: list[dict]) -> None:
    # O runner publica o TOKEN e registra no gateway logo em seguida (dentro
    # do connect()) — retry curto cobre a janela entre o print e o registro.
    ws = None
    for _ in range(25):
        candidate = await websockets.connect(f"ws://127.0.0.1:{GATEWAY_PORT}/twilio/{token}")
        await asyncio.sleep(0.2)
        if candidate.protocol.close_code is None:
            ws = candidate
            break
    if ws is None:
        raise RuntimeError("gateway nunca aceitou o token da chamada (runner não registrou?)")
    await ws.send(json.dumps({"event": "connected"}))
    await ws.send(json.dumps({"event": "start", "start": {"streamSid": "MZ_prova_gw"}}))
    speech = audioop.lin2ulaw(tone_pcm(8000, 600), 2)
    silence = audioop.lin2ulaw(b"\x00" * int(8000 * 1.2) * 2, 2)
    for blob in (speech, silence):
        for i in range(0, len(blob), 160):  # frames de 20ms
            await ws.send(
                json.dumps(
                    {
                        "event": "media",
                        "media": {"payload": base64.b64encode(blob[i : i + 160]).decode()},
                    }
                )
            )
            await asyncio.sleep(0)  # não sufoca o loop
    # coleta o áudio que o runner responde
    try:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=8.0)
            msg = json.loads(raw)
            if msg.get("event") == "media":
                received.append(msg)
    except (TimeoutError, asyncio.TimeoutError):
        pass
    await ws.send(json.dumps({"event": "stop"}))
    await ws.close()


def main() -> int:
    print(f"▶ prova E2E do media gateway (imagem {IMAGE})")
    gateway = subprocess.Popen(
        ["uv", "--directory", str(REPO / "gateway"), "run", "echo-media-gateway"],
        env={
            **os.environ,
            "ECHO_MEDIA_GATEWAY_PORT": str(GATEWAY_PORT),
            "ECHO_MEDIA_GATEWAY_AUTH_TOKEN": AUTH_TOKEN,
        },
    )
    docker = None
    try:
        wait_gateway_up()
        print(f"  gateway de pé em :{GATEWAY_PORT} (auth ligada)")

        docker = subprocess.Popen(
            [
                "docker", "run", "--rm",
                "--entrypoint", "python",
                "-e", "TWILIO_ACCOUNT_SID=AC_prova",
                "-e", "TWILIO_AUTH_TOKEN=fake",
                "-e", "TWILIO_FROM_NUMBER=+15550000001",
                "-e", f"ECHO_MEDIA_GATEWAY_URL=ws://host.docker.internal:{GATEWAY_PORT}",
                "-e", f"ECHO_MEDIA_GATEWAY_TOKEN={AUTH_TOKEN}",
                "-v", f"{REPO / 'scripts' / 'gateway-proof-runner-side.py'}:/proof.py:ro",
                IMAGE,
                "/proof.py",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # O runner publica o TOKEN antes do connect(); connect() só retorna
        # quando o Twilio-sim manda o `start` — então o host prossegue já no
        # TOKEN (esperar mais seria deadlock).
        token = None
        runner_lines: list[str] = []
        assert docker.stdout is not None
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            line = docker.stdout.readline()
            if not line:
                break
            runner_lines.append(line.rstrip())
            print(f"  [runner] {line.rstrip()}")
            if line.startswith("TOKEN="):
                token = line.strip().split("=", 1)[1]
                break
        if not token:
            raise RuntimeError("runner não publicou o token da chamada")

        received: list[dict] = []
        asyncio.run(simulated_twilio(token, received))

        for line in docker.stdout:
            runner_lines.append(line.rstrip())
            print(f"  [runner] {line.rstrip()}")
        exit_code = docker.wait(timeout=30)

        out = "\n".join(runner_lines)
        assert "UTTERANCE_RECEBIDA" in out, "runner não recebeu a fala do Twilio-sim"
        assert "CALL_ENDED reason=completed" in out, "runner não viu o fim da chamada"
        assert "PROVA_RUNNER_OK" in out, "prova do lado runner não completou"
        assert exit_code == 0, f"container saiu com {exit_code}"
        assert received, "Twilio-sim não recebeu áudio de volta do runner"
        total = sum(len(base64.b64decode(m["media"]["payload"])) for m in received)
        assert all(m["streamSid"] == "MZ_prova_gw" for m in received)

        print(
            f"✅ PROVA OK — proxy bidirecional: runner recebeu 1 utterance; "
            f"Twilio-sim recebeu {len(received)} frames ({total} bytes mu-law) de volta; "
            f"call_ended propagado; container exit 0"
        )
        return 0
    finally:
        if docker is not None and docker.poll() is None:
            docker.kill()
        gateway.terminate()
        gateway.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
