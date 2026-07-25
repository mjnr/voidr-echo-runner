"""Lado-runner da prova E2E do media gateway — roda DENTRO do container.

Instancia o TwilioMediaStreamTransport real em modo gateway com um client
Twilio fake (REST não é o alvo da prova; o alvo é o caminho de mídia
WSS runner→gateway←Twilio). Publica o token da chamada no stdout para o
driver do host conectar o "Twilio simulado", espera uma fala, responde com
áudio e valida o encerramento.

Uso (ver scripts/prove_gateway_e2e.py):
    docker run --entrypoint python -e ECHO_MEDIA_GATEWAY_URL=... \
      -v .../gateway-proof-runner-side.py:/proof.py voidr-echo-runner:dev /proof.py
"""

import asyncio
import base64
import math
import struct

from voidr_echo_runner.twilio_transport import (
    PIPELINE_SAMPLE_RATE,
    TwilioMediaStreamTransport,
)


class FakeCall:
    sid = "CA_gateway_proof"


class FakeCallContext:
    def update(self, **kwargs):
        pass


class FakeCalls:
    def create(self, **kwargs):
        print(f"CALLS_CREATE twiml={kwargs.get('twiml')}", flush=True)
        return FakeCall()

    def __call__(self, sid):
        return FakeCallContext()


class FakeTwilioClient:
    calls = FakeCalls()


def tone_pcm(sample_rate: int, ms: int, freq: int = 440, amp: int = 12000) -> bytes:
    n = int(sample_rate * ms / 1000)
    return b"".join(
        struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / sample_rate)))
        for i in range(n)
    )


async def main() -> None:
    transport = TwilioMediaStreamTransport("+5511999990000", client=FakeTwilioClient())
    assert transport.gateway_mode, "prova exige ECHO_MEDIA_GATEWAY_URL"
    token = transport.public_url.rsplit("/", 1)[1]
    print(f"TOKEN={token}", flush=True)

    await transport.connect()
    print("GATEWAY_REGISTRADO", flush=True)

    msg = await transport.receive(timeout=30.0)
    assert msg["type"] == "audio", f"esperava fala do agente, veio {msg}"
    pcm = base64.b64decode(msg["data"])
    print(f"UTTERANCE_RECEBIDA bytes={len(pcm)} rate={msg['sample_rate']}", flush=True)

    await transport.send_audio(tone_pcm(PIPELINE_SAMPLE_RATE, 300))
    print("AUDIO_ENVIADO ms=300", flush=True)

    ended = await transport.receive(timeout=30.0)
    assert ended.get("name") == "call_ended", f"esperava call_ended, veio {ended}"
    print(f"CALL_ENDED reason={ended.get('reason')}", flush=True)
    await transport.hangup()
    print("PROVA_RUNNER_OK", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
