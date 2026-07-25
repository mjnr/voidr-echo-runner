"""voidr echo media gateway — WSS público estável para o Twilio Media Streams.

Problema: o Media Stream do Twilio precisa ALCANÇAR o processo do runner
(servidor WS de mídia). No dev local isso é um tunnel por máquina
(TWILIO_STREAM_PUBLIC_URL → porta 8990); na nuvem o runner é um pod efêmero
atrás de NAT, sem endereço público. Um LoadBalancer por job seria caro e
lento de provisionar (minutos vs. segundos de setup de chamada).

Desenho: UM Deployment estável com endpoint WSS público, pareando as duas
pontas por um token de chamada:

    runner (pod)  ── ws outbound ──▶  /runner/{token}   (Bearer auth)
    Twilio        ── wss inbound ──▶  /twilio/{token}   (capability URL)

1. O runner gera um token aleatório por chamada (128 bits), conecta OUTBOUND
   em `/runner/{token}` autenticando com `Authorization: Bearer
   $ECHO_MEDIA_GATEWAY_TOKEN`, e só então cria a chamada Twilio com TwiML
   `<Stream url="wss://{gateway}/twilio/{token}"/>`.
2. O Twilio conecta em `/twilio/{token}`. O gateway pareia as duas conexões
   e faz proxy CEGO e bidirecional dos frames (JSON do Media Streams passa
   intacto — zero parsing, zero estado de mídia).
3. DTMF mid-call (update de TwiML + re-<Connect><Stream>): o lado Twilio cai
   e RECONECTA com o mesmo token; a conexão do runner permanece e a nova
   conexão Twilio assume o slot. A sessão morre quando o runner desconecta.

Segurança: o caminho do runner exige o shared secret; o caminho do Twilio
não tem como carregar auth (o Twilio não envia headers custom em Media
Streams), então a segurança é o token por chamada não-adivinhável + TTL de
sessão amarrado à conexão do runner. Frames de mídia nunca são logados.

Env vars:
    ECHO_MEDIA_GATEWAY_PORT         porta de escuta (default 8991)
    ECHO_MEDIA_GATEWAY_AUTH_TOKEN   shared secret exigido em /runner/{token}
                                    (sem ele o gateway roda ABERTO — só dev)
"""

from __future__ import annotations

import asyncio
import http
import os
import signal
import sys
import time
from dataclasses import dataclass, field

import websockets
from websockets.asyncio.server import ServerConnection, serve

CLOSE_UNAUTHORIZED = 4401
CLOSE_UNKNOWN_TOKEN = 4404
CLOSE_RUNNER_GONE = 4410
CLOSE_DUPLICATE_RUNNER = 4409


def _log(msg: str) -> None:
    print(f"[echo-media-gateway] {msg}", flush=True)


@dataclass
class CallSession:
    """Uma chamada em curso: a conexão persistente do runner + o slot Twilio."""

    token: str
    runner: ServerConnection
    twilio: ServerConnection | None = None
    created_at: float = field(default_factory=time.monotonic)
    twilio_attach_count: int = 0


class MediaGateway:
    def __init__(self, auth_token: str | None):
        self.auth_token = auth_token
        self.sessions: dict[str, CallSession] = {}

    # -- roteamento -----------------------------------------------------------

    async def handle(self, ws: ServerConnection) -> None:
        path = (ws.request.path if ws.request else "") or ""
        parts = [p for p in path.split("?")[0].split("/") if p]
        if len(parts) != 2 or parts[0] not in ("runner", "twilio"):
            await ws.close(CLOSE_UNKNOWN_TOKEN, "unknown path")
            return
        role, token = parts
        if role == "runner":
            await self._handle_runner(ws, token)
        else:
            await self._handle_twilio(ws, token)

    # -- lado runner (registro outbound do pod) --------------------------------

    def _runner_authorized(self, ws: ServerConnection) -> bool:
        if not self.auth_token:
            return True  # dev/local — produção SEMPRE define o token
        header = (ws.request.headers.get("Authorization", "") if ws.request else "").strip()
        return header == f"Bearer {self.auth_token}"

    async def _handle_runner(self, ws: ServerConnection, token: str) -> None:
        if not self._runner_authorized(ws):
            _log(f"runner rejeitado (auth) token={token[:8]}…")
            await ws.close(CLOSE_UNAUTHORIZED, "bad gateway token")
            return
        if token in self.sessions:
            _log(f"runner duplicado token={token[:8]}…")
            await ws.close(CLOSE_DUPLICATE_RUNNER, "token already registered")
            return
        session = CallSession(token=token, runner=ws)
        self.sessions[token] = session
        _log(f"runner registrado token={token[:8]}… (sessões ativas: {len(self.sessions)})")
        try:
            # Frames runner → Twilio (áudio da persona, marks, clear...).
            # Sem Twilio conectado (setup da chamada / janela de reconexão do
            # DTMF) o frame é descartado — mesma semântica do servidor local
            # quando o stream caiu.
            async for message in ws:
                twilio = session.twilio
                if twilio is not None:
                    try:
                        await twilio.send(message)
                    except websockets.ConnectionClosed:
                        session.twilio = None
        except websockets.ConnectionClosed:
            pass
        finally:
            self.sessions.pop(token, None)
            twilio = session.twilio
            if twilio is not None:
                # Runner sumiu (fim de chamada ou crash do pod) — derruba o
                # lado Twilio para a operadora encerrar o stream.
                await _close_quietly(twilio, CLOSE_RUNNER_GONE, "runner disconnected")
            _log(f"runner saiu token={token[:8]}… (sessões ativas: {len(self.sessions)})")

    # -- lado Twilio (Media Streams) -------------------------------------------

    async def _handle_twilio(self, ws: ServerConnection, token: str) -> None:
        session = self.sessions.get(token)
        if session is None:
            _log(f"twilio com token desconhecido token={token[:8]}…")
            await ws.close(CLOSE_UNKNOWN_TOKEN, "unknown call token")
            return
        previous = session.twilio
        if previous is not None:
            # Reconexão (DTMF TwiML update): a conexão nova assume o slot; a
            # antiga normalmente já está fechando do lado Twilio.
            await _close_quietly(previous, 1000, "replaced by reconnect")
        session.twilio = ws
        session.twilio_attach_count += 1
        _log(f"twilio conectado token={token[:8]}… (attach #{session.twilio_attach_count})")
        try:
            # Frames Twilio → runner (start/media/stop do Media Streams).
            async for message in ws:
                try:
                    await session.runner.send(message)
                except websockets.ConnectionClosed:
                    break
        except websockets.ConnectionClosed:
            pass
        finally:
            if session.twilio is ws:
                session.twilio = None
            _log(f"twilio saiu token={token[:8]}…")


async def _close_quietly(ws: ServerConnection, code: int, reason: str) -> None:
    try:
        await ws.close(code, reason)
    except Exception:  # noqa: BLE001 — conexão já podre não pode travar o teardown
        pass


def _health_check(connection: ServerConnection, request):
    if request.path in ("/healthz", "/health"):
        return connection.respond(http.HTTPStatus.OK, "ok\n")
    return None


async def run_gateway(
    host: str = "0.0.0.0",
    port: int = 8991,
    auth_token: str | None = None,
    *,
    ready: asyncio.Event | None = None,
) -> None:
    gateway = MediaGateway(auth_token)
    async with serve(
        gateway.handle,
        host,
        port,
        process_request=_health_check,
        # Media Streams manda frames de 20ms; ping padrão do websockets segura
        # NATs/LBs intermediários. max_size default (1 MiB) é folgado para
        # frames de mídia (~400 bytes) e barra payloads abusivos.
    ) as server:
        _log(
            f"escutando em {host}:{port} "
            f"(auth {'LIGADA' if auth_token else 'DESLIGADA — só dev'})"
        )
        if ready is not None:
            ready.set()
        await server.serve_forever()


def main() -> None:
    port = int(os.environ.get("ECHO_MEDIA_GATEWAY_PORT", "8991"))
    auth_token = os.environ.get("ECHO_MEDIA_GATEWAY_AUTH_TOKEN") or None
    if not auth_token:
        print(
            "AVISO: ECHO_MEDIA_GATEWAY_AUTH_TOKEN não definido — o registro de "
            "runners está SEM auth (aceitável só em dev local)",
            file=sys.stderr,
        )

    async def _serve() -> None:
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        task = asyncio.create_task(run_gateway(port=port, auth_token=auth_token))
        await stop.wait()
        task.cancel()

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
