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
import contextlib
import hmac
import http
import json
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

import httpx
import websockets
from websockets.asyncio.server import ServerConnection, serve

from .auth import (
    CapabilityError,
    CapabilityVerifier,
    RateLimitExceeded,
    ReplayGuard,
    VoiceRateLimiter,
)
from .observability import VoiceObservability
from .providers import ProviderConfigurationError, VoiceProviders
from .secrets import load_gateway_secrets
from .state import (
    MemoryVoiceStateBackend,
    RedisVoiceStateBackend,
    StateBackendUnavailable,
    VoiceStateBackend,
    unpack_relay,
)

CLOSE_UNAUTHORIZED = 4401
CLOSE_UNKNOWN_TOKEN = 4404
CLOSE_RUNNER_GONE = 4410
CLOSE_DUPLICATE_RUNNER = 4409
CLOSE_SCOPE_DENIED = 4403
CLOSE_RATE_LIMITED = 4429
CLOSE_BAD_REQUEST = 4400
CLOSE_UPSTREAM_ERROR = 4502
MAX_STT_AUDIO_BYTES = 16_000 * 2 * 120
MAX_STT_CHUNKS = 2_048
MAX_STT_UPSTREAM_BYTES = 2**20
MAX_TTS_AUDIO_BYTES = 16_000 * 2 * 120
MAX_TTS_CHUNKS = 2_048
STT_DEADLINE_S = 150
TTS_DEADLINE_S = 45
CLIENT_REQUEST_TIMEOUT_S = 10
RELAY_LEASE_TTL_MS = 15_000
RELAY_RENEW_INTERVAL_S = 5.0


class ClientReadTimeout(TimeoutError):
    pass


class UpstreamDeadlineExceeded(TimeoutError):
    pass


class RelaySuperseded(RuntimeError):
    pass


@dataclass
class ReadinessState:
    max_age_s: float = 15.0
    last_redis_success: float | None = None
    redis_error: str | None = None
    providers_ready: bool = True
    configuration_error: str | None = None

    def mark_success(self) -> None:
        self.last_redis_success = time.monotonic()
        self.redis_error = None

    def mark_failure(self, exc: BaseException) -> None:
        self.redis_error = type(exc).__name__

    def set_provider_configuration(self, ready: bool, error: str | None = None) -> None:
        self.providers_ready = ready
        self.configuration_error = error

    def ready(self) -> bool:
        return (
            self.providers_ready
            and self.configuration_error is None
            and self.last_redis_success is not None
            and time.monotonic() - self.last_redis_success <= self.max_age_s
            and self.redis_error is None
        )


def _log(msg: str) -> None:
    print(f"[echo-media-gateway] {msg}", flush=True)


class MediaGateway:
    def __init__(
        self,
        auth_token: str | None,
        *,
        verifier: CapabilityVerifier | None = None,
        providers: VoiceProviders | None = None,
        limiter: VoiceRateLimiter | None = None,
        observability: VoiceObservability | None = None,
        state_backend: VoiceStateBackend | None = None,
        require_tls: bool = False,
        allow_insecure_runner_auth: bool = False,
    ):
        self.auth_token = auth_token
        self.require_tls = require_tls
        self.allow_insecure_runner_auth = allow_insecure_runner_auth
        self.verifier = verifier
        self.providers = providers or VoiceProviders()
        self.limiter = limiter or VoiceRateLimiter()
        self.observability = observability or VoiceObservability(
            lambda line: _log(f"audit={line}")
        )
        self.state_backend = state_backend or MemoryVoiceStateBackend()

    # -- roteamento -----------------------------------------------------------

    async def handle(self, ws: ServerConnection) -> None:
        if self.require_tls and not self._secure_request(ws):
            await ws.close(CLOSE_SCOPE_DENIED, "tls_required")
            return
        path = (ws.request.path if ws.request else "") or ""
        parsed = urlsplit(path)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) == 3 and parts[:2] == ["v1", "stt"]:
            await self._handle_stt(ws, parts[2], parse_qs(parsed.query))
            return
        if len(parts) == 3 and parts[:2] == ["v1", "tts"]:
            await self._handle_tts(ws, parts[2], parse_qs(parsed.query))
            return
        if len(parts) != 2 or parts[0] not in ("runner", "twilio"):
            await ws.close(CLOSE_UNKNOWN_TOKEN, "unknown path")
            return
        role, token = parts
        if role == "runner":
            await self._handle_runner(ws, token)
        else:
            await self._handle_twilio(ws, token)

    @staticmethod
    def _secure_request(ws: ServerConnection) -> bool:
        if ws.transport.get_extra_info("ssl_object") is not None:
            return True
        headers = ws.request.headers if ws.request else {}
        forwarded = headers.get("X-Forwarded-Proto", "").split(",", 1)[0].strip().lower()
        return forwarded in {"https", "wss"}

    async def _voice_claims(
        self, ws: ServerConnection, provider: str, model: str
    ):
        if self.verifier is None:
            raise CapabilityError("voice_auth_not_configured")
        headers = ws.request.headers if ws.request else {}
        authorization = headers.get("Authorization", "")
        if not authorization.startswith("Bearer "):
            raise CapabilityError("missing_token")
        return await self.verifier.verify(
            authorization.removeprefix("Bearer ").strip(),
            headers.get("X-Voice-Request-Id", ""),
            provider,
            model,
        )

    @staticmethod
    def _param(query: dict[str, list[str]], name: str, default: str) -> str:
        values = query.get(name)
        value = values[0] if values else default
        if len(value) > 128:
            raise CapabilityError("invalid_parameter")
        return value

    async def _handle_stt(
        self, ws: ServerConnection, provider: str, query: dict[str, list[str]]
    ) -> None:
        model = self._param(
            query,
            "model",
            self.providers.stt_alias
            or "echo-stt-deepgram-nova-2@id:2026-07-26",
        )
        started = time.monotonic()
        tags = {"provider": provider, "model": model}
        try:
            if provider != "litellm":
                raise CapabilityError("unknown_provider")
            claims = await self._voice_claims(ws, provider, model)
            tags = claims.tags(provider, model)
            sample_rate = int(self._param(query, "sample_rate", "16000"))
            if sample_rate != 16_000:
                raise CapabilityError("unsupported_audio_format")
            chunks: list[bytes] = []
            total = 0
            async with asyncio.timeout(STT_DEADLINE_S):
                async for message in ws:
                    if isinstance(message, bytes):
                        total += len(message)
                        if len(chunks) >= MAX_STT_CHUNKS or total > MAX_STT_AUDIO_BYTES:
                            raise CapabilityError("media_limit_exceeded")
                        chunks.append(message)
                        continue
                    try:
                        control = json.loads(message)
                    except json.JSONDecodeError as exc:
                        raise CapabilityError("invalid_request") from exc
                    if control != {"type": "CloseStream"}:
                        raise CapabilityError("invalid_request")
                    break
            pcm = b"".join(chunks)
            if not pcm or len(pcm) % 2:
                raise CapabilityError("invalid_pcm")
            async with self.limiter.acquire(claims.org, provider):
                transcript = await self.providers.transcribe(
                    pcm=pcm,
                    model=model,
                    sample_rate=sample_rate,
                    language=self._param(query, "language", "pt"),
                    tags={**tags, "modality": "stt"},
                )
            await ws.send(
                json.dumps(
                    {
                        "type": "transcription",
                        "sequence": 0,
                        "is_final": True,
                        "text": transcript,
                    }
                )
            )
            await _close_quietly(ws, 1000, "stt_complete")
            self.observability.audit(
                "voice_stt",
                tags,
                status="ok",
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        except CapabilityError as exc:
            self.observability.audit(
                "voice_stt", tags, status="rejected", error_code=exc.code
            )
            code = CLOSE_BAD_REQUEST
            if exc.code == "scope_denied":
                code = CLOSE_SCOPE_DENIED
            elif exc.code in {"missing_token", "bad_signature", "expired", "replay"}:
                code = CLOSE_UNAUTHORIZED
            await _close_quietly(ws, code, exc.code)
        except RateLimitExceeded as exc:
            self.observability.audit(
                "voice_stt", tags, status="rejected", error_code=exc.code
            )
            await _close_quietly(ws, CLOSE_RATE_LIMITED, exc.code)
        except (TimeoutError, asyncio.TimeoutError):
            self.observability.audit(
                "voice_stt", tags, status="error", error_code="deadline_exceeded"
            )
            await _close_quietly(ws, CLOSE_UPSTREAM_ERROR, "deadline_exceeded")
        except (ProviderConfigurationError, OSError, websockets.WebSocketException):
            self.observability.audit(
                "voice_stt", tags, status="error", error_code="upstream_error"
            )
            await _close_quietly(ws, CLOSE_UPSTREAM_ERROR, "upstream_error")

    async def _proxy_voice_frames(self, client, upstream) -> None:
        async def forward_client() -> None:
            chunks = 0
            total = 0
            async for message in client:
                chunks += 1
                size = len(message.encode()) if isinstance(message, str) else len(message)
                total += size
                if chunks > MAX_STT_CHUNKS or total > MAX_STT_AUDIO_BYTES:
                    raise CapabilityError("media_limit_exceeded")
                await upstream.send(message)

        async def forward_upstream() -> None:
            total = 0
            chunks = 0
            async for message in upstream:
                chunks += 1
                total += len(message.encode()) if isinstance(message, str) else len(message)
                if chunks > MAX_STT_CHUNKS or total > MAX_STT_UPSTREAM_BYTES:
                    raise CapabilityError("upstream_limit_exceeded")
                await client.send(message)

        tasks = {
            asyncio.create_task(forward_client()),
            asyncio.create_task(forward_upstream()),
        }
        try:
            async with asyncio.timeout(STT_DEADLINE_S):
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    task.result()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            for task in tasks:
                with contextlib.suppress(
                    asyncio.CancelledError, websockets.ConnectionClosed
                ):
                    await task
            await _close_quietly(client, 1000, "stt_complete")
            close = getattr(upstream, "close", None)
            if close is not None:
                with contextlib.suppress(Exception):
                    await close()

    async def _handle_tts(
        self, ws: ServerConnection, provider: str, query: dict[str, list[str]]
    ) -> None:
        defaults = {
            "litellm": self.providers.tts_alias
            or "echo-tts-elevenlabs-flash-v2-5@id:2026-07-26",
        }
        model = self._param(query, "model", defaults.get(provider, "unknown"))
        started = time.monotonic()
        tags = {"provider": provider, "model": model}
        chunks = 0
        try:
            if provider not in defaults:
                raise CapabilityError("unknown_provider")
            claims = await self._voice_claims(ws, provider, model)
            tags = claims.tags(provider, model)
            try:
                raw = await asyncio.wait_for(
                    ws.recv(), timeout=CLIENT_REQUEST_TIMEOUT_S
                )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise ClientReadTimeout from exc
            if not isinstance(raw, str) or len(raw.encode()) > 32_768:
                raise CapabilityError("invalid_request")
            try:
                request = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise CapabilityError("invalid_request") from exc
            if not isinstance(request, dict):
                raise CapabilityError("invalid_request")
            text = request.get("text")
            voice = request.get("voice")
            output_format = request.get("output_format", "pcm_16000")
            if (
                not isinstance(text, str)
                or not 1 <= len(text) <= 5_000
                or not isinstance(voice, str)
                or not 1 <= len(voice) <= 128
                or not isinstance(output_format, str)
                or not 1 <= len(output_format) <= 32
            ):
                raise CapabilityError("invalid_request")
            if output_format != "pcm_16000":
                raise CapabilityError("unsupported_audio_format")
            if not claims.allows_voice(provider, voice):
                raise CapabilityError("voice_scope_denied")
            stream = getattr(self.providers, provider)(
                text=text,
                voice=voice,
                model=model,
                output_format=output_format,
                tags={**tags, "modality": "tts"},
            )
            async with self.limiter.acquire(claims.org, provider):
                await ws.send(
                    json.dumps(
                        {
                            "type": "audio",
                            "encoding": "pcm_s16le",
                            "sample_rate": 16_000,
                            "channels": 1,
                        }
                    )
                )
                total = 0
                try:
                    try:
                        async with asyncio.timeout(TTS_DEADLINE_S):
                            async for chunk in stream:
                                if not isinstance(chunk, bytes) or not chunk:
                                    continue
                                chunks += 1
                                total += len(chunk)
                                if (
                                    chunks > MAX_TTS_CHUNKS
                                    or total > MAX_TTS_AUDIO_BYTES
                                ):
                                    raise CapabilityError("media_limit_exceeded")
                                await ws.send(chunk)
                    except (TimeoutError, asyncio.TimeoutError) as exc:
                        raise UpstreamDeadlineExceeded from exc
                finally:
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        await close()
                if total == 0 or total % 2:
                    raise CapabilityError("invalid_pcm")
            await ws.send(json.dumps({"type": "end", "chunks": chunks}))
            self.observability.audit(
                "voice_tts",
                tags,
                status="ok",
                duration_ms=int((time.monotonic() - started) * 1000),
                chunks=chunks,
            )
            await _close_quietly(ws, 1000, "tts_complete")
        except CapabilityError as exc:
            self.observability.audit(
                "voice_tts", tags, status="rejected", error_code=exc.code
            )
            code = (
                CLOSE_SCOPE_DENIED
                if exc.code in {"scope_denied", "voice_scope_denied"}
                else CLOSE_BAD_REQUEST
            )
            if exc.code in {"missing_token", "bad_signature", "expired", "replay"}:
                code = CLOSE_UNAUTHORIZED
            await _close_quietly(ws, code, exc.code)
        except RateLimitExceeded as exc:
            self.observability.audit(
                "voice_tts", tags, status="rejected", error_code=exc.code
            )
            await _close_quietly(ws, CLOSE_RATE_LIMITED, exc.code)
        except ClientReadTimeout:
            self.observability.audit(
                "voice_tts", tags, status="rejected", error_code="request_timeout"
            )
            await _close_quietly(ws, CLOSE_BAD_REQUEST, "request_timeout")
        except UpstreamDeadlineExceeded:
            self.observability.audit(
                "voice_tts",
                tags,
                status="error",
                error_code="upstream_deadline_exceeded",
            )
            await _close_quietly(
                ws, CLOSE_UPSTREAM_ERROR, "upstream_deadline_exceeded"
            )
        except (ProviderConfigurationError, httpx.HTTPError, OSError):
            self.observability.audit(
                "voice_tts", tags, status="error", error_code="upstream_error"
            )
            await _close_quietly(ws, CLOSE_UPSTREAM_ERROR, "upstream_error")

    # -- lado runner (registro outbound do pod) --------------------------------

    def _runner_authorized(self, ws: ServerConnection) -> bool:
        if not self.auth_token:
            return self.allow_insecure_runner_auth
        header = (ws.request.headers.get("Authorization", "") if ws.request else "").strip()
        return bool(self.auth_token) and hmac.compare_digest(
            header, f"Bearer {self.auth_token}"
        )

    async def _handle_runner(self, ws: ServerConnection, token: str) -> None:
        if not self._runner_authorized(ws):
            _log("runner rejeitado (auth)")
            await ws.close(CLOSE_UNAUTHORIZED, "bad gateway token")
            return
        owner = uuid.uuid4().hex
        subscription = await self.state_backend.subscribe_relay(
            token, "twilio_to_runner"
        )
        try:
            if not await self.state_backend.claim_runner(
                token, owner, RELAY_LEASE_TTL_MS
            ):
                await ws.close(CLOSE_DUPLICATE_RUNNER, "token already registered")
                return
            _log("runner registrado")

            async def incoming() -> None:
                async for message in ws:
                    await self.state_backend.publish_relay(
                        token, "runner_to_twilio", message
                    )

            async def outgoing() -> None:
                while True:
                    payload = await subscription.get()
                    if payload:
                        _, message = unpack_relay(payload)
                        await ws.send(message)

            async def renew() -> None:
                while True:
                    await asyncio.sleep(RELAY_RENEW_INTERVAL_S)
                    if not await self.state_backend.renew_runner(
                        token, owner, RELAY_LEASE_TTL_MS
                    ):
                        raise StateBackendUnavailable("relay_lease_lost")

            await self._run_relay_tasks(incoming(), outgoing(), renew())
        except (websockets.ConnectionClosed, RelaySuperseded):
            pass
        except StateBackendUnavailable:
            await _close_quietly(ws, CLOSE_UPSTREAM_ERROR, "relay_unavailable")
        finally:
            with contextlib.suppress(StateBackendUnavailable):
                await self.state_backend.release_runner(token, owner)
            await subscription.close()
            _log("runner saiu")

    # -- lado Twilio (Media Streams) -------------------------------------------

    async def _handle_twilio(self, ws: ServerConnection, token: str) -> None:
        owner = uuid.uuid4().hex
        subscription = await self.state_backend.subscribe_relay(
            token, "runner_to_twilio"
        )
        try:
            if not await self.state_backend.runner_exists(token):
                await ws.close(CLOSE_UNKNOWN_TOKEN, "unknown call token")
                return
            await self.state_backend.claim_twilio(
                token, owner, RELAY_LEASE_TTL_MS
            )
            _log("twilio conectado")

            async def incoming() -> None:
                async for message in ws:
                    if not await self.state_backend.publish_relay(
                        token, "twilio_to_runner", message, owner
                    ):
                        raise RelaySuperseded

            async def outgoing() -> None:
                while True:
                    payload = await subscription.get()
                    if not payload:
                        await _close_quietly(
                            ws, CLOSE_RUNNER_GONE, "runner disconnected"
                        )
                        raise RelaySuperseded
                    target, message = unpack_relay(payload)
                    if target != owner:
                        raise RelaySuperseded
                    await ws.send(message)

            async def renew() -> None:
                while True:
                    await asyncio.sleep(RELAY_RENEW_INTERVAL_S)
                    if not await self.state_backend.runner_exists(token):
                        await _close_quietly(
                            ws, CLOSE_RUNNER_GONE, "runner disconnected"
                        )
                        raise RelaySuperseded
                    if not await self.state_backend.renew_twilio(
                        token, owner, RELAY_LEASE_TTL_MS
                    ):
                        raise RelaySuperseded

            await self._run_relay_tasks(incoming(), outgoing(), renew())
        except (websockets.ConnectionClosed, RelaySuperseded):
            pass
        except StateBackendUnavailable:
            await _close_quietly(ws, CLOSE_UPSTREAM_ERROR, "relay_unavailable")
        finally:
            with contextlib.suppress(StateBackendUnavailable):
                await self.state_backend.release_twilio(token, owner)
            await subscription.close()
            _log("twilio saiu")

    @staticmethod
    async def _run_relay_tasks(*coroutines) -> None:
        tasks = {asyncio.create_task(coroutine) for coroutine in coroutines}
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            for task in tasks:
                with contextlib.suppress(
                    asyncio.CancelledError, websockets.ConnectionClosed, Exception
                ):
                    await task


async def _close_quietly(ws: ServerConnection, code: int, reason: str) -> None:
    try:
        await ws.close(code, reason)
    except Exception:  # noqa: BLE001 — conexão já podre não pode travar o teardown
        pass


def _health_check(
    observability: VoiceObservability,
    metrics_token: str | None = None,
    readiness: ReadinessState | None = None,
):
    def process(connection: ServerConnection, request):
        if request.path in ("/healthz", "/health"):
            return connection.respond(http.HTTPStatus.OK, "ok\n")
        if request.path == "/readyz":
            if readiness is not None and readiness.ready():
                return connection.respond(http.HTTPStatus.OK, "ready\n")
            return connection.respond(
                http.HTTPStatus.SERVICE_UNAVAILABLE, "not ready\n"
            )
        if request.path == "/metrics":
            authorization = request.headers.get("Authorization", "")
            if not metrics_token or not hmac.compare_digest(
                authorization, f"Bearer {metrics_token}"
            ):
                return connection.respond(http.HTTPStatus.NOT_FOUND, "not found\n")
            return connection.respond(http.HTTPStatus.OK, observability.prometheus())
        return None

    return process


async def run_gateway(
    host: str = "0.0.0.0",
    port: int = 8991,
    auth_token: str | None = None,
    *,
    signing_secret: str | None = None,
    verifier: CapabilityVerifier | None = None,
    providers: VoiceProviders | None = None,
    limiter: VoiceRateLimiter | None = None,
    observability: VoiceObservability | None = None,
    state_backend: VoiceStateBackend | None = None,
    metrics_token: str | None = None,
    readiness: ReadinessState | None = None,
    ready: asyncio.Event | None = None,
    require_tls: bool = False,
    allow_insecure_runner_auth: bool = False,
) -> None:
    observability = observability or VoiceObservability(lambda line: _log(f"audit={line}"))
    if verifier is None and signing_secret:
        verifier = CapabilityVerifier(signing_secret)
    gateway = MediaGateway(
        auth_token,
        verifier=verifier,
        providers=providers,
        limiter=limiter,
        observability=observability,
        state_backend=state_backend,
        require_tls=require_tls,
        allow_insecure_runner_auth=allow_insecure_runner_auth,
    )
    async with serve(
        gateway.handle,
        host,
        port,
        process_request=_health_check(observability, metrics_token, readiness),
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


def _redis_backend(runtime: str) -> RedisVoiceStateBackend:
    import redis.asyncio as redis

    local = runtime in {"local", "dev", "development", "test"}
    redis_url = os.environ.get("REDIS_URL", "").strip()
    if redis_url:
        parsed = urlsplit(redis_url)
        if not local and parsed.scheme != "rediss":
            raise SystemExit("production Redis requires REDIS_URL with rediss://")
        insecure_reqs = {
            value.lower()
            for value in parse_qs(parsed.query).get("ssl_cert_reqs", [])
        }
        if not local and insecure_reqs & {"none", "optional"}:
            raise SystemExit("production Redis must validate TLS certificates")
        kwargs = {"socket_connect_timeout": 5, "socket_timeout": 5}
        if parsed.scheme == "rediss":
            kwargs.update(ssl_cert_reqs="required", ssl_check_hostname=True)
        client = redis.from_url(redis_url, **kwargs)
    else:
        if not local:
            raise SystemExit("production Redis requires REDIS_URL with rediss://")
        host = os.environ.get("REDIS_HOST", "").strip()
        if not host:
            raise SystemExit("REDIS_HOST or REDIS_URL is required for Redis voice state")
        client = redis.Redis(
            host=host,
            port=int(os.environ.get("REDIS_PORT", "6379")),
            password=os.environ.get("REDIS_PASSWORD") or None,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return RedisVoiceStateBackend(client)


def _state_backend(runtime: str) -> VoiceStateBackend:
    backend_name = os.environ.get("VOICE_GATEWAY_STATE_BACKEND", "").strip().lower()
    local = runtime in {"local", "dev", "development", "test"}
    if backend_name == "memory":
        if not local:
            raise SystemExit("memory voice state is allowed only in local/dev/test")
        return MemoryVoiceStateBackend()
    if backend_name != "redis":
        raise SystemExit(
            "VOICE_GATEWAY_STATE_BACKEND must be explicitly set to redis"
            + (" or memory in local/dev/test" if local else "")
        )
    return _redis_backend(runtime)


async def _redis_health_loop(
    backend: VoiceStateBackend, readiness: ReadinessState, interval_s: float
) -> None:
    while True:
        await asyncio.sleep(interval_s)
        try:
            await backend.ping()
        except Exception as exc:  # noqa: BLE001 - readiness must fail closed
            readiness.mark_failure(exc)
        else:
            readiness.mark_success()


def main() -> None:
    load_gateway_secrets()
    port = int(os.environ.get("ECHO_MEDIA_GATEWAY_PORT", "8991"))
    auth_token = os.environ.get("ECHO_MEDIA_GATEWAY_AUTH_TOKEN") or None
    signing_secret = os.environ.get("VOICE_GATEWAY_SIGNING_SECRET") or None
    metrics_token = os.environ.get("VOICE_GATEWAY_METRICS_TOKEN") or None
    runtime = os.environ.get("ECHO_RUNTIME_ENV", "").strip().lower()
    local = runtime in {"local", "dev", "development", "test"}
    if runtime not in {"local", "dev", "development", "test", "cloud", "staging", "prod", "production"}:
        raise SystemExit("ECHO_RUNTIME_ENV must explicitly identify local/dev or production")
    if not local and any(
        os.environ.get(name) for name in ("DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY")
    ):
        raise SystemExit(
            "direct provider credentials are forbidden in the production voice gateway"
        )
    if not signing_secret and not local:
        raise SystemExit(
            "VOICE_GATEWAY_SIGNING_SECRET is required outside local/dev/test"
        )
    if not local and (not auth_token or len(auth_token.encode()) < 32):
        raise SystemExit(
            "ECHO_MEDIA_GATEWAY_AUTH_TOKEN (at least 32 bytes) is required in production"
        )
    if not local and (not metrics_token or len(metrics_token.encode()) < 32):
        raise SystemExit(
            "VOICE_GATEWAY_METRICS_TOKEN (at least 32 bytes) is required in production"
        )
    allow_insecure = (
        local
        and os.environ.get(
            "ECHO_MEDIA_GATEWAY_ALLOW_INSECURE_RUNNER_AUTH", ""
        ).strip()
        == "1"
    )
    if not auth_token and not allow_insecure:
        raise SystemExit(
            "ECHO_MEDIA_GATEWAY_AUTH_TOKEN is required; local/dev may explicitly set "
            "ECHO_MEDIA_GATEWAY_ALLOW_INSECURE_RUNNER_AUTH=1"
        )
    if not auth_token:
        print(
            "AVISO: ECHO_MEDIA_GATEWAY_AUTH_TOKEN não definido — o registro de "
            "runners está SEM auth por opt-in explícito de dev local",
            file=sys.stderr,
        )

    async def _serve() -> None:
        backend = _state_backend(runtime)
        await backend.ping()
        redis_ping_interval = float(
            os.environ.get("VOICE_GATEWAY_REDIS_PING_INTERVAL_S", "5")
        )
        if redis_ping_interval <= 0:
            raise SystemExit("VOICE_GATEWAY_REDIS_PING_INTERVAL_S must be positive")
        readiness = ReadinessState(max_age_s=max(15.0, redis_ping_interval * 3))
        readiness.mark_success()
        providers = VoiceProviders(require_tls=not local)
        enabled_providers = {
            item.strip().lower()
            for item in os.environ.get(
                "VOICE_GATEWAY_ENABLED_PROVIDERS", ""
            ).split(",")
            if item.strip()
        }
        providers_ready, provider_error = providers.readiness(enabled_providers)
        readiness.set_provider_configuration(providers_ready, provider_error)
        health_task = asyncio.create_task(
            _redis_health_loop(backend, readiness, redis_ping_interval)
        )
        verifier = (
            CapabilityVerifier(
                signing_secret,
                replay_guard=ReplayGuard(backend=backend),
            )
            if signing_secret
            else None
        )
        limiter = VoiceRateLimiter(backend=backend)
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        task = asyncio.create_task(
            run_gateway(
                port=port,
                auth_token=auth_token,
                verifier=verifier,
                limiter=limiter,
                providers=providers,
                state_backend=backend,
                metrics_token=metrics_token,
                readiness=readiness,
                require_tls=not local,
                allow_insecure_runner_auth=allow_insecure,
            )
        )
        try:
            await stop.wait()
        finally:
            task.cancel()
            health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            with contextlib.suppress(asyncio.CancelledError):
                await health_task
            await backend.close()

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
