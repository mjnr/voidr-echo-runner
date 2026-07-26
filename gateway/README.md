# voidr echo media gateway

Endpoint **WSS público estável** para o Twilio Media Streams alcançar o
`voidr-echo-runner` rodando como pod efêmero (GKE Job). Substitui o tunnel
por-máquina (`TWILIO_STREAM_PUBLIC_URL` → porta 8990) do dev local.

## O problema

O transporte Twilio do runner sobe um servidor WebSocket e manda no TwiML
`<Connect><Stream url="wss://..."/>` — o Twilio precisa **conectar de volta**
nesse servidor. Um pod de Job não tem endereço público, e um LoadBalancer por
job custa caro e demora minutos para provisionar (o setup da chamada é
segundos).

## O desenho

```
                       ┌────────────────────────────┐
   PSTN  ──chamada──▶  │           Twilio           │
                       └─────────────┬──────────────┘
                                     │ wss (inbound)
                                     ▼
                     wss://media-gw.voidr.co/twilio/{token}
                       ┌────────────────────────────┐
                       │     echo-media-gateway     │   Deployment estável
                       │  pareia por {token} e faz  │   + Service/Ingress TLS
                       │  proxy cego bidirecional   │
                       └─────────────▲──────────────┘
                                     │ wss (OUTBOUND do pod)
                     /runner/{token} │ Authorization: Bearer <secret>
                       ┌─────────────┴──────────────┐
                       │  voidr-echo-runner (pod)   │   GKE Job por shard
                       │  TwilioMediaStreamTransport │
                       └────────────────────────────┘
```

1. O runner gera um **token aleatório por chamada** (128 bits), conecta
   *outbound* em `/runner/{token}` (Bearer `ECHO_MEDIA_GATEWAY_TOKEN`) e só
   então cria a chamada com TwiML apontando para `/twilio/{token}`.
2. O gateway pareia as duas conexões e repassa frames nos dois sentidos sem
   parsear nada (o JSON do Media Streams passa intacto).
3. **DTMF mid-call** (update de TwiML + re-`<Connect><Stream>`): o lado
   Twilio cai e reconecta com o mesmo token; a conexão do runner permanece.
4. Runner desconectou (fim da chamada ou pod morto) → o gateway fecha o lado
   Twilio (code 4410) e libera o token.

Segurança: o registro de runner exige o shared secret; o caminho `/twilio/`
não tem como carregar auth (Media Streams não envia headers custom), então a
proteção é o token de chamada não-adivinhável, com vida amarrada à conexão
do runner. Frames de mídia nunca são logados.

## Voice provider gateway

O mesmo processo pode expor transporte de capability, mas todo upstream é
LiteLLM:

- `WS /v1/stt/litellm`: recebe um utterance PCM e encaminha multipart para
  `/audio/transcriptions`.
- `WS /v1/tts/litellm`: devolve chunks encaminhados de `/audio/speech`,
  terminando com `{"type":"end","chunks":N}`.

Essas rotas exigem `Authorization: Bearer <capability>` e
`X-Voice-Request-Id`. A capability compacta HMAC-SHA256 contém `org`,
`execution`, `shard`, `providers`, `models`, `voices`, `iat`, `exp`, `jti` e
`max_requests`. Assinatura, expiração, allowlists não vazias de modelos e voice
IDs por provider, escopo, nonce/replay e limite de usos são validados antes de abrir
chamada ao LiteLLM. Replay, limite de usos, token bucket de CPS e
semáforo de concorrência por organização/provider usam operações Lua atômicas
no Redis compartilhado, com TTL/leases renováveis. Falha do Redis fecha o
acesso em produção.

Logs de auditoria contêm somente tags (`org`, `execution`, `shard`, `provider`,
`model`), status, código de erro, duração e número de chunks. Áudio, texto,
transcript, tokens e chaves nunca são logados. Métricas estão em `/metrics`.

## Rodar local

```bash
uv sync
ECHO_RUNTIME_ENV=local \
VOICE_GATEWAY_STATE_BACKEND=memory \
ECHO_MEDIA_GATEWAY_AUTH_TOKEN=dev-secret \
uv run echo-media-gateway
# healthcheck: curl http://localhost:8991/healthz
```

Testes: `uv run pytest`. Prova E2E local (Twilio simulado → gateway →
runner em Docker): `../scripts/prove-gateway-e2e.sh` (ver docs/CLOUD-EXECUTION.md).

## Envs (gateway)

| Env | Default | Descrição |
|---|---|---|
| `ECHO_MEDIA_GATEWAY_PORT` | `8991` | Porta de escuta |
| `ECHO_RUNTIME_ENV` | *(obrigatório)* | `local/dev/test` ou `cloud/staging/prod/production` |
| `ECHO_MEDIA_GATEWAY_AUTH_TOKEN` | *(obrigatório)* | Shared secret de `/runner/{token}`; mínimo 32 bytes em produção |
| `ECHO_MEDIA_GATEWAY_ALLOW_INSECURE_RUNNER_AUTH` | `0` | Opt-in `1` sem auth, somente em local/dev/test |
| `VOICE_GATEWAY_SIGNING_SECRET` | *(obrigatório fora de local/dev/test)* | Chave HMAC para validar capabilities efêmeras (mínimo 32 bytes) |
| `VOICE_GATEWAY_STATE_BACKEND` | *(obrigatório)* | `redis` em produção; `memory` somente explicitamente em local/test |
| `VOICE_GATEWAY_ENABLED_PROVIDERS` | *(obrigatório para readiness)* | Deve ser exatamente `litellm` |
| `REDIS_HOST` / `REDIS_PORT` | *(somente local/test)* | Redis plaintext para desenvolvimento |
| `REDIS_URL` | *(obrigatório em produção)* | `rediss://` com certificado e hostname validados; estado, leases e relay Pub/Sub compartilhados entre réplicas |
| `LITELLM_BASE_URL` | *(obrigatório)* | Endpoint interno do LiteLLM |
| `LITELLM_API_KEY` | *(obrigatório)* | Virtual key restrita aos aliases Echo |
| `LITELLM_TTS_MODEL` | *(obrigatório)* | Alias TTS imutável e pinado |
| `LITELLM_STT_MODEL` | *(obrigatório)* | Alias STT imutável e pinado |

## Envs (runner, via ENVIRONMENT_PARAMS ou processo)

| Env | Descrição |
|---|---|
| `ECHO_MEDIA_GATEWAY_URL` | Base pública `wss://` usada no TwiML (`/twilio/{token}`) — liga o modo gateway |
| `ECHO_MEDIA_GATEWAY_RUNNER_URL` | Base de registro outbound; em produção deve ser `wss://`, sem downgrade; default = a pública |
| `ECHO_MEDIA_GATEWAY_TOKEN` | Shared secret do registro (par do `ECHO_MEDIA_GATEWAY_AUTH_TOKEN`) |
| `LITELLM_BASE_URL` | Gateway único de text/image/audio |
| `LITELLM_API_KEY` | Virtual key org-scoped entregue em envelope de uso único |
| `LITELLM_TTS_MODEL` / `LITELLM_STT_MODEL` | Aliases pinados |

Chamadas diretas a Deepgram/ElevenLabs só são aceitas com
`ECHO_RUNTIME_ENV=local|dev` **e** `VOICE_ALLOW_DIRECT_PROVIDERS=1`.

## Benchmark TTS

O script `../scripts/benchmark_voice_tts.py` mede por amostra TTFB, latência
total, chunks, bytes, erros e uma verificação estrutural de áudio não vazio:

```bash
LITELLM_BASE_URL=https://... LITELLM_API_KEY=... \
  uv run python scripts/benchmark_voice_tts.py --voice VOICE_ID --runs 20
```

A virtual key precisa permitir os aliases Echo pinados. O relatório compara
contra um JSON baseline; não existe fallback direto em produção.

## Deploy K8s (exemplo)

Manifests de exemplo em [`k8s/`](k8s/): Deployment (2 réplicas), Service,
`BackendConfig` com timeout longo (chamadas duram minutos) e Ingress GCE com
TLS. O gateway é stateless por sessão-de-token em memória — réplicas > 1
exigem **afinidade por sessão** (as duas pontas do mesmo token precisam cair
na MESMA réplica). O manifest usa `sessionAffinity: ClientIP` no Service
como base, mas o par runner/Twilio tem IPs diferentes: para múltiplas
réplicas, o roteamento correto é por hash do PATH (nginx ingress
`upstream-hash-by: $request_uri`) — documentado no manifest. Com 1 réplica
(suficiente para o volume de chamadas atual: cada frame tem ~400 bytes a
cada 20ms por chamada) nada disso importa.

Imagem: `docker build -t echo-media-gateway gateway/` (Dockerfile próprio).
