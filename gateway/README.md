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
                                     │ ws (OUTBOUND do pod)
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

## Rodar local

```bash
uv sync
ECHO_MEDIA_GATEWAY_AUTH_TOKEN=dev-secret uv run echo-media-gateway
# healthcheck: curl http://localhost:8991/healthz
```

Testes: `uv run pytest`. Prova E2E local (Twilio simulado → gateway →
runner em Docker): `../scripts/prove-gateway-e2e.sh` (ver docs/CLOUD-EXECUTION.md).

## Envs (gateway)

| Env | Default | Descrição |
|---|---|---|
| `ECHO_MEDIA_GATEWAY_PORT` | `8991` | Porta de escuta |
| `ECHO_MEDIA_GATEWAY_AUTH_TOKEN` | *(vazio)* | Shared secret exigido em `/runner/{token}`. Sem ele o gateway roda aberto — **só dev** |

## Envs (runner, via ENVIRONMENT_PARAMS ou processo)

| Env | Descrição |
|---|---|
| `ECHO_MEDIA_GATEWAY_URL` | Base pública `wss://` usada no TwiML (`/twilio/{token}`) — liga o modo gateway |
| `ECHO_MEDIA_GATEWAY_RUNNER_URL` | Base interna para o registro outbound do runner (ex.: `ws://echo-media-gateway.voidr-runners.svc.cluster.local:8991`); default = a pública |
| `ECHO_MEDIA_GATEWAY_TOKEN` | Shared secret do registro (par do `ECHO_MEDIA_GATEWAY_AUTH_TOKEN`) |

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
