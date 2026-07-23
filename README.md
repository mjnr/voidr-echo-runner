# voidr-echo-runner

Runner do **voidr echo**: um agente tester sintético que "liga" para um agente
conversacional alvo, navega a URA de acesso por DTMF, conduz a conversa
encarnando uma persona e avalia a trajetória contra um journey flow (máquina de
estados). Irmão de `voidr-runner` (Playwright) e `voidr-k6-runner` (k6) — mesmo
contrato de report, outra mídia.

O modo texto roda **100% offline** contra o
[`vivo-autopilot-mock`](../vivo-autopilot-mock). O modo áudio usa STT/TTS reais
(Deepgram + ElevenLabs via Pipecat) e o transporte Twilio faz chamadas PSTN de
verdade. O playground de personas (`echo-runner chat`) conversa com a persona
via LLM — sempre através do hive, nunca com chave de LLM local. Nenhum dado
real: ANIs fake, segredos só via env (`.env` gitignored, ver `.env.example`).

## Como rodar

```bash
# terminal 1 — o alvo (mock)
cd ../vivo-autopilot-mock && uv sync && uv run vivo-mock

# terminal 2 — o runner
uv sync
export MOCK_ACCESS_CODE=919021552   # resolvido pelo placeholder {{env.MOCK_ACCESS_CODE}} do case
uv run echo-runner run --case cases/consulta-saldo-tc-001.yaml --target ws://localhost:8765/ws --seed 42
```

Exit code: `0` passed · `1` failed · `2` erro de setup. Flags: `--mode
text|audio` (default text), `--brain scripted|llm` (default scripted),
`--personas`, `--out`, `--run-id`.

### Smoke E2E (gate da fase 1)

```bash
./smoke/run-smoke.sh
```

Sobe o mock limpo → roda os 2 cases (espera PASSED) → reinicia o mock com
`MOCK_DEVIATION=jornada_errada` → roda de novo (espera FAILED com
`errorMessage` citando `jornada_errada`) → imprime resumo PASS/FAIL. Totalmente
offline. Testes unitários: `uv run pytest`.

## Modo áudio (`--mode audio`)

Áudio real nos dois sentidos contra o mock (protocolo WS ganha mensagens
`audio`, PCM s16le mono 16 kHz base64, uma fala completa por mensagem — ver
README do mock):

```bash
# exige DEEPGRAM_API_KEY e ELEVENLABS_API_KEY (no .env dos dois repos)
uv run echo-runner run --case cases/consulta-saldo-tc-001.yaml \
  --target ws://localhost:8765/ws --seed 42 --mode audio
```

- **STT**: `DeepgramSTTService` do Pipecat (websocket streaming, `nova-2`,
  `language=pt-BR`) — cada fala do agente chega como áudio e o transcript é o
  que o STT ouviu de verdade (`entries[].source == "stt"`).
- **TTS**: `ElevenLabsHttpTTSService` do Pipecat (`eleven_flash_v2_5`,
  `language=pt`, PCM 16 kHz). Vozes por persona (premade da voice library,
  em `personas/catalog.yaml`):

| Persona | Voz ElevenLabs | voiceId |
|---|---|---|
| `dona-marcia-58-mineira` | Matilda (feminina, meia-idade) | `XrExE9yKIg1WjnnlVkGX` |
| `carlos-34-paulista` | Liam (masculino, jovem) | `TX3LPaxmHKxFdv7VOQHJ` |
| agente do mock | Sarah (via `MOCK_TTS_VOICE_ID`) | `EXAVITQu4vr4xnSDxMaL` |

- O cérebro continua o `ScriptedBrain` determinístico — o áudio é a camada
  nova; a arquitetura é `CallRunner` (inalterado) + `AudioTransportAdapter`
  (`audio.py`), que converte áudio→texto e texto→áudio com pipelines Pipecat
  reais por turno.
- Artifacts extras em `out/<run-id>/`: `call.redacted.wav` (estéreo: L =
  tester, R = agente, PII com beep de 1 kHz — ver "Redação de PII"; o cru só
  com `ECHO_KEEP_RAW_AUDIO=1`) e `meta.audio` no `timeline.json` (voiceId,
  turnos STT/TTS, duração, beeps).

### Smoke de áudio (consome créditos Deepgram/ElevenLabs)

```bash
./smoke/run-smoke-audio.sh
```

Mesma matriz do smoke texto (happy paths PASSED + desvios FAILED citando
`jornada_errada`, mais o case `tc-003` em que a persona dita um CPF sintético),
validando também `call.redacted.wav` (com beep de 1 kHz no intervalo do CPF),
a ausência do `call.wav` cru e transcript vindo de STT real.
Custo típico da matriz completa: ~35k caracteres de TTS (~USD 1–2 no plano
ElevenLabs) + ~10 min de STT Deepgram (~USD 0.06).

## Anatomia de um run

1. **Dial plan**: executa os `dtmf_steps` do case contra a URA (mensagens
   `dtmf` no protocolo WS do mock; ver README do mock).
2. **Conversa**: a persona (cérebro `ScriptedBrain`, determinístico e seedado)
   responde cada turno do agente a partir de `goalTemplate` + `vocabulary` +
   regras por keyword. Mesma seed ⇒ mesmas falas.
3. **Rastreio**: um classificador v0 por keywords mapeia cada turno do agente
   para um estado do journey flow (JSON, formato da seção 6.1 do
   ARCHITECTURE.md) e registra a trajetória.
4. **Avaliação local v0**: `assert.flow` (`must_visit` / `must_not_visit` /
   `max_turns`) → passed/failed.
5. **Redação de PII** (sempre ativa — seção abaixo): transcript, timeline,
   report e áudio são redigidos ANTES de qualquer persistência.
6. **Artifacts** em `./out/<run-id>/`:
   - `transcript.json` — diarizado (`ura`/`agent`/`tester`), timestamps, estado
     classificado por turno do agente, PII como placeholders (`[CPF_1]`)
   - `timeline.json` — eventos (dtmf, turnos, transições de estado,
     encerramento) + trajetória + metadados (persona@version+seed, flow id,
     `piiRedactionReport`)
   - `report.json` — contrato do runner Voidr (idêntico ao voidr-k6-runner):

```json
{
  "stats":   {"total": 1, "passed": 1, "failed": 0, "flaky": 0, "skipped": 0, "durationMs": 92},
  "results": [{"name": "consulta-saldo-tc-001", "status": "passed", "durationMs": 92,
               "errorMessage": "só quando failed"}]
}
```

## Redação de PII (seção 10 do ARCHITECTURE.md)

Pré-requisito para chamadas reais com massas da Vivo: **nenhum dado sensível
em texto plano em nenhum artifact, payload ou log**. A redação é **sempre
ativa** no `run` e no `serve-execution` (`--no-redaction` existe só para dev,
com warning) e roda em `redaction.py` + `audio_redaction.py`, em duas camadas:

1. **Deny-list de massas (a camada crítica)** — os valores injetados via
   `{{env.X}}` no case e os valores de `massa`/`dial_plan` são conhecidos no
   runtime e redigidos por igualdade E por fuzzy de dígitos: com/sem
   pontuação, espaçados, ditados dígito a dígito ("nove um nove zero..."),
   inclusive "meia" = 6. O placeholder é rastreável sem expor o valor:
   `[MASSA_MOCK_ACCESS_CODE]`, `[MASSA_ANI]`.
2. **Detectores genéricos BR** — regex + validadores para precisão:
   - CPF (com máscara, sem máscara e por extenso; dígitos verificadores
     validados) → `[CPF_n]`
   - CNPJ (dígitos verificadores) → `[CNPJ_n]`
   - telefone BR (+55, DDD, celular/fixo, por extenso) → `[TELEFONE_n]`
   - cartão (Luhn) → `[CARTAO_n]`, CEP → `[CEP_n]`, e-mail → `[EMAIL_n]`
   - data de nascimento **em contexto** ("nascida em ...") → `[DATA_NASCIMENTO_n]`
   - fail-closed: qualquer sequência ditada de ≥ 8 dígitos que não casou com
     nada acima é tratada como potencial ANI → `[NUMERO_n]`
     (números inválidos NÃO escapam: CPF com DV errado vira `[NUMERO_n]`)

Mesma entidade ⇒ mesmo placeholder na sessão inteira (o transcript continua
legível). O `timeline.json` (meta) e o serve-execution carregam o
`piiRedactionReport` — contagem de entidades por tipo, nunca os valores.

**Áudio**: os segmentos do WAV cujo texto contém PII são re-transcritos via
Deepgram prerecorded (word-level timestamps) e os intervalos das palavras de
PII recebem **beep de 1 kHz** (±120 ms de padding), gerando
`call.redacted.wav`. Se o alinhamento por palavras falhar, o turno INTEIRO é
beepado (fail-closed). O `call.wav` cru é **descartado por default** — só é
mantido com `ECHO_KEEP_RAW_AUDIO=1`. Custo/latência: pós-chamada (zero impacto
na conversa ao vivo), 1 chamada REST por segmento com PII.

**Caminhos protegidos**: artifacts (`transcript/timeline/report`), payload do
`POST /echo/sessions` (transcript, timeline, deviations, target), history
enviado ao `persona-turn` do hive (que também valida — 422 com PII em claro) e
os eventos `dtmf_sent` da timeline (código de acesso/ANI digitados).

**Engine**: detector próprio (regex + DV/Luhn + parser de dígitos falados),
zero dependência extra. [Microsoft Presidio](https://github.com/microsoft/presidio)
com spaCy `pt_core_news_lg` fica como segunda passada **opt-in** para NER de
nomes/endereços (custo: ~500 MB de modelo + dependência pesada — instale
`presidio-analyzer` + `presidio-anonymizer` e rode sobre o transcript já
redigido; os recognizers custom de CPF/CNPJ daqui continuam necessários, o
Presidio não os traz nativos).

## Formatos de entrada

**Voice test case** (`cases/*.yaml`, seção 4.4 do ARCHITECTURE.md):

```yaml
id: consulta-saldo-tc-001
channel: voice
persona: { base: dona-marcia-58-mineira, variant_seed: 42 }
massa: { ani: "11900000001", profile: MASSA_ANI_SALDO }
dial_plan:
  dtmf_steps:
    - wait_for_prompt_matching: "codigo de acesso"
      send: "{{env.MOCK_ACCESS_CODE}}"
    - wait_for_prompt_matching: "digite o numero"
      send: "11900000001#"
journey_flow: ../flows/consulta-saldo-v1.json
goal: "frase-objetivo que preenche o {goal} do goalTemplate da persona"
assert:
  flow:
    must_visit: [saudacao, identificacao, diagnostico_saldo, oferta_recarga, envio_deep_link]
    must_not_visit: [transferencia_humano, jornada_errada]
    max_turns: 14
```

**Personas** (`personas/catalog.yaml`, schema da seção 5.4): `demographics`,
`temperament`, `speech` (voiceId placeholder até a fase de áudio),
`goalTemplate` (com `{goal}`), `vocabulary`. v0 traz 2 personas curadas
(`dona-marcia-58-mineira`, `carlos-34-paulista`).

**Journey flows** (`flows/*.json`, seção 6.1): estados com `expects`, `next`,
`terminal`, `evidence` + extensão `keywords` usada pelo classificador v0 (um
LLM classifier substitui na fase seguinte, mesma interface). O estado virtual
`jornada_errada` carrega keywords da jornada *errada* — é assim que o desvio é
detectado.

## Transporte Twilio (chamadas PSTN reais)

`--target tel:+<E164>` (exige `--mode audio`) usa `TwilioMediaStreamTransport`
(`twilio_transport.py`):

1. `calls.create` outbound com TwiML inline `<Connect><Stream/>` apontando para
   o servidor WebSocket local do runner (Media Streams, 8 kHz μ-law via
   `TwilioFrameSerializer` do Pipecat), `send_digits` com pausas `w` (montado
   automaticamente do `dial_plan` do case) e **gravação dual-channel** ligada.
2. Áudio de entrada é segmentado em falas por VAD de energia
   (`UtteranceSegmenter`) e entra no mesmo `AudioTransportAdapter` do modo
   áudio local (STT/TTS/brain idênticos).
3. **DTMF mid-call**: a Twilio *não* suporta DTMF outbound por Media Streams
   bidirecional, e o endpoint REST `/Calls/{sid}/Play.json` **não existe**
   (404 code 20404, verificado em chamada real). O caminho implementado é o
   workaround suportado: update do TwiML com `<Play digits>` +
   re-`<Connect><Stream>` — o stream cai e reconecta em ~1–3 s (validado ao
   vivo). Prefira `send_digits` na criação da chamada para navegar a URA.

Envs: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` e
`TWILIO_STREAM_PUBLIC_URL` (URL pública do tunnel):

```bash
# terminal 1 — tunnel para a porta do Media Streams server (default 8990)
# IMPORTANTE: use 127.0.0.1, não localhost — o cloudflared alterna para ::1
# (IPv6) com localhost e gera 502 intermitente no tunnel.
cloudflared tunnel --url http://127.0.0.1:8990   # ou: ngrok http 8990
# copie a URL pública para o .env como TWILIO_STREAM_PUBLIC_URL=wss://<host>

# terminal 2 — chamada real (SÓ nas janelas contratuais, com coordenação)
uv run echo-runner run --case cases/consulta-saldo-tc-001.yaml \
  --target tel:+55XXXXXXXXXXX --mode audio --seed 42
```

Validação real executada (números próprios da conta, ~20 s de chamadas):
Media Stream conectado via cloudflared em ~3 s, 8.2 s de fala inbound
atravessando `TwilioFrameSerializer` → segmenter → pipeline, DTMF mid-call com
reconexão do stream em 2.9 s e áudio fluindo após, hangup limpo via REST.
**Nunca aponte para números Vivo/IBM fora das janelas contratuais.**

## Playground de personas: `echo-runner chat`

"Dar play e conversar com a persona": você digita como o agente da Vivo e a
persona responde no personagem, com LLM real. Seguindo a regra 8.5 do
`ARCHITECTURE.md`, **o runner não tem chave de LLM** — o `LLMBrain` chama o
gateway síncrono do hive (`POST {HIVE_URL}/echo/persona-turn`, auth
`Bearer HIVE_GATEWAY_TOKEN`), que roteia DeepSeek v4 Pro → Sonnet (escalação)
e registra billing por organização.

```bash
# envs no .env: HIVE_URL, HIVE_GATEWAY_TOKEN, VOIDR_ORG_ID
uv run echo-runner chat --persona dona-marcia-58-mineira \
  [--journey flows/consulta-saldo-v1.json] [--voice] [--escalate] \
  [--seed 42] [--goal "quero saber meu saldo"]
```

- **`--journey`** — mantém o `journeyState` atualizado com o classificador de
  estados por keywords sobre o que você digita (a persona fica contextualizada
  na jornada; o estado corrente aparece no rodapé de cada turno). Sem a flag,
  usa o estado genérico `conversa-livre`.
- **`--voice`** — além do texto, sintetiza a resposta com a voz ElevenLabs da
  persona (`speech.voiceId` do catálogo) e toca no alto-falante (afplay).
  Sem `ELEVENLABS_API_KEY`, degrada para texto com aviso.
- **`--escalate`** — todos os turnos no modelo de escalação (Sonnet).
- **`--goal`** — preenche o `{goal}` do `goalTemplate` da persona.
- **Comandos no prompt**: `/escalate` (força Sonnet no próximo turno),
  `/state` (mostra o journeyState), `/help`, `/quit` (ou Ctrl-D).

Cada turno mostra o modelo usado e o custo (do `usage` do hive); o custo
acumulado da conversa é impresso na saída. Erros do hive viram mensagens
claras: `400` payload, `422` PII em claro no history (redija `<CPF>`,
`<TELEFONE>`), `502` gateway LLM indisponível.

O mesmo cérebro funciona no modo de teste: `echo-runner run ... --brain llm`
executa o case com a persona LLM em vez do `ScriptedBrain`.

Para desenvolvimento local, suba o hive da worktree com Mongo/Redis locais e
aponte `HIVE_URL` para ele (ex.: `http://localhost:3210`).

## O que é stub / plugável

| Camada | v0 | Como ativa |
|---|---|---|
| Cérebro da persona | `ScriptedBrain` (determinístico, seedado — default dos testes) | `LLMBrain` **implementado** via hive (`--brain llm` / `echo-runner chat`) — envs `HIVE_URL`, `HIVE_GATEWAY_TOKEN`, `VOIDR_ORG_ID`; sem chave de LLM no runner |
| Modo áudio | **implementado** (Deepgram + ElevenLabs via Pipecat) | `--mode audio` + `DEEPGRAM_API_KEY` + `ELEVENLABS_API_KEY`; TTS Azure (`AZURE_SPEECH_KEY`) segue stub |
| Transporte | `LocalWebSocketTransport` (mock, texto e áudio) | `TwilioMediaStreamTransport` **implementado** — `tel:+E164` + envs `TWILIO_*` + tunnel público (seção acima) |

## Modo service: `echo-runner serve-execution`

Modo de integração com o voidr-service (mesmo contrato HTTP do
`voidr-k6-runner`): o runner é disparado como job — 1 chamada = 1 shard — e
faz todo o ciclo sozinho. Implementado em `src/voidr_echo_runner/service_mode.py`.

```bash
VOIDR_API_URL=http://localhost:3010 \
EXECUTION_ID=<execId> \
VOIDR_ORG_ID=<orgId> \
VOIDR_CLIENT_ID=sa_... VOIDR_CLIENT_SECRET=sk_... \
SHARDS_CURRENT=1 SHARDS_TOTAL=2 \
ENVIRONMENT_PARAMS='{"BASE_URL":"...","IBM_TEST_NUMBER":"ws://..."}' \
uv run echo-runner serve-execution --out out
```

Fluxo executado:

1. **Auth** — usa `VOIDR_ACCESS_TOKEN` se o dispatch já injetou um token
   pré-mintado; senão `POST /v1/service-accounts/token` (client credentials).
2. **Resolução** — `GET /v1/executions/:id` → pega o target do shard
   (`SHARDS_CURRENT` é 1-based, `targets[shard-1]`), carrega o case no plan
   (`GET /v1/test-plans/:planId`, subdocumento `voice`), a persona
   (`GET /v1/echo/personas/:id`) e o journey flow
   (`GET /v1/echo/journey-flows/:id`). Placeholders `{{env.*}}` no `dialPlan`
   e no target são resolvidos com `ENVIRONMENT_PARAMS`.
3. **Chamada** — `PUT /shards/:i` com `RUNNING`, executa contra o alvo
   (`dialPlan.to`, ex.: `ws://localhost:8765/ws`) com o core deste CLI
   (persona brain seedada + classificador de trajetória + avaliador
   `flowAssert`).
4. **Artifacts** — pede signed URLs em `POST /v1/executions/:id/artifacts/upload-urls`
   e sobe `report.json` (contrato k6: `stats` + `results[{name,status,durationMs,errorMessage?}]`,
   `name` = slug do case) em `shards/{i}/reporter/json/test-results.json`,
   mais `transcript.json` e `timeline.json`.
5. **Report** — `PUT /v1/executions/:id/shards/:i` com `FINISHED`/`FAILED`;
   o service parseia o report (`voice-report.parser.ts`) e persiste
   `testCaseResults` quando o último shard finaliza.
6. **Sessão** — `POST /v1/echo/sessions` com transcript, trajetória, status
   (`passed | deviation | escalation | abandoned | env_failure` — mesma
   precedência do `voice-eval.service.ts`), `deviations[]`, métricas e paths
   dos artifacts.

### Envs do contrato

| Env | Obrigatória | Descrição |
|---|---|---|
| `VOIDR_API_URL` | sim | Base do voidr-service (ex.: `http://localhost:3010`; `/v1` é adicionado) |
| `EXECUTION_ID` | sim | Execution `provider: VOICE` criada no service |
| `VOIDR_ORG_ID` | sim | Organization id (Auth0) |
| `VOIDR_CLIENT_ID` / `VOIDR_CLIENT_SECRET` | sim* | Credenciais da service account |
| `VOIDR_ACCESS_TOKEN` | não | Token pré-mintado pelo dispatch; dispensa o par client id/secret |
| `SHARDS_CURRENT` / `SHARDS_TOTAL` | sim | Shard deste job (1-based) / total |
| `ENVIRONMENT_PARAMS` | sim | JSON com secrets do environment (resolve `{{env.*}}`) |
| `MOCK_*`, `HIVE_*`, … | não | Passam direto para o core (mesmos knobs do modo CLI) |

\* obrigatório se `VOIDR_ACCESS_TOKEN` não vier.

### Dev local (sem GKE)

O voidr-service com `ECHO_LOCAL_RUNNER_SCRIPT=<path>/scripts/serve-execution.sh`
(e `NODE_ENV != production`) despacha cada shard de execution VOICE spawnando
esse wrapper localmente com o env acima — logs em `out/serve-logs/`. Em
produção o dispatch é um job GKE com a imagem `GKE_ECHO_RUNNER_IMAGE`, mesmo
contrato de env.
