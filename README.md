# voidr-echo-runner

Runner do **voidr echo**: um agente tester sintético que "liga" para um agente
conversacional alvo, navega a URA de acesso por DTMF, conduz a conversa
encarnando uma persona e avalia a trajetória contra um journey flow (máquina de
estados). Irmão de `voidr-runner` (Playwright) e `voidr-k6-runner` (k6) — mesmo
contrato de report, outra mídia.

v0 roda **100% offline** em modo texto contra o
[`vivo-autopilot-mock`](../vivo-autopilot-mock). Nenhum dado real: ANIs fake,
segredos apenas como placeholders `{{env.*}}`.

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
5. **Artifacts** em `./out/<run-id>/`:
   - `transcript.json` — diarizado (`ura`/`agent`/`tester`), timestamps, estado
     classificado por turno do agente
   - `timeline.json` — eventos (dtmf, turnos, transições de estado,
     encerramento) + trajetória + metadados (persona@version+seed, flow id)
   - `report.json` — contrato do runner Voidr (idêntico ao voidr-k6-runner):

```json
{
  "stats":   {"total": 1, "passed": 1, "failed": 0, "flaky": 0, "skipped": 0, "durationMs": 92},
  "results": [{"name": "consulta-saldo-tc-001", "status": "passed", "durationMs": 92,
               "errorMessage": "só quando failed"}]
}
```

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

## O que é stub / plugável

| Camada | v0 | Como ativa |
|---|---|---|
| Cérebro da persona | `ScriptedBrain` (determinístico, seedado) | `LLMBrain` atrás de `OPENAI_API_KEY`/`GEMINI_API_KEY` — stub em `brain.py`, interface `PersonaBrain` estável |
| Modo áudio | falha rápido com mensagem clara | `--mode audio` + `DEEPGRAM_API_KEY` (STT) e `ELEVENLABS_API_KEY` ou `AZURE_SPEECH_KEY` (TTS); montagem do pipeline Pipecat documentada em `audio.py` |
| Transporte | `LocalWebSocketTransport` (mock) | `TwilioTransport` atrás de `TWILIO_ACCOUNT_SID`/`TWILIO_AUTH_TOKEN`/`TWILIO_FROM_NUMBER` — stub estruturado em `transport.py` com o desenho Media Streams + `calls.create(send_digits=...)` + DTMF mid-call |

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
| `MOCK_*`, `OPENAI_API_KEY`, … | não | Passam direto para o core (mesmos knobs do modo CLI) |

\* obrigatório se `VOIDR_ACCESS_TOKEN` não vier.

### Dev local (sem GKE)

O voidr-service com `ECHO_LOCAL_RUNNER_SCRIPT=<path>/scripts/serve-execution.sh`
(e `NODE_ENV != production`) despacha cada shard de execution VOICE spawnando
esse wrapper localmente com o env acima — logs em `out/serve-logs/`. Em
produção o dispatch é um job GKE com a imagem `GKE_ECHO_RUNNER_IMAGE`, mesmo
contrato de env.
