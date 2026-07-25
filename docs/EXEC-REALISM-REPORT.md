# EXEC-REALISM — Realismo conversacional na execução de jornadas

Missão: elevar o realismo da execução do voidr echo (testes de voz da Vivo)
para o nível de uma ligação real — persona que usa temperamento, letramento,
glossário e emoção, esquece dados, hesita, demora e soa como telefone.

Data: 2026-07-25. Repos: `voidr-echo-runner` (master), `voidr-hive`
(release-agent-test), `vivo-autopilot-mock` (master). Service: **zero
mudanças necessárias** (ver §4).

---

## 1. Síntese da pesquisa (EN + ZH)

### User simulators para voice agents
- **τ-voice / τ³-bench (Sierra, 2026; arXiv 2509.23124)** — o simulador de
  usuário de voz aplica *G.711 µ-law companding a 8 kHz*, mistura de ruído
  ambiental e degradação de canal ao áudio do usuário simulado; inclui
  comportamentos não-colaborativos e detecção de alucinação do simulador
  (o desvio do tester não conta contra o agente — mesma regra adotada no
  nosso fidelity/judge).
- **EVA-Bench (ServiceNow)** — suíte de perturbações para avaliar voice
  agents: ruídos de fundo por preset e degradação de conexão, aplicados de
  forma determinística/reprodutível.
- LLM user simulators "out-of-the-box" exibem memória sobre-humana e
  colaboração excessiva; realismo exige **injeção explícita** de esquecimento
  e não-colaboração ("Simulating Human Memory with LLMs", 2026).

### Disfluências e hesitações em TTS
- Fillers e pausas silenciosas se concentram onde o planejamento conceitual é
  difícil — ditar números de documento é o caso clássico; autocorreção é
  esperada (Interspeech 2015, "micro-structure of disfluencies";
  PMC4203439). Estratégias: anotação explícita no texto (o que usamos — o
  LLM gera "ai, peraí... é..." e reticências), predição automática, ou
  geração implícita pelo TTS.

### Latência de resposta humana
- Floor-transfer offsets são unimodais e assimétricos à direita, moda de
  100–300 ms em conversa casual (Stivers et al. 2009, PNAS; Levinson &
  Torreira 2015); respostas de TAREFA ao telefone (recall de dados) ficam
  em 700–1900 ms, mediana ~1200 ms percebida como humana (CHI'26,
  "Quantifying Latencies"). A velocidade de turn-taking é bem modelada por
  **distribuição gamma** com parâmetros por traço do falante (SIGDial 2025,
  "Modeling Turn-Taking Speed and Speaker Characteristics") — é exatamente o
  modelo do `Humanizer.reply_delay_s` (gamma k=2, base condicionada a
  idade/INAF/emoção/tamanho da fala + segundos extras de "buscar documento").

### Canal telefônico
- Banda clássica de telefonia 300–3400 Hz (ITU G.712; idiap
  acoustic-simulator); artefatos de codec via companding µ-law G.711 de
  8 bits; injeção de ruído de fundo com nível calibrado (dBFS) e cor
  (hiss/hum/rumble). Aplicar SÓ no canal do simulador, preservando o STT do
  lado do agente.

### Full-duplex / turn-taking (ZH)
- Pesquisa em chinês (关键词: 用户模拟器, 语音对话系统 测试, 全双工语音交互,
  拟人化 TTS 停顿 迟疑, 智能客服 压测 仿真用户): o trabalho industrial de
  Alibaba (阿里小蜜 — simulação de usuário para 智能客服/压测), Baidu
  (DuerOS 全双工免唤醒 — full-duplex com barge-in e predição de fim de turno)
  e iFlytek (讯飞 全双工语音交互) converge nos mesmos pontos: (a) o simulador
  precisa de latência variável e backchannel para estressar o turn-taking do
  agente; (b) pausas/hesitações "拟人化" (humanizadas) são injetadas no texto
  antes do TTS; (c) teste de robustez usa ruído e canal degradado
  determinísticos. Full-duplex/barge-in ficou como pendência (ver §6).

---

## 2. Diagnóstico — onde a execução perdia a riqueza da persona

O playground (echo-playground.service → hive `persona-turn`) sempre usou o
condicionamento rico (identidade, OCEAN, letramento, vocabulário, probes).
A execução **não**:

1. **Brain errado por default**: `serve_execution` usava
   `ECHO_BRAIN=scripted` como default. O `ScriptedBrain` é determinístico
   (goal em 1ª pessoa + reformulação) e ignora temperamento/letramento/
   glossário/emoção — a "persona pragmática" relatada pelo fundador era o
   scripted brain. **Fix**: default `llm` quando `HIVE_URL`+
   `HIVE_GATEWAY_TOKEN` existem; fallback scripted com aviso alto.
2. **Sem massa pessoal**: nenhum caminho levava CPF/nascimento à persona —
   ela não tinha O QUE esquecer nem ditar. **Fix**: contrato
   `ENVIRONMENT_PARAMS.ECHO_MASSA` + fallback `identity.facts`, com
   placeholders `{{massa.*}}` (PII nunca passa pelo LLM — guard do hive).
3. **Sem timing humano**: resposta do brain era enviada imediatamente
   (gaps de ~0–5 ms nos timestamps do transcript ANTES).
4. **Sem canal telefônico**: TTS ElevenLabs full-band direto no WAV.
5. **Jornadas sem espaço**: nenhuma jornada do mock pedia CPF/nascimento
   nem usava jargão mapeável ao glossário.

## 3. O que mudou, por repo

### voidr-echo-runner (master)
| Commit | Conteúdo |
|---|---|
| `5596cc8` | E3: letramento + vocabulário de glossário no payload persona-turn |
| `f4d6d4b` | E4: `jargao_tecnico` dinâmico pelo vocabulário da persona |
| `a1039af` | ScriptedBrain: goal em 1ª pessoa + reformulação |
| `8d58783` | **EXEC-REALISM core**: brain LLM default na execução; `humanize.py` (MassaFacts + lapsos + latência gamma); `callfx.py` (banda 300–3400 Hz + µ-law + ambience seedada); runner/audio/cli/service_mode wiring; 40 testes novos |
| `21d2980` | flow+case `segunda-via-fatura`; persona Márcia com literacy INAF rudimentar + glossaryVocabulary local |

Knobs (ENVIRONMENT_PARAMS ou env): `ECHO_BRAIN` (llm|scripted),
`ECHO_MASSA` (JSON), `ECHO_HUMAN_REALISM=0`, `ECHO_HUMAN_TIMING=0`,
`ECHO_CALL_AMBIENCE` (`none|quiet|home|office|street[:level]` ou JSON).
Tudo determinístico por (persona, seed). CLI: `--ambience`, `--no-humanize`.

### voidr-hive (release-agent-test)
| Commit | Conteúdo |
|---|---|
| `153435a` | Contrato **v2.2**: `personalData` (rótulo+placeholder) e `turnDirectives` no persona-turn (guard de PII cobre os campos novos); persona-fidelity com bloco "Realismo humano simulado" (lapsos/hesitações ≠ break); echo-judge instruído a julgar a REAÇÃO do agente às imperfeições, nunca a imperfeição em si |

### vivo-autopilot-mock (master)
| Commit | Conteúdo |
|---|---|
| `81d795f` | Transições por `pattern` (regex) + jornada `segunda-via-fatura` (CPF + nascimento + jargão "fatura em aberto"/"titularidade") |
| `950b982` | Prioridade código-de-barras sobre SMS (cliente cita "SMS" ao recusá-lo) |

### voidr-service
Nenhuma mudança necessária: secrets do environment já fluem genericamente
para `ENVIRONMENT_PARAMS` (ECHO_CALL_AMBIENCE/ECHO_BRAIN configuráveis por
environment secret), e o agente da frente de massa já dispatcha
`ECHO_MASSA` com o contrato exato que o runner consome
(`executions.service.ts`, resolução journey massa > seed data > persona).

## 4. Prova — ANTES × DEPOIS (execução real contra o mock :8768)

Caso `segunda-via-fatura-tc-004`, persona `dona-marcia-58-mineira`, seed 7,
hive isolado :3007, massa sintética via `ECHO_MASSA`
(`{"cpf":"390.533.447-05", "birthDate":"12/03/1966", ...}` — dígitos
verificadores válidos por algoritmo, não é documento real).

### ANTES (`out/proof-before`, brain scripted, sem humanização) — FAILED
```text
+0ms  agent  Oi! Eu sou o assistente virtual da Vivo...
+0ms  tester Oi, moço... então, a conta da Vivo venceu e você perdeu o papel — quer a segunda via...   ← 3ª pessoa vazada do goal
+0ms  agent  ...preciso do CPF do titular da linha...
+0ms  tester É... sim, sou eu, o titular da linha, pode confirmar.        ← não tem CPF para ditar
+0ms  agent  Sem problema, eu aguardo...
+0ms  tester É... sim, sou eu, o titular da linha, pode confirmar.        ← repete verbatim
      ...reprompts esgotados → transferencia_humano (must_not_visit violado)
```
Gaps de 0–1 ms entre agent e tester; nunca dita o CPF; cai em transferência.

### DEPOIS (`out/proof-after-text`, brain LLM + humanize + ECHO_MASSA) — PASSED
```text
+2833ms tester Oi... é, moço, eu tô ligando porque... a conta da Vivo venceu e eu perdi o papel,
               sabe? Eu queria... hã, outra conta pra pagar.                 ← sinônimo popular do glossário
        agent  ...preciso do CPF do titular da linha...
+6514ms tester Ai, peraí... deixa eu pegar aqui... onde foi que eu anotei esse trem... hã... é... [MASSA_CPF]
                                                                             ← lapso de memória + busca + dita
        agent  ...me confirma também a sua data de nascimento...
+2971ms tester É... deixa eu ver... hã... [MASSA_BIRTHDATE].
        agent  Consta uma fatura em aberto na titularidade da linha...
+3815ms tester Ai, moço... oitenta e nove e noventa? Nossa, tá caro esse trem... cê pode repetir o dia
               do vencimento? Eu me perdi aqui.                              ← INAF rudimentar: se perde em 2 dados
+3676ms tester Hã... pera, moço... cê falou SMS? Isso é aquela mensagenzinha no celular, né?...
                                                                             ← baixa fluência digital
```
Trajetória completa `saudacao → pedido_cpf → confirmacao_nascimento →
info_fatura → envio_sms`. Timeline auditável: `humanized_turn` turn 2 com
`delayMs=4186, memoryLapse=true, lapseCategory=cpf`;
`meta.humanize={massaSource: environment, seed: 7}`. O CPF real foi
substituído FORA do LLM e redigido nos artefatos (`[MASSA_CPF]`).

### Áudio (`out/proof-after-audio/call.redacted.wav`, 116.9 s) — PASSED
Mesma conversa por Deepgram STT + ElevenLabs TTS. Análise espectral do WAV:
- canal tester (persona): energia >3.6 kHz / in-band = **0.00005** e
  <250 Hz = **0.00002** → banda telefônica 300–3400 Hz confirmada + µ-law;
- canal agent: 0.025 / 0.561 → full-band normal (STT do agente intacto);
- `meta.audio.channelFx={preset: quiet, level: 1.0}`; 2 beeps de redação de
  PII (CPF + nascimento ditados, 9.2 s beepados).
Hesitações e latências sobreviveram ao caminho TTS→STT (gaps de 2.6–6.1 s
nos timestamps do tester).

### Fidelity — sem break
`POST :3007/echo/persona-fidelity` no transcript DEPOIS:
**fidelityScore 1.0, personaBreak false** (Sonnet). literacyAdherence:
"usou corretamente os sinônimos populares esperados"; characterBreak e
emotionalCoherence aprovados com confiança 0.95+ — as hesitações/lapsos
não foram tratados como quebra (bloco "Realismo humano simulado").

## 5. Validações
- runner: `uv run pytest` → **203 passed** (163 → 203; +40 da missão).
- hive: vitest unit echo → 68 passed; `tsc --noEmit` limpo.
- mock: `uv run pytest` → 24 passed.

## 6. Pendências
1. **Full-duplex/barge-in**: a persona ainda não interrompe o agente no
   meio da fala (pesquisa ZH aponta como próximo degrau de realismo).
2. **Pausas INTRA-fala no áudio**: hesitações hoje viram texto ("..."), o
   TTS as interpreta; inserir silêncios reais entre segmentos de TTS daria
   controle fino.
3. **E2E via service :3019**: a prova rodou pelo CLI do runner (mesmo
   CallRunner/wiring do `serve-execution`, coberto por testes); falta um
   dispatch completo service→GKE→runner com `ECHO_MASSA` resolvido pela
   frente de massa do outro agente quando ela estabilizar.
4. **Glossário resolvido pelo service na execução**: a partição
   `glossaryVocabulary` da persona vem do service no playground; na execução
   offline usamos a partição local do catálogo — conectar quando o dispatch
   enviar o vocabulário resolvido.
5. **Ambience por persona**: `speech.backgroundNoise` (ex. `tv_sala` da
   Márcia) poderia mapear para presets do `callfx` automaticamente.
