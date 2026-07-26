#!/usr/bin/env bash
# Smoke E2E de ÁUDIO: voidr-echo-runner (--mode audio) <-> vivo-autopilot-mock.
#
# Mesma matriz do smoke texto, mas com áudio real fluindo nos dois sentidos
# por uma virtual key LiteLLM (TTS + STT — pode consumir créditos upstream):
#
#   1. Mock limpo: 2 cases -> espera PASSED.
#   2. Mock com MOCK_DEVIATION=jornada_errada: 2 cases -> espera FAILED
#      com errorMessage citando o desvio.
#
# Além do report, valida por run: out/<run-id>/call.redacted.wav gravado
# (estéreo tester/agente, PII com beep; o cru é descartado por default) e
# transcript.json com falas do agente vindas do STT real
# (entries[].source == "stt"). O case tc-003 dita um CPF sintético e o
# check_redaction_audio confere placeholder no transcript + beep no WAV.
#
# Requer LITELLM_BASE_URL, LITELLM_API_KEY e aliases TTS/STT governados.
#
# Uso: ./smoke/run-smoke-audio.sh   (de qualquer diretório)
set -u

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOCK_DIR="${MOCK_DIR:-$RUNNER_DIR/../vivo-autopilot-mock}"
PORT="${MOCK_PORT:-8765}"
TARGET="ws://localhost:$PORT/ws"
OUT_DIR="$RUNNER_DIR/out"
export MOCK_ACCESS_CODE="${MOCK_ACCESS_CODE:-919021552}"
export MOCK_PORT="$PORT"

MOCK_PID=""
FAILURES=0
SUMMARY=()

log() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

stop_mock() {
  if [[ -n "$MOCK_PID" ]]; then
    pkill -TERM -P "$MOCK_PID" 2>/dev/null || true
    kill "$MOCK_PID" 2>/dev/null || true
    MOCK_PID=""
  fi
  lsof -ti "tcp:$PORT" 2>/dev/null | xargs kill 2>/dev/null || true
  sleep 0.5
}
trap stop_mock EXIT

start_mock() {
  local deviation="$1"
  stop_mock
  log "subindo mock (MOCK_DEVIATION=$deviation)"
  (cd "$MOCK_DIR" && MOCK_DEVIATION="$deviation" "$MOCK_DIR/.venv/bin/vivo-mock" \
    > "/tmp/vivo-mock-smoke-audio-$deviation.log" 2>&1) &
  MOCK_PID=$!
  disown "$MOCK_PID" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if curl -sf -m 1 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then return 0; fi
    sleep 0.5
  done
  echo "ERRO: mock não subiu (log em /tmp/vivo-mock-smoke-audio-$deviation.log)"
  exit 2
}

# run_case <label> <case-file> <seed> <expected: passed|failed> [<substring esperada no errorMessage>]
run_case() {
  local label="$1" case_file="$2" seed="$3" expected="$4" expect_error="${5:-}"
  local run_id="smoke-audio-$label"
  rm -rf "${OUT_DIR:?}/$run_id"  # run-ids fixos: artifacts de execuções antigas não podem vazar
  log "runner --mode audio: $label (esperado: $expected)"
  (cd "$RUNNER_DIR" && set -o pipefail && uv run --no-sync echo-runner run \
    --case "$case_file" --target "$TARGET" --seed "$seed" --mode audio \
    --out "$OUT_DIR" --run-id "$run_id" 2>&1 | grep -v 'pipecat-ai-flows'; exit "${PIPESTATUS[0]}")
  local exit_code=$?
  local run_dir="$OUT_DIR/$run_id"
  local verdict
  verdict=$("$RUNNER_DIR/.venv/bin/python" - "$run_dir" "$expected" "$expect_error" <<'PY'
import json, sys, wave
from pathlib import Path

run_dir, expected, expect_error = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
problems = []
try:
    report = json.load(open(run_dir / "report.json"))
except Exception as exc:
    print(f"FAIL: report ilegível ({exc})"); sys.exit(0)
result = report["results"][0]
stats = report["stats"]
if result["status"] != expected:
    problems.append(f"status={result['status']} esperado={expected}")
if expected == "passed" and stats["passed"] != 1:
    problems.append(f"stats.passed={stats['passed']}")
if expected == "failed":
    msg = result.get("errorMessage", "")
    if stats["failed"] != 1:
        problems.append(f"stats.failed={stats['failed']}")
    if not msg:
        problems.append("errorMessage ausente")
    elif expect_error and expect_error not in msg:
        problems.append(f"errorMessage não cita {expect_error!r}: {msg[:100]}")

# --- validações específicas do modo áudio ---
# Redação de PII sempre ativa: só o call.redacted.wav é persistido
# (o cru exigiria ECHO_KEEP_RAW_AUDIO=1).
wav_path = run_dir / "call.redacted.wav"
if not wav_path.exists():
    problems.append("call.redacted.wav ausente")
else:
    with wave.open(str(wav_path)) as wav:
        seconds = wav.getnframes() / wav.getframerate()
        if wav.getnchannels() != 2:
            problems.append(f"call.redacted.wav com {wav.getnchannels()} canal(is), esperado 2")
        if seconds < 3:
            problems.append(f"call.redacted.wav muito curto ({seconds:.1f}s)")
if (run_dir / "call.wav").exists():
    problems.append("call.wav cru presente (deveria ser descartado por default)")

transcript = json.load(open(run_dir / "transcript.json"))
agent_entries = [e for e in transcript["entries"] if e["speaker"] == "agent"]
if not agent_entries:
    problems.append("transcript sem falas do agente")
elif not all(e.get("source") == "stt" for e in agent_entries):
    problems.append("fala do agente sem source=stt (não veio do STT real)")

print("FAIL: " + "; ".join(problems) if problems else "OK")
PY
)
  if [[ "$expected" == "passed" && $exit_code -ne 0 ]]; then verdict="FAIL: exit code $exit_code (esperado 0)"; fi
  if [[ "$expected" == "failed" && $exit_code -ne 1 ]]; then verdict="FAIL: exit code $exit_code (esperado 1)"; fi
  if [[ "$verdict" == OK ]]; then
    SUMMARY+=("PASS  $label")
  else
    SUMMARY+=("FAIL  $label — $verdict")
    FAILURES=$((FAILURES + 1))
  fi
}

log "preparando ambientes (uv sync)"
(cd "$MOCK_DIR" && uv sync -q) && (cd "$RUNNER_DIR" && uv sync -q) || { echo "uv sync falhou"; exit 2; }

# check_redaction_audio <run-id>: transcript com [CPF_1], WAV redigido com
# beep de 1 kHz no canal do tester (detecção por taxa de cruzamentos por zero)
check_redaction_audio() {
  local run_id="$1"
  local verdict
  verdict=$("$RUNNER_DIR/.venv/bin/python" - "$OUT_DIR/$run_id" <<'PY'
import json, sys, wave
from pathlib import Path
run_dir = Path(sys.argv[1])
problems = []
blob = (run_dir / "transcript.json").read_text()
if "39053344705" in blob.replace(".", "").replace("-", ""):
    problems.append("CPF sintético em claro no transcript")
if "[CPF_1]" not in blob:
    problems.append("placeholder [CPF_1] ausente do transcript")
with wave.open(str(run_dir / "call.redacted.wav")) as wav:
    rate, frames = wav.getframerate(), wav.readframes(wav.getnframes())
left = [int.from_bytes(frames[i*4:i*4+2], "little", signed=True) for i in range(len(frames)//4)]
win = rate // 10  # janelas de 100 ms
beeped = False
for start in range(0, len(left) - win, win):
    w = left[start:start+win]
    if max(abs(s) for s in w) < 3000:
        continue
    changes = sum(1 for a, b in zip(w, w[1:]) if (a >= 0) != (b >= 0))
    freq = changes / 2 / (len(w) / rate)
    if abs(freq - 1000) < 80:
        beeped = True
        break
if not beeped:
    problems.append("nenhum beep de 1kHz no canal do tester do call.redacted.wav")
print("FAIL: " + "; ".join(problems) if problems else "OK")
PY
)
  if [[ "$verdict" == OK ]]; then
    SUMMARY+=("PASS  redacao-cpf-beep")
  else
    SUMMARY+=("FAIL  redacao-cpf-beep — $verdict")
    FAILURES=$((FAILURES + 1))
  fi
}

start_mock none
run_case "consulta-saldo-ok" "cases/consulta-saldo-tc-001.yaml" 42 passed
run_case "bloqueio-financeiro-ok" "cases/bloqueio-financeiro-tc-002.yaml" 7 passed
run_case "redacao-cpf" "cases/consulta-saldo-tc-003-cpf.yaml" 42 passed
check_redaction_audio "smoke-audio-redacao-cpf"

start_mock jornada_errada
run_case "consulta-saldo-desvio" "cases/consulta-saldo-tc-001.yaml" 42 failed "jornada_errada"
run_case "bloqueio-financeiro-desvio" "cases/bloqueio-financeiro-tc-002.yaml" 7 failed "jornada_errada"

stop_mock

log "resumo do smoke de áudio"
for line in "${SUMMARY[@]}"; do echo "  $line"; done
TOTAL=${#SUMMARY[@]}
if [[ $FAILURES -eq 0 ]]; then
  echo -e "\n\033[32mSMOKE AUDIO PASS\033[0m ($TOTAL/$TOTAL)"
  exit 0
else
  echo -e "\n\033[31mSMOKE AUDIO FAIL\033[0m ($FAILURES falha(s))"
  exit 1
fi
