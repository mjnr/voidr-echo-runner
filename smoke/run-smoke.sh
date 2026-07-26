#!/usr/bin/env bash
# Smoke E2E offline: echo-runner -> Hive v3 stub -> vivo-autopilot-mock.
#
#   1. Por default sobe um boundary stub Hive v3 estrito, sem provider externo.
#      SMOKE_HIVE_MODE=real usa um Hive real e exige as envs do contrato.
#   2. Sobe o mock limpo (porta 8765) e roda os 2 cases -> espera PASSED.
#   3. Reinicia o mock com MOCK_DEVIATION=jornada_errada e roda os 2 cases
#      -> espera FAILED com errorMessage citando o desvio (jornada_errada).
#
# Uso: ./smoke/run-smoke.sh   (de qualquer diretório)
set -u

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOCK_DIR="${MOCK_DIR:-$RUNNER_DIR/../vivo-autopilot-mock}"
PORT="${MOCK_PORT:-8765}"
TARGET="ws://localhost:$PORT/ws"
OUT_DIR="$RUNNER_DIR/out"
HIVE_MODE="${SMOKE_HIVE_MODE:-stub}"
HIVE_PORT="${SMOKE_HIVE_PORT:-18765}"
export MOCK_ACCESS_CODE="${MOCK_ACCESS_CODE:-919021552}"
export MOCK_PORT="$PORT"

MOCK_PID=""
HIVE_PID=""
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

stop_hive() {
  if [[ -n "$HIVE_PID" ]]; then
    kill "$HIVE_PID" 2>/dev/null || true
    wait "$HIVE_PID" 2>/dev/null || true
    HIVE_PID=""
  fi
}

cleanup() {
  stop_mock
  stop_hive
}
trap cleanup EXIT

configure_hive() {
  case "$HIVE_MODE" in
    stub)
      export HIVE_URL="http://127.0.0.1:$HIVE_PORT"
      export HIVE_GATEWAY_TOKEN="smoke-hive-v3-token"
      export VOIDR_ORG_ID="00000000-0000-4000-8000-000000000003"
      export HIVE_ECHO_PERSONA_V3_MODEL_REVISION="deepseek-v4-pro@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      ;;
    real)
      local missing=()
      local name
      for name in HIVE_URL HIVE_GATEWAY_TOKEN VOIDR_ORG_ID HIVE_ECHO_PERSONA_V3_MODEL_REVISION; do
        if [[ -z "${!name:-}" ]]; then missing+=("$name"); fi
      done
      if (( ${#missing[@]} )); then
        echo "ERRO: SMOKE_HIVE_MODE=real exige as envs: ${missing[*]}" >&2
        echo "Nenhum valor de secret foi exibido." >&2
        exit 2
      fi
      ;;
    *)
      echo "ERRO: SMOKE_HIVE_MODE deve ser 'stub' (default) ou 'real'." >&2
      exit 2
      ;;
  esac
}

start_hive() {
  if [[ "$HIVE_MODE" == "real" ]]; then
    log "usando Hive v3 real configurado por ambiente"
    return
  fi
  log "subindo boundary stub Hive persona-turn v3"
  (
    cd "$RUNNER_DIR"
    SMOKE_HIVE_PORT="$HIVE_PORT" \
      SMOKE_HIVE_TOKEN="$HIVE_GATEWAY_TOKEN" \
      SMOKE_HIVE_MODEL_REVISION="$HIVE_ECHO_PERSONA_V3_MODEL_REVISION" \
      "$RUNNER_DIR/.venv/bin/python" "$RUNNER_DIR/smoke/hive_v3_stub.py" \
      > "/tmp/echo-runner-hive-v3-smoke.log" 2>&1
  ) &
  HIVE_PID=$!
  disown "$HIVE_PID" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if curl -sf -m 1 "$HIVE_URL/healthz" >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$HIVE_PID" 2>/dev/null; then break; fi
    sleep 0.2
  done
  echo "ERRO: boundary stub Hive v3 não subiu (log em /tmp/echo-runner-hive-v3-smoke.log)" >&2
  exit 2
}

start_mock() {
  local deviation="$1"
  stop_mock
  log "subindo mock (MOCK_DEVIATION=$deviation)"
  (cd "$MOCK_DIR" && MOCK_DEVIATION="$deviation" "$MOCK_DIR/.venv/bin/vivo-mock" \
    > "/tmp/vivo-mock-smoke-$deviation.log" 2>&1) &
  MOCK_PID=$!
  disown "$MOCK_PID" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if curl -sf -m 1 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then return 0; fi
    sleep 0.5
  done
  echo "ERRO: mock não subiu (log em /tmp/vivo-mock-smoke-$deviation.log)"
  exit 2
}

# run_case <label> <case-file> <seed> <expected: passed|failed> [<substring esperada no errorMessage>]
run_case() {
  local label="$1" case_file="$2" seed="$3" expected="$4" expect_error="${5:-}"
  local run_id="smoke-$label"
  rm -rf "${OUT_DIR:?}/$run_id"  # run-ids fixos: artifacts de execuções antigas não podem vazar
  log "runner: $label (esperado: $expected)"
  (cd "$RUNNER_DIR" && uv run --no-sync echo-runner run \
    --case "$case_file" --target "$TARGET" --seed "$seed" \
    --out "$OUT_DIR" --run-id "$run_id")
  local exit_code=$?
  local report="$OUT_DIR/$run_id/report.json"
  local verdict
  verdict=$("$RUNNER_DIR/.venv/bin/python" - "$report" "$expected" "$expect_error" <<'PY'
import json, sys
report_path, expected, expect_error = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    report = json.load(open(report_path))
except Exception as exc:
    print(f"FAIL: report ilegível ({exc})"); sys.exit(0)
result = report["results"][0]
stats = report["stats"]
problems = []
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

configure_hive

log "preparando ambientes (uv sync)"
(cd "$MOCK_DIR" && uv sync -q) && (cd "$RUNNER_DIR" && uv sync -q) || { echo "uv sync falhou"; exit 2; }

start_hive

# check_redaction <run-id>: transcript salvo deve ter [CPF_1] e nunca o CPF cru
check_redaction() {
  local run_id="$1"
  local verdict
  verdict=$("$RUNNER_DIR/.venv/bin/python" - "$OUT_DIR/$run_id" <<'PY'
import json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
problems = []
blob = (run_dir / "transcript.json").read_text() + (run_dir / "timeline.json").read_text()
if "39053344705" in blob.replace(".", "").replace("-", ""):
    problems.append("CPF sintético em claro no transcript/timeline")
if "[CPF_1]" not in blob:
    problems.append("placeholder [CPF_1] ausente do transcript")
# No contrato v3 o Hive recebe o goal já redigido e devolve [CPF_1]. Nesse
# caminho correto não há CPF cru para o redator de persistência contar de novo.
print("FAIL: " + "; ".join(problems) if problems else "OK")
PY
)
  if [[ "$verdict" == OK ]]; then
    SUMMARY+=("PASS  redacao-cpf-artifacts")
  else
    SUMMARY+=("FAIL  redacao-cpf-artifacts — $verdict")
    FAILURES=$((FAILURES + 1))
  fi
}

start_mock none
run_case "consulta-saldo-ok" "cases/consulta-saldo-tc-001.yaml" 42 passed
run_case "bloqueio-financeiro-ok" "cases/bloqueio-financeiro-tc-002.yaml" 7 passed
run_case "redacao-cpf" "cases/consulta-saldo-tc-003-cpf.yaml" 42 passed
check_redaction "smoke-redacao-cpf"

start_mock jornada_errada
run_case "consulta-saldo-desvio" "cases/consulta-saldo-tc-001.yaml" 42 failed "jornada_errada"
run_case "bloqueio-financeiro-desvio" "cases/bloqueio-financeiro-tc-002.yaml" 7 failed "jornada_errada"

stop_mock

log "resumo do smoke"
for line in "${SUMMARY[@]}"; do echo "  $line"; done
TOTAL=${#SUMMARY[@]}
if [[ $FAILURES -eq 0 ]]; then
  echo -e "\n\033[32mSMOKE PASS\033[0m ($TOTAL/$TOTAL)"
  exit 0
else
  echo -e "\n\033[31mSMOKE FAIL\033[0m ($FAILURES falha(s))"
  exit 1
fi
