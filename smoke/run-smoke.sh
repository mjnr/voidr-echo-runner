#!/usr/bin/env bash
# Smoke E2E offline: voidr-echo-runner <-> vivo-autopilot-mock (sem chaves, sem rede externa).
#
#   1. Sobe o mock limpo (porta 8765) e roda os 2 cases -> espera PASSED.
#   2. Reinicia o mock com MOCK_DEVIATION=jornada_errada e roda os 2 cases
#      -> espera FAILED com errorMessage citando o desvio (jornada_errada).
#
# Uso: ./smoke/run-smoke.sh   (de qualquer diretório)
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

log "preparando ambientes (uv sync)"
(cd "$MOCK_DIR" && uv sync -q) && (cd "$RUNNER_DIR" && uv sync -q) || { echo "uv sync falhou"; exit 2; }

start_mock none
run_case "consulta-saldo-ok" "cases/consulta-saldo-tc-001.yaml" 42 passed
run_case "bloqueio-financeiro-ok" "cases/bloqueio-financeiro-tc-002.yaml" 7 passed

start_mock jornada_errada
run_case "consulta-saldo-desvio" "cases/consulta-saldo-tc-001.yaml" 42 failed "jornada_errada"
run_case "bloqueio-financeiro-desvio" "cases/bloqueio-financeiro-tc-002.yaml" 7 failed "jornada_errada"

stop_mock

log "resumo do smoke"
for line in "${SUMMARY[@]}"; do echo "  $line"; done
if [[ $FAILURES -eq 0 ]]; then
  echo -e "\n\033[32mSMOKE PASS\033[0m (4/4)"
  exit 0
else
  echo -e "\n\033[31mSMOKE FAIL\033[0m ($FAILURES falha(s))"
  exit 1
fi
