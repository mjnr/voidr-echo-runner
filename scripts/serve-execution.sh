#!/usr/bin/env bash
# Wrapper invoked by the voidr-service local VOICE dispatch (dev only —
# ECHO_LOCAL_RUNNER_SCRIPT). Receives the standard runner env contract
# (VOIDR_API_URL, EXECUTION_ID, VOIDR_ORG_ID, VOIDR_CLIENT_ID/SECRET,
# VOIDR_ACCESS_TOKEN, SHARDS_CURRENT/TOTAL, ENVIRONMENT_PARAMS) and runs one
# shard. Logs go to out/serve-logs/.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

LOG_DIR="$REPO_DIR/out/serve-logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${EXECUTION_ID:-unknown}-shard-${SHARDS_CURRENT:-1}.log"

uv run echo-runner serve-execution --out "$REPO_DIR/out" 2>&1 | tee -a "$LOG_FILE"
