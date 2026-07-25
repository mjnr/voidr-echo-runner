#!/usr/bin/env bash
# Wrapper de dispatch VOICE que roda o shard num CONTAINER da imagem de nuvem
# (voidr-echo-runner:dev) — a prova canônica da paridade cloud no dev local.
# Plugue no voidr-service via ECHO_LOCAL_RUNNER_SCRIPT: recebe o mesmo
# contrato de env do GKE Job (VOIDR_API_URL, EXECUTION_ID, VOIDR_ORG_ID,
# credenciais, SHARDS_CURRENT/TOTAL, ENVIRONMENT_PARAMS) e o repassa ao
# container. Logs em out/serve-logs/*-docker.log.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/out/serve-logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${EXECUTION_ID:-unknown}-shard-${SHARDS_CURRENT:-1}-docker.log"
IMAGE="${ECHO_RUNNER_IMAGE:-voidr-echo-runner:dev}"

# Dentro do container, localhost é o próprio container — URLs do host
# (service, hive, mock ws://) viram host.docker.internal. Só afeta o dev
# local: na nuvem nada aponta para localhost.
rewrite_host() {
  printf '%s' "$1" \
    | sed -e 's#//localhost#//host.docker.internal#g' \
          -e 's#//127\.0\.0\.1#//host.docker.internal#g'
}

VOIDR_API_URL_DOCKER="$(rewrite_host "${VOIDR_API_URL:-http://localhost:3000}")"
# Atenção: ${VAR:-{}} em bash fecha a expansão no primeiro `}` e vaza um `}`
# literal para dentro do JSON — por isso o default vai numa atribuição própria.
ENV_PARAMS_RAW="${ENVIRONMENT_PARAMS:-}"
[ -n "$ENV_PARAMS_RAW" ] || ENV_PARAMS_RAW='{}'
ENVIRONMENT_PARAMS_DOCKER="$(rewrite_host "$ENV_PARAMS_RAW")"

echo "▶ serve-execution-docker imagem=$IMAGE execution=${EXECUTION_ID:-?} shard=${SHARDS_CURRENT:-1}/${SHARDS_TOTAL:-1}" >>"$LOG_FILE"

exec docker run --rm \
  -e CLOUD_PROVIDER="${CLOUD_PROVIDER:-local}" \
  -e VOIDR_ORG_ID \
  -e EXECUTION_ID \
  -e VOIDR_CLIENT_ID \
  -e VOIDR_CLIENT_SECRET \
  -e VOIDR_ACCESS_TOKEN \
  -e SHARDS_CURRENT \
  -e SHARDS_TOTAL \
  -e VOIDR_API_URL="$VOIDR_API_URL_DOCKER" \
  -e ENVIRONMENT_PARAMS="$ENVIRONMENT_PARAMS_DOCKER" \
  "$IMAGE" >>"$LOG_FILE" 2>&1
