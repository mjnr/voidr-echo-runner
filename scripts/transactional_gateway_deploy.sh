#!/usr/bin/env bash
set -euo pipefail

: "${NAMESPACE:?NAMESPACE is required}"
: "${DEPLOYMENT:?DEPLOYMENT is required}"
: "${SERVICE:?SERVICE is required}"
: "${IMAGE_DIGEST:?IMAGE_DIGEST is required}"
CONTAINER="${CONTAINER:-echo-media-gateway}"
PORT="${PORT:-8080}"
SMOKE_PATH="${SMOKE_PATH:-/readyz}"
BENCHMARK_COMMAND="${BENCHMARK_COMMAND:-:}"
DRY_RUN="${DRY_RUN:-0}"
revision="${DEPLOYMENT}-$(date +%Y%m%d%H%M%S)"
candidate_app="${DEPLOYMENT}-candidate-${revision##*-}"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%s\n' \
    "DRY-RUN: clone deployment/$DEPLOYMENT to deployment/$revision" \
    "DRY-RUN: provision candidate app=$candidate_app without Service traffic" \
    "DRY-RUN: wait readiness, benchmark and smoke $SMOKE_PATH" \
    "DRY-RUN: promote service/$SERVICE only after success" \
    "DRY-RUN: restore prior selector on failure"
  exit 0
fi

old_selector="$(kubectl -n "$NAMESPACE" get service "$SERVICE" -o jsonpath='{.spec.selector.app}')"
cleanup() { kubectl -n "$NAMESPACE" delete service "${revision}-smoke" --ignore-not-found >/dev/null; }
rollback() {
  kubectl -n "$NAMESPACE" patch service "$SERVICE" --type merge \
    -p "{\"spec\":{\"selector\":{\"app\":\"${old_selector}\"}}}" >/dev/null || true
  cleanup
}
trap rollback ERR
trap cleanup EXIT

kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" -o json |
  jq --arg name "$revision" --arg app "$candidate_app" --arg container "$CONTAINER" \
    --arg image "$IMAGE_DIGEST" '
      del(.metadata.uid,.metadata.resourceVersion,.metadata.generation,.metadata.creationTimestamp,
          .metadata.managedFields,.status) |
      .metadata.name=$name |
      .metadata.labels.app=$app |
      .spec.selector.matchLabels.app=$app |
      .spec.template.metadata.labels.app=$app |
      (.spec.template.spec.containers[] | select(.name==$container) | .image)=$image
    ' | kubectl apply -f -
kubectl -n "$NAMESPACE" rollout status "deployment/$revision" --timeout=10m
kubectl -n "$NAMESPACE" expose deployment "$revision" --name "${revision}-smoke" \
  --port "$PORT" --target-port "$PORT"
kubectl -n "$NAMESPACE" run "${revision}-probe" --rm -i --restart=Never \
  --image=curlimages/curl:8.15.0 -- \
  --fail --retry 8 "http://${revision}-smoke:${PORT}${SMOKE_PATH}"
K8S_CANDIDATE_SERVICE="${revision}-smoke" K8S_CANDIDATE_NAMESPACE="$NAMESPACE" \
  bash -ceu "$BENCHMARK_COMMAND"
kubectl -n "$NAMESPACE" patch service "$SERVICE" --type merge \
  -p "{\"spec\":{\"selector\":{\"app\":\"${candidate_app}\"}}}"
trap - ERR
