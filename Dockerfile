# voidr-echo-runner — imagem de execução em nuvem (GKE Job / serve-execution).
#
# Build (local, arch nativa):
#   docker build -t voidr-echo-runner:dev .
# Build para GKE (amd64) + push (ver docs/CLOUD-EXECUTION.md):
#   docker buildx build --platform linux/amd64 \
#     -t southamerica-east1-docker.pkg.dev/<proj>/<repo>/voidr-echo-runner:<tag> --push .
#
# O ENTRYPOINT é `echo-runner serve-execution` porque o GKE Job do
# voidr-service NÃO seta `command` — o contrato inteiro chega por env vars
# (VOIDR_API_URL, EXECUTION_ID, VOIDR_ORG_ID, VOIDR_CLIENT_ID/SECRET ou
# VOIDR_ACCESS_TOKEN, SHARDS_CURRENT/TOTAL, ENVIRONMENT_PARAMS). Secrets de
# plataforma (gateway/Hive/Twilio e tokens) são injetados diretamente pelo
# service. ENVIRONMENT_PARAMS contém apenas dados/configuração do cliente e
# nunca pode sobrescrever essas coordenadas governadas.

# ── estágio 1: resolve dependências e instala o projeto ─────────────────────
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Camada de deps separada do código: rebuilds de código não re-resolvem o lock.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ── estágio 2: runtime enxuto ────────────────────────────────────────────────
FROM python:3.13-slim-bookworm

# ca-certificates: TLS para Twilio/Deepgram/ElevenLabs/GCS signed URLs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 runner \
    && mkdir -p /work/out && chown -R runner:runner /work

COPY --from=builder --chown=runner:runner /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

USER runner
WORKDIR /work

# Porta do servidor Media Streams (modo tunnel local; no modo gateway o runner
# só faz conexões outbound e nenhuma porta precisa ser exposta).
EXPOSE 8990

ENTRYPOINT ["echo-runner", "serve-execution"]
CMD ["--out", "/work/out"]
