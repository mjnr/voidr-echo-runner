# echo-media-gateway — build a partir do diretório gateway/:
#   docker build -t echo-media-gateway .
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.13-slim-bookworm
RUN useradd --create-home --uid 10001 gateway
COPY --from=builder --chown=gateway:gateway /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
USER gateway
EXPOSE 8991
ENTRYPOINT ["echo-media-gateway"]
