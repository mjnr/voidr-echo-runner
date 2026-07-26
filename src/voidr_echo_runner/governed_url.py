"""SSRF/TLS policy for Hive and LiteLLM clients."""

from __future__ import annotations

import os
from urllib.parse import urlsplit

LOCAL_RUNTIMES = frozenset({"local", "dev", "development", "test"})
DEFAULT_HOSTS = (
    "llm.voidr.co,hive.voidr.co,*.hive.voidr.co,*.run.app,"
    "*.svc,*.svc.cluster.local,localhost,127.0.0.1,*.test"
)


def _matches(host: str, rule: str) -> bool:
    rule = rule.strip().lower()
    if rule.startswith("*."):
        suffix = rule[1:]
        return host.endswith(suffix) and len(host) > len(suffix)
    return host == rule


def validate_governed_url(value: str, *, name: str) -> str:
    parsed = urlsplit(value)
    runtime = os.environ.get("ECHO_RUNTIME_ENV", "").strip().lower()
    local = runtime in LOCAL_RUNTIMES or (
        not runtime and "PYTEST_CURRENT_TEST" in os.environ
    )
    if parsed.username or parsed.password:
        raise RuntimeError(f"{name} must not contain userinfo")
    if parsed.scheme != "https" and not (local and parsed.scheme == "http"):
        raise RuntimeError(f"{name} must use https outside local/test")
    if not parsed.hostname:
        raise RuntimeError(f"{name} must contain a valid host")
    rules = os.environ.get("AI_EGRESS_HOST_ALLOWLIST", DEFAULT_HOSTS).split(",")
    if not any(_matches(parsed.hostname.lower(), rule) for rule in rules):
        raise RuntimeError(f"{name} host is not in AI_EGRESS_HOST_ALLOWLIST")
    return value.rstrip("/")
