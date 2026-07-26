"""Text normalization and {{env.*}} placeholder resolution."""

from __future__ import annotations

import os
import re
import unicodedata

ENV_PLACEHOLDER = re.compile(r"\{\{\s*env\.([A-Za-z0-9_]+)\s*\}\}")


def normalize(text: str) -> str:
    """Lowercase and strip accents ('não' -> 'nao') for keyword matching."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def keyword_matches(keyword: str, text: str) -> bool:
    """Multi-word keywords match as substring; single words on word boundary."""
    norm_kw = normalize(keyword)
    norm_text = normalize(text)
    if " " in norm_kw:
        return norm_kw in norm_text
    return re.search(rf"\b{re.escape(norm_kw)}\b", norm_text) is not None


def resolve_env_placeholders(value: str, captured: dict[str, str] | None = None) -> str:
    """Resolve {{env.NAME}} placeholders from os.environ, failing loudly.

    When `captured` is given, every substituted (NAME, value) pair is recorded
    there — the PII redaction deny-list is built from these known values.
    """

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if resolved is None:
            raise KeyError(
                f"environment variable {name!r} required by placeholder "
                f"{match.group(0)!r} is not set. For local runs against the mock, "
                f"e.g.: export MOCK_ACCESS_CODE=919021552"
            )
        if captured is not None:
            captured[name] = resolved
        return resolved

    return ENV_PLACEHOLDER.sub(_sub, value)


def resolve_placeholders_deep(obj, captured: dict[str, str] | None = None):
    """Recursively resolve placeholders in a YAML-loaded structure."""
    if isinstance(obj, str):
        return resolve_env_placeholders(obj, captured)
    if isinstance(obj, list):
        return [resolve_placeholders_deep(v, captured) for v in obj]
    if isinstance(obj, dict):
        return {k: resolve_placeholders_deep(v, captured) for k, v in obj.items()}
    return obj
