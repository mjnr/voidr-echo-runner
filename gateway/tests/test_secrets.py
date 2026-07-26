from pathlib import Path

from echo_media_gateway.secrets import load_gateway_secrets


def test_gateway_never_loads_direct_provider_credentials(tmp_path: Path):
    (tmp_path / "DEEPGRAM_API_KEY").write_text("deepgram\n", encoding="utf-8")
    (tmp_path / "ELEVENLABS_API_KEY").write_text("elevenlabs\n", encoding="utf-8")
    (tmp_path / "LITELLM_API_KEY").write_text("virtual\n", encoding="utf-8")
    env: dict[str, str] = {}

    loaded = load_gateway_secrets(env, tmp_path)

    assert loaded == ["LITELLM_API_KEY"]
    assert env["LITELLM_API_KEY"] == "virtual"
    assert "DEEPGRAM_API_KEY" not in env
    assert "ELEVENLABS_API_KEY" not in env


def test_environment_override_wins_without_logging_secret_values(tmp_path: Path):
    (tmp_path / "VOICE_GATEWAY_SIGNING_SECRET").write_text(
        "file-value\n", encoding="utf-8"
    )
    env = {"VOICE_GATEWAY_SIGNING_SECRET": "injected-value"}

    assert load_gateway_secrets(env, tmp_path) == []
    assert env["VOICE_GATEWAY_SIGNING_SECRET"] == "injected-value"
