from pathlib import Path

from echo_media_gateway.secrets import load_gateway_secrets


def test_gateway_loads_direct_provider_credentials(tmp_path: Path):
    (tmp_path / "DEEPGRAM_API_KEY").write_text("deepgram\n", encoding="utf-8")
    (tmp_path / "ELEVENLABS_API_KEY").write_text("elevenlabs\n", encoding="utf-8")
    (tmp_path / "LITELLM_API_KEY").write_text("virtual\n", encoding="utf-8")
    env: dict[str, str] = {}

    loaded = load_gateway_secrets(env, tmp_path)

    assert loaded == ["DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY"]
    assert env["DEEPGRAM_API_KEY"] == "deepgram"
    assert env["ELEVENLABS_API_KEY"] == "elevenlabs"
    assert "LITELLM_API_KEY" not in env


def test_environment_override_wins_without_logging_secret_values(tmp_path: Path):
    (tmp_path / "VOICE_GATEWAY_SIGNING_SECRET").write_text(
        "file-value\n", encoding="utf-8"
    )
    env = {"VOICE_GATEWAY_SIGNING_SECRET": "injected-value"}

    assert load_gateway_secrets(env, tmp_path) == []
    assert env["VOICE_GATEWAY_SIGNING_SECRET"] == "injected-value"
