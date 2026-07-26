from pathlib import Path

from voidr_echo_runner.projected_secrets import load_projected_secrets


def test_never_loads_gateway_provider_secrets_into_runner(tmp_path: Path):
    (tmp_path / "DEEPGRAM_API_KEY").write_text("projected-deepgram\n", encoding="utf-8")
    (tmp_path / "ELEVENLABS_API_KEY").write_text("projected-elevenlabs\n", encoding="utf-8")
    env: dict[str, str] = {}

    loaded = load_projected_secrets(env, tmp_path)

    assert loaded == []
    assert "DEEPGRAM_API_KEY" not in env
    assert "ELEVENLABS_API_KEY" not in env


def test_existing_runner_environment_wins_over_projected_file(tmp_path: Path):
    (tmp_path / "TWILIO_AUTH_TOKEN").write_text("projected-value", encoding="utf-8")
    env = {"TWILIO_AUTH_TOKEN": "local-value"}

    loaded = load_projected_secrets(env, tmp_path)

    assert loaded == []
    assert env["TWILIO_AUTH_TOKEN"] == "local-value"


def test_loads_hive_gateway_token_from_csi_file(tmp_path: Path):
    (tmp_path / "HIVE_GATEWAY_TOKEN").write_text("projected-hive-token\n", encoding="utf-8")
    env: dict[str, str] = {}

    loaded = load_projected_secrets(env, tmp_path)

    assert loaded == ["HIVE_GATEWAY_TOKEN"]
    assert env["HIVE_GATEWAY_TOKEN"] == "projected-hive-token"


def test_managed_runtime_loads_gateway_token_but_drops_provider_keys(tmp_path: Path):
    (tmp_path / "ECHO_MEDIA_GATEWAY_TOKEN").write_text("scoped-capability\n")
    (tmp_path / "DEEPGRAM_API_KEY").write_text("must-not-load\n")
    (tmp_path / "ELEVENLABS_API_KEY").write_text("must-not-load\n")
    env = {
        "ECHO_RUNTIME_ENV": "production",
        "DEEPGRAM_API_KEY": "inherited-key-must-be-removed",
    }

    loaded = load_projected_secrets(env, tmp_path)

    assert loaded == ["ECHO_MEDIA_GATEWAY_TOKEN"]
    assert env["ECHO_MEDIA_GATEWAY_TOKEN"] == "scoped-capability"
    assert "VOICE_GATEWAY_TOKEN" not in env
    assert "DEEPGRAM_API_KEY" not in env
    assert "ELEVENLABS_API_KEY" not in env


def test_gateway_manifest_uses_read_only_csi_and_digest_without_secret_env():
    manifest = (
        Path(__file__).parents[1] / "gateway/k8s/echo-media-gateway.yaml"
    ).read_text(encoding="utf-8")

    assert "driver: secrets-store-gke.csi.k8s.io" in manifest
    assert "readOnly: true" in manifest
    assert "VOIDR_SECRET_DIR" in manifest
    assert "secretKeyRef:" not in manifest
    assert "envFrom:" not in manifest
    assert "@sha256:" in manifest
    assert ":TAG" not in manifest
    assert "replicas: 3" in manifest
    assert "kind: PodDisruptionBudget" in manifest
    assert "minAvailable: 1" in manifest
    assert "topologySpreadConstraints:" in manifest
    assert "podAntiAffinity:" in manifest
    assert "VOICE_GATEWAY_ENABLED_PROVIDERS" in manifest
