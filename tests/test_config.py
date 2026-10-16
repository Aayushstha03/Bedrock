from pathlib import Path

import pytest

from bedrock.core.config import Settings


def test_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BEDROCK_QDRANT_URL", "http://localhost:6333/")
    monkeypatch.setenv("BEDROCK_VAULT_PATH", "/tmp/vault")
    monkeypatch.setenv("BEDROCK_MODEL_CACHE_PATH", "/tmp/models")
    monkeypatch.setenv("BEDROCK_COLLECTION", "test_entries")
    monkeypatch.setenv("BEDROCK_DENSE_MODEL", "dense/test")
    monkeypatch.setenv("BEDROCK_SPARSE_MODEL", "sparse/test")
    monkeypatch.setenv("BEDROCK_PORT", "9090")
    monkeypatch.setenv("BEDROCK_EXCLUDED_FILES", "tags.md, index.md")

    settings = Settings.from_env()

    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.vault_path == Path("/tmp/vault")
    assert settings.model_cache_path == Path("/tmp/models")
    assert settings.collection_name == "test_entries"
    assert settings.port == 9090
    assert settings.excluded_files == ("tags.md", "index.md")
    assert settings.dense_model == "dense/test"
    assert settings.sparse_model == "sparse/test"
    assert settings.validate() == ()


def test_settings_reject_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BEDROCK_PORT", "0")
    with pytest.raises(ValueError, match="positive"):
        Settings.from_env()


def test_settings_report_contract_errors(tmp_path: Path) -> None:
    settings = Settings("qdrant", Path("relative"), tmp_path, "", "0.0.0.0", 8080)
    assert len(settings.validate()) == 3


def test_tag_aliases_must_target_known_tags(tmp_path: Path) -> None:
    settings = Settings(
        "http://qdrant",
        tmp_path,
        tmp_path,
        "entries",
        "0.0.0.0",
        8080,
        known_tags=("journal",),
        tag_aliases=(("recipes", "recipe"),),
    )
    assert "every tag alias must target a known canonical tag" in settings.validate()


def test_dense_token_window_defaults_to_the_model_declared_length() -> None:
    from bedrock.core.config import DENSE_TOKEN_WINDOW

    # all-MiniLM-L6-v2 declares max_seq_length 256 in sentence_bert_config.json.
    # FastEmbed's own default of 128 is half of that.
    assert DENSE_TOKEN_WINDOW == 256


def test_a_token_window_past_the_positional_limit_is_rejected(tmp_path: Path) -> None:
    settings = Settings(
        "http://qdrant:6333",
        tmp_path,
        tmp_path,
        "entries",
        "127.0.0.1",
        8080,
        dense_token_window=1024,
    )
    assert any("must not exceed 512" in error for error in settings.validate())


def test_a_tiny_token_window_is_rejected(tmp_path: Path) -> None:
    settings = Settings(
        "http://qdrant:6333",
        tmp_path,
        tmp_path,
        "entries",
        "127.0.0.1",
        8080,
        dense_token_window=8,
    )
    assert any("at least 32" in error for error in settings.validate())


def test_llm_settings_default_to_lm_studio_on_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BEDROCK_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("BEDROCK_LLM_MODEL", raising=False)
    settings = Settings.from_env()
    assert settings.llm_base_url == "http://host.docker.internal:1234/v1"
    assert settings.llm_model == "google/gemma-4-e4b"
    assert settings.validate() == ()

    assert settings.llm_timeout == 120.0
    monkeypatch.setenv("BEDROCK_LLM_TIMEOUT", "0")
    with pytest.raises(ValueError, match="positive"):
        Settings.from_env()
    monkeypatch.delenv("BEDROCK_LLM_TIMEOUT")

    assert settings.llm_temperature == 0.0
    monkeypatch.setenv("BEDROCK_LLM_TEMPERATURE", "0.7")
    assert Settings.from_env().llm_temperature == 0.7
    # Zero is the useful default, not a mistake, so the bound is a range.
    monkeypatch.setenv("BEDROCK_LLM_TEMPERATURE", "0")
    assert Settings.from_env().llm_temperature == 0.0
    for rejected in ("-0.1", "2.1"):
        monkeypatch.setenv("BEDROCK_LLM_TEMPERATURE", rejected)
        with pytest.raises(ValueError, match=r"between 0\.0 and 2\.0"):
            Settings.from_env()
    monkeypatch.delenv("BEDROCK_LLM_TEMPERATURE")

    monkeypatch.setenv("BEDROCK_LLM_BASE_URL", "http://127.0.0.1:1234/v1/")
    assert Settings.from_env().llm_base_url == "http://127.0.0.1:1234/v1"


def test_llm_settings_reject_a_non_http_base_url(tmp_path: Path) -> None:
    settings = Settings(
        "http://qdrant",
        tmp_path,
        tmp_path,
        "entries",
        "0.0.0.0",
        8080,
        llm_base_url="localhost:1234",
        llm_model="",
    )
    errors = settings.validate()
    assert "BEDROCK_LLM_BASE_URL must use http or https" in errors
    assert "BEDROCK_LLM_MODEL cannot be empty" in errors


def test_an_empty_environment_value_means_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compose substitutes an empty string for a key an older `.env` lacks.

    Refusing it would stop `bedrock serve`, and with it the `health` command an
    operator would run to find out why the container restarts.
    """
    monkeypatch.setenv("BEDROCK_LLM_TIMEOUT", "")
    monkeypatch.setenv("BEDROCK_LLM_TEMPERATURE", "")
    monkeypatch.setenv("BEDROCK_DENSE_TOKEN_WINDOW", "")
    monkeypatch.setenv("BEDROCK_PORT", "")
    settings = Settings.from_env()
    assert settings.llm_timeout == 120.0
    assert settings.llm_temperature == 0.0
    assert settings.dense_token_window == 256
    assert settings.port == 8080

    # A value that is present and wrong is still rejected.
    monkeypatch.setenv("BEDROCK_LLM_TIMEOUT", "not-a-number")
    with pytest.raises(ValueError, match="must be a number"):
        Settings.from_env()
