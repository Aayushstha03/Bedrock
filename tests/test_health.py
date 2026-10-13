import json
from dataclasses import replace
from pathlib import Path
from urllib.error import URLError

from bedrock.core.config import Settings
from bedrock.runtime.health import health_report, llm_health


def test_health_report_combines_dependencies(monkeypatch, tmp_path: Path) -> None:
    settings = Settings("http://qdrant:6333", tmp_path, tmp_path / "models", "entries", "127.0.0.1", 8080)
    monkeypatch.setattr("bedrock.runtime.health.qdrant_health", lambda _: {"ready": True})
    monkeypatch.setattr("bedrock.runtime.health.readiness", lambda _: {"ready": True})
    monkeypatch.setattr("bedrock.runtime.health.index_health", lambda _: {"ready": True})
    report = health_report(settings)
    assert report["ready"] is True
    assert report["vault"]["read_only_expected"] is True


def test_health_is_false_when_any_dependency_fails(monkeypatch, tmp_path: Path) -> None:
    settings = Settings("http://qdrant:6333", tmp_path, tmp_path / "models", "entries", "127.0.0.1", 8080)
    monkeypatch.setattr("bedrock.runtime.health.qdrant_health", lambda _: {"ready": False})
    monkeypatch.setattr("bedrock.runtime.health.readiness", lambda _: {"ready": True})
    assert health_report(settings)["ready"] is False


def test_health_is_false_for_incompatible_index(monkeypatch, tmp_path: Path) -> None:
    settings = Settings("http://qdrant:6333", tmp_path, tmp_path / "models", "entries", "127.0.0.1", 8080)
    monkeypatch.setattr("bedrock.runtime.health.qdrant_health", lambda _: {"ready": True})
    monkeypatch.setattr("bedrock.runtime.health.readiness", lambda _: {"ready": True})
    monkeypatch.setattr(
        "bedrock.runtime.health.index_health",
        lambda _: {"ready": False, "errors": ["payload indexes are missing: entry_id"]},
    )

    report = health_report(settings)
    assert report["ready"] is False
    assert "payload indexes" in report["index"]["errors"][0]


def test_health_is_false_when_models_are_missing(monkeypatch, tmp_path: Path) -> None:
    settings = Settings("http://qdrant:6333", tmp_path, tmp_path / "models", "entries", "127.0.0.1", 8080)
    monkeypatch.setattr("bedrock.runtime.health.qdrant_health", lambda _: {"ready": True})
    monkeypatch.setattr("bedrock.runtime.health.index_health", lambda _: {"ready": True})
    monkeypatch.setattr(
        "bedrock.runtime.health.readiness",
        lambda _: {"ready": False, "reason": "model cache has not been prefetched"},
    )
    report = health_report(settings)
    assert report["ready"] is False
    assert "not been prefetched" in report["models"]["reason"]


def _ready_settings(tmp_path: Path) -> Settings:
    return Settings("http://qdrant:6333", tmp_path, tmp_path / "models", "entries", "127.0.0.1", 8080)


def test_a_closed_lm_studio_never_makes_the_deployment_unhealthy(monkeypatch, tmp_path: Path) -> None:
    settings = _ready_settings(tmp_path)
    monkeypatch.setattr("bedrock.runtime.health.qdrant_health", lambda _: {"ready": True})
    monkeypatch.setattr("bedrock.runtime.health.readiness", lambda _: {"ready": True})
    monkeypatch.setattr("bedrock.runtime.health.index_health", lambda _: {"ready": True})
    monkeypatch.setattr(
        "bedrock.runtime.health.llm_health",
        lambda _: {"ready": False, "required": False, "error": "Connection refused"},
    )

    report = health_report(settings, include_llm=True)
    assert report["llm"]["ready"] is False
    assert report["llm"]["required"] is False
    assert report["ready"] is True


def test_the_liveness_probe_does_not_reach_for_lm_studio(monkeypatch, tmp_path: Path) -> None:
    settings = _ready_settings(tmp_path)
    monkeypatch.setattr("bedrock.runtime.health.qdrant_health", lambda _: {"ready": True})
    monkeypatch.setattr("bedrock.runtime.health.readiness", lambda _: {"ready": True})
    monkeypatch.setattr("bedrock.runtime.health.index_health", lambda _: {"ready": True})

    def fail(_: Settings):  # pragma: no cover - must not run
        raise AssertionError("the container liveness probe must not call LM Studio")

    monkeypatch.setattr("bedrock.runtime.health.llm_health", fail)
    assert "llm" not in health_report(settings)


def test_llm_health_is_ready_only_when_the_configured_model_is_served(monkeypatch, tmp_path: Path) -> None:
    import io

    settings = _ready_settings(tmp_path)
    # Derived from the settings, not written out: the configured model changes
    # when a different one is loaded, and this test is about the comparison.
    served = {"data": [{"id": settings.llm_model}, {"id": "bge-m3"}]}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(
        "bedrock.runtime.health.urlopen",
        lambda url, timeout=1.0: Response(json.dumps(served).encode()),
    )
    report = llm_health(settings)
    assert report["ready"] is True
    assert report["available_models"] == sorted(["bge-m3", settings.llm_model])

    missing = llm_health(replace(settings, llm_model="not-loaded"))
    assert missing["ready"] is False


def test_llm_health_reports_a_refused_connection_as_not_ready(monkeypatch, tmp_path: Path) -> None:
    def refuse(url: str, timeout: float = 1.0):
        raise URLError("Connection refused")

    monkeypatch.setattr("bedrock.runtime.health.urlopen", refuse)
    report = llm_health(_ready_settings(tmp_path))
    assert report["ready"] is False
    assert report["required"] is False
    assert "Connection refused" in report["error"]


def test_llm_health_reports_a_malformed_model_listing_as_not_ready(monkeypatch, tmp_path: Path) -> None:
    """An OpenAI-compatible server need not return the shape OpenAI returns.

    Raising here would lose the Qdrant, index, model, and vault report that
    `bedrock health` was actually run for.
    """
    import io

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    for body in ('[{"id": "m"}]', '{"data": ["m"]}', "not json", '{"data": null}'):
        monkeypatch.setattr(
            "bedrock.runtime.health.urlopen",
            lambda url, timeout=1.0, payload=body: Response(payload.encode()),
        )
        report = llm_health(_ready_settings(tmp_path))
        assert report["ready"] is False
        assert report["required"] is False
