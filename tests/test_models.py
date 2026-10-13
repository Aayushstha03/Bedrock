from pathlib import Path

from bedrock.core.config import Settings
from bedrock.runtime.models import prefetch, readiness


class DenseEncoder:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def embed(self, texts: list[str]):
        yield [0.0] * 384


class SparseVector:
    indices = [1, 2]
    values = [0.5, 0.25]


class SparseEncoder:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def embed(self, texts: list[str]):
        yield SparseVector()


def settings(tmp_path: Path) -> Settings:
    return Settings("http://qdrant:6333", tmp_path, tmp_path / "models", "entries", "0.0.0.0", 8080)


def test_prefetch_validates_models_and_writes_versioned_marker(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    assert readiness(configured)["ready"] is False

    result = prefetch(configured, DenseEncoder, SparseEncoder)

    assert result["ready"] is True
    assert result["dense_dimensions"] == 384
    assert readiness(configured)["ready"] is True


def test_readiness_rejects_different_model_configuration(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    prefetch(configured, DenseEncoder, SparseEncoder)
    changed = Settings(
        configured.qdrant_url,
        configured.vault_path,
        configured.model_cache_path,
        configured.collection_name,
        configured.host,
        configured.port,
        dense_model="different/model",
    )
    assert readiness(changed) == {
        "ready": False,
        "reason": "cached model versions do not match configuration",
    }
