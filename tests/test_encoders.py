from pathlib import Path

import pytest

from bedrock.core.config import Settings
from bedrock.index.encoders import LocalEncoders


class Dense:
    def __init__(self, size: int = 384) -> None:
        self.size = size

    def embed(self, texts: list[str]):
        return ([0.25] * self.size for _ in texts)


class SparseVector:
    indices = [2, 7]
    values = [0.8, 0.2]


class Sparse:
    def embed(self, texts: list[str]):
        return (SparseVector() for _ in texts)


def settings(tmp_path: Path) -> Settings:
    return Settings("http://qdrant:6333", tmp_path, tmp_path, "entries", "127.0.0.1", 8080)


def test_encoders_create_named_dense_and_sparse_vectors(tmp_path: Path) -> None:
    vectors = LocalEncoders(settings(tmp_path), Dense(), Sparse()).encode(["one", "two"])
    assert len(vectors) == 2
    assert len(vectors[0]["semantic"]) == 384
    assert vectors[0]["text-bm25"].indices == [2, 7]


def test_encoders_reject_wrong_dense_dimension(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="expected 384"):
        LocalEncoders(settings(tmp_path), Dense(12), Sparse()).encode(["bad"])
