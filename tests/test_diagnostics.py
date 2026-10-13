from pathlib import Path
from types import SimpleNamespace

from qdrant_client import models

from bedrock.core.config import Settings
from bedrock.index.diagnostics import inspect_collection
from bedrock.index.schema import PAYLOAD_INDEXES


def settings(tmp_path: Path) -> Settings:
    return Settings("http://unused", tmp_path, tmp_path, "entries", "127.0.0.1", 8080)


class Client:
    def __init__(self, with_alias: bool = True) -> None:
        self.with_alias = with_alias

    def get_aliases(self):
        aliases = (
            [SimpleNamespace(alias_name="entries", collection_name="entries__build")]
            if self.with_alias
            else []
        )
        return SimpleNamespace(aliases=aliases)

    def get_collection(self, name: str):
        assert name == "entries__build"
        params = SimpleNamespace(
            vectors={
                "semantic": SimpleNamespace(size=384, distance=models.Distance.COSINE),
            },
            sparse_vectors={"text-bm25": object()},
        )
        return SimpleNamespace(
            config=SimpleNamespace(params=params),
            payload_schema={field: object() for field, _ in PAYLOAD_INDEXES},
            points_count=38,
        )


def test_inspection_reports_compatible_active_collection(tmp_path: Path) -> None:
    report = inspect_collection(settings(tmp_path), Client())
    assert report == {
        "ready": True,
        "collection": "entries",
        "physical_collection": "entries__build",
        "points": 38,
        "errors": [],
    }


def test_inspection_reports_missing_alias(tmp_path: Path) -> None:
    report = inspect_collection(settings(tmp_path), Client(with_alias=False))
    assert report["ready"] is False
    assert "does not exist" in report["reason"]
