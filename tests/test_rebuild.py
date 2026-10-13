from pathlib import Path
from types import SimpleNamespace

import pytest

from bedrock.core.config import Settings
from bedrock.index.rebuild import IndexRebuilder


class Encoders:
    def encode(self, texts: list[str]):
        return [{"semantic": [0.0] * 384, "text-bm25": {"indices": [1], "values": [1.0]}} for _ in texts]


class Client:
    def __init__(self, count_offset: int = 0, previous: str | None = "entries__old") -> None:
        self.count_offset = count_offset
        self.previous = previous
        self.collections: set[str] = set()
        self.points: list[object] = []
        self.alias_operations = None
        self.deleted: list[str] = []

    def get_aliases(self):
        aliases = []
        if self.previous:
            aliases.append(SimpleNamespace(alias_name="entries", collection_name=self.previous))
        return SimpleNamespace(aliases=aliases)

    def create_collection(self, collection_name: str, **_: object) -> None:
        self.collections.add(collection_name)

    def create_payload_index(self, **_: object) -> None:
        pass

    def upsert(self, *, points: list[object], **_: object) -> None:
        self.points.extend(points)

    def count(self, **_: object):
        return SimpleNamespace(count=len(self.points) + self.count_offset)

    def update_collection_aliases(self, change_aliases_operations: list[object]) -> None:
        self.alias_operations = change_aliases_operations

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)
        self.collections.remove(name)


def settings(vault: Path) -> Settings:
    return Settings("http://qdrant:6333", vault, vault / "models", "entries", "127.0.0.1", 8080)


def test_rebuild_drops_the_collection_the_alias_left_behind() -> None:
    vault = Path(__file__).parent / "fixtures" / "vault"
    client = Client()
    client.collections.add("entries__old")

    IndexRebuilder(settings(vault), client=client, encoders=Encoders()).rebuild()

    assert client.deleted == ["entries__old"]
    assert "entries__old" not in client.collections


def test_first_rebuild_has_no_previous_collection_to_drop() -> None:
    vault = Path(__file__).parent / "fixtures" / "vault"
    client = Client(previous=None)

    IndexRebuilder(settings(vault), client=client, encoders=Encoders()).rebuild()

    assert client.deleted == []


def test_rebuild_validates_points_then_atomically_switches_alias() -> None:
    vault = Path(__file__).parent / "fixtures" / "vault"
    client = Client()
    report = IndexRebuilder(settings(vault), client=client, encoders=Encoders()).rebuild()

    assert report.points == len(client.points) > 0
    assert report.previous_collection == "entries__old"
    assert len(client.alias_operations) == 2
    payload = client.points[0].payload
    # One digest now stands in for the parser, model, encoder, tokenizer, and
    # embedding-input versions. Assert it is a real digest and that it tells
    # two different entries apart, which is all `_embedding_compatible` asks.
    fingerprints = {point.payload["embedding_fingerprint"] for point in client.points}
    assert all(len(value) == 64 for value in fingerprints)
    assert len(fingerprints) > 1
    assert payload["start_line"] >= 1


def test_failed_validation_preserves_alias_and_deletes_temporary_collection() -> None:
    vault = Path(__file__).parent / "fixtures" / "vault"
    client = Client(count_offset=-1)
    with pytest.raises(RuntimeError, match="validation failed"):
        IndexRebuilder(settings(vault), client=client, encoders=Encoders()).rebuild()

    assert client.alias_operations is None
    assert len(client.deleted) == 1
