from pathlib import Path
from types import SimpleNamespace

from qdrant_client import models

from bedrock.core.config import Settings
from bedrock.index.schema import PAYLOAD_INDEXES, create_collection, initialize_index


class Client:
    def __init__(self) -> None:
        self.created = None
        self.indexes: list[tuple[str, object]] = []

    def create_collection(self, **kwargs: object) -> None:
        self.created = kwargs

    def create_payload_index(self, **kwargs: object) -> None:
        self.indexes.append((kwargs["field_name"], kwargs["field_schema"]))


def test_collection_has_named_dense_sparse_and_filter_indexes() -> None:
    client = Client()
    create_collection(client, "build")

    dense = client.created["vectors_config"]["semantic"]
    sparse = client.created["sparse_vectors_config"]["text-bm25"]
    assert dense.size == 384
    assert dense.distance is models.Distance.COSINE
    assert sparse.modifier is models.Modifier.IDF
    assert client.indexes == list(PAYLOAD_INDEXES)


class InitClient(Client):
    def __init__(self) -> None:
        super().__init__()
        self.aliases: list[SimpleNamespace] = []
        self.alias_operations: list[object] = []

    def get_aliases(self):
        return SimpleNamespace(aliases=self.aliases)

    def update_collection_aliases(self, *, change_aliases_operations: list[object]) -> None:
        self.alias_operations = change_aliases_operations
        operation = change_aliases_operations[0]
        self.aliases = [
            SimpleNamespace(
                alias_name=operation.create_alias.alias_name,
                collection_name=operation.create_alias.collection_name,
            )
        ]

    def collection_exists(self, _: str) -> bool:
        return False


def test_index_initialization_creates_alias_once(tmp_path: Path) -> None:
    settings = Settings("http://unused", tmp_path, tmp_path, "entries", "127.0.0.1", 8080)
    client = InitClient()

    first = initialize_index(settings, client)
    second = initialize_index(settings, client)

    assert first.created is True
    assert first.physical_collection.startswith("entries__init_")
    assert second.created is False
    assert second.physical_collection == first.physical_collection
    assert len(client.alias_operations) == 1
