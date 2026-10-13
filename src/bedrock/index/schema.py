"""Qdrant collection schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from qdrant_client import models

from bedrock.core.config import Settings
from bedrock.index.qdrant import alias_target

DENSE_VECTOR = "semantic"
SPARSE_VECTOR = "text-bm25"

PAYLOAD_INDEXES: tuple[tuple[str, models.PayloadSchemaType], ...] = (
    ("vault_id", models.PayloadSchemaType.KEYWORD),
    ("note_id", models.PayloadSchemaType.KEYWORD),
    ("entry_id", models.PayloadSchemaType.KEYWORD),
    ("path", models.PayloadSchemaType.KEYWORD),
    ("note_title", models.PayloadSchemaType.KEYWORD),
    ("note_type", models.PayloadSchemaType.KEYWORD),
    ("entry_type", models.PayloadSchemaType.KEYWORD),
    ("heading", models.PayloadSchemaType.KEYWORD),
    ("heading_path", models.PayloadSchemaType.KEYWORD),
    ("note_date", models.PayloadSchemaType.DATETIME),
    ("entry_date", models.PayloadSchemaType.DATETIME),
    ("entry_date_end", models.PayloadSchemaType.DATETIME),
    ("tags", models.PayloadSchemaType.KEYWORD),
    ("outgoing_note_ids", models.PayloadSchemaType.KEYWORD),
    ("note_aliases", models.PayloadSchemaType.KEYWORD),
    ("outgoing_link_keys", models.PayloadSchemaType.KEYWORD),
    ("text", models.PayloadSchemaType.TEXT),
    ("board.column", models.PayloadSchemaType.KEYWORD),
    ("board.status", models.PayloadSchemaType.KEYWORD),
    ("board.checked", models.PayloadSchemaType.BOOL),
    ("modified_at", models.PayloadSchemaType.DATETIME),
)


def create_collection(client: object, name: str) -> None:
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(size=384, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
                modifier=models.Modifier.IDF,
            )
        },
    )
    for field_name, field_schema in PAYLOAD_INDEXES:
        client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )


@dataclass(frozen=True, slots=True)
class IndexInitReport:
    collection: str
    physical_collection: str
    created: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def initialize_index(settings: Settings, client: Any) -> IndexInitReport:
    target = alias_target(client, settings.collection_name)
    if target is not None:
        return IndexInitReport(settings.collection_name, target, False)

    physical = f"{settings.collection_name}__init_{uuid4().hex}"
    try:
        create_collection(client, physical)
        client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=physical,
                        alias_name=settings.collection_name,
                    )
                )
            ]
        )
    except Exception:
        if client.collection_exists(physical):
            client.delete_collection(physical)
        raise
    return IndexInitReport(settings.collection_name, physical, True)
