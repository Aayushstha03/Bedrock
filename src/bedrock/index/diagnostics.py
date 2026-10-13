"""Machine-readable collection integrity diagnostics."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from bedrock.core.config import Settings
from bedrock.index.qdrant import alias_target
from bedrock.index.schema import DENSE_VECTOR, PAYLOAD_INDEXES, SPARSE_VECTOR


def inspect_collection(settings: Settings, client: Any | None = None) -> dict[str, object]:
    client = client or QdrantClient(url=settings.qdrant_url)
    target = alias_target(client, settings.collection_name)
    if target is None:
        return {
            "ready": False,
            "collection": settings.collection_name,
            "reason": "active collection alias does not exist",
        }

    info = client.get_collection(target)
    vectors = info.config.params.vectors
    sparse_vectors = info.config.params.sparse_vectors or {}
    payload_schema = info.payload_schema
    dense = vectors.get(DENSE_VECTOR) if isinstance(vectors, dict) else None
    missing_indexes = [field for field, _ in PAYLOAD_INDEXES if field not in payload_schema]
    errors: list[str] = []
    if dense is None or dense.size != 384 or str(dense.distance).casefold().split(".")[-1] != "cosine":
        errors.append("semantic vector schema is incompatible")
    if SPARSE_VECTOR not in sparse_vectors:
        errors.append("text-bm25 sparse vector is missing")
    if missing_indexes:
        errors.append(f"payload indexes are missing: {', '.join(missing_indexes)}")
    return {
        "ready": not errors,
        "collection": settings.collection_name,
        "physical_collection": target,
        "points": info.points_count,
        "errors": errors,
    }
