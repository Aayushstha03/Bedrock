"""Shared access patterns over the Qdrant client."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

SCROLL_PAGE = 256


def vault_condition(vault_id: str) -> models.FieldCondition:
    """Match the one vault every stored point is tagged with."""
    return models.FieldCondition(key="vault_id", match=models.MatchValue(value=vault_id))


def alias_target(client: Any, alias_name: str) -> str | None:
    """Return the physical collection an alias points at, or None."""
    return next(
        (alias.collection_name for alias in client.get_aliases().aliases if alias.alias_name == alias_name),
        None,
    )


def scroll_all(
    client: Any,
    collection: str,
    scroll_filter: models.Filter,
    *,
    with_vectors: bool = False,
) -> list[Any]:
    """Read every record matching one filter, one page at a time."""
    records: list[Any] = []
    offset: object | None = None
    while True:
        page, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=SCROLL_PAGE,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        records.extend(page)
        if offset is None:
            return records
