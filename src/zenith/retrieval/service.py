"""Metadata, literal, lexical, semantic, and hybrid retrieval over the active index."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from zenith.core.config import Settings
from zenith.core.contracts import (
    QUERY_TEXT_FIELDS,
    EntryType,
    KanbanData,
    Link,
    LinkResolution,
    NoteType,
    Page,
    QueryPlan,
    ResultOrder,
    RetrievalMode,
    SearchResult,
)
from zenith.index.encoders import LocalEncoders
from zenith.index.qdrant import scroll_all, vault_condition
from zenith.index.schema import DENSE_VECTOR, SPARSE_VECTOR
from zenith.retrieval.filters import build_filter
from zenith.retrieval.literal import verify_literal


class Retriever:
    def __init__(
        self,
        settings: Settings,
        *,
        vault_id: str = "personal",
        client: Any | None = None,
        encoders: Any | None = None,
    ) -> None:
        self.settings = settings
        self.vault_id = vault_id
        self.client = client or QdrantClient(url=settings.qdrant_url)
        self._encoders = encoders

    @property
    def encoders(self) -> Any:
        if self._encoders is None:
            self._encoders = LocalEncoders(self.settings)
        return self._encoders

    def search(self, plan: QueryPlan) -> tuple[SearchResult, ...]:
        return self.search_page(plan).items

    def search_page(self, plan: QueryPlan) -> Page[SearchResult]:
        filter_ = build_filter(plan, self.vault_id)
        if plan.mode is RetrievalMode.METADATA:
            return self._metadata(filter_, plan)
        if plan.mode is RetrievalMode.LITERAL:
            return self._literal(filter_, plan)
        if plan.mode is RetrievalMode.LEXICAL:
            return self._lexical(filter_, plan)
        if plan.mode is RetrievalMode.SEMANTIC:
            return self._semantic(filter_, plan)
        return self._hybrid(filter_, plan)

    def get_entry(self, entry_id: str) -> SearchResult | None:
        filter_ = models.Filter(
            must=[
                vault_condition(self.vault_id),
                models.FieldCondition(key="entry_id", match=models.MatchValue(value=entry_id)),
            ]
        )
        results = self._ordered(filter_, QueryPlan(mode=RetrievalMode.METADATA, limit=1)).items
        return results[0] if results else None

    def get_note_entries(self, note_id: str) -> tuple[SearchResult, ...]:
        return self.metadata_all(QueryPlan(mode=RetrievalMode.METADATA, note_id=note_id))

    def metadata_all(self, plan: QueryPlan | None = None) -> tuple[SearchResult, ...]:
        plan = plan or QueryPlan(mode=RetrievalMode.METADATA)
        if plan.mode is not RetrievalMode.METADATA:
            raise ValueError("metadata_all requires metadata mode")
        records = self._scroll_all(build_filter(plan, self.vault_id))
        records.sort(key=_by_position)
        if plan.order is ResultOrder.DATE_DESC:
            records.sort(key=_effective_date, reverse=True)
        return tuple(_result(record.payload, RetrievalMode.METADATA) for record in records)

    def get_backlinks(self, note_id: str) -> tuple[SearchResult, ...]:
        return self.get_backlinks_page(note_id).items

    def get_backlinks_page(self, note_id: str, *, limit: int = 10, offset: int = 0) -> Page[SearchResult]:
        filter_ = models.Filter(
            must=[
                vault_condition(self.vault_id),
                models.FieldCondition(key="outgoing_note_ids", match=models.MatchValue(value=note_id)),
            ]
        )
        return self._ordered(filter_, QueryPlan(mode=RetrievalMode.METADATA, limit=limit, offset=offset))

    def search_within(
        self,
        note_id: str,
        query: str,
        *,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        section: str | None = None,
        limit: int = 1,
    ) -> tuple[SearchResult, ...]:
        texts = dict.fromkeys(QUERY_TEXT_FIELDS[mode], query)
        return self.search(QueryPlan(mode=mode, note_id=note_id, section=section, limit=limit, **texts))

    def _ordered(self, filter_: models.Filter, plan: QueryPlan | None = None) -> Page[SearchResult]:
        """Read every matching record, in stable document order."""
        records = self._scroll_all(filter_)
        records.sort(key=_by_position)
        if plan is None:
            return _page(records, QueryPlan(mode=RetrievalMode.METADATA), RetrievalMode.METADATA)
        if plan.order is ResultOrder.DATE_DESC:
            # A stable second sort keeps path/line ordering for equal dates and
            # puts entries carrying no chronology after every dated entry.
            records.sort(key=_effective_date, reverse=True)
        return _page(records, plan, RetrievalMode.METADATA)

    def _metadata(self, filter_: models.Filter, plan: QueryPlan) -> Page[SearchResult]:
        return self._ordered(filter_, plan)

    def _literal(self, filter_: models.Filter, plan: QueryPlan) -> Page[SearchResult]:
        if not plan.literal_text:
            raise ValueError("literal mode requires literal_text")
        candidates = self._scroll_all(filter_)
        verified = [
            record for record in candidates if verify_literal(plan.literal_text, record.payload["text"])
        ]
        verified.sort(key=_by_position)
        return _page(verified, plan, plan.mode, verified=True)

    def _lexical(self, filter_: models.Filter, plan: QueryPlan) -> Page[SearchResult]:
        if not plan.lexical_text:
            raise ValueError("lexical mode requires lexical_text")
        sparse_vector = self.encoders.encode_sparse([plan.lexical_text])[0]
        return self._vector_search(sparse_vector, SPARSE_VECTOR, filter_, plan)

    def _semantic(self, filter_: models.Filter, plan: QueryPlan) -> Page[SearchResult]:
        if not plan.semantic_text:
            raise ValueError("semantic mode requires semantic_text")
        dense_vector = self.encoders.encode_dense([plan.semantic_text])[0]
        return self._vector_search(dense_vector, DENSE_VECTOR, filter_, plan)

    def _vector_search(
        self, vector: Any, using: str, filter_: models.Filter, plan: QueryPlan
    ) -> Page[SearchResult]:
        response = self.client.query_points(
            collection_name=self.settings.collection_name,
            query=vector,
            using=using,
            query_filter=filter_,
            limit=plan.limit + 1,
            offset=plan.offset,
            with_payload=True,
        )
        return _vector_page(response.points, plan)

    def _hybrid(self, filter_: models.Filter, plan: QueryPlan) -> Page[SearchResult]:
        if not plan.lexical_text or not plan.semantic_text:
            raise ValueError("hybrid mode requires both lexical_text and semantic_text")
        dense_vector = self.encoders.encode_dense([plan.semantic_text])[0]
        sparse_vector = self.encoders.encode_sparse([plan.lexical_text])[0]
        requested = plan.offset + plan.limit + 1
        prefetch_limit = max(requested * 4, requested)
        response = self.client.query_points(
            collection_name=self.settings.collection_name,
            prefetch=[
                models.Prefetch(query=dense_vector, using=DENSE_VECTOR, filter=filter_, limit=prefetch_limit),
                models.Prefetch(
                    query=sparse_vector, using=SPARSE_VECTOR, filter=filter_, limit=prefetch_limit
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=filter_,
            limit=plan.limit + 1,
            offset=plan.offset,
            with_payload=True,
        )
        return _vector_page(response.points, plan)

    def _scroll_all(self, filter_: models.Filter) -> list[Any]:
        return scroll_all(self.client, self.settings.collection_name, filter_)


def _by_position(record: Any) -> tuple[str, int]:
    return (record.payload["path"], record.payload["start_line"])


def _effective_date(record: Any) -> str:
    return record.payload.get("entry_date") or record.payload.get("note_date") or ""


def _page(
    records: list[Any], plan: QueryPlan, mode: RetrievalMode, *, verified: bool | None = None
) -> Page[SearchResult]:
    start = plan.offset
    selected = records[start : start + plan.limit + 1]
    has_more = len(selected) > plan.limit
    if has_more:
        selected.pop()
    return Page(
        tuple(_result(record.payload, mode, verified=verified) for record in selected),
        has_more,
        start + plan.limit if has_more else None,
    )


def _vector_page(points: list[Any], plan: QueryPlan) -> Page[SearchResult]:
    has_more = len(points) > plan.limit
    selected = points[: plan.limit]
    return Page(
        tuple(_result(point.payload, plan.mode, score=point.score) for point in selected),
        has_more,
        plan.offset + plan.limit if has_more else None,
    )


def _result(
    payload: dict[str, Any],
    mode: RetrievalMode,
    *,
    score: float | None = None,
    verified: bool | None = None,
) -> SearchResult:
    board = payload.get("board")
    return SearchResult(
        entry_id=payload["entry_id"],
        note_id=payload["note_id"],
        path=payload["path"],
        note_title=payload["note_title"],
        note_type=NoteType(payload["note_type"]),
        entry_type=EntryType(payload["entry_type"]),
        mode=mode,
        text=payload["text"],
        heading=payload["heading"],
        heading_path=tuple(payload["heading_path"]),
        start_line=payload["start_line"],
        end_line=payload["end_line"],
        note_date=_short_date(payload["note_date"]),
        entry_date=_short_date(payload["entry_date"]),
        tags=tuple(payload["tags"]),
        outgoing_links=tuple(_link(link) for link in payload["outgoing_links"]),
        score=score,
        verified=verified,
        board=KanbanData(**board) if board is not None else None,
    )


def _short_date(value: str | None) -> str | None:
    return value.split("T", 1)[0] if value else None


def _link(payload: dict[str, Any]) -> Link:
    return Link(
        target_text=payload["target_text"],
        target_note_id=payload["target_note_id"],
        target_heading=payload["target_heading"],
        alias=payload["alias"],
        resolution=LinkResolution(payload["resolution"]),
        line=payload["line"],
    )
