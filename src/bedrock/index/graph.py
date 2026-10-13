"""Deterministic, unbounded note-network export from the active index."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from itertools import combinations
from typing import Any

from qdrant_client import QdrantClient, models

from bedrock.core.config import Settings
from bedrock.core.contracts import (
    GraphEdgeType,
    LinkResolution,
    NetworkEdge,
    NetworkGraph,
    NetworkNode,
    NoteType,
)
from bedrock.index.qdrant import scroll_all, vault_condition


class GraphExporter:
    def __init__(
        self,
        settings: Settings,
        *,
        vault_id: str = "personal",
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.vault_id = vault_id
        self.client = client or QdrantClient(url=settings.qdrant_url)

    def export(self) -> NetworkGraph:
        payloads = self._payloads()
        payloads.sort(key=lambda item: (item["path"], item["start_line"], item["entry_id"]))
        nodes = self._nodes(payloads)
        edges = [*self._link_edges(payloads), *self._tag_edges(nodes), *self._date_edges(payloads)]
        edges.sort(key=_edge_sort_key)
        return NetworkGraph(nodes=nodes, edges=tuple(edges))

    def _payloads(self) -> list[dict[str, Any]]:
        filter_ = models.Filter(must=[vault_condition(self.vault_id)])
        records = scroll_all(self.client, self.settings.collection_name, filter_)
        return [record.payload for record in records]

    @staticmethod
    def _nodes(payloads: list[dict[str, Any]]) -> tuple[NetworkNode, ...]:
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for payload in payloads:
            grouped[payload["note_id"]].append(payload)
        nodes = []
        for note_id, entries in grouped.items():
            first = entries[0]
            tags = sorted({tag for entry in entries for tag in entry.get("tags", [])})
            dates = sorted(
                {
                    date
                    for entry in entries
                    for date in (_date(entry.get("note_date")), _date(entry.get("entry_date")))
                    if date is not None
                }
            )
            nodes.append(
                NetworkNode(
                    note_id=note_id,
                    path=first["path"],
                    title=first["note_title"],
                    note_type=NoteType(first["note_type"]),
                    tags=tuple(tags),
                    dates=tuple(dates),
                )
            )
        nodes.sort(key=lambda node: (node.path, node.note_id))
        return tuple(nodes)

    @staticmethod
    def _link_edges(payloads: list[dict[str, Any]]) -> list[NetworkEdge]:
        edges: list[NetworkEdge] = []
        for payload in payloads:
            for ordinal, link in enumerate(payload.get("outgoing_links", [])):
                if (
                    link.get("resolution") != LinkResolution.RESOLVED.value
                    or link.get("target_note_id") is None
                ):
                    continue
                parts = (
                    payload["entry_id"],
                    str(ordinal),
                    link["target_note_id"],
                    str(link.get("line")),
                )
                edges.append(
                    NetworkEdge(
                        edge_id=_edge_id(GraphEdgeType.INTERNAL_LINK, *parts),
                        source_note_id=payload["note_id"],
                        target_note_id=link["target_note_id"],
                        edge_type=GraphEdgeType.INTERNAL_LINK,
                        source_entry_id=payload["entry_id"],
                        line=link.get("line"),
                        target_heading=link.get("target_heading"),
                    )
                )
        return edges

    @staticmethod
    def _tag_edges(nodes: tuple[NetworkNode, ...]) -> list[NetworkEdge]:
        edges: list[NetworkEdge] = []
        for left, right in combinations(sorted(nodes, key=lambda node: node.note_id), 2):
            shared = tuple(sorted(set(left.tags) & set(right.tags)))
            if shared:
                edges.append(
                    NetworkEdge(
                        edge_id=_edge_id(GraphEdgeType.SHARED_TAG, left.note_id, right.note_id, *shared),
                        source_note_id=left.note_id,
                        target_note_id=right.note_id,
                        edge_type=GraphEdgeType.SHARED_TAG,
                        tags=shared,
                    )
                )
        return edges

    @staticmethod
    def _date_edges(payloads: list[dict[str, Any]]) -> list[NetworkEdge]:
        grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for payload in payloads:
            for field in ("note_date", "entry_date"):
                value = _date(payload.get(field))
                if value is not None:
                    grouped[(field, value)].append(payload)

        edges: list[NetworkEdge] = []
        for (field, value), entries in sorted(grouped.items()):
            ordered = sorted(entries, key=lambda item: (item["note_id"], item["entry_id"]))
            for left, right in combinations(ordered, 2):
                if left["note_id"] == right["note_id"]:
                    continue
                source, target = left, right
                parts = (field, value, source["entry_id"], target["entry_id"])
                edges.append(
                    NetworkEdge(
                        edge_id=_edge_id(GraphEdgeType.SHARED_DATE, *parts),
                        source_note_id=source["note_id"],
                        target_note_id=target["note_id"],
                        edge_type=GraphEdgeType.SHARED_DATE,
                        source_entry_id=source["entry_id"],
                        target_entry_id=target["entry_id"],
                        date=value,
                        date_field=field,
                    )
                )
        return edges


def _date(value: str | None) -> str | None:
    return value.split("T", 1)[0] if value else None


def _edge_id(edge_type: GraphEdgeType, *parts: str) -> str:
    material = "\x1f".join((edge_type.value, *parts))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _edge_sort_key(edge: NetworkEdge) -> tuple[object, ...]:
    return (
        edge.edge_type.value,
        edge.source_note_id,
        edge.target_note_id,
        edge.source_entry_id or "",
        edge.target_entry_id or "",
        edge.line or 0,
        edge.date_field or "",
        edge.date or "",
        edge.edge_id,
    )
