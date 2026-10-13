from __future__ import annotations

from types import SimpleNamespace

from bedrock.core.config import Settings
from bedrock.core.contracts import GraphEdgeType
from bedrock.index.graph import GraphExporter


class GraphClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = payloads

    def scroll(self, **_: object):
        return [SimpleNamespace(payload=payload) for payload in self.payloads], None


def payload(
    note: str,
    entry: str,
    *,
    tags: list[str] | None = None,
    note_date: str | None = None,
    entry_date: str | None = None,
    links: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "vault_id": "personal",
        "note_id": note,
        "entry_id": entry,
        "path": f"{note}.md",
        "note_title": note.title(),
        "note_type": "standard",
        "start_line": 1,
        "tags": tags or [],
        "note_date": note_date,
        "entry_date": entry_date,
        "outgoing_links": links or [],
    }


def settings(tmp_path):
    return Settings("http://unused", tmp_path, tmp_path, "entries", "127.0.0.1", 8080)


def test_graph_export_is_complete_evidenced_unbounded_and_deterministic(tmp_path) -> None:
    resolved_links = [
        {
            "target_note_id": f"target-{number}",
            "target_heading": "Details",
            "resolution": "resolved",
            "line": number + 2,
        }
        for number in range(7)
    ]
    resolved_links.append(
        {
            "target_note_id": None,
            "target_heading": None,
            "resolution": "missing",
            "line": 20,
        }
    )
    payloads = [
        payload(
            "hub",
            "hub-entry",
            tags=["shared", "hub-only"],
            note_date="2026-08-20T00:00:00Z",
            entry_date="2026-08-18T00:00:00Z",
            links=resolved_links,
        ),
        payload(
            "isolated",
            "isolated-entry",
            note_date="2026-08-21T00:00:00Z",
        ),
    ]
    payloads.extend(
        payload(
            f"target-{number}",
            f"target-entry-{number}",
            tags=["shared"] if number == 0 else [],
            note_date="2026-08-20T00:00:00Z" if number == 0 else None,
            entry_date="2026-08-18T00:00:00Z" if number == 1 else None,
        )
        for number in range(7)
    )

    exporter = GraphExporter(settings(tmp_path), client=GraphClient(list(reversed(payloads))))
    first = exporter.export()
    second = exporter.export()

    assert first.to_dict() == second.to_dict()
    assert len(first.nodes) == 9
    assert any(node.note_id == "isolated" for node in first.nodes)

    links = [edge for edge in first.edges if edge.edge_type is GraphEdgeType.INTERNAL_LINK]
    assert len(links) == 7  # graph export does not inherit the five-note context budget
    assert all(edge.source_entry_id == "hub-entry" and edge.line is not None for edge in links)

    tag_edges = [edge for edge in first.edges if edge.edge_type is GraphEdgeType.SHARED_TAG]
    assert len(tag_edges) == 1
    assert tag_edges[0].tags == ("shared",)

    date_edges = [edge for edge in first.edges if edge.edge_type is GraphEdgeType.SHARED_DATE]
    assert {(edge.date, edge.date_field) for edge in date_edges} == {
        ("2026-08-20", "note_date"),
        ("2026-08-18", "entry_date"),
    }
    assert not any({edge.source_note_id, edge.target_note_id} == {"hub", "isolated"} for edge in date_edges)
