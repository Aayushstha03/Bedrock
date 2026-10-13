import pytest

from bedrock.core.contracts import (
    EntryType,
    Link,
    LinkResolution,
    NoteType,
    QdrantPayload,
    QueryPlan,
    SourceRange,
)


def test_source_range_is_positive_and_inclusive() -> None:
    assert SourceRange(3, 5).end_line == 5
    with pytest.raises(ValueError):
        SourceRange(0, 1)
    with pytest.raises(ValueError):
        SourceRange(5, 4)


def test_query_plan_enforces_traversal_budget() -> None:
    assert QueryPlan(link_depth=2, max_linked_notes=5).link_depth == 2
    with pytest.raises(ValueError, match="link_depth"):
        QueryPlan(link_depth=3)
    with pytest.raises(ValueError, match="max_linked_notes"):
        QueryPlan(max_linked_notes=6)


def test_payload_serializes_enums_tuples_and_nested_contracts() -> None:
    payload = QdrantPayload(
        vault_id="personal",
        note_id="note",
        entry_id="entry",
        path="projects/News Resolution.md",
        note_title="News Resolution",
        note_type=NoteType.STANDARD,
        entry_type=EntryType.PROJECT_UPDATE,
        text="status loops",
        heading="2026-08-19",
        heading_path=("News Resolution", "2026-08-19"),
        start_line=3,
        end_line=4,
        note_date=None,
        entry_date="2026-08-19",
        tags=("work",),
        outgoing_note_ids=("target",),
        outgoing_links=(Link("Target", "target", resolution=LinkResolution.RESOLVED),),
        web_links=(),
        content_hash="sha256",
        modified_at="2026-08-20T00:00:00+00:00",
        embedding_fingerprint="deadbeef",
    ).to_dict()

    assert payload["note_type"] == "standard"
    assert payload["heading_path"] == ["News Resolution", "2026-08-19"]
    assert payload["outgoing_links"][0]["resolution"] == "resolved"
