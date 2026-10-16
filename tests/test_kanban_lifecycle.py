"""Lifecycle blocks, indexed payloads, and public task filters agree."""

import warnings
from pathlib import Path

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from qdrant_client import QdrantClient
from test_retrieval import HashEncoders

from bedrock.agent import tools as agent_tools
from bedrock.agent.sources import Sources
from bedrock.core.config import Settings
from bedrock.core.contracts import QueryPlan, WarningType
from bedrock.index.incremental import IncrementalIndexer
from bedrock.index.rebuild import IndexRebuilder
from bedrock.library import Bedrock
from bedrock.parser.service import VaultParser
from bedrock.vault.service import VaultTools


def write_board(tmp_path: Path, cards: str) -> tuple[Settings, Path]:
    path = tmp_path / "kanban" / "Lifecycle.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("---\nkanban-plugin: board\n---\n\n## Doing\n\n" + cards)
    return Settings("http://unused", tmp_path, tmp_path / "models", "entries", "localhost", 8080), path


def test_only_named_block_establishes_lifecycle_dates(tmp_path: Path) -> None:
    settings, path = write_board(
        tmp_path,
        """- [ ] New card @{2026-01-01}
  - Nested details #project_idea [[Related]]

  ```bedrock-dates
  created_at: 2026-10-16T00:30:00+05:45
  started_at: 2026-10-16T10:30:00Z
  completed_at:
  due_date: 2026-10-20
  ```
- [ ] Old card created_at: 2020-01-01 @{2026-10-01}
  ```yaml
  due_date: 2026-10-20
  ```
""",
    )
    note = VaultParser(settings).parse_file(path)
    new, old = note.entries
    assert new.board.created_at == "2026-10-15T18:45:00+00:00"
    assert new.board.started_at == "2026-10-16T10:30:00+00:00"
    assert new.board.completed_at is None
    assert new.board.due_date == "2026-10-20"
    assert "Nested details" in new.text
    assert new.tags == ("project_idea",)
    assert new.outgoing_links[0].target_text == "Related"
    assert "created_at" not in new.text
    assert "due_date" not in new.embedding_text
    assert new.entry_date == "2026-01-01"
    assert old.entry_date == "2026-10-01"
    assert old.board.created_at is None and old.board.due_date is None
    assert "due_date" in old.text  # Ordinary code remains searchable.


@pytest.mark.parametrize(
    "content",
    [
        "created_at: 2026-10-16",  # No timezone-aware timestamp.
        "due_date: 2026-02-30",
        "due_date: 2026-10-20\n  due_date: 2026-10-21",
        "due_Date: 2026-10-20",  # Exact, explicit keys only.
    ],
)
def test_bad_fields_warn_and_do_not_establish_dates(tmp_path: Path, content: str) -> None:
    settings, path = write_board(tmp_path, f"- [ ] Card\n  ```bedrock-dates\n  {content}\n  ```\n")
    note = VaultParser(settings).parse_file(path)
    assert any(w.kind is WarningType.INVALID_KANBAN_DATES for w in note.warnings)
    assert note.entries[0].board.created_at is None
    assert note.entries[0].board.due_date is None
    assert note.entries[0].text == "Card"


def test_multiple_blocks_are_ambiguous_and_unclosed_blocks_are_rejected(tmp_path: Path) -> None:
    block = "  ```bedrock-dates\n  due_date: 2026-10-20\n  ```\n"
    for content in (block + block, block.rsplit("  ```", 1)[0]):
        settings, path = write_board(tmp_path, "- [ ] Card\n" + content)
        note = VaultParser(settings).parse_file(path)
        assert note.entries[0].board.due_date is None
        assert any(w.kind is WarningType.INVALID_KANBAN_DATES for w in note.warnings)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"created_at_from": "2026-02-30"},
        {"started_at_to": "2026-10-16T10:00:00"},
        {"started_at_to": "2026-10-16T10:00:00+01:99"},
        {"completed_at_from": "2026-10-17", "completed_at_to": "2026-10-16"},
        {"due_date_from": "2026-10-16T10:00:00Z"},
    ],
)
def test_invalid_search_bounds_fail_before_querying(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        QueryPlan(**kwargs)


@pytest.fixture
def indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings, path = write_board(
        tmp_path,
        """- [ ] Active
  ```bedrock-dates
  created_at: 2026-10-16T23:59:59.999999Z
  started_at: 2026-10-17T00:30:00+05:45
  due_date: 2026-10-20
  ```
- [x] Finished
  ```bedrock-dates
  created_at: 2026-10-01T08:00:00Z
  completed_at: 2026-10-16T10:00:00Z
  due_date: 2026-10-19
  ```
- [ ] Legacy @{2026-10-16}
""",
    )
    client = QdrantClient(":memory:")
    encoders = HashEncoders()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        IndexRebuilder(settings, client=client, encoders=encoders).rebuild()
    vault = Bedrock(settings, client=client, encoders=encoders)
    # Embedded Qdrant cannot report payload indexes; bypass readiness only.
    monkeypatch.setattr(VaultTools, "_require_index", lambda self: None)
    yield vault, path
    client.close()


def test_lifecycle_filters_and_metadata_on_every_task_surface(indexed) -> None:
    vault, _ = indexed
    tools = VaultTools(vault)
    page = tools.find_tasks(
        created_at_from="2026-10-16",
        created_at_to="2026-10-16",
        started_at_from="2026-10-16",
        due_date_from="2026-10-20",
        due_date_to="2026-10-20",
    )
    assert [item.text for item in page.items] == ["Active"]
    assert page.items[0].task.due_date == "2026-10-20"
    assert page.items[0].task.created_at == "2026-10-16T23:59:59.999999+00:00"
    assert [item.text for item in tools.find_tasks(completed_at_from="2026-10-16").items] == ["Finished"]
    assert tools.find_tasks(started_at_from="2026-10-17").items == []  # Stored instants, not local dates.
    assert [item.text for item in tools.find_tasks(date_from="2026-10-16").items] == ["Legacy"]
    assert tools.find_tasks(completed_at_from="2026-10-16", due_date_from="2026-10-20").items == []
    assert [item.text for item in vault.find_kanban_cards(due_date_to="2026-10-19")] == ["Finished"]
    ctx = RunContext(deps=Sources(vault), model=TestModel(), usage=RunUsage())
    result = agent_tools.find_tasks(ctx, due_date_from="2026-10-20", due_date_to="2026-10-20")
    assert len(result["items"]) == 1
    assert result["items"][0]["task"]["due_date"] == "2026-10-20"
    records, _ = vault.client.scroll("entries", limit=20)
    active = next(record.payload for record in records if record.payload["text"] == "Active")
    assert active["board"]["due_date"] == "2026-10-20T00:00:00Z"
    assert active["board"]["created_at"] == "2026-10-16T23:59:59.999999+00:00"


def test_date_only_edit_updates_payload_without_changing_identity_or_prose(indexed) -> None:
    vault, path = indexed
    original = vault.find_kanban_cards(due_date_from="2026-10-20")[0]
    path.write_text(path.read_text().replace("due_date: 2026-10-20", "due_date: 2026-10-22"))
    report = IncrementalIndexer(vault.settings, client=vault.client, encoders=vault.encoders).reindex(
        ["kanban/Lifecycle.md"]
    )
    assert report.embeddings_generated == 0
    assert not vault.find_kanban_cards(due_date_to="2026-10-20", started_at_from="2026-10-01")
    updated = vault.find_kanban_cards(due_date_from="2026-10-22")[0]
    assert updated.entry_id == original.entry_id
    assert updated.text == original.text
    assert updated.board.due_date == "2026-10-22"
