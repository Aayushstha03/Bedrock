"""Regression coverage for the standalone, agent-independent tool contract."""

import asyncio

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from test_agent_tools import ctx  # noqa: F401 - shared hermetic indexed-vault fixture

from bedrock.index import diagnostics
from bedrock.vault import service as service_module
from bedrock.vault.mcp import build_server
from bedrock.vault.service import TOOL_NAMES, IndexUnavailable, VaultTools


@pytest.fixture
def tools(ctx, monkeypatch):  # noqa: F811 - pytest fixture injection
    # Qdrant's embedded test engine does not implement payload indexes. Keep
    # alias/vector checks real, discount only that known backend limitation.
    def inspect(settings, client):
        report = diagnostics.inspect_collection(settings, client)
        if "errors" in report:
            report["errors"] = [
                e for e in report["errors"] if not e.startswith("payload indexes are missing")
            ]
            report["ready"] = not report["errors"]
        return report

    monkeypatch.setattr(service_module, "inspect_collection", inspect)
    yield VaultTools(ctx.deps.vault)
    ctx.deps.vault.client.close()


def test_search_can_scope_a_note_section_and_entry_type(tools):
    page = tools.search_notes(
        "pipeline",
        mode="literal",
        note="projects/News Resolution.md",
        section="2026-08-19",
        entry_types=["project_update"],
    )
    assert page.items
    assert all(
        item.path == "projects/News Resolution.md"
        and item.entry_date == "2026-08-19"
        and item.exact_match_verified
        for item in page.items
    )
    assert all(item.entry_id and item.note_id and item.start_line <= item.end_line for item in page.items)
    assert tools.read_note(page.items[0].note_id).path == page.items[0].path
    assert (
        tools.expand_context(page.items[0].entry_id, link_depth=0).items[0].entry_id == page.items[0].entry_id
    )


def test_read_every_character_and_detect_edits_between_pages(tools):
    path = tools.vault.settings.vault_path / "Large.md"
    original = "# Large\n" + "x" * 30000 + "\n" + "tail\n" * 250
    path.write_text(original)
    chunks = []
    args = {"note": "Large", "line_count": 20}
    for _ in range(30):
        result = tools.read_note(**args)
        assert len(result.content) <= 12000
        chunks.append(result.content)
        if not result.has_more:
            break
        args.update(
            start_line=result.next_line,
            char_offset=result.next_char_offset or 0,
            expected_hash=result.content_hash,
        )
    else:
        pytest.fail("read pagination failed to terminate")
    assert "".join(chunks) == original
    path.write_text(original + "edit")
    with pytest.raises(ValueError, match="changed between pages"):
        tools.read_note("Large", expected_hash=result.content_hash)


def test_empty_note_and_out_of_range_read(tools):
    (tools.vault.settings.vault_path / "Empty.md").write_text("")
    result = tools.read_note("Empty")
    assert result.content == "" and result.end_line is None and not result.has_more
    with pytest.raises(ValueError, match="exceeds"):
        tools.read_note("Empty", start_line=2)


def test_board_path_isolation_and_empty_structure(tools):
    root = tools.vault.settings.vault_path
    (root / "kanban/Other.md").write_text(
        "---\nkanban-plugin: board\n---\n# Kitchen App\n## ToDo\n- [ ] OTHER_BOARD\n## Empty Column\n"
    )
    (root / "kanban/Empty.md").write_text("---\nkanban-plugin: board\n---\n# Empty\n## Backlog\n")
    tools.vault.reindex(full=True)
    boards = tools.list_boards().items
    assert next(b for b in boards if b.path == "kanban/Empty.md").columns == ["Backlog"]
    assert "Empty Column" in next(b for b in boards if b.path == "kanban/Other.md").columns
    for page in (
        tools.find_tasks(board="kanban/Kitchen App.md"),
        tools.vault.find_kanban_cards(board="kanban/Kitchen App.md"),
    ):
        cards = page.items if hasattr(page, "items") else page
        assert cards and all(c.path == "kanban/Kitchen App.md" for c in cards)
    other = next(b for b in boards if b.path == "kanban/Other.md")
    assert tools.find_tasks(board=other.note_id, query="OTHER_BOARD", mode="literal").items
    assert not tools.find_tasks(board="kanban/Empty.md", columns=["Backlog"]).items
    with pytest.raises(ValueError, match="ambiguous"):
        tools.find_tasks(board="Kitchen App")


def test_catalog_filters_before_paging_and_keeps_empty_notes(tools):
    (tools.vault.settings.vault_path / "New Empty.md").write_text("# New Empty\n")
    page = tools.list_notes(query="new empty", limit=1)
    assert len(page.items) == 1 and page.items[0].indexed_entries == 0
    page = tools.list_notes(path_prefix="projects/", limit=1)
    assert page.items[0].path.startswith("projects/")
    assert page.has_more
    second = tools.list_notes(path_prefix="projects/", limit=1, offset=page.next_offset)
    assert second.items[0].note_id != page.items[0].note_id


@pytest.mark.parametrize(
    "args",
    [
        {"mode": "literal"},
        {"query": "   "},
        {"mode": "metadata", "query": "text"},
        {"date_from": "2026-02-30"},
        {"date_from": "2026-09-01", "date_to": "2026-01-01"},
        {"limit": 51},
        {"offset": -1},
    ],
)
def test_invalid_search_is_actionable(tools, args):
    with pytest.raises(ValueError):
        tools.search_notes(**args)


def test_status_detects_changes_and_index_failure_is_not_empty_results(tools):
    assert tools.vault_status().freshness == "current"
    path = tools.vault.settings.vault_path / "projects/News Resolution.md"
    path.write_text(path.read_text() + "\n## New section\nA new decision.\n")
    status = tools.vault_status()
    assert status.freshness == "stale" and status.pending_entries > 0
    tools.vault.reindex()
    assert tools.vault_status().freshness == "current"
    physical = tools.vault_status().index["physical_collection"]
    tools.vault.client.delete_collection(physical)
    assert not tools.vault_status().ready
    with pytest.raises(IndexUnavailable):
        tools.search_notes("anything")
    # Original Markdown stays readable even when derived storage is down.
    assert tools.read_note("projects/News Resolution.md").content


def test_all_tools_through_real_mcp_session_and_error_results(tools):
    async def exercise():
        async with create_connected_server_and_client_session(build_server(tools)) as client:
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == set(TOOL_NAMES)
            assert all(tool.annotations.readOnlyHint and tool.outputSchema for tool in listed.tools)
            assert "ctx" not in str([tool.inputSchema for tool in listed.tools])
            page = await client.call_tool("search_notes", {"query": "pipeline", "mode": "literal"})
            assert not page.isError
            source = page.structuredContent["items"][0]
            requests = {
                "list_tags": {},
                "list_notes": {"query": "News"},
                "read_note": {"note": source["note_id"]},
                "find_backlinks": {"note": source["path"]},
                "find_tasks": {},
                "expand_context": {"entry_id": source["entry_id"], "max_items": 2},
                "list_boards": {},
                "vault_status": {},
            }
            for name, arguments in requests.items():
                result = await client.call_tool(name, arguments)
                assert not result.isError, (name, result)
                assert isinstance(result.structuredContent, dict)
            missing = await client.call_tool("read_note", {"note": "No Such Note"})
            assert missing.isError and "not found" in missing.content[0].text
            invalid = await client.call_tool("search_notes", {"limit": 999})
            assert invalid.isError
            # Independent client session can use canonical IDs from the first.
        async with create_connected_server_and_client_session(build_server(tools)) as client:
            result = await client.call_tool("expand_context", {"entry_id": source["entry_id"]})
            assert not result.isError

    asyncio.run(exercise())


def test_backlink_candidate_discovery_includes_more_than_first_page(tools):
    root = tools.vault.settings.vault_path
    (root / "Target.md").write_text("# Target\n\n## Detail\nTarget evidence.\n")
    for index in range(15):
        (root / f"Backlink{index:02}.md").write_text(f"# Backlink {index}\n\nSee [[Target]].\n")
    tools.vault.reindex(full=True)
    target = tools.search_notes("Target evidence", mode="literal", note="Target").items[0]
    backlinks = tools.vault.retriever.get_backlinks(target.note_id)
    assert len(backlinks) == 15
    first = tools.find_backlinks(target.note_id, limit=10)
    second = tools.find_backlinks(target.note_id, limit=10, offset=first.next_offset)
    assert len(first.items) == 10 and len(second.items) == 5 and not second.has_more
    context = tools.expand_context(target.entry_id, max_notes=5)
    assert context.truncated and "max_notes" in context.budget_reasons


def test_missing_qdrant_becomes_index_unavailable(tools, monkeypatch):
    def unavailable(*args):
        raise ConnectionError("Qdrant is down")

    monkeypatch.setattr(service_module, "inspect_collection", unavailable)
    with pytest.raises(IndexUnavailable, match="Qdrant is down"):
        tools.search_notes("decision")
    assert tools.vault_status().freshness == "unknown"


def test_stdio_cli_protocol_and_complete_read(tools):
    import os
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def exercise():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "bedrock.runtime.cli", "mcp"],
            env={**os.environ, "BEDROCK_VAULT_PATH": str(tools.vault.settings.vault_path)},
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
            await client.initialize()
            assert {tool.name for tool in (await client.list_tools()).tools} == set(TOOL_NAMES)
            result = await client.call_tool("read_note", {"note": "News Resolution", "line_count": 2})
            assert not result.isError
            assert result.structuredContent["path"] == "projects/News Resolution.md"
            assert result.structuredContent["has_more"]

    asyncio.run(exercise())


def test_context_diagnostics_are_bounded_and_disclose_omissions(tools):
    path = tools.vault.settings.vault_path / "Unresolved.md"
    path.write_text("# Unresolved\n\n## Links\n" + " ".join(f"[[Missing{index}]]" for index in range(25)))
    tools.vault.reindex(full=True)
    entry = tools.search_notes("Missing0", mode="literal", note="Unresolved").items[0]
    result = tools.expand_context(entry.entry_id)
    assert len(result.unresolved_links) == 20
    assert result.truncated and "max_diagnostics" in result.budget_reasons


def test_tag_registry_uses_active_settings_without_index():
    from dataclasses import replace
    from types import SimpleNamespace

    from bedrock.core.config import Settings

    settings = Settings.from_env()
    settings = replace(
        settings, known_tags=("project_idea", "journal", "custom"), tag_aliases=(("journaling", "journal"),)
    )
    service = VaultTools(SimpleNamespace(settings=settings))
    tags = service.list_tags().items
    assert [tag.name for tag in tags] == ["project_idea", "journal", "custom"]
    assert tags[0].description and "scoping" in tags[0].description
    assert tags[1].aliases == ["journaling"]
    assert tags[2].description is None
