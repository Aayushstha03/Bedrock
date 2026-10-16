"""Exercise real source mutations with an isolated vault and embedded Qdrant."""

import hashlib
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from test_retrieval import HashEncoders

from bedrock.core.config import Settings
from bedrock.core.contracts import QueryPlan, RetrievalMode
from bedrock.index.rebuild import IndexRebuilder
from bedrock.library import Bedrock
from bedrock.parser.service import VaultParser
from bedrock.vault.service import VaultTools


@pytest.fixture
def vault(tmp_path: Path):
    root = tmp_path / "vault"
    root.mkdir()
    settings = Settings("http://unused", root, tmp_path / "models", "entries", "localhost", 8080)
    client = QdrantClient(":memory:")
    encoders = HashEncoders()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        IndexRebuilder(settings, client=client, encoders=encoders).rebuild()
    instance = Bedrock(settings, client=client, encoders=encoders)
    yield instance
    client.close()


def board(vault: Bedrock):
    return vault.kanban.create_board("kanban/Sandbox.md", "board-request")


def test_complete_workflow_dates_identity_indexing_and_archive(vault: Bedrock, monkeypatch):
    clock = iter(
        [
            "2026-10-16T08:00:00+00:00",
            "2026-10-16T09:00:00+00:00",
            "2026-10-16T10:00:00+00:00",
            "2026-10-17T11:00:00+00:00",
        ]
    )
    monkeypatch.setattr("bedrock.vault.editor._now", lambda: next(clock))
    created = board(vault)
    assert created.saved and created.index_status == "updated"
    added = vault.kanban.add_task(
        created.path,
        "ToDo",
        "Experiment",
        created.content_hash,
        "card-request",
        "- Nested details #idea\n- Link [[Another note]]",
        "2026-10-20",
    )
    task_id, entry_id = added.task.task_id, added.task.entry_id
    assert added.task.created_at == "2026-10-16T08:00:00+00:00"
    assert added.task.started_at is None and not added.task.checked
    changed = vault.kanban.update_task(
        added.path,
        task_id,
        added.content_hash,
        title="Edited experiment",
        content="- Revised details\n  - Keep nested child",
        due_date="2026-10-21",
    )
    assert changed.task.created_at == added.task.created_at
    assert changed.task.task_id == task_id and changed.task.entry_id == entry_id
    assert changed.task.content == "- Revised details\n  - Keep nested child"
    moving = vault.kanban.move_task(changed.path, task_id, "Doing", changed.content_hash)
    started = moving.task.started_at
    assert started == "2026-10-16T09:00:00+00:00"
    assert not moving.task.checked
    done = vault.kanban.move_task(moving.path, task_id, "Completed", moving.content_hash)
    assert done.task.completed_at == "2026-10-16T10:00:00+00:00" and done.task.checked
    assert done.task.entry_id == entry_id
    assert len(vault.find_kanban_cards(completed_at_from="2026-10-16", due_date_from="2026-10-21")) == 1
    archived = vault.kanban.archive_task(done.path, task_id, done.content_hash)
    assert archived.task.archived and archived.task.checked
    assert archived.task.completed_at == done.task.completed_at
    assert vault.find_kanban_cards() == ()
    history = vault.find_kanban_cards(archived=True)
    assert history[0].entry_id == entry_id
    assert vault.find_kanban_cards(archived=None) == history
    read = vault.kanban.get_task(done.path, task_id)
    assert read.task.archived
    reopened = vault.kanban.move_task(done.path, task_id, "Doing", read.content_hash)
    assert not reopened.task.archived and not reopened.task.checked
    assert reopened.task.completed_at is None and reopened.task.started_at == started
    redone = vault.kanban.move_task(done.path, task_id, "Completed", reopened.content_hash)
    assert redone.task.completed_at == "2026-10-17T11:00:00+00:00"
    parsed = VaultParser(vault.settings).parse_file(vault.settings.vault_path / done.path).entries[0]
    assert parsed.board.due_date == "2026-10-21" and parsed.board.completed_at == redone.task.completed_at
    assert "bedrock-dates" not in parsed.text and "created_at" not in parsed.embedding_text
    assert task_id not in parsed.text and task_id not in parsed.embedding_text
    payload = vault.client.retrieve("entries", [entry_id])[0].payload
    assert payload["board"]["task_id"] == task_id
    assert payload["board"]["completed_at"] == redone.task.completed_at
    assert payload["board"]["due_date"] == "2026-10-21T00:00:00Z"


def test_retries_do_not_duplicate_or_overwrite(vault: Bedrock):
    first = board(vault)
    added = vault.kanban.add_task(first.path, "ToDo", "One", first.content_hash, "unique")
    replay = vault.kanban.add_task(first.path, "ToDo", "One", first.content_hash, "unique")
    assert replay.replayed and replay.content_hash == added.content_hash
    assert replay.task.task_id == added.task.task_id
    recreated = vault.kanban.create_board(first.path, "board-request")
    assert recreated.replayed and recreated.content_hash == added.content_hash
    with pytest.raises(ValueError, match="already exists"):
        vault.kanban.create_board(first.path, "different")
    with pytest.raises(ValueError, match="revision conflict"):
        vault.kanban.update_task(first.path, added.task.task_id, first.content_hash, title="Stale")
    assert len(vault.find_kanban_cards()) == 1


def test_unrelated_notes_cards_frontmatter_and_settings_are_preserved(vault: Bedrock):
    note = vault.settings.vault_path / "Keep.md"
    note.write_text("Do not alter me.\n")
    original = b"""---\r\nkanban-plugin: board\r\ncustom: keep\r\n---\r\n\r\n## ToDo\r\n\r\n- [ ] Legacy @{2026-01-01}\r\n  - Existing nested content\r\n\r\n- [ ] Untouched\r\n  - Keep this byte-for-byte\r\n\r\n## Doing\r\n\r\n## Completed\r\n\r\n**Complete**\r\n\r\n%% kanban:settings\r\n```\r\n{"kanban-plugin":"board","date-trigger":"!"}\r\n```\r\n%%\r\n"""
    destination = vault.settings.vault_path / "kanban/Legacy.md"
    destination.parent.mkdir()
    destination.write_bytes(original)
    parsed = VaultParser(vault.settings).parse_file(destination)
    old = parsed.entries[0]
    read = vault.kanban.get_task("kanban/Legacy.md", old.entry_id)
    assert read.task.created_at is None
    changed = vault.kanban.move_task("kanban/Legacy.md", old.entry_id, "Doing", read.content_hash)
    assert changed.task.created_at is None and changed.task.started_at
    assert changed.backup_path and Path(changed.backup_path).read_bytes() == original
    result = destination.read_bytes()
    assert b"- [ ] Untouched\r\n  - Keep this byte-for-byte\r\n" in result
    assert result.startswith(original[: original.index(b"## ToDo")])
    assert result[result.index(b"%% kanban:settings") :] == original[original.index(b"%% kanban:settings") :]
    assert "Legacy @{2026-01-01}" in changed.task.markdown
    assert b"\n" not in result.replace(b"\r\n", b"")
    assert note.read_text() == "Do not alter me.\n"
    assert len(discover(vault)) == 2


def discover(vault: Bedrock):
    return list(vault.settings.vault_path.rglob("*.md"))


def test_reordering_keeps_every_other_card(vault: Bedrock):
    result = board(vault)
    ids = []
    for name in ("One", "Two", "Three"):
        result = vault.kanban.add_task(result.path, "ToDo", name, result.content_hash, name)
        ids.append(result.task.task_id)
    result = vault.kanban.move_task(result.path, ids[-1], "ToDo", result.content_hash, position=0)
    cards = vault.find_kanban_cards()
    assert [card.text for card in cards] == ["Three", "One", "Two"]
    result = vault.kanban.move_task(result.path, ids[-1], "ToDo", result.content_hash, position=2)
    assert [card.text for card in vault.find_kanban_cards()] == ["One", "Two", "Three"]


@pytest.mark.parametrize(
    "path",
    [
        "../outside.md",
        "/absolute.md",
        "Notes.md",
        "kanban/../Notes.md",
        "kanban/not-markdown.txt",
        "kanban/.obsidian/hidden.md",
    ],
)
def test_edit_paths_are_confined(vault: Bedrock, path: str):
    with pytest.raises(ValueError):
        vault.kanban.create_board(path, "unsafe")
    assert not discover(vault)


def test_symlinks_and_hardlinks_cannot_redirect_writes(vault: Bedrock, tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = vault.settings.vault_path
    (root / "kanban").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        vault.kanban.create_board("kanban/Unsafe.md", "unsafe")
    assert not list(outside.iterdir())
    (root / "kanban").unlink()
    created = board(vault)
    source = root / created.path
    (root / "alias.md").hardlink_to(source)
    with pytest.raises(ValueError, match="hard link"):
        vault.kanban.add_task(created.path, "ToDo", "Unsafe", created.content_hash, "unsafe")


def test_index_failure_does_not_disguise_a_saved_write(vault: Bedrock, monkeypatch):
    def unavailable(paths):
        raise RuntimeError("Qdrant offline")

    monkeypatch.setattr(vault, "reindex", unavailable)
    created = board(vault)
    assert created.saved and created.index_status == "failed" and "Qdrant offline" in created.index_error
    added = vault.kanban.add_task(created.path, "Doing", "Saved", created.content_hash, "offline")
    assert added.saved and added.index_status == "failed"
    assert vault.kanban.get_task(created.path, added.task.task_id).task.title == "Saved"
    retry = vault.kanban.add_task(created.path, "Doing", "Saved", created.content_hash, "offline")
    assert retry.replayed and retry.task.task_id == added.task.task_id


def test_native_complete_marker_and_literal_code_are_respected(vault: Bedrock):
    created = vault.kanban.create_board("kanban/Custom.md", "custom", ["Plan", "Shipping!"])
    path = vault.settings.vault_path / created.path
    path.write_text(path.read_text().replace("## Shipping!\n", "## Shipping!\n\n**Complete**\n"))
    revision = hashlib.sha256(path.read_bytes()).hexdigest()
    added = vault.kanban.add_task(
        created.path,
        "Shipping!",
        "Finished",
        revision,
        "finished",
        "```markdown\n- [ ] This is code, not another card\n## Not a column\n```",
    )
    assert added.task.checked and added.task.completed_at
    assert len(vault.find_kanban_cards()) == 1
    assert added.task.created_at
    assert len(VaultParser(vault.settings).parse_file(path).metadata["columns"]) == 2


def test_dates_can_be_cleared_and_old_creation_supplied(vault: Bedrock):
    created = board(vault)
    added = vault.kanban.add_task(
        created.path, "Completed", "Direct complete", created.content_hash, "completed", due_date="2026-10-20"
    )
    assert added.task.completed_at and added.task.checked and added.task.started_at is None
    changed = vault.kanban.update_task(
        created.path,
        added.task.task_id,
        added.content_hash,
        due_date="",
        created_at="2020-01-01T10:00:00+05:45",
    )
    assert changed.task.due_date is None
    assert changed.task.created_at == "2020-01-01T04:15:00+00:00"
    assert changed.task.completed_at == added.task.completed_at
    with pytest.raises(ValueError):
        vault.kanban.update_task(
            created.path, changed.task.task_id, changed.content_hash, due_date="2026-02-30"
        )


def test_two_writers_with_one_revision_cannot_both_commit(vault: Bedrock, monkeypatch):
    # Exercise independent editor instances against one board, without sharing
    # an embedded Qdrant client across threads.
    monkeypatch.setattr(vault, "reindex", lambda paths: None)
    created = board(vault)

    def write(name):
        try:
            return vault.kanban.add_task(created.path, "ToDo", name, created.content_hash, name).saved
        except ValueError as error:
            assert "revision conflict" in str(error)
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(write, ["One", "Two"])) == [False, True]
    assert len(VaultParser(vault.settings).parse_file(vault.settings.vault_path / created.path).entries) == 1


def test_mutation_tools_work_without_readiness_gate(vault: Bedrock):
    tools = VaultTools(vault)
    created = tools.create_board("kanban/MCP.md", "mcp")
    added = tools.add_task(created.path, "ToDo", "Tool task", created.content_hash, "task")
    read = tools.get_task(created.path, added.task.task_id)
    changed = tools.update_task(created.path, read.task.task_id, read.content_hash, content="Details")
    moved = tools.move_task(created.path, changed.task.task_id, "Completed", changed.content_hash)
    archived = tools.archive_task(created.path, moved.task.task_id, moved.content_hash)
    assert archived.task.archived
    result = vault.retriever.search(QueryPlan(mode=RetrievalMode.METADATA, kanban_archived=True))
    assert len(result) == 1 and result[0].board.completed_at


def test_external_edit_before_replace_is_not_overwritten(vault: Bedrock, monkeypatch):
    created = board(vault)
    path = vault.settings.vault_path / created.path
    original = path.read_bytes()
    backup = vault.kanban._backup
    external = original + b"\n<!-- Another editor saved this -->\n"

    def edit_between_check_and_replace(snapshot):
        result = backup(snapshot)
        path.write_bytes(external)
        return result

    monkeypatch.setattr(vault.kanban, "_backup", edit_between_check_and_replace)
    with pytest.raises(ValueError, match="changed before saving"):
        vault.kanban.add_task(created.path, "ToDo", "Stale", created.content_hash, "stale")
    assert path.read_bytes() == external
    assert not list(path.parent.glob(".bedrock-*.tmp"))


def test_invalid_managed_metadata_blocks_edits_but_leaves_source_untouched(vault: Bedrock):
    created = board(vault)
    added = vault.kanban.add_task(created.path, "ToDo", "Bad dates", created.content_hash, "bad")
    path = vault.settings.vault_path / created.path
    path.write_text(path.read_text().replace("    created_at:", "    due_Date:"))
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    with pytest.raises(ValueError, match="invalid bedrock-dates"):
        vault.kanban.update_task(created.path, added.task.task_id, digest, title="Unsafe")
    assert path.read_bytes() == original


def test_native_ids_and_archive_survive_plugin_style_four_space_reindent(vault: Bedrock):
    created = board(vault)
    added = vault.kanban.add_task(
        created.path,
        "Doing",
        "Keep ID",
        created.content_hash,
        "native",
        content="- Nested\n\n```python\nprint('hello')\n```",
    )
    path = vault.settings.vault_path / created.path
    assert any(
        line.startswith("- [ ] Keep ID ^") and added.task.task_id in line
        for line in path.read_text().splitlines()
    )
    reread = vault.kanban.get_task(created.path, added.task.task_id)
    assert reread.task.entry_id == added.task.entry_id
    assert reread.task.content == "- Nested\n\n```python\nprint('hello')\n```"


def test_archive_does_not_confuse_an_existing_regular_archive_column(vault: Bedrock):
    created = board(vault)
    path = vault.settings.vault_path / created.path
    path.write_text(path.read_text().replace("## Doing", "## Archive\n\n## Doing"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    added = vault.kanban.add_task(created.path, "Archive", "Still active", digest, "regular")
    archived = vault.kanban.archive_task(created.path, added.task.task_id, added.content_hash)
    assert archived.task.archived
    assert "Archive" in archived.columns  # Existing active column is preserved.
    assert path.read_text().count("## Archive") == 2
    assert vault.kanban.get_task(created.path, added.task.task_id).task.archived


def test_real_mcp_editing_session_and_date_filters(vault: Bedrock, monkeypatch):
    import asyncio

    from mcp.shared.memory import create_connected_server_and_client_session

    from bedrock.vault.mcp import build_server
    from bedrock.vault.service import WRITE_TOOL_NAMES

    monkeypatch.setattr(VaultTools, "_require_index", lambda self: None)

    async def exercise():
        async with create_connected_server_and_client_session(build_server(VaultTools(vault))) as client:
            listed = await client.list_tools()
            for tool in listed.tools:
                assert tool.annotations.readOnlyHint == (tool.name not in WRITE_TOOL_NAMES)
                assert tool.outputSchema

            async def call(name, args):
                result = await client.call_tool(name, args)
                assert not result.isError, result
                return result.structuredContent

            board = await call("create_board", {"path": "kanban/Protocol.md", "request_id": "board"})
            task = await call(
                "add_task",
                {
                    "board": board["path"],
                    "column": "ToDo",
                    "title": "Protocol task",
                    "expected_hash": board["content_hash"],
                    "request_id": "task",
                    "due_date": "2026-10-20",
                },
            )
            task_id = task["task"]["task_id"]
            read = await call("get_task", {"board": board["path"], "task": task_id})
            changed = await call(
                "update_task",
                {
                    "board": board["path"],
                    "task": task_id,
                    "expected_hash": read["content_hash"],
                    "content": "- Kept as nested content",
                },
            )
            for column in ("Doing", "Completed"):
                changed = await call(
                    "move_task",
                    {
                        "board": board["path"],
                        "task": task_id,
                        "column": column,
                        "expected_hash": changed["content_hash"],
                    },
                )
            assert (
                changed["task"]["checked"]
                and changed["task"]["started_at"]
                and changed["task"]["completed_at"]
            )
            page = await call(
                "find_tasks",
                {
                    "due_date_from": "2026-10-20",
                    "due_date_to": "2026-10-20",
                    "completed_at_from": changed["task"]["completed_at"],
                },
            )
            assert len(page["items"]) == 1
            archived = await call(
                "archive_task",
                {"board": board["path"], "task": task_id, "expected_hash": changed["content_hash"]},
            )
            assert archived["task"]["archived"]
            assert (await call("find_tasks", {}))["items"] == []
            assert len((await call("find_tasks", {"archived": True}))["items"]) == 1
            conflict = await client.call_tool(
                "move_task",
                {
                    "board": board["path"],
                    "task": task_id,
                    "column": "Doing",
                    "expected_hash": changed["content_hash"],
                },
            )
            assert conflict.isError

    asyncio.run(asyncio.wait_for(exercise(), timeout=10))


@pytest.mark.parametrize("column", ["on progress", "Working"])
def test_existing_vault_working_column_names_stamp_started_at(vault: Bedrock, column: str):
    created = vault.kanban.create_board("kanban/Named States.md", "states", ["Plan", column, "Completed"])
    added = vault.kanban.add_task(created.path, "Plan", "Start me", created.content_hash, "start")
    moved = vault.kanban.move_task(created.path, added.task.task_id, column, added.content_hash)
    assert moved.task.started_at and not moved.task.checked
    assert vault.find_kanban_cards(statuses=("doing",))[0].board.started_at == moved.task.started_at
