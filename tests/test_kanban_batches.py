"""Batch atomicity, small-model workflows, retries, concurrency, and recoverable removal."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from test_kanban_editor import vault as _vault_fixture

from bedrock.parser.discovery import discover_markdown
from bedrock.parser.service import VaultParser
from bedrock.vault.batch_editor import BatchKanbanEditor
from bedrock.vault.mcp import build_server
from bedrock.vault.service import VaultTools

vault = _vault_fixture


def setup(vault):
    result = vault.kanban.create_board("kanban/Batches.md", "board")
    return result.path


def test_multiple_independent_additions_do_not_need_or_conflict_on_board_hash(vault):
    board = setup(vault)
    for batch in range(3):
        result = vault.kanban.add_cards(
            board, [{"title": f"Task {batch}-{i}", "column": "ToDo"} for i in range(5)], f"request-{batch}"
        )
        assert result.saved and len(result.items) == 5 and result.index_status == "updated"
    assert len(vault.find_kanban_cards(limit=30)) == 15


def test_batch_saves_and_indexes_once_and_keeps_one_original_backup(vault, monkeypatch):
    board = setup(vault)
    source = vault.settings.vault_path / board
    original = source.read_bytes()
    saves, indexed = [], []
    save = vault.kanban._save
    reindex = vault.reindex

    def save_count(path, raw, original):
        if path == source:
            saves.append(path)
        return save(path, raw, original)

    def index_count(paths):
        indexed.append(paths)
        return reindex(paths)

    monkeypatch.setattr(vault.kanban, "_save", save_count)
    monkeypatch.setattr(vault, "reindex", index_count)
    result = vault.kanban.add_cards(board, [{"title": str(i), "column": "Doing"} for i in range(10)], "ten")
    assert len(saves) == len(indexed) == 1
    assert Path(result.backup_path).read_bytes() == original
    assert len(list(source.parent.glob(".bedrock-backups/*.bak"))) == 1
    assert all(card.started_at and card.created_at and not card.checked for card in result.items)


@pytest.mark.parametrize("operation", ["add", "edit", "move", "archive"])
def test_bad_item_saves_no_part_of_the_batch(vault, operation):
    board = setup(vault)
    initial = vault.kanban.add_cards(
        board, [{"title": "One", "column": "ToDo"}, {"title": "Two", "column": "Doing"}], "seed"
    )
    source = vault.settings.vault_path / board
    raw = source.read_bytes()
    a, b = initial.items
    if operation == "add":
        cards = [{"title": "Valid", "column": "ToDo"}, {"title": "Invalid", "column": "Missing"}]
    elif operation == "edit":
        cards = [
            {"task_id": a.task_id, "version": a.version, "title": "Changed"},
            {"task_id": b.task_id, "version": "stale", "title": "Unsafe"},
        ]
    elif operation == "move":
        cards = [{"task_id": a.task_id, "column": "Doing"}, {"task_id": b.task_id, "column": "Missing"}]
    else:
        cards = [{"task_id": a.task_id}, {"task_id": "missing"}]
    method = {
        "add": vault.kanban.add_cards,
        "edit": vault.kanban.edit_cards,
        "move": vault.kanban.move_cards,
        "archive": vault.kanban.archive_cards,
    }[operation]
    with pytest.raises(ValueError, match=r"cards\[1\].*batch not saved"):
        method(board, cards, "invalid")
    assert source.read_bytes() == raw
    assert len(vault.find_kanban_cards()) == 2


def test_card_versions_survive_unrelated_writes_but_reject_changes_to_same_card(vault):
    board = setup(vault)
    first = vault.kanban.add_cards(board, [{"title": "Target", "column": "ToDo"}], "first").items[0]
    vault.kanban.add_cards(board, [{"title": "Unrelated", "column": "ToDo"}], "second")
    edited = vault.kanban.edit_cards(
        board, [{"task_id": first.task_id, "version": first.version, "content": "Updated"}], "edit"
    )
    assert edited.items[0].version != first.version
    assert vault.kanban.get_task(board, first.task_id).task.content == "Updated"
    with pytest.raises(ValueError, match="Card version conflict"):
        vault.kanban.edit_cards(
            board, [{"task_id": first.task_id, "version": first.version, "title": "Overwrite"}], "stale-edit"
        )
    indexed = next(card for card in vault.find_kanban_cards() if card.board.task_id == first.task_id)
    assert indexed.board.version == edited.items[0].version


def test_chained_moves_see_previous_items_and_return_final_state(vault, monkeypatch):
    board = setup(vault)
    card = vault.kanban.add_cards(
        board, [{"title": "Chain", "column": "ToDo", "due_date": "2026-10-25"}], "seed"
    ).items[0]
    clock = iter(["2026-10-16T01:00:00+00:00", "2026-10-16T02:00:00+00:00"])
    monkeypatch.setattr("bedrock.vault.editor._now", lambda: next(clock))
    result = vault.kanban.move_cards(
        board,
        [
            {"task_id": card.task_id, "from_column": "ToDo", "column": "Doing"},
            {"task_id": card.task_id, "from_column": "Doing", "column": "Completed"},
        ],
        "chain",
    )
    assert all(item.column == "Completed" and item.checked for item in result.items)
    assert result.items[0].started_at == "2026-10-16T01:00:00+00:00"
    assert result.items[0].completed_at == "2026-10-16T02:00:00+00:00"
    assert result.items[0].due_date == card.due_date
    archived = vault.kanban.archive_cards(board, [{"task_id": card.task_id}], "archive")
    assert archived.items[0].archived
    reopened = vault.kanban.move_cards(board, [{"task_id": card.task_id, "column": "ToDo"}], "reopen")
    assert not reopened.items[0].checked and reopened.items[0].completed_at is None
    assert reopened.items[0].started_at == result.items[0].started_at


def test_add_and_edit_retries_are_durable_and_different_arguments_are_rejected(vault):
    board = setup(vault)
    cards = [{"title": "One", "column": "ToDo"}, {"title": "Two", "column": "Doing"}]
    added = vault.kanban.add_cards(board, cards, "stable")
    another_editor = BatchKanbanEditor(vault)
    replay = another_editor.add_cards(board, cards, "stable")
    assert replay.replayed and replay.content_hash == added.content_hash
    assert len(vault.find_kanban_cards()) == 2
    with pytest.raises(ValueError, match="different arguments"):
        another_editor.add_cards(board, [{"title": "Different", "column": "ToDo"}], "stable")
    edits = [
        {"task_id": card.task_id, "version": card.version, "title": card.title + " edited"}
        for card in added.items
    ]
    changed = another_editor.edit_cards(board, edits, "edit")
    repeated = vault.kanban.edit_cards(board, edits, "edit")
    assert repeated.replayed and repeated.content_hash == changed.content_hash


def test_parallel_appends_all_succeed_and_parallel_identical_retries_add_once(vault, monkeypatch):
    monkeypatch.setattr(vault, "reindex", lambda paths: None)
    board = setup(vault)

    def add(index):
        return BatchKanbanEditor(vault).add_cards(
            board, [{"title": str(index), "column": "ToDo"}], f"request-{index}"
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(add, range(8)))
    assert all(result.saved for result in results)
    source = vault.settings.vault_path / board
    assert len(VaultParser(vault.settings).parse_file(source).entries) == 8
    with ThreadPoolExecutor(max_workers=4) as pool:
        retries = list(pool.map(lambda _: add(20), range(4)))
    assert sum(result.replayed for result in retries) == 3
    assert len(VaultParser(vault.settings).parse_file(source).entries) == 9


def test_pending_receipt_recovers_a_saved_batch_after_receipt_finalization_failure(vault, monkeypatch):
    board = setup(vault)
    writer = vault.kanban._write_receipt
    calls = 0

    def fail_final(path, receipt):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("Simulated crash before receipt finalization")
        writer(path, receipt)

    monkeypatch.setattr(vault.kanban, "_write_receipt", fail_final)
    cards = [{"title": "Saved", "column": "ToDo"}]
    first = vault.kanban.add_cards(board, cards, "recover")
    assert first.saved
    retry = BatchKanbanEditor(vault).add_cards(board, cards, "recover")
    assert retry.replayed
    assert len(vault.find_kanban_cards()) == 1


def test_index_failure_reports_saved_batch_and_retry_recovers_without_duplication(vault, monkeypatch):
    board = setup(vault)
    reindex = vault.reindex

    def unavailable(paths):
        raise RuntimeError("index offline")

    monkeypatch.setattr(vault, "reindex", unavailable)
    cards = [{"title": "Offline", "column": "ToDo"}]
    saved = vault.kanban.add_cards(board, cards, "offline")
    assert saved.saved and saved.index_status == "failed"
    monkeypatch.setattr(vault, "reindex", reindex)
    retried = vault.kanban.add_cards(board, cards, "offline")
    assert retried.replayed and retried.index_status == "updated"
    assert len(vault.find_kanban_cards()) == 1


def test_delete_board_requires_preview_token_is_recoverable_and_retry_safe(vault):
    board = setup(vault)
    vault.kanban.add_cards(board, [{"title": "Retain me", "column": "ToDo"}], "seed")
    source = vault.settings.vault_path / board
    original = source.read_bytes()
    preview = vault.kanban.delete_board(board)
    assert not preview.saved and preview.status == "preview" and preview.card_count == 1
    assert "Warn the user" in preview.warning and source.exists()
    deleted = vault.kanban.delete_board(board, preview.confirmation_token)
    assert deleted.saved and deleted.status == "deleted" and deleted.index_status == "updated"
    assert not source.exists()
    assert Path(deleted.trash_path).read_bytes() == original
    assert not discover_markdown(vault.settings)
    assert not vault.find_kanban_cards(archived=None)
    source.write_bytes(original)  # Simulate restoring/recreating a board.
    replay = vault.kanban.delete_board(board, preview.confirmation_token)
    assert replay.replayed and source.read_bytes() == original


def test_changed_board_invalidates_delete_preview(vault):
    board = setup(vault)
    preview = vault.kanban.delete_board(board)
    vault.kanban.add_cards(board, [{"title": "New", "column": "ToDo"}], "after-preview")
    with pytest.raises(ValueError, match="changed since deletion preview"):
        vault.kanban.delete_board(board, preview.confirmation_token)
    assert (vault.settings.vault_path / board).exists()


@pytest.mark.parametrize(
    "board",
    ["Plain.md", "kanban/../Plain.md", "kanban/.bedrock-trash/hidden.md", "kanban/.bedrock-state/hidden.md"],
)
def test_batch_and_delete_cannot_modify_ordinary_notes_or_internal_storage(vault, board):
    note = vault.settings.vault_path / "Plain.md"
    note.write_text("Immutable\n")
    with pytest.raises(ValueError):
        vault.kanban.add_cards(board, [{"title": "Unsafe", "column": "ToDo"}], "unsafe")
    with pytest.raises(ValueError):
        vault.kanban.delete_board(board)
    assert note.read_text() == "Immutable\n"


def test_batch_reports_legacy_id_upgrade_and_keeps_unknown_creation_date(vault):
    board = setup(vault)
    source = vault.settings.vault_path / board
    source.write_text(
        source.read_text().replace("## ToDo\n", "## ToDo\n\n- [ ] Legacy\n    - Nested content\n")
    )
    legacy = VaultParser(vault.settings).parse_file(source).entries[0]
    result = vault.kanban.move_cards(
        board,
        [
            {"task_id": legacy.entry_id, "column": "Doing"},
            {"task_id": legacy.entry_id, "column": "Completed"},
        ],
        "legacy-chain",
    )
    assert result.items[0].task_id.startswith("bedrock-")
    assert result.items[0].created_at is None and result.items[0].started_at and result.items[0].completed_at
    assert result.items[0].task_id == result.items[1].task_id


def test_small_model_mcp_workflow_including_concurrent_adds(vault, monkeypatch):
    from mcp.shared.memory import create_connected_server_and_client_session

    monkeypatch.setattr(VaultTools, "_require_index", lambda self: None)

    async def exercise():
        async with create_connected_server_and_client_session(build_server(VaultTools(vault))) as client:

            async def call(name, args):
                result = await client.call_tool(name, args)
                assert not result.isError, result
                return result.structuredContent

            listing = await client.list_tools()
            names = {tool.name for tool in listing.tools}
            assert {"add_cards", "edit_cards", "move_cards", "archive_cards", "delete_board"} <= names
            assert not {"add_task", "update_task", "move_task", "archive_task"} & names
            created = await call("create_board", {"path": "Protocol", "request_id": "board"})
            board = created["path"]
            assert board == "kanban/Protocol.md"
            calls = await asyncio.gather(
                *[
                    call(
                        "add_cards",
                        {
                            "board": board,
                            "cards": [{"title": f"Task {i}", "column": "ToDo"}],
                            "request_id": f"add-{i}",
                        },
                    )
                    for i in range(4)
                ]
            )
            items = [result["items"][0] for result in calls]
            changed = await call(
                "edit_cards",
                {
                    "board": board,
                    "cards": [
                        {"task_id": item["task_id"], "version": item["version"], "content": "- Changed"}
                        for item in items
                    ],
                    "request_id": "edit-all",
                },
            )
            moved = await call(
                "move_cards",
                {
                    "board": board,
                    "cards": [
                        {"task_id": item["task_id"], "from_column": "ToDo", "column": "Completed"}
                        for item in changed["items"]
                    ],
                    "request_id": "move-all",
                },
            )
            assert all(item["completed_at"] and item["checked"] for item in moved["items"])
            archived = await call(
                "archive_cards",
                {
                    "board": board,
                    "cards": [{"task_id": item["task_id"]} for item in moved["items"]],
                    "request_id": "archive-all",
                },
            )
            assert all(item["archived"] for item in archived["items"])
            preview = await call("delete_board", {"board": board})
            assert preview["status"] == "preview" and preview["card_count"] == 4
            deleted = await call(
                "delete_board", {"board": board, "confirmation_token": preview["confirmation_token"]}
            )
            assert deleted["saved"] and Path(deleted["trash_path"]).is_file()

    asyncio.run(asyncio.wait_for(exercise(), timeout=15))


@pytest.mark.parametrize(
    "cards",
    [
        [],
        [{"title": "Many", "column": "ToDo"}] * 31,
        [{"title": "Unknown field", "column": "ToDo", "expected_hash": "obsolete"}],
    ],
)
def test_invalid_batch_shapes_do_not_save(vault, cards):
    board = setup(vault)
    source = vault.settings.vault_path / board
    raw = source.read_bytes()
    with pytest.raises(ValueError):
        vault.kanban.add_cards(board, cards, "invalid-shape")
    assert source.read_bytes() == raw


def test_delete_failed_before_rename_can_retry_with_the_same_preview(vault, monkeypatch):
    import os

    board = setup(vault)
    preview = vault.kanban.delete_board(board)
    rename = os.rename

    def fail(source, destination):
        raise OSError("Simulated rename failure")

    monkeypatch.setattr(os, "rename", fail)
    with pytest.raises(OSError, match="rename failure"):
        vault.kanban.delete_board(board, preview.confirmation_token)
    assert (vault.settings.vault_path / board).is_file()
    monkeypatch.setattr(os, "rename", rename)
    result = vault.kanban.delete_board(board, preview.confirmation_token)
    assert result.saved and Path(result.trash_path).is_file()


def test_delete_rejects_a_plain_note_even_inside_the_kanban_folder(vault):
    board = setup(vault)
    note = vault.settings.vault_path / "kanban/Plain.md"
    note.write_text("# Ordinary immutable note\n")
    with pytest.raises(ValueError, match="kanban-plugin"):
        vault.kanban.delete_board("kanban/Plain.md")
    assert note.read_text() == "# Ordinary immutable note\n"
    assert (vault.settings.vault_path / board).exists()


def test_internal_symlinks_cannot_redirect_receipts_or_trash(vault, tmp_path):
    board = setup(vault)
    root = vault.settings.vault_path / "kanban"
    outside = tmp_path / "outside"
    outside.mkdir()
    # A per-board receipt directory is reached only after the shared lock check.
    (root / ".bedrock-trash").symlink_to(outside, target_is_directory=True)
    preview = vault.kanban.delete_board(board)
    with pytest.raises(ValueError, match="symlink"):
        vault.kanban.delete_board(board, preview.confirmation_token)
    assert not list(outside.iterdir())
    assert (vault.settings.vault_path / board).exists()


def _process_append(root, number):
    from bedrock.core.config import Settings
    from bedrock.library import Bedrock

    api = Bedrock(
        Settings("http://unused", Path(root), Path(root) / "models", "unused", "localhost", 8080),
        client=object(),
    )
    api.reindex = lambda paths: None
    return api.kanban.add_cards(
        "kanban/Batches.md", [{"title": str(number), "column": "ToDo"}], f"process-{number}"
    ).saved


def test_separate_processes_coordinate_through_shared_board_lock(vault):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    board = setup(vault)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(_process_append, str(vault.settings.vault_path), i) for i in range(4)]
        assert all(future.result(timeout=20) for future in futures)
    assert len(VaultParser(vault.settings).parse_file(vault.settings.vault_path / board).entries) == 4


@pytest.mark.parametrize("name", ["Project", "Project.md", "kanbans/Project", "kanbans/Project.md"])
def test_create_board_resolves_names_using_configured_root(vault, monkeypatch, name):
    from dataclasses import replace

    vault.settings = replace(vault.settings, kanban_root="kanbans")
    monkeypatch.setattr(vault, "reindex", lambda paths: None)
    created = vault.kanban.create_board(name, "named-board")
    assert created.path == "kanbans/Project.md"
    assert (vault.settings.vault_path / created.path).is_file()
    replayed = vault.kanban.create_board("Project", "named-board")
    assert replayed.replayed and replayed.content_hash == created.content_hash
    with pytest.raises(ValueError, match="already exists"):
        vault.kanban.create_board("Project.md", "different-request")


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
        ".",
        "..",
        ".md",
        "../Escape",
        "/tmp/Escape.md",
        "kanban/../Escape",
        "Project.txt",
        "Project\nInjected",
        "Wrong/Project.md",
    ],
)
def test_friendly_board_names_still_reject_unsafe_inputs(vault, name):
    with pytest.raises(ValueError):
        vault.kanban.create_board(name, "unsafe-name")
    assert not discover_markdown(vault.settings)
