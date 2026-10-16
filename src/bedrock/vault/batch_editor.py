"""Explicit, all-or-nothing card batches with durable request receipts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from bedrock.core.identity import normalize_vault_path
from bedrock.vault.editor import KanbanEditor, _Board, _hash, _internal_directory, _request
from bedrock.vault.kanban_contracts import (
    BoardDeletion,
    CardEdit,
    CardMove,
    CardSummary,
    CardsWrite,
    CardTarget,
    KanbanWrite,
    NewCard,
)

MAX_BATCH = 30


class _StagedEditor(KanbanEditor):
    """Reuse single-card transformations, changing only an in-memory snapshot."""

    def __init__(self, vault, board: _Board) -> None:
        super().__init__(vault)
        self.snapshot = board
        self.aliases: dict[str, str] = {}

    def _lock(self, _path):
        return nullcontext()

    def _load(self, name: str) -> _Board:
        if normalize_vault_path(name) != self.snapshot.relative:
            raise ValueError("Batches cannot cross board files")
        return self.snapshot

    def _task(self, board, task):
        return super()._task(board, self.aliases.get(task, task))

    def _commit(self, board, text, task_id, edited=None) -> KanbanWrite:
        self._validate_transition(board, text, task_id, edited)
        if edited is not None:
            entry = next(entry for entry in board.entries if entry.source.start_line == edited.start + 1)
            self.aliases[entry.entry_id] = task_id
        self.snapshot = self._parse(board.relative, text.encode("utf-8"))
        return self._result(self.snapshot, task_id)

    def _result(self, board, task_id=None, *, replayed=False, backup=None) -> KanbanWrite:
        task = None
        if task_id:
            column, span, entry = self._task(board, task_id)
            task = self._view(board, column, span, entry)
        return KanbanWrite(
            path=board.relative,
            columns=[col.name for col in board.columns if not col.archived],
            content_hash=board.revision,
            task=task,
            replayed=replayed,
            index_status="updated",
            backup_path=backup,
        )


class BatchKanbanEditor(KanbanEditor):
    def add_cards(self, board: str, cards: list[NewCard], request_id: str) -> CardsWrite:
        return self._batch(board, cards, request_id, "add", NewCard)

    def edit_cards(self, board: str, cards: list[CardEdit], request_id: str) -> CardsWrite:
        return self._batch(board, cards, request_id, "edit", CardEdit)

    def move_cards(self, board: str, cards: list[CardMove], request_id: str) -> CardsWrite:
        return self._batch(board, cards, request_id, "move", CardMove)

    def archive_cards(self, board: str, cards: list[CardTarget], request_id: str) -> CardsWrite:
        return self._batch(board, cards, request_id, "archive", CardTarget)

    def _batch(
        self, board: str, cards: list, request_id: str, kind: str, model: type[BaseModel]
    ) -> CardsWrite:
        request_id = _request(request_id)
        if not 1 <= len(cards) <= MAX_BATCH:
            raise ValueError(f"cards must contain 1 to {MAX_BATCH} items")
        inputs = []
        for index, card in enumerate(cards):
            try:
                inputs.append(model.model_validate(card))
            except ValueError as error:
                raise ValueError(f"cards[{index}]: {error}; batch not saved") from error
        fingerprint = _digest({"operation": kind, "cards": [card.model_dump() for card in inputs]})
        path = self._path(board)
        with self._lock(path):
            receipt_path = self._receipt_path(board, request_id)
            receipt = _read_receipt(receipt_path)
            if receipt:
                if receipt["fingerprint"] != fingerprint:
                    raise ValueError(
                        "request_id was already used with different arguments; use a new ID for a new request"
                    )
                current = path.read_bytes() if path.exists() else None
                if receipt["state"] == "committed" or (
                    current is not None and _hash(current) == receipt["after_hash"]
                ):
                    result = CardsWrite.model_validate(receipt["result"])
                    result.replayed = True
                    result.index_status, result.index_error = self._sync(board)
                    receipt.update(state="committed", result=result.model_dump())
                    self._write_receipt(receipt_path, receipt)
                    return result
                if current is None or _hash(current) != receipt["before_hash"]:
                    raise ValueError(
                        "Previous request outcome is uncertain after an external change. Inspect this board before issuing new edits; this retry made no changes"
                    )
            original = self._load(board)
            staged = _StagedEditor(self.vault, original)
            identifiers = []
            for index, card in enumerate(inputs):
                try:
                    result = self._apply(staged, kind, card, request_id, index)
                    identifiers.append(result.task.task_id)
                except (ValueError, LookupError) as error:
                    raise ValueError(f"cards[{index}]: {error}; entire batch not saved") from error
            final = staged.snapshot
            summaries = []
            for identifier in identifiers:
                col, span, entry = staged._task(final, identifier)
                view = staged._view(final, col, span, entry)
                summaries.append(CardSummary.model_validate(view.model_dump()))
            backup = self._backup(original) if final.raw != original.raw else None
            result = CardsWrite(
                path=final.relative,
                content_hash=final.revision,
                items=summaries,
                backup_path=backup,
                index_status="failed",
                index_error="Index sync pending",
            )
            receipt = {
                "state": "pending",
                "fingerprint": fingerprint,
                "before_hash": original.revision,
                "after_hash": final.revision,
                "result": result.model_dump(),
            }
            # Prepare receipt first: a retry can recognize a save that completed
            # before the process lost its response or finalized its receipt.
            self._write_receipt(receipt_path, receipt)
            if final.raw != original.raw:
                self._save(original.path, final.raw, original.raw)
            elif original.path.read_bytes() != original.raw:
                raise ValueError("Board changed during batch validation; batch not saved")
            result.index_status, result.index_error = self._sync(board)
            receipt.update(state="committed", result=result.model_dump())
            # The pending receipt is durable and recognizes the saved source on
            # retry. Do not disguise a saved batch if receipt finalization fails.
            with suppress(OSError):
                self._write_receipt(receipt_path, receipt)
            return result

    @staticmethod
    def _apply(staged, kind, card, request_id, index):
        snapshot = staged.snapshot
        board, revision = snapshot.relative, snapshot.revision
        if kind == "add":
            key = f"batch-{hashlib.sha256(request_id.encode()).hexdigest()}-{index}"
            return staged.add_task(board, card.column, card.title, revision, key, card.content, card.due_date)
        col, span, entry = staged._task(snapshot, card.task_id)
        if kind == "edit":
            view = staged._view(snapshot, col, span, entry)
            if card.version != view.version:
                raise ValueError(
                    f"Card version conflict for {card.task_id}; current version={view.version}. get_task and reread before editing"
                )
            return staged.update_task(
                board, card.task_id, revision, card.title, card.content, card.due_date, card.created_at
            )
        if kind == "move":
            if card.from_column is not None and card.from_column.casefold() != col.name.casefold():
                raise ValueError(f"Source column conflict for {card.task_id}: current column={col.name!r}")
            return staged.move_task(board, card.task_id, card.column, revision, card.position)
        return staged.archive_task(board, card.task_id, revision)

    def delete_board(self, board: str, confirmation_token: str | None = None) -> BoardDeletion:
        path = self._path(board)
        root = self.vault.settings.vault_path.resolve() / self.vault.settings.kanban_root
        if confirmation_token is not None and not re.fullmatch(r"[0-9a-f]{64}", confirmation_token):
            raise ValueError("Use the confirmation_token returned by a delete_board preview")
        with self._lock(path):
            if confirmation_token:
                directory = _internal_directory(root, ".bedrock-trash", confirmation_token)
                receipt_path = directory / "receipt.json"
                receipt = _read_receipt(receipt_path)
                if receipt:
                    if receipt["board"] != normalize_vault_path(board):
                        raise ValueError("Deletion token belongs to a different board")
                    trash = directory / "board.md.bak"
                    if trash.is_symlink():
                        raise ValueError("Trashed board cannot be a symlink")
                    if trash.exists():
                        if not trash.is_file() or _hash(trash.read_bytes()) != receipt["content_hash"]:
                            raise ValueError("Trashed board does not match its deletion receipt")
                        result = BoardDeletion.model_validate(receipt["result"])
                        result.replayed = True
                        result.index_status, result.index_error = self._sync(board)
                        return result
                    if not path.exists():
                        raise ValueError(
                            "Deletion outcome is uncertain: source and retained board are unavailable"
                        )
                    # A prepared receipt without a retained file can mean rename
                    # failed. Revalidate the current preview before retrying it.
            loaded = self._load(board)
            identity = path.stat()
            token = _digest(
                {
                    "board": loaded.relative,
                    "hash": loaded.revision,
                    "inode": identity.st_ino,
                    "mtime": identity.st_mtime_ns,
                }
            )
            count = sum(len(col.cards) for col in loaded.columns)
            warning = f"This removes {loaded.relative} and all {count} cards from the active vault. The complete file is retained in Kanban trash for recovery. Warn the user before executing deletion."
            preview = BoardDeletion(
                path=loaded.relative,
                status="preview",
                warning=warning,
                card_count=count,
                confirmation_token=token,
            )
            if confirmation_token is None:
                return preview
            if confirmation_token != token:
                raise ValueError(
                    "Board changed since deletion preview; request a new preview and warn the user again"
                )
            directory = _internal_directory(root, ".bedrock-trash", token)
            trash = directory / "board.md.bak"
            if trash.exists() or trash.is_symlink():
                raise ValueError("Deletion storage already exists; inspect it before retrying")
            result = preview.model_copy(
                update={
                    "status": "deleted",
                    "saved": True,
                    "trash_path": str(trash),
                    "index_status": "failed",
                    "index_error": "Index sync pending",
                }
            )
            receipt = {
                "board": loaded.relative,
                "content_hash": loaded.revision,
                "result": result.model_dump(),
            }
            # A failed rename leaves source untouched; recovery requires the
            # actual retained file as well as this prepared receipt.
            self._write_receipt(directory / "receipt.json", receipt)
            if path.is_symlink() or path.read_bytes() != loaded.raw:
                raise ValueError("Board changed before deletion; no deletion performed")
            os.rename(path, trash)
            _sync_directory(path.parent)
            _sync_directory(directory)
            result.index_status, result.index_error = self._sync(board)
            return result

    def _receipt_path(self, board: str, request_id: str) -> Path:
        root = self.vault.settings.vault_path.resolve() / self.vault.settings.kanban_root
        directory = _internal_directory(
            root,
            ".bedrock-state",
            "requests",
            hashlib.sha256(normalize_vault_path(board).encode()).hexdigest(),
        )
        return directory / (hashlib.sha256(request_id.encode()).hexdigest() + ".json")

    def _sync(self, board: str) -> tuple[str, str | None]:
        try:
            self.vault.reindex([normalize_vault_path(board)])
            return "updated", None
        except Exception as error:
            return "failed", str(error)

    def _write_receipt(self, path: Path, receipt: dict) -> None:
        if path.is_symlink():
            raise ValueError("Request receipt cannot be a symlink")
        original = path.read_bytes() if path.exists() else None
        self._save(path, json.dumps(receipt, ensure_ascii=False, sort_keys=True).encode(), original)


def _read_receipt(path: Path) -> dict[str, Any] | None:
    if path.is_symlink():
        raise ValueError("Request receipt cannot be a symlink")
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Invalid request receipt; inspect internal Kanban storage")
    return value


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
