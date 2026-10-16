"""Revision-checked, atomic edits confined to configured Kanban board files."""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
import tempfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid4, uuid5

import yaml

from bedrock.core.contracts import ParsedEntry
from bedrock.core.identity import kanban_card_version, normalize_vault_path
from bedrock.core.task_dates import TASK_DATE_FIELDS, task_date_value
from bedrock.parser.discovery import discover_markdown
from bedrock.parser.kanban import canonical_status, parse_settings
from bedrock.parser.kanban_dates import lifecycle_dates
from bedrock.parser.kanban_structure import BLOCK_ID_RE, CARD_RE, CardSpan, ColumnSpan, board_columns
from bedrock.parser.markdown import markdown_tokens, parse_frontmatter
from bedrock.parser.service import VaultParser
from bedrock.vault.kanban_contracts import EditableTask, KanbanWrite, TaskRead

if TYPE_CHECKING:
    from bedrock.library import Bedrock

MAX_CARD_CHARS = 64000


@dataclass
class _Board:
    path: Path
    relative: str
    raw: bytes
    lines: list[str]
    columns: tuple[ColumnSpan, ...]
    footer: int
    revision: str
    entries: tuple[ParsedEntry, ...]


class KanbanEditor:
    def __init__(self, vault: Bedrock) -> None:
        self.vault = vault

    def create_board(self, path: str, request_id: str, columns: list[str] | None = None) -> KanbanWrite:
        request_id = _request(request_id)
        path = self._creation_path(path)
        destination = self._path(path)
        names = columns if columns is not None else ["ToDo", "Doing", "Completed"]
        if not names or len(names) > 30:
            raise ValueError("Provide between 1 and 30 columns")
        names = [_single(name, "column") for name in names]
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("Column names must be unique, ignoring case")
        if any(name.casefold() == "archive" for name in names):
            raise ValueError("Archive is reserved for the plugin's archive section")
        with self._lock(destination):
            if destination.exists():
                board = self._load(path)
                frontmatter = parse_frontmatter(board.raw.decode("utf-8"))
                if frontmatter.values.get("bedrock-board-request") == request_id:
                    return self._result(board, replayed=True)
                raise ValueError("Board already exists; create_board never overwrites files")
            destination.parent.mkdir(parents=True, exist_ok=True)
            frontmatter = yaml.safe_dump(
                {"kanban-plugin": "board", "bedrock-board-request": request_id}, sort_keys=False
            )
            text = "---\n" + frontmatter + "---\n\n"
            for name in names:
                text += f"## {name}\n\n"
                if canonical_status(name) == "complete":
                    text += "**Complete**\n\n"
            text += '%% kanban:settings\n```\n{"kanban-plugin":"board"}\n```\n%%\n'
            self._save(destination, text.encode("utf-8"), original=None)
            return self._result(self._load(path))

    def get_task(self, board: str, task: str) -> TaskRead:
        loaded = self._load(board)
        column, span, entry = self._task(loaded, task)
        return TaskRead(content_hash=loaded.revision, task=self._view(loaded, column, span, entry))

    def add_task(
        self,
        board: str,
        column: str,
        title: str,
        expected_hash: str,
        request_id: str,
        content: str = "",
        due_date: str | None = None,
        position: int | None = None,
    ) -> KanbanWrite:
        title = _single(title, "title")
        _body(content)
        request_id = _request(request_id)
        task_id = "bedrock-" + str(
            uuid5(NAMESPACE_URL, f"{self.vault.vault_id}:{normalize_vault_path(board)}:{request_id}")
        )
        due = task_date_value("due_date", due_date) if due_date is not None else None
        with self._lock(self._path(board)):
            loaded = self._load(board)
            if any(span.block_id == task_id for col in loaded.columns for span in col.cards):
                return self._result(loaded, task_id, replayed=True)
            _revision(loaded, expected_hash)
            target = self._column(loaded, column)
            now = _now()
            dates = {"created_at": now}
            if _status(target) == "doing":
                dates["started_at"] = now
            if _status(target) == "complete":
                dates["completed_at"] = now
            if due:
                dates["due_date"] = due
            card = self._render(
                title,
                ["    " + line if line else "" for line in content.splitlines()],
                task_id,
                dates,
                _status(target) == "complete",
                _newline(loaded),
            )
            updated = self._insert(loaded, target, card, position)
            return self._commit(loaded, updated, task_id)

    def update_task(
        self,
        board: str,
        task: str,
        expected_hash: str,
        title: str | None = None,
        content: str | None = None,
        due_date: str | None = None,
        created_at: str | None = None,
    ) -> KanbanWrite:
        if title is not None:
            title = _single(title, "title")
        if content is not None:
            _body(content)
        due = task_date_value("due_date", due_date) if due_date else None
        created = task_date_value("created_at", created_at) if created_at is not None else None
        with self._lock(self._path(board)):
            loaded = self._load(board)
            _revision(loaded, expected_hash)
            column, span, entry = self._task(loaded, task)
            task_id = span.block_id or "bedrock-" + str(uuid4())
            old_title, old_body, dates = self._parts(loaded, span)
            if due_date is not None:
                if due:
                    dates["due_date"] = due
                else:
                    dates.pop("due_date", None)
            if created is not None:
                dates["created_at"] = created
            body = (
                old_body
                if content is None
                else ["    " + line if line else "" for line in content.splitlines()]
            )
            card = self._render(
                title if title is not None else old_title,
                body,
                task_id,
                dates,
                entry.board.checked if column.archived else _status(column) == "complete",
                _newline(loaded),
            )
            lines = loaded.lines.copy()
            lines[span.start : span.end] = card.splitlines(keepends=True)
            return self._commit(loaded, "".join(lines), task_id, span)

    def move_task(
        self,
        board: str,
        task: str,
        column: str,
        expected_hash: str,
        position: int | None = None,
    ) -> KanbanWrite:
        with self._lock(self._path(board)):
            loaded = self._load(board)
            _revision(loaded, expected_hash)
            source, span, _entry = self._task(loaded, task)
            target = self._column(loaded, column)
            title, body, dates = self._parts(loaded, span)
            task_id = span.block_id or "bedrock-" + str(uuid4())
            status = _status(target)
            if status == "doing" and not dates.get("started_at"):
                dates["started_at"] = _now()
            if status == "complete":
                if source.archived or _status(source) != "complete" or not dates.get("completed_at"):
                    dates["completed_at"] = _now()
            else:
                dates.pop("completed_at", None)
            card = self._render(title, body, task_id, dates, status == "complete", _newline(loaded))
            return self._relocate(loaded, span, target, card, task_id, position)

    def archive_task(self, board: str, task: str, expected_hash: str) -> KanbanWrite:
        with self._lock(self._path(board)):
            loaded = self._load(board)
            _revision(loaded, expected_hash)
            column, span, entry = self._task(loaded, task)
            if column.archived:
                return self._result(loaded, task, replayed=True)
            title, body, dates = self._parts(loaded, span)
            task_id = span.block_id or "bedrock-" + str(uuid4())
            card = self._render(title, body, task_id, dates, entry.board.checked, _newline(loaded))
            target = next((col for col in loaded.columns if col.archived), None)
            if target is not None:
                return self._relocate(loaded, span, target, card, task_id, None)
            lines = loaded.lines.copy()
            del lines[span.start : span.end]
            boundary = loaded.footer - (span.end - span.start)
            newline = _newline(loaded)
            lines[boundary:boundary] = (
                newline + "***" + newline * 2 + "## Archive" + newline * 2 + card + newline
            ).splitlines(keepends=True)
            return self._commit(loaded, "".join(lines), task_id, span)

    def _lock(self, path: Path):
        root = self.vault.settings.vault_path.resolve() / self.vault.settings.kanban_root
        return _locked(path, root)

    def _creation_path(self, value: str) -> str:
        """Accept a board name while preserving explicit, scoped path support."""
        raw = value.strip()
        if not raw or any(ord(char) < 32 or ord(char) == 127 for char in raw):
            raise ValueError("Provide a nonempty board name without control characters")
        relative = Path(normalize_vault_path(raw))
        root = self.vault.settings.kanban_root
        if len(relative.parts) == 1:
            relative = Path(root) / relative
        elif not relative.parts or relative.parts[0] != root:
            raise ValueError(f"Use a plain board name, or an explicit path beneath {root}/")
        if not relative.name or relative.name in {".", "..", ".md"}:
            raise ValueError("Provide a board name before the optional .md extension")
        if not relative.suffix:
            relative = relative.with_name(relative.name + ".md")
        elif relative.suffix != ".md":
            raise ValueError("Provide a board name with no extension or with .md")
        return relative.as_posix()

    def _path(self, raw: str) -> Path:
        relative = Path(normalize_vault_path(raw))
        settings = self.vault.settings
        if relative.suffix != ".md" or len(relative.parts) < 2 or relative.parts[0] != settings.kanban_root:
            raise ValueError(f"Use an explicit .md path beneath {settings.kanban_root}/")
        excluded = {name.casefold() for name in settings.excluded_directories} | {
            ".bedrock-state",
            ".bedrock-trash",
            ".bedrock-backups",
        }
        if any(part.casefold() in excluded for part in relative.parts[:-1]) or relative.name.casefold() in {
            name.casefold() for name in settings.excluded_files
        }:
            raise ValueError("Cannot edit an excluded board")
        root = settings.vault_path.resolve(strict=True)
        destination = root / relative
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError("Kanban edits do not follow symlinks")
        if not destination.resolve().is_relative_to(root):
            raise ValueError("Board path escapes vault")
        if destination.exists() and (not destination.is_file() or destination.stat().st_nlink != 1):
            raise ValueError("Board must be a regular file with a single hard link")
        return destination

    def _load(self, name: str) -> _Board:
        name = normalize_vault_path(name)
        path = self._path(name)
        if not path.is_file():
            raise LookupError(f"Kanban board not found: {name}")
        if not discover_markdown(self.vault.settings, [name]):
            raise ValueError("Board is unavailable or excluded")
        raw = path.read_bytes()
        board = self._parse(name, raw)
        if path.read_bytes() != raw:
            raise ValueError("Board changed while reading; reread before editing")
        return board

    def _parse(self, name: str, raw: bytes) -> _Board:
        path = self._path(name)
        text = raw.decode("utf-8")
        lines = text.splitlines(keepends=True)
        frontmatter = parse_frontmatter(text, name)
        if frontmatter.warning or frontmatter.values.get("kanban-plugin") != "board":
            raise ValueError("Editing requires valid frontmatter with kanban-plugin: board")
        _, warning, footer = parse_settings(text.splitlines(), name)
        if warning:
            raise ValueError(f"Repair the board settings before editing: {warning.message}")
        columns = board_columns(
            markdown_tokens(text), lines, frontmatter.body_start, footer if footer is not None else len(lines)
        )
        active_names = [col.name.casefold() for col in columns if not col.archived]
        if (
            not active_names
            or len(set(active_names)) != len(active_names)
            or sum(col.archived for col in columns) > 1
        ):
            raise ValueError("Editing requires distinct active columns and at most one archive section")
        ids = [card.block_id for col in columns for card in col.cards if card.block_id]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate card block IDs; repair the board before editing")
        parsed = VaultParser(self.vault.settings, self.vault.vault_id).parse_content(
            name, text.replace("\r\n", "\n").replace("\r", "\n")
        )
        if parsed.content_hash != _hash(raw):
            raise ValueError("Board changed while reading; reread before editing")
        entries = parsed.entries
        return _Board(
            path, name, raw, lines, columns, footer if footer is not None else len(lines), _hash(raw), entries
        )

    @staticmethod
    def _column(board: _Board, name: str) -> ColumnSpan:
        matches = [
            col for col in board.columns if not col.archived and col.name.casefold() == name.casefold()
        ]
        if not matches:
            raise ValueError(
                f"Unknown active column {name!r}; available: {[col.name for col in board.columns if not col.archived]}"
            )
        return matches[0]

    @staticmethod
    def _task(board: _Board, task: str) -> tuple[ColumnSpan, CardSpan, ParsedEntry]:
        by_line = {entry.source.start_line - 1: entry for entry in board.entries}
        for col in board.columns:
            for span in col.cards:
                entry = by_line[span.start]
                if task in {span.block_id, entry.entry_id}:
                    return col, span, entry
        raise LookupError(f"Task not found in {board.relative}; use find_tasks or reread its current ID")

    def _parts(self, board: _Board, span: CardSpan) -> tuple[str, list[str], dict[str, str]]:
        tokens = markdown_tokens(board.raw.decode("utf-8"))
        dates, warnings = lifecycle_dates(
            tokens, [line.rstrip("\r\n") for line in board.lines], span.start, span.end, board.relative
        )
        if warnings:
            raise ValueError("Repair the task's invalid bedrock-dates block before editing")
        removed: set[int] = set()
        for token in tokens:
            if (
                token.type == "fence"
                and token.info.strip() == "bedrock-dates"
                and token.map
                and span.start <= token.map[0] < span.end
            ):
                removed.update(range(*token.map))
        first = CARD_RE.match(board.lines[span.start].rstrip("\r\n")).group(2)
        title = BLOCK_ID_RE.sub("", first)
        body = [
            line.rstrip("\r\n")
            for index, line in enumerate(board.lines)
            if span.start < index < span.end and index not in removed
        ]
        while body and not body[-1].strip():
            body.pop()
        return title, body, dates

    def _view(self, board: _Board, column: ColumnSpan, span: CardSpan, entry: ParsedEntry) -> EditableTask:
        title, body, dates = self._parts(board, span)
        markdown = "".join(board.lines[span.start : span.end])
        if len(markdown) > MAX_CARD_CHARS:
            raise ValueError("Task exceeds the editing limit; use paginated read_note")
        return EditableTask(
            task_id=span.block_id or entry.entry_id,
            version=_card_version(board, column, span),
            entry_id=entry.entry_id,
            board=board.relative,
            column=column.name,
            archived=column.archived,
            checked=entry.board.checked,
            title=title,
            content=dedent("\n".join(body)).strip("\n"),
            markdown=markdown,
            **{key: dates.get(key) for key in TASK_DATE_FIELDS},
        )

    @staticmethod
    def _render(
        title: str, body: list[str], task_id: str, dates: dict[str, str], checked: bool, newline: str
    ) -> str:
        lines = [f"- [{'x' if checked else ' '}] {title} ^{task_id}", *body]
        if dates:
            lines += [
                "",
                "    ```bedrock-dates",
                *[f"    {key}: {dates[key]}" for key in TASK_DATE_FIELDS if key in dates],
                "    ```",
            ]
        result = newline.join(lines) + newline
        if len(result) > MAX_CARD_CHARS:
            raise ValueError("Task exceeds the editing limit")
        return result

    @staticmethod
    def _insert(board: _Board, column: ColumnSpan, card: str, position: int | None) -> str:
        count = len(column.cards)
        position = count if position is None else position
        if not 0 <= position <= count:
            raise ValueError(f"position must be between 0 and {count}")
        at = (
            column.cards[position].start
            if position < count
            else column.cards[-1].end
            if count
            else column.end
        )
        lines = board.lines.copy()
        newline = _newline(board)
        # Blank separators keep metadata fences inside their parent list item.
        prefix = newline if at and not lines[at - 1].endswith(("\n", "\r")) else ""
        lines[at:at] = (prefix + card + newline).splitlines(keepends=True)
        return "".join(lines)

    def _relocate(
        self, board: _Board, span: CardSpan, target: ColumnSpan, card: str, task_id: str, position: int | None
    ) -> KanbanWrite:
        # Insert and remove as independent source patches, ordered backwards.
        remaining = tuple(item for item in target.cards if item.start != span.start)
        position = len(remaining) if position is None else position
        if not 0 <= position <= len(remaining):
            raise ValueError(f"position must be between 0 and {len(remaining)}")
        at = (
            remaining[position].start
            if position < len(remaining)
            else remaining[-1].end
            if remaining
            else target.end
        )
        if span.start <= at <= span.end:
            at = span.start
        lines = board.lines.copy()
        if at >= span.end:
            lines[at:at] = (card + _newline(board)).splitlines(keepends=True)
            del lines[span.start : span.end]
        else:
            del lines[span.start : span.end]
            lines[at:at] = (card + _newline(board)).splitlines(keepends=True)
        return self._commit(board, "".join(lines), task_id, span)

    def _commit(self, board: _Board, text: str, task_id: str, edited: CardSpan | None = None) -> KanbanWrite:
        if board.path.read_bytes() != board.raw:
            raise ValueError("Board changed during editing; get_task/read_note again before retrying")
        self._validate_transition(board, text, task_id, edited)
        backup = self._backup(board)
        self._save(board.path, text.encode("utf-8"), board.raw)
        return self._result(self._load(board.relative), task_id, backup=backup)

    @staticmethod
    def _validate_transition(board: _Board, text: str, task_id: str, edited: CardSpan | None) -> None:
        # Validate the proposed structure before replacing any source file.
        columns = board_columns(
            markdown_tokens(text),
            text.splitlines(keepends=True),
            parse_frontmatter(text).body_start,
            parse_settings(text.splitlines(), board.relative)[2] or len(text.splitlines()),
        )
        proposed_lines = text.splitlines(keepends=True)

        def fingerprint(lines: list[str], span: CardSpan) -> str:
            return hashlib.sha256("".join(lines[span.start : span.end]).rstrip("\r\n").encode()).hexdigest()

        untouched = Counter(
            fingerprint(board.lines, span) for col in board.columns for span in col.cards if span != edited
        )
        proposed = Counter(fingerprint(proposed_lines, span) for col in columns for span in col.cards)
        after_ids = [span.block_id for col in columns for span in col.cards]
        count_before = sum(len(col.cards) for col in board.columns)
        if (
            after_ids.count(task_id) != 1
            or len(after_ids) != count_before + (edited is None)
            or untouched - proposed
        ):
            raise ValueError("Edit would lose or alter another task; no changes saved")

    @staticmethod
    def _backup(board: _Board) -> str:
        directory = board.path.parent / ".bedrock-backups"
        if directory.is_symlink():
            raise ValueError("Backup directory cannot be a symlink")
        directory.mkdir(mode=0o700, exist_ok=True)
        destination = directory / f"{board.path.name}.{hashlib.sha256(board.raw).hexdigest()}.bak"
        if destination.is_symlink():
            raise ValueError("Backup cannot be a symlink")
        try:
            with destination.open("xb") as stream:
                stream.write(board.raw)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            if destination.read_bytes() != board.raw:
                raise ValueError("Existing backup does not match this board revision") from None
        return str(destination)

    @staticmethod
    def _save(path: Path, raw: bytes, original: bytes | None) -> None:
        descriptor, name = tempfile.mkstemp(prefix=".bedrock-", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
                os.fchmod(
                    stream.fileno(), stat.S_IMODE(path.stat().st_mode) if original is not None else 0o644
                )
            if original is None:
                os.link(temporary, path)  # Exclusive publication: never replace an existing file.
            else:
                if path.is_symlink() or path.read_bytes() != original:
                    raise ValueError("Board changed before saving; reread before retrying")
                os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _result(
        self, board: _Board, task_id: str | None = None, *, replayed: bool = False, backup: str | None = None
    ) -> KanbanWrite:
        task = None
        if task_id:
            column, span, entry = self._task(board, task_id)
            task = self._view(board, column, span, entry)
        try:
            self.vault.reindex([board.relative])
            status, error = "updated", None
        except Exception as failure:
            status, error = "failed", str(failure)
        return KanbanWrite(
            path=board.relative,
            content_hash=board.revision,
            columns=[col.name for col in board.columns if not col.archived],
            task=task,
            replayed=replayed,
            index_status=status,
            index_error=error,
            backup_path=backup,
        )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _hash(raw: bytes) -> str:
    # Match read_note's universal-newline text hash, retaining raw bytes for backups/CAS.
    text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _revision(board: _Board, expected: str) -> None:
    if not expected or expected != board.revision:
        raise ValueError("Board revision conflict; get_task/read_note again and use its content_hash")


def _newline(board: _Board) -> str:
    return "\r\n" if b"\r\n" in board.raw else "\n"


def _request(value: str) -> str:
    if not value.strip() or len(value) > 128:
        raise ValueError(
            "request_id must be a unique nonempty string of at most 128 characters; reuse it only for retries"
        )
    return value


def _single(value: str, label: str) -> str:
    if (
        not value.strip()
        or len(value) > 1000
        or any(char in value for char in "\r\n\x00")
        or BLOCK_ID_RE.search(value)
    ):
        raise ValueError(f"{label} must be a nonempty single line without a trailing block ID")
    return value.strip()


def _body(value: str) -> None:
    if (
        len(value) > MAX_CARD_CHARS
        or "\x00" in value
        or "%% kanban:settings" in value
        or any(
            token.type == "fence" and token.info.strip() == "bedrock-dates"
            for token in markdown_tokens(value)
        )
    ):
        raise ValueError(
            "Content must fit the card limit and cannot contain reserved bedrock-dates or Kanban settings blocks"
        )


def _status(column: ColumnSpan) -> str | None:
    return "complete" if column.complete else canonical_status(column.name)


@contextmanager
def _locked(path: Path, root: Path) -> Iterator[None]:
    directory = _internal_directory(root, ".bedrock-state", "locks")
    if directory.is_symlink():
        raise ValueError("Lock directory cannot be a symlink")
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.stat().st_uid != os.getuid():
        raise ValueError("Lock directory belongs to another user")
    name = hashlib.sha256(path.relative_to(root).as_posix().encode()).hexdigest()
    descriptor = os.open(directory / name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _internal_directory(root: Path, *parts: str) -> Path:
    current = root
    if current.is_symlink():
        raise ValueError("Internal storage root cannot be a symlink")
    current.mkdir(parents=True, exist_ok=True)
    for part in parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Internal storage directories cannot be symlinks")
        current.mkdir(mode=0o700, exist_ok=True)
    return current


def _card_version(board: _Board, column: ColumnSpan, span: CardSpan) -> str:
    raw = "".join(board.lines[span.start : span.end])
    return kanban_card_version(column.name, column.archived, raw)
