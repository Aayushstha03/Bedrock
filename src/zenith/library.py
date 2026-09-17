"""Composable public library over Zenith's parser, index, and retrieval layers."""

from __future__ import annotations

import unicodedata
from collections import Counter
from pathlib import PurePosixPath
from typing import Any

from qdrant_client import QdrantClient

from zenith.core.config import Settings
from zenith.core.contracts import (
    ContextExpansion,
    EntryType,
    IndexWarning,
    KanbanBoard,
    Link,
    NetworkGraph,
    NoteContent,
    NoteSummary,
    NoteType,
    NoteView,
    Page,
    ParsedNote,
    QueryPlan,
    ResultOrder,
    RetrievalMode,
    SearchResult,
    WarningType,
)
from zenith.index.graph import GraphExporter
from zenith.index.incremental import IncrementalIndexer, IncrementalReport
from zenith.index.links import resolve_link, resolve_links
from zenith.index.rebuild import IndexRebuilder, RebuildReport
from zenith.parser.discovery import discover_markdown
from zenith.parser.service import VaultParser
from zenith.retrieval.context import ContextExpander
from zenith.retrieval.service import Retriever


class Zenith:
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
        self.encoders = encoders
        self._notes: tuple[ParsedNote, ...] | None = None
        self._resolved: tuple[ParsedNote, ...] | None = None
        self._fingerprint: tuple[tuple[str, int, int], ...] | None = None
        self.retriever = Retriever(
            settings,
            vault_id=vault_id,
            client=self.client,
            encoders=encoders,
        )

    def reindex(
        self, paths: list[str] | None = None, *, full: bool = False
    ) -> RebuildReport | IncrementalReport:
        if full:
            if paths:
                raise ValueError("full rebuild does not accept paths")
            return IndexRebuilder(
                self.settings,
                vault_id=self.vault_id,
                client=self.client,
                encoders=self.encoders,
            ).rebuild()
        return IncrementalIndexer(
            self.settings,
            vault_id=self.vault_id,
            client=self.client,
            encoders=self.encoders,
        ).reindex(paths)

    def find_entries(
        self,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        tags_all: tuple[str, ...] = (),
        tags_any: tuple[str, ...] = (),
        note: str | None = None,
        section: str | None = None,
        entry_types: tuple[EntryType, ...] = (),
        exact_text: str | None = None,
        lexical_text: str | None = None,
        semantic_text: str | None = None,
        mode: RetrievalMode | None = None,
        limit: int = 10,
    ) -> tuple[SearchResult, ...]:
        return self.find_entries_page(
            date_from=date_from,
            date_to=date_to,
            tags_all=tags_all,
            tags_any=tags_any,
            note=note,
            section=section,
            entry_types=entry_types,
            exact_text=exact_text,
            lexical_text=lexical_text,
            semantic_text=semantic_text,
            mode=mode,
            limit=limit,
        ).items

    def find_entries_page(
        self,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        tags_all: tuple[str, ...] = (),
        tags_any: tuple[str, ...] = (),
        note: str | None = None,
        section: str | None = None,
        entry_types: tuple[EntryType, ...] = (),
        exact_text: str | None = None,
        lexical_text: str | None = None,
        semantic_text: str | None = None,
        mode: RetrievalMode | None = None,
        limit: int = 10,
        offset: int = 0,
        order: ResultOrder = ResultOrder.DOCUMENT,
    ) -> Page[SearchResult]:
        note_id = self.get_note(note).note_id if note is not None else None
        selected_mode = _mode(mode, exact_text, lexical_text, semantic_text)
        return self.retriever.search_page(
            QueryPlan(
                mode=selected_mode,
                date_from=date_from,
                date_to=date_to,
                tags_all=tags_all,
                tags_any=tags_any,
                note_id=note_id,
                section=section,
                entry_types=entry_types,
                literal_text=exact_text,
                lexical_text=lexical_text,
                semantic_text=semantic_text,
                limit=limit,
                offset=offset,
                order=order,
            )
        )

    def get_note(self, path_or_title: str) -> NoteView:
        note = self._match_note(path_or_title)
        entries = self.retriever.get_note_entries(note.note_id)
        return NoteView(note.note_id, note.path, note.title, note.note_type, entries)

    def read_note(self, path_or_title: str) -> NoteContent:
        """Read one complete note from the vault as Markdown.

        Indexed entries are split to fit the dense model and carry cleaned
        prose. This returns the file itself, including frontmatter, code, and
        URLs, for the cases where the whole note is the answer.
        """
        note = self._match_note(path_or_title)
        vault = self.settings.vault_path.resolve()
        # `note.path` comes from vault discovery, so it is already inside the
        # vault and outside every excluded directory. Re-check anyway: no
        # caller-supplied string may ever become a path that leaves the mount.
        resolved = (vault / note.path).resolve()
        if not resolved.is_relative_to(vault):
            raise ValueError(f"path escapes vault: {note.path}")
        return NoteContent(
            note_id=note.note_id,
            path=note.path,
            title=note.title,
            note_type=note.note_type,
            content=resolved.read_text(encoding="utf-8"),
        )

    def _match_note(self, path_or_title: str) -> ParsedNote:
        notes = self._parsed_notes()
        needle = _key(path_or_title)
        path_matches = [
            note
            for note in notes
            if needle
            in {
                _key(note.path),
                _key(PurePosixPath(note.path).with_suffix("").as_posix()),
            }
        ]
        title_matches = [note for note in notes if _key(note.title) == needle]
        stem_matches = [note for note in notes if _key(PurePosixPath(note.path).stem) == needle]
        matches = path_matches or title_matches or stem_matches
        if not matches:
            raise LookupError(f"note not found: {path_or_title}")
        if len(matches) > 1:
            paths = ", ".join(sorted(note.path for note in matches))
            raise ValueError(f"ambiguous note title {path_or_title!r}: {paths}")
        return matches[0]

    def get_entry(self, entry_id: str) -> SearchResult:
        result = self.retriever.get_entry(entry_id)
        if result is None:
            raise LookupError(f"entry not found: {entry_id}")
        return result

    def get_outgoing_links(self, note_id: str, entry_id: str | None = None) -> tuple[Link, ...]:
        if entry_id is not None:
            entry = self.get_entry(entry_id)
            if entry.note_id != note_id:
                raise ValueError(f"entry {entry_id} does not belong to note {note_id}")
            return entry.outgoing_links
        entries = self._entries_for_note(note_id)
        return tuple(link for entry in entries for link in entry.outgoing_links)

    def get_backlinks(self, note_id: str) -> tuple[SearchResult, ...]:
        self._entries_for_note(note_id)
        return self.get_backlinks_page(note_id).items

    def get_backlinks_page(self, note_id: str, *, limit: int = 10, offset: int = 0) -> Page[SearchResult]:
        self._entries_for_note(note_id)
        return self.retriever.get_backlinks_page(note_id, limit=limit, offset=offset)

    def resolve_link(self, source_note_id: str, target_text: str) -> Link:
        notes = self._parsed_notes()
        return resolve_link(notes, source_note_id, target_text)

    def search_within(
        self,
        note_id: str,
        query: str,
        mode: RetrievalMode,
        section: str | None = None,
        *,
        limit: int = 10,
    ) -> tuple[SearchResult, ...]:
        self._entries_for_note(note_id)
        return self.retriever.search_within(note_id, query, mode=mode, section=section, limit=limit)

    def expand_context(
        self,
        entry_id: str,
        *,
        link_depth: int = 1,
        nearby_days: int = 3,
        max_notes: int = 5,
    ) -> ContextExpansion:
        return ContextExpander(self.retriever).expand(
            entry_id,
            link_depth=link_depth,
            nearby_days=nearby_days,
            max_notes=max_notes,
        )

    def get_index_warnings(
        self, kind: WarningType | None = None, path: str | None = None
    ) -> tuple[IndexWarning, ...]:
        notes = self._resolved_notes()
        warnings = [warning for note in notes for warning in note.warnings]
        if kind is not None:
            warnings = [warning for warning in warnings if warning.kind is kind]
        if path is not None:
            warnings = [warning for warning in warnings if _key(warning.path) == _key(path)]
        return tuple(sorted(warnings, key=lambda item: (item.path, item.line or 0, item.kind.value)))

    def list_kanban_boards(self) -> tuple[KanbanBoard, ...]:
        cards = self.retriever.metadata_all(
            QueryPlan(
                mode=RetrievalMode.METADATA,
                entry_types=(EntryType.KANBAN_CARD,),
            )
        )
        grouped: dict[str, list[SearchResult]] = {}
        for card in cards:
            grouped.setdefault(card.note_id, []).append(card)
        boards = [_board(group) for group in grouped.values()]
        return tuple(sorted(boards, key=lambda board: (board.path, board.note_id)))

    def get_kanban_board(self, board_name_or_path: str) -> KanbanBoard:
        needle = _key(board_name_or_path)
        boards = self.list_kanban_boards()
        path_matches = [
            board
            for board in boards
            if needle in {_key(board.path), _key(PurePosixPath(board.path).with_suffix("").as_posix())}
        ]
        matches = path_matches or [board for board in boards if _key(board.title) == needle]
        if not matches:
            raise LookupError(f"Kanban board not found: {board_name_or_path}")
        if len(matches) > 1:
            raise ValueError(f"ambiguous Kanban board: {board_name_or_path}")
        return matches[0]

    def find_kanban_cards(
        self,
        *,
        board: str | None = None,
        columns: tuple[str, ...] = (),
        statuses: tuple[str, ...] = (),
        checked: bool | None = None,
        tags: tuple[str, ...] = (),
        date_from: str | None = None,
        date_to: str | None = None,
        exact_text: str | None = None,
        semantic_text: str | None = None,
        limit: int = 10,
    ) -> tuple[SearchResult, ...]:
        return self.find_kanban_cards_page(
            board=board,
            columns=columns,
            statuses=statuses,
            checked=checked,
            tags=tags,
            date_from=date_from,
            date_to=date_to,
            exact_text=exact_text,
            semantic_text=semantic_text,
            limit=limit,
        ).items

    def find_kanban_cards_page(
        self,
        *,
        board: str | None = None,
        columns: tuple[str, ...] = (),
        statuses: tuple[str, ...] = (),
        checked: bool | None = None,
        tags: tuple[str, ...] = (),
        date_from: str | None = None,
        date_to: str | None = None,
        exact_text: str | None = None,
        semantic_text: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> Page[SearchResult]:
        board_name = self.get_kanban_board(board).title if board is not None else None
        selected_mode = _mode(None, exact_text, None, semantic_text)
        return self.retriever.search_page(
            QueryPlan(
                mode=selected_mode,
                tags_all=tags,
                date_from=date_from,
                date_to=date_to,
                entry_types=(EntryType.KANBAN_CARD,),
                literal_text=exact_text,
                semantic_text=semantic_text,
                kanban_board=board_name,
                kanban_columns=columns,
                kanban_statuses=statuses,
                kanban_checked=checked,
                limit=limit,
                offset=offset,
            )
        )

    def list_notes_page(
        self, *, note_type: NoteType | None = None, limit: int = 20, offset: int = 0
    ) -> Page[NoteSummary]:
        """List every vault note, including empty notes Qdrant cannot represent."""
        notes = self._parsed_notes()
        counts = Counter(entry.note_id for entry in self.retriever.metadata_all())
        summaries = [
            NoteSummary(note.note_id, note.path, note.title, note.note_type, counts[note.note_id])
            for note in notes
            if note_type is None or note.note_type is note_type
        ]
        summaries.sort(key=lambda note: _key(note.path))
        start = max(0, offset)
        selected = summaries[start : start + limit + 1]
        has_more = len(selected) > limit
        if has_more:
            selected.pop()
        return Page(tuple(selected), has_more, start + limit if has_more else None)

    def export_graph(self) -> NetworkGraph:
        return GraphExporter(self.settings, vault_id=self.vault_id, client=self.client).export()

    def _entries_for_note(self, note_id: str) -> tuple[SearchResult, ...]:
        results = self.retriever.get_note_entries(note_id)
        if not results and not any(note.note_id == note_id for note in self._parsed_notes()):
            raise LookupError(f"note not found: {note_id}")
        return results

    def _parsed_notes(self) -> tuple[ParsedNote, ...]:
        """Return the parsed vault, reparsing only when a Markdown file changed.

        Every call fingerprints the vault: one `rglob` walk, and roughly three
        stat calls per note between `discover_markdown` and this function. That
        is still far cheaper than reading, tokenizing, and chunking every file,
        and it notices edits made by the watcher or by any other process, not
        only edits made through `reindex`.
        """
        fingerprint = self._vault_fingerprint()
        if self._notes is None or self._fingerprint != fingerprint:
            self._notes = VaultParser(self.settings, self.vault_id).parse_vault()
            self._fingerprint = fingerprint
            self._resolved = None
        return self._notes

    def _resolved_notes(self) -> tuple[ParsedNote, ...]:
        notes = self._parsed_notes()
        if self._resolved is None:
            self._resolved = resolve_links(notes)
        return self._resolved

    def _vault_fingerprint(self) -> tuple[tuple[str, int, int], ...]:
        vault = self.settings.vault_path.resolve()
        fingerprint: list[tuple[str, int, int]] = []
        for path in discover_markdown(self.settings):
            stat = path.stat()
            fingerprint.append((path.relative_to(vault).as_posix(), stat.st_mtime_ns, stat.st_size))
        return tuple(fingerprint)


def _mode(
    requested: RetrievalMode | None,
    exact_text: str | None,
    lexical_text: str | None,
    semantic_text: str | None,
) -> RetrievalMode:
    if requested is not None:
        return requested
    if exact_text is not None:
        return RetrievalMode.LITERAL
    if lexical_text is not None and semantic_text is not None:
        return RetrievalMode.HYBRID
    if lexical_text is not None:
        return RetrievalMode.LEXICAL
    if semantic_text is not None:
        return RetrievalMode.SEMANTIC
    return RetrievalMode.METADATA


def _board(cards: list[SearchResult]) -> KanbanBoard:
    ordered = tuple(
        sorted(
            cards,
            key=lambda card: (
                card.board.column_position if card.board else 0,
                card.board.card_position if card.board else 0,
                card.entry_id,
            ),
        )
    )
    first = ordered[0]
    assert first.board is not None
    columns = tuple(dict.fromkeys(card.board.column for card in ordered if card.board is not None))
    return KanbanBoard(first.note_id, first.path, first.note_title, columns, ordered)


def _key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold().strip().replace("\\", "/")
