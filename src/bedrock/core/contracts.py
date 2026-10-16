"""Storage-independent contracts shared by parsing and indexing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from bedrock.core.task_dates import validate_task_date_ranges


class NoteType(StrEnum):
    LOG = "log"
    STANDARD = "standard"
    KANBAN = "kanban"


class EntryType(StrEnum):
    DAILY_SECTION = "daily_section"
    PROJECT_UPDATE = "project_update"
    FREEFORM_SECTION = "freeform_section"
    FREEFORM_CHUNK = "freeform_chunk"
    KANBAN_CARD = "kanban_card"


class LinkResolution(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"


class WarningType(StrEnum):
    INVALID_DATE = "invalid_date"
    UNKNOWN_TAG = "unknown_tag"
    AMBIGUOUS_LINK = "ambiguous_link"
    MISSING_LINK = "missing_link"
    INVALID_KANBAN_DATES = "invalid_kanban_dates"
    INVALID_KANBAN_SETTINGS = "invalid_kanban_settings"
    MISSING_KANBAN_MARKER = "missing_kanban_marker"
    TRUNCATED_EMBEDDING_INPUT = "truncated_embedding_input"
    PARSER_FAILURE = "parser_failure"


class RetrievalMode(StrEnum):
    METADATA = "metadata"
    LITERAL = "literal"
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class ResultOrder(StrEnum):
    DOCUMENT = "document"
    DATE_DESC = "date_desc"


class ContextLabel(StrEnum):
    DIRECT_EVIDENCE = "direct_evidence"
    FOLLOWED_LINK = "followed_link"
    BACKLINK = "backlink"
    NEARBY_HISTORY = "nearby_history"
    INFERENCE_INPUT = "inference_input"


class GraphEdgeType(StrEnum):
    INTERNAL_LINK = "internal_link"
    SHARED_TAG = "shared_tag"
    SHARED_DATE = "shared_date"


@dataclass(frozen=True, slots=True)
class SourceRange:
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError("source range must use positive inclusive lines")


@dataclass(frozen=True, slots=True)
class Link:
    target_text: str
    target_note_id: str | None = None
    target_heading: str | None = None
    alias: str | None = None
    resolution: LinkResolution = LinkResolution.MISSING
    line: int | None = None


@dataclass(frozen=True, slots=True)
class KanbanData:
    column: str
    status: str | None
    column_position: int
    card_position: int
    checked: bool
    task_id: str | None = None
    archived: bool = False
    card_time: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    due_date: str | None = None


@dataclass(frozen=True, slots=True)
class NoteRef:
    """How every contract identifies the note it belongs to."""

    note_id: str
    path: str
    title: str
    note_type: NoteType


@dataclass(frozen=True, slots=True)
class NoteSummary(NoteRef):
    """One vault-catalog note, including notes with no indexed entries."""

    indexed_entries: int


@dataclass(frozen=True, slots=True)
class NoteHeader:
    """The little of a note that vault-wide link resolution needs.

    A full parse produces one of these, and so does a cheap read of the file
    head, which is what lets a scoped update resolve links without reparsing
    every note in the vault.
    """

    note_id: str
    path: str
    title: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedEntry:
    entry_id: str
    note_id: str
    path: str
    note_title: str
    note_type: NoteType
    entry_type: EntryType
    text: str
    embedding_text: str
    heading: str | None
    heading_path: tuple[str, ...]
    source: SourceRange
    note_date: str | None = None
    entry_date: str | None = None
    tags: tuple[str, ...] = ()
    outgoing_links: tuple[Link, ...] = ()
    web_links: tuple[str, ...] = ()
    content_hash: str = ""
    board: KanbanData | None = None
    entry_date_end: str | None = None


@dataclass(frozen=True, slots=True)
class IndexWarning:
    kind: WarningType
    path: str
    message: str
    line: int | None = None


@dataclass(frozen=True, slots=True)
class ParsedNote(NoteRef):
    content_hash: str
    entries: tuple[ParsedEntry, ...]
    warnings: tuple[IndexWarning, ...] = ()
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class QueryPlan:
    mode: RetrievalMode = RetrievalMode.HYBRID
    date_from: str | None = None
    date_to: str | None = None
    tags_all: tuple[str, ...] = ()
    tags_any: tuple[str, ...] = ()
    note_id: str | None = None
    section: str | None = None
    entry_types: tuple[EntryType, ...] = ()
    literal_text: str | None = None
    lexical_text: str | None = None
    semantic_text: str | None = None
    link_depth: int = 1
    nearby_days: int = 3
    max_linked_notes: int = 5
    limit: int = 10
    offset: int = 0
    order: ResultOrder = ResultOrder.DOCUMENT
    kanban_board: str | None = None
    kanban_column: str | None = None
    kanban_status: str | None = None
    kanban_columns: tuple[str, ...] = ()
    kanban_statuses: tuple[str, ...] = ()
    kanban_archived: bool | None = None
    kanban_checked: bool | None = None
    created_at_from: str | None = None
    created_at_to: str | None = None
    started_at_from: str | None = None
    started_at_to: str | None = None
    completed_at_from: str | None = None
    completed_at_to: str | None = None
    due_date_from: str | None = None
    due_date_to: str | None = None

    def __post_init__(self) -> None:
        validate_task_date_ranges(self)
        if not 0 <= self.link_depth <= 2:
            raise ValueError("link_depth must be between 0 and 2")
        if not 1 <= self.max_linked_notes <= 5:
            raise ValueError("max_linked_notes must be between 1 and 5")
        if self.limit <= 0:
            raise ValueError("limit must be positive")
        if self.offset < 0:
            object.__setattr__(self, "offset", 0)


@dataclass(frozen=True, slots=True)
class Page[PageItem]:
    """A bounded page over a deterministic or relevance-ranked result set."""

    items: tuple[PageItem, ...]
    has_more: bool
    next_offset: int | None = None


# Which query text each retrieval mode consumes. One mapping, so a caller that
# turns a mode and a query into a plan never has to restate the rule.
QUERY_TEXT_FIELDS: dict[RetrievalMode, tuple[str, ...]] = {
    RetrievalMode.METADATA: (),
    RetrievalMode.LITERAL: ("literal_text",),
    RetrievalMode.LEXICAL: ("lexical_text",),
    RetrievalMode.SEMANTIC: ("semantic_text",),
    RetrievalMode.HYBRID: ("lexical_text", "semantic_text"),
}


@dataclass(frozen=True, slots=True)
class QdrantPayload:
    vault_id: str
    note_id: str
    entry_id: str
    path: str
    note_title: str
    note_type: NoteType
    entry_type: EntryType
    text: str
    heading: str | None
    heading_path: tuple[str, ...]
    start_line: int
    end_line: int
    note_date: str | None
    entry_date: str | None
    tags: tuple[str, ...]
    outgoing_note_ids: tuple[str, ...]
    outgoing_links: tuple[Link, ...]
    web_links: tuple[str, ...]
    content_hash: str
    modified_at: str
    # One digest over every input that decides whether a stored vector is still
    # valid: the parser, both model names, the encoder and tokenizer versions,
    # the embedding-input format, and the embedding text itself. A point whose
    # fingerprint no longer matches is re-embedded.
    embedding_fingerprint: str
    # The other names this note answers to, so a scoped update can tell which
    # links a rename or a retitle just broke without reparsing the vault.
    note_aliases: tuple[str, ...] = ()
    # One normalized lookup key per outgoing link, exactly the string link
    # resolution compares. Indexed, so the notes that point at a changed note
    # can be found with a query instead of a full-vault parse.
    outgoing_link_keys: tuple[str, ...] = ()
    board: KanbanData | None = None
    entry_date_end: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class SearchResult:
    entry_id: str
    note_id: str
    path: str
    note_title: str
    note_type: NoteType
    entry_type: EntryType
    mode: RetrievalMode
    text: str
    heading: str | None
    heading_path: tuple[str, ...]
    start_line: int
    end_line: int
    note_date: str | None
    entry_date: str | None
    tags: tuple[str, ...]
    outgoing_links: tuple[Link, ...]
    score: float | None = None
    verified: bool | None = None
    board: KanbanData | None = None
    entry_date_end: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class NoteView(NoteRef):
    entries: tuple[SearchResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class NoteContent(NoteRef):
    """One complete Markdown note, exactly as it is stored in the vault."""

    content: str

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class KanbanBoard:
    note_id: str
    path: str
    title: str
    columns: tuple[str, ...]
    cards: tuple[SearchResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class ContextItem:
    result: SearchResult
    labels: tuple[ContextLabel, ...]
    depth: int
    via_note_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class TraversalDiagnostic:
    source_note_id: str
    target_text: str
    resolution: LinkResolution
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class SectionContinuation:
    """Read omitted section content with read_note, starting at start_line."""

    note_id: str
    section: str
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class ContextExpansion:
    source_entry_id: str
    items: tuple[ContextItem, ...]
    diagnostics: tuple[TraversalDiagnostic, ...] = ()
    inspected_note_ids: tuple[str, ...] = ()
    truncated: bool = False
    budget_reasons: tuple[str, ...] = ()
    section_continuations: tuple[SectionContinuation, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class NetworkNode(NoteRef):
    tags: tuple[str, ...] = ()
    dates: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NetworkEdge:
    edge_id: str
    source_note_id: str
    target_note_id: str
    edge_type: GraphEdgeType
    source_entry_id: str | None = None
    target_entry_id: str | None = None
    line: int | None = None
    target_heading: str | None = None
    tags: tuple[str, ...] = ()
    date: str | None = None
    date_field: str | None = None


@dataclass(frozen=True, slots=True)
class NetworkGraph:
    nodes: tuple[NetworkNode, ...]
    edges: tuple[NetworkEdge, ...]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
