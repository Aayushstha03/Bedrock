"""Version-one structured tool outputs, independent of any agent framework."""

from typing import Literal

from pydantic import BaseModel, Field

from bedrock.core.contracts import EntryType, NoteType, RetrievalMode


class Task(BaseModel):
    task_id: str | None = None
    archived: bool = False
    column: str
    status: str | None
    checked: bool
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    due_date: str | None = None
    card_time: str | None


class Evidence(BaseModel):
    entry_id: str
    note_id: str
    path: str
    title: str
    note_type: NoteType
    entry_type: EntryType
    heading: str | None
    heading_path: list[str]
    start_line: int
    end_line: int
    note_date: str | None
    entry_date: str | None
    entry_date_end: str | None = None
    tags: list[str]
    mode: RetrievalMode
    text: str
    text_truncated: bool = False
    exact_match_verified: bool | None = None
    task: Task | None = None


class EntryPage(BaseModel):
    items: list[Evidence]
    has_more: bool
    next_offset: int | None = None


class NoteInfo(BaseModel):
    note_id: str
    path: str
    title: str
    note_type: NoteType
    indexed_entries: int


class NotePage(BaseModel):
    items: list[NoteInfo]
    has_more: bool
    next_offset: int | None = None


class NoteRead(BaseModel):
    note_id: str
    path: str
    title: str
    content: str
    content_hash: str
    total_lines: int
    start_line: int
    end_line: int | None
    has_more: bool
    next_line: int | None = None
    # A giant single line is paged by character offset instead of becoming
    # permanently unreadable or overrunning the response budget.
    next_char_offset: int | None = None


class BoardInfo(BaseModel):
    note_id: str
    path: str
    title: str
    columns: list[str]
    indexed_cards: int


class BoardPage(BaseModel):
    items: list[BoardInfo]
    has_more: bool
    next_offset: int | None = None


class RelatedEvidence(Evidence):
    relationships: list[str]
    hops: int
    via_note_id: str | None


class SectionContinuation(BaseModel):
    note_id: str
    section: str
    start_line: int
    end_line: int


class ContextResult(BaseModel):
    source_entry_id: str
    items: list[RelatedEvidence]
    unresolved_links: list[dict[str, str | int | None]]
    truncated: bool
    budget_reasons: list[str]
    inspected_note_ids: list[str]
    section_continuations: list[SectionContinuation] = Field(default_factory=list)


class VaultStatus(BaseModel):
    contract_version: str = "1"
    ready: bool
    index: dict[str, object]
    vault_available: bool
    freshness: Literal["current", "stale", "unknown"]
    freshness_scope: str = "searchable entries; observation is not an atomic snapshot"
    watcher_state: Literal["unknown"] = "unknown"
    pending_entries: int | None = None
    errors: list[str] = Field(default_factory=list)


class TagInfo(BaseModel):
    name: str
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)


class TagList(BaseModel):
    items: list[TagInfo]
