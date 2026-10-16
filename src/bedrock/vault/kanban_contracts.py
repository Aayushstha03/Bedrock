"""Results of source-based Kanban reads and writes."""

from typing import Literal

from pydantic import BaseModel


class EditableTask(BaseModel):
    version: str
    task_id: str
    entry_id: str
    board: str
    column: str
    archived: bool
    checked: bool
    title: str
    content: str
    markdown: str
    created_at: str | None
    started_at: str | None
    completed_at: str | None
    due_date: str | None


class TaskRead(BaseModel):
    content_hash: str
    task: EditableTask


class KanbanWrite(BaseModel):
    path: str
    content_hash: str
    columns: list[str]
    task: EditableTask | None = None
    saved: bool = True
    replayed: bool = False
    index_status: Literal["updated", "failed"]
    index_error: str | None = None
    backup_path: str | None = None


class NewCard(BaseModel):
    """One append command; no board hash or insertion index is needed."""

    model_config = {"extra": "forbid", "strict": True}
    title: str
    column: str
    content: str = ""
    due_date: str | None = None


class CardEdit(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    task_id: str
    version: str
    title: str | None = None
    content: str | None = None
    due_date: str | None = None
    created_at: str | None = None


class CardMove(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    task_id: str
    column: str
    from_column: str | None = None
    position: int | None = None


class CardTarget(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    task_id: str


class CardSummary(BaseModel):
    task_id: str
    entry_id: str
    version: str
    title: str
    column: str
    archived: bool
    checked: bool
    created_at: str | None
    started_at: str | None
    completed_at: str | None
    due_date: str | None


class CardsWrite(BaseModel):
    path: str
    content_hash: str
    items: list[CardSummary]
    saved: bool = True
    replayed: bool = False
    index_status: Literal["updated", "failed"]
    index_error: str | None = None
    backup_path: str | None = None


class BoardDeletion(BaseModel):
    path: str
    status: Literal["preview", "deleted"]
    warning: str
    card_count: int
    confirmation_token: str
    saved: bool = False
    replayed: bool = False
    trash_path: str | None = None
    index_status: Literal["updated", "failed"] | None = None
    index_error: str | None = None
