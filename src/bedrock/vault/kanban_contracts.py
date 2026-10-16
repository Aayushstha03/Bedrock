"""Results of source-based Kanban reads and writes."""

from typing import Literal

from pydantic import BaseModel


class EditableTask(BaseModel):
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
