"""The tool surface the answering model is given over the notes vault.

Deliberately narrow. Every tool takes flat arguments, returns a compact
projection, and reports a failure the model can act on rather than raising.
The library keeps the wider surface; `reindex` writes and `export_graph` is
unbounded, so neither is offered here.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, Any

from pydantic import Field
from pydantic_ai import ModelRetry, RunContext

from bedrock.agent import projections
from bedrock.agent.sources import Sources
from bedrock.core.contracts import NoteType, ResultOrder
from bedrock.vault.kanban_contracts import CardEdit, CardMove, CardTarget, NewCard

_SOURCE_ID = re.compile(r"s\d+", re.IGNORECASE)


def search_notes(
    ctx: RunContext[Sources],
    query: str | None = None,
    exact: bool = False,
    date_from: str | None = None,
    date_to: str | None = None,
    tags: list[str] | None = None,
    limit: int = 5,
    offset: int = 0,
) -> dict[str, Any]:
    """Search the notes vault and return the entries that match.

    Args:
        query: What to look for. Omit it to browse entries by date instead.
        exact: Return only entries containing the query as a literal phrase.
            Use this when the question depends on the exact wording. Results
            then carry `exact_match_verified`.
        date_from: Earliest date to include, written as YYYY-MM-DD.
        date_to: Latest date to include, written as YYYY-MM-DD.
        tags: Return only entries that carry all of these tags.
        limit: How many entries to return, at most 20.
        offset: How many matching entries to skip before this page.
    """
    with _repair():
        if exact and not query:
            raise ValueError("exact search requires a query")
        results = ctx.deps.vault.find_entries_page(
            exact_text=query if exact else None,
            lexical_text=None if exact else query,
            semantic_text=None if exact else query,
            date_from=date_from,
            date_to=date_to,
            tags_all=tuple(tags or ()),
            limit=max(1, min(limit, 20)),
            offset=max(0, offset),
            order=ResultOrder.DATE_DESC if query is None else ResultOrder.DOCUMENT,
        )
    return projections.page(ctx.deps, results, match=query if exact else None)


def read_note(ctx: RunContext[Sources], name: str) -> dict[str, Any]:
    """Read one complete note as Markdown.

    Search returns fragments of a note. Read the whole note when the question
    is about the note itself, or when a fragment is not enough to answer it.

    Args:
        name: The note's title, or its path inside the vault. A search result's
            `note` or `path` value works.
    """
    # An `id` reaches here often enough to be worth naming: it is the first
    # field of every result, so it is the identifier the model has just read,
    # and the library would only answer "note not found", which does not say
    # what to send instead.
    if _SOURCE_ID.fullmatch(name.strip()):
        raise ModelRetry(
            f"{name!r} is a source id, not a note name. Pass the `note` or "
            "`path` value from the same result instead."
        )
    with _repair():
        return projections.note(ctx.deps, ctx.deps.vault.read_note(name))


def expand_context(ctx: RunContext[Sources], source: str) -> dict[str, Any]:
    """Find what surrounds one entry: linked notes, backlinks, nearby days.

    Each returned item is labelled with how it relates to the entry, so an
    answer can separate direct evidence from context that was reached through
    a link or from history that merely happened nearby in time.

    Args:
        source: The `id` of a result another tool returned, such as s3.
    """
    with _repair():
        entry_id = ctx.deps.entry_id(source)
        return projections.context(ctx.deps, ctx.deps.vault.expand_context(entry_id))


def find_backlinks(ctx: RunContext[Sources], name: str, limit: int = 10, offset: int = 0) -> dict[str, Any]:
    """Find the entries in other notes that link to this note.

    Use this to learn where a note is referred to, and in what terms.

    Args:
        name: The note's title, or its path inside the vault.
        limit: How many backlinks to return, at most 20.
        offset: How many backlinks to skip before this page.
    """
    with _repair():
        note = ctx.deps.vault.get_note(name)
        return projections.page(
            ctx.deps,
            ctx.deps.vault.get_backlinks_page(
                note.note_id, limit=max(1, min(limit, 20)), offset=max(0, offset)
            ),
        )


def find_tasks(
    ctx: RunContext[Sources],
    board: str | None = None,
    columns: list[str] | None = None,
    statuses: list[str] | None = None,
    checked: bool | None = None,
    archived: bool | None = False,
    query: str | None = None,
    tags: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    created_at_from: str | None = None,
    created_at_to: str | None = None,
    started_at_from: str | None = None,
    started_at_to: str | None = None,
    completed_at_from: str | None = None,
    completed_at_to: str | None = None,
    due_date_from: str | None = None,
    due_date_to: str | None = None,
    limit: int = 10,
    offset: int = 0,
) -> dict[str, Any]:
    """Find task cards on the Kanban boards in the vault.

    Args:
        board: Restrict to one board by name or path. Omit to search them all.
        columns: Restrict to these columns, for example ToDo or Doing. An
            unknown column name is rejected, and the reply names the real ones.
        statuses: Restrict to board statuses, such as todo, doing, or complete.
        checked: True for checked cards, False for unchecked, omit for both.
        archived: False for active tasks, True for archived, None for both.
        query: Words to search for by meaning. This finds related cards; it
            does not prove any card contains these words.
        tags: Return only cards carrying all of these tags.
        date_from: Earliest explicit card date to include, as YYYY-MM-DD.
        date_to: Latest explicit card date to include, as YYYY-MM-DD. Dates do
            not imply due, scheduled, or completion meaning.
        created_at_from: Inclusive earliest created_at; YYYY-MM-DD (UTC day) or timestamp with timezone.
        created_at_to: Inclusive latest created_at; YYYY-MM-DD (UTC day) or timestamp with timezone.
        started_at_from: Inclusive earliest started_at; YYYY-MM-DD (UTC day) or timestamp with timezone.
        started_at_to: Inclusive latest started_at; YYYY-MM-DD (UTC day) or timestamp with timezone.
        completed_at_from: Inclusive earliest completed_at; YYYY-MM-DD (UTC day) or timestamp with timezone.
        completed_at_to: Inclusive latest completed_at; YYYY-MM-DD (UTC day) or timestamp with timezone.
        due_date_from: Inclusive earliest due_date; YYYY-MM-DD.
        due_date_to: Inclusive latest due_date; YYYY-MM-DD.
        limit: How many cards to return, at most 50.
        offset: How many matching cards to skip before this page.
    """
    with _repair():
        results = ctx.deps.vault.find_kanban_cards_page(
            board=board,
            columns=_columns(ctx, board, columns),
            statuses=tuple(statuses or ()),
            checked=checked,
            archived=archived,
            semantic_text=query,
            tags=tuple(tags or ()),
            date_from=date_from,
            date_to=date_to,
            created_at_from=created_at_from,
            created_at_to=created_at_to,
            started_at_from=started_at_from,
            started_at_to=started_at_to,
            completed_at_from=completed_at_from,
            completed_at_to=completed_at_to,
            due_date_from=due_date_from,
            due_date_to=due_date_to,
            limit=max(1, min(limit, 50)),
            offset=max(0, offset),
        )
    return projections.page(ctx.deps, results)


def list_notes(
    ctx: RunContext[Sources],
    note_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """List notes in the vault, including notes with no indexed content.

    Args:
        note_type: Restrict to log, standard, or kanban notes.
        limit: How many notes to return, at most 50.
        offset: How many notes to skip before this page.
    """
    with _repair():
        selected = NoteType(note_type) if note_type is not None else None
        return projections.notes(
            ctx.deps.vault.list_notes_page(
                note_type=selected, limit=max(1, min(limit, 50)), offset=max(0, offset)
            )
        )


def _columns(ctx: RunContext[Sources], board: str | None, requested: list[str] | None) -> tuple[str, ...]:
    """Resolve column names against the boards, and reject the ones that miss.

    Columns reach Qdrant as a case-sensitive match, so "To Do" against a stored
    "ToDo" silently returns nothing. The model would then report that a board
    has no such cards. A bad board name already fails loudly; a bad column name
    must too, and the failure names the columns that exist.
    """
    if not requested:
        return ()
    boards = (ctx.deps.vault.get_kanban_board(board),) if board else ctx.deps.vault.list_kanban_boards()
    known = {column.casefold(): column for item in boards for column in item.columns}
    unknown = [name for name in requested if name.casefold() not in known]
    if unknown:
        raise ValueError(
            f"unknown Kanban column {', '.join(repr(name) for name in unknown)}; "
            f"this board has {', '.join(sorted(known.values()))}"
        )
    return tuple(known[name.casefold()] for name in requested)


def create_board(
    ctx: RunContext[Sources], path: str, request_id: str, columns: list[str] | None = None
) -> dict[str, Any]:
    """Create a new Kanban board without overwriting any existing file.

    Args:
        path: Plain board name, e.g. Project. Folder and .md are automatic;
            an explicit path under the configured Kanban root also works.
        request_id: Unique creation key; reuse only to retry this same request.
        columns: Column names; defaults to ToDo, Doing, Completed.
    """
    with _repair():
        return ctx.deps.vault.kanban.create_board(path, request_id, columns).model_dump()


def get_task(ctx: RunContext[Sources], board: str, task: str) -> dict[str, Any]:
    """Read a complete card and board revision before editing.

    First locate the board with list_notes(note_type="kanban"); follow
    next_offset while has_more if needed. Copy its returned full vault-relative
    path, including all folders and .md, into board. Never guess the folder or
    pass a title, source label, or absolute filesystem path. Reuse a verified
    path already returned in this conversation.

    Args:
        board: Explicit vault-relative .md board path.
        task: Returned task_id, entry_id, or source label such as s1.
    """
    with _repair():
        selected = ctx.deps.entry_id(task) if task.casefold().startswith("s") and task[1:].isdigit() else task
        return ctx.deps.vault.kanban.get_task(board, selected).model_dump()


def add_cards(
    ctx: RunContext[Sources],
    board: str,
    cards: Annotated[list[NewCard], Field(min_length=1, max_length=30)],
    request_id: str,
) -> dict[str, Any]:
    """Append one or more cards with no board hash; save the entire list once.

    First locate the board with list_notes(note_type="kanban"); follow
    next_offset while has_more if needed. Copy its returned full vault-relative
    path, including all folders and .md, into board. Never guess the folder or
    pass a title, source label, or absolute filesystem path. Reuse a verified
    path already returned in this conversation.
    Read the board with read_note(name=that_path) to discover exact column
    names, including on empty boards. Use create_board only when a new board
    is requested, then reuse its returned path and columns. Ask the user to
    choose if multiple boards match.

    Args:
        board: Explicit .md path beneath the configured Kanban root.
        cards: List of 1-30 objects with title, column, optional content and due_date.
        request_id: Unique call ID; retry the identical list with the same ID.
    """
    with _repair():
        return ctx.deps.vault.kanban.add_cards(board, cards, request_id).model_dump()


def edit_cards(
    ctx: RunContext[Sources],
    board: str,
    cards: Annotated[list[CardEdit], Field(min_length=1, max_length=30)],
    request_id: str,
) -> dict[str, Any]:
    """Edit a list atomically, checking only each affected card's version.

    First locate the board with list_notes(note_type="kanban"); follow
    next_offset while has_more if needed. Copy its returned full vault-relative
    path, including all folders and .md, into board. Never guess the folder or
    pass a title, source label, or absolute filesystem path. Reuse a verified
    path already returned in this conversation.
    Find the cards with find_tasks(board=that_path), then read each with
    get_task using the same path and returned task identifier before modifying.

    Args:
        board: Explicit .md board path.
        cards: List of task_id/version pairs from get_task, plus optional title,
            content, due_date, created_at. Omitted fields stay; empty body/date clears.
        request_id: Unique call ID; reuse only for identical retries.
    """
    with _repair():
        return ctx.deps.vault.kanban.edit_cards(board, cards, request_id).model_dump()


def move_cards(
    ctx: RunContext[Sources],
    board: str,
    cards: Annotated[list[CardMove], Field(min_length=1, max_length=30)],
    request_id: str,
) -> dict[str, Any]:
    """Move a list in order within one board, managing dates and checkboxes.

    First locate the board with list_notes(note_type="kanban"); follow
    next_offset while has_more if needed. Copy its returned full vault-relative
    path, including all folders and .md, into board. Never guess the folder or
    pass a title, source label, or absolute filesystem path. Reuse a verified
    path already returned in this conversation.
    Find the cards with find_tasks(board=that_path), then read each with
    get_task using the same path and returned task identifier before modifying.

    Args:
        board: Explicit .md board path; cross-board moves are unsupported.
        cards: List of task_id/column pairs, optional from_column precondition
            and zero-based position. Later moves see earlier ones. All save or none.
        request_id: Unique call ID; reuse only for identical retries.
    """
    with _repair():
        return ctx.deps.vault.kanban.move_cards(board, cards, request_id).model_dump()


def archive_cards(
    ctx: RunContext[Sources],
    board: str,
    cards: Annotated[list[CardTarget], Field(min_length=1, max_length=30)],
    request_id: str,
) -> dict[str, Any]:
    """Remove cards from active use while retaining all content in the archive.

    First locate the board with list_notes(note_type="kanban"); follow
    next_offset while has_more if needed. Copy its returned full vault-relative
    path, including all folders and .md, into board. Never guess the folder or
    pass a title, source label, or absolute filesystem path. Reuse a verified
    path already returned in this conversation.
    Find the cards with find_tasks(board=that_path), then read each with
    get_task using the same path and returned task identifier before modifying.

    Args:
        board: Explicit .md board path.
        cards: List of 1-30 objects containing task_id; no board hash needed.
        request_id: Unique call ID; reuse only for identical retries.
    """
    with _repair():
        return ctx.deps.vault.kanban.archive_cards(board, cards, request_id).model_dump()


def delete_board(
    ctx: RunContext[Sources], board: str, confirmation_token: str | None = None
) -> dict[str, Any]:
    """Preview deletion, warn the user, then move the board and its cards to trash.

    Args:
        board: Explicit .md board path; ordinary notes cannot be deleted.
        confirmation_token: Omit for preview. Warn the user BEFORE executing
            with the returned token. Changed boards need a fresh preview/warning.
    """
    with _repair():
        return ctx.deps.vault.kanban.delete_board(board, confirmation_token).model_dump()


TOOLS = [
    search_notes,
    read_note,
    expand_context,
    find_backlinks,
    find_tasks,
    list_notes,
    create_board,
    get_task,
    add_cards,
    edit_cards,
    move_cards,
    archive_cards,
    delete_board,
]


@contextmanager
def _repair() -> Iterator[None]:
    """Turn a library error into an instruction the model can act on.

    The library already says what went wrong and, for an ambiguous name, which
    notes it could have meant. `ModelRetry` hands that text back to the model
    so it can correct the call instead of the run failing.
    """
    try:
        yield
    except (LookupError, ValueError, OSError) as error:
        raise ModelRetry(str(error)) from error
