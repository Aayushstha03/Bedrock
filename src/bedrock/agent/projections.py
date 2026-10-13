"""Compact, model-facing views of Bedrock's retrieval contracts.

Library contracts carry everything a caller might need. A language model needs
far less, and every field it does not need costs context it could have spent on
evidence. These functions decide what a tool actually returns.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from bedrock.agent.sources import Sources
from bedrock.core.contracts import (
    ContextExpansion,
    NoteContent,
    NoteSummary,
    Page,
    SearchResult,
)

# Most entries fit the dense token window, but `pack` deliberately emits a
# single paragraph larger than the budget rather than cutting a sentence in
# half, so an entry can still overrun this. A whole note has no bound at all.
MAX_ENTRY_CHARS = 1200
MAX_NOTE_CHARS = 12000


def entry(sources: Sources, result: SearchResult, *, match: str | None = None) -> dict[str, Any]:
    """Project one search result down to what an answer needs to cite it.

    `match` is the literal phrase a caller verified. Clipping is centred on it,
    so an entry too large to send whole still carries the evidence for its own
    verification instead of an unrelated opening paragraph.
    """
    text, truncated = _clip(result.text, MAX_ENTRY_CHARS, match=match)
    date, date_kind = _date(result)
    # `id` leads. A model reads a result in key order and cites the first
    # identifier it meets, so the first one it meets is the only one it is
    # allowed to print. The note title, the heading, and the line range stay in
    # the result because they are evidence a model reasons over, but it never
    # has to assemble them into a citation: the label already is one.
    projected: dict[str, Any] = {
        "id": sources.label(result),
        "note": result.note_title,
        "heading": result.heading,
        "lines": f"{result.start_line}-{result.end_line}",
        "path": result.path,
        "date": date,
        "date_kind": date_kind,
        "tags": list(result.tags),
        "text": text,
    }
    if truncated:
        projected["text_truncated"] = True
    if result.entry_date_end:
        projected["date_end"] = result.entry_date_end
    # Only literal retrieval verifies an exact phrase. Similarity never does,
    # and must never be reported as if it had. Neither may a clipped body that
    # no longer contains the phrase: the model would be told the wording is
    # proven while holding text that does not show it.
    if result.verified is not None and (not result.verified or not truncated or _contains(text, match)):
        projected["exact_match_verified"] = result.verified
    if result.board is not None:
        projected["task"] = {
            "board": result.note_title,
            "column": result.board.column,
            "status": result.board.status,
            "checked": result.board.checked,
        }
    return projected


def entries(
    sources: Sources, results: tuple[SearchResult, ...], *, match: str | None = None
) -> list[dict[str, Any]]:
    return [entry(sources, result, match=match) for result in results]


def page(sources: Sources, result: Page[SearchResult], *, match: str | None = None) -> dict[str, Any]:
    projected: dict[str, Any] = {
        "items": entries(sources, result.items, match=match),
        "has_more": result.has_more,
    }
    if result.next_offset is not None:
        projected["next_offset"] = result.next_offset
    return projected


def notes(result: Page[NoteSummary]) -> dict[str, Any]:
    projected: dict[str, Any] = {
        "items": [
            {
                "note": item.title,
                "path": item.path,
                "note_type": str(item.note_type),
                "indexed_entries": item.indexed_entries,
            }
            for item in result.items
        ],
        "has_more": result.has_more,
    }
    if result.next_offset is not None:
        projected["next_offset"] = result.next_offset
    return projected


def note(sources: Sources, content: NoteContent) -> dict[str, Any]:
    text, truncated = _clip(content.content, MAX_NOTE_CHARS)
    projected: dict[str, Any] = {
        "id": sources.label_note(content),
        "note": content.title,
        "path": content.path,
        "note_type": str(content.note_type),
        "content": text,
    }
    if truncated:
        projected["content_truncated"] = True
    return projected


def context(sources: Sources, expansion: ContextExpansion) -> dict[str, Any]:
    """Project expanded context, keeping how each item was reached.

    That is the point of expanding at all. An answer must say whether a fact is
    direct evidence, something a link led to, or nearby history in another
    note. (This `evidence` field is unrelated to the citation ids: it says how
    an item was reached, not how to cite it.)
    """
    items = []
    for item in expansion.items:
        projected = entry(sources, item.result)
        projected["evidence"] = [str(label) for label in item.labels]
        projected["hops"] = item.depth
        items.append(projected)
    unresolved = [
        {"target": diagnostic.target_text, "reason": str(diagnostic.resolution)}
        for diagnostic in expansion.diagnostics
    ]
    # No `source_entry_id`. The caller named the source by its label, and
    # echoing an internal identifier back only invites the model to print it.
    projected: dict[str, Any] = {"items": items}
    if unresolved:
        projected["unresolved_links"] = unresolved
    projected["truncated"] = expansion.truncated
    projected["budget_reasons"] = list(expansion.budget_reasons)
    if expansion.section_continuations:
        projected["section_continuations"] = [
            asdict(continuation) for continuation in expansion.section_continuations
        ]
    return projected


def _date(result: SearchResult) -> tuple[str | None, str | None]:
    """Report which date an entry carries, and where that date came from.

    An entry date is the date written on the entry. A note date belongs to the
    dated note the entry sits in. An entry with neither is undated, and must
    never be presented as if it happened on any particular day.
    """
    if result.entry_date:
        return result.entry_date, "entry_date"
    if result.note_date:
        return result.note_date, "note_date"
    return None, None


def _contains(text: str, match: str | None) -> bool:
    return bool(match) and match.casefold() in text.casefold()


def _clip(text: str, limit: int, *, match: str | None = None) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    start = 0
    if match:
        found = text.casefold().find(match.casefold())
        if found >= 0:
            # Centre the window on the phrase, then pull it back inside the text.
            start = max(0, min(found - (limit - len(match)) // 2, len(text) - limit))
    clipped = text[start : start + limit].strip()
    prefix = "[... truncated]\n" if start else ""
    return f"{prefix}{clipped}\n[... truncated]", True
