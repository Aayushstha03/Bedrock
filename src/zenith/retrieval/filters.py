"""Deterministic Qdrant filter construction from a QueryPlan."""

from __future__ import annotations

from qdrant_client import models

from zenith.core.contracts import QueryPlan
from zenith.index.qdrant import vault_condition


def build_filter(plan: QueryPlan, vault_id: str) -> models.Filter:
    must: list[models.Condition] = [vault_condition(vault_id)]

    if plan.note_id:
        must.append(models.FieldCondition(key="note_id", match=models.MatchValue(value=plan.note_id)))

    if plan.entry_types:
        must.append(
            models.FieldCondition(
                key="entry_type",
                match=models.MatchAny(any=[entry_type.value for entry_type in plan.entry_types]),
            )
        )

    if plan.section:
        must.append(
            models.Filter(
                should=[
                    models.FieldCondition(key="heading", match=models.MatchValue(value=plan.section)),
                    models.FieldCondition(key="heading_path", match=models.MatchValue(value=plan.section)),
                ]
            )
        )

    must.extend(
        models.FieldCondition(key="tags", match=models.MatchValue(value=tag)) for tag in plan.tags_all
    )

    if plan.tags_any:
        must.append(
            models.Filter(
                should=[
                    models.FieldCondition(key="tags", match=models.MatchValue(value=tag))
                    for tag in plan.tags_any
                ]
            )
        )

    if plan.date_from or plan.date_to:
        date_range = models.DatetimeRange(
            gte=f"{plan.date_from}T00:00:00Z" if plan.date_from else None,
            lte=f"{plan.date_to}T23:59:59Z" if plan.date_to else None,
        )
        must.append(
            models.Filter(
                should=[
                    models.FieldCondition(key="note_date", range=date_range),
                    models.FieldCondition(key="entry_date", range=date_range),
                    # Inclusive overlap: stored start <= query end, stored
                    # end >= query start. Single dates use the branch above.
                    models.Filter(
                        must=[
                            models.FieldCondition(
                                key="entry_date", range=models.DatetimeRange(lte=date_range.lte)
                            ),
                            models.FieldCondition(
                                key="entry_date_end", range=models.DatetimeRange(gte=date_range.gte)
                            ),
                        ]
                    ),
                ]
            )
        )

    if plan.kanban_board:
        # A card's `note_title` is its board's title, and every Kanban query is
        # already scoped to `EntryType.KANBAN_CARD`, so this needs no field of
        # its own.
        must.append(models.FieldCondition(key="note_title", match=models.MatchValue(value=plan.kanban_board)))
    if plan.kanban_column:
        must.append(
            models.FieldCondition(key="board.column", match=models.MatchValue(value=plan.kanban_column))
        )
    if plan.kanban_columns:
        must.append(
            models.FieldCondition(key="board.column", match=models.MatchAny(any=list(plan.kanban_columns)))
        )
    if plan.kanban_status:
        must.append(
            models.FieldCondition(key="board.status", match=models.MatchValue(value=plan.kanban_status))
        )
    if plan.kanban_statuses:
        must.append(
            models.FieldCondition(key="board.status", match=models.MatchAny(any=list(plan.kanban_statuses)))
        )
    if plan.kanban_checked is not None:
        must.append(
            models.FieldCondition(key="board.checked", match=models.MatchValue(value=plan.kanban_checked))
        )

    return models.Filter(must=must)
