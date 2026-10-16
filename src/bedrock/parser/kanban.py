"""Specialized `/kanban` parsing layered on the CommonMark AST."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from markdown_it.token import Token

from bedrock.core.config import Settings
from bedrock.core.contracts import (
    EntryType,
    IndexWarning,
    KanbanData,
    NoteType,
    ParsedEntry,
    SourceRange,
    WarningType,
)
from bedrock.core.identity import entry_id
from bedrock.parser.kanban_dates import lifecycle_dates
from bedrock.parser.kanban_structure import BLOCK_ID_RE, CARD_RE, board_columns
from bedrock.parser.markdown import (
    Frontmatter,
    Heading,
    meaningful_line_range,
    prose_between,
    valid_iso_date,
)
from bedrock.parser.tokens import estimate_tokens

# Obsidian Kanban writes card dates and times behind configurable triggers.
# The plugin defaults, from `defaultDateTrigger` and `defaultTimeTrigger` in
# its own source, are `@` and `@@`.
DEFAULT_DATE_TRIGGER = "@"
DEFAULT_TIME_TRIGGER = "@@"
TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")
SETTINGS_START_RE = re.compile(r"^%%\s*kanban:settings\s*$")
STATUS_ALIASES = {
    "todo": {"todo", "to do", "backlog"},
    "doing": {"doing", "in progress", "on progress", "working", "active"},
    "complete": {"complete", "completed", "done"},
}


def _trigger(plugin_settings: dict[str, Any] | None, key: str, fallback: str) -> str:
    value = (plugin_settings or {}).get(key)
    return value if isinstance(value, str) and value.strip() else fallback


def _annotation_re(trigger: str) -> re.Pattern[str]:
    """Match `@{value}` and the daily-note form `@[[value]]` for one trigger."""
    escaped = re.escape(trigger)
    return re.compile(rf"{escaped}(?:\{{([^}}]*)\}}|\[\[([^\]]*)\]\])")


def card_annotations(
    text: str, path: str, line: int, date_trigger: str, time_trigger: str
) -> tuple[str, str | None, str | None, tuple[str, ...], list[IndexWarning]]:
    """Pull dates and times out of card text and hand back the clean prose.

    The plugin syntax is board configuration, not prose, so it never reaches
    the searchable text or the embedding input.
    """
    warnings: list[IndexWarning] = []
    consumed: list[str] = []
    date: str | None = None
    time: str | None = None
    seen_dates: list[str] = []

    # A longer trigger can start with a shorter one, so strip longest first.
    for kind, trigger in sorted(
        (("time", time_trigger), ("date", date_trigger)), key=lambda item: -len(item[1])
    ):
        # `kind` is bound as a default so the closure reads what this iteration
        # is stripping, not whatever the loop variable holds later.
        def take(match: re.Match[str], kind: str = kind) -> str:
            nonlocal date, time
            raw = (match.group(1) if match.group(1) is not None else match.group(2) or "").strip()
            consumed.append(raw)
            if kind == "time":
                if TIME_RE.fullmatch(raw):
                    time = time or raw
                    return " "
                warnings.append(
                    IndexWarning(WarningType.INVALID_DATE, path, f"invalid Kanban card time: {raw}", line)
                )
                return " "
            valid = valid_iso_date(raw)
            if valid is None:
                warnings.append(
                    IndexWarning(WarningType.INVALID_DATE, path, f"invalid Kanban card date: {raw}", line)
                )
                return " "
            seen_dates.append(valid)
            date = date or valid
            return " "

        text = _annotation_re(trigger).sub(take, text)

    unique_dates = sorted(set(seen_dates))
    if len(unique_dates) > 1:
        warnings.append(
            IndexWarning(
                WarningType.INVALID_DATE,
                path,
                f"Kanban card carries more than one date: {', '.join(unique_dates)}; using {date}",
                line,
            )
        )
    return " ".join(text.split()), date, time, tuple(consumed), warnings


def canonical_status(column: str) -> str | None:
    normalized = " ".join(column.casefold().split())
    return next((status for status, aliases in STATUS_ALIASES.items() if normalized in aliases), None)


def parse_settings(
    lines: list[str], path: str
) -> tuple[dict[str, Any] | None, IndexWarning | None, int | None]:
    start = next(
        (index for index, line in enumerate(lines) if SETTINGS_START_RE.fullmatch(line.strip())), None
    )
    if start is None:
        return None, None, None
    fence_start = next(
        (index for index in range(start + 1, len(lines)) if lines[index].strip() in {"```json", "```"}), None
    )
    fence_end = (
        next((index for index in range(fence_start + 1, len(lines)) if lines[index].strip() == "```"), None)
        if fence_start is not None
        else None
    )
    if fence_start is None or fence_end is None:
        warning = IndexWarning(
            WarningType.INVALID_KANBAN_SETTINGS, path, "Kanban settings block is incomplete", start + 1
        )
        return None, warning, start
    try:
        value = json.loads("\n".join(lines[fence_start + 1 : fence_end]))
    except json.JSONDecodeError as exc:
        warning = IndexWarning(
            WarningType.INVALID_KANBAN_SETTINGS,
            path,
            f"invalid Kanban settings JSON: {exc.msg}",
            fence_start + 2,
        )
        return None, warning, start
    if not isinstance(value, dict):
        return (
            None,
            IndexWarning(
                WarningType.INVALID_KANBAN_SETTINGS, path, "Kanban settings must be an object", start + 1
            ),
            start,
        )
    return value, None, start


def parse_kanban_entries(
    *,
    lines: list[str],
    tokens: list[Token],
    parsed_headings: tuple[Heading, ...],
    frontmatter: Frontmatter,
    path: str,
    title: str,
    note_uuid: str,
    settings: Settings,
) -> tuple[tuple[ParsedEntry, ...], tuple[IndexWarning, ...], dict[str, Any]]:
    warnings: list[IndexWarning] = []
    if frontmatter.values.get("kanban-plugin") != "board":
        warnings.append(
            IndexWarning(
                WarningType.MISSING_KANBAN_MARKER, path, "Kanban note is missing `kanban-plugin: board`"
            )
        )
    plugin_settings, settings_warning, settings_start = parse_settings(lines, path)
    if settings_warning:
        warnings.append(settings_warning)

    date_trigger = _trigger(plugin_settings, "date-trigger", DEFAULT_DATE_TRIGGER)
    time_trigger = _trigger(plugin_settings, "time-trigger", DEFAULT_TIME_TRIGGER)

    columns = board_columns(
        tokens, lines, frontmatter.body_start, settings_start if settings_start is not None else len(lines)
    )
    heading_by_line = {heading.line - 1: heading for heading in parsed_headings}
    # Metadata fences belong to the source card but never its searchable prose.
    prose_tokens = [
        token for token in tokens if not (token.type == "fence" and token.info.strip() == "bedrock-dates")
    ]
    entries: list[ParsedEntry] = []
    for column_position, column in enumerate(columns):
        status = "complete" if column.complete else canonical_status(column.name)
        heading = heading_by_line[column.start]
        for card_position, span in enumerate(column.cards):
            line_index, next_card = span.start, span.end
            match = CARD_RE.match(lines[line_index])
            prose = prose_between(
                prose_tokens, line_index, next_card, path, settings.known_tags, settings.tag_aliases
            )
            dates, lifecycle_warnings = lifecycle_dates(tokens, lines, line_index, next_card, path)
            warnings.extend(lifecycle_warnings)
            visible = re.sub(r"^\[[ xX]\]\s*", "", prose.text).strip()
            if span.block_id:
                visible = visible.replace(f" ^{span.block_id}", "", 1)
            if not visible:
                visible = BLOCK_ID_RE.sub("", match.group(2)).strip()
            visible, card_date, card_time, consumed, date_warnings = card_annotations(
                visible, path, line_index + 1, date_trigger, time_trigger
            )
            warnings.extend(date_warnings)
            # No second fallback to the raw line here: a card whose whole text
            # is an annotation has no prose, and reinstating the raw text would
            # put the plugin syntax straight back into the searchable text.
            # `@[[2026-06-06]]` is a date the plugin renders as a daily-note
            # link. It was consumed as a date, so it is not also a real link.
            links = tuple(link for link in prose.links if link.target_text not in consumed)
            structural_key = (
                f"block:{span.block_id}"
                if span.block_id
                else f"{column.name}/{card_position}:{visible.casefold()}"
            )
            card_id = str(entry_id(note_uuid, EntryType.KANBAN_CARD, structural_key))
            body_range = meaningful_line_range(lines, line_index, next_card)
            entries.append(
                ParsedEntry(
                    entry_id=card_id,
                    note_id=note_uuid,
                    path=path,
                    note_title=title,
                    note_type=NoteType.KANBAN,
                    entry_type=EntryType.KANBAN_CARD,
                    text=visible,
                    embedding_text=(
                        f"Board: {title}\nColumn: {column.name}\n"
                        + (f"Date: {card_date}\n" if card_date else "")
                        + f"Status: {status or 'custom'}\nTask: {visible}"
                    ),
                    heading=column.name,
                    heading_path=heading.path,
                    source=SourceRange(line_index + 1, body_range.end_line if body_range else line_index + 1),
                    entry_date=card_date,
                    tags=prose.tags,
                    outgoing_links=links,
                    web_links=prose.web_links,
                    content_hash=hashlib.sha256(visible.encode("utf-8")).hexdigest(),
                    board=KanbanData(
                        column=column.name,
                        status=status,
                        column_position=column_position,
                        card_position=card_position,
                        checked=match.group(1).casefold() == "x",
                        card_time=card_time,
                        task_id=span.block_id,
                        archived=column.archived,
                        **dates,
                    ),
                )
            )
            warnings.extend(prose.warnings)
            # A card is one atomic point, so it is never divided. When its text
            # overruns the encoder the only honest signal is a warning.
            cost = estimate_tokens(entries[-1].embedding_text)
            if cost > settings.dense_token_window:
                warnings.append(
                    IndexWarning(
                        WarningType.TRUNCATED_EMBEDDING_INPUT,
                        path,
                        f"Kanban card needs about {cost} tokens but only "
                        f"{settings.dense_token_window} reach the dense model; a card is never "
                        "divided, so the rest is not semantically searchable",
                        line_index + 1,
                    )
                )
    metadata = {
        "kanban_settings": plugin_settings,
        "columns": [column.name for column in columns if not column.archived],
    }
    return tuple(entries), tuple(warnings), metadata
