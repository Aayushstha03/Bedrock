"""Source spans shared by Kanban parsing and surgical card edits."""

import re
from dataclasses import dataclass

from markdown_it.token import Token

CARD_RE = re.compile(r"^- \[([ xX])\]\s+(.*)$")
BLOCK_ID_RE = re.compile(r"\s+\^([A-Za-z0-9-]+)$")


@dataclass(frozen=True)
class CardSpan:
    start: int
    end: int
    block_id: str | None


@dataclass(frozen=True)
class ColumnSpan:
    name: str
    start: int
    end: int
    complete: bool
    archived: bool
    cards: tuple[CardSpan, ...]


def board_columns(tokens: list[Token], lines: list[str], body_start: int, end: int) -> tuple[ColumnSpan, ...]:
    headings = [
        (token.map[0], tokens[index + 1].content)
        for index, token in enumerate(tokens)
        if token.type == "heading_open"
        and token.tag == "h2"
        and token.level == 0
        and token.map is not None
        and body_start <= token.map[0] < end
    ]
    breaks = {token.map[0] for token in tokens if token.type == "hr" and token.level == 0 and token.map}
    cards = {
        token.map[0]: token.map[1]
        for token in tokens
        if token.type == "list_item_open"
        and token.level == 1
        and token.map is not None
        and CARD_RE.match(lines[token.map[0]])
    }
    result: list[ColumnSpan] = []
    for index, (start, name) in enumerate(headings):
        stop = headings[index + 1][0] if index + 1 < len(headings) else end
        preceding = start - 1
        while preceding >= body_start and not lines[preceding].strip():
            preceding -= 1
        archived = name.casefold() == "archive" and preceding in breaks
        if index + 1 < len(headings):
            boundary = stop - 1
            while boundary > start and not lines[boundary].strip():
                boundary -= 1
            if boundary in breaks and headings[index + 1][1].casefold() == "archive":
                stop = boundary
        spans = []
        for first, last in cards.items():
            if start < first < stop:
                last = min(last, stop)
                while last > first + 1 and not lines[last - 1].strip():
                    last -= 1
                match = BLOCK_ID_RE.search(lines[first].rstrip("\r\n"))
                spans.append(CardSpan(first, last, match.group(1) if match else None))
        complete = any(
            token.type == "inline"
            and token.level == 1
            and token.content == "**Complete**"
            and token.map
            and start < token.map[0] < (spans[0].start if spans else stop)
            for token in tokens
        )
        result.append(ColumnSpan(name, start, stop, complete, archived, tuple(spans)))
    return tuple(result)
