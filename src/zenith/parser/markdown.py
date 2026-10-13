"""CommonMark AST helpers and searchable text extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import yaml
from markdown_it import MarkdownIt
from markdown_it.token import Token

from zenith.core.contracts import IndexWarning, Link, SourceRange, WarningType

# Vault dates have one deliberately non-configurable representation. Using
# ASCII digit classes and fullmatch prevents timestamps, prefixes, suffixes,
# slashes, and Unicode digit lookalikes from being interpreted as chronology.
DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
TAG_RE = re.compile(r"(?<![\w/#])#([A-Za-z][\w-]*)")
WIKI_RE = re.compile(r"\[\[([^\]|#]+)(?:#([^\]|]+))?(?:\|([^\]]+))?\]\]")
URL_RE = re.compile(r"https?://[^\s<>]+")


@dataclass(frozen=True, slots=True)
class Frontmatter:
    values: dict[str, Any]
    body_start: int
    warning: IndexWarning | None = None


@dataclass(frozen=True, slots=True)
class Heading:
    level: int
    text: str
    line: int
    content_start: int
    content_end: int
    path: tuple[str, ...]
    active_date: str | None


@dataclass(frozen=True, slots=True)
class Prose:
    text: str
    tags: tuple[str, ...]
    links: tuple[Link, ...]
    web_links: tuple[str, ...]
    warnings: tuple[IndexWarning, ...]


def parse_frontmatter(content: str, path: str = "") -> Frontmatter:
    def failure(body_start: int, message: str) -> Frontmatter:
        return Frontmatter({}, body_start, IndexWarning(WarningType.PARSER_FAILURE, path, message, 1))

    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return Frontmatter({}, 0)
    try:
        closing = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration:
        return failure(0, "frontmatter opening marker has no closing marker")
    raw = "\n".join(lines[1:closing])
    try:
        values = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        return failure(closing + 1, f"invalid frontmatter: {exc}")
    if not isinstance(values, dict):
        return failure(closing + 1, "frontmatter must be a mapping")
    return Frontmatter(values, closing + 1)


def markdown_tokens(content: str) -> list[Token]:
    return MarkdownIt("commonmark").parse(content)


def valid_iso_date(value: str) -> str | None:
    """Return a real calendar date only when value is exactly YYYY-MM-DD."""
    if not DATE_RE.fullmatch(value):
        return None
    try:
        date.fromisoformat(value)
    except ValueError:
        return None
    return value


def headings(tokens: list[Token], total_lines: int) -> tuple[Heading, ...]:
    raw: list[tuple[int, int, str]] = []
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.map is None:
            continue
        inline = tokens[index + 1]
        raw.append((int(token.tag[1:]), token.map[0], inline.content.strip()))

    result: list[Heading] = []
    stack: list[tuple[int, str, str | None]] = []
    for index, (level, line, text) in enumerate(raw):
        while stack and stack[-1][0] >= level:
            stack.pop()
        own_date = valid_iso_date(text)
        inherited_date = next((item[2] for item in reversed(stack) if item[2]), None)
        path = (*(item[1] for item in stack), text)
        next_line = raw[index + 1][1] if index + 1 < len(raw) else total_lines
        result.append(Heading(level, text, line + 1, line + 1, next_line, path, own_date or inherited_date))
        stack.append((level, text, own_date or inherited_date))
    return tuple(result)


def frontmatter_aliases(values: dict[str, Any] | None) -> tuple[str, ...]:
    """Read the alternative names a note declares, under either spelling."""
    metadata = values or {}
    raw = metadata.get("aliases", metadata.get("alias", ()))
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list):
        return tuple(item for item in raw if isinstance(item, str))
    return ()


def note_title(frontmatter: Frontmatter, parsed_headings: tuple[Heading, ...], fallback: str) -> str:
    explicit = frontmatter.values.get("title")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    first_h1 = next((heading.text for heading in parsed_headings if heading.level == 1), None)
    return first_h1 or fallback


def prose_between(
    tokens: list[Token],
    start_zero: int,
    end_zero: int,
    path: str,
    known_tags: tuple[str, ...],
    tag_aliases: tuple[tuple[str, str], ...],
) -> Prose:
    chunks: list[str] = []
    searchable_chunks: list[str] = []
    web_links: list[str] = []
    chunk_lines: list[int] = []
    for token in tokens:
        if token.map is None:
            continue
        if token.type in {"fence", "code_block"}:
            # A code block can span several paragraph-based chunks. Include
            # only its overlapping content lines, without fence delimiters.
            content_start = token.map[0] + (1 if token.type == "fence" else 0)
            code_lines = token.content.splitlines(keepends=True)
            first = max(start_zero - content_start, 0)
            last = min(end_zero - content_start, len(code_lines))
            if first < last:
                searchable_chunks.append("".join(code_lines[first:last]).rstrip("\n"))
            continue
        if token.type != "inline":
            continue
        if token.map[0] < start_zero or token.map[0] >= end_zero:
            continue
        parts: list[str] = []
        searchable_parts: list[str] = []
        for child in token.children or ():
            if child.type in {"text", "softbreak", "hardbreak"}:
                value = "\n" if child.type != "text" else child.content
                parts.append(value)
                searchable_parts.append(value)
            elif child.type == "code_inline":
                searchable_parts.append(child.content)
                # Keep prose on either side separate for tag/link extraction.
                parts.append(" " * len(child.content))
            elif child.type == "link_open":
                href = child.attrGet("href")
                if href and href.startswith(("http://", "https://")):
                    web_links.append(href)
        text = "".join(parts).strip()
        searchable_text = "".join(searchable_parts).strip()
        if searchable_text:
            searchable_chunks.append(searchable_text)
        if text:
            chunks.append(text)
            chunk_lines.append(token.map[0] + 1)

    text = "\n".join(chunks).strip()
    spans = _chunk_spans(chunks, chunk_lines)
    web_links.extend(match.group(0).rstrip(".,);") for match in URL_RE.finditer(text))

    known = {tag.casefold() for tag in known_tags}
    aliases = dict(tag_aliases)
    tags: list[str] = []
    warnings: list[IndexWarning] = []
    for match in TAG_RE.finditer(text):
        raw_tag = match.group(1).casefold()
        tag = aliases.get(raw_tag, raw_tag)
        if tag in known:
            if tag not in tags:
                tags.append(tag)
        else:
            warnings.append(IndexWarning(WarningType.UNKNOWN_TAG, path, f"unknown tag: #{raw_tag}"))

    links = [
        Link(
            target_text=match.group(1).strip(),
            target_heading=match.group(2).strip() if match.group(2) else None,
            alias=match.group(3).strip() if match.group(3) else None,
            line=_line_at(spans, text, match.start()),
        )
        for match in WIKI_RE.finditer(text)
    ]
    return Prose(
        text="\n".join(searchable_chunks).strip(),
        tags=tuple(tags),
        links=tuple(links),
        web_links=tuple(dict.fromkeys(web_links)),
        warnings=tuple(warnings),
    )


def _chunk_spans(chunks: list[str], chunk_lines: list[int]) -> tuple[tuple[int, int], ...]:
    """Pair each chunk's offset in the joined text with its first source line."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for chunk, line in zip(chunks, chunk_lines, strict=True):
        spans.append((offset, line))
        offset += len(chunk) + 1  # the newline `prose_between` joins chunks with
    return tuple(spans)


def _line_at(spans: tuple[tuple[int, int], ...], text: str, position: int) -> int | None:
    """Return the source line the character at `position` came from.

    Blocks are separated by blank lines the joined text does not keep, so
    counting newlines from the start would drift. Anchor on the chunk the
    position falls in, then count the newlines inside that chunk alone; each
    one is a softbreak, and a softbreak is one source line.
    """
    anchor = next(
        (span for span in reversed(spans) if span[0] <= position),
        None,
    )
    if anchor is None:
        return None
    offset, line = anchor
    return line + text.count("\n", offset, position)


def meaningful_line_range(lines: list[str], start_zero: int, end_zero: int) -> SourceRange | None:
    meaningful = [index for index in range(start_zero, min(end_zero, len(lines))) if lines[index].strip()]
    if not meaningful:
        return None
    return SourceRange(meaningful[0] + 1, meaningful[-1] + 1)
