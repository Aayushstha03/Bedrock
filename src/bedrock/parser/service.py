"""Deterministic orchestration for log, standard, and Kanban notes."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from bedrock.core.config import Settings
from bedrock.core.contracts import (
    EntryType,
    IndexWarning,
    NoteType,
    ParsedEntry,
    ParsedNote,
    SourceRange,
    WarningType,
)
from bedrock.core.identity import entry_id, note_id
from bedrock.parser.chunking import Chunk, pack, paragraphs
from bedrock.parser.discovery import classify_path, discover_markdown
from bedrock.parser.kanban import parse_kanban_entries
from bedrock.parser.markdown import (
    Heading,
    Prose,
    headings,
    markdown_tokens,
    meaningful_line_range,
    note_title,
    parse_frontmatter,
    prose_between,
    valid_iso_date,
)
from bedrock.parser.tokens import content_budget, estimate_tokens

PARSER_VERSION = "3.5.0"


class VaultParser:
    def __init__(self, settings: Settings, vault_id: str = "personal") -> None:
        self.settings = settings
        self.vault_id = vault_id

    def parse_vault(self, requested: list[str] | None = None) -> tuple[ParsedNote, ...]:
        return tuple(self.parse_file(path) for path in discover_markdown(self.settings, requested))

    def parse_file(self, file_path: Path) -> ParsedNote:
        vault = self.settings.vault_path.resolve()
        resolved = file_path.resolve()
        relative = resolved.relative_to(vault).as_posix()
        content = resolved.read_text(encoding="utf-8")
        lines = content.splitlines()
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        note_uuid = str(note_id(self.vault_id, relative))
        note_type = classify_path(relative, self.settings)
        frontmatter = parse_frontmatter(content, relative)
        tokens = markdown_tokens(content)
        parsed_headings = tuple(
            heading for heading in headings(tokens, len(lines)) if heading.line > frontmatter.body_start
        )
        title = note_title(frontmatter, parsed_headings, resolved.stem)
        warnings: list[IndexWarning] = []
        if frontmatter.warning:
            warnings.append(frontmatter.warning)

        metadata = dict(frontmatter.values)
        if note_type is NoteType.KANBAN:
            entries, kanban_warnings, kanban_metadata = parse_kanban_entries(
                lines=lines,
                tokens=tokens,
                parsed_headings=parsed_headings,
                frontmatter=frontmatter,
                path=relative,
                title=title,
                note_uuid=note_uuid,
                settings=self.settings,
            )
            metadata.update(kanban_metadata)
            warnings.extend(kanban_warnings)
        else:
            note_date = (
                self._log_date(resolved.stem, lines, frontmatter.body_start)
                if note_type is NoteType.LOG
                else None
            )
            entries, entry_warnings = self._parse_open_entries(
                note_type=note_type,
                note_date=note_date,
                lines=lines,
                tokens=tokens,
                parsed_headings=parsed_headings,
                body_start=frontmatter.body_start,
                relative=relative,
                title=title,
                note_uuid=note_uuid,
            )
            warnings.extend(entry_warnings)
            if note_type is NoteType.LOG and note_date is None:
                warnings.append(
                    IndexWarning(WarningType.INVALID_DATE, relative, "log note has no valid date")
                )

        return ParsedNote(
            note_id=note_uuid,
            path=relative,
            title=title,
            note_type=note_type,
            content_hash=content_hash,
            entries=entries,
            warnings=tuple(dict.fromkeys(warnings)),
            metadata=metadata,
        )

    def _log_date(self, stem: str, lines: list[str], body_start: int) -> str | None:
        from_filename = valid_iso_date(stem)
        if from_filename:
            return from_filename
        first = next((line.strip() for line in lines[body_start:] if line.strip()), "")
        return valid_iso_date(first)

    def _parse_open_entries(
        self,
        *,
        note_type: NoteType,
        note_date: str | None,
        lines: list[str],
        tokens: list,
        parsed_headings: tuple[Heading, ...],
        body_start: int,
        relative: str,
        title: str,
        note_uuid: str,
    ) -> tuple[tuple[ParsedEntry, ...], tuple[IndexWarning, ...]]:
        entries: list[ParsedEntry] = []
        warnings: list[IndexWarning] = []
        occurrences: defaultdict[tuple[str, ...], int] = defaultdict(int)
        duplicates: defaultdict[str, int] = defaultdict(int)
        first_heading_zero = parsed_headings[0].line - 1 if parsed_headings else len(lines)

        loose_start = body_start
        if note_type is NoteType.LOG and note_date:
            first_nonempty = next(
                (index for index in range(body_start, first_heading_zero) if lines[index].strip()), None
            )
            if first_nonempty is not None and valid_iso_date(lines[first_nonempty].strip()) == note_date:
                loose_start = first_nonempty + 1

        preamble_type = EntryType.DAILY_SECTION if note_type is NoteType.LOG else EntryType.FREEFORM_CHUNK
        for chunk, prose in self._chunks(
            tokens=tokens,
            lines=lines,
            start_zero=loose_start,
            end_zero=first_heading_zero,
            relative=relative,
            title=title,
            heading=None,
            note_type=note_type,
            note_date=note_date,
            entry_date=None,
            warnings=warnings,
        ):
            body_range = meaningful_line_range(lines, chunk.start_zero, chunk.end_zero)
            entries.append(
                self._entry(
                    note_uuid,
                    relative,
                    title,
                    note_type,
                    preamble_type,
                    prose.text,
                    None,
                    (),
                    body_range or SourceRange(chunk.start_zero + 1, chunk.end_zero),
                    note_date,
                    None,
                    prose,
                    _unique_key(f"preamble:{_digest(prose.text)}", duplicates),
                )
            )
            warnings.extend(prose.warnings)

        for heading in parsed_headings:
            entry_date_end = heading.active_date_end if note_type is not NoteType.LOG else None
            if note_type is NoteType.LOG:
                entry_type = EntryType.DAILY_SECTION
                entry_date = None
            elif heading.active_date:
                entry_type = EntryType.PROJECT_UPDATE
                entry_date = heading.active_date
            else:
                entry_type = EntryType.FREEFORM_SECTION
                entry_date = None

            pieces = self._chunks(
                tokens=tokens,
                lines=lines,
                start_zero=heading.content_start,
                end_zero=heading.content_end,
                relative=relative,
                title=title,
                heading=heading.text,
                note_type=note_type,
                note_date=note_date,
                entry_date=entry_date,
                entry_date_end=entry_date_end,
                warnings=warnings,
            )
            if not pieces:
                continue
            occurrences[heading.path] += 1
            section_key = f"{'/'.join(heading.path)}:{occurrences[heading.path]}"
            for position, (chunk, prose) in enumerate(pieces):
                body_range = meaningful_line_range(lines, chunk.start_zero, chunk.end_zero)
                # The first piece is anchored to the heading so evidence and
                # anchored links still point at the section they belong to.
                if position == 0 or body_range is None:
                    start_line = heading.line
                else:
                    start_line = body_range.start_line
                end_line = body_range.end_line if body_range else heading.line
                structural_key = (
                    section_key
                    if len(pieces) == 1
                    else _unique_key(f"{section_key}:{_digest(prose.text)}", duplicates)
                )
                entries.append(
                    self._entry(
                        note_uuid,
                        relative,
                        title,
                        note_type,
                        entry_type,
                        prose.text,
                        heading.text,
                        heading.path,
                        SourceRange(start_line, end_line),
                        note_date,
                        entry_date,
                        prose,
                        structural_key,
                        entry_date_end=entry_date_end,
                    )
                )
                warnings.extend(prose.warnings)
        return tuple(entries), tuple(warnings)

    def _chunks(
        self,
        *,
        tokens: list,
        lines: list[str],
        start_zero: int,
        end_zero: int,
        relative: str,
        title: str,
        heading: str | None,
        note_type: NoteType,
        note_date: str | None,
        entry_date: str | None,
        warnings: list[IndexWarning],
        entry_date_end: str | None = None,
    ) -> list[tuple[Chunk, Prose]]:
        """Split one line range into chunks that fit the dense token budget."""
        blocks = paragraphs(lines, start_zero, end_zero)
        if not blocks:
            return []

        def extract(start: int, end: int) -> Prose:
            return prose_between(
                tokens, start, end, relative, self.settings.known_tags, self.settings.tag_aliases
            )

        per_paragraph = [extract(block.start_zero, block.end_zero) for block in blocks]
        # Every chunk's tags are a subset of the range's tags, so pricing the
        # label prefix with all of them keeps the budget conservative.
        all_tags = tuple(dict.fromkeys(tag for prose in per_paragraph for tag in prose.tags))
        prefix = _embedding_text(
            note_type, title, heading, note_date, entry_date, all_tags, "", entry_date_end
        )
        budget = content_budget(prefix, self.settings.dense_token_window)

        result: list[tuple[Chunk, Prose]] = []
        for chunk in pack(blocks, [estimate_tokens(prose.text) for prose in per_paragraph], budget):
            prose = extract(chunk.start_zero, chunk.end_zero)
            if not prose.text:
                continue
            cost = estimate_tokens(prose.text)
            if cost > budget:
                warnings.append(
                    IndexWarning(
                        WarningType.TRUNCATED_EMBEDDING_INPUT,
                        relative,
                        f"one paragraph needs about {cost} tokens but only {budget} reach the "
                        f"dense model; about {cost - budget} tokens are not semantically searchable",
                        chunk.start_zero + 1,
                    )
                )
            result.append((chunk, prose))
        return result

    @staticmethod
    def _entry(
        note_uuid: str,
        relative: str,
        title: str,
        note_type: NoteType,
        entry_type: EntryType,
        text: str,
        heading: str | None,
        heading_path: tuple[str, ...],
        source: SourceRange,
        note_date: str | None,
        entry_date: str | None,
        prose,
        structural_key: str,
        entry_date_end: str | None = None,
    ) -> ParsedEntry:
        identifier = str(entry_id(note_uuid, entry_type, structural_key))
        return ParsedEntry(
            entry_id=identifier,
            note_id=note_uuid,
            path=relative,
            note_title=title,
            note_type=note_type,
            entry_type=entry_type,
            text=text,
            embedding_text=_embedding_text(
                note_type, title, heading, note_date, entry_date, prose.tags, text, entry_date_end
            ),
            heading=heading,
            heading_path=heading_path,
            source=source,
            note_date=note_date,
            entry_date=entry_date,
            entry_date_end=entry_date_end,
            tags=prose.tags,
            outgoing_links=prose.links,
            web_links=prose.web_links,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _unique_key(key: str, duplicates: defaultdict[str, int]) -> str:
    """Keep identical text in one note distinct without depending on position."""
    duplicates[key] += 1
    return f"{key}:{duplicates[key]}"


def _embedding_text(
    note_type: NoteType,
    title: str,
    heading: str | None,
    note_date: str | None,
    entry_date: str | None,
    tags: tuple[str, ...],
    text: str,
    entry_date_end: str | None = None,
) -> str:
    labels = [f"Note: {title}"]
    if note_type is NoteType.LOG and note_date:
        labels.append(f"Date: {note_date}")
    if entry_date:
        labels.append(f"Date: {entry_date}" + (f" to {entry_date_end}" if entry_date_end else ""))
    if heading:
        labels.append(f"Section: {heading}")
    if tags:
        labels.append(f"Tags: {', '.join(tags)}")
    labels.append(f"Content: {text}")
    return "\n".join(labels)
