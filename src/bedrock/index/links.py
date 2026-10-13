"""Deterministic vault-wide internal-link resolution."""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from bedrock.core.contracts import (
    IndexWarning,
    Link,
    LinkResolution,
    NoteHeader,
    ParsedEntry,
    ParsedNote,
    WarningType,
)
from bedrock.parser.markdown import frontmatter_aliases


@dataclass(frozen=True, slots=True)
class Catalog:
    """Every name the vault answers to, and the note behind each one.

    Building this needs every note, but only four fields of each. Resolving one
    note against it needs nothing else, which is the split a scoped update
    depends on.
    """

    by_name: dict[str, list[NoteHeader]]
    by_path: dict[str, NoteHeader]


def note_header(note: ParsedNote) -> NoteHeader:
    return NoteHeader(note.note_id, note.path, note.title, frontmatter_aliases(note.metadata))


def build_catalog(headers: Iterable[NoteHeader]) -> Catalog:
    by_name: defaultdict[str, list[NoteHeader]] = defaultdict(list)
    by_path: dict[str, NoteHeader] = {}
    for header in headers:
        by_path[path_key(header.path)] = header
        for name in _name_keys(header):
            by_name[name].append(header)
    return Catalog(by_name, by_path)


def note_names(header: NoteHeader) -> frozenset[str]:
    """Every key a wikilink can use to reach this note, path form included."""
    return frozenset({*_name_keys(header), path_key(header.path)})


def link_key(target_text: str) -> str:
    """The one normalized string a wikilink is compared against."""
    without_suffix = _without_suffix(target_text)
    return _key(without_suffix.strip("/") if "/" in without_suffix else without_suffix)


def path_key(path: str) -> str:
    return _key(PurePosixPath(path).with_suffix("").as_posix())


def resolve_links(notes: tuple[ParsedNote, ...]) -> tuple[ParsedNote, ...]:
    catalog = build_catalog(note_header(note) for note in notes)
    return tuple(resolve_note(note, catalog) for note in notes)


def resolve_note(note: ParsedNote, catalog: Catalog) -> ParsedNote:
    warnings = list(note.warnings)
    entries: list[ParsedEntry] = []
    for entry in note.entries:
        links: list[Link] = []
        for link in entry.outgoing_links:
            candidates = _candidates(link.target_text, catalog)
            if len(candidates) == 1:
                links.append(
                    replace(
                        link,
                        target_note_id=candidates[0].note_id,
                        resolution=LinkResolution.RESOLVED,
                    )
                )
            elif len(candidates) > 1:
                links.append(replace(link, resolution=LinkResolution.AMBIGUOUS))
                warnings.append(
                    IndexWarning(
                        WarningType.AMBIGUOUS_LINK,
                        note.path,
                        f"ambiguous internal link: [[{link.target_text}]]",
                        link.line,
                    )
                )
            else:
                links.append(replace(link, resolution=LinkResolution.MISSING))
                warnings.append(
                    IndexWarning(
                        WarningType.MISSING_LINK,
                        note.path,
                        f"missing internal link: [[{link.target_text}]]",
                        link.line,
                    )
                )
        entries.append(replace(entry, outgoing_links=tuple(links)))
    return replace(note, entries=tuple(entries), warnings=tuple(dict.fromkeys(warnings)))


def resolve_link(notes: tuple[ParsedNote, ...], source_note_id: str, target_text: str) -> Link:
    if not any(note.note_id == source_note_id for note in notes):
        raise LookupError(f"source note not found: {source_note_id}")
    raw = target_text.strip()
    if raw.startswith("[[") and raw.endswith("]]"):
        raw = raw[2:-2].strip()
    destination, alias_separator, alias = raw.partition("|")
    target, separator, heading = destination.partition("#")
    if not target.strip():
        raise ValueError("target_text must name a note")
    catalog = build_catalog(note_header(note) for note in notes)
    candidates = _candidates(target, catalog)
    if len(candidates) == 1:
        return Link(
            target_text=target.strip(),
            target_note_id=candidates[0].note_id,
            target_heading=heading.strip() if separator and heading.strip() else None,
            alias=alias.strip() if alias_separator and alias.strip() else None,
            resolution=LinkResolution.RESOLVED,
        )
    return Link(
        target_text=target.strip(),
        target_heading=heading.strip() if separator and heading.strip() else None,
        alias=alias.strip() if alias_separator and alias.strip() else None,
        resolution=(LinkResolution.AMBIGUOUS if candidates else LinkResolution.MISSING),
    )


def _name_keys(header: NoteHeader) -> frozenset[str]:
    names = {PurePosixPath(header.path).stem, header.title, *header.aliases}
    return frozenset(_key(name) for name in names if name.strip())


def _candidates(target: str, catalog: Catalog) -> list[NoteHeader]:
    without_suffix = _without_suffix(target)
    key = link_key(target)
    if "/" in without_suffix:
        match = catalog.by_path.get(key)
        return [match] if match is not None else []
    unique = {header.note_id: header for header in catalog.by_name.get(key, [])}
    return sorted(unique.values(), key=lambda header: header.path.casefold())


def _without_suffix(target: str) -> str:
    normalized = target.strip().replace("\\", "/")
    return normalized[:-3] if normalized.casefold().endswith(".md") else normalized


def _key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold().strip()
