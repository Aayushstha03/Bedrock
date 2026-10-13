"""Cheap note headers, read without a full parse.

A scoped index update still needs every note's names, because link resolution
is vault-wide. Reading the file head for them costs a small fraction of a full
parse, so this module produces the same title `VaultParser` would, from the
frontmatter and the first level-one heading alone.
"""

from __future__ import annotations

import re
from pathlib import Path

from bedrock.core.config import Settings
from bedrock.core.contracts import NoteHeader
from bedrock.core.identity import note_id
from bedrock.parser.discovery import discover_markdown
from bedrock.parser.markdown import Frontmatter, frontmatter_aliases, parse_frontmatter

# CommonMark allows up to three spaces of indent, and an ATX heading may close
# with its own run of hashes. Four spaces would be an indented code block.
ATX_H1 = re.compile(r"^ {0,3}#\s+(.*?)(?:\s+#+)?\s*$")
SETEXT_H1 = re.compile(r"^ {0,3}=+\s*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def read_headers(settings: Settings, vault_id: str = "personal") -> tuple[NoteHeader, ...]:
    """Read one header for every note in the vault."""
    vault = settings.vault_path.resolve()
    return tuple(read_header(path, vault, vault_id) for path in discover_markdown(settings))


def read_header(path: Path, vault: Path, vault_id: str = "personal") -> NoteHeader:
    resolved = path.resolve()
    relative = resolved.relative_to(vault).as_posix()
    content = resolved.read_text(encoding="utf-8")
    frontmatter = parse_frontmatter(content, relative)
    return NoteHeader(
        note_id=str(note_id(vault_id, relative)),
        path=relative,
        title=header_title(content, frontmatter, resolved.stem),
        aliases=frontmatter_aliases(frontmatter.values),
    )


def header_title(content: str, frontmatter: Frontmatter, fallback: str) -> str:
    """Reproduce `note_title` without tokenizing the whole document."""
    explicit = frontmatter.values.get("title")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return _first_h1(content.splitlines(), frontmatter.body_start) or fallback


def _first_h1(lines: list[str], body_start: int) -> str | None:
    fence: str | None = None
    for index in range(body_start, len(lines)):
        line = lines[index]
        marker = FENCE.match(line)
        if marker is not None:
            if fence is None:
                fence = marker.group(1)[0]
            elif marker.group(1)[0] == fence:
                fence = None
            continue
        if fence is not None:
            continue
        atx = ATX_H1.match(line)
        if atx is not None and atx.group(1).strip():
            return atx.group(1).strip()
        # A run of equals signs under a non-blank line is a setext level-one
        # heading, which markdown-it reports the same way as `# Title`.
        if SETEXT_H1.match(line) and index > body_start and lines[index - 1].strip():
            return lines[index - 1].strip()
    return None
