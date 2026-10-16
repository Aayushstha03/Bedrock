"""Vault discovery and the only path-based note classification rules."""

from __future__ import annotations

from pathlib import Path

from bedrock.core.config import Settings
from bedrock.core.contracts import NoteType
from bedrock.core.identity import normalize_vault_path
from bedrock.parser.markdown import valid_iso_date

INTERNAL_DIRECTORIES = frozenset({".bedrock-state", ".bedrock-backups", ".bedrock-trash"})


def classify_path(relative_path: str, settings: Settings) -> NoteType:
    """Classify one note by where it sits, and by what it is called.

    A note named for a day is a log wherever it sits. One configured root does
    not describe a real vault: daily notes collect under `logs/`, under an
    archive, under a per-client folder. A note left standard is a note the
    parser never dates, and the answering model is then handed a result whose
    title reads `2026-08-20` next to a claim that the entry has no date. It
    resolves that contradiction by reading the date off the title, which is
    exactly the guess the dating rule exists to prevent.

    The Kanban root is checked first, so a board named for a date stays a
    board.
    """
    path = Path(normalize_vault_path(relative_path))
    root = path.parts[0].casefold() if len(path.parts) > 1 else ""
    if root == settings.kanban_root.casefold():
        return NoteType.KANBAN
    if root == settings.log_root.casefold() or valid_iso_date(path.stem):
        return NoteType.LOG
    return NoteType.STANDARD


def discover_markdown(settings: Settings, requested: list[str] | None = None) -> tuple[Path, ...]:
    vault = settings.vault_path.resolve()
    excluded = {name.casefold() for name in settings.excluded_directories} | INTERNAL_DIRECTORIES
    excluded_files = {name.casefold() for name in settings.excluded_files}
    if requested:
        candidates = [vault / normalize_vault_path(path) for path in requested]
    else:
        candidates = list(vault.rglob("*.md"))

    discovered: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(vault)
        except ValueError as exc:
            raise ValueError(f"path escapes vault: {candidate}") from exc
        if resolved.suffix.casefold() != ".md" or not resolved.is_file():
            continue
        if any(part.casefold() in excluded for part in relative.parts[:-1]):
            continue
        if relative.name.casefold() in excluded_files:
            continue
        discovered.append(resolved)
    return tuple(sorted(set(discovered), key=lambda path: path.relative_to(vault).as_posix().casefold()))
