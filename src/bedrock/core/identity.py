"""Deterministic Qdrant-compatible UUID identity generation."""

from __future__ import annotations

import unicodedata
from pathlib import PurePosixPath
from uuid import NAMESPACE_URL, UUID, uuid5

IDENTITY_VERSION = "1"


def normalize_vault_path(path: str) -> str:
    raw = unicodedata.normalize("NFC", path.replace("\\", "/"))
    if raw.startswith("/"):
        raise ValueError(f"invalid vault-relative path: {path!r}")
    normalized = raw.strip("/")
    candidate = PurePosixPath(normalized)
    if not normalized or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"invalid vault-relative path: {path!r}")
    return candidate.as_posix()


def note_id(vault_id: str, path: str) -> UUID:
    if not vault_id.strip():
        raise ValueError("vault_id cannot be empty")
    key = f"bedrock:v{IDENTITY_VERSION}:note:{vault_id}:{normalize_vault_path(path)}"
    return uuid5(NAMESPACE_URL, key)


def entry_id(note: UUID | str, entry_type: str, structural_key: str) -> UUID:
    if not entry_type.strip() or not structural_key.strip():
        raise ValueError("entry_type and structural_key cannot be empty")
    key = f"bedrock:v{IDENTITY_VERSION}:entry:{note}:{entry_type}:{structural_key}"
    return uuid5(NAMESPACE_URL, key)
