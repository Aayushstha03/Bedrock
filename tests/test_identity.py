from uuid import UUID

import pytest

from bedrock.core.identity import entry_id, normalize_vault_path, note_id


def test_paths_normalize_cross_platform_separators() -> None:
    assert normalize_vault_path(r"projects\News Resolution.md") == "projects/News Resolution.md"


def test_note_ids_are_deterministic_and_vault_scoped() -> None:
    first = note_id("personal", "projects/News Resolution.md")
    assert isinstance(first, UUID)
    assert first == note_id("personal", "projects/News Resolution.md")
    assert first != note_id("work", "projects/News Resolution.md")


def test_entry_ids_are_structurally_scoped() -> None:
    note = note_id("personal", "projects/News Resolution.md")
    first = entry_id(note, "project_update", "News Resolution/2026-08-19")
    assert first == entry_id(note, "project_update", "News Resolution/2026-08-19")
    assert first != entry_id(note, "project_update", "News Resolution/2026-08-17")


@pytest.mark.parametrize("path", ["", "../outside.md", "folder/../../outside.md", "/absolute.md"])
def test_invalid_vault_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValueError):
        normalize_vault_path(path)
