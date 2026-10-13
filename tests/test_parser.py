import json
from pathlib import Path

from bedrock.core.config import Settings
from bedrock.core.contracts import EntryType, NoteType, WarningType
from bedrock.parser.service import VaultParser

FIXTURES = Path(__file__).parent / "fixtures"
VAULT = FIXTURES / "vault"


def parser() -> VaultParser:
    settings = Settings(
        "http://qdrant:6333",
        VAULT.resolve(),
        Path("/tmp/models"),
        "entries",
        "127.0.0.1",
        8080,
    )
    return VaultParser(settings)


def by_path(paths: list[str]):
    return {note.path: note for note in parser().parse_vault(paths)}


def summary(note):
    data = {
        "note_type": note.note_type.value,
        "entry_types": [entry.entry_type.value for entry in note.entries],
    }
    if note.note_type is NoteType.LOG:
        data["dates"] = [entry.note_date for entry in note.entries]
        data["headings"] = [entry.heading for entry in note.entries]
        data["tags"] = [list(entry.tags) for entry in note.entries]
    elif note.note_type is NoteType.KANBAN:
        data["columns"] = note.metadata["columns"]
        data["checked"] = [entry.board.checked for entry in note.entries]
    else:
        data["dates"] = [entry.entry_date for entry in note.entries]
    return data


def test_documented_shapes_match_golden_summary() -> None:
    expected = json.loads((FIXTURES / "expected" / "parser_summary.json").read_text())
    notes = by_path(list(expected))
    assert {path: summary(note) for path, note in notes.items()} == expected


def test_a_daily_note_outside_the_log_root_is_still_dated() -> None:
    """`Daily/2026-08-20.md` is byte-identical to `logs/2026-08-20.md`.

    Only the directory differs. A vault keeps daily notes in more than one
    place, and a note the parser leaves undated reaches the answering model as
    a result titled `2026-08-20` that claims to carry no date.
    """
    note = by_path(["Daily/2026-08-20.md"])["Daily/2026-08-20.md"]

    assert note.note_type is NoteType.LOG
    assert [entry.note_date for entry in note.entries] == ["2026-08-20"] * 3
    assert [entry.heading for entry in note.entries] == ["Work", "Thoughts", "Meals"]
    # The same file under the log root parses to the same thing.
    logged = by_path(["logs/2026-08-20.md"])["logs/2026-08-20.md"]
    assert summary(note) == summary(logged)


def test_standard_note_mixes_dated_and_undated_entries() -> None:
    note = by_path(["messy/deep/random/Mixed Note.md"])["messy/deep/random/Mixed Note.md"]
    detail = next(entry for entry in note.entries if entry.heading == "Detail")
    more_notes = next(entry for entry in note.entries if entry.heading == "More notes")
    assert detail.entry_type is EntryType.PROJECT_UPDATE
    assert detail.entry_date == "2026-08-18"
    assert more_notes.entry_type is EntryType.FREEFORM_SECTION
    assert more_notes.entry_date is None


def test_code_is_searchable_but_tags_and_links_come_only_from_prose() -> None:
    note = by_path(["freeform/Reference.md"])["freeform/Reference.md"]
    entry = note.entries[0]
    assert entry.tags == ("journal",)
    links = [link for item in note.entries for link in item.outgoing_links]
    assert [link.target_text for link in links] == [
        "News Resolution",
        "Does Not Exist",
        "Shared",
    ]
    # Each link reports the line it is written on. They share one paragraph, so
    # a link used to inherit the line of the first prose in its section.
    assert [link.line for link in links] == [20, 21, 22]
    text = "\n".join(item.text for item in note.entries)
    assert "#recipe [[Not a link]]" in text
    assert 'print("[[Still not a link]]")' in text
    assert "```python" not in text
    assert "https://example.test/page#fragment" in entry.web_links
    assert [warning.kind for warning in note.warnings] == [WarningType.UNKNOWN_TAG]


def test_log_date_fallback_and_invalid_date_warning() -> None:
    notes = by_path(["logs/from-content.md", "logs/no-date.md"])
    fallback = notes["logs/from-content.md"]
    assert all(entry.note_date == "2026-08-21" for entry in fallback.entries)
    assert fallback.entries[0].text == "Loose opening thought before any heading."
    invalid = notes["logs/no-date.md"]
    assert any(warning.kind is WarningType.INVALID_DATE for warning in invalid.warnings)
    assert invalid.entries


def test_dates_in_standard_prose_do_not_establish_chronology() -> None:
    note = by_path(["Daily/from-content.md"])["Daily/from-content.md"]
    assert note.note_type is NoteType.STANDARD
    assert all(entry.entry_date is None for entry in note.entries)


def test_kanban_preserves_state_order_nested_content_and_settings() -> None:
    notes = by_path(["kanban/Kitchen App.md", "kanban/Inconsistent.md"])
    kitchen = notes["kanban/Kitchen App.md"]
    assert kitchen.metadata["columns"] == ["ToDo", "Doing", "complete"]
    assert "Preserve this nested detail" in kitchen.entries[1].text
    assert "kanban-plugin" not in "\n".join(entry.text for entry in kitchen.entries)
    inconsistent = notes["kanban/Inconsistent.md"]
    assert inconsistent.entries[0].board.checked is True
    assert inconsistent.entries[0].board.status == "doing"
    assert inconsistent.entries[1].board.checked is False
    assert inconsistent.entries[1].board.status == "complete"
    assert any(warning.kind is WarningType.INVALID_KANBAN_SETTINGS for warning in inconsistent.warnings)


def test_parser_is_deterministic_and_embedding_text_has_context() -> None:
    first = parser().parse_vault(["projects/News Resolution.md"])[0]
    second = parser().parse_vault(["projects/News Resolution.md"])[0]
    assert first == second
    assert first.entries[0].embedding_text.startswith("Note: News Resolution\nDate: 2026-08-19")
    assert first.entries[0].source.start_line == 3


def test_all_source_ranges_are_valid_for_the_fixture_vault() -> None:
    for note in parser().parse_vault():
        line_count = len((VAULT / note.path).read_text().splitlines())
        for entry in note.entries:
            assert 1 <= entry.source.start_line <= entry.source.end_line <= line_count
