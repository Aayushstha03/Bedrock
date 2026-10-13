"""Kanban cards carry dates like every other entry type."""

from pathlib import Path

from bedrock.core.config import Settings
from bedrock.core.contracts import WarningType
from bedrock.parser.kanban import card_annotations
from bedrock.parser.service import VaultParser

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


def settings(vault: Path) -> Settings:
    return Settings("http://unused", vault, vault / "models", "entries", "127.0.0.1", 8080)


def parse(vault: Path):
    return {note.path: note for note in VaultParser(settings(vault)).parse_vault()}


def board(vault: Path, path: str):
    # The same card text can appear in more than one column, so key on both.
    return {(entry.board.column, entry.text): entry for entry in parse(vault)[path].entries}


def test_card_date_becomes_the_entry_date() -> None:
    cards = board(FIXTURE_VAULT, "kanban/Kitchen App.md")
    assert cards[("ToDo", "Timer for stuff in fridge/pantry/freezer")].entry_date == "2026-08-22"
    assert cards[("complete", "Finished kitchen setup")].entry_date == "2026-08-20"


def test_a_card_without_a_date_stays_undated() -> None:
    cards = board(FIXTURE_VAULT, "kanban/Kitchen App.md")
    assert cards[("ToDo", "Third todo card")].entry_date is None
    # A Kanban board file is not itself dated.
    assert all(entry.note_date is None for entry in parse(FIXTURE_VAULT)["kanban/Kitchen App.md"].entries)


def test_the_daily_note_link_form_is_a_date_and_not_a_link() -> None:
    card = board(FIXTURE_VAULT, "kanban/Kitchen App.md")[("ToDo", "Card dated by daily-note link")]
    assert card.entry_date == "2026-08-20"
    # `@[[2026-08-20]]` is how the plugin renders a date, not a real reference.
    assert card.outgoing_links == ()


def test_a_real_wiki_link_on_a_card_still_resolves() -> None:
    card = board(FIXTURE_VAULT, "kanban/Kitchen App.md")[
        ("ToDo", "Fourth todo card with [[News Resolution]]")
    ]
    assert [link.target_text for link in card.outgoing_links] == ["News Resolution"]


def test_plugin_syntax_never_reaches_searchable_text_or_the_embedding() -> None:
    for entry in parse(FIXTURE_VAULT)["kanban/Kitchen App.md"].entries:
        assert "@{" not in entry.text
        assert "@@{" not in entry.text
        assert "@{" not in entry.embedding_text
        assert "@@{" not in entry.embedding_text


def test_the_card_time_is_kept_as_structured_metadata() -> None:
    card = board(FIXTURE_VAULT, "kanban/Kitchen App.md")[("complete", "Finished kitchen setup")]
    assert card.board.card_time == "14:30"
    assert card.board.checked is True


def test_an_impossible_date_warns_and_dates_nothing() -> None:
    note = parse(FIXTURE_VAULT)["kanban/Kitchen App.md"]
    card = next(entry for entry in note.entries if entry.text == "Card with an impossible date")
    assert card.entry_date is None
    assert any(w.kind is WarningType.INVALID_DATE and "2026-02-30" in w.message for w in note.warnings)


def test_the_entry_date_reaches_the_embedding_input() -> None:
    card = board(FIXTURE_VAULT, "kanban/Kitchen App.md")[("complete", "Finished kitchen setup")]
    assert "Date: 2026-08-20" in card.embedding_text


def test_a_custom_date_trigger_from_board_settings_is_honoured(tmp_path: Path) -> None:
    (tmp_path / "kanban").mkdir()
    (tmp_path / "kanban" / "Custom.md").write_text(
        "---\nkanban-plugin: board\n---\n\n"
        "## ToDo\n\n- [ ] Custom trigger card !{2026-07-04}\n\n"
        "%% kanban:settings\n```json\n"
        '{"kanban-plugin":"board","date-trigger":"!"}\n'
        "```\n%%\n"
    )
    card = board(tmp_path, "kanban/Custom.md")[("ToDo", "Custom trigger card")]
    assert card.entry_date == "2026-07-04"


def test_more_than_one_date_on_a_card_warns_and_takes_the_first(tmp_path: Path) -> None:
    (tmp_path / "kanban").mkdir()
    (tmp_path / "kanban" / "Two.md").write_text(
        "---\nkanban-plugin: board\n---\n\n## ToDo\n\n- [ ] Two dates @{2026-05-12} @{2026-06-01}\n"
    )
    note = parse(tmp_path)["kanban/Two.md"]
    assert note.entries[0].entry_date == "2026-05-12"
    assert any(
        w.kind is WarningType.INVALID_DATE and "more than one date" in w.message for w in note.warnings
    )


def test_annotations_are_stripped_longest_trigger_first() -> None:
    # `@@` starts with `@`, so a naive pass would read `@{14:30}` as a date.
    text, date, time, consumed, warnings = card_annotations(
        "Call the plumber @{2026-05-12} @@{14:30}", "x.md", 1, "@", "@@"
    )
    assert (text, date, time) == ("Call the plumber", "2026-05-12", "14:30")
    assert warnings == []
    assert set(consumed) == {"2026-05-12", "14:30"}


def test_find_kanban_cards_accepts_a_date_range() -> None:
    from inspect import signature

    from bedrock.library import Bedrock

    parameters = signature(Bedrock.find_kanban_cards).parameters
    assert "date_from" in parameters and "date_to" in parameters


def test_a_card_that_is_only_an_annotation_leaks_no_plugin_syntax(tmp_path: Path) -> None:
    (tmp_path / "kanban").mkdir()
    (tmp_path / "kanban" / "Bare.md").write_text(
        "---\nkanban-plugin: board\n---\n\n## ToDo\n\n"
        "- [ ] @{2026-05-12}\n- [ ] @@{14:30}\n- [ ] @[[2026-06-06]]\n"
    )
    entries = parse(tmp_path)["kanban/Bare.md"].entries
    assert len(entries) == 3
    for entry in entries:
        # The card has no prose at all. Falling back to the raw line would put
        # the plugin syntax straight back into the searchable text.
        assert entry.text == ""
        for marker in ("@{", "@@{", "@[["):
            assert marker not in entry.text
            assert marker not in entry.embedding_text
    assert [entry.entry_date for entry in entries] == ["2026-05-12", None, "2026-06-06"]
    assert entries[1].board.card_time == "14:30"
