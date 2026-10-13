"""Coverage for the chunking path: no entry silently overruns the encoder."""

from itertools import pairwise
from pathlib import Path

from bedrock.core.config import Settings
from bedrock.core.contracts import EntryType, WarningType
from bedrock.parser.service import VaultParser
from bedrock.parser.tokens import content_budget, estimate_tokens

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


def settings(vault: Path) -> Settings:
    return Settings("http://unused", vault, vault / "models", "entries", "127.0.0.1", 8080)


def parse(vault: Path):
    return {note.path: note for note in VaultParser(settings(vault)).parse_vault()}


def test_oversized_section_becomes_several_entries_that_keep_their_heading() -> None:
    note = parse(FIXTURE_VAULT)["freeform/Oversized.md"]
    pieces = [entry for entry in note.entries if entry.heading == "Migration Notes"]

    assert len(pieces) > 1
    assert {entry.entry_type for entry in pieces} == {EntryType.FREEFORM_SECTION}
    assert {entry.heading_path for entry in pieces} == {("Oversized Reference", "Migration Notes")}
    assert len({entry.entry_id for entry in pieces}) == len(pieces)


def test_the_tail_of_an_oversized_section_is_still_indexed() -> None:
    note = parse(FIXTURE_VAULT)["freeform/Oversized.md"]
    text = "\n".join(entry.text for entry in note.entries)
    # Before section chunking this term sat past the encoder window and was
    # unreachable by dense search even though the entry claimed those lines.
    assert "saffron" in text
    tail = next(entry for entry in note.entries if "saffron" in entry.text)
    assert estimate_tokens(tail.embedding_text) <= 128


def test_a_section_that_fits_stays_one_entry() -> None:
    note = parse(FIXTURE_VAULT)["freeform/Oversized.md"]
    assert len([entry for entry in note.entries if entry.heading == "Short Section"]) == 1


def test_a_headless_note_produces_freeform_chunks() -> None:
    note = parse(FIXTURE_VAULT)["freeform/Headless Long.md"]
    assert len(note.entries) > 1
    assert {entry.entry_type for entry in note.entries} == {EntryType.FREEFORM_CHUNK}
    assert all(entry.heading is None for entry in note.entries)
    assert all(entry.heading_path == () for entry in note.entries)


def test_every_fixture_entry_fits_the_dense_token_budget() -> None:
    # Assert the invariant that actually matters: the complete embedding input,
    # labels included, stays inside the window. Reconstructing the label prefix
    # per entry type is fragile, and Kanban cards do not use a `Content:` label
    # at all.
    window = settings(FIXTURE_VAULT).dense_token_window
    for note in parse(FIXTURE_VAULT).values():
        for entry in note.entries:
            estimate = estimate_tokens(entry.embedding_text)
            assert estimate <= window, f"{entry.path} {entry.heading or entry.text!r} = {estimate}"


def test_chunk_line_ranges_stay_inside_the_note_and_never_overlap() -> None:
    for note in parse(FIXTURE_VAULT).values():
        total = len((FIXTURE_VAULT / note.path).read_text().splitlines())
        for entry in note.entries:
            assert 1 <= entry.source.start_line <= entry.source.end_line <= total

        # Chunks of one section must partition it, not overlap. Compare only
        # within a heading, because a section's first chunk starts at the
        # heading line and Kanban cards are ordered by column, not by line.
        by_heading: dict[tuple[str, ...], list] = {}
        for entry in note.entries:
            if entry.entry_type is not EntryType.KANBAN_CARD:
                by_heading.setdefault(entry.heading_path, []).append(entry)
        for pieces in by_heading.values():
            ordered = sorted(pieces, key=lambda item: item.source.start_line)
            for earlier, later in pairwise(ordered):
                assert earlier.source.end_line < later.source.start_line, (
                    f"{note.path} chunks overlap: {earlier.source} then {later.source}"
                )


def test_chunk_identity_survives_an_insertion_earlier_in_the_note(tmp_path: Path) -> None:
    # Size each paragraph from the real budget so exactly one fills a chunk,
    # which is the shape an oversized note actually has. Deriving it keeps the
    # test honest if the token window ever changes.
    budget = content_budget("Note: Loose\nContent: ")

    def paragraph(index: int) -> str:
        words = ["reindexing", "identifiers", "embeddings", "alias", "switching", "removal"]
        body = " ".join(words * (budget // len(words)))
        return f"Paragraph {index} discusses {body}."

    body = "\n\n".join(paragraph(index) for index in range(5))
    note_path = tmp_path / "Loose.md"
    note_path.write_text(body + "\n")
    before = {entry.text: entry.entry_id for entry in parse(tmp_path)["Loose.md"].entries}
    assert len(before) >= 4, "each paragraph should fill its own chunk"

    note_path.write_text(paragraph(99) + "\n\n" + body + "\n")
    after = {entry.text: entry.entry_id for entry in parse(tmp_path)["Loose.md"].entries}

    shared = set(before) & set(after)
    assert len(shared) >= 4, "unchanged paragraphs should keep their chunk"
    # A positional key would renumber every chunk after the insertion and force
    # a re-encode of text that did not change.
    assert all(before[text] == after[text] for text in shared)


def test_repeated_identical_chunks_keep_distinct_identifiers(tmp_path: Path) -> None:
    paragraph = "This paragraph repeats word for word inside the very same note."
    (tmp_path / "Repeat.md").write_text("\n\n".join([paragraph] * 3) + "\n")
    entries = parse(tmp_path)["Repeat.md"].entries
    assert len({entry.entry_id for entry in entries}) == len(entries)


def test_an_unsplittable_paragraph_warns_instead_of_truncating_silently(tmp_path: Path) -> None:
    giant = " ".join(f"sentence{index} about indexing and retrieval" for index in range(120))
    (tmp_path / "Giant.md").write_text(f"# Giant\n\n## Section\n\n{giant}\n")
    note = parse(tmp_path)["Giant.md"]

    warning = next(w for w in note.warnings if w.kind is WarningType.TRUNCATED_EMBEDDING_INPUT)
    assert "not semantically searchable" in warning.message
    # One paragraph cannot be divided without cutting a sentence, so it stays
    # a single entry and the warning is the only honest signal available.
    assert len([entry for entry in note.entries if entry.heading == "Section"]) == 1


def test_parsing_is_repeatable() -> None:
    first = parse(FIXTURE_VAULT)
    second = parse(FIXTURE_VAULT)
    assert [(p, [e.entry_id for e in n.entries]) for p, n in sorted(first.items())] == [
        (p, [e.entry_id for e in n.entries]) for p, n in sorted(second.items())
    ]


def test_an_oversized_kanban_card_warns_and_stays_one_point(tmp_path: Path) -> None:
    body = " ".join(f"detail{index} about the migration and the rollback plan" for index in range(80))
    (tmp_path / "kanban").mkdir()
    (tmp_path / "kanban" / "Board.md").write_text(
        f"---\nkanban-plugin: board\n---\n\n# Board\n\n## ToDo\n\n- [ ] Card one\n\t{body}\n"
    )
    note = parse(tmp_path)["kanban/Board.md"]

    assert len(note.entries) == 1
    assert note.entries[0].entry_type is EntryType.KANBAN_CARD
    warning = next(w for w in note.warnings if w.kind is WarningType.TRUNCATED_EMBEDDING_INPUT)
    assert "a card is never divided" in warning.message


def test_a_normal_kanban_card_does_not_warn() -> None:
    note = parse(FIXTURE_VAULT)["kanban/Kitchen App.md"]
    assert not [w for w in note.warnings if w.kind is WarningType.TRUNCATED_EMBEDDING_INPUT]


def test_the_encoder_window_matches_what_the_parser_budgets_against() -> None:
    """The parser's budget and the encoder's real limit must never diverge.

    FastEmbed defaults this model to 128 positions while the model itself
    declares 256. If `LocalEncoders` did not pin the tokenizer, raising
    `BEDROCK_DENSE_TOKEN_WINDOW` would widen the parser's chunks while the
    encoder kept cutting at its own default, silently losing the difference.
    """
    from fastembed import TextEmbedding

    from bedrock.index.encoders import LocalEncoders

    configured = Settings(
        "http://unused",
        FIXTURE_VAULT,
        FIXTURE_VAULT / "models",
        "entries",
        "127.0.0.1",
        8080,
        dense_token_window=192,
    )
    encoders = LocalEncoders(
        configured, dense=TextEmbedding(model_name=configured.dense_model), sparse=object()
    )
    assert encoders._tokenizer().truncation["max_length"] == 192


def test_a_stub_encoder_without_a_tokenizer_is_left_alone() -> None:
    from bedrock.index.encoders import LocalEncoders

    encoders = LocalEncoders(settings(FIXTURE_VAULT), dense=object(), sparse=object())
    assert encoders._tokenizer() is None
    assert encoders.truncated_inputs == 0
