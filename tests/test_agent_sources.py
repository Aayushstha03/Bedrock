"""The labels an answering run cites, and what they resolve back to."""

from zenith.agent.sources import Sources
from zenith.core.contracts import (
    EntryType,
    NoteContent,
    NoteType,
    RetrievalMode,
    SearchResult,
)


def _result(entry_id: str, *, title: str = "News Resolution", start: int = 6) -> SearchResult:
    return SearchResult(
        entry_id,
        "n1",
        "projects/News Resolution.md",
        title,
        NoteType.STANDARD,
        EntryType.FREEFORM_SECTION,
        RetrievalMode.HYBRID,
        "text",
        "2026-08-17",
        (),
        start,
        start + 3,
        None,
        None,
        (),
        (),
    )


_EXPECTED = {
    "note": "News Resolution",
    "path": "projects/News Resolution.md",
    "heading": "2026-08-17",
    "lines": "6-9",
}


def test_labels_count_up_and_an_entry_seen_twice_keeps_one() -> None:
    """Search, backlinks, and context expansion overlap.

    Two labels for one entry would read as two independent pieces of evidence
    for the same claim, so the second sighting reuses the first label.
    """
    sources = Sources(None)

    assert sources.label(_result("e1")) == "s1"
    assert sources.label(_result("e2", start=13)) == "s2"
    assert sources.label(_result("e1")) == "s1"
    assert sources.label(_result("e3", start=20)) == "s3"


def test_a_cited_label_resolves_to_the_entry_behind_it() -> None:
    sources = Sources(None)
    label = sources.label(_result("e1"))

    assert sources.cited(f"You added a parsing model. [{label}]") == {
        "s1": {
            "note": "News Resolution",
            "path": "projects/News Resolution.md",
            "heading": "2026-08-17",
            "lines": "6-9",
        }
    }


def test_an_invented_label_is_reported_rather_than_dropped() -> None:
    """An answer citing evidence that does not exist must not look like one
    citing evidence that does."""
    sources = Sources(None)
    sources.label(_result("e1"))

    assert sources.cited("A claim. [s1] Another. [s7]") == {
        "s1": {
            "note": "News Resolution",
            "path": "projects/News Resolution.md",
            "heading": "2026-08-17",
            "lines": "6-9",
        },
        "s7": {"unknown": True},
    }


def test_a_bare_id_in_prose_is_not_a_citation() -> None:
    """The brackets are what make an id a citation.

    An answer is free to talk about an S3 bucket. Reading that as a citation
    would invent one, or credit a real source the sentence never cited.
    """
    sources = Sources(None)
    sources.label(_result("e1"))
    sources.label(_result("e2", start=13))
    sources.label(_result("e3", start=20))

    assert sources.cited("I moved the backups to s3 and left s1 alone.") == {}
    assert sources.cited("A claim. [s3]") == {"s3": _EXPECTED | {"lines": "20-23"}}


def test_a_sentence_resting_on_several_sources_resolves_all_of_them() -> None:
    sources = Sources(None)
    sources.label(_result("e1"))
    sources.label(_result("e2", start=13))

    both = {"s1": _EXPECTED, "s2": _EXPECTED | {"lines": "13-16"}}
    assert sources.cited("A claim. [s1][s2]") == both
    assert sources.cited("A claim. [s1, s2]") == {}
    assert sources.citation_check("A claim. [s1, s2]").valid is False


def test_a_bracket_carrying_no_id_cites_nothing() -> None:
    """The instructions reject these forms, so a model that writes one anyway
    has cited nothing, not something."""
    sources = Sources(None)
    sources.label(_result("e1"))

    assert sources.cited("A claim. [News Resolution]") == {}
    assert sources.cited("A claim. [News Resolution, 2026-08-17, lines 6-9]") == {}
    assert sources.cited("A claim. [s1, lines 6-9]") == {}
    assert sources.citation_check("A claim. [s1, lines 6-9]").valid is False


def test_citation_check_requires_issued_sources_to_be_cited() -> None:
    sources = Sources(None)
    sources.label(_result("e1"))

    assert sources.citation_check("I found an answer.").errors == (
        "answer used vault evidence but contains no valid issued citation",
    )
    assert sources.citation_check("I found an answer. [s1]").valid is True
    assert sources.citation_check("I found an answer. s1").valid is False
    assert sources.citation_check("I found an answer. [s99]").valid is False


def test_a_whole_note_resolves_without_claiming_a_line_range() -> None:
    """`read_note` has no line range to cite, which is why a citation built
    from one could not be written for it at all. A label can."""
    sources = Sources(None)
    label = sources.label_note(
        NoteContent("n1", "AI safety 2026.md", "AI safety 2026", NoteType.STANDARD, "body")
    )

    assert sources.cited(f"The links are these. [{label}]") == {
        "s1": {"note": "AI safety 2026", "path": "AI safety 2026.md"}
    }


def test_the_labels_a_run_may_expand_are_named_when_one_misses() -> None:
    sources = Sources(None)
    sources.label(_result("e1"))

    assert sources.entry_id("s1") == "e1"
    try:
        sources.entry_id("s4")
    except LookupError as error:
        assert "The results so far are s1." in str(error)
    else:  # pragma: no cover - the lookup must fail
        raise AssertionError("an unknown label must not resolve")
