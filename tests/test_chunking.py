from bedrock.parser.chunking import Chunk, pack, paragraphs
from bedrock.parser.tokens import (
    DEFAULT_TOKEN_WINDOW,
    MINIMUM_CONTENT_TOKENS,
    content_budget,
    estimate_tokens,
)


def test_paragraphs_split_on_blank_lines() -> None:
    lines = ["a" * 10, "", "b" * 10, "", "c" * 10]
    assert paragraphs(lines, 0, len(lines)) == (Chunk(0, 1), Chunk(2, 3), Chunk(4, 5))


def test_empty_ranges_make_no_paragraphs() -> None:
    assert paragraphs(["", "  "], 0, 2) == ()


def test_pack_merges_paragraphs_until_the_budget_is_reached() -> None:
    blocks = (Chunk(0, 1), Chunk(2, 3), Chunk(4, 5))
    assert pack(blocks, (4, 4, 4), budget=8) == (Chunk(0, 3), Chunk(4, 5))


def test_pack_never_divides_one_paragraph() -> None:
    blocks = (Chunk(0, 9),)
    assert pack(blocks, (500,), budget=8) == (Chunk(0, 9),)


def test_pack_rejects_mismatched_costs() -> None:
    try:
        pack((Chunk(0, 1),), (1, 2), budget=8)
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("mismatched costs must raise")


def test_estimate_never_undercounts_real_wordpiece_shapes() -> None:
    # Reference counts came from the pinned tokenizer; the estimate must stay
    # at or above each one so the parser's budget cannot overrun the model.
    reference = {
        "I went to the market this morning and bought some fresh bread and a bit of cheese.": 18,
        "The retrieval pipeline reclusters candidate documents after fusion deterministically.": 15,
        "2026-08-19 and 2026-08-20 were both logged, see 2026-01-01.": 25,
        "2026-08-19 2026-08-20 2026-08-21 2026-08-22 2026-08-23 2026-08-24": 36,
        "Call resolve_links(notes, source_note_id) then check target_entry_id != None.": 24,
        "See https://example.com/some/deep/path?query=1 for the writeup.": 23,
        "会議のメモを書きました。今日は雨が降っています。": 24,
        "Shipped it 🚀🎉 and everyone was happy 😀": 8,
        "- [ ] Todo item with **bold** and `inline code` and [[Wiki Link#Anchor]]": 27,
    }
    for text, real_tokens in reference.items():
        assert estimate_tokens(text) >= real_tokens, text


def test_estimate_is_empty_for_blank_text() -> None:
    assert estimate_tokens("   \n  ") == 0


def test_content_budget_shrinks_as_the_label_prefix_grows() -> None:
    short = content_budget("Note: A\nContent: ")
    long = content_budget(
        "Note: A Much Longer Note Title\nSection: Deep Heading\nTags: work, journal\nContent: "
    )
    assert short > long
    assert short < DEFAULT_TOKEN_WINDOW


def test_content_budget_never_drops_below_the_floor() -> None:
    assert content_budget("word " * 500) == MINIMUM_CONTENT_TOKENS
