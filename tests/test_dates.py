import pytest

from bedrock.parser.markdown import heading_dates, headings, markdown_tokens, valid_iso_date


@pytest.mark.parametrize("value", ["2026-08-20", "2000-02-29", "0001-01-01"])
def test_exact_yyyy_mm_dd_dates_are_valid(value: str) -> None:
    assert valid_iso_date(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "2026-8-20",
        "26-08-20",
        "2026/08/20",
        "2026-08-20T00:00:00Z",
        "2026-08-20 update",
        "date: 2026-08-20",
        " 2026-08-20",
        "2026-08-20 ",
        "2026-02-29",
        "２０２６-０８-２０",
    ],
)
def test_every_other_date_representation_is_invalid(value: str) -> None:
    assert valid_iso_date(value) is None


def test_only_exact_date_heading_establishes_chronology() -> None:
    content = "## 2026-08-20\nExact.\n## 2026-08-21 update\nNot dated.\n"
    parsed = headings(markdown_tokens(content), len(content.splitlines()))
    assert [heading.active_date for heading in parsed] == ["2026-08-20", None]


@pytest.mark.parametrize("value", ["2026-10-01 to 2026-10-02", "2026-10-01 to 2026-10-01"])
def test_heading_date_intervals_are_inclusive_and_explicit(value):
    assert heading_dates(value) == tuple(value.split(" to "))


@pytest.mark.parametrize(
    "value",
    [
        "2026-10-02 to 2026-10-01",
        "2026-02-29 to 2026-03-01",
        "2026-10-01 - 2026-10-02",
        "2026-10-01 to 2026-10-02 recap",
    ],
)
def test_invalid_or_noncanonical_heading_intervals_are_undated(value):
    assert heading_dates(value) == (None, None)


def test_children_inherit_intervals_but_own_dates_override_them():
    content = "## 2026-10-01 to 2026-10-02\nSummary.\n### Detail\nBody.\n### 2026-10-03\nLater.\n## Other\nUndated.\n"
    parsed = headings(markdown_tokens(content), len(content.splitlines()))
    assert [(h.active_date, h.active_date_end) for h in parsed] == [
        ("2026-10-01", "2026-10-02"),
        ("2026-10-01", "2026-10-02"),
        ("2026-10-03", None),
        (None, None),
    ]
