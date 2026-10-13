from pathlib import Path

import pytest

from bedrock.core.config import Settings
from bedrock.core.contracts import NoteType
from bedrock.parser.discovery import classify_path, discover_markdown

VAULT = Path(__file__).parent / "fixtures" / "vault"


def settings() -> Settings:
    return Settings("http://qdrant:6333", VAULT.resolve(), Path("/tmp/models"), "entries", "127.0.0.1", 8080)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("logs/2026-08-20.md", NoteType.LOG),
        ("logs/nested/another.md", NoteType.LOG),
        ("kanban/Kitchen App.md", NoteType.KANBAN),
        ("projects/News Resolution.md", NoteType.STANDARD),
        ("misc/logs/not-special.md", NoteType.STANDARD),
        ("root.md", NoteType.STANDARD),
        # A note named for a day is a log wherever it sits. One configured
        # root does not describe a real vault, and a daily note the parser
        # leaves standard is a daily note it never dates.
        ("Daily/2026-08-20.md", NoteType.LOG),
        ("archive/daily notes/2026-08-20.md", NoteType.LOG),
        ("2026-08-20.md", NoteType.LOG),
        # The filename has to be a real calendar date, not merely shaped like
        # one, and not a date with anything else around it.
        ("Daily/2026-13-40.md", NoteType.STANDARD),
        ("Daily/2026-08-20 review.md", NoteType.STANDARD),
        # The Kanban root wins, so a board named for a date stays a board.
        ("kanban/2026-08-20.md", NoteType.KANBAN),
    ],
)
def test_note_type_comes_from_the_configured_roots_or_a_dated_filename(path: str, expected: NoteType) -> None:
    assert classify_path(path, settings()) is expected


def test_discovery_is_recursive_sorted_and_excludes_configured_directories() -> None:
    paths = discover_markdown(settings())
    relative = [path.relative_to(VAULT).as_posix() for path in paths]
    assert relative == sorted(relative, key=str.casefold)
    assert "messy/deep/random/Mixed Note.md" in relative
    assert ".obsidian/ignored.md" not in relative


def test_requested_paths_cannot_escape_vault() -> None:
    with pytest.raises(ValueError):
        discover_markdown(settings(), ["../outside.md"])
