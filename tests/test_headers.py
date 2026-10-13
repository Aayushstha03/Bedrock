"""The cheap header must agree with the full parser, or scoped updates drift."""

from pathlib import Path

import pytest

from bedrock.core.config import Settings
from bedrock.parser.headers import read_header, read_headers
from bedrock.parser.markdown import frontmatter_aliases
from bedrock.parser.service import VaultParser

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


def settings(vault: Path) -> Settings:
    return Settings("http://unused", vault, vault / "models", "entries", "127.0.0.1", 8080)


def test_header_agrees_with_the_full_parser_across_the_fixture_vault() -> None:
    parsed = VaultParser(settings(FIXTURE_VAULT)).parse_vault()
    headers = {header.path: header for header in read_headers(settings(FIXTURE_VAULT))}

    assert set(headers) == {note.path for note in parsed}
    for note in parsed:
        header = headers[note.path]
        assert header.note_id == note.note_id
        assert header.title == note.title
        assert header.aliases == frontmatter_aliases(note.metadata)


@pytest.mark.parametrize(
    "content",
    [
        "# Plain\n\nbody\n",
        "# Closed Hashes ###\n\nbody\n",
        "   # Indented Three\n\nbody\n",
        "Setext Title\n============\n\nbody\n",
        "```\n# Not A Heading\n```\n\n# Real Heading\n\nbody\n",
        "~~~\n# Fenced Out\n~~~\n\n# After Fence\n\nbody\n",
        "---\ntitle: From Frontmatter\n---\n\n# Ignored\n",
        "---\ntitle: '   '\n---\n\n# Blank Title Falls Through\n",
        "---\naliases: [one, two]\n---\n\n# Aliased\n",
        "no heading at all, so the filename wins\n",
        "## Only Level Two\n\nbody\n",
        "#NoSpace is not a heading\n\n# Actual\n",
    ],
)
def test_header_title_matches_the_parser_on_awkward_documents(tmp_path: Path, content: str) -> None:
    note = tmp_path / "Fallback Name.md"
    note.write_text(content)
    parsed = VaultParser(settings(tmp_path)).parse_file(note)
    header = read_header(note, tmp_path.resolve())

    assert header.title == parsed.title
    assert header.aliases == frontmatter_aliases(parsed.metadata)
