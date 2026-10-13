from pathlib import Path

from bedrock.core.config import Settings
from bedrock.core.contracts import LinkResolution, WarningType
from bedrock.index.links import resolve_link, resolve_links
from bedrock.parser.service import VaultParser

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


def settings(vault: Path) -> Settings:
    return Settings("http://unused", vault, vault / "models", "entries", "127.0.0.1", 8080)


def test_fixture_links_resolve_missing_and_ambiguous_deterministically() -> None:
    notes = resolve_links(VaultParser(settings(FIXTURE_VAULT)).parse_vault())
    reference = next(note for note in notes if note.path == "freeform/Reference.md")
    # The section spans several chunks, so collect links across all of them.
    links = [link for entry in reference.entries for link in entry.outgoing_links]

    assert [link.resolution for link in links] == [
        LinkResolution.RESOLVED,
        LinkResolution.MISSING,
        LinkResolution.AMBIGUOUS,
    ]
    assert links[0].target_note_id is not None
    assert links[0].target_heading == "2026-08-19"
    assert {warning.kind for warning in reference.warnings} >= {
        WarningType.MISSING_LINK,
        WarningType.AMBIGUOUS_LINK,
    }


def test_frontmatter_alias_and_explicit_path_resolve(tmp_path: Path) -> None:
    (tmp_path / "projects").mkdir()
    (tmp_path / "projects" / "Target.md").write_text(
        "---\naliases: [Short Name]\n---\n# Long Target\n\nBody.\n"
    )
    (tmp_path / "Source.md").write_text("# Source\n\n[[Short Name]] and [[projects/Target.md]].\n")
    notes = resolve_links(VaultParser(settings(tmp_path)).parse_vault())
    source = next(note for note in notes if note.path == "Source.md")
    links = source.entries[0].outgoing_links
    assert all(link.resolution is LinkResolution.RESOLVED for link in links)
    assert links[0].target_note_id == links[1].target_note_id


def test_public_link_resolution_accepts_wiki_heading_and_alias() -> None:
    notes = VaultParser(settings(FIXTURE_VAULT)).parse_vault()
    source = next(note for note in notes if note.path == "freeform/Reference.md")
    link = resolve_link(
        notes,
        source.note_id,
        "[[News Resolution#2026-08-19|project update]]",
    )
    assert link.resolution is LinkResolution.RESOLVED
    assert link.target_heading == "2026-08-19"
    assert link.alias == "project update"
