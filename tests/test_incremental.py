from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import models

from bedrock.core.config import Settings
from bedrock.index.incremental import IncrementalIndexer
from bedrock.index.links import resolve_links
from bedrock.index.rebuild import note_alias_map
from bedrock.parser.service import VaultParser


class TrackingEncoders:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, texts: list[str]):
        self.texts.extend(texts)
        return [
            {
                "semantic": [float(len(text))] * 384,
                "text-bm25": models.SparseVector(indices=[1], values=[1.0]),
            }
            for text in texts
        ]


def _matches(payload: dict, condition) -> bool:
    """Model the only Qdrant filter shapes the indexer builds."""
    stored = payload.get(condition.key)
    match = condition.match
    wanted = {match.value} if isinstance(match, models.MatchValue) else set(match.any)
    if isinstance(stored, (list, tuple)):
        return bool(set(stored) & wanted)
    return stored in wanted


class StateClient:
    def __init__(self) -> None:
        self.records: dict[str, SimpleNamespace] = {}
        self.upsert_calls = 0
        self.deleted: list[str] = []
        self.fail_delete_once = False
        self.scrolled_points = 0
        self.retrieved_points = 0

    def scroll(self, *, scroll_filter=None, **_: object):
        records = list(self.records.values())
        if scroll_filter is not None:
            records = [
                record
                for record in records
                if all(_matches(record.payload, condition) for condition in scroll_filter.must)
            ]
        self.scrolled_points += len(records)
        return records, None

    def retrieve(self, *, ids: list[str], **_: object):
        found = [self.records[str(point_id)] for point_id in ids if str(point_id) in self.records]
        self.retrieved_points += len(found)
        return found

    def upsert(self, *, points: list[object], **_: object) -> None:
        self.upsert_calls += 1
        for point in points:
            self.records[str(point.id)] = SimpleNamespace(
                id=point.id,
                payload=point.payload,
                vector=point.vector,
            )

    def delete(self, *, points_selector: list[str], **_: object) -> None:
        if self.fail_delete_once:
            self.fail_delete_once = False
            raise RuntimeError("interrupted delete")
        self.deleted.extend(points_selector)
        for point_id in points_selector:
            self.records.pop(point_id)


def settings(vault: Path) -> Settings:
    return Settings("http://unused", vault, vault / "models", "entries", "127.0.0.1", 8080)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def assert_matches_full_rebuild(client: StateClient, indexer: IncrementalIndexer, vault: Path) -> None:
    """The whole point of a scoped update: it must land where a rebuild would."""
    notes = resolve_links(VaultParser(settings(vault)).parse_vault())
    aliases = note_alias_map(notes)
    expected = {
        entry.entry_id: indexer._payload(entry, aliases.get(entry.note_id, ()))
        for note in notes
        for entry in note.entries
    }
    assert {point_id: record.payload for point_id, record in client.records.items()} == expected


def test_unchanged_run_skips_writes_and_embeddings(tmp_path: Path) -> None:
    write(tmp_path / "Note.md", "# Note\n\n## Section\nhello\n")
    client = StateClient()
    encoders = TrackingEncoders()
    indexer = IncrementalIndexer(settings(tmp_path), client=client, encoders=encoders)

    first = indexer.reindex()
    calls_after_first = client.upsert_calls
    encoded_after_first = len(encoders.texts)
    second = indexer.reindex()

    assert first.inserted == first.embeddings_generated == 1
    assert second.skipped == 1
    assert second.updated == second.inserted == second.deleted == 0
    assert second.embeddings_generated == 0
    assert client.upsert_calls == calls_after_first
    assert len(encoders.texts) == encoded_after_first

    write(tmp_path / "Note.md", "# Note\n\n## Section\ngoodbye\n")
    changed = indexer.reindex(["Note.md"])
    assert changed.updated == changed.embeddings_generated == 1
    assert changed.inserted == changed.deleted == 0


def test_change_add_and_delete_converge_without_reencoding_unchanged_inputs(tmp_path: Path) -> None:
    source_path = tmp_path / "Source.md"
    first_target = tmp_path / "one" / "Target.md"
    second_target = tmp_path / "two" / "Target.md"
    write(source_path, "# Source\n\nSee [[Target]].\n")
    write(first_target, "# First Target\n\nBody one.\n")
    client = StateClient()
    encoders = TrackingEncoders()
    indexer = IncrementalIndexer(settings(tmp_path), client=client, encoders=encoders)
    initial = indexer.reindex()
    assert initial.embeddings_generated == 2

    write(second_target, "# Second Target\n\nBody two.\n")
    added = indexer.reindex(["two/Target.md"])
    assert added.inserted == 1
    assert added.embeddings_generated == 1
    assert added.embeddings_reused == 1
    source = next(record for record in client.records.values() if record.payload["path"] == "Source.md")
    assert source.payload["outgoing_links"][0]["resolution"] == "ambiguous"

    second_target.unlink()
    deleted = indexer.reindex(["two/Target.md"])
    assert deleted.deleted == 1
    assert deleted.embeddings_generated == 0
    assert deleted.embeddings_reused == 1
    source = next(record for record in client.records.values() if record.payload["path"] == "Source.md")
    assert source.payload["outgoing_links"][0]["resolution"] == "resolved"

    renamed_target = tmp_path / "one" / "Renamed.md"
    first_target.rename(renamed_target)
    renamed = indexer.reindex(["one/Target.md", "one/Renamed.md"])
    assert renamed.inserted == renamed.deleted == 1
    # The path changed, so every entry id changed, but the title did not, so
    # the embedding text and its fingerprint did not either. The vector moves
    # across to the new id and the encoder is never called.
    assert renamed.embeddings_generated == 0
    assert renamed.embeddings_reused == 2
    source = next(record for record in client.records.values() if record.payload["path"] == "Source.md")
    assert source.payload["outgoing_links"][0]["resolution"] == "missing"

    assert_matches_full_rebuild(client, indexer, tmp_path)


def test_interrupted_update_converges_when_rerun(tmp_path: Path) -> None:
    first = tmp_path / "First.md"
    second = tmp_path / "Second.md"
    write(first, "# First\n\nOne.\n")
    write(second, "# Second\n\nTwo.\n")
    client = StateClient()
    indexer = IncrementalIndexer(settings(tmp_path), client=client, encoders=TrackingEncoders())
    assert indexer.reindex().inserted == 2

    second.unlink()
    client.fail_delete_once = True
    with pytest.raises(RuntimeError, match="interrupted"):
        indexer.reindex(["Second.md"])

    recovered = indexer.reindex(["Second.md"])
    assert recovered.deleted == 1
    assert {record.payload["path"] for record in client.records.values()} == {"First.md"}


def indexer_for(vault: Path) -> tuple[StateClient, IncrementalIndexer]:
    client = StateClient()
    return client, IncrementalIndexer(settings(vault), client=client, encoders=TrackingEncoders())


def link_of(client: StateClient, path: str) -> dict:
    record = next(r for r in client.records.values() if r.payload["path"] == path)
    return record.payload["outgoing_links"][0]


def test_rename_without_rewritten_links_still_breaks_the_links(tmp_path: Path) -> None:
    """A `mv`, or Obsidian with link updating off, never touches the referrer."""
    write(tmp_path / "news_resolution.md", "Body of the resolution note.\n")
    write(tmp_path / "OtherA.md", "# Other A\n\nSee [[news_resolution]].\n")
    client, indexer = indexer_for(tmp_path)
    indexer.reindex()
    assert link_of(client, "OtherA.md")["resolution"] == "resolved"

    (tmp_path / "news_resolution.md").rename(tmp_path / "incident_resolution.md")
    report = indexer.reindex(["news_resolution.md", "incident_resolution.md"])

    # OtherA.md was never written to, so only the reverse queries can find it.
    assert link_of(client, "OtherA.md")["resolution"] == "missing"
    assert report.deleted == 1
    assert_matches_full_rebuild(client, indexer, tmp_path)


def test_rename_of_a_titled_note_reuses_every_vector(tmp_path: Path) -> None:
    write(tmp_path / "news_resolution.md", "# News Resolution\n\n## Detail\nbody\n")
    client, indexer = indexer_for(tmp_path)
    indexer.reindex()

    (tmp_path / "news_resolution.md").rename(tmp_path / "incident_resolution.md")
    report = indexer.reindex(["news_resolution.md", "incident_resolution.md"])

    # The title comes from the heading, so the embedding text never changed.
    assert report.embeddings_generated == 0
    assert report.embeddings_reused == report.inserted == report.deleted == 1
    assert_matches_full_rebuild(client, indexer, tmp_path)


def test_deleting_one_twin_makes_an_ambiguous_link_resolve(tmp_path: Path) -> None:
    """The deleted note is the only place its old names can be read from."""
    write(tmp_path / "one" / "Target.md", "Body one.\n")
    write(tmp_path / "two" / "Target.md", "Body two.\n")
    write(tmp_path / "Source.md", "# Source\n\nSee [[Target]].\n")
    client, indexer = indexer_for(tmp_path)
    indexer.reindex()
    assert link_of(client, "Source.md")["resolution"] == "ambiguous"

    (tmp_path / "two" / "Target.md").unlink()
    indexer.reindex(["two/Target.md"])

    assert link_of(client, "Source.md")["resolution"] == "resolved"
    assert_matches_full_rebuild(client, indexer, tmp_path)


def test_creating_a_note_resolves_the_links_that_were_waiting_for_it(tmp_path: Path) -> None:
    write(tmp_path / "Source.md", "# Source\n\nSee [[Target]].\n")
    client, indexer = indexer_for(tmp_path)
    indexer.reindex()
    assert link_of(client, "Source.md")["resolution"] == "missing"

    write(tmp_path / "Target.md", "# Target\n\nBody.\n")
    indexer.reindex(["Target.md"])

    assert link_of(client, "Source.md")["resolution"] == "resolved"
    assert_matches_full_rebuild(client, indexer, tmp_path)


def test_removing_an_alias_breaks_the_links_that_used_it(tmp_path: Path) -> None:
    """Old aliases are readable only because the payload stores them."""
    write(tmp_path / "Target.md", "---\naliases: [nickname]\n---\n\n# Target\n\nBody.\n")
    write(tmp_path / "Source.md", "# Source\n\nSee [[nickname]].\n")
    client, indexer = indexer_for(tmp_path)
    indexer.reindex()
    assert link_of(client, "Source.md")["resolution"] == "resolved"

    write(tmp_path / "Target.md", "# Target\n\nBody.\n")
    indexer.reindex(["Target.md"])

    assert link_of(client, "Source.md")["resolution"] == "missing"
    assert_matches_full_rebuild(client, indexer, tmp_path)


def test_a_scoped_update_leaves_unrelated_notes_alone(tmp_path: Path) -> None:
    for index in range(12):
        write(tmp_path / f"Note{index}.md", f"# Note {index}\n\n## Body\nText {index}.\n")
    client, indexer = indexer_for(tmp_path)
    indexer.reindex()
    before = {point_id: record.payload for point_id, record in client.records.items()}

    write(tmp_path / "Note3.md", "# Note 3\n\n## Body\nRewritten.\n")
    client.scrolled_points = 0
    report = indexer.reindex(["Note3.md"])

    assert report.deleted == 0
    assert report.updated == 1
    # Twelve notes are indexed; a scoped run must not read all of their points.
    assert client.scrolled_points < len(before)
    changed = {k for k, v in before.items() if client.records[k].payload != v}
    assert {client.records[point_id].payload["path"] for point_id in changed} == {"Note3.md"}
    assert_matches_full_rebuild(client, indexer, tmp_path)


def test_deleting_an_unlinked_note_parses_nothing_and_encodes_nothing(tmp_path: Path) -> None:
    """An empty scope must not read as "no filter" and reparse the vault."""
    for index in range(6):
        write(tmp_path / f"Note{index}.md", f"# Note {index}\n\n## Body\nText {index}.\n")
    client, indexer = indexer_for(tmp_path)
    indexer.reindex()

    (tmp_path / "Note4.md").unlink()
    report = indexer.reindex(["Note4.md"])

    assert report.notes == 0
    assert report.inserted == report.updated == report.embeddings_generated == 0
    assert report.deleted == 1
    assert_matches_full_rebuild(client, indexer, tmp_path)
