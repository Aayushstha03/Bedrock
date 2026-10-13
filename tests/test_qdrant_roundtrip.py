import warnings
from pathlib import Path

from qdrant_client import QdrantClient, models

from bedrock.core.config import Settings
from bedrock.index.links import resolve_links
from bedrock.index.rebuild import IndexRebuilder, note_alias_map
from bedrock.index.schema import create_collection
from bedrock.parser.service import VaultParser

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


class Encoders:
    def encode(self, texts: list[str]):
        return [
            {
                "semantic": [float(index)] * 384,
                "text-bm25": models.SparseVector(indices=[index + 1], values=[1.0]),
            }
            for index, _ in enumerate(texts)
        ]


def test_all_payloads_and_vectors_round_trip_through_qdrant(tmp_path: Path) -> None:
    settings = Settings(
        "http://unused",
        FIXTURE_VAULT,
        tmp_path / "models",
        "entries",
        "127.0.0.1",
        8080,
    )
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        create_collection(client, "entries")
    indexer = IndexRebuilder(settings, client=client, encoders=Encoders())
    notes = resolve_links(VaultParser(settings).parse_vault())
    entries = [entry for note in notes for entry in note.entries]
    indexer._upsert("entries", entries, note_alias_map(notes))

    records, next_offset = client.scroll("entries", limit=100, with_payload=True, with_vectors=True)
    assert next_offset is None
    expected = {entry.entry_id: indexer._payload(entry) for entry in entries}
    assert {str(record.id): record.payload for record in records} == expected
    assert all(len(record.vector["semantic"]) == 384 for record in records)
    assert all(record.vector["text-bm25"].indices for record in records)

    daily = next(
        record
        for record in records
        if record.payload["path"] == "logs/2026-08-20.md" and record.payload["heading"] == "Thoughts"
    )
    assert daily.payload["note_date"] == "2026-08-20T00:00:00Z"
    assert daily.payload["entry_date"] is None
    assert daily.payload["tags"] == ["journal"]
