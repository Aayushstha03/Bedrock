"""Atomic, deterministic full-vault rebuilds."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient, models

from zenith.core.config import Settings
from zenith.core.contracts import ParsedEntry, ParsedNote, QdrantPayload
from zenith.index.encoders import LocalEncoders
from zenith.index.links import link_key, resolve_links
from zenith.index.qdrant import alias_target
from zenith.index.schema import create_collection
from zenith.parser.markdown import frontmatter_aliases
from zenith.parser.service import PARSER_VERSION, VaultParser

ENCODER_VERSION = "fastembed-0.7.3"
TOKENIZER_VERSION = "bm25-english-stemming-stopwords-v1"
EMBEDDING_INPUT_VERSION = "3"


@dataclass(frozen=True, slots=True)
class RebuildReport:
    collection: str
    physical_collection: str
    notes: int
    points: int
    warnings: int
    previous_collection: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class IndexRebuilder:
    def __init__(
        self,
        settings: Settings,
        *,
        vault_id: str = "personal",
        client: Any | None = None,
        encoders: Any | None = None,
    ) -> None:
        self.settings = settings
        self.vault_id = vault_id
        self.client = client or QdrantClient(url=settings.qdrant_url)
        self._encoders = encoders

    @property
    def encoders(self) -> Any:
        if self._encoders is None:
            self._encoders = LocalEncoders(self.settings)
        return self._encoders

    def rebuild(self) -> RebuildReport:
        notes = resolve_links(VaultParser(self.settings, self.vault_id).parse_vault())
        entries = [entry for note in notes for entry in note.entries]
        temporary = f"{self.settings.collection_name}__build_{uuid4().hex}"
        previous = alias_target(self.client, self.settings.collection_name)

        try:
            create_collection(self.client, temporary)
            self._upsert(temporary, entries, note_alias_map(notes))
            actual = self.client.count(collection_name=temporary, exact=True).count
            if actual != len(entries):
                raise RuntimeError(
                    f"rebuild validation failed: expected {len(entries)} points, found {actual}"
                )
            self._verify_sources_unchanged(notes)
            self._activate(temporary, previous)
        except Exception:
            if self.client.collection_exists(temporary):
                self.client.delete_collection(temporary)
            raise

        # The alias now points at the new collection, so the old one is dead
        # weight. Leaving it behind kept a full copy of the vault per rebuild.
        if previous is not None and self.client.collection_exists(previous):
            self.client.delete_collection(previous)

        return RebuildReport(
            collection=self.settings.collection_name,
            physical_collection=temporary,
            notes=len(notes),
            points=len(entries),
            warnings=sum(len(note.warnings) for note in notes),
            previous_collection=previous,
        )

    def _upsert(
        self,
        collection: str,
        entries: list[ParsedEntry],
        aliases: dict[str, tuple[str, ...]],
        batch_size: int = 64,
    ) -> None:
        for start in range(0, len(entries), batch_size):
            batch = entries[start : start + batch_size]
            vectors = self.encoders.encode([entry.embedding_text for entry in batch])
            points = [
                models.PointStruct(
                    id=entry.entry_id,
                    vector=vector,
                    payload=self._payload(entry, aliases.get(entry.note_id, ())),
                )
                for entry, vector in zip(batch, vectors, strict=True)
            ]
            self.client.upsert(collection_name=collection, points=points, wait=True)

    def _payload(self, entry: ParsedEntry, note_aliases: tuple[str, ...] = ()) -> dict[str, object]:
        try:
            modified_at = datetime.fromtimestamp(
                (self.settings.vault_path / Path(entry.path)).stat().st_mtime, UTC
            ).isoformat()
        except OSError as error:
            raise RuntimeError(f"vault changed during indexing: {entry.path}") from error
        outgoing_ids = tuple(
            link.target_note_id for link in entry.outgoing_links if link.target_note_id is not None
        )
        return QdrantPayload(
            vault_id=self.vault_id,
            note_id=entry.note_id,
            entry_id=entry.entry_id,
            path=entry.path,
            note_title=entry.note_title,
            note_type=entry.note_type,
            entry_type=entry.entry_type,
            text=entry.text,
            heading=entry.heading,
            heading_path=entry.heading_path,
            start_line=entry.source.start_line,
            end_line=entry.source.end_line,
            note_date=_qdrant_date(entry.note_date),
            entry_date=_qdrant_date(entry.entry_date),
            tags=entry.tags,
            outgoing_note_ids=outgoing_ids,
            outgoing_links=entry.outgoing_links,
            web_links=entry.web_links,
            content_hash=entry.content_hash,
            modified_at=modified_at,
            embedding_fingerprint=self._embedding_fingerprint(entry.embedding_text),
            note_aliases=note_aliases,
            outgoing_link_keys=tuple(
                dict.fromkeys(link_key(link.target_text) for link in entry.outgoing_links)
            ),
            board=entry.board,
        ).to_dict()

    def _embedding_fingerprint(self, embedding_text: str) -> str:
        """Digest every input that decides whether a stored vector is still valid.

        The fixed parts lead and the free-form text goes last, so no embedding
        text can imitate a different parser or model by carrying a separator.
        """
        parts = (
            PARSER_VERSION,
            self.settings.dense_model,
            self.settings.sparse_model,
            ENCODER_VERSION,
            TOKENIZER_VERSION,
            EMBEDDING_INPUT_VERSION,
            embedding_text,
        )
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def _verify_sources_unchanged(self, notes: tuple[ParsedNote, ...]) -> None:
        for note in notes:
            source = self.settings.vault_path / Path(note.path)
            try:
                current_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            except OSError as error:
                raise RuntimeError(f"vault changed during rebuild: {note.path}") from error
            if current_hash != note.content_hash:
                raise RuntimeError(f"vault changed during rebuild: {note.path}")

    def _activate(self, temporary: str, previous: str | None) -> None:
        operations: list[object] = []
        if previous is not None:
            operations.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=self.settings.collection_name)
                )
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=temporary,
                    alias_name=self.settings.collection_name,
                )
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=operations)


def note_alias_map(notes: tuple[ParsedNote, ...]) -> dict[str, tuple[str, ...]]:
    """Aliases live on the note, but a payload is built from one entry."""
    return {note.note_id: frontmatter_aliases(note.metadata) for note in notes}


def _qdrant_date(value: str | None) -> str | None:
    return f"{value}T00:00:00Z" if value else None
