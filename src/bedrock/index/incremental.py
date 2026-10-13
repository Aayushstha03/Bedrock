"""Idempotent incremental indexing that converges to a clean rebuild."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from qdrant_client import models

from bedrock.core.contracts import NoteHeader, ParsedEntry, ParsedNote
from bedrock.core.identity import normalize_vault_path, note_id
from bedrock.index.links import build_catalog, note_names, resolve_links, resolve_note
from bedrock.index.qdrant import scroll_all, vault_condition
from bedrock.index.rebuild import IndexRebuilder, note_alias_map
from bedrock.index.schema import DENSE_VECTOR, SPARSE_VECTOR
from bedrock.parser.headers import read_headers
from bedrock.parser.service import VaultParser


@dataclass(frozen=True, slots=True)
class IncrementalReport:
    collection: str
    requested_paths: tuple[str, ...]
    notes: int
    inserted: int
    updated: int
    deleted: int
    skipped: int
    embeddings_generated: int
    embeddings_reused: int
    warnings: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _VectorPlan:
    """Which entries need encoding, which reuse a stored vector, and the tally."""

    encode: list[ParsedEntry]
    reused: dict[str, object]
    inserted: int
    updated: int
    skipped: int


class IncrementalIndexer(IndexRebuilder):
    def reindex(self, paths: list[str] | None = None) -> IncrementalReport:
        """Converge the index with the vault.

        Without `paths` this reparses and rechecks the whole vault, which is
        what a manual update and the watcher's startup pass want. With `paths`
        it narrows the work to the notes that changed and the notes whose links
        point at them, which is what a save wants.
        """
        requested = tuple(sorted({normalize_vault_path(path) for path in paths or ()}, key=str.casefold))
        parser = VaultParser(self.settings, self.vault_id)
        if requested:
            notes, scope = self._scoped_notes(parser, requested)
        else:
            notes, scope = resolve_links(parser.parse_vault()), None

        entries = [entry for note in notes for entry in note.entries]
        aliases = note_alias_map(notes)
        existing = self._existing_points(scope)
        stale_ids = sorted(set(existing) - {entry.entry_id for entry in entries})

        payloads = {entry.entry_id: self._payload(entry, aliases.get(entry.note_id, ())) for entry in entries}
        plan = self._plan(entries, payloads, existing)

        generated = self.encoders.encode([entry.embedding_text for entry in plan.encode])
        vectors: dict[str, object] = dict(plan.reused)
        vectors.update((entry.entry_id, vector) for entry, vector in zip(plan.encode, generated, strict=True))
        points = [
            models.PointStruct(
                id=entry.entry_id,
                vector=vectors[entry.entry_id],
                payload=payloads[entry.entry_id],
            )
            for entry in entries
            if entry.entry_id in vectors
        ]

        self._verify_sources_unchanged(notes)
        for start in range(0, len(points), 64):
            self.client.upsert(
                collection_name=self.settings.collection_name,
                points=points[start : start + 64],
                wait=True,
            )
        if stale_ids:
            self.client.delete(
                collection_name=self.settings.collection_name,
                points_selector=stale_ids,
                wait=True,
            )

        return IncrementalReport(
            collection=self.settings.collection_name,
            requested_paths=requested,
            notes=len(notes),
            inserted=plan.inserted,
            updated=plan.updated,
            deleted=len(stale_ids),
            skipped=plan.skipped,
            embeddings_generated=len(plan.encode),
            embeddings_reused=len(plan.reused),
            warnings=sum(len(note.warnings) for note in notes),
        )

    def _plan(
        self,
        entries: list[ParsedEntry],
        payloads: dict[str, dict[str, object]],
        existing: dict[str, Any],
    ) -> _VectorPlan:
        """Decide which entries need a fresh vector and which can borrow one."""
        # A vector belongs to an embedding fingerprint, not to an entry id. A
        # rename gives every entry a new id but leaves the fingerprint alone,
        # so looking the vector up this way makes a rename cost no encoding.
        sources = {
            record.payload.get("embedding_fingerprint"): str(record.id) for record in existing.values()
        }
        encode: list[ParsedEntry] = []
        borrow: dict[str, str] = {}
        inserted = updated = skipped = 0
        for entry in entries:
            record = existing.get(entry.entry_id)
            payload = payloads[entry.entry_id]
            if record is not None and record.payload == payload:
                skipped += 1
                continue
            if record is None:
                inserted += 1
            else:
                updated += 1
            source = sources.get(payload["embedding_fingerprint"])
            if source is None:
                encode.append(entry)
            else:
                borrow[entry.entry_id] = source

        by_id = {entry.entry_id: entry for entry in entries}
        stored = self._vectors(set(borrow.values()))
        reused: dict[str, object] = {}
        for entry_id, source in borrow.items():
            vector = stored.get(source)
            if _has_vectors(vector):
                reused[entry_id] = vector
            else:
                encode.append(by_id[entry_id])
        return _VectorPlan(encode, reused, inserted, updated, skipped)

    def _scoped_notes(
        self, parser: VaultParser, requested: tuple[str, ...]
    ) -> tuple[tuple[ParsedNote, ...], set[str]]:
        """Parse the changed notes, plus every note whose links they move.

        The changed ids come from the paths and not from the parse, so a note
        that was deleted is still in scope and its points are still swept.
        """
        changed = {str(note_id(self.vault_id, path)) for path in requested}
        headers = read_headers(self.settings, self.vault_id)
        catalog = build_catalog(headers)

        names: set[str] = set()
        for header in headers:
            if header.note_id in changed:
                names |= note_names(header)
        for header in self._stored_headers(changed):
            names |= note_names(header)

        scope = changed | self._linking_notes(names, changed)
        in_scope = [header.path for header in headers if header.note_id in scope]
        # Every note in scope was deleted, so there is nothing left to parse.
        # An empty request would otherwise read as "no filter" and parse the
        # whole vault, against points scoped to the deletion.
        parsed = parser.parse_vault(in_scope) if in_scope else ()
        return tuple(resolve_note(note, catalog) for note in parsed), scope

    def _stored_headers(self, note_ids: set[str]) -> tuple[NoteHeader, ...]:
        """Read the names the changed notes had before this change."""
        seen: dict[str, NoteHeader] = {}
        for record in self._scroll(_match_any("note_id", note_ids)):
            payload = record.payload
            stored = str(payload.get("note_id"))
            if stored not in seen:
                seen[stored] = NoteHeader(
                    note_id=stored,
                    path=str(payload.get("path", "")),
                    title=str(payload.get("note_title", "")),
                    aliases=tuple(payload.get("note_aliases") or ()),
                )
        return tuple(seen.values())

    def _linking_notes(self, names: set[str], changed: set[str]) -> set[str]:
        """Find the notes whose links point at, or used to point at, a change."""
        found: set[str] = set()
        for condition in (_match_any("outgoing_link_keys", names), _match_any("outgoing_note_ids", changed)):
            if condition is None:
                continue
            for record in self._scroll(condition):
                found.add(str(record.payload.get("note_id")))
        return found

    def _existing_points(self, note_ids: set[str] | None = None) -> dict[str, Any]:
        """Read the points in scope. Payloads only: most vectors go unused."""
        if note_ids is not None and not note_ids:
            return {}
        condition = None if note_ids is None else _match_any("note_id", note_ids)
        return {str(record.id): record for record in self._scroll(condition)}

    def _scroll(self, condition: models.FieldCondition | None) -> list[Any]:
        must: list[Any] = [vault_condition(self.vault_id)]
        if condition is not None:
            must.append(condition)
        return scroll_all(self.client, self.settings.collection_name, models.Filter(must=must))

    def _vectors(self, point_ids: set[str]) -> dict[str, object]:
        if not point_ids:
            return {}
        records = self.client.retrieve(
            collection_name=self.settings.collection_name,
            ids=sorted(point_ids),
            with_payload=False,
            with_vectors=True,
        )
        return {str(record.id): record.vector for record in records}


def _match_any(key: str, values: set[str]) -> models.FieldCondition | None:
    if not values:
        return None
    return models.FieldCondition(key=key, match=models.MatchAny(any=sorted(values)))


def _has_vectors(vector: object) -> bool:
    return isinstance(vector, dict) and DENSE_VECTOR in vector and SPARSE_VECTOR in vector
