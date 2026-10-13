"""Source-backed JSON graph snapshots, independent of index readiness."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import PurePosixPath
from threading import Lock
from typing import Any

from bedrock.core.config import Settings
from bedrock.core.contracts import EntryType, LinkResolution, ParsedEntry, SourceRange
from bedrock.index.links import link_key, resolve_links
from bedrock.parser.markdown import markdown_tokens, parse_frontmatter, prose_between
from bedrock.parser.service import VaultParser

TAG = re.compile(r"(?<![\w/#])#([\w-]+(?:/[\w-]+)*)", re.UNICODE)


class GraphSnapshotCache:
    """Atomically publish a snapshot only at startup and watcher callbacks."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._snapshot: dict[str, Any] | None = None
        self._cosmograph: dict[str, Any] | None = None
        self._error: Exception | None = None
        self._lock = Lock()

    def rebuild(self, _paths: tuple[str, ...] = ()) -> None:
        # Every change can affect vault-wide resolution, including aliases.
        with self._lock:
            try:
                snapshot = graph_snapshot(self.settings)
                cosmograph = cosmograph_data(snapshot)
            except Exception as error:
                self._error = error
                raise
            self._snapshot = snapshot
            self._cosmograph = cosmograph
            self._error = None

    def get(self) -> dict[str, Any]:
        with self._lock:
            if self._error is not None:
                raise self._error
            if self._snapshot is None:
                self._snapshot = graph_snapshot(self.settings)
                self._cosmograph = cosmograph_data(self._snapshot)
            return self._snapshot

    def get_cosmograph(self) -> dict[str, Any]:
        self.get()
        with self._lock:
            if self._error is not None:
                raise self._error
            assert self._cosmograph is not None
            return self._cosmograph


def cosmograph_data(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Flat points and links tables for Cosmograph's JSON data importer."""
    degree = dict.fromkeys((node["id"] for node in snapshot["nodes"]), 0)
    links = []
    for edge in snapshot["edges"]:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1
        links.append(
            {
                "id": edge["id"],
                "source": edge["source"],
                "target": edge["target"],
                "type": edge["type"],
                "weight": len(edge.get("occurrences", [])) or 1,
            }
        )
    points = [
        {
            "id": node["id"],
            "label": node["label"],
            "type": node["type"],
            "note_id": node.get("note_id", ""),
            "path": node.get("path", ""),
            "folder": node.get("folder", ""),
            "note_type": node.get("note_type", ""),
            "tags": ", ".join(node.get("tags", [])),
            "degree": degree[node["id"]],
        }
        for node in snapshot["nodes"]
    ]
    return {"points": points, "links": links}


def graph_snapshot(settings: Settings, *, vault_id: str = "personal") -> dict[str, Any]:
    """Return all discovered Markdown notes, tags, and wikilink relationships.

    IDs and ordering are deterministic. Repeated links collapse into one
    directed edge with occurrence evidence; ambiguous links remain diagnostic.
    """
    if not settings.vault_path.is_dir():
        raise FileNotFoundError("configured vault directory is unavailable")
    notes = VaultParser(settings, vault_id).parse_vault()
    graph_notes = []
    tags_by_note: dict[str, list[str]] = {}
    for note in notes:
        content = (settings.vault_path / note.path).read_text(encoding="utf-8")
        frontmatter = parse_frontmatter(content, note.path)
        tokens = markdown_tokens(content)
        # Collect tags from prose only, excluding code, URLs and link destinations.
        tags = set()
        for token in tokens:
            if token.type != "inline" or token.map is None or token.map[0] < frontmatter.body_start:
                continue
            text = "".join(child.content if child.type == "text" else " " for child in token.children or ())
            tags.update(match.group(1).casefold() for match in TAG.finditer(text))
        raw_tags = frontmatter.values.get("tags", [])
        if isinstance(raw_tags, str):
            raw_tags = re.split(r"[\s,]+", raw_tags)
        if isinstance(raw_tags, list):
            tags.update(
                tag.strip().removeprefix("#").casefold()
                for tag in raw_tags
                if isinstance(tag, str) and tag.strip().removeprefix("#")
            )
        tags_by_note[note.note_id] = sorted(tags)
        prose = prose_between(tokens, frontmatter.body_start, len(content.splitlines()), note.path, (), ())
        # One full-document entry keeps links in headings and otherwise empty
        # notes, without inheriting indexing chunks or retrieval budgets.
        entry = ParsedEntry(
            entry_id=note.note_id,
            note_id=note.note_id,
            path=note.path,
            note_title=note.title,
            note_type=note.note_type,
            entry_type=EntryType.FREEFORM_SECTION,
            text="",
            embedding_text="",
            heading=None,
            heading_path=(),
            source=SourceRange(1, max(1, len(content.splitlines()))),
            outgoing_links=prose.links,
        )
        graph_notes.append(replace(note, entries=(entry,)))

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    unresolved = []

    def edge(source: str, target: str, kind: str) -> dict[str, Any]:
        key = (source, target, kind)
        if key not in edges:
            material = "\x1f".join(key)
            edges[key] = {
                "id": hashlib.sha256(material.encode()).hexdigest(),
                "source": source,
                "target": target,
                "type": kind,
            }
        return edges[key]

    for note in resolve_links(tuple(graph_notes)):
        note_node = f"note:{note.note_id}"
        tags = tags_by_note[note.note_id]
        nodes[note_node] = {
            "id": note_node,
            "type": "note",
            "label": note.title,
            "note_id": note.note_id,
            "path": note.path,
            "folder": str(PurePosixPath(note.path).parent),
            "note_type": note.note_type.value,
            "tags": tags,
        }
        for tag in tags:
            tag_node = f"tag:{tag}"
            nodes[tag_node] = {"id": tag_node, "type": "tag", "label": f"#{tag}"}
            edge(note_node, tag_node, "tag")
        for link in note.entries[0].outgoing_links:
            if link.resolution is LinkResolution.AMBIGUOUS:
                unresolved.append(
                    {
                        "source": note_node,
                        "target": link.target_text,
                        "line": link.line,
                        "resolution": "ambiguous",
                    }
                )
                continue
            if link.resolution is LinkResolution.RESOLVED:
                target = f"note:{link.target_note_id}"
            else:
                target = f"missing:{link_key(link.target_text)}"
                nodes.setdefault(target, {"id": target, "type": "missing", "label": link.target_text})
            relationship = edge(note_node, target, "link")
            relationship.setdefault("occurrences", []).append(
                {"line": link.line, "heading": link.target_heading, "alias": link.alias}
            )

    return {
        "schema_version": 1,
        "vault_id": vault_id,
        "nodes": sorted(nodes.values(), key=lambda node: node["id"]),
        "edges": sorted(edges.values(), key=lambda edge: (edge["type"], edge["source"], edge["target"])),
        "unresolved_links": unresolved,
    }
