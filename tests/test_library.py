import re
import shutil
import warnings
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from bedrock.core.config import Settings
from bedrock.core.contracts import LinkResolution, RetrievalMode, WarningType
from bedrock.core.identity import note_id
from bedrock.index.rebuild import IndexRebuilder
from bedrock.library import Bedrock
from bedrock.parser.service import VaultParser

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


class HashEncoders:
    def encode(self, texts: list[str]):
        return [
            {"semantic": dense, "text-bm25": sparse}
            for dense, sparse in zip(self.encode_dense(texts), self.encode_sparse(texts), strict=True)
        ]

    def encode_dense(self, texts: list[str]) -> list[list[float]]:
        return [[float(hash((text, index)) % 7) for index in range(384)] for text in texts]

    def encode_sparse(self, texts: list[str]) -> list[models.SparseVector]:
        vectors = []
        for text in texts:
            indices = sorted({abs(hash(word)) % 997 for word in re.findall(r"[a-z0-9]+", text.lower())}) or [
                0
            ]
            vectors.append(models.SparseVector(indices=indices, values=[1.0] * len(indices)))
        return vectors


def indexed_api(tmp_path: Path) -> Bedrock:
    settings = Settings(
        "http://unused",
        FIXTURE_VAULT,
        tmp_path / "models",
        "entries",
        "127.0.0.1",
        8080,
    )
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    encoders = HashEncoders()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        IndexRebuilder(settings, client=client, encoders=encoders).rebuild()
    return Bedrock(settings, client=client, encoders=encoders)


def test_library_note_entry_search_and_link_operations(tmp_path: Path) -> None:
    api = indexed_api(tmp_path)
    project = api.get_note("News Resolution")
    assert project.path == "projects/News Resolution.md"
    assert api.get_note("projects/News Resolution").note_id == project.note_id
    assert api.get_entry(project.entries[0].entry_id) == project.entries[0]

    daily_id = str(note_id("personal", "logs/2026-08-20.md"))
    links = api.get_outgoing_links(daily_id)
    assert any(link.target_note_id == project.note_id for link in links)
    assert any(result.note_id == daily_id for result in api.get_backlinks(project.note_id))

    resolved = api.resolve_link(daily_id, "News Resolution#2026-08-19")
    assert resolved.resolution is LinkResolution.RESOLVED
    assert resolved.target_heading == "2026-08-19"
    assert api.resolve_link(daily_id, "Does Not Exist").resolution is LinkResolution.MISSING

    literal = api.find_entries(exact_text="fried chicken")
    assert literal and all(result.verified is True for result in literal)
    scoped = api.search_within(project.note_id, "pipeline", RetrievalMode.LEXICAL, limit=1)
    assert scoped and scoped[0].note_id == project.note_id

    with pytest.raises(ValueError, match="ambiguous note title"):
        api.get_note("Shared")
    with pytest.raises(LookupError, match="note not found"):
        api.get_note("Unknown")


def test_library_warning_and_kanban_operations(tmp_path: Path) -> None:
    api = indexed_api(tmp_path)
    warnings = api.get_index_warnings(WarningType.INVALID_KANBAN_SETTINGS)
    assert len(warnings) == 1
    assert warnings[0].path == "kanban/Inconsistent.md"

    boards = api.list_kanban_boards()
    assert {board.title for board in boards} == {"Inconsistent", "Kitchen App"}
    kitchen = api.get_kanban_board("kanban/Kitchen App.md")
    assert kitchen.columns == ("ToDo", "Doing", "complete")

    open_todo = api.find_kanban_cards(
        board="Kitchen App", columns=("ToDo",), statuses=("todo",), checked=False, limit=20
    )
    assert len(open_todo) == 6
    assert all(card.board and card.board.column == "ToDo" for card in open_todo)


def test_library_context_graph_and_reindex_surface(tmp_path: Path) -> None:
    api = indexed_api(tmp_path)
    daily = api.get_note("logs/2026-08-20.md")
    work = next(entry for entry in daily.entries if entry.heading == "Work")
    expansion = api.expand_context(work.entry_id)
    assert len(expansion.inspected_note_ids) <= 5
    assert len(expansion.items) > 1

    first = api.export_graph().to_dict()
    assert first == api.export_graph().to_dict()
    report = api.reindex()
    assert report.inserted == report.updated == report.deleted == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rebuilt = api.reindex(full=True)
    assert rebuilt.notes == 18
    # 44, not 45. `Daily/2026-08-20.md` is a log now that a dated filename
    # makes one, so its leading `2026-08-20` line is the note's date and no
    # longer parses into a headless entry of its own.
    assert rebuilt.points == 44
    with pytest.raises(ValueError, match="does not accept paths"):
        api.reindex(["Note.md"], full=True)


def test_library_parses_the_vault_once_until_a_note_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, vault)
    settings = Settings("http://unused", vault, tmp_path / "models", "entries", "127.0.0.1", 8080)
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    encoders = HashEncoders()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        IndexRebuilder(settings, client=client, encoders=encoders).rebuild()
    api = Bedrock(settings, client=client, encoders=encoders)

    parses: list[object] = []
    original = VaultParser.parse_vault

    def counting(self: VaultParser, requested: list[str] | None = None):
        parses.append(requested)
        return original(self, requested)

    monkeypatch.setattr(VaultParser, "parse_vault", counting)

    project = api.get_note("News Resolution")
    api.get_note("News Resolution")
    api.get_index_warnings()
    api.get_index_warnings()
    api.resolve_link(project.note_id, "Kitchen App")
    assert len(parses) == 1

    note = vault / "projects" / "News Resolution.md"
    note.write_text(
        note.read_text(encoding="utf-8") + "\n## Added\n\nNew prose.\n",
        encoding="utf-8",
    )
    api.get_note("News Resolution")
    assert len(parses) == 2
    api.get_index_warnings()
    assert len(parses) == 2


def test_library_reads_one_complete_note_from_the_vault(tmp_path: Path) -> None:
    api = indexed_api(tmp_path)
    note = api.read_note("News Resolution")
    assert note.path == "projects/News Resolution.md"
    assert note.note_id == api.get_note("News Resolution").note_id
    on_disk = (FIXTURE_VAULT / "projects" / "News Resolution.md").read_text(encoding="utf-8")
    assert note.content == on_disk

    # The file itself, not the indexed prose: entry text is cleaned and split.
    entries = api.get_note("News Resolution").entries
    assert len(note.content) > max(len(entry.text) for entry in entries)

    assert api.read_note("projects/News Resolution.md").content == on_disk
    assert api.read_note("projects/News Resolution").content == on_disk

    with pytest.raises(ValueError, match="ambiguous note title"):
        api.read_note("Shared")
    with pytest.raises(LookupError, match="note not found"):
        api.read_note("Unknown")
    with pytest.raises(LookupError, match="note not found"):
        api.read_note("../../../etc/passwd")
