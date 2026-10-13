import re
import warnings
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from zenith.core.config import Settings
from zenith.core.contracts import EntryType, LinkResolution, QueryPlan, RetrievalMode
from zenith.core.identity import note_id as compute_note_id
from zenith.index.rebuild import IndexRebuilder
from zenith.index.schema import create_collection
from zenith.retrieval.literal import verify_literal
from zenith.retrieval.service import Retriever

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"
DENSE_DIMS = 384


class HashEncoders:
    """Deterministic but content-meaningless vectors, for populating a real vault."""

    def encode(self, texts: list[str]):
        return [
            {"semantic": dense, "text-bm25": sparse}
            for dense, sparse in zip(self.encode_dense(texts), self.encode_sparse(texts), strict=True)
        ]

    def encode_dense(self, texts: list[str]) -> list[list[float]]:
        return [[float(hash((text, i)) % 7) for i in range(DENSE_DIMS)] for text in texts]

    def encode_sparse(self, texts: list[str]) -> list[models.SparseVector]:
        vectors = []
        for text in texts:
            words = re.findall(r"[a-z0-9]+", text.lower())
            indices = sorted({abs(hash(word)) % 997 for word in words}) or [0]
            vectors.append(models.SparseVector(indices=indices, values=[1.0] * len(indices)))
        return vectors


class QueryEncoders:
    """Query-only encoders returning caller-controlled vectors by exact text lookup."""

    def __init__(self, dense: dict[str, list[float]], sparse: dict[str, models.SparseVector]) -> None:
        self._dense = dense
        self._sparse = sparse

    def encode_dense(self, texts: list[str]) -> list[list[float]]:
        return [self._dense[text] for text in texts]

    def encode_sparse(self, texts: list[str]) -> list[models.SparseVector]:
        return [self._sparse[text] for text in texts]


def make_settings(vault: Path, tmp_path: Path) -> Settings:
    return Settings("http://unused", vault, tmp_path / "models", "entries", "127.0.0.1", 8080)


def real_client(tmp_path: Path) -> QdrantClient:
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        create_collection(client, "entries")
    return client


def index_fixture_vault(tmp_path: Path) -> tuple[QdrantClient, Settings]:
    settings = make_settings(FIXTURE_VAULT, tmp_path)
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    IndexRebuilder(settings, client=client, encoders=HashEncoders()).rebuild()
    return client, settings


def base_payload(**overrides: object) -> dict[str, object]:
    payload = {
        "vault_id": "personal",
        "note_id": "note-1",
        "entry_id": "e1",
        "path": "note.md",
        "note_title": "Note",
        "note_type": "standard",
        "entry_type": "freeform_section",
        "text": "some body text",
        "heading": "Section",
        "heading_path": ["Section"],
        "start_line": 1,
        "end_line": 2,
        "note_date": None,
        "entry_date": None,
        "tags": [],
        "outgoing_note_ids": [],
        "outgoing_links": [],
        "web_links": [],
        "content_hash": "x",
        "modified_at": "2026-01-01T00:00:00Z",
        "embedding_fingerprint": "",
        "board": None,
    }
    payload.update(overrides)
    return payload


def upsert_point(
    client: QdrantClient,
    point_id: str,
    payload: dict[str, object],
    *,
    dense: list[float] | None = None,
    sparse: models.SparseVector | None = None,
) -> None:
    client.upsert(
        "entries",
        points=[
            models.PointStruct(
                id=point_id,
                vector={
                    "semantic": dense or [0.0] * DENSE_DIMS,
                    "text-bm25": sparse or models.SparseVector(indices=[1], values=[0.01]),
                },
                payload=payload,
            )
        ],
    )


# --- literal.verify_literal (pure function) ---------------------------------


def test_verify_literal_ignores_case_but_requires_contiguous_phrase() -> None:
    assert verify_literal("fried chicken", "made fried chicken today")
    assert verify_literal("Fried Chicken", "made fried chicken today")
    assert verify_literal("fried chicken", "MADE FRIED CHICKEN TODAY")
    assert not verify_literal("fried chicken", "chicken was fried well")


def test_verify_literal_normalizes_unicode_forms() -> None:
    decomposed = "café"  # "café" as e + combining acute accent
    precomposed = "café"
    assert verify_literal(precomposed, f"the {decomposed} closed")
    assert verify_literal(decomposed, f"the {precomposed} closed")


# --- real fixture vault: literal + metadata + filter-leak safety ------------


def test_literal_search_finds_and_verifies_exact_text(tmp_path: Path) -> None:
    client, settings = index_fixture_vault(tmp_path)
    retriever = Retriever(settings, client=client, encoders=HashEncoders())
    target_note = str(compute_note_id("personal", "logs/2026-08-20.md"))

    results = retriever.search(
        QueryPlan(mode=RetrievalMode.LITERAL, literal_text="fried chicken", note_id=target_note)
    )

    assert len(results) == 1
    result = results[0]
    assert result.path == "logs/2026-08-20.md"
    assert result.heading == "Meals"
    assert result.verified is True
    assert result.note_date == "2026-08-20"
    assert result.entry_date is None


def test_literal_search_rejects_non_contiguous_word_matches(tmp_path: Path) -> None:
    client, settings = index_fixture_vault(tmp_path)
    retriever = Retriever(settings, client=client, encoders=HashEncoders())

    results = retriever.search(QueryPlan(mode=RetrievalMode.LITERAL, literal_text="chicken fried"))
    assert results == ()


def test_metadata_search_orders_deterministically_by_path_and_line(tmp_path: Path) -> None:
    client, settings = index_fixture_vault(tmp_path)
    retriever = Retriever(settings, client=client, encoders=HashEncoders())
    target_note = str(compute_note_id("personal", "logs/2026-08-20.md"))

    results = retriever.search(
        QueryPlan(mode=RetrievalMode.METADATA, note_id=target_note, tags_any=("journal", "recipe"))
    )
    assert [(r.path, r.start_line) for r in results] == sorted((r.path, r.start_line) for r in results)
    assert [r.heading for r in results] == ["Thoughts", "Meals"]


@pytest.mark.parametrize("mode", [RetrievalMode.LEXICAL, RetrievalMode.SEMANTIC])
def test_hard_note_filter_never_leaks_across_vector_modes(tmp_path: Path, mode: RetrievalMode) -> None:
    client, settings = index_fixture_vault(tmp_path)
    retriever = Retriever(settings, client=client, encoders=HashEncoders())
    target_note = str(compute_note_id("personal", "logs/2026-08-20.md"))

    plan = QueryPlan(
        mode=mode,
        note_id=target_note,
        lexical_text="chicken",
        semantic_text="chicken",
        limit=50,
    )
    results = retriever.search(plan)
    assert results
    assert all(r.note_id == target_note for r in results)


def test_hybrid_filter_never_leaks(tmp_path: Path) -> None:
    client, settings = index_fixture_vault(tmp_path)
    retriever = Retriever(settings, client=client, encoders=HashEncoders())
    target_note = str(compute_note_id("personal", "logs/2026-08-20.md"))

    plan = QueryPlan(
        mode=RetrievalMode.HYBRID,
        note_id=target_note,
        lexical_text="chicken",
        semantic_text="chicken",
        limit=50,
    )
    results = retriever.search(plan)
    assert results
    assert all(r.note_id == target_note for r in results)


# --- synthetic points: filter correctness -----------------------------------


def test_tags_all_requires_every_tag_and_tags_any_requires_one(tmp_path: Path) -> None:
    client = real_client(tmp_path)
    settings = make_settings(FIXTURE_VAULT, tmp_path)
    upsert_point(client, 1, base_payload(entry_id="p1", tags=["journal", "work"]))
    upsert_point(client, 2, base_payload(entry_id="p2", tags=["journal"]))
    upsert_point(client, 3, base_payload(entry_id="p3", tags=["recipe"]))
    retriever = Retriever(settings, client=client)

    all_results = retriever.search(QueryPlan(mode=RetrievalMode.METADATA, tags_all=("journal", "work")))
    assert {r.entry_id for r in all_results} == {"p1"}

    any_results = retriever.search(QueryPlan(mode=RetrievalMode.METADATA, tags_any=("work", "recipe")))
    assert {r.entry_id for r in any_results} == {"p1", "p3"}


def test_note_id_and_section_filters(tmp_path: Path) -> None:
    client = real_client(tmp_path)
    settings = make_settings(FIXTURE_VAULT, tmp_path)
    upsert_point(
        client, 1, base_payload(entry_id="p1", note_id="note-a", heading="Intro", heading_path=["Intro"])
    )
    upsert_point(
        client,
        2,
        base_payload(entry_id="p2", note_id="note-b", heading="Body", heading_path=["Parent", "Body"]),
    )
    retriever = Retriever(settings, client=client)

    by_note = retriever.search(QueryPlan(mode=RetrievalMode.METADATA, note_id="note-a"))
    assert {r.entry_id for r in by_note} == {"p1"}

    by_section_heading = retriever.search(QueryPlan(mode=RetrievalMode.METADATA, section="Body"))
    assert {r.entry_id for r in by_section_heading} == {"p2"}

    by_section_path = retriever.search(QueryPlan(mode=RetrievalMode.METADATA, section="Parent"))
    assert {r.entry_id for r in by_section_path} == {"p2"}


def test_entry_type_and_kanban_filters(tmp_path: Path) -> None:
    client = real_client(tmp_path)
    settings = make_settings(FIXTURE_VAULT, tmp_path)
    upsert_point(
        client,
        1,
        base_payload(
            entry_id="p1",
            entry_type="kanban_card",
            note_title="Kitchen",
            board={
                "column": "Doing",
                "status": None,
                "column_position": 0,
                "card_position": 0,
                "checked": False,
            },
        ),
    )
    upsert_point(
        client,
        2,
        base_payload(
            entry_id="p2",
            entry_type="kanban_card",
            note_title="Kitchen",
            board={
                "column": "Done",
                "status": None,
                "column_position": 1,
                "card_position": 0,
                "checked": True,
            },
        ),
    )
    upsert_point(client, 3, base_payload(entry_id="p3", entry_type="freeform_section"))
    retriever = Retriever(settings, client=client)

    by_type = retriever.search(QueryPlan(mode=RetrievalMode.METADATA, entry_types=(EntryType.KANBAN_CARD,)))
    assert {r.entry_id for r in by_type} == {"p1", "p2"}

    by_column = retriever.search(
        QueryPlan(mode=RetrievalMode.METADATA, kanban_board="Kitchen", kanban_column="Doing")
    )
    assert {r.entry_id for r in by_column} == {"p1"}

    by_checked = retriever.search(QueryPlan(mode=RetrievalMode.METADATA, kanban_checked=True))
    result = by_checked[0]
    assert {r.entry_id for r in by_checked} == {"p2"}
    assert result.board.column == "Done"


def test_date_range_matches_note_date_or_entry_date_while_staying_distinct(tmp_path: Path) -> None:
    client = real_client(tmp_path)
    settings = make_settings(FIXTURE_VAULT, tmp_path)
    upsert_point(client, 1, base_payload(entry_id="p1", note_date="2026-01-01T00:00:00Z", entry_date=None))
    upsert_point(client, 2, base_payload(entry_id="p2", note_date=None, entry_date="2026-01-01T00:00:00Z"))
    upsert_point(client, 3, base_payload(entry_id="p3", note_date="2027-01-01T00:00:00Z", entry_date=None))
    retriever = Retriever(settings, client=client)

    results = retriever.search(
        QueryPlan(mode=RetrievalMode.METADATA, date_from="2026-01-01", date_to="2026-01-01")
    )
    by_id = {r.entry_id: r for r in results}
    assert set(by_id) == {"p1", "p2"}
    assert by_id["p1"].note_date == "2026-01-01" and by_id["p1"].entry_date is None
    assert by_id["p2"].entry_date == "2026-01-01" and by_id["p2"].note_date is None


def test_outgoing_links_round_trip_with_full_evidence(tmp_path: Path) -> None:
    client = real_client(tmp_path)
    settings = make_settings(FIXTURE_VAULT, tmp_path)
    upsert_point(
        client,
        1,
        base_payload(
            entry_id="p1",
            outgoing_note_ids=["note-2"],
            outgoing_links=[
                {
                    "target_text": "Target Note",
                    "target_note_id": "note-2",
                    "target_heading": "Section",
                    "alias": "the target",
                    "resolution": "resolved",
                    "line": 5,
                }
            ],
        ),
    )
    retriever = Retriever(settings, client=client)

    result = retriever.search(QueryPlan(mode=RetrievalMode.METADATA))[0]
    assert len(result.outgoing_links) == 1
    link = result.outgoing_links[0]
    assert link.target_text == "Target Note"
    assert link.target_note_id == "note-2"
    assert link.target_heading == "Section"
    assert link.alias == "the target"
    assert link.resolution is LinkResolution.RESOLVED
    assert link.line == 5


def test_entry_lookup_backlinks_and_note_scoped_search_use_resolved_payloads(
    tmp_path: Path,
) -> None:
    client, settings = index_fixture_vault(tmp_path)
    retriever = Retriever(settings, client=client, encoders=HashEncoders())
    daily_id = str(compute_note_id("personal", "logs/2026-08-20.md"))
    project_id = str(compute_note_id("personal", "projects/News Resolution.md"))

    daily = retriever.search(QueryPlan(mode=RetrievalMode.METADATA, note_id=daily_id, section="Work"))[0]
    assert retriever.get_entry(daily.entry_id) == daily

    backlinks = retriever.get_backlinks(project_id)
    assert {result.path for result in backlinks} >= {
        "freeform/Reference.md",
        "logs/2026-08-20.md",
    }

    within = retriever.search_within(
        project_id,
        "pipeline clustering",
        mode=RetrievalMode.LEXICAL,
        limit=1,
    )
    assert len(within) == 1
    assert within[0].note_id == project_id


def test_vault_id_isolates_results(tmp_path: Path) -> None:
    client = real_client(tmp_path)
    settings = make_settings(FIXTURE_VAULT, tmp_path)
    upsert_point(client, 1, base_payload(entry_id="p1", vault_id="personal"))
    upsert_point(client, 2, base_payload(entry_id="p2", vault_id="work"))
    retriever = Retriever(settings, vault_id="personal", client=client)

    results = retriever.search(QueryPlan(mode=RetrievalMode.METADATA, limit=100))
    assert {r.entry_id for r in results} == {"p1"}


# --- synthetic points: lexical / semantic / hybrid ranking mechanics -------


def test_lexical_semantic_and_hybrid_surface_different_signals(tmp_path: Path) -> None:
    client = real_client(tmp_path)
    settings = make_settings(FIXTURE_VAULT, tmp_path)

    lexical_hit_sparse = models.SparseVector(indices=[10, 11], values=[2.0, 2.0])
    semantic_hit_dense = [0.0] * DENSE_DIMS
    semantic_hit_dense[0] = 1.0

    p1_dense = [0.0] * DENSE_DIMS
    p1_dense[2] = 1.0  # orthogonal to the semantic query
    p3_dense = [0.0] * DENSE_DIMS
    p3_dense[1] = 1.0  # orthogonal to the semantic query

    upsert_point(client, 1, base_payload(entry_id="p1"), dense=p1_dense, sparse=lexical_hit_sparse)
    upsert_point(
        client,
        2,
        base_payload(entry_id="p2"),
        dense=semantic_hit_dense,
        sparse=models.SparseVector(indices=[50], values=[0.01]),
    )
    upsert_point(
        client,
        3,
        base_payload(entry_id="p3"),
        dense=p3_dense,
        sparse=models.SparseVector(indices=[60], values=[0.01]),
    )

    encoders = QueryEncoders(
        dense={"find-semantic": semantic_hit_dense},
        sparse={"find-lexical": lexical_hit_sparse},
    )
    retriever = Retriever(settings, client=client, encoders=encoders)

    lexical = retriever.search(QueryPlan(mode=RetrievalMode.LEXICAL, lexical_text="find-lexical", limit=5))
    assert lexical[0].entry_id == "p1"
    assert lexical[0].score is not None

    semantic = retriever.search(
        QueryPlan(mode=RetrievalMode.SEMANTIC, semantic_text="find-semantic", limit=5)
    )
    assert semantic[0].entry_id == "p2"

    hybrid = retriever.search(
        QueryPlan(
            mode=RetrievalMode.HYBRID, lexical_text="find-lexical", semantic_text="find-semantic", limit=2
        )
    )
    assert {r.entry_id for r in hybrid} == {"p1", "p2"}


def test_hybrid_requires_both_query_texts(tmp_path: Path) -> None:
    client = real_client(tmp_path)
    settings = make_settings(FIXTURE_VAULT, tmp_path)
    retriever = Retriever(settings, client=client, encoders=QueryEncoders({}, {}))

    with pytest.raises(ValueError, match="hybrid mode requires both"):
        retriever.search(QueryPlan(mode=RetrievalMode.HYBRID, lexical_text="only lexical"))


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("2026-10-01", "2026-10-01", True),
        ("2026-10-02", "2026-10-02", True),
        ("2026-09-30", "2026-09-30", False),
        ("2026-10-03", "2026-10-03", False),
        ("2026-09-30", "2026-10-03", True),
        ("2026-10-02", None, True),
        ("2026-10-03", None, False),
        (None, "2026-10-01", True),
        (None, "2026-09-30", False),
    ],
)
def test_date_filters_match_inclusive_entry_intervals(tmp_path, start, end, expected):
    client = real_client(tmp_path)
    try:
        upsert_point(
            client,
            1,
            base_payload(
                entry_id="recap",
                entry_date="2026-10-01T00:00:00Z",
                entry_date_end="2026-10-02T00:00:00Z",
            ),
        )
        retriever = Retriever(make_settings(FIXTURE_VAULT, tmp_path), client=client)
        matches = retriever.search(QueryPlan(mode=RetrievalMode.METADATA, date_from=start, date_to=end))
        assert bool(matches) is expected
        if matches:
            assert matches[0].entry_date == "2026-10-01"
            assert matches[0].entry_date_end == "2026-10-02"
    finally:
        client.close()


def test_compacted_log_and_heading_link_round_trip(tmp_path):
    from zenith.retrieval.context import ContextExpander

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Source.md").write_text("## Work\nSee [[Project#2026-10-01 to 2026-10-02]].\n")
    (vault / "Project.md").write_text(
        "## 2026-10-01 to 2026-10-02\n"
        + "First decision. " * 30
        + "\n\n"
        + "Second decision. " * 30
        + "\n"
        + "### Detail\nChild evidence.\n"
        + "## 2026-10-03\nLater evidence.\n"
    )
    settings = make_settings(vault, tmp_path)
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        IndexRebuilder(settings, client=client, encoders=HashEncoders()).rebuild()
        retriever = Retriever(settings, client=client)
        entries = retriever.metadata_all(
            QueryPlan(
                mode=RetrievalMode.METADATA,
                date_from="2026-10-02",
                date_to="2026-10-02",
            )
        )
        assert len(entries) >= 3
        assert all(e.entry_date == "2026-10-01" and e.entry_date_end == "2026-10-02" for e in entries)
        source = next(e for e in retriever.metadata_all() if e.path == "Source.md")
        expansion = ContextExpander(retriever).expand(source.entry_id)
        assert [i.result.entry_id for i in expansion.items[1:]] == [e.entry_id for e in entries]
        assert not expansion.truncated
        partial = ContextExpander(retriever).expand(source.entry_id, max_items=2)
        continuation = partial.section_continuations[0]
        assert continuation.start_line == entries[1].start_line
        assert continuation.end_line == entries[-1].end_line
    finally:
        client.close()
