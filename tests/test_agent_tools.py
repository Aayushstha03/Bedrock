"""The tool surface the answering model is given, tested without a model."""

import re
import shutil
import typing
import warnings
from pathlib import Path

import pytest
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import Tool
from pydantic_ai.usage import RunUsage
from qdrant_client import QdrantClient, models

from bedrock.agent import tools
from bedrock.agent.projections import MAX_ENTRY_CHARS, MAX_NOTE_CHARS
from bedrock.agent.sources import Sources
from bedrock.core.config import Settings
from bedrock.index.rebuild import IndexRebuilder
from bedrock.library import Bedrock

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


@pytest.fixture
def ctx(tmp_path: Path) -> RunContext[Sources]:
    vault = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, vault)
    settings = Settings("http://unused", vault, tmp_path / "models", "entries", "127.0.0.1", 8080)
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    encoders = HashEncoders()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        IndexRebuilder(settings, client=client, encoders=encoders).rebuild()
    api = Bedrock(settings, client=client, encoders=encoders)
    return RunContext(deps=Sources(api), model=TestModel(), usage=RunUsage())


def test_every_tool_declares_a_flat_described_schema() -> None:
    for function in tools.TOOLS:
        definition = Tool(function, require_parameter_descriptions=True).tool_def
        assert definition.description
        schema = definition.parameters_json_schema
        for name, spec in schema.get("properties", {}).items():
            assert spec.get("description"), f"{definition.name}.{name} has no description"
            # Flat arguments only: a local model handles scalars and string
            # lists, and fumbles nested objects.
            types = {spec.get("type")} | {option.get("type") for option in spec.get("anyOf", [])}
            assert types <= {"string", "integer", "boolean", "array", "null", None}
            if "array" in types:
                assert spec.get("items", spec.get("anyOf", [{}])[0].get("items", {})).get("type") == "string"


def test_every_tool_takes_an_annotated_run_context() -> None:
    """An unannotated `ctx` is silently treated as a model-supplied argument.

    pydantic-ai decides by annotation, not by parameter name, so dropping the
    annotation would hand the model a `ctx` argument to invent and would stop
    the run's tool calls being counted against the usage limits.
    """
    import inspect

    for function in tools.TOOLS:
        first = next(iter(inspect.signature(function).parameters.values()))
        annotation = inspect.get_annotations(function, eval_str=True)[first.name]
        assert typing.get_origin(annotation) is RunContext
        assert typing.get_args(annotation) == (Sources,)


def test_the_write_and_unbounded_operations_are_not_offered() -> None:
    offered = {function.__name__ for function in tools.TOOLS}
    assert offered == {
        "search_notes",
        "read_note",
        "expand_context",
        "find_backlinks",
        "find_tasks",
        "list_notes",
    }
    assert "reindex" not in offered
    assert "export_graph" not in offered


def test_search_projects_entries_with_citable_fields(ctx: RunContext[Sources]) -> None:
    page = tools.search_notes(ctx, "pipeline", limit=3)
    assert page
    results = page["items"]
    first = results[0]
    assert set(first) >= {"id", "note", "path", "lines", "date", "date_kind", "text"}
    assert re.fullmatch(r"\d+-\d+", first["lines"])
    # `id` is the citation, and the only identifier the model is given. A
    # result carrying an entry_id would hand it a second one to print.
    assert [result["id"] for result in results] == ["s1", "s2", "s3"][: len(results)]
    assert "entry_id" not in first
    # Similarity is not proof of an exact phrase, so it must not claim to be.
    assert "exact_match_verified" not in first


def test_exact_search_reports_a_verified_literal_match(ctx: RunContext[Sources]) -> None:
    page = tools.search_notes(ctx, "fried chicken", exact=True)
    assert page["items"]
    assert all(result["exact_match_verified"] is True for result in page["items"])


def test_an_undated_entry_is_never_given_a_date(ctx: RunContext[Sources]) -> None:
    for result in tools.search_notes(ctx, "pipeline", limit=20)["items"]:
        if result["date_kind"] is None:
            assert result["date"] is None
        else:
            assert result["date_kind"] in {"entry_date", "note_date"}
            assert result["date"]


def test_read_note_returns_the_whole_file_within_a_cap(ctx: RunContext[Sources]) -> None:
    note = tools.read_note(ctx, "News Resolution")
    assert note["path"] == "projects/News Resolution.md"
    assert note["content"].startswith("# News Resolution")
    assert len(note["content"]) <= MAX_NOTE_CHARS + 32
    assert "content_truncated" not in note


def test_expand_context_keeps_the_evidence_labels(ctx: RunContext[Sources]) -> None:
    source = tools.search_notes(ctx, "pipeline", limit=1)["items"][0]["id"]
    expansion = tools.expand_context(ctx, source)
    # No entry id is echoed back. The caller named the source by its label, and
    # an internal identifier in the result is one the model may print.
    assert "source_entry_id" not in expansion
    assert expansion["items"]
    labels = {label for item in expansion["items"] for label in item["evidence"]}
    assert "direct_evidence" in labels
    assert all("hops" in item for item in expansion["items"])


def test_backlinks_and_tasks_resolve_by_name(ctx: RunContext[Sources]) -> None:
    backlinks = tools.find_backlinks(ctx, "News Resolution")
    assert backlinks["items"]
    assert all("path" in result for result in backlinks["items"])

    open_cards = tools.find_tasks(ctx, board="Kitchen App", columns=["ToDo"], checked=False)
    assert open_cards["items"]
    assert all(card["task"]["column"] == "ToDo" for card in open_cards["items"])
    assert all(card["task"]["checked"] is False for card in open_cards["items"])


def test_a_bad_name_becomes_an_instruction_the_model_can_act_on(
    ctx: RunContext[Sources],
) -> None:
    with pytest.raises(ModelRetry, match="note not found"):
        tools.read_note(ctx, "No Such Note")

    # An ambiguous name must hand back the candidates, not just the failure.
    with pytest.raises(ModelRetry, match="ambiguous note title") as ambiguous:
        tools.read_note(ctx, "Shared")
    assert ".md" in str(ambiguous.value)

    with pytest.raises(ModelRetry, match="not a source this answer has seen"):
        tools.expand_context(ctx, "s99")

    # `id` is the first field of every result, so it is the identifier the
    # model has just read when it reaches for a note name.
    with pytest.raises(ModelRetry, match="source id, not a note name"):
        tools.read_note(ctx, "s1")


def test_limits_are_clamped_rather_than_trusted(ctx: RunContext[Sources]) -> None:
    assert len(tools.search_notes(ctx, "the", limit=999)["items"]) <= 20
    assert tools.search_notes(ctx, "the", limit=0)["items"]


def test_collection_tools_page_and_continue(ctx: RunContext[Sources]) -> None:
    first = tools.search_notes(ctx, "the", limit=1, offset=-10)
    assert len(first["items"]) == 1
    assert first["has_more"] is True
    assert first["next_offset"] == 1

    second = tools.search_notes(ctx, "the", limit=1, offset=first["next_offset"])
    assert second["items"]
    assert second["items"][0]["id"] != first["items"][0]["id"]


def test_queryless_search_orders_dated_entries_before_undated_entries(ctx: RunContext[Sources]) -> None:
    page = tools.search_notes(ctx, limit=20)
    dates = [item["date"] for item in page["items"]]
    assert any(date is not None for date in dates)
    assert any(date is None for date in dates)
    assert dates == sorted((date for date in dates if date is not None), reverse=True) + [None] * dates.count(None)


def test_list_notes_includes_a_note_with_no_indexed_entries(ctx: RunContext[Sources]) -> None:
    (ctx.deps.vault.settings.vault_path / "Empty.md").write_text("# Empty\n")

    page = tools.list_notes(ctx, note_type="standard", limit=50)
    empty = next(note for note in page["items"] if note["path"] == "Empty.md")
    assert empty == {
        "note": "Empty",
        "path": "Empty.md",
        "note_type": "standard",
        "indexed_entries": 0,
    }
    assert all(note["note_type"] == "standard" for note in page["items"])


def test_a_clipped_entry_only_claims_a_verified_match_it_can_show() -> None:
    """`pack` emits an oversized single paragraph rather than cut a sentence.

    Such an entry gets clipped here, so a head-clip can drop the very phrase
    the match was verified against and still assert it was proven.
    """
    from bedrock.agent import projections
    from bedrock.core.contracts import EntryType, NoteType, RetrievalMode, SearchResult

    phrase = "fried chicken"
    text = "a" * 3000 + f" {phrase} " + "b" * 3000
    result = SearchResult(
        "e",
        "n",
        "p.md",
        "N",
        NoteType.STANDARD,
        EntryType.FREEFORM_SECTION,
        RetrievalMode.LITERAL,
        text,
        None,
        (),
        1,
        9,
        None,
        None,
        (),
        (),
        verified=True,
    )

    windowed = projections.entry(Sources(None), result, match=phrase)
    assert windowed["text_truncated"] is True
    assert phrase in windowed["text"]
    assert windowed["exact_match_verified"] is True
    assert len(windowed["text"]) <= MAX_ENTRY_CHARS + 64

    # Without the phrase to centre on, the clip cannot show the proof, so the
    # claim of proof is withheld rather than made on unseen text.
    blind = projections.entry(Sources(None), result)
    assert blind["text_truncated"] is True
    assert phrase not in blind["text"]
    assert "exact_match_verified" not in blind


def test_an_unknown_kanban_column_names_the_real_ones(ctx: RunContext[Sources]) -> None:
    """A case-sensitive miss would otherwise read as an empty board."""
    with pytest.raises(ModelRetry, match="unknown Kanban column") as unknown:
        tools.find_tasks(ctx, board="Kitchen App", columns=["To Do"])
    assert "ToDo" in str(unknown.value)

    # A different case is a near miss, not a mistake, so it is resolved.
    assert tools.find_tasks(ctx, board="Kitchen App", columns=["todo"])


def test_find_tasks_never_claims_a_card_contains_the_query(ctx: RunContext[Sources]) -> None:
    cards = tools.find_tasks(ctx, board="Kitchen App", query="recipe", limit=3)["items"]
    assert cards
    assert all("exact_match_verified" not in card for card in cards)


def test_tasks_filter_explicit_dates_tags_and_statuses(ctx: RunContext[Sources]) -> None:
    dated = tools.find_tasks(ctx, date_from="2026-08-20", date_to="2026-08-20")["items"]
    assert dated
    assert all(card["date"] == "2026-08-20" and card["date_kind"] == "entry_date" for card in dated)

    tagged = tools.find_tasks(ctx, tags=["recipe"])["items"]
    assert tagged and all("recipe" in card["tags"] for card in tagged)

    complete = tools.find_tasks(ctx, statuses=["complete"])["items"]
    assert complete and all(card["task"]["status"] == "complete" for card in complete)


def test_search_narrows_by_tag_and_by_date(ctx: RunContext[Sources]) -> None:
    """The filters are how a question about a period, or a topic, stays honest.

    A model that cannot narrow searches the whole vault and answers from
    whatever ranked highest, which is not the same as what the question asked.
    """
    tagged = tools.search_notes(ctx, "tag", tags=["journal"], limit=5)["items"]
    assert tagged
    assert all("journal" in result["tags"] for result in tagged)

    dated = tools.search_notes(
        ctx, "pipeline", date_from="2026-08-19", date_to="2026-08-19", limit=10
    )["items"]
    assert dated
    assert all(result["date"] == "2026-08-19" for result in dated)


def test_expand_context_reports_the_links_it_could_not_resolve(
    ctx: RunContext[Sources],
) -> None:
    """A link that leads nowhere is evidence too.

    Dropping it silently would leave the model to read the remaining items as
    the whole of what the entry points at, and answer as if nothing were
    missing.
    """
    source = tools.search_notes(ctx, "Missing", exact=True, limit=5)["items"][0]["id"]
    unresolved = tools.expand_context(ctx, source)["unresolved_links"]
    reasons = {item["target"]: item["reason"] for item in unresolved}
    assert reasons["Does Not Exist"] == "missing"
    assert reasons["Shared"] == "ambiguous"


def test_a_column_is_resolved_across_every_board_when_none_is_named(
    ctx: RunContext[Sources],
) -> None:
    """Without a board, a column has to be checked against all of them.

    `Done` exists on one board only. Resolving it against a single board would
    reject a column that really does exist somewhere in the vault.
    """
    assert tools.find_tasks(ctx, columns=["Done"], limit=20)

    # Named against the board that lacks it, the same column is still a miss,
    # and the reply names that board's columns rather than the whole vault's.
    with pytest.raises(ModelRetry, match="unknown Kanban column") as missing:
        tools.find_tasks(ctx, board="Kitchen App", columns=["Done"])
    listed = str(missing.value).split("this board has ")[1]
    assert "ToDo" in listed
    assert "Done" not in listed


def test_an_oversized_note_is_marked_as_a_fragment() -> None:
    """A whole note has no size bound, so `read_note` can still clip one.

    The flag is what stops the model concluding that a note does not mention
    something, from a part of it that does not.
    """
    from bedrock.agent import projections
    from bedrock.core.contracts import NoteContent, NoteType

    content = NoteContent("n", "p.md", "N", NoteType.STANDARD, "a" * (MAX_NOTE_CHARS + 500))
    projected = projections.note(Sources(None), content)

    assert projected["content_truncated"] is True
    assert projected["content"].endswith("[... truncated]")
    assert len(projected["content"]) <= MAX_NOTE_CHARS + 64


def test_a_library_bug_is_not_disguised_as_a_model_mistake() -> None:
    """`_repair` turns a bad argument into a retry. A defect is not one.

    Handing a real fault back to the model as a retry would spend the whole
    usage limit re-calling a tool that cannot work, and hide the fault.
    """
    with pytest.raises(TypeError), tools._repair():
        raise TypeError("the library is broken")


def test_a_whole_note_is_citable_and_cannot_be_expanded(
    ctx: RunContext[Sources],
) -> None:
    """A note read whole has a label, but no entry behind it.

    Its label is what makes a fact drawn from `read_note` citable at all: the
    projection has no line range to cite, so a citation assembled from one
    could not be written. Expansion still needs a real entry, and says so.
    """
    note = tools.read_note(ctx, "News Resolution")
    assert note["id"].startswith("s")
    assert ctx.deps.cited(f"a fact [{note['id']}]") == {
        note["id"]: {"note": "News Resolution", "path": "projects/News Resolution.md"}
    }

    with pytest.raises(ModelRetry, match="is a whole note, not an entry"):
        tools.expand_context(ctx, note["id"])
