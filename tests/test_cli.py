import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from zenith.core.config import Settings
from zenith.runtime.cli import build_parser, main


@pytest.fixture
def cli_settings(monkeypatch, tmp_path: Path) -> Settings:
    settings = Settings("http://unused", tmp_path, tmp_path, "entries", "127.0.0.1", 8080)
    monkeypatch.setattr("zenith.runtime.cli.Settings.from_env", lambda: settings)
    return settings


@pytest.mark.parametrize(
    "argv",
    [
        ["index", "init"],
        ["index", "rebuild"],
        ["search", "query", "--mode", "hybrid"],
        ["ask", "what did I do last week?"],
        ["note", "get", "Note"],
        ["note", "read", "Note"],
        ["entry", "get", "entry-id"],
        ["links", "outgoing", "note-id"],
        ["links", "backlinks", "note-id"],
        ["links", "resolve", "note-id", "Target"],
        ["search-within", "note-id", "query"],
        ["context", "entry-id"],
        ["warnings", "--type", "missing_link"],
        ["kanban", "list"],
        ["kanban", "get", "Board"],
        ["kanban", "find", "task", "--checked", "false"],
        ["graph", "export"],
        ["diagnose"],
    ],
)
def test_phase_7_command_surface_parses(argv: list[str]) -> None:
    assert build_parser().parse_args(argv).command


def test_search_command_emits_stable_json(monkeypatch, cli_settings, capsys) -> None:
    calls: list[dict[str, object]] = []

    class FakeZenith:
        def __init__(self, settings: Settings) -> None:
            assert settings is cli_settings

        def find_entries(self, **kwargs: object):
            calls.append(kwargs)
            return (SimpleNamespace(to_dict=lambda: {"entry_id": "entry-1"}),)

    monkeypatch.setattr("zenith.runtime.cli.Zenith", FakeZenith)
    code = main(
        [
            "search",
            "fried chicken",
            "--mode",
            "literal",
            "--tag-all",
            "recipe",
            "--limit",
            "3",
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {"results": [{"entry_id": "entry-1"}]}
    assert calls[0]["exact_text"] == "fried chicken"
    assert calls[0]["tags_all"] == ("recipe",)
    assert calls[0]["limit"] == 3


def test_cli_validation_errors_are_json_and_exit_two(monkeypatch, cli_settings, capsys) -> None:
    monkeypatch.setattr("zenith.runtime.cli.Zenith", lambda _: object())

    code = main(["search", "--mode", "literal"])
    error = json.loads(capsys.readouterr().err)

    assert code == 2
    assert error["error"]["type"] == "ValueError"
    assert "requires query text" in error["error"]["message"]


def test_unknown_command_is_json_and_exit_two(cli_settings, capsys) -> None:
    code = main(["not-a-command"])
    error = json.loads(capsys.readouterr().err)
    assert code == 2
    assert error["error"]["type"] == "ValueError"


def test_note_read_command_emits_the_markdown_file(monkeypatch, cli_settings, capsys) -> None:
    class FakeZenith:
        def __init__(self, settings: Settings) -> None:
            assert settings is cli_settings

        def read_note(self, path_or_title: str):
            assert path_or_title == "News Resolution"
            return SimpleNamespace(
                to_dict=lambda: {
                    "content": "# News Resolution\n",
                    "note_id": "uuid",
                    "note_type": "standard",
                    "path": "projects/News Resolution.md",
                    "title": "News Resolution",
                }
            )

        def get_note(self, path_or_title: str):  # pragma: no cover - must not run
            raise AssertionError("note read must not fall through to note get")

    monkeypatch.setattr("zenith.runtime.cli.Zenith", FakeZenith)
    assert main(["note", "read", "News Resolution"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["content"] == "# News Resolution\n"
    assert list(payload) == sorted(payload)


def test_ask_refuses_to_answer_from_an_unready_index(monkeypatch, cli_settings, capsys) -> None:
    """Silence from an unready index reads exactly like an honest empty answer."""
    monkeypatch.setattr("zenith.runtime.cli.Zenith", lambda settings: object())
    monkeypatch.setattr(
        "zenith.runtime.cli.health_report",
        lambda settings: {"ready": False, "index": {"ready": False}},
    )

    def must_not_reach(settings):  # pragma: no cover - must not run
        raise AssertionError("LM Studio must not be contacted for an unready index")

    monkeypatch.setattr("zenith.runtime.cli.llm_health", must_not_reach)
    assert main(["ask", "what did I do?"]) == 1
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["type"] == "IndexUnavailable"
    assert error["health"]["index"]["ready"] is False


def test_ask_refuses_to_run_when_lm_studio_is_not_serving_the_model(
    monkeypatch, cli_settings, capsys
) -> None:
    monkeypatch.setattr("zenith.runtime.cli.Zenith", lambda settings: object())
    monkeypatch.setattr("zenith.runtime.cli.health_report", lambda settings: {"ready": True})
    monkeypatch.setattr(
        "zenith.runtime.cli.llm_health",
        lambda settings: {"ready": False, "required": False, "error": "Connection refused"},
    )

    def must_not_build(*_: object, **__: object):  # pragma: no cover - must not run
        raise AssertionError("the agent must not be built without a served model")

    monkeypatch.setattr("zenith.agent.build_agent", must_not_build)
    assert main(["ask", "what did I do?"]) == 1
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["type"] == "LLMUnavailable"
    assert "Serve on Local Network" in error["error"]["message"]
    assert error["llm"]["ready"] is False


def test_ask_reports_the_answer_with_the_evidence_it_looked_at(monkeypatch, cli_settings, capsys) -> None:
    from pydantic_ai import RunContext
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    import zenith.agent.service as agent_service

    deps = object()
    reached: list[object] = []

    def search_notes(ctx: RunContext[object], query: str) -> list[dict]:
        """Search the notes vault.

        Args:
            query: What to look for.
        """
        reached.append(ctx.deps)
        return [{"id": "s1", "path": "projects/News Resolution.md"}]

    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("search_notes", {"query": "pipeline"})])
        return ModelResponse(parts=[TextPart("You worked on the pipeline. [s1]")])

    monkeypatch.setattr("zenith.runtime.cli.Zenith", lambda settings: deps)
    monkeypatch.setattr("zenith.runtime.cli.health_report", lambda settings: {"ready": True})
    monkeypatch.setattr("zenith.runtime.cli.llm_health", lambda settings: {"ready": True})
    monkeypatch.setattr(agent_service, "TOOLS", [search_notes])
    monkeypatch.setattr(agent_service, "build_model", lambda settings: FunctionModel(script))

    assert main(["ask", "what did I do?", "--model", "lfm2.5-8b-a1b"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["answer"] == "You worked on the pipeline. [s1]"
    # This tool bypasses the projections, so it issued no label. A citation the
    # run cannot account for is reported as one, not quietly rendered as a note.
    assert payload["citations"] == {"s1": {"unknown": True}}
    assert payload["citation_valid"] is False
    assert payload["citation_errors"] == ["unknown citation [s1]"]
    assert payload["model"] == "lfm2.5-8b-a1b"
    assert payload["question"] == "what did I do?"
    # The trace is part of the result: an answer is only as good as what it read.
    assert payload["tool_calls"] == [{"arguments": {"query": "pipeline"}, "tool": "search_notes"}]
    assert payload["usage"]["requests"] == 3
    assert payload["usage"]["tool_calls"] == 1
    # The agent is handed the run's source ledger, holding the library.
    assert [item.vault for item in reached] == [deps]
    assert list(payload) == sorted(payload)


def test_ask_corrects_a_missing_citation(monkeypatch, cli_settings, capsys) -> None:
    from pydantic_ai import RunContext
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, UserPromptPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    import zenith.agent.service as agent_service
    from zenith.agent.sources import Sources
    from zenith.core.contracts import NoteContent, NoteType

    def search_notes(ctx: RunContext[Sources], query: str) -> dict:
        """Return one source.

        Args:
            query: What to look for.
        """
        label = ctx.deps.label_note(NoteContent("n", "Note.md", "Note", NoteType.STANDARD, "body"))
        return {"items": [{"id": label, "note": "Note", "path": "Note.md"}], "has_more": False}

    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(
            isinstance(part, UserPromptPart) and "Rewrite your immediately previous answer" in str(part.content)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart("The note says this. [s1]")])
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("search_notes", {"query": "note"})])
        return ModelResponse(parts=[TextPart("The note says this.")])

    monkeypatch.setattr("zenith.runtime.cli.Zenith", lambda settings: object())
    monkeypatch.setattr("zenith.runtime.cli.health_report", lambda settings: {"ready": True})
    monkeypatch.setattr("zenith.runtime.cli.llm_health", lambda settings: {"ready": True})
    monkeypatch.setattr(agent_service, "TOOLS", [search_notes])
    monkeypatch.setattr(agent_service, "build_model", lambda settings: FunctionModel(script))

    assert main(["ask", "what does the note say?"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["answer"] == "The note says this. [s1]"
    assert payload["citation_valid"] is True
    assert payload["citation_errors"] == []
    assert payload["citations"] == {"s1": {"note": "Note", "path": "Note.md"}}


def test_ask_leaves_the_agent_stack_unimported_for_other_commands(monkeypatch, cli_settings) -> None:
    """Importing pydantic-ai costs over a second; only `ask` may pay it."""
    import subprocess
    import sys

    probe = "import sys, zenith.runtime.cli;sys.exit(1 if 'pydantic_ai' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", probe]).returncode == 0
