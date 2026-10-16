"""The answering agent's tool loop, proved without LM Studio or a network."""

import json
import re
import shutil
import warnings
from pathlib import Path

import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from qdrant_client import QdrantClient, models

from bedrock.agent import DEFAULT_LIMITS, Sources, build_agent
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
def vault_api(tmp_path: Path) -> Sources:
    vault = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, vault)
    settings = Settings("http://unused", vault, tmp_path / "models", "entries", "127.0.0.1", 8080)
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    encoders = HashEncoders()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        IndexRebuilder(settings, client=client, encoders=encoders).rebuild()
    # The agent depends on the run's source ledger, not on the library, so the
    # labels it hands out survive from one tool call to the next.
    return Sources(Bedrock(settings, client=client, encoders=encoders))


@pytest.fixture
def settings(vault_api: Sources) -> Settings:
    return vault_api.vault.settings


def _returns(messages: list[ModelMessage], tool_name: str) -> list[object]:
    """Collect what the tools actually handed back to the model."""
    returned = []
    returned.extend(
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name
    )
    return returned


EVERY_TOOL = {
    "search_notes",
    "read_note",
    "expand_context",
    "find_backlinks",
    "find_tasks",
    "list_notes",
    "create_board",
    "get_task",
    "add_task",
    "update_task",
    "move_task",
    "archive_task",
}


def test_every_tool_is_declared_and_succeeds_in_a_real_agent_run(settings, vault_api) -> None:
    """Walk all read and scoped edit tools with real arguments in one agent run.

    TestModel invents argument values, which a name or identifier tool rightly
    rejects. Driving the loop with real values is what proves the tools work,
    not only that they are registered.
    """
    declared: list[set[str]] = []

    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        declared.append({tool.name for tool in info.function_tools})
        done = {
            part.tool_name
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        }
        if "search_notes" not in done:
            return _call("search_notes", {"query": "pipeline", "limit": 3})
        found = _payload(_returns(messages, "search_notes")[0])["items"]
        if "read_note" not in done:
            return _call("read_note", {"name": found[0]["path"]})
        if "expand_context" not in done:
            return _call("expand_context", {"source": found[0]["id"]})
        if "find_backlinks" not in done:
            return _call("find_backlinks", {"name": "News Resolution"})
        if "find_tasks" not in done:
            return _call("find_tasks", {"board": "Kitchen App", "limit": 3})
        if "list_notes" not in done:
            return _call("list_notes", {"note_type": "standard", "limit": 3})
        if "create_board" not in done:
            return _call("create_board", {"path": "kanban/Agent Playground.md", "request_id": "agent-board"})
        created = _payload(_returns(messages, "create_board")[0])
        if "add_task" not in done:
            return _call(
                "add_task",
                {
                    "board": created["path"],
                    "column": "ToDo",
                    "title": "Agent experiment",
                    "expected_hash": created["content_hash"],
                    "request_id": "agent-task",
                    "due_date": "2026-10-20",
                },
            )
        added = _payload(_returns(messages, "add_task")[0])
        task_id = added["task"]["task_id"]
        if "get_task" not in done:
            return _call("get_task", {"board": created["path"], "task": task_id})
        read = _payload(_returns(messages, "get_task")[0])
        if "update_task" not in done:
            return _call(
                "update_task",
                {
                    "board": created["path"],
                    "task": task_id,
                    "expected_hash": read["content_hash"],
                    "content": "- Real agent tool update",
                },
            )
        changed = _payload(_returns(messages, "update_task")[0])
        if "move_task" not in done:
            return _call(
                "move_task",
                {
                    "board": created["path"],
                    "task": task_id,
                    "column": "Completed",
                    "expected_hash": changed["content_hash"],
                },
            )
        moved = _payload(_returns(messages, "move_task")[0])
        if "archive_task" not in done:
            return _call(
                "archive_task",
                {"board": created["path"], "task": task_id, "expected_hash": moved["content_hash"]},
            )
        archived = _payload(_returns(messages, "archive_task")[0])
        assert archived["task"]["archived"] and archived["task"]["checked"]
        assert archived["task"]["completed_at"] and archived["index_status"] == "updated"
        return ModelResponse(parts=[TextPart("done")])

    agent = build_agent(settings, model=FunctionModel(script))
    result = agent.run_sync("walk every tool", deps=vault_api)

    assert declared[0] == EVERY_TOOL
    succeeded = {
        part.tool_name
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    assert succeeded == EVERY_TOOL
    # No tool asked the model to correct itself: every call was answered.
    assert not [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, RetryPromptPart)
    ]


def test_test_model_still_exercises_the_tools_it_can_guess_arguments_for(settings, vault_api) -> None:
    """A guessed argument must fail as a repairable retry, never as a crash."""
    agent = build_agent(settings, model=TestModel(call_tools=["search_notes", "find_tasks"]))
    result = agent.run_sync("anything", deps=vault_api)
    called = {
        part.tool_name
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    }
    assert called == {"search_notes", "find_tasks"}


def _call(name: str, args: dict) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(name, args)])


def _payload(content: object) -> list[dict]:
    return json.loads(content) if isinstance(content, str) else content


def test_the_model_can_search_then_follow_an_entry_into_its_context(settings, vault_api) -> None:
    """A two-step loop: the second call uses an id the first call produced."""
    steps: list[str] = []

    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        searched = _returns(messages, "search_notes")
        expanded = _returns(messages, "expand_context")
        if not searched:
            steps.append("search")
            return ModelResponse(parts=[ToolCallPart("search_notes", {"query": "pipeline", "limit": 3})])
        if not expanded:
            entries = _payload(searched[0])["items"]
            steps.append("expand")
            return ModelResponse(parts=[ToolCallPart("expand_context", {"source": entries[0]["id"]})])
        context = _payload(expanded[0])
        labels = {label for item in context["items"] for label in item["evidence"]}
        steps.append("answer")
        return ModelResponse(parts=[TextPart(f"labels={sorted(labels)}")])

    agent = build_agent(settings, model=FunctionModel(script))
    result = agent.run_sync("what happened on the pipeline?", deps=vault_api)

    assert steps == ["search", "expand", "answer"]
    assert "direct_evidence" in result.output


def test_a_wrong_note_name_comes_back_as_a_repairable_retry(settings, vault_api) -> None:
    """The library's own error text must reach the model and let it correct."""
    seen_retries: list[str] = []

    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, RetryPromptPart):
                        content = part.content
                        if isinstance(content, str) and content not in seen_retries:
                            seen_retries.append(content)
        if not seen_retries:
            return ModelResponse(parts=[ToolCallPart("read_note", {"name": "Shared"})])
        if not _returns(messages, "read_note"):
            # The retry text names the candidates, so a corrected call is possible.
            candidate = re.search(r"([\w /]+\.md)", seen_retries[0]).group(1)
            return ModelResponse(parts=[ToolCallPart("read_note", {"name": candidate})])
        return ModelResponse(parts=[TextPart("recovered")])

    agent = build_agent(settings, model=FunctionModel(script))
    result = agent.run_sync("read the shared note", deps=vault_api)

    assert seen_retries and "ambiguous note title" in seen_retries[0]
    assert ".md" in seen_retries[0]
    assert result.output == "recovered"
    assert _returns(result.all_messages(), "read_note")


def test_a_runaway_tool_loop_is_stopped_by_the_usage_limits(settings, vault_api) -> None:
    def never_answers(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart("search_notes", {"query": "again"})])

    agent = build_agent(settings, model=FunctionModel(never_answers))
    with pytest.raises(UsageLimitExceeded):
        agent.run_sync("spin", deps=vault_api, usage_limits=DEFAULT_LIMITS)

    assert DEFAULT_LIMITS.request_limit == 12
    assert DEFAULT_LIMITS.tool_calls_limit == 20


def test_the_instructions_state_the_citation_and_dating_rules(settings, vault_api) -> None:
    captured: list[str] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured.append(info.instructions or "")
        return ModelResponse(parts=[TextPart("done")])

    build_agent(settings, model=FunctionModel(capture)).run_sync("hi", deps=vault_api)

    instructions = captured[0]
    # The model cites a label, not a citation it assembles. The worked example
    # and the rejected forms are what it actually follows, so both are part of
    # the contract now, not prose that may be reworded away.
    assert "parsing model. [s2]" in instructions
    assert "[s2][s5]" in instructions
    assert "[News Resolution, 2026-08-17, lines 6-9]" in instructions
    assert "Put nothing inside the brackets except the id" in instructions
    assert "undated" in instructions
    assert "exact_match_verified" in instructions


def test_the_agent_is_built_against_lm_studio_by_default(settings) -> None:
    agent = build_agent(settings)
    assert agent.model.model_name == settings.llm_model
    assert agent.model.base_url.rstrip("/") == settings.llm_base_url
    profile = agent.model.profile
    assert profile["openai_supports_strict_tool_definition"] is False
    assert profile["openai_supports_tool_choice_required"] is False
    # A hosted API's timeout and retry defaults are wrong for a local model:
    # 600 seconds retried twice reads to a person as a hung command.
    client = agent.model.client
    assert client.timeout == settings.llm_timeout
    assert client.max_retries == 0


def test_the_configured_temperature_reaches_the_request(settings, vault_api) -> None:
    """Temperature is a request parameter, not model state.

    LM Studio holds its own preset, and the value sent here overrides it, so
    the setting is only real if it arrives with the request.
    """
    from dataclasses import replace

    captured: list[object] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured.append((info.model_settings or {}).get("temperature"))
        return ModelResponse(parts=[TextPart("done")])

    build_agent(settings, model=FunctionModel(capture)).run_sync("hi", deps=vault_api)
    assert captured[0] == 0.0

    warm = replace(settings, llm_temperature=0.7)
    build_agent(warm, model=FunctionModel(capture)).run_sync("hi", deps=vault_api)
    assert captured[1] == 0.7
