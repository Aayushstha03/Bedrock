"""Stable JSON command line and lightweight health service."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty
from threading import Event
from typing import Any, TextIO
from urllib.parse import urlsplit

from qdrant_client import QdrantClient

from bedrock.core.config import Settings
from bedrock.core.contracts import QUERY_TEXT_FIELDS, EntryType, RetrievalMode, WarningType
from bedrock.index.diagnostics import inspect_collection
from bedrock.index.incremental import IncrementalIndexer
from bedrock.index.schema import initialize_index
from bedrock.library import Bedrock
from bedrock.parser.service import VaultParser
from bedrock.parser.watcher import VaultWatcher
from bedrock.runtime.health import encode_report, health_report, llm_health
from bedrock.runtime.models import prefetch, readiness
from bedrock.vault.events import HEARTBEAT_SECONDS, EventBus, GraphUpdates, NoteChangePublisher, encode_sse
from bedrock.vault.graph import GraphSnapshotCache


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _print(data: object, *, file: TextIO | None = None) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str), file=file or sys.stdout)


def _handler(
    settings: Settings, snapshots: GraphSnapshotCache | None = None, events: EventBus | None = None
) -> type[BaseHTTPRequestHandler]:
    snapshots = snapshots or GraphSnapshotCache(settings)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/events" and events is not None:
                self._events()
                return
            graph_routes = {
                "/graph/snapshot",
                "/graph/cosmograph",
                "/graph/cosmograph/points.json",
                "/graph/cosmograph/links.json",
            }
            if path in graph_routes:
                filename = path.rsplit("/", 1)[-1] if path.endswith(".json") else None
                name = filename.removesuffix(".json") if filename else path.rsplit("/", 1)[-1]
                try:
                    response = snapshots.get_response(name)
                except (OSError, ValueError):
                    self._json(503, {"error": "vault snapshot unavailable"})
                    return
                # GET uses weak comparison for If-None-Match, including lists.
                validators = self.headers.get("If-None-Match", "")
                unchanged = any(
                    tag.strip() == "*" or tag.strip().removeprefix("W/") == response.etag
                    for tag in validators.split(",")
                )
                self.send_response(304 if unchanged else 200)
                self.send_header("ETag", response.etag)
                self.send_header("Cache-Control", "private, no-cache")
                if not unchanged:
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(response.body)))
                    if filename is not None:
                        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.end_headers()
                if not unchanged:
                    self.wfile.write(response.body)
                return
            if self.path not in {"/health", "/healthz"}:
                self.send_error(404)
                return
            report = health_report(settings)
            body = encode_report(report)
            self.send_response(200 if report["ready"] else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: object) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _events(self) -> None:
            assert events is not None
            stream = events.open_stream()
            self.close_connection = True
            # Bound writes to clients that stop reading; queues are bounded too.
            self.connection.settimeout(10)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                self.wfile.write(b"retry: 2000\n: connected; revalidate current state\n\n")
                self.wfile.flush()
                while True:
                    try:
                        event = stream.get(timeout=HEARTBEAT_SECONDS)
                    except Empty:
                        body = b": heartbeat\n\n"
                    else:
                        if event is None:
                            return
                        body = encode_sse(event)
                    self.wfile.write(body)
                    self.wfile.flush()
            except OSError:
                pass  # Client disconnected or stopped reading.
            finally:
                events.close_stream(stream)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(prog="bedrock")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="run the health and graph snapshot HTTP service")
    commands.add_parser("health", help="check Qdrant, index, models, vault, and configuration")
    commands.add_parser("diagnose", help="emit combined health and index diagnostics")
    mcp = commands.add_parser("mcp", help="run the read-only Bedrock MCP server")
    mcp.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    mcp.add_argument("--host", default="127.0.0.1")
    mcp.add_argument("--port", type=int, default=8000)

    parse = commands.add_parser("parse", help="parse the configured vault without indexing")
    parse.add_argument("paths", nargs="*", help="optional vault-relative Markdown paths")
    commands.add_parser("watch", help="watch the vault and index debounced Markdown changes")

    index = commands.add_parser("index", help="manage the Qdrant index")
    index_commands = index.add_subparsers(dest="index_command", required=True)
    index_commands.add_parser("init", help="initialize an empty active collection")
    index_commands.add_parser("rebuild", help="atomically rebuild the complete vault index")
    update = index_commands.add_parser("update", help="incrementally converge the active index")
    update.add_argument("paths", nargs="*", help="optional changed vault-relative paths")
    index_commands.add_parser("inspect", help="inspect active collection integrity")

    models = commands.add_parser("models", help="manage local embedding models")
    model_commands = models.add_subparsers(dest="model_command", required=True)
    model_commands.add_parser("prefetch", help="download and validate pinned models")
    model_commands.add_parser("ready", help="inspect model-cache readiness")

    search = commands.add_parser("search", help="search indexed entries")
    search.add_argument("query", nargs="?", help="query text; omit for metadata mode")
    search.add_argument("--mode", choices=_values(RetrievalMode), default="hybrid")
    _add_entry_filters(search)

    ask = commands.add_parser("ask", help="answer a question from the notes vault")
    ask.add_argument("question")
    ask.add_argument("--model", default=None, help="override BEDROCK_LLM_MODEL for this question")

    note = commands.add_parser("note", help="retrieve notes")
    note_commands = note.add_subparsers(dest="note_command", required=True)
    note_get = note_commands.add_parser("get")
    note_get.add_argument("path_or_title")
    note_read = note_commands.add_parser("read", help="read one complete note as Markdown")
    note_read.add_argument("path_or_title")

    entry = commands.add_parser("entry", help="retrieve entries")
    entry_commands = entry.add_subparsers(dest="entry_command", required=True)
    entry_get = entry_commands.add_parser("get")
    entry_get.add_argument("entry_id")

    links = commands.add_parser("links", help="inspect and resolve internal links")
    link_commands = links.add_subparsers(dest="link_command", required=True)
    outgoing = link_commands.add_parser("outgoing")
    outgoing.add_argument("note_id")
    outgoing.add_argument("--entry-id")
    backlinks = link_commands.add_parser("backlinks")
    backlinks.add_argument("note_id")
    resolve = link_commands.add_parser("resolve")
    resolve.add_argument("source_note_id")
    resolve.add_argument("target_text")

    within = commands.add_parser("search-within", help="search within one note")
    within.add_argument("note_id")
    within.add_argument("query")
    within.add_argument("--mode", choices=_values(RetrievalMode), default="hybrid")
    within.add_argument("--section")
    within.add_argument("--limit", type=int, default=10)

    context = commands.add_parser("context", help="expand bounded linked context")
    context.add_argument("entry_id")
    context.add_argument("--link-depth", type=int, default=1)
    context.add_argument("--nearby-days", type=int, default=3)
    context.add_argument("--max-notes", type=int, default=5)

    warnings = commands.add_parser("warnings", help="list index warnings")
    warnings.add_argument("--type", choices=_values(WarningType))
    warnings.add_argument("--path")

    kanban = commands.add_parser("kanban", help="query Kanban boards and cards")
    kanban_commands = kanban.add_subparsers(dest="kanban_command", required=True)
    kanban_commands.add_parser("list")
    kanban_get = kanban_commands.add_parser("get")
    kanban_get.add_argument("board")
    kanban_find = kanban_commands.add_parser("find")
    kanban_find.add_argument("query", nargs="?")
    kanban_find.add_argument("--board")
    kanban_find.add_argument("--column", action="append", default=[])
    kanban_find.add_argument("--status", action="append", default=[])
    kanban_find.add_argument("--checked", choices=("true", "false"))
    kanban_find.add_argument("--tag", action="append", default=[])
    kanban_find.add_argument("--date-from")
    kanban_find.add_argument("--date-to")
    kanban_find.add_argument("--exact", action="store_true")
    kanban_find.add_argument("--limit", type=int, default=10)

    graph = commands.add_parser("graph", help="export the complete note network")
    graph.add_subparsers(dest="graph_command", required=True).add_parser("export")
    return parser


def _add_entry_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--date-from")
    parser.add_argument("--date-to")
    parser.add_argument("--tag-all", action="append", default=[])
    parser.add_argument("--tag-any", action="append", default=[])
    parser.add_argument("--note")
    parser.add_argument("--section")
    parser.add_argument("--entry-type", action="append", choices=_values(EntryType), default=[])
    parser.add_argument("--limit", type=int, default=10)


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return _run(args, Settings.from_env())
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        _print(
            {"error": {"message": str(exc), "type": type(exc).__name__}},
            file=sys.stderr,
        )
        return 2 if isinstance(exc, (LookupError, ValueError)) else 1


def _run(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "mcp":
        from bedrock.vault.mcp import build_server
        from bedrock.vault.service import VaultTools

        if not 1 <= args.port <= 65535:
            raise ValueError("MCP port must be 1-65535")
        vault = Bedrock(settings)
        try:
            build_server(VaultTools(vault), host=args.host, port=args.port).run(transport=args.transport)
        finally:
            vault.client.close()
        return 0
    if args.command == "serve":
        snapshots = GraphSnapshotCache(settings)
        events = EventBus()
        GraphUpdates(snapshots, events)
        changes = NoteChangePublisher(settings, events)
        server = ThreadingHTTPServer((settings.host, settings.port), _handler(settings, snapshots, events))
        try:
            with VaultWatcher(settings, changes.changed) as watcher:
                watcher.catch_up()
                server.serve_forever()
        finally:
            events.close()
            server.server_close()
        return 0
    if args.command == "health":
        report = health_report(settings, include_llm=True)
        _print(report)
        return 0 if report["ready"] else 1
    if args.command == "diagnose":
        report = {
            "health": health_report(settings, include_llm=True),
            "index": inspect_collection(settings),
        }
        _print(report)
        return 0 if report["health"]["ready"] and report["index"]["ready"] else 1
    if args.command == "parse":
        result = VaultParser(settings).parse_vault(args.paths or None)
        _print({"notes": [asdict(note) for note in result]})
        return 0
    if args.command == "watch":
        indexer = IncrementalIndexer(settings)

        def changed(paths: tuple[str, ...]) -> None:
            _print(indexer.reindex(list(paths)).to_dict())

        def failed(error: BaseException) -> None:
            _print(
                {"error": {"message": str(error), "type": type(error).__name__}},
                file=sys.stderr,
            )

        try:
            with VaultWatcher(settings, changed, on_error=failed) as watcher:
                # Index once before waiting, so edits made while the watcher
                # was down are not stranded until the next manual update.
                watcher.catch_up()
                Event().wait()
        except KeyboardInterrupt:
            return 0
        return 0

    if args.command == "models":
        report = prefetch(settings) if args.model_command == "prefetch" else readiness(settings)
        _print(report)
        return 0 if report["ready"] else 1

    if args.command == "index":
        if args.index_command == "init":
            report = initialize_index(settings, QdrantClient(url=settings.qdrant_url)).to_dict()
        elif args.index_command == "rebuild":
            report = Bedrock(settings).reindex(full=True).to_dict()
        elif args.index_command == "update":
            report = Bedrock(settings).reindex(args.paths or None).to_dict()
        else:
            report = inspect_collection(settings)
        _print(report)
        return 0 if report.get("ready", True) else 1

    api = Bedrock(settings)
    if args.command == "ask":
        return _ask(args, settings, api)
    if args.command == "search":
        mode = RetrievalMode(args.mode)
        texts = _query_texts(mode, args.query)
        results = api.find_entries(
            date_from=args.date_from,
            date_to=args.date_to,
            tags_all=tuple(args.tag_all),
            tags_any=tuple(args.tag_any),
            note=args.note,
            section=args.section,
            entry_types=tuple(EntryType(value) for value in args.entry_type),
            mode=mode,
            limit=args.limit,
            **texts,
        )
        _print({"results": [result.to_dict() for result in results]})
    elif args.command == "note":
        if args.note_command == "read":
            _print(api.read_note(args.path_or_title).to_dict())
        else:
            _print(api.get_note(args.path_or_title).to_dict())
    elif args.command == "entry":
        _print(api.get_entry(args.entry_id).to_dict())
    elif args.command == "links":
        if args.link_command == "outgoing":
            links = api.get_outgoing_links(args.note_id, args.entry_id)
            _print({"links": [asdict(link) for link in links]})
        elif args.link_command == "backlinks":
            results = api.get_backlinks(args.note_id)
            _print({"results": [result.to_dict() for result in results]})
        else:
            _print(asdict(api.resolve_link(args.source_note_id, args.target_text)))
    elif args.command == "search-within":
        results = api.search_within(
            args.note_id,
            args.query,
            RetrievalMode(args.mode),
            args.section,
            limit=args.limit,
        )
        _print({"results": [result.to_dict() for result in results]})
    elif args.command == "context":
        _print(
            api.expand_context(
                args.entry_id,
                link_depth=args.link_depth,
                nearby_days=args.nearby_days,
                max_notes=args.max_notes,
            ).to_dict()
        )
    elif args.command == "warnings":
        warnings = api.get_index_warnings(
            WarningType(args.type) if args.type else None,
            args.path,
        )
        _print({"warnings": [asdict(warning) for warning in warnings]})
    elif args.command == "kanban":
        if args.kanban_command == "list":
            _print({"boards": [board.to_dict() for board in api.list_kanban_boards()]})
        elif args.kanban_command == "get":
            _print(api.get_kanban_board(args.board).to_dict())
        else:
            results = api.find_kanban_cards(
                board=args.board,
                columns=tuple(args.column),
                statuses=tuple(args.status),
                checked=_optional_bool(args.checked),
                tags=tuple(args.tag),
                date_from=args.date_from,
                date_to=args.date_to,
                exact_text=args.query if args.exact else None,
                semantic_text=args.query if args.query and not args.exact else None,
                limit=args.limit,
            )
            _print({"results": [result.to_dict() for result in results]})
    else:
        _print(api.export_graph().to_dict())
    return 0


def _ask(args: argparse.Namespace, settings: Settings, api: Bedrock) -> int:
    # Imported here, not at module scope: pulling in the agent stack costs
    # more than a second, and every other command would pay it for nothing.
    from pydantic_ai.exceptions import UsageLimitExceeded

    from bedrock.agent import DEFAULT_LIMITS, Sources, build_agent

    if args.model:
        settings = replace(settings, llm_model=args.model)

    # Check the index before the model. A model that searches an unready index
    # finds nothing and reports, confidently, that the notes say nothing. That
    # failure is worse than refusing, because it looks like an answer.
    report = health_report(settings)
    if not report["ready"]:
        _print(
            {
                "error": {
                    "message": (
                        "the vault index is not ready to answer questions; "
                        "run `bedrock diagnose` for the failing dependency"
                    ),
                    "type": "IndexUnavailable",
                },
                "health": report,
            },
            file=sys.stderr,
        )
        return 1

    status = llm_health(settings)
    if not status["ready"]:
        _print(
            {
                "error": {
                    "message": (
                        f"LM Studio is not serving {settings.llm_model!r} at "
                        f"{settings.llm_base_url}. Start LM Studio, load the model, "
                        "and enable Serve on Local Network."
                    ),
                    "type": "LLMUnavailable",
                },
                "llm": status,
            },
            file=sys.stderr,
        )
        return 1

    # One ledger per question. The model cites `s1`, `s2`; the ledger is what
    # turns those back into a note, a heading, and a line range for the reader.
    sources = Sources(api)
    agent = build_agent(settings)
    result = agent.run_sync(args.question, deps=sources, usage_limits=DEFAULT_LIMITS)
    citation_check = sources.citation_check(result.output)
    if not citation_check.valid:
        original = result
        try:
            result = agent.run_sync(
                "Rewrite your immediately previous answer so it follows the citation rules. "
                f"Validation errors: {'; '.join(citation_check.errors)}. "
                "Keep the answer grounded in the tool results already shown; do not invent sources.",
                message_history=result.all_messages(),
                deps=sources,
                usage_limits=DEFAULT_LIMITS,
                usage=result.usage,
            )
        except UsageLimitExceeded:
            result = original
        citation_check = sources.citation_check(result.output)
    _print(
        {
            "answer": result.output,
            "citation_errors": list(citation_check.errors),
            "citation_valid": citation_check.valid,
            "citations": sources.cited(result.output),
            "model": settings.llm_model,
            "question": args.question,
            "tool_calls": _tool_calls(result),
            "usage": _usage(result),
        }
    )
    return 0


def _tool_calls(result: object) -> list[dict[str, object]]:
    """List what the model actually looked at, in order.

    A local model's answer is only as good as the evidence it retrieved, so
    the trace is part of the result rather than a debugging extra.
    """
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    return [
        {"arguments": part.args_as_dict(), "tool": part.tool_name}
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]


def _usage(result: object) -> dict[str, int | None]:
    usage = result.usage
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "requests": usage.requests,
        "tool_calls": usage.tool_calls,
    }


def _query_texts(mode: RetrievalMode, query: str | None) -> dict[str, str | None]:
    fields = QUERY_TEXT_FIELDS[mode]
    if not fields:
        return {}
    if not query:
        raise ValueError(f"{mode.value} mode requires query text")
    # `find_entries` calls the literal field `exact_text`; a plan calls it
    # `literal_text`. Every other name is shared.
    return {("exact_text" if field == "literal_text" else field): query for field in fields}


def _optional_bool(value: str | None) -> bool | None:
    return None if value is None else value == "true"


def _values(enum: Any) -> tuple[str, ...]:
    return tuple(item.value for item in enum)


if __name__ == "__main__":
    sys.exit(main())
