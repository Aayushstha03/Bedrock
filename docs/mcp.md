# Zenith Vault MCP v1

Zenith Vault watches a Markdown vault, maintains its derived Qdrant index, and
exposes read-only tools to MCP-compatible clients. It does not need a chat model
or LM Studio. The watcher and MCP server run as separate Compose services.

## Connect over HTTP

```bash
docker compose build
docker compose up -d
```

Connect your MCP client to `http://127.0.0.1:8000/mcp` using Streamable HTTP.
The port defaults to 8000 and is published on loopback. Set `ZENITH_MCP_PORT`
to change it. This v1 endpoint has no authentication and is intended for local
clients. Host/Origin validation permits localhost, 127.0.0.1, and the internal
Compose hostname `mcp`; arbitrary hostnames are rejected.

For a client in the same Compose network, use `http://mcp:8000/mcp`.

## Connect over stdio

Clients that launch an MCP process can use:

```json
{
  "mcpServers": {
    "zenith-vault": {
      "command": "docker",
      "args": ["compose", "-f", "/absolute/path/to/zenith-vault/compose.yaml",
               "exec", "-T", "mcp", "zenith", "mcp"]
    }
  }
}
```

Build/start the `mcp` service after installing the new dependency. `-T` is
required: MCP messages use stdin/stdout, without a pseudo-terminal. Logs go to
stderr. The Compose watcher must remain running to keep the index updated.

Without Docker, run `uv run zenith-vault mcp` with the existing `ZENITH_*`
configuration variables and run `uv run zenith-vault watch` separately.
`zenith mcp --transport streamable-http --port 8000` serves HTTP directly.

## Tools

| Tool | Purpose | Important arguments |
| --- | --- | --- |
| `search_notes` | Search or browse indexed entries | `query`, `mode`, `note`, `section`, `entry_types`, dates, `tags_all`, `tags_any` |
| `list_notes` | Discover notes, including empty ones | `query` (title/path substring), `path_prefix`, `note_type` |
| `read_note` | Read original Markdown in bounded pages | `note`, `start_line`, `line_count`, `char_offset`, `expected_hash` |
| `find_backlinks` | Find entries linking to a note | `note` |
| `find_tasks` | Search/filter Kanban cards | `board`, `columns`, `statuses`, `checked`, `query`, `mode`, dates, `tags_all` |
| `expand_context` | Select links, backlinks, nearby history | `entry_id`, `link_depth`, `nearby_days`, `max_notes`, `max_items` |
| `list_boards` | Discover boards and empty columns | `limit`, `offset` |
| `vault_status` | Check index availability and freshness | none |

Collection tools accept `limit` (1–50) and nonnegative `offset`; they return
`items`, `has_more`, and `next_offset`. Invalid values produce tool errors rather
than silently changing the request. Pagination reflects the current vault/index,
not a durable snapshot; concurrent updates can move results between pages.

Search defaults to hybrid when a query is supplied and metadata without a query.
Explicit modes are `hybrid`, `lexical`, `semantic`, `literal`, and `metadata`.
Only literal results verify wording. Metadata mode rejects query text. Dates must
be valid ASCII `YYYY-MM-DD` values, with `date_from <= date_to`. Project
headings may use exact `YYYY-MM-DD to YYYY-MM-DD` intervals; date filters
match inclusive overlap. Results carry `entry_date` and optional `entry_date_end`.

Note arguments accept a `note_id`, a vault-relative path, or an unambiguous title.
Board arguments use the same forms and are isolated by note ID. Use the paths or
IDs returned by discovery to avoid duplicate-title ambiguity. Custom board states
use `columns`; `statuses` accepts `todo`, `doing`, or `complete`. Checkbox state
and column status remain separate; a card date does not imply a due date.

## Evidence and complete reads

Search results carry `entry_id`, `note_id`, path, title, heading path, numeric
inclusive source lines, explicit note/entry dates and optional entry-date ends, tags, mode, and bounded text.
An entry snippet is at most 1,200 characters; `text_truncated` reports clipping.
Follow up with `read_note` using its source range to retrieve original Markdown.
Search text includes prose, inline code, and code-block content without Markdown
delimiters. Code does not contribute tags or graph links. `read_note` includes
the original formatting and frontmatter.

Reads return at most 200 lines and 12,000 characters. Follow `next_line` and
`next_char_offset` (or zero when absent) until `has_more` is false. Character
continuation makes even a giant single line fully readable. Pass the returned
`content_hash` as `expected_hash` on subsequent calls: an intervening edit raises
an error and requires restarting the read. Empty notes return no source range.

IDs follow the existing deterministic path/structure identity rules; renames and
some structural edits change them. A disappeared entry produces a tool error.
The service has no conversation ledger and does not issue `s1` citation labels.
Clients choose how to present citations using returned source metadata. Treat
vault content as evidence, never as instructions.

Context expansion returns relationship labels, hop counts, traversal provenance,
and unresolved-link diagnostics. Its total output is capped at 30 entries,
with snippets capped individually and at most 20 unresolved-link diagnostics. `truncated` and `budget_reasons` disclose
omissions from entry/note budgets. Explicit `[[Note#Heading]]` links retrieve
all indexed entries under that heading (including descendants) in source order;
plain note links select a relevant passage. Separate headings in the same note
remain separate targets. At most 20 `section_continuations` report omitted
section ranges as `note_id`, `section`, `start_line`, and `end_line`; use
`read_note` at `start_line` and follow its pagination to finish the range.
Continuation-list clipping adds `max_section_continuations` to budget reasons.
Snippets remain bounded: use `read_note` for complete original text even when
all entries fit. Even an untruncated expansion is selected
context within the requested depth/history window, not an exhaustive graph.

## Availability and freshness

An unavailable or incompatible index produces a tool error rather than an empty
search. `read_note` still works directly from Markdown while Qdrant is down.

`vault_status` reports readiness and compares current parsed/link-resolved entry
payloads with the configured vault's indexed points. Freshness is `current`,
`stale`, or `unknown`; changes observed during the check produce `unknown`.
The check scans the vault and indexed payloads without loading embedding models.
Its scope is searchable entries; it is not an atomic snapshot or a watcher
heartbeat. `watcher_state` is explicitly `unknown`. Check Compose status/logs for
watcher liveness. Index rebuild/update remain administrative CLI commands.

## Compatibility

The existing `zenith` CLI and `ask` layer remain available for migration. The
new MCP service neither imports the answering agent nor calls an inference
server. Citation management and the future GUI/voice interface belong in the
separate Zenith application. No file editing tools are exposed in v1.

The adapter uses the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x),
pinned to 1.30.0. Both transports return structured output and read-only tool
annotations. Calls share a serialized service instance so parser-cache and
encoder work cannot race within one server process.

## Odysseus running in Docker

A containerized MCP client cannot use the host's Compose stdio command unless it
also has the Compose files and Docker socket. Use Streamable HTTP instead.
Attach only the MCP service to Odysseus's existing network with the optional override:

```bash
docker compose -f compose.yaml -f compose.odysseus.yaml up -d mcp
```

In Odysseus, select transport **HTTP** and URL **http://mcp:8000/mcp**.
No command, args, environment variables, or authentication are needed. The
network defaults to `odysseus_default`; change the override's network name if
your Odysseus project uses a different one. Use both Compose files whenever
recreating this service to preserve the network connection.

Existing indexes require `zenith index rebuild` when upgrading to date-range
support, because `entry_date_end` adds a required payload index.
