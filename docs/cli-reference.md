# Bedrock CLI and Library Reference

Bedrock commands run inside the application container. Examples in this document
use the complete invocation:

```bash
docker compose exec bedrock bedrock COMMAND
```

The container runs the working tree, which `compose.yaml` binds at `/app/src`,
so a command reflects an edit without a rebuild. `bedrock serve` and
`bedrock watch` and `bedrock mcp` are already running and keep their own code
until `docker compose restart bedrock watch mcp`.

All normal command results are deterministic, pretty-printed JSON. Property
names are sorted so unchanged commands are easy to diff.

## Output and exit-code contract

| Exit code | Meaning |
| --- | --- |
| `0` | Command completed successfully. |
| `1` | A dependency, health, runtime, or index-readiness check failed. |
| `2` | The request was invalid, or an identifier was missing or ambiguous. |

Invalid operations write a JSON error to stderr:

```json
{
  "error": {
    "message": "note not found: Definitely-Missing",
    "type": "LookupError"
  }
}
```

`bedrock serve` is a long-running HTTP service and `bedrock watch` is a
long-running event stream. Help text from `--help` is intentionally human-readable
rather than JSON.

## Shared response objects

### Search result

Search, note, entry, backlink, context, and Kanban commands use this evidence
object:

```json
{
  "entry_id": "uuid",
  "note_id": "uuid",
  "path": "projects/News Resolution.md",
  "note_title": "News Resolution",
  "note_type": "standard",
  "entry_type": "project_update",
  "mode": "lexical",
  "text": "Original searchable text",
  "heading": "2026-08-19",
  "heading_path": ["News Resolution", "2026-08-19"],
  "start_line": 3,
  "end_line": 4,
  "note_date": null,
  "entry_date": "2026-08-19",
  "tags": [],
  "outgoing_links": [],
  "score": 7.3312263,
  "verified": null,
  "board": null
}
```

Important fields:

- `note_type`: `log`, `standard`, or `kanban`.
- `entry_type`: `daily_section`, `project_update`, `freeform_section`,
  `freeform_chunk`, or `kanban_card`.
- `score`: populated by lexical, semantic, and hybrid retrieval.
- `verified`: `true` for a literally verified result; otherwise `null`.
- `note_date` and `entry_date` remain distinct. Optional `entry_date_end`
  records the inclusive end of a compacted project-log interval. Date filters
  include intervals overlapping the requested dates.
- Dates are returned as plain calendar dates. The stored Qdrant payload
  holds them as UTC datetimes, so a raw payload reads
  `2026-08-19T00:00:00Z` where this object reads `2026-08-19`.
- `start_line` and `end_line` are inclusive source lines.

### Internal link

```json
{
  "target_text": "News Resolution",
  "target_note_id": "uuid-or-null",
  "target_heading": "2026-08-19",
  "alias": "project update",
  "resolution": "resolved",
  "line": 18
}
```

`resolution` is `resolved`, `missing`, or `ambiguous`.

### Kanban data

Kanban cards include:

```json
{
  "column": "ToDo",
  "status": "todo",
  "column_position": 0,
  "card_position": 1,
  "checked": false
}
```

Canonical statuses are `todo`, `doing`, and `complete`. A custom column can
have a `null` status.

## Service and diagnostics

### `mcp`

```bash
docker compose exec -T bedrock bedrock mcp
# Or the Compose mcp service at http://127.0.0.1:8000/mcp
```

Runs the read-only Bedrock MCP server. Defaults to stdio; use
`--transport streamable-http`, `--host`, and `--port` for HTTP.
See [mcp.md](mcp.md) for the version-one tool contracts, client setup,
pagination, and availability behavior.

### `serve`

```bash
docker compose exec bedrock bedrock serve
```

Starts the long-running health and graph snapshot HTTP server. Compose already
uses this as the application container's default process and publishes it at
`http://localhost:8080` (loopback only). After code changes, run
`docker compose restart bedrock` to load the updated routes.

- `GET /health`
- `GET /healthz`
- HTTP `200` when ready
- HTTP `503` when any required component is not ready

Graph routes and the `GET /events` notification stream are described below.

### `health`

```bash
docker compose exec bedrock bedrock health
```

Checks configuration, Qdrant connectivity, active-index compatibility, pinned
model readiness, the vault mount, and LM Studio.

The `llm` block reports the answering model and is marked `"required": false`.
It never changes the top-level `ready` flag and never changes the exit code. A
closed LM Studio is a normal state: parsing, indexing, and the watcher run
without it, so the index never competes for VRAM.

The container liveness endpoint at `/healthz` omits the `llm` block entirely.
Container health must not depend on a desktop application, or spend its timeout
budget waiting for one.

```json
{
  "configuration": {"errors": [], "ready": true},
  "llm": {
    "available_models": ["google/gemma-4-e4b"],
    "base_url": "http://host.docker.internal:1234/v1",
    "model": "google/gemma-4-e4b",
    "ready": true,
    "required": false
  },
  "index": {
    "collection": "bedrock_entries",
    "errors": [],
    "physical_collection": "bedrock_entries__build_...",
    "points": 38,
    "ready": true
  },
  "models": {"ready": true},
  "qdrant": {"ready": true, "status": 200},
  "ready": true,
  "vault": {"path": "/vault", "read_only_expected": true, "ready": true}
}
```

### `diagnose`

```bash
docker compose exec bedrock bedrock diagnose
```

Returns both the complete health report and focused index inspection:

```json
{
  "health": {"ready": true},
  "index": {"ready": true, "points": 38}
}
```

### `parse [PATH ...]`

```bash
docker compose exec bedrock bedrock parse
docker compose exec bedrock bedrock parse logs/2026-08-20.md
```

Parses all Markdown or only the supplied vault-relative paths without changing
Qdrant. It returns storage-independent parser contracts:

```json
{
  "notes": [
    {
      "note_id": "uuid",
      "path": "logs/2026-08-20.md",
      "title": "2026-08-20",
      "note_type": "log",
      "content_hash": "sha256",
      "entries": [],
      "warnings": [],
      "metadata": {}
    }
  ]
}
```

This is the raw parser output. Vault-wide link resolution occurs during index
operations and warning inspection.

### `watch`

```bash
docker compose exec bedrock bedrock watch
```

Watches recursive Markdown changes and incrementally indexes each debounced
path batch. It emits one incremental report per batch and runs until interrupted.

## Models and index lifecycle

### `models prefetch`

```bash
docker compose run --rm model-prefetch
# Equivalent application command:
docker compose exec bedrock bedrock models prefetch
```

Downloads and validates the pinned dense and sparse models, then writes the
versioned readiness marker to the persistent model cache.

### `models ready`

```bash
docker compose exec bedrock bedrock models ready
```

Reports whether the cache matches the configured models and whether dense
vectors have 384 dimensions.

### `index init`

```bash
docker compose exec bedrock bedrock index init
```

Creates an empty schema-compatible physical collection and active alias only
when no active alias exists. It is idempotent and does not parse or index notes.

```json
{
  "collection": "bedrock_entries",
  "created": false,
  "physical_collection": "bedrock_entries__build_..."
}
```

### `index rebuild`

```bash
docker compose exec bedrock bedrock index rebuild
```

Parses and resolves the entire vault, generates embeddings, validates a new
physical collection, and atomically switches the active alias.

```json
{
  "collection": "bedrock_entries",
  "notes": 16,
  "physical_collection": "bedrock_entries__build_...",
  "points": 38,
  "previous_collection": "bedrock_entries__build_...",
  "warnings": 7
}
```

### `index update [PATH ...]`

```bash
docker compose exec bedrock bedrock index update
docker compose exec bedrock bedrock index update projects/News\ Resolution.md
```

Converges the active index incrementally. Optional paths document the triggering
change batch; global link resolution still ensures the result equals a rebuild.

```json
{
  "collection": "bedrock_entries",
  "requested_paths": [],
  "notes": 16,
  "inserted": 0,
  "updated": 0,
  "deleted": 0,
  "skipped": 38,
  "embeddings_generated": 0,
  "embeddings_reused": 0,
  "warnings": 7
}
```

### `index inspect`

```bash
docker compose exec bedrock bedrock index inspect
```

Validates the alias, dense and sparse vector configuration, payload indexes,
and indexed payload schema version.

## Entry search

### `search [QUERY]`

```text
bedrock search [QUERY]
  [--mode metadata|literal|lexical|semantic|hybrid]
  [--date-from YYYY-MM-DD] [--date-to YYYY-MM-DD]
  [--tag-all TAG]... [--tag-any TAG]...
  [--note PATH_OR_TITLE] [--section HEADING]
  [--entry-type TYPE]... [--limit N]
```

Examples:

```bash
# Exact, case-insensitive, Unicode-normalized phrase verification
docker compose exec bedrock bedrock search "fried chicken" --mode literal

# BM25 lexical retrieval
docker compose exec bedrock bedrock search "pipeline clustering" --mode lexical

# MiniLM semantic retrieval
docker compose exec bedrock bedrock search "food going bad" --mode semantic

# Dense + BM25 retrieval fused with RRF
docker compose exec bedrock bedrock search "project status flow" --mode hybrid

# Metadata-only filtering; query text is omitted
docker compose exec bedrock bedrock search --mode metadata \
  --date-from 2026-08-17 --date-to 2026-08-20 \
  --tag-any journal --tag-any work --limit 20

# Every repeated --tag-all value is required
docker compose exec bedrock bedrock search --mode metadata \
  --tag-all journal --tag-all work

# Constrain retrieval to a note and entry type
docker compose exec bedrock bedrock search pipeline --mode lexical \
  --note "News Resolution" --entry-type project_update
```

The result envelope is:

```json
{
  "results": [
    {"entry_id": "uuid", "path": "note.md", "mode": "hybrid"}
  ]
}
```

The default mode is `hybrid`, so query text is required unless `--mode
metadata` is selected. Hard filters apply before vector retrieval.

### `search-within NOTE_ID QUERY`

```bash
docker compose exec bedrock bedrock search-within NOTE_ID pipeline \
  --mode lexical --section 2026-08-19 --limit 3
```

Supports the same five modes while applying a mandatory note-ID filter and an
optional heading filter. Returns `{"results": [SearchResult, ...]}`.

## Note and entry lookup

### `note get PATH_OR_TITLE`

```bash
docker compose exec bedrock bedrock note get "News Resolution"
docker compose exec bedrock bedrock note get projects/News\ Resolution.md
```

Accepts a vault-relative path, extensionless path, exact note title, or filename
stem. Ambiguous titles or stems return exit code `2` rather than guessing.

```json
{
  "note_id": "uuid",
  "path": "projects/News Resolution.md",
  "title": "News Resolution",
  "note_type": "standard",
  "entries": []
}
```

### `ask QUESTION`

```bash
docker compose exec bedrock bedrock ask "what did I work on last week?"
docker compose exec bedrock bedrock ask "..." --model lfm2.5-8b-a1b
```

Answers a question from the vault. The model reaches the notes only through
six tools in `bedrock.agent`: search, note listing, whole-note read, context
expansion, backlinks, and Kanban cards. It cannot write, and it cannot export
the graph. Collection tools use bounded `items` / `has_more` / `next_offset`
pages. Query-less note search browses explicitly dated entries newest-first,
then undated entries; Kanban date filters match only explicit card annotations.

The model does not write citations. Every result a tool returns carries a
short `id`, `s1`, `s2`, `s3`, and the model cites the id. The answer therefore
reads `You added an llm based parsing model. [s2]`, and the `citations` object
resolves each cited id to the note, heading, and line range behind it:

```json
{
  "answer": "You added an llm based parsing model. [s2]",
  "citations": {
    "s2": {
      "note": "News Resolution",
      "path": "projects/News Resolution.md",
      "heading": "2026-08-17",
      "lines": "6-9"
    }
  }
}
```

Ids are issued per question and count from `s1` each time; they mean nothing
outside the answer built from them. One entry reached twice keeps one id. A
note read whole through `read_note` gets an id too, and resolves to the note
without a line range, because there is no one range to claim. An id the run
never issued resolves to `{"unknown": true}`, so an answer citing evidence
that does not exist does not read like one citing evidence that does.

Each citation is one bracketed id, such as `[s2]`; several sources use
`[s2][s5]`, not `[s2, s5]`. Bedrock checks that cited ids were issued and that
answers using evidence contain valid citations. It makes one correction pass
inside the same request/tool-call budget. The output always includes
`citation_valid` and `citation_errors`; an exhausted correction returns the
answer with `citation_valid: false` rather than silently treating it as
grounded.

`--model` overrides `BEDROCK_LLM_MODEL` for one question, which is how to
compare two loaded models without editing the environment.

The command checks the index first, then LM Studio, before it builds the agent.
An unready index returns exit code `1` rather than letting the model search it,
find nothing, and report that the notes say nothing. An unreachable LM Studio,
or one serving a different model, also returns exit code `1` and says what to
fix rather than failing part-way through a run.

One answer waits `BEDROCK_LLM_TIMEOUT` seconds, 120 by default. The OpenAI
client would otherwise wait 600 seconds and retry twice, which reads as a hung
command.

`BEDROCK_LLM_TEMPERATURE` sets the sampling temperature, 0.0 to 2.0, and 0.0 by
default. The value goes out with every request, so it decides the sampling
rather than any preset held by the LM Studio server. Raising it loosens the
citation and grounding rules the instructions depend on, so raise it only to
compare answers, never to make them better.

Temperature 0.0 makes an answer repeatable, not deterministic. A
mixture-of-experts model served by LM Studio was measured giving two different
answers over six runs of one question, on a byte-identical prompt. Do not build
anything on two runs agreeing.

`tool_calls` lists what the model actually looked at, in order. An answer is
only as good as the evidence behind it, so the trace is part of the result
rather than a debugging extra. Pipe the answer alone with `jq -r .answer`.

```json
{
  "answer": "You traced the pipeline flow on 2026-08-17 ...",
  "model": "google/gemma-4-e4b",
  "question": "what did I work on last week?",
  "tool_calls": [
    {"arguments": {"query": "pipeline", "limit": 5}, "tool": "search_notes"}
  ],
  "usage": {
    "input_tokens": 2841,
    "output_tokens": 173,
    "requests": 3,
    "tool_calls": 1
  }
}
```

A run that spins without answering stops at the usage limits and returns exit
code `1`.

### `note read PATH_OR_TITLE`

```bash
docker compose exec bedrock bedrock note read "News Resolution"
```

Returns the complete Markdown file, including frontmatter, code, and URLs.
Accepts the same identifiers as `note get`, and reports the same exit code `2`
for an ambiguous or missing note.

Use `note get` to see how a note was indexed, and `note read` to see what the
note actually says. Indexed entries carry cleaned prose split to fit the dense
model; this command carries the file.

```json
{
  "content": "# News Resolution\n",
  "note_id": "uuid",
  "note_type": "standard",
  "path": "projects/News Resolution.md",
  "title": "News Resolution"
}
```

### `entry get ENTRY_ID`

```bash
docker compose exec bedrock bedrock entry get ENTRY_ID
```

Returns one `SearchResult` object or a JSON `LookupError` with exit code `2`.

## Links and context

### `links outgoing NOTE_ID [--entry-id ENTRY_ID]`

```bash
docker compose exec bedrock bedrock links outgoing NOTE_ID
docker compose exec bedrock bedrock links outgoing NOTE_ID --entry-id ENTRY_ID
```

Returns every outgoing link for the note, or only links from the selected entry:

```json
{"links": [{"target_text": "Target", "resolution": "resolved"}]}
```

### `links backlinks NOTE_ID`

```bash
docker compose exec bedrock bedrock links backlinks NOTE_ID
```

Returns entry-level `SearchResult` evidence for every point whose resolved
outgoing links contain the target note ID.

### `links resolve SOURCE_NOTE_ID TARGET_TEXT`

```bash
docker compose exec bedrock bedrock links resolve SOURCE_NOTE_ID \
  "News Resolution#2026-08-19"
docker compose exec bedrock bedrock links resolve SOURCE_NOTE_ID \
  "[[News Resolution#2026-08-19|project update]]"
```

Resolves an arbitrary target against the current vault catalog and returns one
`Link`. Missing and ambiguous targets are returned as explicit resolutions;
they are not guessed.

### `context ENTRY_ID`

```bash
docker compose exec bedrock bedrock context ENTRY_ID \
  --link-depth 1 --nearby-days 3 --max-notes 5
```

Constraints:

- Default depth `1`; maximum depth `2`.
- Default nearby-history window `±3` days.
- Default and maximum linked-note budget `5`.
- Cycles terminate deterministically.

```json
{
  "source_entry_id": "uuid",
  "inspected_note_ids": ["uuid"],
  "diagnostics": [],
  "items": [
    {
      "depth": 0,
      "labels": ["direct_evidence", "inference_input"],
      "result": {},
      "via_note_id": null
    },
    {
      "depth": 1,
      "labels": ["followed_link", "inference_input"],
      "result": {},
      "via_note_id": "uuid"
    }
  ]
}
```

Possible relationship labels are `direct_evidence`, `followed_link`,
`backlink`, and `nearby_history`. Related context is never relabeled as direct
evidence. Traversal diagnostics contain `source_note_id`, `target_text`,
`resolution`, and `line` for missing or ambiguous links encountered.

## Warnings

### `warnings [--type TYPE] [--path PATH]`

```bash
docker compose exec bedrock bedrock warnings
docker compose exec bedrock bedrock warnings --type missing_link
docker compose exec bedrock bedrock warnings --path freeform/Reference.md
```

Warning types:

- `invalid_date`
- `unknown_tag`
- `ambiguous_link`
- `missing_link`
- `invalid_kanban_settings`
- `missing_kanban_marker`
- `truncated_embedding_input`
- `parser_failure`

```json
{
  "warnings": [
    {
      "kind": "missing_link",
      "path": "freeform/Reference.md",
      "message": "missing internal link: [[Does Not Exist]]",
      "line": 5
    }
  ]
}
```

Warnings are recomputed from the current read-only vault with vault-wide link
resolution; this command does not mutate the index.

## Kanban

### `kanban list`

```bash
docker compose exec bedrock bedrock kanban list
```

Returns `{"boards": [{"title": "Board", "cards": []}]}`. Each board has `note_id`, `path`,
`title`, ordered `columns`, and ordered `cards`.

### `kanban get BOARD_OR_PATH`

```bash
docker compose exec bedrock bedrock kanban get "Kitchen App"
docker compose exec bedrock bedrock kanban get kanban/Kitchen\ App.md
```

Returns one complete `KanbanBoard`. Missing or ambiguous boards exit `2`.

### `kanban find [QUERY]`

```text
bedrock kanban find [QUERY]
  [--board BOARD] [--column COLUMN]... [--status STATUS]...
  [--checked true|false] [--tag TAG]... [--date-from YYYY-MM-DD]
  [--date-to YYYY-MM-DD] [--archived true|false|all] [--exact] [--limit N]
```

Examples:

```bash
# All unchecked cards in one column
docker compose exec bedrock bedrock kanban find \
  --board "Kitchen App" --column ToDo --checked false

# Semantic card search
docker compose exec bedrock bedrock kanban find "food expiring" --checked false

# Literally verified card search
docker compose exec bedrock bedrock kanban find "shopping list" --exact

# Cards dated on one day, from their `@{YYYY-MM-DD}` annotation
docker compose exec bedrock bedrock kanban find --date-from 2026-06-06 --date-to 2026-06-06
```

Without query text, the command performs metadata lookup. With query text it
uses semantic retrieval unless `--exact` is set. Repeated columns and statuses
use any-match semantics; repeated tags are all required.

`--date-from` and `--date-to` filter on the card's `entry_date`, which comes
from the Obsidian Kanban date annotation. Undated cards never match a date
range. The same cards also appear in a plain `bedrock search --date-from ...`
next to daily sections and project updates.

`--archived` defaults to `false`; `true` selects archived cards and `all` includes both.

Lifecycle filters are also available as `--created-at-from/to`,
`--started-at-from/to`, `--completed-at-from/to`, and `--due-date-from/to`
(each pair represents two flags, e.g. `--due-date-from` and `--due-date-to`).
Timestamp ranges accept dates spanning whole UTC days or timezone-aware ISO
instants; due-date ranges accept calendar dates only. Missing fields do not match,
and all supplied ranges combine with AND. These match explicit
[`bedrock-dates` card fields](operations.md#kanban-lifecycle-dates), independently
of the legacy `--date-from` / `--date-to` annotations. The library methods
`find_kanban_cards` and `find_kanban_cards_page` expose matching underscore-named
arguments, e.g. `created_at_from` and `due_date_to`.

```bash
bedrock kanban find --started-at-from 2026-10-12 --due-date-to 2026-10-20
```

## Network graph

### HTTP snapshot and Cosmograph downloads

```bash
curl --fail http://localhost:8080/graph/snapshot -o graph.json
curl --fail http://localhost:8080/graph/cosmograph/points.json -o points.json
curl --fail http://localhost:8080/graph/cosmograph/links.json -o links.json
```

`GET /graph/snapshot` returns `schema_version`, `vault_id`, `nodes`, `edges`,
and `unresolved_links`. It reads the source vault without Qdrant or embedding
models and includes empty and isolated notes. Nodes have `id`, `type`, and
`label`; note nodes also have `note_id`, `path`, `folder`, `note_type`, and
`tags`. Node types are `note`, `tag`, and `missing`. Edges have `id`, `source`,
`target`, and `type` (`link` or `tag`). Directed wikilinks collapse repeated
references into an edge with `occurrences` (`line`, `heading`, `alias`). Each
note connects to its inline/frontmatter tag nodes, including nested and
unconfigured tags. Missing wikilinks produce placeholder nodes; ambiguous
targets appear in `unresolved_links`. Markdown-style links and attachments
are not included.

The service caches the snapshot at startup and rebuilds it after debounced
Watchdog Markdown create/modify/delete/move events or directory structural
changes. Requests do not scan or reparse files. The usual debounce is two
seconds, capped at ten seconds of continuous activity, plus build time.
Rebuild failures return JSON with HTTP 503 and retry through the watcher.
Configured directory exclusions apply.

`GET /graph/cosmograph` returns flat `points` and `links` arrays from the same
cache. The `/graph/cosmograph/points.json` and `/graph/cosmograph/links.json`
routes download these arrays individually. Import both in Cosmograph and map
point ID to `id`, label to `label`, and link endpoints to `source`/`target`.
Use `type`, `folder`, or `note_type` for point colors and `degree` for size.
Link `weight` counts wikilink occurrences (tag membership has weight 1).
Download both after edits settle to keep the tables consistent.

Every graph route supports conditional GET with a strong content-derived
`ETag`. Save the response's ETag and send it in `If-None-Match` on the next
request to the same route. Matching validators return HTTP 304 with ETag and
no response body; keep existing data and skip rerendering. Nonmatching
validators return HTTP 200, the current JSON, and its ETag. Weak validators,
comma-separated validator lists, and `*` are supported. Unavailable snapshots
still return HTTP 503, even with a matching validator. Each representation
has its own ETag, computed with its encoded JSON only when the cache rebuilds.
Identical JSON retains the same ETag across rebuilds and process restarts.
Responses use `Cache-Control: private, no-cache` so consumers revalidate before
using a privately cached response.

```bash
curl -i -H 'If-None-Match: "ETAG_FROM_PREVIOUS_RESPONSE"' \
  http://localhost:8080/graph/snapshot
```

### Live notifications (`GET /events`)

```bash
curl -N http://localhost:8080/events
```

Returns `text/event-stream` with named SSE events, an SSE `id` matching the
JSON `sequence`, a 15-second comment heartbeat, and a two-second reconnect
interval. Event payloads look like:

```text
id: 42
event: notes.changed
data: {"type":"notes.changed","paths":["projects/zenith.md"],"deleted_paths":[],"rescan":false,"sequence":42}

id: 43
event: graph.updated
data: {"type":"graph.updated","etag":"\"abc123\"","sequence":43}
```

`notes.changed` is published after filesystem debouncing, before graph
rebuilding. Paths are vault-relative Markdown paths; deletions include old
rename paths. Prose-only edits still notify. A structural change/startup
catch-up uses `rescan: true`, includes current discovered paths and known
deleted paths, and asks clients to refresh all relevant state.

`graph.updated` is published only after the rebuilt snapshot is available and
its `/graph/snapshot` ETag changes. Failed builds do not emit graph updates;
the existing watcher retries. Initial successful graph preparation can emit
an update. These events do not signal retrieval/index readiness. The other
graph representations have their own ETags; use conditional GET for the
representation the GUI renders.

This is a notification channel, not a durable event log. There is no replay
for `Last-Event-ID`, sequences reset with the process, and disconnected clients
may miss events. Revalidate the graph and refresh the open note on every
connection/reconnection. Each client has a bounded queue of 64 events; overflow
disconnects it so it can reconnect and resync. Disconnections clean up stream
subscriptions. Consumers use HTTP/MCP to read current content.

### `graph export`

```bash
docker compose exec -T bedrock bedrock graph export > graph.json
```

Exports the complete deterministic graph without context-expansion depth,
date-window, or note-count limits:

```json
{
  "nodes": [
    {
      "note_id": "uuid",
      "path": "note.md",
      "title": "Note",
      "note_type": "standard",
      "tags": ["work"],
      "dates": ["2026-08-20"]
    }
  ],
  "edges": [
    {
      "edge_id": "sha256",
      "source_note_id": "uuid",
      "target_note_id": "uuid",
      "edge_type": "internal_link",
      "source_entry_id": "uuid",
      "target_entry_id": null,
      "line": 4,
      "target_heading": null,
      "tags": [],
      "date": null,
      "date_field": null
    }
  ]
}
```

Edge types:

- `internal_link`: directed, one edge per resolved link occurrence.
- `shared_tag`: undirected relationship represented once in canonical order;
  `tags` contains every shared tag.
- `shared_date`: exact entry pairing for a common `note_date` or `entry_date`;
  `date` and `date_field` identify the evidence.

## Python library mapping

Every command below is a thin adapter over the public library, with one
exception: `ask` has no library method. The answering agent lives in
`bedrock.agent` and is built per question, beside the library rather than
inside it.

```python
from bedrock import Bedrock
from bedrock.core.config import Settings

api = Bedrock(Settings.from_env())
```

| Library method | CLI equivalent |
| --- | --- |
| `reindex(paths=None, full=False)` | `index update` / `index rebuild` |
| `find_entries(...)` | `search` |
| `get_note(path_or_title)` | `note get` |
| `read_note(path_or_title)` | `note read` |
| `get_entry(entry_id)` | `entry get` |
| `get_outgoing_links(note_id, entry_id=None)` | `links outgoing` |
| `get_backlinks(note_id)` | `links backlinks` |
| `resolve_link(source_note_id, target_text)` | `links resolve` |
| `search_within(note_id, query, mode, section=None)` | `search-within` |
| `expand_context(entry_id, ...)` | `context` |
| `get_index_warnings(kind=None, path=None)` | `warnings` |
| `list_kanban_boards()` | `kanban list` |
| `get_kanban_board(board_name_or_path)` | `kanban get` |
| `find_kanban_cards(...)` | `kanban find` |
| `export_graph()` | `graph export` |

Kanban mutation operations are available through `bedrock.kanban` in Python and
through MCP: `create_board`, `add_cards`, `edit_cards`, `move_cards`,
`archive_cards`, `delete_board`, plus the `get_task` reader. Card mutations
accept lists of 1–30 items in one board and return Pydantic result models.
There are no mutation CLI subcommands. See [the MCP editing contract](mcp.md#kanban-editing).

Library methods return frozen typed dataclasses or tuples of them. Aggregate
response contracts provide `to_dict()`; individual `Link` and `IndexWarning`
values are compatible with `dataclasses.asdict()`.
