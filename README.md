# Bedrock

*Work in progress. I am building it as I go, and it is just getting started.*

## Why

I work on many projects at the same time. AI makes each one move fast. I test an
idea, build it, check the result against the last attempt, and make a decision.
Some decisions are right. Some are wrong. There are many of them, and they come
quickly.

So I take notes. I write a journal entry each day, a log for each experiment, a
card for each task, and a record of each decision and its reason. The notes work
well when I write them. They work badly when I must find them again. The
information is in the vault, but it takes a long time to open each file, read
each heading, and remember which day held the answer.

Bedrock is the tool for that second part. It makes the lookup fast, simple, and
one step. I ask a question in the words I remember, and Bedrock gives me the
entries I wrote.

It is also one tool for all of it. My journals, my Kanban boards, my project
updates, and my freeform logs are different shapes of Markdown, but they hold
one history. Bedrock reads all of them, indexes each entry the same way, and
searches them together.

## What it is

A local-first search index for a Markdown vault. The Markdown stays the source
of truth and is mounted read-only. Qdrant holds the derived index and nothing
else, so the whole index can be thrown away and rebuilt from the notes.

Bedrock cuts notes into entries — daily sections, dated project updates,
freeform sections, and Kanban cards — and indexes each one with a dense vector,
a BM25 sparse vector, and a payload carrying its path, heading path, dates, and
line range. Search it by metadata, exact phrase, keyword, meaning, or a hybrid
of the last two. Expose those entries through a read-only MCP server so any
MCP-compatible agent can search and read the vault. The indexer and watcher need no chat model.


## Start it

```bash
cp .env.example .env          # then set BEDROCK_VAULT_PATH to your vault
docker compose build
docker compose run --rm model-prefetch
docker compose up -d
```

`.env.example` is the template, and it documents every setting.
The default vault path points at the fixture vault, which is safe to index as-is.

### Vault graph snapshot

`bedrock serve` exposes `GET /graph/snapshot` on the configured HTTP host/port:

```bash
curl http://localhost:8080/graph/snapshot
```

The JSON contains `schema_version`, `vault_id`, `nodes`, `edges`, and
`unresolved_links`. Nodes have stable `id`, `type` (`note`, `tag`, or `missing`),
and `label` fields. Note nodes also carry `note_id`, `path`, `folder`,
`note_type`, and `tags`, useful for coloring and filtering a graph view.
Edges have `id`, `source`, `target`, and `type` (`link` or `tag`). Link edges
point from the referring note to its target; repeated wikilinks share one edge
with `occurrences` containing `line`, `heading`, and `alias`. Tag edges connect
each note to its tag nodes, avoiding a dense mesh of shared-tag note pairs.

The snapshot reads all discovered Markdown notes directly, including empty
and isolated notes, respects configured directory exclusions, and needs no
Qdrant or embedding models. Tags include inline and YAML frontmatter tags,
including unconfigured and nested tags; tag spelling is case-folded without
applying search-index tag aliases. Code examples are excluded. Wikilinks use
the existing vault-wide title/path/alias resolver. Missing targets get placeholder
nodes; ambiguous targets appear in `unresolved_links` without invented edges.
Markdown-style links, attachments, and inferred similarity/date edges are not
included. A missing or unreadable vault returns JSON with HTTP 503.
The HTTP service keeps a snapshot in memory, builds it at startup, and rebuilds
it only after the same debounced Watchdog Markdown create/modify/delete/move
events used by the index watcher. Requests reuse the cached value without
walking or reparsing the vault. Updates normally appear after the two-second
debounce (at most ten seconds of continuous activity, plus build time).
Directory creates, moves, and deletions also trigger full convergence, covering
structural changes that may not emit individual Markdown events.
The HTTP service runs its own observer using that shared watcher layer because
the indexing watcher runs in a separate process. Failed rebuilds retry through
the watcher and return HTTP 503 until recovery rather than serving stale success.
The existing `bedrock graph export` remains the separate index-backed export.

For [Cosmograph](https://run.cosmograph.app/), download the two flat JSON tables:

- Points: `http://localhost:8080/graph/cosmograph/points.json`
- Links: `http://localhost:8080/graph/cosmograph/links.json`
- Combined JSON (`points` and `links` arrays): `http://localhost:8080/graph/cosmograph`

Add both files in Cosmograph's Data Sources. Map point ID to `id`, point label
to `label`, and link endpoints to `source` and `target`. Optionally color points
by `type`, `folder`, or `note_type`, and size by `degree`. Links carry `type`
and `weight` (wikilink occurrence count, or 1 for tag membership). Point IDs
match the original snapshot; isolated notes remain in the points table.
These tables rebuild alongside the cached snapshot on watcher events. The two
file endpoints return attachment headers for browser downloads. Download both
after edits have settled so they describe the same vault state.

Build the index, then search it:

```bash
docker compose exec bedrock bedrock index rebuild
docker compose exec bedrock bedrock search "pipeline clustering" --mode hybrid
```

The watcher runs as its own service and reindexes each saved file, so the index
follows the vault without another command.

The MCP endpoint is **http://127.0.0.1:8000/mcp** (Streamable HTTP).
See [docs/mcp.md](docs/mcp.md) for client configuration, the eight tools, and
complete note reading. Citation presentation belongs to the consuming agent.

The legacy `bedrock ask` command needs LM Studio running on the host with
"Serve on Local Network" enabled. Nothing else does. Parsing, indexing, and the watcher all run with LM
Studio closed, so the index never competes with a chat model for VRAM.

The Qdrant dashboard is at <http://localhost:6333/dashboard>.

## Vault conventions

- **Kanban boards:** keep all boards under the configured top-level
  `BEDROCK_KANBAN_ROOT` folder (default `kanban`). Classification uses the folder,
  not the `kanban-plugin: board` property. Boards elsewhere are treated as
  ordinary notes; keep ordinary notes out of the board folder.
- **Tags and properties:** tag filters use the curated `BEDROCK_KNOWN_TAGS` set
  and configured aliases from inline prose tags. Frontmatter tags and arbitrary
  properties are not search filters. Store meaningful information in the body.
- **Dates:** use exact `YYYY-MM-DD` filenames for daily notes. Inside the
  configured logs folder, an exact date on the first nonempty body line is also
  accepted. Dated headings in ordinary notes must contain only `YYYY-MM-DD`;
  child sections inherit their date. For a compacted project log, use a heading
  containing exactly `YYYY-MM-DD to YYYY-MM-DD`, e.g.
  `### 2026-10-01 to 2026-10-02`; endpoints are inclusive and date filters
  match any overlapping day. Results expose the start as `entry_date` and the
  end as `entry_date_end`. Dates mentioned in prose or properties
  and filesystem timestamps do not establish chronology.
- **Kanban state:** checkbox state and column status are separate. `todo`,
  `to do`, and `backlog` map to `todo`; `doing`, `in progress`, and `active` map
  to `doing`; `complete`, `completed`, and `done` map to `complete`, ignoring
  case. Filter custom states by their actual column names from `list_boards`.
- **Kanban dates:** card dates use `@{YYYY-MM-DD}` or `@[[YYYY-MM-DD]]`, and
  times use `@@{HH:MM}` or `@@[[HH:MM]]`, unless board settings override the
  triggers. A card exposes one date and one time; the date is not automatically
  interpreted as a due date. Multiple distinct dates produce a warning and
  retain the first date.
- **Links:** use `[[Note]]` or `[[Note#Heading]]` (including a vault-relative
  note path such as `[[projects/Note.md#Heading]]`). Heading names must match
  the stored heading exactly. Explicit heading links expand every indexed
  entry in the section, including child headings, within the output budget.
  `section_continuations` identifies omitted source lines for `read_note`.
  Block references and same-note heading links
  are not supported by graph traversal.
- **Technical snippets:** inline code and code blocks are searchable alongside
  prose. Tags and wikilinks inside code remain literal text rather than tags
  or graph connections. Original Markdown is available through `read_note`.

After upgrading to date-range support, rebuild the existing index to create
the new date-end payload index and refresh searchable entries:

```bash
docker compose restart bedrock watch mcp
docker compose exec bedrock bedrock index rebuild
```

## Working on it

### Project rename

Bedrock uses the `bedrock` package/CLI, `BEDROCK_*` settings, and the `bedrock`
Compose project and application service. Earlier names are not supported.
The local `.env` must use the new setting names. Deterministic IDs also use the
Bedrock namespace; clients should discover new IDs after rebuilding the index.

Stop the previous Compose deployment before starting Bedrock to release its
ports. The renamed Compose project creates new Qdrant/model-cache volumes;
previous volumes are retained, but are not automatically reused. Follow the
startup instructions to build the image, prefetch models, and rebuild the index.
An existing HTTP MCP client can keep its endpoint when the configured port is
unchanged; stdio clients must use the `bedrock` command and updated repo path.

### Development

```bash
uv sync --dev
uv run pytest
```

Compose binds `./src` over the image, and `PYTHONPATH` puts the mount first, so
the containers run the working tree. Edit a file and the next
`docker compose exec` picks it up with no rebuild. The two long-running
processes hold their own code:

```bash
docker compose restart bedrock watch mcp
```

Rebuild the image only when a dependency changes.

## Documentation

- [docs/mcp.md](docs/mcp.md) — the Bedrock v1 tool contract and MCP setup.
- [docs/design.md](docs/design.md) — how Bedrock works, and why. The decisions,
  the indexing pipeline, the retrieval modes, the invariants, and the known
  limits.
- [docs/cli-reference.md](docs/cli-reference.md) — every command, option,
  response shape, exit code, and Python API method.
