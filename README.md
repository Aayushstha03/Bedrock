# Bedrock

A local-first Markdown knowledge layer for personal agents. Bedrock makes
journals, project logs, freeform notes, and Kanban cards searchable together,
with source locations that let you inspect the evidence behind each result.

Built as the vault layer for **Zenith**, Bedrock can also run independently
through its CLI, Python API, or MCP server. Markdown remains the
source of truth; Qdrant holds a rebuildable derived index. Parsing, indexing,
search, and the watcher need no chat model.

*Work in progress. The knowledge layer is usable today; Zenith's wider
assistant, GUI, and voice experience are separate work.*

For the full technical breakdown, read the [architecture and design](docs/design.md).
For integration details, use the [MCP guide](docs/mcp.md) or
[CLI/API reference](docs/cli-reference.md).

## Quickstart

Requires Docker with Docker Compose. Run these commands from the repository root:

```bash
cp .env.example .env
```

Set `BEDROCK_VAULT_PATH` in `.env` to the host path of your Markdown vault.
The default points to the repository's fixture vault, which you can use to try
Bedrock first. [.env.example](.env.example) documents the other settings.

```bash
docker compose build
docker compose run --rm model-prefetch
docker compose up -d
docker compose exec bedrock bedrock index rebuild
docker compose exec bedrock bedrock search "pipeline clustering" --mode hybrid
```

Model prefetching requires network access. After models are cached, normal
indexing and search can run offline. The vault is mounted read-only with a writable overlay for its configured Kanban folder.
Set `BEDROCK_VAULT_USER` to the vault owner's UID:GID (`id -u` / `id -g`). Persistent
volumes hold Qdrant data and the model cache.

The `watch` service keeps the index current after saves and catches up on
startup. The explicit rebuild above establishes a fresh index for first use.

| Local endpoint | Purpose |
| --- | --- |
| `http://127.0.0.1:8000/mcp` | Read-only MCP over Streamable HTTP |
| `http://localhost:8080/healthz` | Application health |
| `http://localhost:8080/graph/snapshot` | Cached source-backed vault graph |
| `http://localhost:8080/events` | Live note/graph notifications over SSE |
| `http://localhost:6333/dashboard` | Qdrant dashboard with default configuration |

MCP and Qdrant addresses/ports are configurable. The MCP endpoint has no
authentication and is intended for local clients; it is published on loopback
by default. See [client setup and transport details](docs/mcp.md).

## Usage

### Search and inspect sources

Commands return JSON unless otherwise documented.

```bash
# Combine keyword and semantic retrieval.
docker compose exec bedrock bedrock search "why the pipeline changed" --mode hybrid

# Verify an exact phrase, ignoring case and normalizing Unicode.
docker compose exec bedrock bedrock search "pipeline clustering" --mode literal

# Browse entries with explicit dates, including overlapping date ranges.
docker compose exec bedrock bedrock search --mode metadata \
  --date-from 2026-10-01 --date-to 2026-10-14

# Read original Markdown using a vault-relative path or unambiguous title.
docker compose exec bedrock bedrock note read "projects/Ascending Project.md"
```

The note path in the last example exists in the default fixture vault; use your
own path when indexing another vault. Search results include note and entry IDs,
paths, heading paths, dates, and source line ranges. MCP excerpts disclose
clipping so clients can follow up with paginated original-source reads.

For links, context, filters, and every command, see the
[CLI and Python API reference](docs/cli-reference.md).

### Query Kanban cards

```bash
docker compose exec bedrock bedrock kanban list
docker compose exec bedrock bedrock kanban find --status doing --checked false
```

Boards live under the configured Kanban root. Column status, checkbox state,
and explicit card dates remain separate. See
[Kanban usage](docs/cli-reference.md#kanban) and
[vault conventions](docs/operations.md#vault-conventions).

### Connect an assistant

Configure an MCP-compatible client to use Streamable HTTP at
`http://127.0.0.1:8000/mcp`. Bedrock exposes fifteen tools for note search, discovery, original-source reads,
backlinks, tasks, context, boards, tags, and availability/freshness checks, plus
scoped board/task editing with revision checks and backups. See
[Kanban editing](docs/mcp.md#kanban-editing).

The [MCP guide](docs/mcp.md) covers HTTP and stdio configuration, tool arguments,
pagination, evidence handling, and containerized clients. Citation presentation
belongs to the consuming assistant.

The legacy `bedrock ask` command can answer through LM Studio running on the
host. It is optional; the MCP and indexing services do not depend on it.
See [the answering-layer contract](docs/design.md#10-legacy-answering-layer).

### Use the graph and live updates

```bash
curl --fail http://localhost:8080/graph/snapshot
curl -N http://localhost:8080/events
```

The HTTP snapshot exposes notes, tags, wikilinks, and missing-target
placeholders, including empty and isolated notes. Cosmograph downloads are
available too. Graph responses support ETags; SSE announces note changes and
changed graph snapshots. These events do not indicate search-index readiness.

A separate `bedrock graph export` command exposes index-backed link,
shared-tag, and shared-date relationships. See the
[graph and event reference](docs/cli-reference.md#network-graph) for schemas,
export commands, caching, reconnection, and visualization setup.

## Technical overview

```text
Read-only Markdown vault
  -> CommonMark parsing and structured entries
  -> vault-wide wikilink resolution
  -> local dense + sparse encoding
  -> Qdrant derived index
  -> filtered retrieval and bounded context
  -> CLI / Python API / MCP

Read-only Markdown vault
  -> HTTP observer and event bus
  -> cached graph snapshot / Cosmograph exports / SSE
```

| Capability | Implementation | Detailed explanation |
| --- | --- | --- |
| Structure-aware parsing | Daily sections, dated updates, freeform sections/chunks, and atomic Kanban cards; code remains searchable | [Entries and pipeline](docs/design.md#2-what-one-point-is) |
| Local hybrid retrieval | CPU FastEmbed, 384-dimensional MiniLM dense vectors, BM25 sparse vectors, and RRF fusion; five search modes | [Retrieval](docs/design.md#6-retrieval) |
| Source-linked evidence | Note identity, heading hierarchy, line ranges, bounded excerpts, and complete source reads | [Evidence and reads](docs/mcp.md#evidence-and-complete-reads) |
| Dependency-aware updates | Debounced watcher batches widen to affected link references; stale entries are removed | [Index updates](docs/design.md#5-two-write-modes) |
| Vector reuse and identity | Deterministic IDs and embedding fingerprints; renames can change IDs while allowing vector reuse | [Identity](docs/design.md#3-names-decide-everything) and [fingerprints](docs/design.md#vectors-follow-the-fingerprint-not-the-id) |
| Safe rebuilds | Build and validate a replacement collection, then atomically switch the live alias | [Rebuild lifecycle](docs/design.md#5-two-write-modes) |
| Explicit chronology | Daily dates, dated headings, inclusive date ranges, and Kanban annotations; no inferred dates | [Date semantics](docs/design.md#7-dates) |
| Bounded related context | Outgoing links, backlinks, and nearby history with provenance and traversal budgets | [Context](docs/design.md#6-retrieval) |
| Live interface data | Source-backed graph cache, content-derived ETags, event bus, and ephemeral SSE | [Graph architecture](docs/design.md#9-live-graph-and-notifications) |
| Reusable agent tools | Fifteen MCP tools over HTTP or stdio, with scoped Kanban editing and no chat model dependency | [MCP contract](docs/mcp.md#tools) |

Bedrock deliberately keeps Qdrant as its only derived datastore and performs
embedding in-process on CPU. The parser, metadata rules, and source references
keep retrieval grounded in the files. The
[design document](docs/design.md) explains the decisions, algorithms, and limits.

## Boundaries and limitations

- Writes are confined to Kanban board/task tools. Other Markdown notes stay read-only.
  Moving across board files and deleting notes/boards are not supported.
- Search tag filters use configured inline tags and aliases. The source-backed
  graph includes broader inline/frontmatter tags; these are distinct contracts.
- Missing or ambiguous links are reported rather than guessed. Graph traversal
  does not support block references or same-note heading links.
- Long indivisible paragraphs and cards can exceed the dense model window;
  warnings disclose this. Excerpt clipping is a separate limit.
- Context expansion is selected, bounded evidence. Paginated tool results are
  not a durable snapshot; clients must handle concurrent edits.
- GUI, voice, and the wider research/assistant workflow belong to Zenith or
  another consuming application.

See [encoder limits](docs/design.md#8-entry-size-and-the-encoder-window),
[vault conventions](docs/operations.md#vault-conventions), and the
[MCP contract](docs/mcp.md) before integrating a client.

## Documentation

| Read this to… | Guide |
| --- | --- |
| Understand the architecture, algorithms, tradeoffs, and invariants | [How Bedrock works](docs/design.md) |
| Connect an agent and implement tool workflows | [MCP integration](docs/mcp.md) |
| Look up commands, JSON shapes, HTTP endpoints, and Python methods | [CLI and library reference](docs/cli-reference.md) |
| Configure notes, diagnose stale results, upgrade, or contribute | [Operations and development](docs/operations.md) |

## Development

Requires Python 3.13+ and uv:

```bash
uv sync --dev
uv run pytest
```

Compose mounts the working `src` tree. Restart `bedrock`, `watch`, and `mcp`
after code changes to reload their long-running processes; dependency changes
require an image rebuild. See [development details](docs/operations.md#development)
and [test coverage](docs/operations.md#what-the-tests-cover).
