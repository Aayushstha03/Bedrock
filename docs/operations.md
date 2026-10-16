# Operations and development

Use this guide to configure your vault, diagnose stale search results, upgrade
an existing deployment, and work on Bedrock. For first-time setup, start with
the [README quickstart](../README.md#quickstart). Command syntax and response
contracts live in the [CLI and library reference](cli-reference.md).

## Configuration

[.env.example](../.env.example) documents the settings used by Docker Compose.
Set `BEDROCK_VAULT_PATH` to your vault before starting services. Keep embedding
models in the persistent model-cache volume; prefetch them before normal use.
The vault is mounted read-only with a writable overlay only for the configured
Kanban folder in `bedrock` and `mcp`. The watcher remains read-only. Qdrant and
model-cache volumes are writable. Set `BEDROCK_VAULT_USER` to the vault owner's
UID:GID (check `id -u` and `id -g`) so the editing processes can write there.
Run model prefetch before serving; the prefetch service owns the model-cache
volume and the editing processes need read access to cached models.

`BEDROCK_EXCLUDED_DIRECTORIES` skips folder names anywhere in the vault.
`BEDROCK_EXCLUDED_FILES` skips comma-separated filenames anywhere in the vault
(e.g. `tags.md`). Both match case-insensitively. After changing exclusions,
recreate the services with `docker compose up -d`; the watcher startup pass
removes newly excluded notes from the index.

The local MCP endpoint has no authentication. Its default published address is
loopback; supported hosts and client setup are documented in the
[MCP guide](mcp.md#connect-over-http).

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
  `to do`, and `backlog` map to `todo`; `doing`, `in progress`, `on progress`,
  `working`, and `active` map
  to `doing`; `complete`, `completed`, and `done` map to `complete`, ignoring
  case. Filter custom states by their actual column names from `list_boards`.
- **Kanban dates:** card dates use `@{YYYY-MM-DD}` or `@[[YYYY-MM-DD]]`, and
  times use `@@{HH:MM}` or `@@[[HH:MM]]`, unless board settings override the
  triggers. These legacy annotations expose one generic date and one time;
  the date is not automatically interpreted as a due date. Multiple distinct
  dates produce a warning and retain the first date. Separate structured
  lifecycle fields are described under [Kanban lifecycle dates](#kanban-lifecycle-dates).
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

## Checking availability and freshness

```bash
docker compose ps
docker compose exec bedrock bedrock health
docker compose exec bedrock bedrock diagnose
docker compose exec bedrock bedrock index inspect
docker compose logs --tail=100 watch
```

`health` checks configuration, vault access, Qdrant, model-cache readiness, and
index compatibility. It also reports the optional answering model, whose
availability does not determine overall health. `diagnose` combines health and
index diagnostics; `index inspect` focuses on the active collection.

An MCP client can call `vault_status` to compare current parsed/link-resolved
entries with indexed payloads. It reports `current`, `stale`, or `unknown`.
This scans the vault and index without embedding models; it is not an atomic
snapshot or a watcher heartbeat. Watcher liveness must be checked through
Compose status and logs. See [the freshness contract](mcp.md#availability-and-freshness).

### If an edit is missing from search

1. Check that the path is inside the configured vault and outside excluded
   directories. Review [vault conventions](#vault-conventions) if dates, tags,
   or board filters are the unexpected part.
2. Check watcher status and logs. Markdown saves normally batch after two
   seconds; continuous edits are capped at ten seconds before processing,
   plus indexing time. Failed batches remain pending and retry with backoff
   up to sixty seconds. Startup runs a full catch-up pass.
3. To converge the whole index manually, run:

   ```bash
   docker compose exec bedrock bedrock index update
   ```

4. If diagnostics report an incompatible schema or model configuration, use
   `bedrock index rebuild` through Compose to build and validate a replacement
   collection before switching the live alias.

A `notes.changed` or `graph.updated` event does not establish search-index
readiness: the HTTP observer and indexing watcher run independently. Graph
failures return HTTP 503 and retry; SSE clients must refresh current state
on reconnect because events are not replayed. See the
[HTTP and event reference](cli-reference.md#network-graph).

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
`docker compose exec` picks it up with no rebuild. The long-running services hold their own code:

```bash
docker compose restart bedrock watch mcp
```

Rebuild the image only when a dependency changes.

## What the tests cover

`uv run pytest` runs the suite. Tests cover Markdown discovery and parsing,
chunking, dates and Kanban annotations, deterministic identity, retrieval and
filters, link resolution and bounded context, rebuilds and incremental updates,
watcher behavior, graph snapshots and events, CLI contracts, and vault tools.
The encoder and Qdrant tests exercise those integration boundaries as well.
These checks verify behavior; they do not establish search quality for every
vault or provide production-scale performance benchmarks.

## Kanban lifecycle dates

Each card can carry one explicitly named, indented fence at the end of its
contents. Keep it inside the card, before the next card or column:

````markdown
- [ ] Implement login
    - Include password reset

    ```bedrock-dates
    created_at: 2026-10-16T10:30:00+05:45
    started_at: 2026-10-16T11:00:00+05:45
    due_date: 2026-10-20
    ```
````

Recognized keys are exactly `created_at`, `started_at`, `completed_at`, and
`due_date`. Lifecycle timestamps require seconds and an explicit timezone;
Bedrock normalizes them to UTC. Due dates are calendar dates (`YYYY-MM-DD`).
Missing or empty values remain unknown. Existing cards receive no invented
creation or lifecycle dates. Older annotations such as `@{2026-10-16}` retain
their generic date meaning and never become due dates or lifecycle timestamps.

Unknown keys, malformed values, duplicate keys, and malformed or duplicate
blocks produce `invalid_kanban_dates` warnings. An ambiguous key is ignored;
multiple blocks invalidate the card's lifecycle fields. Other valid keys in a
single block remain usable. Date-block contents do not enter searchable prose,
tags, links, or embeddings. Original Markdown remains available through reads.

Both full rebuilds and incremental updates store the dates in Qdrant under
`board.created_at`, `board.started_at`, `board.completed_at`, and `board.due_date`.
The collection has datetime payload indexes for all four. Due dates are stored
at UTC midnight and returned as calendar dates. Date-only edits update metadata
without changing the card's current entry ID or embedding input.

When upgrading an existing collection, run a full rebuild to install the new
payload indexes and refresh existing payloads:

```bash
docker compose exec bedrock bedrock index rebuild
```

Restart the long-running Bedrock, MCP, and watcher services to load the new code.
The card editing tools automatically stamp new creation dates and transition
milestones. Moving into Doing stamps the first `started_at`. Completed stamps
`completed_at` and checks the card; reopening removes `completed_at`, unchecks,
and preserves the first `started_at`. Due dates remain explicit and optional.
Existing cards retain unknown creation dates unless supplied explicitly.

## Enabling Kanban edits

After updating the mount/user configuration, recreate the processes (a restart
alone does not apply mount or user changes):

```bash
docker compose up -d bedrock mcp watch
docker compose exec bedrock bedrock index rebuild
```

No note/board deletion tool is exposed. Before each update, move, archive, or
addition to an existing board, Bedrock saves the previous bytes beneath that
board's directory in `.bedrock-backups/<filename>.<revision>.bak`. These are not
Markdown notes and are not indexed. Backups are retained; inspect them before
manually restoring one. New-board creation uses exclusive publication and
refuses to overwrite an existing file. Board paths must be explicit `.md` paths
beneath the configured root; excluded paths, symlinks, and hard-linked files are
rejected.

Task writes use a per-board process lock, require the current board hash, and
check the original bytes again immediately before atomic replacement. Obsidian
does not participate in this lock: avoid simultaneous saves in Obsidian during
a tool write. The final compare/replace has a small unavoidable race with an
uncooperative external editor; backups retain the pre-edit revision.

See [the tool contract](mcp.md#kanban-editing) for retries, revision handling,
archive queries, and saved-source versus index-sync results. The editor supports
cards up to 64,000 characters; use paginated `read_note` for larger existing cards.
