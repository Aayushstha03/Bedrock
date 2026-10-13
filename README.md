# Zenith Vault

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

Zenith Vault is the tool for that second part. It makes the lookup fast, simple, and
one step. I ask a question in the words I remember, and Zenith Vault gives me the
entries I wrote.

It is also one tool for all of it. My journals, my Kanban boards, my project
updates, and my freeform logs are different shapes of Markdown, but they hold
one history. Zenith Vault reads all of them, indexes each entry the same way, and
searches them together.

## What it is

A local-first search index for a Markdown vault. The Markdown stays the source
of truth and is mounted read-only. Qdrant holds the derived index and nothing
else, so the whole index can be thrown away and rebuilt from the notes.

Zenith Vault cuts notes into entries — daily sections, dated project updates,
freeform sections, and Kanban cards — and indexes each one with a dense vector,
a BM25 sparse vector, and a payload carrying its path, heading path, dates, and
line range. Search it by metadata, exact phrase, keyword, meaning, or a hybrid
of the last two. Expose those entries through a read-only MCP server so any
MCP-compatible agent can search and read the vault. The indexer and watcher need no chat model.


## Start it

```bash
cp .env.example .env          # then set ZENITH_VAULT_PATH to your vault
docker compose build
docker compose run --rm model-prefetch
docker compose up -d
```

`.env.example` is the template, and it documents every setting.
The default vault path points at the fixture vault, which is safe to index as-is.

Build the index, then search it:

```bash
docker compose exec zenith zenith index rebuild
docker compose exec zenith zenith search "pipeline clustering" --mode hybrid
```

The watcher runs as its own service and reindexes each saved file, so the index
follows the vault without another command.

The MCP endpoint is **http://127.0.0.1:8000/mcp** (Streamable HTTP).
See [docs/mcp.md](docs/mcp.md) for client configuration, the eight tools, and
complete note reading. Citation presentation belongs to the consuming agent.

The legacy `zenith ask` command needs LM Studio running on the host with
"Serve on Local Network" enabled. Nothing else does. Parsing, indexing, and the watcher all run with LM
Studio closed, so the index never competes with a chat model for VRAM.

The Qdrant dashboard is at <http://localhost:6333/dashboard>.

## Vault conventions

- **Kanban boards:** keep all boards under the configured top-level
  `ZENITH_KANBAN_ROOT` folder (default `kanban`). Classification uses the folder,
  not the `kanban-plugin: board` property. Boards elsewhere are treated as
  ordinary notes; keep ordinary notes out of the board folder.
- **Tags and properties:** tag filters use the curated `ZENITH_KNOWN_TAGS` set
  and configured aliases from inline prose tags. Frontmatter tags and arbitrary
  properties are not search filters. Store meaningful information in the body.
- **Dates:** use exact `YYYY-MM-DD` filenames for daily notes. Inside the
  configured logs folder, an exact date on the first nonempty body line is also
  accepted. Dated headings in ordinary notes must contain only `YYYY-MM-DD`;
  child sections inherit their date. Dates mentioned in prose or properties
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
  the stored heading exactly. Block references and same-note heading links
  are not supported by graph traversal.

## Working on it

```bash
uv sync --dev
uv run pytest
```

Compose binds `./src` over the image, and `PYTHONPATH` puts the mount first, so
the containers run the working tree. Edit a file and the next
`docker compose exec` picks it up with no rebuild. The two long-running
processes hold their own code:

```bash
docker compose restart zenith watch mcp
```

Rebuild the image only when a dependency changes.

## Documentation

- [docs/mcp.md](docs/mcp.md) — the Zenith Vault v1 tool contract and MCP setup.
- [docs/design.md](docs/design.md) — how Zenith works, and why. The decisions,
  the indexing pipeline, the retrieval modes, the invariants, and the known
  limits.
- [docs/cli-reference.md](docs/cli-reference.md) — every command, option,
  response shape, exit code, and Python API method.
