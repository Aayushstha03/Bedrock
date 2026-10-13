# How Zenith works

The single source of truth for what Zenith is and why it is built this way.
Command syntax, flags, and response shapes live in
[cli-reference.md](cli-reference.md).

## 1. The decision

Markdown is the source of truth. Qdrant is the only derived datastore. The
index is disposable: any state Qdrant holds can be rebuilt from the vault, and
nothing in the application writes to the vault.

One Docker Compose project runs everything:

| Service | Role |
| --- | --- |
| `zenith` | Parse, index, embed, query, answer. Runs the health service. |
| `watch` | Watchdog watcher. Indexes each debounced change batch. |
| `mcp` | Read-only, agent-independent vault tools over MCP. |
| `qdrant` | All derived storage and retrieval. Dashboard on `127.0.0.1:6333`. |

Two persistent volumes hold the Qdrant data and the model cache. The vault is a
read-only bind mount.

```text
Markdown vault  ->  parser  ->  entries  ->  link resolution
                                                  |
                                              encoders
                                                  |
                                               Qdrant  ->  retrieval  ->  answers
```

Ruled out, deliberately: SQLite, FTS5, a separate graph database, cloud
inference, and hosted embedding APIs. Embedding runs in the application process
on CPU through FastEmbed. Section 9 explains why that constraint is load
bearing.

## 2. What one point is

A point is one independently searchable entry, never a whole file. Five entry
types exist:

| Entry type | What it is |
| --- | --- |
| `daily_section` | A section inside a daily note, recognized by an exact `YYYY-MM-DD` filename anywhere in the vault or by the configured log root. |
| `project_update` | Content under a `YYYY-MM-DD` heading in any standard note. |
| `freeform_section` | An ordinary heading-delimited section. |
| `freeform_chunk` | Loose or oversized content, split on paragraph boundaries. |
| `kanban_card` | One card in one Kanban column. |

Only two first-level directories carry meaning: the log root and the Kanban
root, both configurable. Outside the Kanban root, an exact `YYYY-MM-DD`
filename also identifies a daily note. Remaining Markdown files are standard
notes, wherever they sit and however they are nested. Directory names like `projects`, `work`, or
`archive` classify nothing.

Whole-note identity travels in every point payload, so a result always carries
its path, heading path, dates, and line range.

### Indexed payload fields

`schema.py` declares these payload indexes. Everything else in the payload is
carried for display and evidence, not for filtering.

| Field | Type | Used for |
| --- | --- | --- |
| `vault_id` | keyword | Vault isolation |
| `note_id`, `entry_id` | keyword | Identity lookup |
| `path`, `note_title` | keyword | Note resolution |
| `note_type`, `entry_type` | keyword | Kind filtering |
| `heading`, `heading_path` | keyword | Section filtering |
| `note_date`, `entry_date` | datetime | Date and range queries |
| `tags` | keyword | Fixed-pool tag filtering |
| `outgoing_note_ids` | keyword | Backlink queries |
| `note_aliases` | keyword | The other names a note answers to |
| `outgoing_link_keys` | keyword | Reverse lookup for scoped updates |
| `text` | text | Declared. No query path reads it yet. |
| `board.column`, `board.status` | keyword | Kanban filtering |
| `board.checked` | bool | Completion filtering |
| `modified_at` | datetime | Maintenance and diagnostics |

Dates are stored as UTC datetimes, so `entry_date` in a raw payload reads
`2026-08-19T00:00:00Z`. Every API and CLI result returns the plain calendar
date, `2026-08-19`.

## 3. Names decide everything

A **note id** is a hash of the vault name and the vault-relative path. A
**entry id** is a hash of the note id, the entry type, and a **structural key**.

The structural key sets what an edit costs:

- A section key is its heading path plus an occurrence number. It holds no
  text, so the section can be rewritten and the entry id survives. The point
  updates in place.
- A chunk with no heading uses a digest of its text. An edit makes a new entry
  id and leaves the old point with no owner.

The second case is why the stale sweep is not optional. Without it, every such
edit leaves a duplicate in the results.

Because a note id comes from the path, a rename is a delete plus a create, and
every entry id under it changes.

## 4. The pipeline

```text
files -> parse -> entries -> resolve links -> encode -> points -> Qdrant
```

| Stage | Module | What it does |
| --- | --- | --- |
| find | `parser/discovery.py` | Collect every `.md` file. Skip excluded folders. |
| parse | `parser/service.py` | Cut a note into entries at its headings, then split any section too long for the encoder. |
| name | `core/identity.py` | Give every note and entry a permanent id. |
| link | `index/links.py` | Resolve each `[[wikilink]]` to `resolved`, `ambiguous`, or `missing`. |
| encode | `index/encoders.py` | Build one dense and one sparse vector per entry. |
| store | `index/rebuild.py`, `index/incremental.py` | Write the points to Qdrant. |

Parsing uses a CommonMark AST, so `# Heading` never becomes a tag. Inline code
and code-block content are searchable, but tags and wikilinks inside code do
not become metadata or graph edges. Kanban plugin settings remain excluded.

Encoding is pinned: `sentence-transformers/all-MiniLM-L6-v2` for the dense
`semantic` vector at 384 dimensions and cosine distance, and `Qdrant/bm25` for
the sparse `text-bm25` vector with English stemming, stopwords, and Qdrant's
IDF modifier. A prefetch command fills the model cache so normal operation runs
offline.

## 5. Two write modes

**Rebuild** — `zenith index rebuild`. Writes into a new collection, validates
it, then moves the alias. It is atomic. A failure never touches the live index,
and its incomplete collection is removed.

**Incremental** — `zenith index update`, and the watcher. Writes into the live
collection. It compares the entries just parsed against the points already
stored:

| Case | Action |
| --- | --- |
| Not stored | insert |
| Payload changed | update |
| Payload identical | skip |
| Stored, no longer parsed | delete |

### A scoped update is not one file

`reindex(paths)` narrows the work, which is what the watcher calls on every
save. Narrowing is not simply parsing the changed file, for two reasons. Link
resolution is vault-wide: adding a note named `Target` must change a
`[[Target]]` in a file nobody edited. And the stale sweep subtracts, so a scope
of one file would make every other point look deleted.

So a scoped update:

1. Takes the changed note ids from the **paths**, not from the parse. A deleted
   file has no parse but still has an id, so its points stay in scope.
2. Reads a **header** for every note — path, id, title, aliases — from the file
   head. This is the catalog link resolution needs, at a fraction of a parse.
3. Collects the changed **names**: the old ones from the index, the new ones
   from the catalog.
4. Widens the scope with two queries: notes whose `outgoing_link_keys` match
   those names, and notes whose `outgoing_note_ids` match the changed ids.
5. Parses only the scope, and resolves its links against the whole catalog.
6. Reads existing points for the scope only, and sweeps inside it.

### Vectors follow the fingerprint, not the id

Every point stores an `embedding_fingerprint`: a digest of the parser version,
both model names, the encoder and tokenizer versions, the input format, and the
embedding text. Vector reuse is keyed by that fingerprint.

The embedding text holds the note **title**, not its path. So renaming a note
that has a heading or a frontmatter title changes every entry id but no
fingerprint, and the rename costs no encoding. Rename a note titled by its
filename and the title really did change, so re-encoding is correct.

## 6. Retrieval

Five modes, all under the same hard metadata filters, which apply before vector
retrieval and never leak:

| Mode | How it retrieves |
| --- | --- |
| `metadata` | Payload filters only. No query text. |
| `literal` | Candidates, then an exact, case-insensitive, Unicode-normalized phrase check against the stored text. |
| `lexical` | BM25 over the sparse vector. |
| `semantic` | Dense retrieval over the `semantic` vector, query embedded by the same model. |
| `hybrid` | Dense and sparse candidates fused with RRF. The default. |

Similarity is never proof of an exact phrase. A literal result is marked
`verified`; a semantic one is not.

**Context expansion** follows resolved outgoing links and backlinks from one
entry. Default depth 1, maximum depth 2, at most 5 linked notes, and a nearby
history window of plus or minus 3 days. Cycles terminate. Every item is labeled
`direct_evidence`, `followed_link`, `backlink`, or `nearby_history`, and
related context is never relabeled as direct evidence. Missing and ambiguous
links stop traversal and appear as diagnostics.

**Graph export** is the unbounded counterpart, meant for a future
visualization layer. One node per note, including notes with no links, tags, or
dates. Three edge types: `internal_link` for each resolved link occurrence in
link direction, `shared_tag` recorded once in canonical order, and `shared_date`
for an exact shared `note_date` or `entry_date`. No depth, count, or
nearby-day budget applies. Two exports of the same vault state are identical.

## 7. Dates

Four date meanings stay separate and never merge:

- `note_date`: the date a daily note represents.
- `entry_date`: a date written on the entry itself.
- `modified_at`: the filesystem timestamp.
- A mentioned date: metadata that establishes no chronology.

A date counts only when the complete value is a valid calendar date in the
exact ASCII `YYYY-MM-DD` form. A heading that is only such a date establishes
`entry_date` for the content it governs. No other representation does.

Undated material stays searchable and stays undated. It does not inherit a date
through proximity, and a link never transfers its date to its target.

### Kanban card dates

Cards carry dates like any other entry. Zenith reads the Obsidian Kanban date
annotation from the card text and stores it as the card's `entry_date`, so a
date query returns cards next to daily sections and project updates:

```markdown
- [ ] Buy saffron @{2026-05-12}
- [x] Finished kitchen setup @{2026-08-20} @@{14:30}
- [ ] Dated by daily-note link @[[2026-08-20]]
```

Zenith records the date only. It does not decide whether a date means due,
scheduled, or done, because the plugin does not record that either. Every card
payload carries `board.checked` and `board.status`, so a caller reads that
meaning from the board's own state.

The `@` and `@@` triggers come from the board's `kanban:settings` block when it
sets `date-trigger` or `time-trigger`, and fall back to the plugin defaults.
The time is kept as `board.card_time`. A date that is not a real calendar date
raises an `invalid_date` warning and dates nothing. A card carrying more than
one date warns and keeps the first.

The annotation never reaches the searchable text or the embedding input, so
plugin syntax cannot pollute BM25 terms or dense vectors. `@[[2026-08-20]]` is
read as a date, not as a link to a note named `2026-08-20`.

## 8. Entry size and the encoder window

The pinned dense model accepts 256 input tokens and truncates the rest without
reporting it. Zenith therefore splits any section, preamble, or headless note
that would overrun the window into several entries, cutting only on paragraph
boundaries. Each piece keeps the heading, heading path, entry type, and entry
date of the section it came from, so chronology and evidence stay intact.

The window is 256 because that is the `max_seq_length` all-MiniLM-L6-v2
declares for itself. FastEmbed would otherwise run it at 128, half the length
the model was fine-tuned for. `LocalEncoders` pins the real tokenizer to
`ZENITH_DENSE_TOKEN_WINDOW`, so the parser's budget and the encoder's actual
limit cannot drift apart. `Settings.validate` rejects any window above 512,
because past that this model's positional embeddings are untrained. A different
model carries a different ceiling, so that guard moves with the model.

The parser budgets tokens with a calibrated, dependency-free estimate, so
parsing stays hermetic and entry identifiers never depend on whether a model is
present. The estimate over-counts ordinary prose, so a section can split
slightly earlier than strictly necessary.

Two cases cannot be divided: a single paragraph larger than the window, and a
Kanban card, which is one atomic point. Both emit a `truncated_embedding_input`
warning instead of losing text silently.

## 9. Legacy answering layer

Zenith Vault v1 exposes its agent-independent tool contract through MCP; see
[mcp.md](mcp.md). It returns canonical identities and source metadata, leaving
citation presentation to the consuming application. The following answering
layer remains for migration to the separate Zenith application.

`zenith ask` puts a model in front of the index. The model does not compile a
query plan; it calls tools, and the trace of what it called is part of the
result.

It reaches the notes through six tools only: `search_notes`, `list_notes`,
`read_note`, `expand_context`, `find_backlinks`, and `find_tasks`. It cannot
write to the vault and it cannot export the graph. Collection tools return a
bounded page with `items`, `has_more`, and, when another page exists,
`next_offset`. Page offsets describe the current vault/index state rather than
a durable snapshot.

`search_notes` accepts no query for date-first metadata browsing: explicitly
dated entries appear newest first, then undated entries in document order. Date
filters include only entries carrying a matching `note_date` or `entry_date`.
`list_notes` reads the vault catalog rather than the index, so empty and
header-only notes remain discoverable. Kanban date filters work the same way:
they match only explicit card annotations and carry no due, scheduled, or done
meaning.

**Citations are ids the run issued.** Every tool result carries a short id,
`s1`, `s2`, `s3`, and the model cites that id rather than writing a citation of
its own. An answer reads `You added an llm based parsing model. [s2]`, and the
result's `citations` object resolves each id back to its note, heading, and
line range. Ids count from `s1` per question and mean nothing outside the
answer built from them. One entry reached twice keeps one id. Multiple sources
use adjacent ids, `[s2][s5]`; `[s2, s5]` is invalid. A whole note read through
`read_note` gets an id with no line range, because there is no one range to
claim. An id the run never issued resolves to `{"unknown": true}`. Zenith
validates citation syntax after every answer. It makes one correction pass
within the original request and tool-call budget; an answer that still fails is
returned visibly marked with `citation_valid: false` and `citation_errors`.

The model is served by LM Studio on the host over its OpenAI-compatible API,
configured through the `ZENITH_LLM_*` variables. Enable "Serve on Local
Network" in LM Studio so the container can reach it, and start its server.

Temperature is 0.0 by default and is sent on every request, so a preset held by
the LM Studio server does not apply. Raising it loosens the citation and
grounding rules the instructions depend on. Temperature 0.0 makes an answer
repeatable, not deterministic: a mixture-of-experts model served by LM Studio
was measured giving two different answers over six runs of one question on a
byte-identical prompt. Do not build anything on two runs agreeing.

`ask` checks the index, then LM Studio, before it builds the agent. An unready
index fails rather than searching nothing and reporting that the notes say
nothing.

**LM Studio is optional, and that is a constraint, not a convenience.** LM
Studio holds a chat model in VRAM. Parsing, indexing, and the watcher must keep
running while it is closed, so they must never compete for that VRAM and must
never depend on a desktop application. This is why embedding runs in-process on
CPU through FastEmbed, and why serving embeddings from LM Studio or any other
external inference server is rejected. `zenith health` reports LM Studio but
never fails because of it, and the container liveness probe at `/healthz` does
not contact it at all.

## 10. Rules that never bend

- Never modify Markdown during indexing.
- Treat the Qdrant collection as rebuildable derived data.
- Keep parser, payload-schema, tokenizer, BM25, and embedding-model versions.
- Use deterministic ids and idempotent upserts. Leave no stale points after a
  rename, a deletion, or a version upgrade.
- `# Heading` never becomes a tag, and a section's tags never leak outward.
- Code is searchable without contributing tag metadata or graph edges; Kanban
  plugin settings remain excluded.
- Never silently resolve an ambiguous note title.
- Never infer a date from a link, and never let a mentioned date establish
  chronology.
- Never infer Kanban status from checkbox state alone.
- Never present semantic similarity as proof of an exact phrase.
- Always keep the path, heading path, and source line range.
- Bound every automatic link traversal.
