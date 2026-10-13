"""Labeled-query comparison of dense, BM25, and hybrid retrieval quality.

Runs real MiniLM/BM25 embeddings against the fixture vault. Requires the
pinned models to already be prefetched, so it is meant to run inside the
`bedrock` container where the model-cache volume is mounted, e.g.:

    BEDROCK_VAULT_PATH=$(pwd)/tests/fixtures/vault \\
        docker compose run --rm --entrypoint python bedrock \\
        -m bedrock.retrieval.eval_quality
"""

from __future__ import annotations

import sys

from bedrock.core.config import Settings
from bedrock.core.contracts import QueryPlan, RetrievalMode
from bedrock.index.rebuild import IndexRebuilder
from bedrock.retrieval.service import Retriever

LABELED_QUERIES: tuple[dict[str, object], ...] = (
    {
        "label": "exact technical terms (lexical-favorable)",
        "lexical_text": "clustering reclustering",
        "semantic_text": "clustering and reclustering the pipeline",
        "expected_path": "projects/News Resolution.md",
        "expected_heading": "2026-08-19",
    },
    {
        "label": "paraphrase with no shared words (semantic-favorable)",
        "lexical_text": "meal cooked with poultry",
        "semantic_text": "a meal I cooked with poultry",
        "expected_path": "logs/2026-08-20.md",
        "expected_heading": "Meals",
    },
    {
        "label": "mood paraphrase with no shared words (semantic-favorable)",
        "lexical_text": "did I enjoy my day",
        "semantic_text": "did I enjoy my day",
        "expected_path": "logs/2026-08-20.md",
        "expected_heading": "Thoughts",
    },
    {
        "label": "kitchen appliance concept (semantic-favorable)",
        "lexical_text": "appliance countdown reminder",
        "semantic_text": "a countdown reminder for a kitchen appliance",
        "expected_path": "kanban/Kitchen App.md",
        "expected_heading": None,
    },
    {
        "label": "exact project term (lexical-favorable)",
        "lexical_text": "llm based parsing summarization",
        "semantic_text": "using a language model to parse and summarize",
        "expected_path": "projects/News Resolution.md",
        "expected_heading": "2026-08-17",
    },
)


def _rank(results: tuple, expected_path: str, expected_heading: str | None) -> int | None:
    for index, result in enumerate(results, start=1):
        if result.path == expected_path and (expected_heading is None or result.heading == expected_heading):
            return index
    return None


def main() -> int:
    settings = Settings.from_env()
    vault_id = "personal"
    IndexRebuilder(settings, vault_id=vault_id).rebuild()
    retriever = Retriever(settings, vault_id=vault_id)

    rows: list[tuple[str, int | None, int | None, int | None]] = []
    for case in LABELED_QUERIES:
        common = {"limit": 5}

        dense = retriever.search(
            QueryPlan(mode=RetrievalMode.SEMANTIC, semantic_text=case["semantic_text"], **common)
        )
        bm25 = retriever.search(
            QueryPlan(mode=RetrievalMode.LEXICAL, lexical_text=case["lexical_text"], **common)
        )
        hybrid = retriever.search(
            QueryPlan(
                mode=RetrievalMode.HYBRID,
                lexical_text=case["lexical_text"],
                semantic_text=case["semantic_text"],
                **common,
            )
        )

        rows.append(
            (
                case["label"],
                _rank(dense, case["expected_path"], case["expected_heading"]),
                _rank(bm25, case["expected_path"], case["expected_heading"]),
                _rank(hybrid, case["expected_path"], case["expected_heading"]),
            )
        )

    header = f"{'query':<55} {'dense':>6} {'bm25':>6} {'hybrid':>6}"
    print(header)
    print("-" * len(header))
    for label, dense_rank, bm25_rank, hybrid_rank in rows:
        print(f"{label:<55} {_fmt(dense_rank):>6} {_fmt(bm25_rank):>6} {_fmt(hybrid_rank):>6}")

    hit_at_5 = {
        "dense": sum(1 for _, d, _, _ in rows if d is not None) / len(rows),
        "bm25": sum(1 for _, _, b, _ in rows if b is not None) / len(rows),
        "hybrid": sum(1 for _, _, _, h in rows if h is not None) / len(rows),
    }
    print()
    print(
        f"hit@5 -- dense: {hit_at_5['dense']:.0%}  bm25: {hit_at_5['bm25']:.0%}  hybrid: {hit_at_5['hybrid']:.0%}"
    )
    return 0


def _fmt(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


if __name__ == "__main__":
    sys.exit(main())
