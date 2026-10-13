"""Paragraph-aware chunking bounded by the embedding token budget."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    start_zero: int
    end_zero: int


def paragraphs(lines: list[str], start_zero: int, end_zero: int) -> tuple[Chunk, ...]:
    """Split a line range on blank lines. Paragraphs are never divided."""
    result: list[Chunk] = []
    cursor = start_zero
    while cursor < end_zero:
        while cursor < end_zero and not lines[cursor].strip():
            cursor += 1
        if cursor >= end_zero:
            break
        paragraph_start = cursor
        while cursor < end_zero and lines[cursor].strip():
            cursor += 1
        result.append(Chunk(paragraph_start, cursor))
    return tuple(result)


def pack(blocks: Sequence[Chunk], costs: Sequence[int], budget: int) -> tuple[Chunk, ...]:
    """Greedily merge adjacent paragraphs while they fit the token budget.

    A paragraph whose own cost exceeds the budget still becomes one chunk.
    Splitting it would cut a sentence in half, so the caller warns instead.
    """
    if len(blocks) != len(costs):
        raise ValueError("blocks and costs must be the same length")
    chunks: list[Chunk] = []
    start: int | None = None
    end = 0
    accumulated = 0
    for block, cost in zip(blocks, costs, strict=True):
        if start is not None and accumulated + cost > budget:
            chunks.append(Chunk(start, end))
            start = None
            accumulated = 0
        if start is None:
            start = block.start_zero
        end = block.end_zero
        accumulated += cost
    if start is not None:
        chunks.append(Chunk(start, end))
    return tuple(chunks)
