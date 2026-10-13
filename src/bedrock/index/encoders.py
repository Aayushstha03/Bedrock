"""Pinned local dense and sparse encoding."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from qdrant_client import models

from bedrock.core.config import Settings

DENSE_DIMENSIONS = 384


class LocalEncoders:
    """Load FastEmbed models once and encode batches into Qdrant vectors."""

    def __init__(
        self,
        settings: Settings,
        dense: Any | None = None,
        sparse: Any | None = None,
    ) -> None:
        if dense is None or sparse is None:
            from fastembed import SparseTextEmbedding, TextEmbedding

            dense = dense or TextEmbedding(
                model_name=settings.dense_model,
                cache_dir=str(settings.model_cache_path),
                local_files_only=True,
            )
            sparse = sparse or SparseTextEmbedding(
                model_name=settings.sparse_model,
                cache_dir=str(settings.model_cache_path),
                local_files_only=True,
                language="english",
                disable_stemmer=False,
            )
        self.dense = dense
        self.sparse = sparse
        self.token_window = settings.dense_token_window
        self.truncated_inputs = 0
        self._pin_token_window()

    def encode(self, texts: Sequence[str]) -> list[dict[str, object]]:
        dense_vectors = self.encode_dense(texts)
        sparse_vectors = self.encode_sparse(texts)
        return [
            {"semantic": dense, "text-bm25": sparse}
            for dense, sparse in zip(dense_vectors, sparse_vectors, strict=True)
        ]

    def encode_dense(self, texts: Sequence[str]) -> list[list[float]]:
        self.truncated_inputs += self._count_truncated(texts)
        dense_vectors = list(self.dense.embed(list(texts)))
        if len(dense_vectors) != len(texts):
            raise RuntimeError("dense encoder returned a different number of vectors than inputs")

        result: list[list[float]] = []
        for dense in dense_vectors:
            dense_values = _float_list(dense)
            if len(dense_values) != DENSE_DIMENSIONS:
                raise RuntimeError(
                    f"dense model returned {len(dense_values)} dimensions; expected {DENSE_DIMENSIONS}"
                )
            result.append(dense_values)
        return result

    def _pin_token_window(self) -> None:
        """Hold the real tokenizer to the window the parser budgets against.

        FastEmbed defaults all-MiniLM-L6-v2 to 128 positions even though the
        model declares 256. Without this the parser would split entries for one
        window while the encoder silently cut them at another, which is the
        exact failure the token budget exists to prevent.
        """
        tokenizer = self._tokenizer()
        if tokenizer is None:
            return
        tokenizer.enable_truncation(max_length=self.token_window)

    def _count_truncated(self, texts: Sequence[str]) -> int:
        """Count inputs the real tokenizer cuts, catching parser-estimate drift.

        The parser budgets tokens with a hermetic approximation. This is the
        only place the pinned tokenizer is authoritative, so a model upgrade
        that narrows the window shows up here instead of silently degrading
        recall.
        """
        tokenizer = self._tokenizer()
        if tokenizer is None:
            return 0
        limit = (tokenizer.truncation or {}).get("max_length")
        if not limit:
            return 0
        truncated = 0
        for text in texts:
            encoded = tokenizer.encode(text, add_special_tokens=True)
            # A padded encoding reports the padded length, so trust the mask.
            length = sum(encoded.attention_mask) if encoded.attention_mask else len(encoded.ids)
            if length >= limit:
                truncated += 1
        return truncated

    def _tokenizer(self) -> Any | None:
        try:
            return self.dense.model.tokenizer
        except AttributeError:
            return None

    def encode_sparse(self, texts: Sequence[str]) -> list[models.SparseVector]:
        sparse_vectors = list(self.sparse.embed(list(texts)))
        if len(sparse_vectors) != len(texts):
            raise RuntimeError("sparse encoder returned a different number of vectors than inputs")

        result: list[models.SparseVector] = []
        for sparse in sparse_vectors:
            indices = _int_list(sparse.indices)
            values = _float_list(sparse.values)
            if not indices or len(indices) != len(values):
                raise RuntimeError("sparse model returned an invalid or empty vector")
            result.append(models.SparseVector(indices=indices, values=values))
        return result


def _float_list(values: Iterable[object]) -> list[float]:
    return [float(value) for value in values]


def _int_list(values: Iterable[object]) -> list[int]:
    return [int(value) for value in values]
