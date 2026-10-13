"""Local FastEmbed model lifecycle."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bedrock.core.config import Settings

MARKER = "ready.json"


def marker_path(settings: Settings) -> Path:
    return settings.model_cache_path / MARKER


def readiness(settings: Settings) -> dict[str, Any]:
    marker = marker_path(settings)
    if not marker.is_file():
        return {"ready": False, "reason": "model cache has not been prefetched"}
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"ready": False, "reason": "model readiness marker is invalid"}
    expected = {"dense_model": settings.dense_model, "sparse_model": settings.sparse_model}
    if any(data.get(key) != value for key, value in expected.items()):
        return {"ready": False, "reason": "cached model versions do not match configuration"}
    return {"ready": True, **data}


def prefetch(
    settings: Settings,
    dense_factory: Callable[..., Any] | None = None,
    sparse_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if dense_factory is None or sparse_factory is None:
        from fastembed import SparseTextEmbedding, TextEmbedding

        dense_factory = dense_factory or TextEmbedding
        sparse_factory = sparse_factory or SparseTextEmbedding

    settings.model_cache_path.mkdir(parents=True, exist_ok=True)
    dense = dense_factory(model_name=settings.dense_model, cache_dir=str(settings.model_cache_path))
    sparse = sparse_factory(model_name=settings.sparse_model, cache_dir=str(settings.model_cache_path))
    dense_vector = next(iter(dense.embed(["bedrock model readiness"])))
    sparse_vector = next(iter(sparse.embed(["bedrock model readiness"])))
    dense_size = len(dense_vector)
    if dense_size != 384:
        raise RuntimeError(f"dense model returned {dense_size} dimensions; expected 384")
    if len(sparse_vector.indices) == 0:
        raise RuntimeError("sparse model returned an empty vector")
    data = {
        "dense_model": settings.dense_model,
        "dense_dimensions": dense_size,
        "sparse_model": settings.sparse_model,
        "prefetched_at": datetime.now(UTC).isoformat(),
    }
    marker_path(settings).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return {"ready": True, **data}
