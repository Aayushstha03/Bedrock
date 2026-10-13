"""Application and dependency health checks."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from bedrock.core.config import Settings
from bedrock.index.diagnostics import inspect_collection
from bedrock.runtime.models import readiness


def qdrant_health(settings: Settings, timeout: float = 2.0) -> dict[str, Any]:
    try:
        with urlopen(f"{settings.qdrant_url}/healthz", timeout=timeout) as response:
            return {"ready": response.status == 200, "status": response.status}
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return {"ready": False, "error": str(exc)}


def index_health(settings: Settings) -> dict[str, Any]:
    try:
        return inspect_collection(settings)
    except Exception as exc:
        return {"ready": False, "collection": settings.collection_name, "error": str(exc)}


def llm_health(settings: Settings, timeout: float = 1.0) -> dict[str, Any]:
    """Report whether LM Studio is serving the configured answering model.

    A closed LM Studio is a normal state, not a fault: it frees the VRAM that
    parsing, indexing, and the watcher must never compete for.
    """
    report: dict[str, Any] = {
        "base_url": settings.llm_base_url,
        "model": settings.llm_model,
        "required": False,
    }
    try:
        with urlopen(f"{settings.llm_base_url}/models", timeout=timeout) as response:
            served = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        return {**report, "ready": False, "error": str(exc)}
    # An OpenAI-compatible server is not obliged to return the shape OpenAI
    # returns. A bare array, or an array of strings, must read as not ready
    # rather than raise out of the report this block is only one part of.
    listed = served.get("data") if isinstance(served, dict) else None
    if not isinstance(listed, list):
        listed = []
    available = sorted(str(item["id"]) for item in listed if isinstance(item, dict) and item.get("id"))
    return {
        **report,
        "ready": settings.llm_model in available,
        "available_models": available,
    }


def health_report(settings: Settings, *, include_llm: bool = False) -> dict[str, Any]:
    """Combine every dependency the container needs to be considered healthy.

    `include_llm` is off by default because the container liveness probe must
    not depend on a desktop application, and must not spend its timeout budget
    waiting for one. `bedrock health` and `bedrock diagnose` turn it on.
    """
    config_errors = settings.validate()
    qdrant = qdrant_health(settings)
    models = readiness(settings)
    index = (
        index_health(settings)
        if qdrant["ready"]
        else {
            "ready": False,
            "collection": settings.collection_name,
            "reason": "Qdrant is unreachable",
        }
    )
    vault = {
        "ready": settings.vault_path.is_dir(),
        "path": str(settings.vault_path),
        "read_only_expected": True,
    }
    ready = not config_errors and qdrant["ready"] and index["ready"] and models["ready"] and vault["ready"]
    report = {
        "ready": ready,
        "configuration": {"ready": not config_errors, "errors": list(config_errors)},
        "qdrant": qdrant,
        "index": index,
        "models": models,
        "vault": vault,
    }
    if include_llm:
        # Deliberately absent from `ready`: the answering model is optional.
        report["llm"] = llm_health(settings)
    return report


def encode_report(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, sort_keys=True) + "\n").encode("utf-8")
