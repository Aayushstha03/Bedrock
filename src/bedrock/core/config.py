"""Typed runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"

# Input positions the pinned dense model accepts before it truncates.
# FastEmbed defaults all-MiniLM-L6-v2 to 128, but the model's own
# sentence_bert_config.json declares 256, which is what it was fine-tuned
# for. `LocalEncoders` pins the real tokenizer to this value.
DENSE_TOKEN_WINDOW = 256

# The answering model is served by LM Studio on the host, over its
# OpenAI-compatible API. It is optional: parsing, indexing, and the watcher run
# with LM Studio closed so the index never competes for VRAM. LM Studio ignores
# the API key, but the OpenAI client requires a non-empty one.
LLM_BASE_URL = "http://host.docker.internal:1234/v1"
LLM_MODEL = "google/gemma-4-e4b"
LLM_API_KEY = "lm-studio"
# Seconds to wait for one answer. The OpenAI client defaults to a 600 second
# read timeout, which turns a stalled local model into a CLI that appears to
# hang. A local model that has not answered in two minutes is stuck.
LLM_TIMEOUT = 120.0
# Sampling temperature for one answer. Zero by default: the same question over
# unchanged notes should not produce a different answer each time. The value is
# sent on every request, so it overrides any preset the LM Studio server holds.
LLM_TEMPERATURE = 0.0
# The OpenAI-compatible range. Above roughly one, a small model stops obeying
# the citation and grounding rules the instructions depend on.
LLM_TEMPERATURE_MAX = 2.0


def _positive_int(name: str, default: int) -> int:
    # Compose substitutes an empty string for a variable an older `.env` does
    # not define. That means unset, not invalid: refusing it would stop every
    # command, including `bedrock serve` and the `health` that would explain it.
    raw = os.getenv(name) or str(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name) or str(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _bounded_float(name: str, default: float, low: float, high: float) -> float:
    """Read a float that is allowed to sit at zero, unlike `_positive_float`.

    A temperature of zero is the useful default rather than a mistake, so the
    bound is a range and not a sign.
    """
    raw = os.getenv(name) or str(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")
    return value


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip().strip("/") for item in os.getenv(name, default).split(",") if item.strip())


def _aliases(name: str, default: str) -> tuple[tuple[str, str], ...]:
    aliases: list[tuple[str, str]] = []
    for item in _csv(name, default):
        alias, separator, canonical = item.partition(":")
        if not separator or not alias.strip() or not canonical.strip():
            raise ValueError(f"{name} entries must use alias:canonical syntax")
        aliases.append((alias.strip().casefold(), canonical.strip().casefold()))
    return tuple(aliases)


@dataclass(frozen=True, slots=True)
class Settings:
    qdrant_url: str
    vault_path: Path
    model_cache_path: Path
    collection_name: str
    host: str
    port: int
    dense_model: str = DENSE_MODEL
    sparse_model: str = SPARSE_MODEL
    log_root: str = "logs"
    kanban_root: str = "kanban"
    excluded_directories: tuple[str, ...] = (".git", ".obsidian", ".trash")
    known_tags: tuple[str, ...] = ("journal", "recipe", "work")
    tag_aliases: tuple[tuple[str, str], ...] = (("journaling", "journal"), ("recipes", "recipe"))
    dense_token_window: int = DENSE_TOKEN_WINDOW
    llm_base_url: str = LLM_BASE_URL
    llm_model: str = LLM_MODEL
    llm_api_key: str = LLM_API_KEY
    llm_timeout: float = LLM_TIMEOUT
    llm_temperature: float = LLM_TEMPERATURE

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            qdrant_url=os.getenv("BEDROCK_QDRANT_URL", "http://qdrant:6333").rstrip("/"),
            vault_path=Path(os.getenv("BEDROCK_VAULT_PATH", "/vault")),
            model_cache_path=Path(os.getenv("BEDROCK_MODEL_CACHE_PATH", "/models")),
            collection_name=os.getenv("BEDROCK_COLLECTION", "bedrock_entries"),
            host=os.getenv("BEDROCK_HOST", "0.0.0.0"),
            port=_positive_int("BEDROCK_PORT", 8080),
            dense_model=os.getenv("BEDROCK_DENSE_MODEL", DENSE_MODEL),
            sparse_model=os.getenv("BEDROCK_SPARSE_MODEL", SPARSE_MODEL),
            log_root=os.getenv("BEDROCK_LOG_ROOT", "logs").strip("/"),
            kanban_root=os.getenv("BEDROCK_KANBAN_ROOT", "kanban").strip("/"),
            excluded_directories=_csv("BEDROCK_EXCLUDED_DIRECTORIES", ".git,.obsidian,.trash"),
            known_tags=_csv("BEDROCK_KNOWN_TAGS", "journal,recipe,work"),
            tag_aliases=_aliases("BEDROCK_TAG_ALIASES", "journaling:journal,recipes:recipe"),
            dense_token_window=_positive_int("BEDROCK_DENSE_TOKEN_WINDOW", DENSE_TOKEN_WINDOW),
            llm_base_url=os.getenv("BEDROCK_LLM_BASE_URL", LLM_BASE_URL).rstrip("/"),
            llm_model=os.getenv("BEDROCK_LLM_MODEL", LLM_MODEL).strip(),
            llm_api_key=os.getenv("BEDROCK_LLM_API_KEY", LLM_API_KEY),
            llm_timeout=_positive_float("BEDROCK_LLM_TIMEOUT", LLM_TIMEOUT),
            llm_temperature=_bounded_float(
                "BEDROCK_LLM_TEMPERATURE", LLM_TEMPERATURE, 0.0, LLM_TEMPERATURE_MAX
            ),
        )

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.qdrant_url.startswith(("http://", "https://")):
            errors.append("BEDROCK_QDRANT_URL must use http or https")
        if not self.collection_name.strip():
            errors.append("BEDROCK_COLLECTION cannot be empty")
        if not self.vault_path.is_absolute():
            errors.append("BEDROCK_VAULT_PATH must be absolute")
        if not self.model_cache_path.is_absolute():
            errors.append("BEDROCK_MODEL_CACHE_PATH must be absolute")
        if not self.log_root or "/" in self.log_root:
            errors.append("BEDROCK_LOG_ROOT must be one root directory name")
        if not self.kanban_root or "/" in self.kanban_root:
            errors.append("BEDROCK_KANBAN_ROOT must be one root directory name")
        if self.log_root.casefold() == self.kanban_root.casefold():
            errors.append("log and Kanban roots must differ")
        known = {tag.casefold() for tag in self.known_tags}
        if any(canonical not in known for _, canonical in self.tag_aliases):
            errors.append("every tag alias must target a known canonical tag")
        if self.dense_token_window < 32:
            errors.append("BEDROCK_DENSE_TOKEN_WINDOW must be at least 32")
        if self.dense_token_window > 512:
            errors.append(
                "BEDROCK_DENSE_TOKEN_WINDOW must not exceed 512, the positional limit "
                "of the supported dense models"
            )
        if not self.llm_base_url.startswith(("http://", "https://")):
            errors.append("BEDROCK_LLM_BASE_URL must use http or https")
        if not self.llm_model:
            errors.append("BEDROCK_LLM_MODEL cannot be empty")
        if not self.llm_api_key:
            errors.append("BEDROCK_LLM_API_KEY cannot be empty")
        return tuple(errors)
