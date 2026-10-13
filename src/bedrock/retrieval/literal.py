"""Strict literal verification against original payload text."""

from __future__ import annotations

import unicodedata


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def verify_literal(query: str, candidate_text: str) -> bool:
    return normalize(query).casefold() in normalize(candidate_text).casefold()
