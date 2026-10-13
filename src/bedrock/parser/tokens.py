"""Deterministic, dependency-free token estimation for embedding budgets.

Parsing must stay hermetic: chunk boundaries decide entry identifiers, so they
cannot depend on whether a model has been downloaded. The parser therefore
estimates token cost with the WordPiece approximation below instead of loading
the real tokenizer.

The divisors are calibrated, not guessed. They were fitted against the pinned
tokenizer over the fixture vault plus English, Spanish, Russian, Japanese, and
Korean samples, and chosen as the loosest values that never under-count any of
them. Normal prose is over-counted by roughly two times, which is the safety
margin: a section splits a little earlier than a tokenizer-aware splitter would
rather than risk content being silently cut.

Deliberately adversarial input, such as a long random identifier or a base64
blob, still fragments faster than this estimate. That case is not silent:
`VaultParser` warns from its own estimate, and `LocalEncoders.truncated_inputs`
counts what the real tokenizer actually cut.
"""

from __future__ import annotations

import math
import re

# WordPiece splits on script and punctuation boundaries before subwords, so
# count runs of letters and digits separately from standalone symbols.
_PIECE_RE = re.compile(r"[^\W\d_]+|\d+|[^\s\w]", re.UNICODE)

# Measured characters per subword piece, worst case over the calibration set.
_ASCII_CHARS_PER_TOKEN = 3
_DIGIT_CHARS_PER_TOKEN = 2

# The pinned model's vocabulary is English uncased, so other scripts fall back
# to very short pieces. Korean syllable blocks are the worst observed case.
_NON_ASCII_TOKENS_PER_CHAR = 3

# The pinned dense model's own declared `max_seq_length`. FastEmbed ships a
# 128 default for it, which is half what the model was fine-tuned for, so
# `LocalEncoders` raises the real tokenizer to whatever this is set to.
# Never exceed 512: past that the positional embeddings are untrained.
DEFAULT_TOKEN_WINDOW = 256

# `[CLS]` and `[SEP]` occupy two positions in every encoded sequence.
SPECIAL_TOKENS = 2

# A chunk narrower than this is not worth emitting as its own point.
MINIMUM_CONTENT_TOKENS = 16


def estimate_tokens(text: str) -> int:
    """Approximate the WordPiece token count, over-counting ordinary prose."""
    total = 0
    for piece in _PIECE_RE.findall(text):
        if piece.isdigit():
            total += max(1, math.ceil(len(piece) / _DIGIT_CHARS_PER_TOKEN))
        elif piece.isalpha() and piece.isascii():
            total += max(1, math.ceil(len(piece) / _ASCII_CHARS_PER_TOKEN))
        elif piece.isalpha():
            total += len(piece) * _NON_ASCII_TOKENS_PER_CHAR
        else:
            total += 1
    return total


def content_budget(prefix: str, token_window: int = DEFAULT_TOKEN_WINDOW) -> int:
    """Tokens left for entry content once labels and specials are paid for."""
    budget = token_window - SPECIAL_TOKENS - estimate_tokens(prefix)
    return max(budget, MINIMUM_CONTENT_TOKENS)
