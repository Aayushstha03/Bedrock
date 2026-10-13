"""The labelled evidence one answering run has been shown.

A citation the model writes by hand is a citation it can get wrong. Assembling
a note title, a heading, and a line range into a template is three chances to
drift, and a small model takes them. So the model never writes a citation at
all. Every result a tool returns carries a short label, `s1`, `s2`, `s3`, and
the model cites the label. This module hands out those labels and turns them
back into the entry they stand for.

The ledger belongs to one run. Labels mean nothing outside the answer that was
built from them, so a new question starts a new ledger and counts from `s1`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from bedrock.core.contracts import NoteContent, SearchResult
from bedrock.library import Bedrock

# Each source citation is one bracketed label, such as `[s1]`. Several sources
# use adjacent citations, `[s1][s4]`; putting multiple ids into one bracket is
# deliberately invalid so the answer grammar stays unambiguous.
CITATION = re.compile(r"\[([^\[\]]*)\]")
LABEL = re.compile(r"\bs\d+\b", re.IGNORECASE)
VALID_CITATION = re.compile(r"\[(s\d+)\]", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Source:
    """What one label stands for, in the terms a reader needs."""

    note: str
    path: str
    heading: str | None = None
    lines: str | None = None
    # The entry a label came from, and the argument `expand_context` needs.
    # A whole note read through `read_note` has no entry and no line range.
    entry_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        projected: dict[str, Any] = {"note": self.note, "path": self.path}
        if self.heading is not None:
            projected["heading"] = self.heading
        if self.lines is not None:
            projected["lines"] = self.lines
        return projected


@dataclass(frozen=True, slots=True)
class CitationCheck:
    valid: bool
    errors: tuple[str, ...] = ()


@dataclass
class Sources:
    """The vault, plus the labels this run has issued over it.

    This is the agent's dependency rather than the library itself, because the
    labels have to outlive a single tool call: the model meets `s3` in a search
    and cites it three tool calls later.
    """

    vault: Bedrock
    _sources: dict[str, Source] = field(default_factory=dict)
    _labels: dict[str, str] = field(default_factory=dict)

    def label(self, result: SearchResult) -> str:
        """Label one entry, reusing the label if this run has shown it before.

        Search, backlinks, and context expansion overlap, and the same entry
        reached twice must keep one label. Two labels for one entry would read
        as two independent pieces of evidence.
        """
        return self._issue(
            result.entry_id,
            Source(
                note=result.note_title,
                path=result.path,
                heading=result.heading,
                lines=f"{result.start_line}-{result.end_line}",
                entry_id=result.entry_id,
            ),
        )

    def label_note(self, content: NoteContent) -> str:
        """Label one whole note.

        A note read whole has no line range to cite, which is why a citation
        the model assembles from `lines` could not be written for it at all.
        A label can: it resolves to the note, and says nothing it cannot prove.
        """
        return self._issue(
            f"note:{content.path}",
            Source(note=content.title, path=content.path),
        )

    def entry_id(self, label: str) -> str:
        """Resolve a label to the entry it stands for.

        Raises `LookupError` with the labels that do exist, so a model that
        invents one is told what it may cite instead of failing the run.
        """
        source = self._sources.get(label.strip().lower())
        if source is None:
            raise LookupError(
                f"{label!r} is not a source this answer has seen. "
                f"{self._known()} Use the `id` of a result you were given."
            )
        if source.entry_id is None:
            raise LookupError(
                f"{label!r} is a whole note, not an entry, so it has no context "
                "to expand. Search the note first, then expand a result of that."
            )
        return source.entry_id

    def cited(self, answer: str) -> dict[str, dict[str, Any]]:
        """Resolve the labels an answer cites, in the order it cites them.

        A label the run never issued is reported too, with no note behind it.
        The model invented it, and an answer that cites evidence which does not
        exist must not look the same as one that cites evidence which does.
        """
        resolved: dict[str, dict[str, Any]] = {}
        for match in VALID_CITATION.finditer(answer):
            label = match.group(1).lower()
            if label in resolved:
                continue
            source = self._sources.get(label)
            resolved[label] = source.to_dict() if source else {"unknown": True}
        return resolved

    def citation_check(self, answer: str) -> CitationCheck:
        """Check the answer's source-reference grammar, not its semantics."""
        errors: list[str] = []
        valid_spans = {
            (match.start(1), match.end(1)): match.group(1).lower()
            for match in VALID_CITATION.finditer(answer)
        }
        known = {label for label in valid_spans.values() if label in self._sources}

        for bracket in CITATION.finditer(answer):
            content = bracket.group(1)
            if LABEL.search(content) and not LABEL.fullmatch(content):
                errors.append(f"malformed citation [{content}]; use one id such as [s1]")
        for match in LABEL.finditer(answer):
            span = (match.start(), match.end())
            label = match.group(0).lower()
            if span not in valid_spans:
                errors.append(f"bare source id {label}; write it as [{label}]")
            elif label not in self._sources:
                errors.append(f"unknown citation [{label}]")
        if self._sources and not known:
            errors.append("answer used vault evidence but contains no valid issued citation")
        return CitationCheck(not errors, tuple(dict.fromkeys(errors)))

    def _issue(self, key: str, source: Source) -> str:
        existing = self._labels.get(key)
        if existing is not None:
            return existing
        label = f"s{len(self._labels) + 1}"
        self._labels[key] = label
        self._sources[label] = source
        return label

    def _known(self) -> str:
        if not self._sources:
            return "No results have been returned yet."
        return f"The results so far are {', '.join(sorted(self._sources))}."
