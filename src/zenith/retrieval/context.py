"""Budgeted graph traversal that keeps direct and related evidence distinct."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, timedelta

from zenith.core.contracts import (
    ContextExpansion,
    ContextItem,
    ContextLabel,
    EntryType,
    LinkResolution,
    QueryPlan,
    RetrievalMode,
    SearchResult,
    SectionContinuation,
    TraversalDiagnostic,
)
from zenith.retrieval.service import Retriever


@dataclass(frozen=True, slots=True)
class _Candidate:
    note_id: str
    label: ContextLabel
    via_note_id: str
    references: int
    section: str | None = None


class ContextExpander:
    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def expand(
        self,
        entry_id: str,
        *,
        link_depth: int = 1,
        nearby_days: int = 3,
        max_notes: int = 5,
        max_items: int = 30,
        mode: RetrievalMode = RetrievalMode.HYBRID,
    ) -> ContextExpansion:
        _validate_budgets(link_depth, nearby_days, max_notes)
        if not 1 <= max_items <= 100:
            raise ValueError("max_items must be between 1 and 100")
        reasons: set[str] = set()
        source = self.retriever.get_entry(entry_id)
        if source is None:
            raise LookupError(f"entry not found: {entry_id}")

        items = [
            ContextItem(
                result=source,
                labels=(ContextLabel.DIRECT_EVIDENCE, ContextLabel.INFERENCE_INPUT),
                depth=0,
            )
        ]
        if link_depth == 0:
            return ContextExpansion(source_entry_id=entry_id, items=tuple(items))

        candidates, diagnostics = self._candidates(source)
        queue = deque((candidate, 1) for candidate in candidates)
        visited = {(source.note_id, None)}
        inspected: list[str] = []
        seen_items = {source.entry_id}
        continuations: list[SectionContinuation] = []
        anchor = _evidence_date(source)
        anchor_end = date.fromisoformat(source.entry_date_end) if source.entry_date_end else anchor

        def section_matches(candidate: _Candidate) -> tuple[SearchResult, ...]:
            return self.retriever.metadata_all(
                QueryPlan(mode=RetrievalMode.METADATA, note_id=candidate.note_id, section=candidate.section)
            )

        def continue_section(candidate: _Candidate, omitted: tuple[SearchResult, ...]) -> None:
            if omitted and candidate.section is not None:
                continuations.append(
                    SectionContinuation(
                        candidate.note_id,
                        candidate.section,
                        omitted[0].start_line,
                        omitted[-1].end_line,
                    )
                )

        while queue:
            candidate, depth = queue.popleft()
            key = (candidate.note_id, candidate.section)
            if key in visited or depth > link_depth:
                continue
            if len(items) >= max_items or (
                candidate.note_id not in inspected and len(inspected) >= max_notes
            ):
                reasons.add("max_items" if len(items) >= max_items else "max_notes")
                # Disclose heading targets that the budget prevented visiting.
                pending = [(candidate, depth), *queue]
                for remaining, remaining_depth in pending:
                    remaining_key = (remaining.note_id, remaining.section)
                    if remaining_key in visited or remaining_depth > link_depth:
                        continue
                    visited.add(remaining_key)
                    if remaining.section is not None:
                        continue_section(remaining, section_matches(remaining))
                break
            visited.add(key)
            if candidate.note_id not in inspected:
                inspected.append(candidate.note_id)

            # Explicit anchors select every indexed entry in source order,
            # including descendants. Plain links still select one passage.
            matches = (
                section_matches(candidate)
                if candidate.section is not None
                else self.retriever.search_within(
                    candidate.note_id,
                    source.text,
                    mode=mode if source.text.strip() else RetrievalMode.METADATA,
                    limit=1,
                )
            )
            if not matches:
                continue
            selected: list[SearchResult] = []
            for position, match in enumerate(matches):
                if match.entry_id in seen_items:
                    selected.append(match)
                    continue
                if len(items) >= max_items:
                    reasons.add("max_items")
                    continue_section(candidate, matches[position:])
                    break
                items.append(
                    ContextItem(
                        result=match,
                        labels=(candidate.label, ContextLabel.INFERENCE_INPUT),
                        depth=depth,
                        via_note_id=candidate.via_note_id,
                    )
                )
                seen_items.add(match.entry_id)
                selected.append(match)

            if depth < link_depth:
                for match in selected:
                    next_candidates, next_diagnostics = self._candidates(match)
                    diagnostics.extend(next_diagnostics)
                    queue.extend((next_candidate, depth + 1) for next_candidate in next_candidates)

            if anchor is not None and any(
                match.entry_type is EntryType.PROJECT_UPDATE or match.entry_date is not None
                for match in selected
            ):
                for history in self._nearby_history(candidate.note_id, anchor, nearby_days, anchor_end):
                    if history.entry_id in seen_items:
                        continue
                    if len(items) >= max_items:
                        reasons.add("max_items")
                        break
                    items.append(
                        ContextItem(
                            result=history,
                            labels=(ContextLabel.NEARBY_HISTORY, ContextLabel.INFERENCE_INPUT),
                            depth=depth,
                            via_note_id=candidate.via_note_id,
                        )
                    )
                    seen_items.add(history.entry_id)

        return ContextExpansion(
            source_entry_id=entry_id,
            items=tuple(items),
            diagnostics=tuple(dict.fromkeys(diagnostics)),
            inspected_note_ids=tuple(inspected),
            truncated=bool(reasons),
            budget_reasons=tuple(sorted(reasons)),
            section_continuations=tuple(dict.fromkeys(continuations)),
        )

    def _candidates(self, result: SearchResult) -> tuple[tuple[_Candidate, ...], list[TraversalDiagnostic]]:
        counts: defaultdict[tuple[str, ContextLabel, str | None], int] = defaultdict(int)
        diagnostics: list[TraversalDiagnostic] = []
        outgoing_ids: set[str] = set()
        for link in result.outgoing_links:
            if link.resolution is LinkResolution.RESOLVED and link.target_note_id is not None:
                key = (link.target_note_id, ContextLabel.FOLLOWED_LINK, link.target_heading)
                counts[key] += 1
                outgoing_ids.add(link.target_note_id)
            else:
                diagnostics.append(
                    TraversalDiagnostic(
                        source_note_id=result.note_id,
                        target_text=link.target_text,
                        resolution=link.resolution,
                        line=link.line,
                    )
                )

        for backlink in self.retriever.get_backlinks(result.note_id):
            if backlink.note_id in outgoing_ids:
                continue
            references = sum(
                1
                for link in backlink.outgoing_links
                if link.resolution is LinkResolution.RESOLVED and link.target_note_id == result.note_id
            )
            if references:
                counts[(backlink.note_id, ContextLabel.BACKLINK, None)] += references

        candidates = [
            _Candidate(
                note_id,
                label,
                result.note_id,
                references,
                section,
            )
            for (note_id, label, section), references in counts.items()
        ]
        candidates.sort(
            key=lambda candidate: (
                0 if candidate.label is ContextLabel.FOLLOWED_LINK else 1,
                -candidate.references,
                candidate.note_id,
                candidate.section or "",
            )
        )
        return tuple(candidates), diagnostics

    def _nearby_history(
        self, note_id: str, anchor: date, nearby_days: int, anchor_end: date | None = None
    ) -> tuple[SearchResult, ...]:
        start = (anchor - timedelta(days=nearby_days)).isoformat()
        end = ((anchor_end or anchor) + timedelta(days=nearby_days)).isoformat()
        results = self.retriever.metadata_all(
            QueryPlan(
                mode=RetrievalMode.METADATA,
                note_id=note_id,
                date_from=start,
                date_to=end,
                entry_types=(EntryType.PROJECT_UPDATE,),
            )
        )
        return tuple(sorted(results, key=lambda item: (_distance(item, anchor), item.start_line)))


def _validate_budgets(link_depth: int, nearby_days: int, max_notes: int) -> None:
    if not 0 <= link_depth <= 2:
        raise ValueError("link_depth must be between 0 and 2")
    if nearby_days < 0:
        raise ValueError("nearby_days must be non-negative")
    if not 1 <= max_notes <= 5:
        raise ValueError("max_notes must be between 1 and 5")


def _evidence_date(result: SearchResult) -> date | None:
    value = result.entry_date or result.note_date
    return date.fromisoformat(value) if value is not None else None


def _distance(result: SearchResult, anchor: date) -> int:
    value = _evidence_date(result)
    if value is None:
        return 10**9
    end = date.fromisoformat(result.entry_date_end) if result.entry_date_end else value
    if value <= anchor <= end:
        return 0
    return min(abs((value - anchor).days), abs((end - anchor).days))
