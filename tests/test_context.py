from __future__ import annotations

from dataclasses import replace

import pytest

from bedrock.core.contracts import (
    ContextLabel,
    EntryType,
    Link,
    LinkResolution,
    NoteType,
    RetrievalMode,
    SearchResult,
)
from bedrock.retrieval.context import ContextExpander


def result(
    entry_id: str,
    note_id: str,
    *,
    note_date: str | None = None,
    entry_date: str | None = None,
    entry_type: EntryType = EntryType.FREEFORM_SECTION,
    links: tuple[Link, ...] = (),
) -> SearchResult:
    return SearchResult(
        entry_id=entry_id,
        note_id=note_id,
        path=f"{note_id}.md",
        note_title=note_id,
        note_type=NoteType.STANDARD,
        entry_type=entry_type,
        mode=RetrievalMode.METADATA,
        text=f"evidence from {entry_id}",
        heading=entry_id,
        heading_path=(entry_id,),
        start_line=1,
        end_line=2,
        note_date=note_date,
        entry_date=entry_date,
        tags=(),
        outgoing_links=links,
    )


class FakeRetriever:
    def __init__(self) -> None:
        self.source = result(
            "daily-entry",
            "daily",
            note_date="2026-08-20",
            links=(
                Link(
                    "Project",
                    "project",
                    target_heading="2026-08-19",
                    resolution=LinkResolution.RESOLVED,
                    line=3,
                ),
                Link("Gone", resolution=LinkResolution.MISSING, line=4),
                Link("Duplicate", resolution=LinkResolution.AMBIGUOUS, line=5),
            ),
        )
        self.project = result(
            "project-current",
            "project",
            entry_date="2026-08-19",
            entry_type=EntryType.PROJECT_UPDATE,
            links=(Link("Daily", "daily", resolution=LinkResolution.RESOLVED, line=8),),
        )
        self.backlink = result("backlink-entry", "backlink")
        self.history = result(
            "project-history",
            "project",
            entry_date="2026-08-17",
            entry_type=EntryType.PROJECT_UPDATE,
        )
        self.within_calls: list[tuple[str, str | None]] = []

    def get_entry(self, entry_id: str):
        return self.source if entry_id == self.source.entry_id else None

    def get_backlinks(self, note_id: str):
        if note_id == "daily":
            return (
                replace(
                    self.backlink,
                    outgoing_links=(Link("Daily", "daily", resolution=LinkResolution.RESOLVED, line=2),),
                ),
            )
        if note_id == "project":
            return (self.source,)
        return ()

    def search_within(self, note_id: str, query: str, *, section: str | None = None, **_: object):
        self.within_calls.append((note_id, section))
        return {
            "project": (self.project,),
            "backlink": (self.backlink,),
        }.get(note_id, ())

    def metadata_all(self, plan):
        if plan.section is not None:
            assert plan.section == "2026-08-19"
            return (self.project,)
        assert plan.note_id == "project"
        assert plan.entry_types == (EntryType.PROJECT_UPDATE,)
        assert (plan.date_from, plan.date_to) == ("2026-08-17", "2026-08-23")
        return (self.project, self.history)


def test_daily_to_project_expansion_labels_context_and_terminates_cycles() -> None:
    retriever = FakeRetriever()
    expansion = ContextExpander(retriever).expand("daily-entry", link_depth=2)

    by_id = {item.result.entry_id: item for item in expansion.items}
    assert by_id["daily-entry"].labels == (
        ContextLabel.DIRECT_EVIDENCE,
        ContextLabel.INFERENCE_INPUT,
    )
    assert ContextLabel.FOLLOWED_LINK in by_id["project-current"].labels
    assert ContextLabel.NEARBY_HISTORY in by_id["project-history"].labels
    assert ContextLabel.BACKLINK in by_id["backlink-entry"].labels
    assert expansion.inspected_note_ids == ("project", "backlink")
    assert retriever.within_calls == [
        ("backlink", None),
    ]
    assert {diagnostic.resolution for diagnostic in expansion.diagnostics} == {
        LinkResolution.MISSING,
        LinkResolution.AMBIGUOUS,
    }
    assert all(
        ContextLabel.DIRECT_EVIDENCE not in item.labels
        for item in expansion.items
        if item.result.entry_id != "daily-entry"
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"link_depth": 3}, "link_depth"),
        ({"nearby_days": -1}, "nearby_days"),
        ({"max_notes": 6}, "max_notes"),
    ],
)
def test_expansion_budgets_are_enforced(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ContextExpander(FakeRetriever()).expand("daily-entry", **kwargs)


def test_zero_depth_returns_only_direct_evidence() -> None:
    expansion = ContextExpander(FakeRetriever()).expand("daily-entry", link_depth=0)
    assert [item.result.entry_id for item in expansion.items] == ["daily-entry"]
    assert expansion.inspected_note_ids == ()


def test_context_caps_total_items_and_reports_history_omissions() -> None:
    expansion = ContextExpander(FakeRetriever()).expand("daily-entry", max_items=2)
    assert len(expansion.items) == 2
    assert expansion.truncated and "max_items" in expansion.budget_reasons


def test_context_reports_unvisited_notes() -> None:
    expansion = ContextExpander(FakeRetriever()).expand("daily-entry", max_notes=1)
    assert expansion.truncated and "max_notes" in expansion.budget_reasons


class SectionRetriever(FakeRetriever):
    def __init__(self):
        super().__init__()
        self.source = replace(
            self.source,
            note_date=None,
            outgoing_links=(
                Link("Project", "project", target_heading="Decision", resolution=LinkResolution.RESOLVED),
                Link("Project", "project", target_heading="Results", resolution=LinkResolution.RESOLVED),
            ),
        )
        self.sections = {
            "Decision": (
                replace(result("decision-1", "project"), start_line=4, end_line=7),
                replace(result("decision-child", "project"), start_line=9, end_line=12),
            ),
            "Results": (replace(result("results", "project"), start_line=15, end_line=19),),
        }

    def get_backlinks(self, note_id):
        return ()

    def metadata_all(self, plan):
        assert plan.mode is RetrievalMode.METADATA
        return self.sections[plan.section]


def test_explicit_links_include_all_chunks_and_distinct_headings_in_one_note():
    retriever = SectionRetriever()
    expansion = ContextExpander(retriever).expand("daily-entry", max_notes=1)
    assert [item.result.entry_id for item in expansion.items] == [
        "daily-entry",
        "decision-1",
        "decision-child",
        "results",
    ]
    assert expansion.inspected_note_ids == ("project",)
    assert not expansion.truncated
    assert not retriever.within_calls


def test_section_budget_returns_source_continuations_for_partial_and_deferred_sections():
    expansion = ContextExpander(SectionRetriever()).expand("daily-entry", max_items=2)
    assert [item.result.entry_id for item in expansion.items] == ["daily-entry", "decision-1"]
    assert expansion.truncated and "max_items" in expansion.budget_reasons
    assert [(c.section, c.start_line, c.end_line) for c in expansion.section_continuations] == [
        ("Decision", 9, 12),
        ("Results", 15, 19),
    ]


def test_links_from_later_section_chunks_are_traversed():
    retriever = SectionRetriever()
    first, child = retriever.sections["Decision"]
    retriever.sections["Decision"] = (
        first,
        replace(child, outgoing_links=(Link("Other", "other", resolution=LinkResolution.RESOLVED),)),
    )
    expansion = ContextExpander(retriever).expand("daily-entry", link_depth=2)
    assert "other" in expansion.inspected_note_ids
    assert ("other", None) in retriever.within_calls
