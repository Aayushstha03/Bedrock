"""Explicit Kanban lifecycle fields and timezone-aware search bounds."""

import re
from datetime import UTC, datetime, time

TASK_DATE_FIELDS = ("created_at", "started_at", "completed_at", "due_date")
DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
TIMESTAMP_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"
)


def task_date_value(field: str, value: str) -> str:
    """Canonical date or UTC timestamp; never infer a missing timezone."""
    if field == "due_date":
        if not DATE_PATTERN.fullmatch(value):
            raise ValueError("due_date must be YYYY-MM-DD")
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    if not TIMESTAMP_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must be an ISO timestamp with seconds and timezone")
    return datetime.fromisoformat(value).astimezone(UTC).isoformat()


def task_date_bound(field: str, value: str, *, end: bool = False) -> datetime:
    """Date-only bounds span the entire UTC day, including fractional seconds."""
    if DATE_PATTERN.fullmatch(value):
        day = datetime.strptime(value, "%Y-%m-%d").date()
        return datetime.combine(day, time.max if end else time.min, UTC)
    if field == "due_date":
        raise ValueError("due_date bounds must be YYYY-MM-DD")
    return datetime.fromisoformat(task_date_value(field, value))


def validate_task_date_ranges(plan: object) -> None:
    for field in TASK_DATE_FIELDS:
        start = getattr(plan, f"{field}_from")
        end = getattr(plan, f"{field}_to")
        lower = task_date_bound(field, start) if start is not None else None
        upper = task_date_bound(field, end, end=True) if end is not None else None
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"{field}_from must not be after {field}_to")
