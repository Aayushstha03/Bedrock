"""Read lifecycle dates only from an explicit per-card bedrock-dates fence."""

from markdown_it.token import Token

from bedrock.core.contracts import IndexWarning, WarningType
from bedrock.core.task_dates import TASK_DATE_FIELDS, task_date_value


def lifecycle_dates(
    tokens: list[Token], lines: list[str], start: int, end: int, path: str
) -> tuple[dict[str, str], list[IndexWarning]]:
    blocks = [
        token
        for token in tokens
        if token.type == "fence"
        and token.info.strip() == "bedrock-dates"
        and token.map is not None
        and start <= token.map[0] < end
    ]
    warnings: list[IndexWarning] = []

    def warn(message: str, line: int) -> None:
        warnings.append(IndexWarning(WarningType.INVALID_KANBAN_DATES, path, message, line))

    if not blocks:
        return {}, warnings
    if len(blocks) > 1:
        warn("card has multiple bedrock-dates blocks; lifecycle dates ignored", start + 1)
        return {}, warnings
    block = blocks[0]
    first, last = block.map
    closing = lines[last - 1].strip()
    if last > end or len(closing) < len(block.markup) or set(closing) != {block.markup[0]}:
        warn("bedrock-dates block must have a closing fence inside its card", first + 1)
        return {}, warnings
    values: dict[str, str] = {}
    seen: set[str] = set()
    for offset, raw in enumerate(block.content.splitlines(), start=first + 2):
        if not raw.strip():
            continue
        key, separator, value = raw.partition(":")
        key, value = key.strip(), value.strip()
        if not separator or key not in TASK_DATE_FIELDS:
            warn(f"unknown or malformed lifecycle date field: {raw.strip()}", offset)
            continue
        if key in seen:
            values.pop(key, None)
            warn(f"duplicate lifecycle date field: {key}; value ignored", offset)
            continue
        seen.add(key)
        if not value:
            continue
        try:
            values[key] = task_date_value(key, value)
        except ValueError as error:
            warn(f"invalid {key}: {value}: {error}", offset)
    return values, warnings
