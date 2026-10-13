import time
from pathlib import Path
from threading import Lock, Thread

from watchdog.events import FileCreatedEvent, FileMovedEvent, FileOpenedEvent

from bedrock.core.config import Settings
from bedrock.parser.watcher import MarkdownChangeHandler


def settings(vault: Path) -> Settings:
    return Settings("http://qdrant:6333", vault, Path("/tmp/models"), "entries", "127.0.0.1", 8080)


def test_watcher_batches_normalized_markdown_paths(tmp_path: Path) -> None:
    received: list[tuple[str, ...]] = []
    handler = MarkdownChangeHandler(settings(tmp_path), received.append, debounce_seconds=60)
    first = tmp_path / "messy" / "Note.md"
    first.parent.mkdir()
    first.touch()
    second = tmp_path / "logs" / "2026-08-20.md"
    second.parent.mkdir()
    second.touch()

    handler.on_any_event(FileCreatedEvent(str(second)))
    handler.on_any_event(FileCreatedEvent(str(first)))
    handler.flush()

    assert received == [("logs/2026-08-20.md", "messy/Note.md")]
    handler.close()


def test_watcher_tracks_both_sides_of_rename_and_ignores_non_markdown(tmp_path: Path) -> None:
    received: list[tuple[str, ...]] = []
    handler = MarkdownChangeHandler(settings(tmp_path), received.append, debounce_seconds=60)
    old = tmp_path / "old.md"
    new = tmp_path / "nested" / "new.md"
    handler.on_any_event(FileMovedEvent(str(old), str(new)))
    handler.on_any_event(FileCreatedEvent(str(tmp_path / "image.png")))
    handler.on_any_event(FileOpenedEvent(str(tmp_path / "opened.md")))
    handler.flush()

    assert received == [("nested/new.md", "old.md")]
    handler.close()


def test_watcher_retries_a_failed_callback_instead_of_losing_the_change(tmp_path: Path) -> None:
    attempts: list[tuple[str, ...]] = []
    errors: list[BaseException] = []

    def callback(paths: tuple[str, ...]) -> None:
        attempts.append(paths)
        if len(attempts) == 1:
            raise RuntimeError("qdrant is down")

    handler = MarkdownChangeHandler(
        settings(tmp_path), callback, debounce_seconds=0.05, on_error=errors.append
    )
    handler.on_any_event(FileCreatedEvent(str(tmp_path / "note.md")))
    handler.flush()

    deadline = time.monotonic() + 5
    while len(attempts) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    handler.close()

    assert attempts == [("note.md",), ("note.md",)]
    assert [str(error) for error in errors] == ["qdrant is down"]


def test_watcher_never_runs_two_index_passes_at_once(tmp_path: Path) -> None:
    running = 0
    overlapped = False
    guard = Lock()

    def callback(_: tuple[str, ...]) -> None:
        nonlocal running, overlapped
        with guard:
            running += 1
            overlapped = overlapped or running > 1
        time.sleep(0.15)
        with guard:
            running -= 1

    handler = MarkdownChangeHandler(settings(tmp_path), callback, debounce_seconds=0.01)
    first = Thread(target=handler.flush)
    handler.on_any_event(FileCreatedEvent(str(tmp_path / "one.md")))
    first.start()
    handler.on_any_event(FileCreatedEvent(str(tmp_path / "two.md")))
    time.sleep(0.3)
    first.join()
    handler.close()

    assert not overlapped


def test_watcher_flushes_when_continuous_activity_reaches_the_ceiling(tmp_path: Path) -> None:
    received: list[tuple[str, ...]] = []
    handler = MarkdownChangeHandler(
        settings(tmp_path), received.append, debounce_seconds=0.2, max_delay_seconds=0.4
    )

    deadline = time.monotonic() + 1.2
    while time.monotonic() < deadline and not received:
        handler.on_any_event(FileCreatedEvent(str(tmp_path / "busy.md")))
        time.sleep(0.05)
    handler.close()

    assert received and received[0] == ("busy.md",)


def test_catch_up_indexes_once_without_any_event(tmp_path: Path) -> None:
    received: list[tuple[str, ...]] = []
    handler = MarkdownChangeHandler(settings(tmp_path), received.append, debounce_seconds=60)

    handler.catch_up()
    handler.close()

    assert received == [()]


def test_directory_structure_changes_request_full_convergence(tmp_path: Path) -> None:
    from watchdog.events import DirDeletedEvent, DirModifiedEvent, DirMovedEvent

    received: list[tuple[str, ...]] = []
    handler = MarkdownChangeHandler(settings(tmp_path), received.append, debounce_seconds=60)
    try:
        handler.on_any_event(FileCreatedEvent(str(tmp_path / "note.md")))
        handler.on_any_event(DirDeletedEvent(str(tmp_path / "removed")))
        handler.flush()
        assert received == [()]
        handler.on_any_event(DirModifiedEvent(str(tmp_path)))
        handler.on_any_event(DirDeletedEvent(str(tmp_path / ".obsidian")))
        handler.flush()
        assert received == [()]
        handler.on_any_event(DirMovedEvent(str(tmp_path / "before"), str(tmp_path / "after")))
        handler.flush()
        assert received == [(), ()]
    finally:
        handler.close()
