"""Watchdog-based, debounced Markdown change observation."""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from threading import Lock, Timer, current_thread
from time import monotonic

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from bedrock.core.config import Settings
from bedrock.parser.discovery import INTERNAL_DIRECTORIES

ChangeCallback = Callable[[tuple[str, ...]], None]
ErrorCallback = Callable[[BaseException], None]

DEBOUNCE_SECONDS = 2.0
MAX_DELAY_SECONDS = 10.0
MAX_RETRY_SECONDS = 60.0


def report_to_stderr(error: BaseException) -> None:
    traceback.print_exception(error, file=sys.stderr)


class MarkdownChangeHandler(FileSystemEventHandler):
    def __init__(
        self,
        settings: Settings,
        callback: ChangeCallback,
        debounce_seconds: float = DEBOUNCE_SECONDS,
        max_delay_seconds: float = MAX_DELAY_SECONDS,
        on_error: ErrorCallback = report_to_stderr,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.callback = callback
        self.on_error = on_error
        self.debounce_seconds = debounce_seconds
        # Continuous activity must not postpone indexing forever, so the
        # sliding debounce never delays a pending change past this ceiling.
        self.max_delay_seconds = max(max_delay_seconds, debounce_seconds)
        self._pending: set[str] = set()
        self._full_pending = False
        self._dirty = False
        self._closed = False
        self._timer: Timer | None = None
        self._deadline: float | None = None
        self._retry_seconds = debounce_seconds
        self._lock = Lock()
        # One callback at a time: two concurrent index passes would each read
        # a half-written view of the collection and delete the other's points.
        self._callback_lock = Lock()

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type not in {"created", "modified", "deleted", "moved"}:
            return
        paths = [event.src_path]
        destination = getattr(event, "dest_path", None)
        if destination:
            paths.append(destination)
        if event.is_directory:
            # A directory removal may have no individual file-delete events.
            # Request full convergence for structural changes, but ignore the
            # directory modification events caused by ordinary file writes.
            if event.event_type == "modified" or not any(self._included_directory(path) for path in paths):
                return
            with self._lock:
                self._full_pending = True
                self._dirty = True
                self._arm(self.debounce_seconds)
            return
        accepted = [relative for path in paths if (relative := self._relative_markdown(path)) is not None]
        if not accepted:
            return
        with self._lock:
            self._pending.update(accepted)
            self._dirty = True
            self._arm(self.debounce_seconds)

    def catch_up(self) -> None:
        """Index once without waiting, so edits made while the watcher was down converge."""
        with self._lock:
            self._dirty = True
        self.flush()

    def flush(self) -> None:
        with self._lock:
            timer = self._timer
            dirty = self._dirty
            paths = () if self._full_pending else tuple(sorted(self._pending, key=str.casefold))
            self._pending.clear()
            self._full_pending = False
            self._dirty = False
            self._timer = None
            self._deadline = None
        if timer is not None and timer is not current_thread():
            timer.cancel()
        if dirty:
            self._run(paths)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self.flush()

    def _run(self, paths: tuple[str, ...]) -> None:
        with self._callback_lock:
            try:
                self.callback(paths)
            except Exception as error:
                self._retry(paths, error)
            else:
                with self._lock:
                    self._retry_seconds = self.debounce_seconds

    def _retry(self, paths: tuple[str, ...], error: Exception) -> None:
        """Keep the change pending and try again, so a transient failure loses nothing."""
        self.on_error(error)
        with self._lock:
            if self._closed:
                return
            self._pending.update(paths)
            self._full_pending = self._full_pending or not paths
            self._dirty = True
            delay = self._retry_seconds
            self._retry_seconds = min(self._retry_seconds * 2, MAX_RETRY_SECONDS)
            self._arm(delay, cap=False)

    def _arm(self, delay: float, *, cap: bool = True) -> None:
        """Schedule the next flush. The caller holds the lock."""
        if cap:
            now = monotonic()
            if self._deadline is None:
                self._deadline = now + self.max_delay_seconds
            delay = min(delay, max(self._deadline - now, 0.0))
        if self._timer is not None:
            self._timer.cancel()
        self._timer = Timer(delay, self.flush)
        self._timer.daemon = True
        self._timer.start()

    def _relative_markdown(self, raw_path: str) -> str | None:
        path = Path(raw_path)
        if path.suffix.casefold() != ".md":
            return None
        try:
            relative = path.resolve().relative_to(self.settings.vault_path.resolve())
        except ValueError:
            return None
        excluded = {name.casefold() for name in self.settings.excluded_directories} | INTERNAL_DIRECTORIES
        if any(part.casefold() in excluded for part in relative.parts[:-1]):
            return None
        if relative.name.casefold() in {name.casefold() for name in self.settings.excluded_files}:
            return None
        return relative.as_posix()

    def _included_directory(self, raw_path: str) -> bool:
        try:
            relative = Path(raw_path).resolve().relative_to(self.settings.vault_path.resolve())
        except ValueError:
            return False
        excluded = {name.casefold() for name in self.settings.excluded_directories} | INTERNAL_DIRECTORIES
        return not any(part.casefold() in excluded for part in relative.parts)


class VaultWatcher:
    def __init__(
        self,
        settings: Settings,
        callback: ChangeCallback,
        debounce_seconds: float = DEBOUNCE_SECONDS,
        max_delay_seconds: float = MAX_DELAY_SECONDS,
        on_error: ErrorCallback = report_to_stderr,
    ) -> None:
        self.handler = MarkdownChangeHandler(
            settings, callback, debounce_seconds, max_delay_seconds, on_error
        )
        self.observer = Observer()
        self.observer.schedule(self.handler, str(settings.vault_path), recursive=True)

    def start(self) -> None:
        self.observer.start()

    def catch_up(self) -> None:
        self.handler.catch_up()

    def stop(self) -> None:
        self.observer.stop()
        self.observer.join()
        self.handler.close()

    def __enter__(self) -> VaultWatcher:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
