"""Ephemeral vault notifications, independent of HTTP transport."""

from __future__ import annotations

import json
from collections.abc import Callable
from queue import Empty, Full, Queue
from threading import Lock
from typing import Any

from bedrock.core.config import Settings
from bedrock.parser.discovery import discover_markdown
from bedrock.vault.graph import GraphSnapshotCache

Notification = dict[str, Any]
Subscriber = Callable[[Notification], None]
HEARTBEAT_SECONDS = 15
QUEUE_SIZE = 64


class EventBus:
    """Ordered notifications with bounded queues; slow clients reconnect to resync."""

    def __init__(self) -> None:
        self._sequence = 0
        self._lock = Lock()
        self._streams: set[Queue[Notification | None]] = set()
        self._subscribers: dict[str, list[Subscriber]] = {}
        self._closed = False

    def subscribe(self, event_type: str, callback: Subscriber) -> None:
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    def open_stream(self) -> Queue[Notification | None]:
        stream: Queue[Notification | None] = Queue(maxsize=QUEUE_SIZE)
        with self._lock:
            if self._closed:
                stream.put_nowait(None)
            else:
                self._streams.add(stream)
        return stream

    def close_stream(self, stream: Queue[Notification | None]) -> None:
        with self._lock:
            self._streams.discard(stream)

    @staticmethod
    def _end_stream(stream: Queue[Notification | None]) -> None:
        # The publisher holds the bus lock, so no other producer can refill it.
        while True:
            try:
                stream.get_nowait()
            except Empty:
                break
        stream.put_nowait(None)

    def publish(self, event_type: str, **data: Any) -> Notification:
        with self._lock:
            if self._closed:
                raise RuntimeError("event bus is closed")
            self._sequence += 1
            event = {**data, "type": event_type, "sequence": self._sequence}
            for stream in tuple(self._streams):
                try:
                    stream.put_nowait(event)
                except Full:
                    self._streams.remove(stream)
                    self._end_stream(stream)
            callbacks = tuple(self._subscribers.get(event_type, []))
        # Consumers see notes.changed before graph preparation begins. Callback
        # failures propagate to the existing watcher retry mechanism.
        for callback in callbacks:
            callback(event)
        return event

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for stream in self._streams:
                self._end_stream(stream)
            self._streams.clear()


def encode_sse(event: Notification) -> bytes:
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event['sequence']}\nevent: {event['type']}\ndata: {payload}\n\n".encode()


class NoteChangePublisher:
    """Translate a debounced watcher batch into current/deleted Markdown paths."""

    def __init__(self, settings: Settings, bus: EventBus) -> None:
        self.settings = settings
        self.bus = bus
        self._known = self._paths()

    def _paths(self) -> set[str]:
        root = self.settings.vault_path.resolve()
        return {path.relative_to(root).as_posix() for path in discover_markdown(self.settings)}

    def changed(self, paths: tuple[str, ...]) -> None:
        if paths:
            current = {path for path in paths if (self.settings.vault_path / path).is_file()}
            deleted = set(paths) - current
            self._known.difference_update(deleted)
            self._known.update(current)
        else:
            # Structural changes and startup catch-up request a full refresh.
            current = self._paths()
            deleted = self._known - current
            self._known = current
        self.bus.publish(
            "notes.changed", paths=sorted(current), deleted_paths=sorted(deleted), rescan=not paths
        )


class GraphUpdates:
    """Rebuild after note notifications; publish only after a new ETag is readable."""

    def __init__(self, snapshots: GraphSnapshotCache, bus: EventBus) -> None:
        self.snapshots = snapshots
        self.bus = bus
        self._etag: str | None = None
        bus.subscribe("notes.changed", self.changed)

    def changed(self, event: Notification) -> None:
        self.snapshots.rebuild(tuple(event["paths"] + event["deleted_paths"]))
        etag = self.snapshots.get_response("snapshot").etag
        if etag != self._etag:
            self._etag = etag
            self.bus.publish("graph.updated", etag=etag)
