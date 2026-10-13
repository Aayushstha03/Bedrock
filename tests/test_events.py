from __future__ import annotations

import io
import json
from queue import Empty
from types import SimpleNamespace

import pytest
from watchdog.events import FileDeletedEvent, FileModifiedEvent, FileMovedEvent

from bedrock.core.config import Settings
from bedrock.parser.watcher import MarkdownChangeHandler
from bedrock.runtime.cli import _handler
from bedrock.vault.events import QUEUE_SIZE, EventBus, GraphUpdates, NoteChangePublisher, encode_sse
from bedrock.vault.graph import GraphSnapshotCache


def settings(tmp_path):
    return Settings("http://unused", tmp_path, tmp_path, "entries", "127.0.0.1", 8080)


def test_note_and_graph_notifications_follow_debounce_and_cache_publication(tmp_path):
    note = tmp_path / "Note.md"
    note.write_text("first prose")
    config = settings(tmp_path)
    bus = EventBus()
    cache = GraphSnapshotCache(config)
    GraphUpdates(cache, bus)
    publisher = NoteChangePublisher(config, bus)
    handler = MarkdownChangeHandler(config, publisher.changed, debounce_seconds=60)
    stream = bus.open_stream()
    seen = []
    bus.subscribe("graph.updated", lambda event: seen.append(cache.get_response("snapshot").etag))
    try:
        handler.catch_up()
        initial_notes = stream.get_nowait()
        initial_graph = stream.get_nowait()
        assert initial_notes["type"] == "notes.changed"
        assert initial_graph["type"] == "graph.updated"
        assert initial_graph["etag"] == seen[-1]
        assert initial_notes["sequence"] < initial_graph["sequence"]

        note.write_text("only prose changes")
        handler.on_any_event(FileModifiedEvent(str(note)))
        handler.on_any_event(FileModifiedEvent(str(note)))
        assert stream.empty()
        handler.flush()
        changed = stream.get_nowait()
        assert changed["type"] == "notes.changed"
        assert changed["paths"] == ["Note.md"]
        assert changed["deleted_paths"] == []
        assert stream.empty()  # Prose edit does not change graph ETag.

        note.write_text("#newtag")
        handler.on_any_event(FileModifiedEvent(str(note)))
        handler.flush()
        assert stream.get_nowait()["type"] == "notes.changed"
        graph = stream.get_nowait()
        assert graph["type"] == "graph.updated"
        assert graph["etag"] == cache.get_response("snapshot").etag == seen[-1]
        assert graph["etag"] != initial_graph["etag"]
    finally:
        handler.close()
        bus.close()


def test_rename_delete_and_full_rescan_notifications(tmp_path):
    old = tmp_path / "Old.md"
    old.write_text("")
    config = settings(tmp_path)
    bus = EventBus()
    publisher = NoteChangePublisher(config, bus)
    stream = bus.open_stream()
    handler = MarkdownChangeHandler(config, publisher.changed, debounce_seconds=60)
    try:
        new = tmp_path / "New.md"
        old.rename(new)
        handler.on_any_event(FileMovedEvent(str(old), str(new)))
        handler.flush()
        event = stream.get_nowait()
        assert event["paths"] == ["New.md"]
        assert event["deleted_paths"] == ["Old.md"]
        new.unlink()
        handler.on_any_event(FileDeletedEvent(str(new)))
        handler.flush()
        assert stream.get_nowait()["deleted_paths"] == ["New.md"]
        new.write_text("")
        publisher.changed(())
        assert stream.get_nowait()["rescan"] is True
        new.unlink()
        publisher.changed(())
        event = stream.get_nowait()
        assert event["rescan"] is True
        assert event["deleted_paths"] == ["New.md"]
    finally:
        handler.close()
        bus.close()


def test_failed_graph_build_only_publishes_notes_and_can_retry(monkeypatch, tmp_path):
    bus = EventBus()
    cache = GraphSnapshotCache(settings(tmp_path))
    GraphUpdates(cache, bus)
    publisher = NoteChangePublisher(settings(tmp_path), bus)
    stream = bus.open_stream()
    original = cache.rebuild

    def fail(paths):
        raise OSError("cannot read vault")

    monkeypatch.setattr(cache, "rebuild", fail)
    with pytest.raises(OSError):
        publisher.changed(())
    assert stream.get_nowait()["type"] == "notes.changed"
    assert stream.empty()
    monkeypatch.setattr(cache, "rebuild", original)
    publisher.changed(())
    assert stream.get_nowait()["type"] == "notes.changed"
    assert stream.get_nowait()["type"] == "graph.updated"
    bus.close()


def test_bus_fanout_sequences_disconnect_and_backpressure():
    bus = EventBus()
    one, two = bus.open_stream(), bus.open_stream()
    first = bus.publish("notes.changed", paths=["Note.md"], deleted_paths=[])
    assert one.get_nowait() == two.get_nowait() == first
    bus.close_stream(one)
    for _ in range(QUEUE_SIZE + 1):
        bus.publish("notes.changed", paths=[], deleted_paths=[])
    assert one.empty()
    assert two.get_nowait() is None  # Overflow closes stream, prompting resync.
    fresh = bus.open_stream()
    later = bus.publish("graph.updated", etag='"etag"')
    assert later["sequence"] > first["sequence"]
    assert fresh.get_nowait() == later
    bus.close()
    assert fresh.get_nowait() is None
    assert bus.open_stream().get_nowait() is None


def test_sse_encoding_contains_named_event_and_json():
    event = {"type": "notes.changed", "sequence": 42, "paths": ["a\n名字.md"], "deleted_paths": []}
    frame = encode_sse(event).decode()
    assert frame.startswith("id: 42\nevent: notes.changed\ndata: ")
    assert frame.endswith("\n\n")
    assert json.loads(frame.split("data: ", 1)[1]) == event
    assert len(frame.splitlines()) == 4  # Paths cannot inject SSE fields.


@pytest.mark.parametrize("disconnect", [False, True])
def test_http_sse_stream_heartbeat_headers_and_cleanup(tmp_path, disconnect):
    bus = EventBus()
    event = {"type": "graph.updated", "sequence": 1, "etag": '"abc"'}
    frames = iter([event, Empty(), None])
    closed = []

    class Stream:
        def get(self, timeout):
            frame = next(frames)
            if isinstance(frame, Empty):
                raise frame
            return frame

    stream = Stream()
    bus.open_stream = lambda: stream
    bus.close_stream = closed.append
    handler_type = _handler(settings(tmp_path), events=bus)
    handler = handler_type.__new__(handler_type)
    handler.path = "/events"
    handler.connection = SimpleNamespace(settimeout=lambda timeout: None)
    statuses, headers = [], {}
    handler.send_response = statuses.append
    handler.send_header = lambda key, value: headers.update({key: value})
    handler.end_headers = lambda: None

    class Output(io.BytesIO):
        def write(self, body):
            if disconnect:
                raise BrokenPipeError
            return super().write(body)

    handler.wfile = Output()
    handler.do_GET()
    assert statuses == [200]
    assert headers["Content-Type"] == "text/event-stream; charset=utf-8"
    assert headers["X-Accel-Buffering"] == "no"
    assert "Content-Length" not in headers
    assert closed == [stream]
    if not disconnect:
        assert encode_sse(event) in handler.wfile.getvalue()
        assert b": heartbeat\n\n" in handler.wfile.getvalue()
