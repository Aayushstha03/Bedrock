from __future__ import annotations

import io
import json

from bedrock.core.config import Settings
from bedrock.runtime.cli import _handler
from bedrock.vault.graph import graph_snapshot


def settings(tmp_path):
    return Settings("http://unused", tmp_path, tmp_path, "entries", "127.0.0.1", 8080)


def test_snapshot_includes_empty_notes_all_tags_and_link_evidence(tmp_path):
    (tmp_path / "Empty.md").write_text("")
    (tmp_path / "Target.md").write_text("---\naliases: [Destination]\n---\n")
    (tmp_path / "Hub.md").write_text(
        "---\ntags: [custom, 'team/backend']\n---\n"
        "# Hub [[Empty]]\n"
        "#custom #team/backend #unconfigured\n"
        "[[Destination#Details|alias]] [[Destination]] [[Absent]]\n"
        "`#inline [[Ignored]]`\n```\n#code [[Ignored]]\n```\n"
    )
    first = graph_snapshot(settings(tmp_path))
    assert first == graph_snapshot(settings(tmp_path))
    nodes = first["nodes"]
    assert len([n for n in nodes if n["type"] == "note"]) == 3
    assert {n["label"] for n in nodes if n["type"] == "tag"} == {"#custom", "#team/backend", "#unconfigured"}
    assert [n["label"] for n in nodes if n["type"] == "missing"] == ["Absent"]
    assert all(
        e["source"] in {n["id"] for n in nodes} and e["target"] in {n["id"] for n in nodes}
        for e in first["edges"]
    )
    links = [e for e in first["edges"] if e["type"] == "link"]
    assert len(links) == 3
    repeated = next(e for e in links if len(e["occurrences"]) == 2)
    assert repeated["occurrences"][0] == {"line": 6, "heading": "Details", "alias": "alias"}


def test_snapshot_ambiguity_exclusions_and_empty_vault(tmp_path):
    assert graph_snapshot(settings(tmp_path))["nodes"] == []
    for folder in ("a", "b", ".obsidian"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "Shared.md").write_text("")
    (tmp_path / "Hub.md").write_text("[[Shared]] [[a/Shared]]")
    snapshot = graph_snapshot(settings(tmp_path))
    assert len(snapshot["nodes"]) == 3
    assert len(snapshot["edges"]) == 1
    assert snapshot["unresolved_links"][0]["resolution"] == "ambiguous"


def response(config, path):
    handler = _handler(config).__new__(_handler(config))
    handler.path = path
    handler.headers = {}
    handler.wfile = io.BytesIO()
    statuses = []
    handler.send_response = statuses.append
    handler.send_header = lambda *args: None
    handler.end_headers = lambda: None
    handler.do_GET()
    return statuses[0], json.loads(handler.wfile.getvalue())


def test_snapshot_http_returns_json_without_qdrant(tmp_path):
    (tmp_path / "Empty.md").write_text("")
    status, body = response(settings(tmp_path), "/graph/snapshot")
    assert status == 200
    assert body["schema_version"] == 1
    assert body["nodes"][0]["path"] == "Empty.md"
    assert body["edges"] == []


def test_snapshot_http_unavailable_vault_is_not_empty(tmp_path):
    status, body = response(settings(tmp_path / "missing"), "/graph/snapshot")
    assert status == 503
    assert body == {"error": "vault snapshot unavailable"}


def test_snapshot_rebuilds_only_on_debounced_markdown_changes(monkeypatch, tmp_path):
    from watchdog.events import FileCreatedEvent, FileDeletedEvent, FileModifiedEvent, FileMovedEvent

    from bedrock.parser.watcher import MarkdownChangeHandler
    from bedrock.vault.graph import GraphSnapshotCache

    builds = []
    original = graph_snapshot

    def build(config):
        builds.append(True)
        return original(config)

    monkeypatch.setattr("bedrock.vault.graph.graph_snapshot", build)
    config = settings(tmp_path)
    cache = GraphSnapshotCache(config)
    handler = MarkdownChangeHandler(config, cache.rebuild, debounce_seconds=60)
    try:
        handler.catch_up()
        initial = cache.get()
        assert cache.get() is initial
        assert len(builds) == 1

        note = tmp_path / "Note.md"
        note.write_text("#custom")
        handler.on_any_event(FileCreatedEvent(str(note)))
        handler.on_any_event(FileModifiedEvent(str(note)))
        handler.flush()
        updated = cache.get()
        assert len(builds) == 2
        assert updated is not initial
        assert any(n["label"] == "#custom" for n in updated["nodes"])
        assert cache.get() is updated

        renamed = tmp_path / "Renamed.md"
        note.rename(renamed)
        handler.on_any_event(FileMovedEvent(str(note), str(renamed)))
        handler.flush()
        assert any(n.get("path") == "Renamed.md" for n in cache.get()["nodes"])

        renamed.unlink()
        handler.on_any_event(FileDeletedEvent(str(renamed)))
        handler.flush()
        assert cache.get()["nodes"] == []
        assert len(builds) == 4

        handler.on_any_event(FileCreatedEvent(str(tmp_path / "image.png")))
        handler.on_any_event(FileCreatedEvent(str(tmp_path / ".obsidian" / "ignored.md")))
        handler.flush()
        cache.get()
        assert len(builds) == 4
    finally:
        handler.close()


def test_failed_rebuild_does_not_serve_stale_success_and_recovers(monkeypatch, tmp_path):
    import pytest

    from bedrock.vault.graph import GraphSnapshotCache

    cache = GraphSnapshotCache(settings(tmp_path))
    cache.rebuild()
    original = graph_snapshot

    def fail(config):
        raise OSError("unavailable")

    monkeypatch.setattr("bedrock.vault.graph.graph_snapshot", fail)
    with pytest.raises(OSError):
        cache.rebuild(("Note.md",))
    with pytest.raises(OSError):
        cache.get()
    monkeypatch.setattr("bedrock.vault.graph.graph_snapshot", original)
    cache.rebuild(("Note.md",))
    assert cache.get()["nodes"] == []


def test_cosmograph_flat_tables_match_snapshot_and_download_routes(tmp_path):
    from bedrock.vault.graph import cosmograph_data

    (tmp_path / "Hub.md").write_text("#custom [[Target]] [[Target]]")
    (tmp_path / "Target.md").write_text("")
    (tmp_path / "Isolated.md").write_text("")
    config = settings(tmp_path)
    snapshot = graph_snapshot(config)
    data = cosmograph_data(snapshot)
    assert {p["id"] for p in data["points"]} == {n["id"] for n in snapshot["nodes"]}
    assert all(
        not isinstance(v, (dict, list)) for row in data["points"] + data["links"] for v in row.values()
    )
    assert next(p for p in data["points"] if p["label"] == "Isolated")["degree"] == 0
    assert next(p for p in data["points"] if p["label"] == "Hub")["degree"] == 2
    assert next(link for link in data["links"] if link["type"] == "link")["weight"] == 2
    point_ids = {p["id"] for p in data["points"]}
    assert all(link["source"] in point_ids and link["target"] in point_ids for link in data["links"])
    assert response(config, "/graph/cosmograph") == (200, data)
    assert response(config, "/graph/cosmograph/points.json") == (200, data["points"])
    assert response(config, "/graph/cosmograph/links.json") == (200, data["links"])


def test_cosmograph_tables_are_cached_and_refreshed_with_snapshot(tmp_path):
    from bedrock.vault.graph import GraphSnapshotCache

    cache = GraphSnapshotCache(settings(tmp_path))
    first = cache.get_cosmograph()
    assert cache.get_cosmograph() is first
    (tmp_path / "Note.md").write_text("#new")
    assert cache.get_cosmograph() is first
    cache.rebuild(("Note.md",))
    updated = cache.get_cosmograph()
    assert updated is not first
    assert len(updated["points"]) == 2
    assert len(updated["links"]) == 1


def test_cosmograph_json_routes_send_download_headers(tmp_path):
    handler_type = _handler(settings(tmp_path))
    handler = handler_type.__new__(handler_type)
    handler.path = "/graph/cosmograph/points.json"
    handler.headers = {}
    handler.wfile = io.BytesIO()
    headers = {}
    handler.send_response = lambda status: None
    handler.send_header = lambda key, value: headers.update({key: value})
    handler.end_headers = lambda: None
    handler.do_GET()
    assert headers["Content-Disposition"] == 'attachment; filename="points.json"'
    assert json.loads(handler.wfile.getvalue()) == []


def http_response(handler_type, path, validator=None):
    handler = handler_type.__new__(handler_type)
    handler.path = path
    handler.headers = {} if validator is None else {"If-None-Match": validator}
    handler.wfile = io.BytesIO()
    headers = {}
    statuses = []
    handler.send_response = statuses.append
    handler.send_header = lambda key, value: headers.update({key: value})
    handler.end_headers = lambda: None
    handler.do_GET()
    return statuses[0], headers, handler.wfile.getvalue()


def test_graph_routes_support_conditional_get_without_serializing_again(monkeypatch, tmp_path):
    import hashlib

    from bedrock.vault.graph import GraphSnapshotCache

    cache = GraphSnapshotCache(settings(tmp_path))
    cache.rebuild()
    handler = _handler(settings(tmp_path), cache)

    def fail(*args, **kwargs):
        raise AssertionError("request must reuse encoded response and ETag")

    monkeypatch.setattr("bedrock.vault.graph.json.dumps", fail)
    for path in (
        "/graph/snapshot",
        "/graph/cosmograph",
        "/graph/cosmograph/points.json",
        "/graph/cosmograph/links.json",
    ):
        status, headers, body = http_response(handler, path)
        assert status == 200
        etag = headers["ETag"]
        assert etag == f'"{hashlib.sha256(body).hexdigest()}"'
        assert headers["Cache-Control"] == "private, no-cache"
        assert int(headers["Content-Length"]) == len(body)
        for validator in (etag, f"W/{etag}", f'"old", W/{etag}', "*"):
            status, not_modified, body = http_response(handler, path, validator)
            assert status == 304
            assert body == b""
            assert not_modified["ETag"] == etag
            assert not_modified["Cache-Control"] == headers["Cache-Control"]
            assert "Content-Length" not in not_modified
        assert http_response(handler, path, '"stale"')[0] == 200


def test_etag_changes_only_when_response_changes_and_survives_restart(tmp_path):
    from bedrock.vault.graph import GraphSnapshotCache

    note = tmp_path / "Note.md"
    note.write_text("hello")
    config = settings(tmp_path)
    cache = GraphSnapshotCache(config)
    handler = _handler(config, cache)
    _, headers, _ = http_response(handler, "/graph/snapshot")
    etag = headers["ETag"]
    note.write_text("changed prose without changing graph")
    cache.rebuild(("Note.md",))
    assert http_response(handler, "/graph/snapshot", etag)[0] == 304
    assert http_response(_handler(config), "/graph/snapshot", etag)[0] == 304
    note.write_text("#newtag")
    cache.rebuild(("Note.md",))
    status, headers, body = http_response(handler, "/graph/snapshot", etag)
    assert status == 200
    assert headers["ETag"] != etag
    assert any(node["label"] == "#newtag" for node in json.loads(body)["nodes"])


def test_conditional_get_does_not_hide_unavailable_vault(tmp_path):
    status, headers, body = http_response(_handler(settings(tmp_path / "missing")), "/graph/snapshot", "*")
    assert status == 503
    assert "ETag" not in headers
    assert json.loads(body) == {"error": "vault snapshot unavailable"}
