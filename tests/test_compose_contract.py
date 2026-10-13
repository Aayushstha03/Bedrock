from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_compose_contract_is_local_persistent_and_read_only() -> None:
    compose = (ROOT / "compose.yaml").read_text()
    assert "image: ${QDRANT_IMAGE}" in compose
    assert '"${QDRANT_BIND_ADDRESS}:${QDRANT_PORT}:6333"' in compose
    assert "qdrant-data:/qdrant/storage" in compose
    assert "model-cache:/models" in compose
    assert "read_only: true" in compose
    assert "internal: true" in compose
    assert 'QDRANT__TELEMETRY_DISABLED: "true"' in compose
    assert "dashboard:" in compose
    assert "condition: service_healthy" in compose
    assert 'profiles: ["tools"]' in compose
    assert 'command: ["bedrock", "models", "prefetch"]' in compose
    # LM Studio runs on the host, so the bedrock service needs a routed network
    # and a host gateway. Qdrant and the index path stay unroutable.
    assert '"host.docker.internal:host-gateway"' in compose
    assert compose.count("internal: true") == 1


def test_every_service_runs_the_working_tree_rather_than_the_baked_copy() -> None:
    """One `docker compose up` runs the source on disk, with no second file.

    The image installs the project, so it still runs with nothing mounted. The
    mount only wins because `PYTHONPATH` puts it ahead of site-packages, and
    the two have to stay together: the mount alone is shadowed by the install,
    and `PYTHONPATH` alone points at a directory that is not there.
    """
    compose = (ROOT / "compose.yaml").read_text()
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "PYTHONPATH=/app/src" in dockerfile
    # Every service that runs the `bedrock` script reads the same source.
    assert compose.count("target: /app/src") == 3
    assert compose.count("source: ./src") == 3
    # The container runs as uid 10001. Writing into the working tree would
    # leave files behind that the host user cannot edit.
    for block in compose.split("- type: bind")[1:]:
        if "target: /app/src" in block:
            assert "read_only: true" in block


def test_the_watcher_runs_as_its_own_service_off_the_routed_network() -> None:
    """Indexing must not depend on the answering model being reachable."""
    compose = (ROOT / "compose.yaml").read_text()
    assert 'command: ["bedrock", "watch"]' in compose
    watch = compose.split("  watch:", 1)[1].split("\n  model-prefetch:", 1)[0]
    assert "condition: service_healthy" in watch
    assert "read_only: true" in watch
    assert "target: /app/src" in watch
    assert "model-cache:/models" in watch
    assert "bedrock-internal" in watch
    # The watcher parses and indexes only. Joining `llm` would give the one
    # long-running vault reader outbound network access it has no use for.
    assert "- llm" not in watch
    assert "host.docker.internal" not in watch


def test_documented_env_contains_all_compose_settings() -> None:
    env = (ROOT / ".env").read_text()
    example = (ROOT / ".env.example").read_text()
    required = {
        "QDRANT_IMAGE",
        "QDRANT_BIND_ADDRESS",
        "QDRANT_PORT",
        "BEDROCK_VAULT_PATH",
        "BEDROCK_COLLECTION",
        "BEDROCK_DENSE_MODEL",
        "BEDROCK_SPARSE_MODEL",
        "BEDROCK_LOG_ROOT",
        "BEDROCK_KANBAN_ROOT",
        "BEDROCK_EXCLUDED_DIRECTORIES",
        "BEDROCK_KNOWN_TAGS",
        "BEDROCK_TAG_ALIASES",
        "RESTART_POLICY",
        "BEDROCK_LLM_BASE_URL",
        "BEDROCK_LLM_MODEL",
        "BEDROCK_LLM_API_KEY",
        "BEDROCK_LLM_TIMEOUT",
        "BEDROCK_LLM_TEMPERATURE",
    }
    for name in required:
        assert f"{name}=" in env
        assert f"{name}=" in example


def test_the_image_installs_every_declared_dependency() -> None:
    """The Dockerfile repeats the dependency list, so it can drift from it.

    The runtime stage installs the project with `--no-deps`, which means a
    dependency added to pyproject.toml and not added here is simply absent
    from the image, and only fails when the code path that needs it runs.
    """
    import re
    import tomllib

    declared = set(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"])
    pinned = set(
        re.findall(r"^\s+\"?([A-Za-z0-9_.\[\]-]+==[0-9][^\s\\\"]*)", (ROOT / "Dockerfile").read_text(), re.M)
    )
    assert declared == pinned, f"Dockerfile and pyproject disagree: {declared ^ pinned}"


def test_runtime_has_no_sqlite_or_cloud_inference_dependency() -> None:
    runtime_files = [*(ROOT / "src").rglob("*.py"), ROOT / "pyproject.toml", ROOT / "compose.yaml"]
    runtime = "\n".join(path.read_text().lower() for path in runtime_files)
    assert "sqlite" not in runtime
    assert "fts5" not in runtime
    assert "cloud_inference" not in runtime


def test_mcp_service_reuses_read_only_vault_and_publishes_loopback() -> None:
    import yaml

    services = yaml.safe_load((ROOT / "compose.yaml").read_text())["services"]
    mcp = services["mcp"]
    assert mcp["environment"] == services["bedrock"]["environment"]
    assert mcp["volumes"] == services["bedrock"]["volumes"]
    assert "127.0.0.1" in mcp["ports"][0]
    assert "streamable-http" in mcp["command"]
    assert "-T" not in mcp["command"]
    assert "bedrock-internal" in mcp["networks"] and "llm" not in mcp["networks"]
