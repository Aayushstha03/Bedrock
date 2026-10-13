FROM python:3.13.7-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY pyproject.toml README.md ./
RUN python -m pip install --prefix=/install \
    fastembed==0.7.3 \
    markdown-it-py==4.0.0 \
    mcp==1.30.0 \
    "pydantic-ai-slim[openai]==2.35.0" \
    pyyaml==6.0.3 \
    qdrant-client==1.15.1 \
    watchdog==6.0.0
COPY src ./src
RUN python -m pip install --prefix=/install --no-deps .

FROM python:3.13.7-slim-bookworm AS runtime

# `PYTHONPATH` is what lets the working tree win. The project is installed
# into site-packages below, so the image runs on its own with nothing mounted.
# A PYTHONPATH entry sits ahead of site-packages in `sys.path`, so when compose
# binds the source at /app/src the container runs that instead, and the
# `bedrock` console script keeps working either way.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    BEDROCK_QDRANT_URL=http://qdrant:6333 \
    BEDROCK_VAULT_PATH=/vault \
    BEDROCK_MODEL_CACHE_PATH=/models \
    BEDROCK_HOST=0.0.0.0 \
    BEDROCK_PORT=8080

RUN groupadd --system --gid 10001 bedrock \
    && useradd --system --uid 10001 --gid bedrock --home-dir /app bedrock \
    && mkdir -p /app /models \
    && chown -R bedrock:bedrock /app /models

COPY --from=builder /install /usr/local
WORKDIR /app
USER bedrock

EXPOSE 8080
CMD ["bedrock", "serve"]
