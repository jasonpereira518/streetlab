FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
# Hosted: a private simulation per WebSocket connection, at most this many at
# once (see docs/hosted-sessions.md for how 2 was chosen). 0 = one shared world.
ENV STREETLAB_MAX_SESSIONS=2
CMD ["streetlab", "serve", "--host", "0.0.0.0", "--port", "8080", "--source", "osm", "--no-stdin-watchdog"]
