FROM python:3.12-slim-bookworm

ARG APP_UID=1000
ARG APP_GID=1000

LABEL org.opencontainers.image.title="SRHarness" \
      org.opencontainers.image.description="A harness for agentic symbolic regression" \
      org.opencontainers.image.url="https://github.com/yuzhTHU/SRHarness" \
      org.opencontainers.image.source="https://github.com/yuzhTHU/SRHarness" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache

RUN groupadd --gid "${APP_GID}" srharness \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin srharness

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --no-cache-dir '.[tools]' \
    && python -m pip check

RUN install -d -o "${APP_UID}" -g "${APP_GID}" \
        /data \
        /usr/local/lib/python3.12/site-packages/sr_harness/skills/custom

USER srharness
WORKDIR /data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import socket; connection = socket.create_connection(('127.0.0.1', 8000), timeout=3); connection.close()"

ENTRYPOINT ["sr-harness"]
CMD ["run", "--save-path", "/data", "--workspace-dir", "/data", "--isolate-users", "--host", "0.0.0.0", "--port", "8000"]
