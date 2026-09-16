# Service image: one-command up, healthcheck included.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONDONTSAVE=1 PIPNODIRWARN=1

WORKDIR /srv/agent-forge
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir ".[serve]"

# Non-root runtime user (defense in depth; no secrets are ever baked in).
RUN useradd --create-home --shell /usr/sbin/nologin agentforge
USER agentforge
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status==200 else 1)"

CMD ["uvicorn", "agent_forge.server.app:APP", "--host", "0.0.0.0", "--port", "8080", "--log-level", "warning"]
