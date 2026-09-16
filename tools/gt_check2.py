
def read(p):
    try:
        return open(p).read()
    except Exception as e:
        return "<<MISSING:" + str(e)
ci = read(".github/workflows/ci.yml")
co = read("compose.yaml")
dk = read("Dockerfile")
li = read("LICENSE")
gi = read(".gitignore")
print("CI ok:", all(s in ci for s in ("pull_request:", "actions/checkout@v4", "astral-sh/setup-uv@v6", "uv sync --extra dev", "uv run pytest -q", "uv run ruff check", "scripts/run_eval.py", "python -m agent_forge.cli")), "| bad key count:", ci.count("pull_request"), "| 'uses:' intact:", "uses:" in ci and "uses=" not in ci)
print("COMPOSE ok:", all(s in co for s in ("services:", "volumes:", "127.0.0.1:8080:8080", "restart: unless-stopped", "AGENTFORGE_API_KEYS", "build: .")))
print("DOCKER ok:", all(s in dk for s in ("FROM python:3.11-slim", "pip install", "agent_forge.server.app:APP", "EXPOSE 8080", "HEALTHCHECK", "useradd", "CMD [")), "| FROM count:", dk.count("FROM"), "| uvicorn:", "uvicorn" in dk)
print("LICENSE ok:", ("Apache License" in li) and ("Version 2.0" in li) and ("apache.org" in li) and ("AS IS" in li), "| ApacheLicensecount:", li.count("Apache License"))
print("GITIGNORE keys:", [s for s in ("venv/", "pycache/", "pycache", "egg-info", "dist/", "build/") if s in gi], "| ruffline:", "ruff" in gi.lower())
