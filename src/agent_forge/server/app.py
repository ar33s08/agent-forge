"""FastAPI service: run the agent over HTTP and keep the run store queryable.

Design notes (interview talking points):
- The app is a *factory*, not a module-level singleton (except the demo APP below):
  tests inject a scripted MockProvider factory and an in-memory RunStore, so no test
  ever touches the network. The provider is built FRESH per request because the mock
  is script-stateful — a shared instance would leak scripts between requests.
- Auth is a single dependency: absent/empty `api_keys` means dev mode (no auth), which
  is fine locally but must be a deliberate choice at the deployment edge.
- Rate limiting is a coarse in-memory token bucket keyed by (api key, minute). It is
  per-process by design; a real multi-worker deployment moves it to redis.
"""
from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable, Sequence

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from agent_forge import __version__
from agent_forge.agent import Agent, AgentConfig
from agent_forge.models import ToolCall
from agent_forge.providers.base import LlmProvider, LlmResponse
from agent_forge.providers.mock import MockProvider
from agent_forge.sessions import RunStore
from agent_forge.tools import ToolRegistry


class RunRequest(BaseModel):
    goal: str= Field(min_length=1)
    session_id: str | None= None


def create_app(
    provider_factory: Callable[[], LlmProvider],
    tools: ToolRegistry,
    *,
    api_keys: Sequence[str] | None= None,
    rate_limit_per_min: int | None= None,
    agent_config: AgentConfig | None= None,
    store_path: str= ":memory:",
) -> FastAPI:
    """Build the service around an injectable provider factory + tool registry."""
    app= FastAPI(title="agent-forge", version=__version__)
    store= RunStore(store_path)

    auth_enabled= bool(api_keys)
    allowed= set(api_keys or ())
    _buckets: dict[tuple[str, int], int]= {}
    _buckets_lock= threading.Lock()

    def guard(x_api_key: str | None= Header(default=None, alias="X-API-Key")) -> None:
        # 1) authentication — dev mode when no keys were configured
        if auth_enabled and (not x_api_key or x_api_key not in allowed):
                raise HTTPException(
                    status_code=401,
                    detail="missing or invalid X-API-Key",
                )
        # 2) coarse per-(key, minute) token bucket
        if rate_limit_per_min is not None:
            minute= int(time.time() // 60)
            bucket= (x_api_key or "-", minute)
            with _buckets_lock:
                used= _buckets.get(bucket, 0)
                if used>= rate_limit_per_min:
                    raise HTTPException(
                        status_code=429, detail="rate limit exceeded, retry next minute"
                    )
                _buckets[bucket]= used + 1
                if len(_buckets)> 10_000:  # keep the map from growing unbounded
                    for old in [k for k in _buckets if k[1]< minute - 1]:
                        del _buckets[old]

    guard_deps= [Depends(guard)]

    @app.get("/healthz", dependencies=guard_deps)
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/runs", status_code=201, dependencies=guard_deps)
    def post_run(req: RunRequest) -> dict:
        agent= Agent(provider_factory(), tools, agent_config)
        run= agent.run(req.goal)
        store.save(run, session_id=req.session_id)
        return run.model_dump(mode="json")

    @app.get("/v1/runs/{run_id}", dependencies=guard_deps)
    def get_run(run_id: str) -> dict:
        run= store.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return run.model_dump(mode="json")

    @app.get("/v1/sessions/{session_id}", dependencies=guard_deps)
    def get_session(session_id: str) -> list[dict]:
        return [r.model_dump(mode="json") for r in store.session(session_id)]

    @app.get("/v1/stats", dependencies=guard_deps)
    def stats() -> dict[str, float]:
        return store.totals()

    app.state.store= store
    return app


# --------------------------------------------------------------------- demo app
# Module-level instance so `uvicorn agent_forge.server.app:APP` just works.
# The provider factory mints a fresh scripted MockProvider per request: a small
# counter lets the demo answer goals without any network access.
_demo_counter= itertools.count(1)


def _demo_provider() -> LlmProvider:
    n= next(_demo_counter)
    return MockProvider(
        [
            LlmResponse(
                content=None,
                tool_calls=[ToolCall(name="add", arguments={"a": n, "b": 40})],
                tokens_in=4,
                tokens_out=2,
            ),
            LlmResponse(content=f"demo answer {n}: got your goal", tokens_in=6, tokens_out=8),
        ]
    )


def _demo_tools() -> ToolRegistry:
    reg= ToolRegistry()

    @reg.register
    def add(a: int, b: int) -> str:
        """Add two integers."""
        return str(a + b)

    return reg


APP= create_app(_demo_provider, _demo_tools())
app= APP  # lowercase alias for `uvicorn agent_forge.server.app:app`
