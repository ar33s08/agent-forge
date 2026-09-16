"""HTTP-level tests for the agent-forge service — scripted mocks, no network."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

SRC= Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient

from agent_forge.models import ToolCall
from agent_forge.providers.base import LlmResponse
from agent_forge.providers.mock import MockProvider
from agent_forge.server.app import create_app
from agent_forge.tools import ToolRegistry


def _tools() -> ToolRegistry:
    reg= ToolRegistry()

    @reg.register
    def add(a: int, b: int) -> str:
        """Add two integers."""
        return str(a + b)

    return reg


def _say_factory(prefix: str= "ok"):
    counter= itertools.count(1)

    def factory() -> MockProvider:
        return MockProvider([LlmResponse(content=f"{prefix}-{next(counter)}")])

    return factory


def _tool_then_say():
    def factory() -> MockProvider:
        return MockProvider(
            [
                LlmResponse(content=None, tool_calls=[ToolCall(name="add", arguments={"a": 2, "b": 3})]),
                LlmResponse(content="total 5"),
            ]
        )

    return factory


def test_healthz_and_dev_mode_no_auth() -> None:
    client= TestClient(create_app(_say_factory(), _tools()))
    r= client.get("/healthz")
    assert r.status_code== 200
    body= r.json()
    assert body["status"]== "ok"
    assert "version" in body


def test_auth_required_401_without_header_200_with() -> None:
    app= create_app(_say_factory(), _tools(), api_keys=["secret"])
    client= TestClient(app)
    assert client.get("/healthz").status_code== 401
    r= client.get("/healthz", headers={"X-API-Key": "secret"})
    assert r.status_code== 200


def test_post_run_returns_201_and_persists() -> None:
    client= TestClient(create_app(_say_factory("fine"), _tools()))
    r= client.post("/v1/runs", json={"goal": "say hi"})
    assert r.status_code== 201
    run= r.json()
    assert run["final_output"]== "fine-1"
    assert run["status"]== "succeeded"
    got= client.get(f"/v1/runs/{run['id']}")
    assert got.status_code== 200
    assert got.json()["id"]== run["id"]


def test_tool_call_flow() -> None:
    client= TestClient(create_app(_tool_then_say(), _tools()))
    r= client.post("/v1/runs", json={"goal": "add 2 and 3"})
    assert r.status_code== 201
    run= r.json()
    assert run["final_output"]== "total 5"
    assert run["tool_calls"][0]["name"]== "add"


def test_unknown_run_id_404() -> None:
    client= TestClient(create_app(_say_factory(), _tools()))
    r= client.get("/v1/runs/nope")
    assert r.status_code== 404
    assert r.json()["detail"]== "run not found"


def test_session_history_chronological() -> None:
    client= TestClient(create_app(_say_factory("s"), _tools()))
    first= client.post("/v1/runs", json={"goal": "one", "session_id": "sess-1"}).json()
    second= client.post("/v1/runs", json={"goal": "two", "session_id": "sess-1"}).json()
    rows= client.get("/v1/sessions/sess-1").json()
    assert [r["id"] for r in rows]== [first["id"], second["id"]]


def test_stats_totals_increment() -> None:
    client= TestClient(create_app(_say_factory(), _tools()))
    assert client.get("/v1/stats").json()["runs"]== 0
    client.post("/v1/runs", json={"goal": "count me"})
    assert client.get("/v1/stats").json()["runs"]== 1


def test_rate_limit_third_request_429() -> None:
    app= create_app(
        _say_factory(), _tools(), api_keys=["key-1"], rate_limit_per_min=2
    )
    client= TestClient(app)
    headers= {"X-API-Key": "key-1"}
    assert client.get("/healthz", headers=headers).status_code== 200
    assert client.get("/healthz", headers=headers).status_code== 200
    assert client.get("/healthz", headers=headers).status_code== 429
