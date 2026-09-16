"""Offline tests for OpenAICompatibleProvider via httpx.MockTransport. No network."""
from __future__ import annotations

import json

import httpx
import pytest

from agent_forge.models import ChatMessage, ToolCall
from agent_forge.providers.openai import OpenAICompatibleProvider


MODEL= "gpt-test-4o"


def make_handler(status_code=200, payload=None, captured=None, side_effect=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured["request"]= request
        if side_effect is not None:
            raise side_effect
        body= payload if payload is not None else {
            "choices": [{"message": {"content": "hello world", "tool_calls": None}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }
        return httpx.Response(status_code, json=body, request=request)

    return httpx.MockTransport(handler=handler)


def make_provider(handler, **kw):
    return OpenAICompatibleProvider(model=MODEL, api_key="sk-test-123", transport=handler, **kw)


def simple_messages():
    return [
        ChatMessage(role="system", content="be nice"),
        ChatMessage(role="user", content="hi there"),
    ]


def test_happy_path_content_and_tokens():
    captured= {}
    provider= make_provider(make_handler(captured=captured))
    resp= provider.complete(simple_messages())
    assert resp.content== "hello world"
    assert resp.tool_calls== []
    assert resp.tokens_in== 11
    assert resp.tokens_out== 7

    req= captured["request"]
    assert str(req.url)== "https://api.openai.com/v1/chat/completions"
    sent= json.loads(req.content)
    assert sent["model"]== MODEL
    assert sent["messages"][0]== {"role": "system", "content": "be nice"}
    assert "tools" not in sent
    assert "tool_choice" not in sent
    assert "temperature" not in sent


def test_message_extras_serialized():
    captured= {}
    provider= make_provider(make_handler(captured=captured))
    msgs= [
        ChatMessage(role="user", content="yo", name="alice"),
        ChatMessage(role="tool", content="42", tool_call_id="tc_1"),
    ]
    provider.complete(msgs)
    sent= json.loads(captured["request"].content)
    assert sent["messages"][0]== {"role": "user", "content": "yo", "name": "alice"}
    assert sent["messages"][1]== {
        "role": "tool",
        "content": "42",
        "tool_call_id": "tc_1",
    }


def test_tools_and_temperature_in_payload():
    captured= {}
    provider= make_provider(make_handler(captured=captured), temperature=0.2)
    specs= [{"type": "function", "function": {"name": "add"}}]
    provider.complete(simple_messages(), tools=specs)
    sent= json.loads(captured["request"].content)
    assert sent["tools"]== specs
    assert sent["tool_choice"]== "auto"
    assert sent["temperature"]== 0.2


def test_tool_calls_parse_with_json_arguments():
    payload= {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_9",
                            "function": {"name": "lookup", "arguments": '{"city": "oslo"}'},
                        },
                        {
                            "function": {"name": "noop", "arguments": "{}"},
                        },
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
    provider= make_provider(make_handler(payload=payload))
    resp= provider.complete(simple_messages())
    assert resp.content is None
    assert resp.tool_calls== [
        ToolCall(id="tc_9", name="lookup", arguments={"city": "oslo"}),
        ToolCall(name="noop", arguments={}),
    ]
    assert resp.tool_calls[1].id  # auto-generated id
    assert (resp.tokens_in, resp.tokens_out)== (3, 2)


def test_bad_arguments_json_raises_value_error():
    payload= {
        "choices": [
            {"message": {"content": None, "tool_calls": [{"function": {"name": "x", "arguments": "not-json{"}}]}}
        ]
    }
    provider= make_provider(make_handler(payload=payload))
    with pytest.raises(ValueError, match="arguments not json"):
        provider.complete(simple_messages())


def test_missing_usage_falls_back_to_heuristic():
    payload= {"choices": [{"message": {"content": "a" * 40}}]}
    provider= make_provider(make_handler(payload=payload))
    resp= provider.complete([ChatMessage(role="user", content="b" * 12)])
    assert resp.tokens_out== 10  # 40 // 4
    assert resp.tokens_in== 3   # 12 // 4


def test_non_2xx_raises_runtime_error_naming_model():
    provider= make_provider(make_handler(status_code=503, payload={"error": "boom"}))
    with pytest.raises(RuntimeError) as exc_info:
        provider.complete(simple_messages())
    assert MODEL in str(exc_info.value)
    assert "provider request failed" in str(exc_info.value)


def test_missing_choices_raises_runtime_error():
    provider= make_provider(make_handler(payload={"oops": True}))
    with pytest.raises(RuntimeError, match="provider request failed"):
        provider.complete(simple_messages())


def test_transport_error_raises_runtime_error():
    provider= make_provider(make_handler(side_effect=httpx.ConnectError("dial tone missing")))
    with pytest.raises(RuntimeError, match="provider request failed"):
        provider.complete(simple_messages())


def test_authorization_header_present():
    captured= {}
    provider= make_provider(make_handler(captured=captured))
    provider.complete(simple_messages())
    assert captured["request"].headers["authorization"]== "Bearer sk-test-123"


def test_missing_api_key_raises_keyerror_naming_env_var(monkeypatch):
    monkeypatch.delenv("AGENTFORGE_OPENAI_API_KEY", raising=False)
    with pytest.raises(KeyError, match="AGENTFORGE_OPENAI_API_KEY"):
        OpenAICompatibleProvider.from_env(model=MODEL)


def test_from_env_wiring(monkeypatch):
    monkeypatch.delenv("AGENTFORGE_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("AGENTFORGE_OPENAI_MODEL", raising=False)
    monkeypatch.setenv("AGENTFORGE_OPENAI_BASE_URL", "https://local.llm/v1")
    monkeypatch.setenv("AGENTFORGE_OPENAI_MODEL", "env-model-7")
    monkeypatch.setenv("AGENTFORGE_OPENAI_API_KEY", "sk-env-1")
    provider= OpenAICompatibleProvider.from_env()
    assert provider.base_url== "https://local.llm/v1"
    assert provider.model== "env-model-7"
    assert provider.api_key== "sk-env-1"
    assert provider.name== "openai"

    captured= {}
    provider._client= httpx.Client(transport=make_handler(captured=captured))
    resp= provider.complete(simple_messages())
    assert resp.content== "hello world"
    assert json.loads(captured["request"].content)["model"]== "env-model-7"


def test_provider_satisfies_base_contract():
    from agent_forge.providers.base import LlmProvider

    provider= make_provider(make_handler())
    assert isinstance(provider, LlmProvider)
    assert provider.name== "openai"
