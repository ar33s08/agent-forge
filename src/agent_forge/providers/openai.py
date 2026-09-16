"""OpenAI-compatible HTTP provider: chat/completions over httpx, no SDK required."""
from __future__ import annotations

import json
import os

import httpx

from agent_forge.models import ChatMessage, ToolCall
from agent_forge.providers.base import LlmResponse

_API_KEY_ENV= "AGENTFORGE_OPENAI_API_KEY"
_BASE_URL_ENV= "AGENTFORGE_OPENAI_BASE_URL"
_MODEL_ENV= "AGENTFORGE_OPENAI_MODEL"


class OpenAICompatibleProvider:
    """Talks to any OpenAI-compatible ``/chat/completions`` endpoint.

    Transient failures (non-2xx, missing choices, transport errors) raise
    ``RuntimeError`` so callers can wrap ``complete`` in ``run_with_retries``.
    """

    name: str= "openai"

    def __init__(
        self,
        *,
        base_url: str= "https://api.openai.com/v1",
        model: str,
        api_key: str | None= None,
        timeout_s: float= 30.0,
        temperature: float | None= None,
        transport: httpx.BaseTransport | None= None,
        client: httpx.Client | None= None,
        name: str= "openai",
    ) -> None:
        self.base_url= base_url.rstrip("/")
        self.model= model
        self.api_key= api_key if api_key is not None else os.environ.get(_API_KEY_ENV)
        self.timeout_s= timeout_s
        self.temperature= temperature
        self._client= client
        self._transport= transport
        if name:
            self.name= name

    @classmethod
    def from_env(
        cls,
        base_url: str | None= None,
        model: str | None= None,
        **kw: object,
    ) -> "OpenAICompatibleProvider":
        """Build from AGENTFORGE_OPENAI_BASE_URL / _MODEL / _API_KEY env vars."""
        api_key= os.environ.get(_API_KEY_ENV)
        if not api_key:
            raise KeyError(f"missing api key; set env var {_API_KEY_ENV}")
        resolved_base= base_url if base_url is not None else os.environ.get(_BASE_URL_ENV)
        if resolved_base is None:
            resolved_base= "https://api.openai.com/v1"
        if model is None:
            model= os.environ.get(_MODEL_ENV)
        if not model:
            raise KeyError(f"missing model; set env var {_MODEL_ENV} or pass model=")
        return cls(base_url=resolved_base, model=str(model), api_key=api_key, **kw)

    # -- wire format ---------------------------------------------------------

    def _serialize_message(self, msg: ChatMessage) -> dict[str, str]:
        payload: dict[str, str]= {"role": msg.role, "content": msg.content}
        if msg.name is not None:
            payload["name"]= msg.name
        if msg.tool_call_id is not None:
            payload["tool_call_id"]= msg.tool_call_id
        return payload

    def _parse_tool_calls(self, raw: list | None) -> list[ToolCall]:
        calls: list[ToolCall]= []
        for item in raw or []:
            fn= item.get("function") or {}
            raw_args= fn.get("arguments") or "{}"
            if isinstance(raw_args, str):
                try:
                    arguments= json.loads(raw_args)
                except ValueError:
                    raise ValueError("arguments not json") from None
            else:
                arguments= raw_args
            call_kwargs: dict[str, object]= {"name": fn.get("name", ""), "arguments": arguments}
            if item.get("id"):
                call_kwargs["id"]= item["id"]
            calls.append(ToolCall(**call_kwargs))
        return calls

    @staticmethod
    def _heuristic_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    # -- the one method ------------------------------------------------------

    def complete(
        self, messages: list[ChatMessage], tools: list[dict] | None= None
    ) -> LlmResponse:
        payload: dict[str, object]= {
            "model": self.model,
            "messages": [self._serialize_message(m) for m in messages],
        }
        if tools:
            payload["tools"]= tools
            payload["tool_choice"]= "auto"
        if self.temperature is not None:
            payload["temperature"]= self.temperature

        headers= {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"]= f"Bearer {self.api_key}"

        if self._client is not None:
            client= self._client
            owns_client= False
        else:
            client= httpx.Client(transport=self._transport, timeout=self.timeout_s)
            owns_client= True
        try:
            try:
                response= client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
            except httpx.HTTPError as exc:
                raise RuntimeError(f"provider request failed: {exc}") from exc
        finally:
            if owns_client:
                client.close()

        if response.status_code< 200 or response.status_code>= 300:
            snippet= response.text[:200]
            raise RuntimeError(
                f"provider request failed: model {self.model!r} returned http {response.status_code}: {snippet}"
            )

        try:
            body= response.json()
        except ValueError as exc:
            raise RuntimeError(f"provider request failed: unparseable json body for model {self.model!r}") from exc

        choices= body.get("choices") if isinstance(body, dict) else None
        if not choices:
            raise RuntimeError(f"provider request failed: missing choices in response for model {self.model!r}")

        message= choices[0].get("message") or {}
        content= message.get("content")
        tool_calls= self._parse_tool_calls(message.get("tool_calls"))

        usage= body.get("usage") or {}
        tokens_in= int(usage.get("prompt_tokens") or usage.get("prompt_tokens_tokens") or 0)
        tokens_out= int(usage.get("completion_tokens") or 0)
        if not tokens_in:
            in_chars= sum(len(m.content) for m in messages)
            tokens_in= max(1, in_chars // 4) if in_chars else 1
        if not tokens_out:
            tokens_out= self._heuristic_tokens(content or "")
        return LlmResponse(content=content, tool_calls=tool_calls, tokens_in=tokens_in, tokens_out=tokens_out)
