"""Deterministic scripted provider — the workhorse for tests and CI evals."""
from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

from agent_forge.models import ChatMessage, ToolCall
from agent_forge.providers.base import LlmResponse


class MockProvider:
    """Replays a script of LlmResponses and records every prompt it was shown."""

    name= "mock"

    def __init__(self, script: Iterable[LlmResponse] | None= None):
        self._script: list[LlmResponse]= list(script or [])
        self._i= 0
        self.prompts: list[list[ChatMessage]]= []
        self.tool_specs_seen: list[list[dict] | None]= []

    # -- ergonomic constructors -------------------------------------------------
    @staticmethod
    def say(text: str, tokens_in: int= 10, tokens_out: int= 10) -> LlmResponse:
        return LlmResponse(content=text, tokens_in=tokens_in, tokens_out=tokens_out)

    @staticmethod
    def say_tools(*calls: ToolCall, tokens_in: int= 10, tokens_out: int= 5) -> LlmResponse:
        return LlmResponse(
            content=None, tool_calls=list(calls), tokens_in=tokens_in, tokens_out=tokens_out
        )

    # -- provider contract ------------------------------------------------------
    def complete(
        self, messages: list[ChatMessage], tools: list[dict] | None= None
    ) -> LlmResponse:
        self.prompts.append(deepcopy(messages))
        self.tool_specs_seen.append(deepcopy(tools))
        if self._i>= len(self._script):
            # Script exhausted — behave like a well-behaved model wrapping up.
            return LlmResponse(content="(script exhausted)", tokens_in=1, tokens_out=2)
        response= self._script[self._i]
        self._i += 1
        return deepcopy(response)


class FlakyProvider(MockProvider):
    """Fails `fail_times` times (with the given exception) before replaying the script."""

    def __init__(
        self,
        script: Iterable[LlmResponse] | None= None,
        fail_times: int= 1,
        exc: Exception | None= None,
    ):
        super().__init__(script)
        self._fail_times= fail_times
        self._exc= exc or RuntimeError("provider 503")

    def complete(
        self, messages: list[ChatMessage], tools: list[dict] | None= None
    ) -> LlmResponse:
        if self._fail_times> 0:
            self._fail_times -= 1
            raise deepcopy(self._exc)
        return super().complete(messages, tools)
