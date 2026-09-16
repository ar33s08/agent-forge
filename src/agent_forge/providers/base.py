"""Provider contract: one synchronous `complete` call in, text-or-tool-calls out."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_forge.models import ChatMessage, ToolCall


@dataclass
class LlmResponse:
    content: str | None= None
    tool_calls: list[ToolCall]= field(default_factory=list)
    tokens_in: int= 0
    tokens_out: int= 0


@runtime_checkable
class LlmProvider(Protocol):
    """Anything that can turn a message list into a reply or a batch of tool calls.

    `tools` is the OpenAI-function-spec list produced by ToolRegistry.specs().
    Implementations are synchronous; callers may wrap them with retries/timeouts.
    """

    name: str

    def complete(
        self, messages: list[ChatMessage], tools: list[dict] | None= None
    ) -> LlmResponse: ...
