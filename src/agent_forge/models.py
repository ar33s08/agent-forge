"""Core pydantic schemas shared across AgentForge."""
from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class ToolCall(BaseModel):
    id: str= Field(default_factory=lambda: new_id("tc"))
    name: str
    arguments: dict[str, Any]= Field(default_factory=dict)


class ToolResult(BaseModel):
    call_id: str
    name: str
    ok: bool= True
    output: str
    latency_ms: float= 0.0


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str= ""
    name: str | None= None
    tool_call_id: str | None= None


class StopReason(StrEnum):
    COMPLETED= "completed"
    MAX_STEPS= "max_steps"
    TOOL_ERROR= "tool_error"
    PROVIDER_ERROR= "provider_error"


class RunStatus(StrEnum):
    RUNNING= "running"
    SUCCEEDED= "succeeded"
    FAILED= "failed"


class TraceStep(BaseModel):
    seq: int
    kind: Literal["llm", "tool"]
    name: str
    input_summary: str= ""
    output_summary: str= ""
    tokens_in: int= 0
    tokens_out: int= 0
    cost_usd: float= 0.0
    latency_ms: float= 0.0


class Run(BaseModel):
    id: str= Field(default_factory=lambda: new_id("run"))
    goal: str
    system_prompt: str | None= None
    status: RunStatus= RunStatus.RUNNING
    steps: list[TraceStep]= Field(default_factory=list)
    tool_calls: list[ToolCall]= Field(default_factory=list)
    final_output: str | None= None
    stop_reason: StopReason | None= None
    error: str | None= None
    total_tokens_in: int= 0
    total_tokens_out: int= 0
    total_cost_usd: float= 0.0
    created_at: float= Field(default_factory=time.time)
    finished_at: float | None= None
