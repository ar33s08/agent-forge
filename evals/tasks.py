"""Declarative eval tasks and report models for the eval-as-CI harness.

An EvalTask is a fully deterministic scenario: the goal handed to the agent, a
scripted MockProvider transcript, and the assertions we gate on (substrings of
the final answer, the multiset of tools the loop actually executed, budgets).
Scripts are stored as plain dict payloads so task packs stay JSON-serializable;
they are hydrated into LlmResponse dataclasses at validation time.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agent_forge.models import ToolCall
from agent_forge.providers.base import LlmResponse


def hydrate_response(payload: Any) -> LlmResponse:
    """Accept an LlmResponse as-is or a plain dict and return the dataclass."""
    if isinstance(payload, LlmResponse):
        return payload
    if isinstance(payload, dict):
        calls= [
            c if isinstance(c, ToolCall) else ToolCall.model_validate(c)
            for c in payload.get("tool_calls", [])
        ]
        return LlmResponse(
            content=payload.get("content"),
            tool_calls=calls,
            tokens_in=int(payload.get("tokens_in", 0)),
            tokens_out=int(payload.get("tokens_out", 0)),
        )
    raise TypeError(f"cannot build LlmResponse from {type(payload).__name__}")


class EvalTask(BaseModel):
    """One golden scenario: scripted provider transcript plus pass assertions."""

    name: str
    goal: str
    provider_script: list[LlmResponse]= Field(default_factory=list)
    expect_contains: list[str]= Field(default_factory=list)
    expect_tool_names: list[str]= Field(default_factory=list)
    max_steps_budget: int= 8
    max_cost_usd: float | None= None

    @field_validator("provider_script", mode="before")
    @classmethod
    def _hydrate_script(cls, value: Any) -> Any:
        if value is None:
            return []
        return [hydrate_response(item) for item in value]

    # -- scoring helpers (pure; the harness consumes them) --------------------
    def expected_multiset(self) -> Counter:
        return Counter(self.expect_tool_names)

    def executed_multiset(self, names: list[str]) -> Counter:
        return Counter(names)

    def contains_ok(self, final_output: str | None) -> bool:
        if final_output is None:
            return not self.expect_contains
        hay= final_output.casefold()
        return all(needle.casefold() in hay for needle in self.expect_contains)


class TaskOutcome(BaseModel):
    """Per-task grading line with the metrics the CI gate aggregates."""

    name: str
    passed: bool
    reason: str= ""
    status: str= ""
    stop_reason: str= ""
    tools_expected: list[str]= Field(default_factory=list)
    tools_executed: list[str]= Field(default_factory=list)
    tool_selection_ok: bool= True
    budget_ok: bool= True
    completed: bool= False
    cost_usd: float= 0.0
    latency_ms: float= 0.0


class EvalReport(BaseModel):
    """Aggregate of TaskOutcome lines; this is what report.json holds."""

    task_count: int= 0
    passed: int= 0
    failed: int= 0
    pass_rate: float= 0.0
    tool_selection_accuracy: float= 0.0
    budget_compliance_rate: float= 0.0
    completed_rate: float= 0.0
    mean_latency_ms: float= 0.0
    outcomes: list[TaskOutcome]= Field(default_factory=list)

    @classmethod
    def from_outcomes(cls, outcomes: list[TaskOutcome]) -> EvalReport:
        n= len(outcomes)
        done= n or 1
        return cls(
            task_count=n,
            passed=sum(1 for o in outcomes if o.passed),
            failed=sum(1 for o in outcomes if not o.passed),
            pass_rate=sum(1 for o in outcomes if o.passed) / done,
            tool_selection_accuracy=sum(1 for o in outcomes if o.tool_selection_ok) / done,
            budget_compliance_rate=sum(1 for o in outcomes if o.budget_ok) / done,
            completed_rate=sum(1 for o in outcomes if o.completed) / done,
            mean_latency_ms=(sum(o.latency_ms for o in outcomes) / done) if n else 0.0,
            outcomes=outcomes,
        )

    def to_json(self) -> str:
        import json

        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)
