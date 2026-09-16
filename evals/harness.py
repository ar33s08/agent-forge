"""EvalHarness: run every EvalTask through a fresh Agent and grade the run.

Isolation rule: each task gets a brand-new ToolRegistry and a brand-new
MockProvider hydrated from the task's own script dicts, so no state can leak
between tasks and results are reproducible across machines.
"""
from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

from agent_forge.agent import Agent, AgentConfig
from agent_forge.providers.mock import MockProvider
from agent_forge.retry import RetryPolicy
from agent_forge.tools import ToolRegistry
from evals.tasks import EvalReport, EvalTask, TaskOutcome


class EvalHarness:
    def __init__(
        self,
        tasks: list[EvalTask],
        registry_factory: Callable[[], ToolRegistry],
        *,
        price_per_1k_in: float= 0.01,
        price_per_1k_out: float= 0.02,
    ):
        self.tasks= list(tasks)
        self.registry_factory= registry_factory
        self.price_per_1k_in= price_per_1k_in
        self.price_per_1k_out= price_per_1k_out

    # ------------------------------------------------------------------ public
    def run_task(self, task: EvalTask) -> TaskOutcome:
        """Execute one task end-to-end and grade it into a TaskOutcome."""
        registry= self.registry_factory()
        provider= MockProvider([_clone(r) for r in task.provider_script])
        config= AgentConfig(
            max_steps=task.max_steps_budget,
            max_cost_usd=task.max_cost_usd,
            price_per_1k_in=self.price_per_1k_in,
            price_per_1k_out=self.price_per_1k_out,
            retry=RetryPolicy(max_attempts=2, base_delay_s=0.001),
        )
        agent= Agent(provider=provider, tools=registry, config=config)

        started= perf_counter()
        run= agent.run(task.goal)
        latency_ms= (perf_counter() - started) * 1000

        executed= [call.name for call in run.tool_calls]
        tool_ok= (
            task.expected_multiset()== task.executed_multiset(executed)
        )
        contains_ok= task.contains_ok(run.final_output)
        succeeded= run.status.value== "succeeded" if hasattr(run.status, "value") else str(
            run.status
        )== "succeeded"
        budget_ok= succeeded and (
            task.max_cost_usd is None
            or run.total_cost_usd<= task.max_cost_usd + 1e-9
        )
        completed= _stop_value(run.stop_reason)== "completed"

        reasons: list[str]= []
        if not succeeded:
            reasons.append(f"status={_status_value(run.status)}")
        if not contains_ok:
            reasons.append("expect_contains miss")
        if not tool_ok:
            reasons.append(f"tool multiset mismatch: expected={sorted(task.expect_tool_names)} executed={sorted(executed)}")
        if not budget_ok:
            reasons.append(
                f"budget breach: cost={run.total_cost_usd:.6f} limit={task.max_cost_usd}"
            )

        return TaskOutcome(
            name=task.name,
            passed=succeeded and contains_ok and tool_ok and budget_ok,
            reason="; ".join(reasons),
            status=_status_value(run.status),
            stop_reason=_stop_value(run.stop_reason),
            tools_expected=list(task.expect_tool_names),
            tools_executed=executed,
            tool_selection_ok=tool_ok,
            budget_ok=budget_ok,
            completed=completed,
            cost_usd=run.total_cost_usd,
            latency_ms=latency_ms,
        )

    def run_all(self) -> EvalReport:
        return EvalReport.from_outcomes([self.run_task(t) for t in self.tasks])


# ---------------------------------------------------------------------- utils
def _clone(response: Any):
    from copy import deepcopy

    return deepcopy(response)


def _enum_value(obj: Any) -> str:
    return str(getattr(obj, "value", obj))


def _status_value(status: Any) -> str:
    return "" if status is None else _enum_value(status)


def _stop_value(stop: Any) -> str:
    return "" if stop is None else _enum_value(stop)
