"""The agent loop: prompt -> LLM -> tools -> feed results -> repeat, with budgets.

Design notes (interview talking points):
- The model never sees an unvalidated payload: every tool result is a ToolResult the
  loop produced, including hard failures (unknown tool, bad args, tool crash), which
  are fed back as observations instead of raising. An agent process must not die
  because a tool had a bad afternoon.
- Two independent kill switches: max_steps bounds the loop, max_cost_usd bounds the
  wallet. Either one stops the run cleanly with a final synthesis call.
- Retry/backoff lives in the provider call, not sprinkled through the loop.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from agent_forge.models import (
    ChatMessage,
    Run,
    RunStatus,
    StopReason,
    TraceStep,
)
from agent_forge.providers.base import LlmProvider
from agent_forge.retry import ProviderError, RetryPolicy, run_with_retries
from agent_forge.tools import ToolRegistry

DEFAULT_SYSTEM= (
    "You are a capable agent. Use tools when they help, then answer the goal directly. "
    "If information is missing, say so instead of inventing it."
)


@dataclass
class AgentConfig:
    max_steps: int= 8
    max_tool_calls: int= 24
    max_cost_usd: float | None= None
    price_per_1k_in: float= 0.0
    price_per_1k_out: float= 0.0
    retry: RetryPolicy= field(default_factory=RetryPolicy)
    system_prompt: str= DEFAULT_SYSTEM


class Agent:
    def __init__(
        self,
        provider: LlmProvider,
        tools: ToolRegistry,
        config: AgentConfig | None= None,
    ):
        self.provider= provider
        self.tools= tools
        self.config= config or AgentConfig()

    # ------------------------------------------------------------------ public
    def run(self, goal: str, *, system_prompt: str | None= None) -> Run:
        cfg= self.config
        run= Run(goal=goal, system_prompt=system_prompt or cfg.system_prompt)
        messages= [
            ChatMessage(role="system", content=run.system_prompt or ""),
            ChatMessage(role="user", content=goal),
        ]
        specs= self.tools.specs() or None
        tool_calls_used= 0
        seq= 0

        for step in range(cfg.max_steps):
            try:
                response, llm_ms= self._call(messages, specs)
            except ProviderError as exc:
                run.error= str(exc)
                run.status= RunStatus.FAILED
                run.stop_reason= StopReason.PROVIDER_ERROR
                run.finished_at= time.time()
                return run
            seq += 1
            cost= _cost(response.tokens_in, response.tokens_out, cfg)
            run.total_tokens_in += response.tokens_in
            run.total_tokens_out += response.tokens_out
            run.total_cost_usd += cost
            run.steps.append(TraceStep(
                seq=seq, kind="llm", name=self.provider.name,
                input_summary=_summary(messages[-1].content),
                output_summary=_summary(response.content or "[tool calls]"),
                tokens_in=response.tokens_in, tokens_out=response.tokens_out,
                cost_usd=cost, latency_ms=llm_ms,
            ))

            if cfg.max_cost_usd is not None and run.total_cost_usd>= cfg.max_cost_usd:
                return self._finish(run, messages, StopReason.MAX_STEPS)
            if response.tool_calls:
                for call in response.tool_calls:
                    if tool_calls_used>= cfg.max_tool_calls:
                        return self._finish(run, messages, StopReason.MAX_STEPS)
                    tool_calls_used += 1
                    result= self.tools.call(call)
                    seq += 1
                    run.tool_calls.append(call)
                    run.steps.append(TraceStep(
                        seq=seq, kind="tool", name=result.name,
                        input_summary=_summary(str(call.arguments)),
                        output_summary=("" if result.ok else "ERROR ") + _summary(result.output),
                        latency_ms=result.latency_ms,
                    ))
                    if not result.ok:
                        run.stop_reason= StopReason.TOOL_ERROR  # sticky, may recover
                    messages.append(ChatMessage(
                        role="tool", content=result.output,
                        name=result.name, tool_call_id=result.call_id,
                    ))
                continue

            # plain text answer -> done
            run.final_output= response.content
            run.status= RunStatus.SUCCEEDED
            run.stop_reason= run.stop_reason or StopReason.COMPLETED
            run.finished_at= time.time()
            return run

        return self._finish(run, messages, StopReason.MAX_STEPS)

    # ----------------------------------------------------------------- helpers
    def _call(self, messages: list[ChatMessage], specs: list[dict] | None):
        started= time.perf_counter()
        try:
            response= run_with_retries(
                lambda: self.provider.complete(list(messages), specs),
                self.config.retry,
                attempts_exhausted=f"provider {self.provider.name!r}",
            )
        except ProviderError as exc:
            raise ProviderError(str(exc)) from exc
        return response, (time.perf_counter() - started) * 1000

    def _finish(self, run: Run, messages: list[ChatMessage], reason: StopReason) -> Run:
        """Budget hit: give the model one last no-tools turn to synthesize what it has.

        stop_reason stays sticky: a run that recovered from a failed tool still reports
        TOOL_ERROR alongside SUCCEEDED, so eval gates can see the wobble.
        """
        cfg= self.config
        if cfg.max_cost_usd is not None and run.total_cost_usd>= cfg.max_cost_usd:
            reason= StopReason.MAX_STEPS  # budget stop still synthesizes
        try:
            final= run_with_retries(
                lambda: self.provider.complete(
                    list(messages) + [ChatMessage(
                        role="user",
                        content="Budget reached. Answer with everything you know now.",
                    )],
                    None,
                ),
                RetryPolicy(max_attempts=2, base_delay_s=0.1),
                attempts_exhausted="final synthesis",
            )
            run.total_tokens_in += final.tokens_in
            run.total_tokens_out += final.tokens_out
            run.total_cost_usd += _cost(final.tokens_in, final.tokens_out, cfg)
            run.final_output= final.content
        except ProviderError:
            run.final_output= None
        run.status= RunStatus.SUCCEEDED if run.final_output else RunStatus.FAILED
        run.stop_reason= reason
        run.finished_at= time.time()
        return run


def _cost(tokens_in: int, tokens_out: int, cfg: AgentConfig) -> float:
    return tokens_in / 1000 * cfg.price_per_1k_in + tokens_out / 1000 * cfg.price_per_1k_out


def _summary(text: str, limit: int= 240) -> str:
    flat= " ".join(text.split())
    return flat if len(flat)<= limit else flat[: limit - 1] + chr(8230)
