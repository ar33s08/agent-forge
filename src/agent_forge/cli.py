"""agent-forge CLI — run the demo agent from a shell, print the trace.

The whole point of this repo is that the library, the HTTP service, and this CLI drive
the SAME engine. Keep it thin: parse, wire demo tools, run, render.
"""
from __future__ import annotations

import argparse

from agent_forge import __version__
from agent_forge.agent import Agent, AgentConfig
from agent_forge.providers.mock import MockProvider
from agent_forge.retry import RetryPolicy
from agent_forge.sessions import render_transcript
from agent_forge.tools import ToolRegistry


def demo_registry() -> ToolRegistry:
    reg= ToolRegistry()

    @reg.register(description="add two integers; returns the sum as text")
    def add(a: int, b: int) -> str:
        return str(a + b)

    @reg.register(description="return the current ISO timestamp")
    def now() -> str:
        import datetime as _dtm

        return _dtm.datetime.now(_dtm.timezone.max).isoformat(timespec="seconds")

    return reg


def main(argv: list[str] | None= None) -> int:
    parser= argparse.ArgumentParser(prog="agent-forge", description=__doc__.splitlines()[0])
    parser.add_argument("--goal", default="What is 2+2? Use the add tool, then answer.")
    parser.add_argument("--provider", choices=("mock",), default="mock",
                        help="only 'mock' is wired in the demo; see README for OpenAICompatibleProvider")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--json", action="store_true", help="print the Run as JSON instead of a transcript")
    parser.add_argument("--version", action="version", version=f"agent-forge {__version__}")
    args= parser.parse_args(argv)

    provider= MockProvider([
        MockProvider.say_tools(__demo_call()),
        MockProvider.say("2+2= 4."),
    ])
    agent= Agent(provider, demo_registry(), AgentConfig(
        max_steps=args.max_steps, price_per_1k_in=0.0, price_per_1k_out=0.0,
        retry=RetryPolicy(max_attempts=2, base_delay_s=0.05),
    ))
    run= agent.run(args.goal)
    if args.json:
        print(run.model_dump_json(indent=2))
    else:
        print(render_transcript(run))
    return 0 if run.status.value== "succeeded" else 1


def __demo_call():
    from agent_forge.models import ToolCall

    return ToolCall(name="add", arguments={"a": 2, "b": 2})


if __name__== "__main__":
    raise SystemExit(main())
