"""Golden task pack: 12 deterministic scenarios covering the loop's behaviors.

Coverage map:
- 2 direct-answer tasks (no tools needed)
- 3 single-tool tasks with correct arguments
- 2 multi-step tool chains ending in a synthesis answer
- 1 wrong-arguments-then-recovery task (tool error fed back, agent recovers)
- 1 unknown-tool-then-known-tool task
- 1 max-steps budget task (repeating tool calls; completes via synthesis)
- 2 cost-budget tasks (priced tokens, hard max_cost_usd ceiling asserted)
"""
from __future__ import annotations

from agent_forge.models import ToolCall
from agent_forge.providers.mock import MockProvider
from agent_forge.tools import ToolRegistry
from evals.tasks import EvalTask

# Valid operators accepted by the calculator tool below.
_OPS= {"add", "sub"}


def build_registry() -> ToolRegistry:
    """Fresh registry with the two scalar-only tools every task shares."""
    reg= ToolRegistry()

    @reg.register(description="Arithmetic on two numbers; op must be add or sub.")
    def calculator(op: str, x: float, y: float) -> str:
        if op not in _OPS:
            return "bad op"
        if op== "add":
            return str(x + y)
        return str(x - y)

    @reg.register(description="Count the words in a piece of text.")
    def wordcount(text: str) -> str:
        return str(len(text.casefold().split()))

    return reg


def _calc(op: str, x: float, y: float) -> ToolCall:
    return ToolCall(name="calculator", arguments={"op": op, "x": x, "y": y})


def _words(text: str) -> ToolCall:
    return ToolCall(name="wordcount", arguments={"text": text})


def tasks() -> list[EvalTask]:
    say, say_tools= MockProvider.say, MockProvider.say_tools
    return [
        # -- 1-2: direct answers -------------------------------------------------
        EvalTask(
            name="direct-greeting",
            goal="Greet the team with the project codename.",
            provider_script=[say("Hello team — codename: FORGE.")],
            expect_contains=["hello", "forge"],
        ),
        EvalTask(
            name="direct-explanation",
            goal="Explain in one line what an agent loop does.",
            provider_script=[
                say(
                    "The loop prompts the model, runs any tool calls it asks for, "
                    "feeds the results back, and stops when a plain answer appears."
                )
            ],
            expect_contains=["loop"],
        ),
        # -- 3-5: single tool, correct args --------------------------------------
        EvalTask(
            name="single-add",
            goal="What is 2 + 3? Use the calculator tool.",
            provider_script=[
                say_tools(_calc("add", 2, 3)),
                say("2 + 3= 5"),
            ],
            expect_contains=["5"],
            expect_tool_names=["calculator"],
        ),
        EvalTask(
            name="single-sub",
            goal="Compute 10 - 4 with the calculator and report the result.",
            provider_script=[
                say_tools(_calc("sub", 10, 4)),
                say("The result is 6."),
            ],
            expect_contains=["result", "6"],
            expect_tool_names=["calculator"],
        ),
        EvalTask(
            name="single-wordcount",
            goal="Count the words in: the quick brown fox jumps",
            provider_script=[
                say_tools(_words("the quick brown fox jumps")),
                say("That sentence has 5 words."),
            ],
            expect_contains=["5 words"],
            expect_tool_names=["wordcount"],
        ),
        # -- 6-7: multi-step chains -----------------------------------------------
        EvalTask(
            name="chain-add-then-multiply-via-add",
            goal="Add 4 and 5, then triple the sum with the calculator.",
            provider_script=[
                say_tools(_calc("add", 4, 5)),
                say_tools(_calc("add", 9, 9), _calc("add", 18, 9)),
                say("4 + 5= 9, tripled it is 27."),
            ],
            expect_contains=["27"],
            expect_tool_names=["calculator", "calculator", "calculator"],
        ),
        EvalTask(
            name="chain-wordcount-then-add",
            goal="Count the words in 'one two three', then add the count to itself.",
            provider_script=[
                say_tools(_words("one two three")),
                say_tools(_calc("add", 3, 3)),
                say("The count is 3 and doubled it is 6."),
            ],
            expect_contains=["6"],
            expect_tool_names=["wordcount", "calculator"],
        ),
        # -- 8: wrong args -> recovery --------------------------------------------
        EvalTask(
            name="recovery-bad-op",
            goal="Compute 7 + 8 using the calculator.",
            provider_script=[
                # Model first reaches for an unsupported op; the tool replies
                # 'bad op' as an observation and the model corrects itself.
                say_tools(_calc("pow", 7, 8)),
                say_tools(_calc("add", 7, 8)),
                say("7 + 8= 15"),
            ],
            expect_contains=["15"],
            # The failed call still counts as executed: multiset includes it.
            expect_tool_names=["calculator", "calculator"],
        ),
        # -- 9: unknown tool -> known tool -----------------------------------------
        EvalTask(
            name="recovery-unknown-then-known",
            goal="Count words in: alpha beta",
            provider_script=[
                say_tools(ToolCall(name="spellcheck", arguments={"text": "alpha beta"})),
                say_tools(_words("alpha beta")),
                say("There are 2 words."),
            ],
            expect_contains=["2 words"],
            expect_tool_names=["spellcheck", "wordcount"],
        ),
        # -- 10: max-steps budget (script keeps calling tools; synthesis wraps up) --
        EvalTask(
            name="budget-max-steps",
            goal="Add 1+1 over and over until I stop you.",
            provider_script=[
                say_tools(_calc("add", 1, 1)),
                say_tools(_calc("add", 1, 1)),
                say_tools(_calc("add", 1, 1)),
                # Budget hit: the loop's one no-tools synthesis call gets this.
                say("Every sum was 2; you stopped me after three."),
            ],
            max_steps_budget=3,
            # Gate on the executed tool multiset: exactly three calculator
            # calls ran before the loop stopped and synthesized.
            expect_tool_names=["calculator", "calculator", "calculator"],
        ),
        # -- 11-12: cost budget ceilings -------------------------------------------
        EvalTask(
            name="budget-cost-single-tool",
            goal="Use the calculator to add 100 and 200, then report it.",
            provider_script=[
                say_tools(_calc("add", 100, 200), tokens_in=5000, tokens_out=2500),
                say("100 + 200= 300", tokens_in=5000, tokens_out=2500),
            ],
            max_cost_usd=0.5,
            expect_contains=["300"],
            expect_tool_names=["calculator"],
        ),
        EvalTask(
            name="budget-cost-wordcount",
            goal="Count the words in 'cost budget check' under the spend cap.",
            provider_script=[
                say_tools(_words("cost budget check"), tokens_in=4000, tokens_out=2000),
                say("It has 3 words.", tokens_in=4000, tokens_out=2000),
            ],
            max_cost_usd=0.5,
            expect_contains=["3 words"],
            expect_tool_names=["wordcount"],
        ),
    ]
