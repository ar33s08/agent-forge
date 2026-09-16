# AgentForge

**Ship LLM agents as real services — not notebooks.**

An agent runtime with the parts that are missing from 90% of agent demos: validated
tool dispatch, per-tool failure isolation, run budgets (steps *and* dollars), persistent
traces, an auth'd HTTP API, a Docker image, and an eval suite wired into CI that fails
the build when agent behavior regresses.

The library, the HTTP service, and the CLI all drive the same engine — one mental model,
three front doors.

```
user goal ─► Agent loop ─► LlmProvider (.complete) ─► content | ToolCalls
                ▲                                            │
                └── ToolResults (observed, never raised) ◄───┘
                                  │
                    ToolRegistry: pydantic-validated, timed, crash-isolated
                                  │
                  RunStore (sqlite): every step, token, and cent persisted
```

## Why this exists

Most agent code stops at "the model called a tool, cool." Production agents need answers
to questions demos never ask:

| Question a demo skips | What AgentForge does |
|---|---|
| What if the model passes garbage to a tool? | Arguments validate against a pydantic model *before* your code runs; the error goes back to the model as an observation it can recover from |
| What if the tool itself crashes? | Caught, converted to a `ToolResult(ok=False)`, fed back as an observation. The agent process does not die because a tool had a bad afternoon |
| What will this cost? | `max_cost_usd` + token pricing stop runaway spend; every trace step carries its own cost; `GET /v1/stats` answers "what have we spent" |
| What if the loop won't stop? | Two independent kill switches (`max_steps`, `max_cost_usd`); either one still buys one final no-tools turn so the run exits with an answer, not a shrug |
| What did it actually do? | Full `TraceStep` stream per run — LLM calls, tool calls, latencies, tokens, cost — persisted to sqlite and rendered by `render_transcript()` for the incident thread |
| Did that deploy change agent *behavior*? | `scripts/run_eval.py` gates CI on pass-rate, tool-selection accuracy, and budget-compliance against a golden task pack |
| Is this a service or a script? | FastAPI API with key auth + rate limiting, Dockerfile with healthcheck, compose file that binds loopback-only by default |

## Quickstart

Requires [uv](https://github.com/astral-sh/uv) and Python 3.11+.

```bash
git clone https://github.com/ar33s08/agent-forge.git && cd agent-forge
uv sync --extra dev
uv run pytest -q                    # unit + integration tests
uv run python scripts/run_eval.py   # the eval-as-CI gate
uv run python -m agent_forge.cli --goal "What is 2+2? Use the add tool, then answer."
```

The CLI prints the full trace for real:

```
# run run_f8e704764378 [succeeded] stop=completed
goal: What is 2+2? Use the add tool, then answer.
- LLM mock | tok 10+5 $0.0000
    out: [tool calls]
- TOOL add | 0ms
    in : {'a': 2, 'b': 2}
    out: 4
total: 20+15 tok, $0.0000
answer: 2+2= 4.
```

## Use it as a library

```python
from agent_forge.agent import Agent, AgentConfig
from agent_forge.providers.openai import OpenAICompatibleProvider
from agent_forge.retry import RetryPolicy
from agent_forge.tools import ToolRegistry

tools = ToolRegistry()

@tools.register(description="add two integers")
def add(a: int, b: int) -> str:
    return str(a + b)          # return value goes back to the model as text

agent = Agent(
    provider=OpenAICompatibleProvider.from_env(model="gpt-4o-mini"),
    tools=tools,
    config=AgentConfig(
        max_steps=8,            # loop bound
        max_cost_usd=0.25,      # wallet bound — stop runaway spend
        price_per_1k_in=0.15, price_per_1k_out=0.60,
        retry=RetryPolicy(max_attempts=4, base_delay_s=0.25),
    ),
)

run = agent.run("What is 12 + 30?")
print(run.final_output)          # "42"
print(run.total_cost_usd)        # what that actually cost
for step in run.steps: ...       # the whole trace, tokens/cost/latency per step
```

### The tool loop, precisely

1. System + goal go to `provider.complete(messages, tools=specs)`.
2. Tool calls come back → each is validated against its pydantic param model → executed
   with timing → *whatever happened* (value, validation error, crash, unknown tool)
   becomes a `tool` message the model sees next turn.
3. A plain-text response ends the run. Budget exhaustion buys one final no-tools
   synthesis turn, so even a stopped run answers.
4. The model never sees an unvalidated payload; your code never sees unvalidated arguments.

## The HTTP service

```python
# my_service.py
from agent_forge.providers.openai import OpenAICompatibleProvider
from agent_forge.server.app import create_app
from agent_forge.tools import ToolRegistry

tools = ToolRegistry()
# ... register tools ...
APP = create_app(
    provider_factory=lambda: OpenAICompatibleProvider.from_env(model="gpt-4o-mini"),
    tools=tools,
    api_keys=["keep-this-off-the-internet"],   # enables auth + rate limiting
    store_path="runs.db",
)
```

```bash
uv run uvicorn my_service:APP --port 8080   # bind is your call; loopback behind a proxy is safe
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/runs` `{"goal": "...", "session_id": "optional"}` | run an agent, returns the persisted run (201) |
| `GET /v1/runs/{id}` | fetch any historical run with its full trace |
| `GET /v1/sessions/{id}` | chronological history for multi-turn sessions |
| `GET /v1/stats` | aggregate runs / tokens / dollars |
| `GET /healthz` | healthcheck (used by the Docker HEALTHCHECK) |

Auth is `X-API-Key` against a configured set; over-limit responses are `429`. With no
keys configured, auth is disabled — fine for local dev, not for anything reachable.

## Evals as a merge gate

`evals/golden/tasks.py` is a growing pack of scripted-agent tasks: correct tool args,
recovery from a bad-argument observation, unknown-tool self-correction, budget
compliance. `scripts/run_eval.py` fails (exit 1) if any gate slips:

- **pass rate** — did the agent reach the expected answer
- **tool-selection accuracy** — right tools, right arguments, right count
- **budget compliance** — cost stayed inside `max_cost_usd`

Because the harness runs on `MockProvider`, CI is deterministic and costs zero API
tokens: same behavioral gate you'd want from a live eval, at test speed. The numbers
from the latest run:

```
(pasted from `uv run python scripts/run_eval.py` output — see CI badge green state)
```

## Providers

- `MockProvider` — scripted, records every prompt it was shown; deterministic tests,
  evals, and CI.
- `FlakyProvider` — fails N times first; proves your retry policy actually retries.
- `OpenAICompatibleProvider` — one HTTP class for any OpenAI-shaped API
  (`base_url` covers Azure-style gateways and vLLM/served-locally stacks). It raises on
  non-2xx so the retry policy does its job.

Writing one? Implement the 3-member `LlmProvider` protocol (`name`, `complete`).

## Operational notes

- **Provider errors are data.** `run()` returns a `Run(status=FAILED,
  stop_reason=PROVIDER_ERROR)` — it does not raise a `ProviderError`. One dead upstream
  does not take your worker pool down with it.
- **`stop_reason` is sticky.** A run that recovered from a failed tool still reports
  `TOOL_ERROR` alongside `SUCCEEDED` — your eval gate sees the wobble, not just the win.
- **sqlite is one-connection-per-store**, thread-safe, sufficient for the single-worker
  service. Swap `RunStore` for Postgres behind the same 5-method surface at scale.
- **No telemetry phone-home.** There is no analytics beacon anywhere in this codebase;
  traces stay in your sqlite file.
- **Security stance:** the compose file binds `127.0.0.1` on purpose. Do not put this
  service on the public internet without a gateway in front of it — the agent holds
  your model key and executes your tools.

## Layout

```
src/agent_forge/
  agent.py        # the loop, budgets, sticky stop reasons, synthesis turn
  tools.py        # registry: pydantic param models from signatures, OpenAI specs
  models.py       # Run / TraceStep / ToolCall / ToolResult — the persisted truth
  retry.py        # RetryPolicy + run_with_retries (backoff lives here, nowhere else)
  sessions.py     # RunStore (sqlite) + transcript renderer
  providers/      # LlmProvider protocol, mock, OpenAI-compatible HTTP
  server/app.py   # create_app() — auth, rate limit, runs API
  cli.py          # python -m agent_forge.cli
evals/            # golden task pack + harness      scripts/run_eval.py  CI gate
tests/            # unit + API integration; the suite can fail on purpose
Dockerfile · compose.yaml · .github/workflows/ci.yml
```

## License

Apache-2.0
