"""Sqlite-backed run/session store — the trace is a first-class artifact.

Every agent run is persisted whole (goal, step trace, tokens, cost) with the aggregate
columns EXTRACTED into real columns (payload JSON alone cannot answer 'what did this
cost'), so the dashboard/eval gates read the same rows the API serves. One connection,
check_same_thread=False, guarded by a lock — good enough for a single-worker service
without dragging in an ORM.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterator

from agent_forge.models import Run

_SCHEMA= """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    status TEXT NOT NULL,
    stop_reason TEXT,
    goal TEXT NOT NULL,
    steps INTEGER NOT NULL DEFAULT 0,
    tool_calls INTEGER NOT NULL DEFAULT 0,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id, created_at);
"""


class RunStore:
    def __init__(self, path: str | Path= ":memory:"):
        self.path= str(path)
        self._lock= threading.Lock()
        self._conn= sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory= sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save(self, run: Run, session_id: str | None= None) -> None:
        row= (
            run.id, session_id, run.status.value,
            run.stop_reason.value if run.stop_reason else None,
            run.goal, len(run.steps), len(run.tool_calls),
            run.total_tokens_in, run.total_tokens_out, run.total_cost_usd,
            run.model_dump_json(), run.created_at, run.finished_at,
        )
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", row
            )
            self._conn.commit()

    def get(self, run_id: str) -> Run | None:
        row= self._conn.execute("SELECT payload FROM runs WHERE id= ?", (run_id,)).fetchone()
        return Run.model_validate_json(row["payload"]) if row else None

    def session(self, session_id: str) -> list[Run]:
        """Chronological history for multi-turn sessions."""
        rows= self._conn.execute(
            "SELECT payload FROM runs WHERE session_id= ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
        return [Run.model_validate_json(r["payload"]) for r in rows]

    def recent(self, limit: int= 20) -> list[Run]:
        rows= self._conn.execute(
            "SELECT payload FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [Run.model_validate_json(r["payload"]) for r in rows][::-1]

    def iter_all(self) -> Iterator[Run]:
        for row in self._conn.execute("SELECT payload FROM runs ORDER BY created_at ASC"):
            yield Run.model_validate_json(row["payload"])

    def totals(self) -> dict[str, float]:
        row= self._conn.execute(
            "SELECT COUNT(*) AS n,"
            " COALESCE(SUM(tokens_in), 0) AS ti,"
            " COALESCE(SUM(tokens_out), 0) AS tout,"
            " COALESCE(SUM(cost_usd), 0) AS c FROM runs"
        ).fetchone()
        return {
            "runs": row["n"],
            "tokens_in": row["ti"],
            "tokens_out": row["tout"],
            "cost_usd": round(row["c"], 6),
        }

    def close(self) -> None:
        self._conn.close()


def render_transcript(run: Run) -> str:
    """Human-readable trace dump — what you paste in an incident thread."""
    stop= run.stop_reason.value if run.stop_reason else "-"
    lines= [f"# run {run.id} [{run.status.value}] stop={stop}"]
    lines.append(f"goal: {run.goal}")
    for s in run.steps:
        tag= f"LLM {s.name}" if s.kind== "llm" else f"TOOL {s.name}"
        bits= [tag, f"{s.latency_ms:.0f}ms"]
        if s.kind== "llm":
            bits.append(f"tok {s.tokens_in}+{s.tokens_out} ${s.cost_usd:.4f}")
        lines.append("- " + " | ".join(bits))
        lines.append(f"    in : {s.input_summary}")
        lines.append(f"    out: {s.output_summary}")
    lines.append(f"total: {run.total_tokens_in}+{run.total_tokens_out} tok, ${run.total_cost_usd:.4f}")
    if run.final_output:
        lines.append("answer: " + run.final_output)
    return "\n".join(lines)
