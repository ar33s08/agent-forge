#!/usr/bin/env python3
"""CI gate: run the golden eval pack and fail if any threshold slips.

Exit 0 only when pass_rate, tool_selection_accuracy and budget_compliance_rate
all meet their thresholds; the full graded report lands in evals/report.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT= Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evals.golden.tasks import build_registry, tasks
from evals.harness import EvalHarness

PASS_RATE= 1.00
TOOL_SELECTION= 1.00
BUDGET= 1.00

THRESHOLDS= {
    "pass_rate": PASS_RATE,
    "tool_selection_accuracy": TOOL_SELECTION,
    "budget_compliance_rate": BUDGET,
}


def main() -> int:
    harness= EvalHarness(tasks(), build_registry)
    report= harness.run_all()

    for outcome in report.outcomes:
        mark= "PASS" if outcome.passed else "FAIL"
        detail= f" — {outcome.reason}" if not outcome.passed else ""
        print(f"{mark:4} {outcome.name}{detail}")

    print("-" * 60)
    print(
        f"tasks={report.task_count} passed={report.passed} failed={report.failed} "
        f"pass_rate={report.pass_rate:.2f} "
        f"tool_selection={report.tool_selection_accuracy:.2f} "
        f"budget={report.budget_compliance_rate:.2f} "
        f"completed={report.completed_rate:.2f}"
    )

    out= ROOT / "evals" / "report.json"
    out.write_text(report.to_json() + "\n")
    print(f"wrote {out}")

    breaches= [
        name
        for name, floor in THRESHOLDS.items()
        if getattr(report, name)< floor
    ]
    for name in breaches:
        print(f"GATE FAIL {name}: {getattr(report, name):.2f}< {THRESHOLDS[name]:.2f}")
    return 1 if breaches else 0


if __name__== "__main__":
    raise SystemExit(main())
