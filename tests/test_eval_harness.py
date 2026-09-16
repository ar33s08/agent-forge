"""Tests for the eval-as-CI harness: the golden pack must be green, and the
grading machinery must be able to detect failure (negative control)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT= Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evals.golden.tasks import build_registry, tasks
from evals.harness import EvalHarness
from evals.tasks import EvalTask


@pytest.fixture()
def golden_report():
    return EvalHarness(tasks(), build_registry).run_all()


def test_golden_pack_is_green(golden_report):
    report= golden_report
    assert report.task_count== 12
    assert report.failed== 0, [o.reason for o in report.outcomes if not o.passed]
    assert report.passed== 12
    assert report.pass_rate== 1.0
    assert report.tool_selection_accuracy== 1.0
    assert report.budget_compliance_rate== 1.0


def test_negative_control_detects_failure():
    """A task whose expect_contains can never match must be graded as failed."""
    bad= EvalTask(
        name="negative-control",
        goal="Answer politely.",
        provider_script=[
            {
                "content": "All good here.",
                "tool_calls": [],
                "tokens_in": 10,
                "tokens_out": 10,
            }
        ],
        expect_contains=["ZZZ-never"],
    )
    report= EvalHarness([bad], build_registry).run_all()
    assert report.task_count== 1
    assert report.failed== 1
    assert report.passed== 0
    assert report.pass_rate== 0.0
    assert "expect_contains miss" in report.outcomes[0].reason


def test_negative_control_wrong_tools_detected():
    """Executed-tool multiset mismatch must flip tool_selection_ok."""
    mismatch= EvalTask(
        name="wrong-tool-mismatch",
        goal="Add 1 and 1.",
        provider_script=[
            {
                "content": None,
                "tool_calls": [
                    {"name": "calculator", "arguments": {"op": "add", "x": 1, "y": 1}}
                ],
                "tokens_in": 10,
                "tokens_out": 5,
            },
            {"content": "It is 2.", "tool_calls": [], "tokens_in": 10, "tokens_out": 10},
        ],
        expect_contains=["2"],
        expect_tool_names=["wordcount"],
    )
    report= EvalHarness([mismatch], build_registry).run_all()
    assert report.failed== 1
    assert report.tool_selection_accuracy== 0.0


def test_tasks_are_json_serializable_scripts():
    """Scripts stored as dict payloads survive a round-trip through the model."""
    for task in tasks():
        payload= json.loads(json.dumps(task.model_dump(mode="json")))
        revived= EvalTask.model_validate(payload)
        assert revived.name== task.name
        assert len(revived.provider_script)== len(task.provider_script)


def test_report_json_written_by_ci_gate(tmp_path):
    """scripts/run_eval.py writes a parseable evals/report.json and exits 0."""
    import subprocess

    proc= subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_eval.py")],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )
    assert proc.returncode== 0, proc.stdout + proc.stderr
    report= json.loads((ROOT / "evals" / "report.json").read_text())
    assert report["task_count"]== 12
    assert report["failed"]== 0
    assert report["pass_rate"]== 1.0
