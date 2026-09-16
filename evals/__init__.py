"""Eval-as-CI harness for AgentForge.

Deterministic, offline evaluation of the agent loop: scripted MockProvider runs
against golden task packs, scored on completion, tool-selection accuracy and
budget compliance. scripts/run_eval.py turns this into a CI gate.
"""
from __future__ import annotations

from evals.harness import EvalHarness
from evals.tasks import EvalReport, EvalTask, TaskOutcome

__all__= ["EvalHarness", "EvalReport", "EvalTask", "TaskOutcome"]
