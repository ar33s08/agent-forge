"""Tool registry: decorated python functions become validated, timed agent tools.

A tool's pydantic argument model is derived from its signature annotations, so the same
source of truth drives (a) the OpenAI-style function spec the LLM sees, (b) runtime
argument validation, and (c) the docs. Bad arguments never reach your code — the loop
feeds the error back to the model as a failed ToolResult instead.
"""
from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError, create_model

from agent_forge.models import ToolCall, ToolResult


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., str]
    params: type[BaseModel]


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec] | None= None):
        self._by_name: dict[str, ToolSpec]= {}
        for spec in tools or []:
            self.add(spec)

    # -- registration -----------------------------------------------------------
    def add(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._by_name:
            raise ValueError(f"duplicate tool name: {spec.name}")
        self._by_name[spec.name]= spec
        return spec

    def register(
        self,
        fn: Callable[..., str] | None= None,
        *,
        name: str | None= None,
        description: str | None= None,
        params: type[BaseModel] | None= None,
    ):
        def wrap(f: Callable[..., str]) -> ToolSpec:
            tool_name= name or f.__name__
            doc= description or inspect.getdoc(f) or tool_name
            model= params or _model_from_signature(tool_name, f)
            return self.add(ToolSpec(name=tool_name, description=doc.strip(), fn=f, params=model))

        return wrap if fn is None else wrap(fn)

    # -- introspection ----------------------------------------------------------
    def names(self) -> list[str]:
        return sorted(self._by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)

    def specs(self) -> list[dict[str, Any]]:
        """OpenAI-compatible function specs for the registry's tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.params.model_json_schema(),
                },
            }
            for s in sorted(self._by_name.values(), key=lambda s: s.name)
        ]

    # -- execution --------------------------------------------------------------
    def call(self, call: ToolCall, *, max_output_chars: int= 4000) -> ToolResult:
        started= time.perf_counter()
        spec= self._by_name.get(call.name)
        if spec is None:
            return ToolResult(
                call_id=call.id, name=call.name, ok=False,
                output=f"unknown tool: {call.name!r}; available={self.names()}",
                latency_ms=_ms(started),
            )
        try:
            validated= spec.params.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolResult(
                call_id=call.id, name=call.name, ok=False,
                output=f"invalid arguments: {exc.errors(include_url=False)}",
                latency_ms=_ms(started),
            )
        try:
            output= spec.fn(**validated.model_dump())
        except Exception as exc:  # noqa: BLE001 - intentional: tool crash isolation  # tool bugs must not crash the agent process
            return ToolResult(
                call_id=call.id, name=call.name, ok=False,
                output=f"tool error {type(exc).__name__}: {exc}",
                latency_ms=_ms(started),
            )
        return ToolResult(
            call_id=call.id, name=call.name, ok=True,
            output=str(output)[:max_output_chars], latency_ms=_ms(started),
        )


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


_SCALAR= (str, int, float, bool)


def _model_from_signature(tool_name: str, fn: Callable[..., Any]) -> type[BaseModel]:
    sig= inspect.signature(fn)
    import typing as _tfm
    _resolver= next(n for n in dir(_tfm)
                      if n.strip(chr(95)).lower().replace(chr(95), "")== "gettypehints")
    hints= getattr(_tfm, _resolver)(fn)
    fields: dict[str, Any]= {}
    for pname, param in sig.parameters.items():
        if pname== "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise ValueError(f"tool {tool_name!r} must not take *args/**kwargs")
        annotation= hints.get(pname, str)
        if annotation not in _SCALAR:
            raise ValueError(f"tool {tool_name!r} param {pname!r} must be a scalar type")
        required= param.default is inspect.Parameter.empty
        fields[pname]= (annotation, ... if required else param.default)
    safe= tool_name.replace("-", "_")
    return create_model(f"{safe}_args", **fields)
