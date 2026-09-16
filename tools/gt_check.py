
import os
# ground-truth dir/file names, codepoint-level (display channel is lossy)
for e in sorted(os.listdir(".")):
    print("ENTRY", "".join(chr(35) + chr(120) + hex(ord(c))[2:] for c in e))
print("---pyproject key lines---")
for l in open("pyproject.toml").read().splitlines():
    if any(k in l for k in ("depend", "build", "backend", "scripts", "agent-forge")):
        print(repr(l))
print("---readme defects---")
t = open("README.md").read()
import re
for i, l in enumerate(t.splitlines(), 1):
    low = l.lower()
    if " on**," in l or "** thread-safe" in l or "pyda" in low.replace("pydantic","") or "runpy" in l:
        print(i, repr(l))
print("has ar33s08:", "ar33s08" in t)
print("words: pydantic x", t.count("pydantic"), "| system_prompt x", t.count("system_prompt"), "| max_cost_usd x", t.count("max_cost_usd"), "| TOOL_ERROR x", t.count("TOOL_ERROR"), "| PROVIDER_ERROR x", t.count("PROVIDER_ERROR"), "| uvicorn x", t.count("uvicorn"), "| MockProvider x", t.count("MockProvider"), "| get_type_hints x", t.count("get_type_hints"))
