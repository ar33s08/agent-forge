"""Verification gate: byte-compile every shipped .py, then import every agent_forge module."""
from __future__ import annotations

import importlib
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main() -> int:
    src_dir = ROOT / "src"
    files = sorted(src_dir.rglob("*.py"))
    for f in files:
        try:
            compile(f.read_bytes(), str(f), "exec")
        except SyntaxError as exc:
            print(f"COMPILE FAIL {f}: {exc}")
            return 1
    sys.path.insert(0, str(src_dir))
    importlib.invalidate_caches()
    mods = ["agent_forge"]
    pkg = src_dir / "agent_forge"
    for p in sorted(pkg.rglob("*.py")):
        rel = str(p.relative_to(src_dir).with_suffix(""))
        mods.append(rel.replace(f"{chr(47)}", "."))
    ok = 0
    for m in mods:
        try:
            importlib.import_module(m)
            ok += 1
        except Exception as exc:
            print(f"IMPORT FAIL {m}: {type(exc).__name__}: {exc}")
            return 1
    print(f"GATE OK: {len(files)} compiled, {ok} imported")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
