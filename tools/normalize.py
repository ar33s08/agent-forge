"""Repo-wide byte normalizer. Canonical tokens are assembled from codepoints so this
file's own text cannot be corrupted in transit; variants are matched by regex class."""
from __future__ import annotations

import re
import sys
from pathlib import Path

DU = chr(95) + chr(95)
PYD = "pyd" + chr(97) + "ntic"          # the real package
STRE = "Str" + chr(69) + "num"
RUFF = chr(114) + "u" + chr(102) * 2
HATCH = "hatchl" + chr(105) + "ng"

RULES = [
    # any 'pyd?antic(-settings)?' that is not canonical -> canonical
    (re.compile(r"pyd[a-z]*ntic(-settings)?"), lambda m: PYD + ("-s" + "ettings" if m.group(1) else "")),
    # StrEnum with wrong middle char
    (re.compile(r"Str[a-z]*[Ee]num"), lambda m: STRE if m.group(0) != STRE else m.group(0)),
    # ryefl/ryef-style mangling of the linter name
    (re.compile(r"ry[e]?[fl]+"), lambda m: RUFF),
    # hatchl[?]?ing build backend
    (re.compile(r"hatchl[a-z]?ng"), lambda m: HATCH if m.group(0) != HATCH else m.group(0)),
    # dunders with single underscores: _future_, _version_, _all_, _init_, _name_, _main_
    (re.compile(r"(?<![a-z0-9])_+(future|init|version|all|name|repr|main|enter|exit)_+(?![a-z0-9])"),
     lambda m: DU + m.group(1) + DU),
    # Callable[[], T] empty-bracket mangling -> Callable[[], T]
    (re.compile(r"Callable\[\s*\]\s*,\s*([A-Z][a-z0-9_]*)\]"), lambda m: "Callable[[], " + m.group(1) + "]"),
    # stray space before version specifiers in toml dep strings
    (re.compile(r"\s+(?=[><=])"), lambda m: ""),
]


# sqlite3 API canon derived at runtime from the stdlib module (never hand-typed)
import sqlite3 as _sqlite3
_SQLITE = tuple(n for n in dir(_sqlite3) if n.startswith("execute") or n.startswith("fetch"))
def _sqlite_fix(m):
    w = m.group(1)
    base = w.rstrip("s")
    for real in _SQLITE:
        if real == base or (real + "s") == w:
            return real
    return w
RULES.append((re.compile(r"\b(execute|executes|executescript|executescripts|fetchone|fetchall|fetchmany|row_factory)\b"),
              lambda m: m.group(0) if m.group(0) in ("execute", "executescript", "fetchone", "fetchall", "fetchmany", "row_factory") else _sqlite_fix(m) if m.group(0) != "row_factory" else "row_factory"))
# COALESCE -> COALESCE (sql keyword; canon built from fragments)
_COALESCE = "COAL" + "ES" + "CE"
RULES.append((re.compile(r"COALES[A-Z]{2}"), lambda m: _COALESCE))

def fix_text(t: str) -> str:
    for rx, repl in RULES:
        t = rx.sub(repl, t)
    return t

def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    changed = 0
    for p in list(root.glob("src/**/*.py")) + list(root.glob("tests/**/*.py")) + \
             list(root.glob("scripts/**/*.py")) + [root / "pyproject.toml"]:
        if not p.is_file():
            continue
        original = p.read_text()
        fixed = fix_text(original)
        if fixed != original:
            p.write_text(fixed)
            changed += 1
            print("fixed", p)
    print("files changed:", changed)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
