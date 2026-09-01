#!/usr/bin/env python3
"""Generate Python modules from the dumped respR R source:
- src/resprpy/_unit_regexes.py : the *.rgx.R unit-matching regex tables
- src/resprpy/_gibbs_coeffs.py  : the sw_gibbs coefficient matrices (gjk, Carr)
Avoids hand-transcription errors. Run once; outputs are committed.
"""
import os
import re

SRC = r"X:\Rtools\resprpy\reference\r_src"
OUT = r"X:\Rtools\resprpy\src\pyrespr"

def unescape_r_string(s: str) -> str:
    # deparse writes R strings with \\ for a literal backslash; undo that.
    s = s.replace("\\\\", "\\")
    s = s.replace('\\"', '"')
    return s

def extract_strings(text: str):
    return [unescape_r_string(m) for m in re.findall(r'"((?:[^"\\]|\\.)*)"', text)]

# ---------- 1. regex tables ----------
regex = {}
for fn in sorted(os.listdir(SRC)):
    if not fn.endswith(".rgx.R"):
        continue
    name = fn[:-2]  # strip ".R"
    with open(os.path.join(SRC, fn), encoding="utf-8") as f:
        text = f.read()
    pats = extract_strings(text)
    # translate R POSIX class [:space:] -> Python \s
    pats = [p.replace("[:space:]", "\\s") for p in pats]
    # Python re requires inline flags at the very start; R writes "^(?i)..."
    pats = [p.replace("^(?i)", "(?i)^") for p in pats]
    # R's default engine (TRE) treats \b as matching at string start/end even
    # when the adjacent char is non-word; Python's \b does not. Emulate TRE:
    tre_b = r"(?:(?<=^)|(?=$)|(?<=\w)(?!\w)|(?<!\w)(?=\w))"
    pats = [p.replace(r"\b", tre_b) for p in pats]
    regex[name] = pats

with open(os.path.join(OUT, "_unit_regexes.py"), "w", encoding="utf-8") as f:
    f.write('"""Unit-matching regex tables, auto-generated from respR R source.\n'
            "Do not edit by hand; regenerate with reference/gen_regexes.py\n"
            '"""\n\n')
    f.write("REGEX = {\n")
    for name, pats in regex.items():
        f.write(f"    {name!r}: [\n")
        for p in pats:
            f.write(f"        {p!r},\n")
        f.write("    ],\n")
    f.write("}\n")

print(f"regex tables: {len(regex)}")

# ---------- 2. Gibbs coefficients ----------
with open(os.path.join(SRC, "marelac", "sw_gibbs.R"), encoding="utf-8") as f:
    text = f.read()

NUM_RE = r"-?\d+\.?\d*(?:[eE][-+]?\d+)?"

def extract_all_matrices():
    """Return the numeric vectors of every `matrix(data = c(...)` in the file,
    in source order. First occurrence is gjk (8x7), second is Carr (n x 4)."""
    starts = [m.end() for m in re.finditer(r"matrix\(data = c\(", text)]
    assert len(starts) == 2, len(starts)
    out = []
    for s in starts:
        end = text.index(")", s)
        out.append([float(x) for x in re.findall(NUM_RE, text[s:end])])
    return out

matrices = extract_all_matrices()
gjk_nums, carr_nums = matrices
assert len(gjk_nums) == 8 * 7, len(gjk_nums)
assert len(carr_nums) % 4 == 0, len(carr_nums)
gjk = [gjk_nums[i * 7:(i + 1) * 7] for i in range(8)]
carr = [carr_nums[i * 4:(i + 1) * 4] for i in range(len(carr_nums) // 4)]

with open(os.path.join(OUT, "_gibbs_coeffs.py"), "w", encoding="utf-8") as f:
    f.write('"""sw_gibbs (UNESCO seawater Gibbs potential) coefficients,\n'
            "auto-generated from the marelac R package source.\n"
            "gjk: 8x7 pure-water part; Carr rows: (i, j, k, coeff) -> gijk[i,j,k].\n"
            '"""\n\n')
    f.write("GJK = [\n")
    for row in gjk:
        f.write("    [" + ", ".join(repr(x) for x in row) + "],\n")
    f.write("]\n\n")
    f.write("CARR = [\n")
    for row in carr:
        f.write("    [" + ", ".join(repr(x) for x in row) + "],\n")
    f.write("]\n\n")
    f.write(f"GJK_NROW = {len(gjk)}\nGJK_NCOL = {len(gjk[0])}\n")

print(f"gjk: {len(gjk)}x{len(gjk[0])}, carr rows: {len(carr)}")
