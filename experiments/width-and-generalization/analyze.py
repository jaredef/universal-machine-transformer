#!/usr/bin/env python3
"""Summarize results.json: for each training set and width, programs run correctly per seed, and accuracy on steps never seen."""
import json
from collections import defaultdict
from pathlib import Path

import sys
R = json.loads((Path(__file__).resolve().parent / (sys.argv[1] if len(sys.argv) > 1 else "results.json")).read_text())["runs"]
by = defaultdict(list)
for r in R:
    by[(r["n"], r["d"])].append(r)
widths = sorted({r["d"] for r in R}); ns = sorted({r["n"] for r in R})
params = {r["d"]: r["params"] for r in R}
print("Programs run correctly (of 12), per seed")
print("| Random machines | " + " | ".join(f"width {d} ({params[d]:,})" for d in widths) + " |")
print("|---|" + "---|" * len(widths))
for n in ns:
    print(f"| {n:,} | " + " | ".join(" / ".join(str(r["programs"]) for r in sorted(by[(n, d)], key=lambda r: r["seed"])) or "–" for d in widths) + " |")
print("\nSteps never seen in training that the network gets right (all seeds pooled)")
print("| Random machines | " + " | ".join(f"width {d}" for d in widths) + " |")
print("|---|" + "---|" * len(widths))
for n in ns:
    cells = []
    for d in widths:
        rs = by[(n, d)]
        u, ur = sum(r["unseen"] for r in rs), sum(r["unseen_right"] for r in rs)
        cells.append(f"{100 * ur / u:.1f}%" if u else "–")
    print(f"| {n:,} | " + " | ".join(cells) + " |")
fails = [r for r in R if r["train_errors"]]
print(f"\nRuns that did not learn their whole training set: {len(fails)}" + (": " + ", ".join(f"width {r['d']} N={r['n']} seed {r['seed']} ({r['train_errors']} wrong)" for r in fails) if fails else ""))
