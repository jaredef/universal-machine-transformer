#!/usr/bin/env python3
"""Error on unseen steps against network size, for each training set: is there a peak where the parameter count is
close to the number of training steps (double descent)?

    python3 curve.py [results.json]
"""
import json, statistics, sys
from collections import defaultdict
from pathlib import Path

R = json.loads((Path(__file__).resolve().parent / (sys.argv[1] if len(sys.argv) > 1 else "results.json")).read_text())["runs"]
by = defaultdict(list)
for r in R:
    if not r.get("fixed_steps"):
        by[(r["n"], r["d"])].append(r)
for n in sorted({k[0] for k in by}):
    cells = sorted((d for (m, d) in by if m == n))
    if len(cells) < 4:
        continue
    samples = by[(n, cells[0])][0]["train_steps"]
    print(f"\nTrained on the first {n:,} random machines: {samples:,} training steps")
    print("| Width | Parameters | Parameters per training step | Seeds | Unseen steps wrong (mean) | Range over seeds | Programs correct (mean of 12) | Adam steps to fit (mean) |")
    print("|---|---|---|---|---|---|---|---|")
    for d in cells:
        rs = by[(n, d)]
        err = [100 * (1 - r["unseen_right"] / r["unseen"]) for r in rs]
        print(f"| {d} | {rs[0]['params']:,} | {rs[0]['params'] / samples:.2f} | {len(rs)} | {statistics.mean(err):.2f}% | "
              f"{min(err):.2f}–{max(err):.2f}% | {statistics.mean(r['programs'] for r in rs):.1f} | "
              f"{statistics.mean(r['fit_at'] or 0 for r in rs):,.0f} |")
