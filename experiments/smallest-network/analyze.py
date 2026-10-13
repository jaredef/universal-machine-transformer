#!/usr/bin/env python3
"""Summarize results.json: one row per network size."""
import json
from collections import defaultdict
from pathlib import Path

R = json.loads((Path(__file__).resolve().parent / "results.json").read_text())["runs"]
by = defaultdict(list)
for r in R:
    by[(r["d"], r["hidden"])].append(r)
print("| Width | MLP | Parameters | Seeds that learn the whole table | Adam steps to fit | Steps still wrong (seeds that did not) |")
print("|---|---|---|---|---|---|")
for (d, h), rs in sorted(by.items(), key=lambda kv: -kv[1][0]["params"]):
    rs = sorted(rs, key=lambda r: r["seed"])
    fit = [r for r in rs if r["errors"] == 0]
    steps = sorted(r["fit_at"] for r in fit)
    print(f"| {d} | {h} | {rs[0]['params']:,} | {len(fit)} of {len(rs)} | "
          f"{(f'{steps[0]:,}' if steps[0] == steps[-1] else f'{steps[0]:,}–{steps[-1]:,}') if steps else '–'} | "
          f"{', '.join(f'{r['errors']:,}' for r in rs if r['errors']) or '–'} |")
