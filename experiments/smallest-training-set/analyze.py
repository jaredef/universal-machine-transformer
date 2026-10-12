#!/usr/bin/env python3
"""Summarize results.json: one row per training-set size (pure Python, standard library)."""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())["runs"]
S = json.loads((HERE / "steps.json").read_text())
NEEDED = sorted(set().union(*map(set, S["tests"].values())))         # every step some demo program needs
by = defaultdict(list)
for r in R:
    by[r["n"]].append(r)

print(f"{len(S['steps']):,} distinct steps in all; the 12 demo programs need {len(NEEDED):,} of them.\n")
print("| Random machines | Training steps | Distinct demo steps never in training | Unseen demo steps predicted right* | "
      "Later pool steps predicted right | Programs run correctly, per seed | Seeds running all 12 |")
print("|---|---|---|---|---|---|---|")
for n in sorted(by):
    rs = sorted(by[n], key=lambda r: r["seed"])
    uns = sum(t["unseen"] for t in rs[0]["tests"].values())             # summed over programs (shared steps count again)
    distinct = sum(1 for i in NEEDED if not 0 <= S["first"][i] < n)
    right = [uns - sum(t["wrong_unseen"] for t in r["tests"].values()) for r in rs]
    later = [(r["later_correct"], r["later_steps"]) for r in rs]
    ok = [sum(t["correct_run"] for t in r["tests"].values()) for r in rs]
    pct = lambda a, b: f"{100 * a / b:.1f}%" if b else "n/a"
    print(f"| {n:,} | {rs[0]['train_steps']:,} | {distinct:,} | "
          f"{pct(sum(right), uns * len(rs))} | {pct(sum(c for c, _ in later), sum(t for _, t in later))} | "
          f"{' / '.join(map(str, ok))} | {sum(o == 12 for o in ok)} of {len(rs)} |")

print("\n*summed over the 12 programs and the seeds; a step two programs share counts twice.\n")
names = list(R[0]["tests"])
print("Per program: seeds whose run is correct, of seeds trained")
print("| Random machines | " + " | ".join(names) + " |")
print("|---|" + "---|" * len(names))
for n in sorted(by):
    rs = by[n]
    print(f"| {n:,} | " + " | ".join(f'{sum(r["tests"][k]["correct_run"] for r in rs)}/{len(rs)}' for k in names) + " |")
