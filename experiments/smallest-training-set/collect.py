#!/usr/bin/env python3
"""Collect the steps for the smallest-training-set experiment (pure Python, standard library).

1. The training pool: the universal machine's lagging-window steps while it runs random machines, generated exactly as
   lag3's training data was (same generator, same seed, same order), continued past lag3's 4,000 machines. For every
   distinct step, record the index of the first machine that produces it: the training set of the first N machines is
   then every step whose first index is below N.
2. The tests: the distinct steps each demonstration program needs on its exact run (the seven held-out machines on
   their 70 inputs, Rules 90, 30, 110 for 16 generations, and the busy beavers BB(2) and BB(4)). A trained network runs a
   program correctly exactly when it predicts every one of those steps correctly, since the decoder is deterministic.

Writes steps.json (committed, so train.py runs without recollecting): every step (window and exact output), its first machine index, and each test
program's steps as indices.
"""
import json, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for d in ("lag3/derived", "lag4/derived", "lag5/derived"):
    sys.path.insert(0, str(ROOT / "machines" / d))
import lag3, lag0, lag1, utm  # noqa: E402
import ca, bb  # noqa: E402

N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
t0 = time.time()
REF = lag3.utm_program(lag3.held_out_source("invert", "0"))
EMIT2NAME = {tuple(e): n for n, e, _ in lag1.Net(REF).vocab_out}


def label(w):
    return EMIT2NAME[tuple(lag0.rule(REF, *w) or [])]


first = {}
rng = random.Random(2)
growth = []
for i in range(N_MAX):                                   # identical to train_lag3.ref.py for i < 4000
    src = lag3.random_source(rng, rng.randint(1, 5), "".join(rng.choice("01_") for _ in range(rng.randint(1, 6))),
                             "RRRLS" if i % 2 else "LRS", 0.5)
    p = lag3.utm_program(src)
    seen = set()
    lag3.lag_run(p, lag3.Exact(p), max_decode=400_000, seen=seen)
    for w in seen:
        first.setdefault(w, i)
    if (i + 1) in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 16000, 32000) or (i + 1) % 1000 == 0:
        growth.append((i + 1, len(first)))
        print(f"{i + 1} machines: {len(first)} distinct steps ({time.time() - t0:.0f}s)", flush=True)

tests = {}


def need(name, prog, runner):
    exact = lag3.Exact(prog)
    runner(exact)
    tests.setdefault(name, set()).update(exact.cache)


sys.argv = sys.argv[:1]
import lag3w  # noqa: E402
for name, tape in lag3w.suite():
    src = lag3.held_out_source(name, tape)
    p = lag3.utm_program(src)
    need(name, p, lambda ex: lag3.lag_run(p, ex))
print(f"held-out runs done ({time.time() - t0:.0f}s)", flush=True)
import lag4  # noqa: E402
for rule in (90, 30, 110):
    p = lag3.utm_program(ca.ca_source(rule))
    need(f"rule{rule}", p, lambda ex: lag4.draw(rule, ex, 16))
    print(f"rule {rule} done ({time.time() - t0:.0f}s)", flush=True)
import lag5  # noqa: E402
for name in bb.CHAMPIONS:
    src = bb.bb_source(name)
    need(name, lag3.utm_program(src), lambda ex: lag5.trace(src, ex))

steps = sorted(set(first) | set().union(*tests.values()))
index = {w: i for i, w in enumerate(steps)}
out = {"n_max": N_MAX, "growth": growth,
       "vin": lag0.alphabet(REF), "vout": [n for n, _, _ in lag1.Net(REF).vocab_out],
       "steps": ["\u0001".join(w) for w in steps], "labels": [label(w) for w in steps],
       "first": [first.get(w, -1) for w in steps],
       "tests": {k: sorted(index[w] for w in v) for k, v in tests.items()}}
(HERE / "steps.json").write_text(json.dumps(out, separators=(",", ":")))
print(f"{len(steps)} steps, {len(tests)} test programs, wrote steps.json ({time.time() - t0:.0f}s)")
