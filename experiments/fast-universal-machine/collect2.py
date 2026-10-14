#!/usr/bin/env python3
"""Milestone 2: version 2's step table. Does it close, as version 1's did at 14,849 steps?

Collects the lagging-window steps that universal machine version 2 takes while running random machines (windows.py: read
off direct runs, checked equal to the decoder's), recording for each distinct step the first machine that produced it,
and its exact output. Version 2's 256 states need machines that use them: the random machines here have 1 to 256 states
(half up to 8, the rest spread evenly on a log scale), on random tapes of 1 to 8 cells, each run for at most
MAX_UNIVERSAL universal steps.

Then the steps each demonstration program needs, including the dragon curve's machine spread among 64 and 256 states,
to see whether the table collected from random machines covers them.

    python3 collect2.py [machines]       writes steps2.json; prints the table's growth
"""
import json, math, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import windows, utm2, check  # noqa: E402
import lag0  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
MAX_UNIVERSAL = 300_000
RULES = {(s, r): (n, w, m) for s, r, n, w, m in utm2.utm_rules()}
UTM = lag0.parse_tm0(utm2.utm_source("_"))                  # the universal machine itself, for the exact output of a step
t0 = time.time()


def random_machine(rng, n_states, moves, halt):
    states = [f"s{i}" for i in range(n_states)]
    lines = []
    for q in states:
        for sym in "01_":
            if rng.random() < 0.12:
                continue
            nxt = rng.choice(states) if rng.random() > 0.25 * halt else rng.choice(["qa", "qa", "qr"])
            lines.append(f"{q} {sym} -> {nxt} {rng.choice('01_')} {rng.choice(moves)}")
    tape = "".join(rng.choice("01_") for _ in range(rng.randint(1, 8)))
    return lag0.parse_tm0(f"blank _\nstart s0\naccept qa\nreject qr\ntape {tape}\n---\n" + "\n".join(lines) + "\n"), tape


def size(rng):
    return rng.randint(1, 8) if rng.random() < 0.5 else int(round(math.exp(rng.uniform(math.log(8), math.log(256)))))


first, growth = {}, []
rng = random.Random(2)
for i in range(N):
    S = size(rng)
    prog, tape = random_machine(rng, S, "RRRLS" if i % 2 else "LRS", 0.5 if S <= 8 else 0.15)
    W, steps, halted = windows.collect(RULES, utm2.encode(prog, tape), max_steps=MAX_UNIVERSAL)
    for w in W:
        first.setdefault(w, i)
    if (i + 1) in (1, 10, 30, 100, 300) or (i + 1) % 250 == 0:
        growth.append((i + 1, len(first)))
        print(f"{i + 1} machines: {len(first):,} distinct steps ({time.time() - t0:.0f}s)", flush=True)

tests = {}
for name, prog, tape in check.programs():
    if name.startswith("random"):
        continue
    key = name.split("(")[0].strip()
    W, _, _ = windows.collect(RULES, utm2.encode(prog, tape), max_steps=3_000_000)
    tests.setdefault(key, set()).update(W)
for S in (64, 256):                                          # the dragon curve's machine, its states spread among S
    dragon = ["s _ -> inc _ R", "inc 1 -> inc 0 R", "inc 0 -> back 1 L", "inc _ -> qa _ S", "back 0 -> back 0 L", "back 1 -> back 1 L",
              "back _ -> low _ R", "low 0 -> low 0 R", "low 1 -> nxt 1 R", "nxt 0 -> sayR 0 L", "nxt _ -> sayR _ L", "nxt 1 -> sayL 1 L",
              "sayR 1 -> home 1 L", "sayL 1 -> home 1 L", "home 0 -> home 0 L", "home _ -> inc _ R"]
    prog = check.source(dragon + [f"z{j} 0 -> z{j} 0 R" for j in range(S - 8)], "s", "_" + "0" * 6)
    real = utm2.states_of(prog); rest = real[1:]; random.Random(S).shuffle(rest)
    W, _, _ = windows.collect(RULES, utm2.encode(prog, "_" + "0" * 6, [real[0]] + rest), max_steps=10 ** 7)
    tests[f"dragon among {S} states"] = W
print(f"tests collected ({time.time() - t0:.0f}s)", flush=True)

steps = sorted(set(first) | set().union(*tests.values()))
index = {w: k for k, w in enumerate(steps)}
labels = []
for w in steps:
    out = lag0.rule(UTM, *w)
    labels.append(" ".join(out) if out is not None else "<halt>")
vin = sorted({t for w in steps for t in w})
vout = sorted(set(labels))
for name, W in tests.items():
    missing = sum(1 for w in W if w not in first)
    print(f"  {name}: needs {len(W):,} steps; {missing:,} never produced by the random machines")
(HERE / "steps2.json").write_text(json.dumps({
    "machines": N, "growth": growth, "vin": vin, "vout": vout,
    "steps": ["\u0001".join(w) for w in steps], "labels": labels, "first": [first.get(w, -1) for w in steps],
    "tests": {k: sorted(index[w] for w in v) for k, v in tests.items()}}, separators=(",", ":")))
print(f"{len(steps):,} steps ({len(first):,} from random machines), {len(vin)} input tokens, {len(vout)} outputs; "
      f"wrote steps2.json ({time.time() - t0:.0f}s)")
