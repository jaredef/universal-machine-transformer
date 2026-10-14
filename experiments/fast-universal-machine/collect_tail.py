#!/usr/bin/env python3
"""Close the tail of version 2's step table with "tour" machines.

The steps random machines miss in programs of hundreds of states come from a jump crossing the rule being read, whose
next-state field is partly used: they depend on the current state's number, the target's bit pattern, and the jump's level
together, so covering them needs many different transitions executed in programs of hundreds of states. Random machines
of that size run only a few dozen steps within collect2.py's budget. A tour machine instead gives every state one rule
that sends it to a fresh random state (random symbol written, random move), so each run executes hundreds of different
transitions, each with a random target.

Adds the tours' steps to steps2.json's random-machine steps (as later machines), and checks coverage on programs the tours
were not chosen for: the dragon curve's machine spread among 128 states, Collatz among 200, Rule 110 among 256.

    python3 collect_tail.py [tours]       writes steps3.json
"""
import json, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import windows, utm2, check, ca  # noqa: E402
import lag0  # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 200
RULES = {(s, r): (n, w, m) for s, r, n, w, m in utm2.utm_rules()}
UTM = lag0.parse_tm0(utm2.utm_source("_"))
t0 = time.time()

S2 = json.loads((HERE / "steps2.json").read_text())
first = {tuple(s.split("\u0001")): f for s, f in zip(S2["steps"], S2["first"]) if f >= 0}
base, growth = S2["machines"], []
print(f"{len(first):,} steps from {base:,} random machines", flush=True)


def tour(rng, S):
    """Every state reads 0 1 _ alike: write a random symbol, move at random, go to a random other state; one state of the
    tour, chosen at random, accepts instead, so the run is a random walk of length about S."""
    states = [f"t{i}" for i in range(S)]
    end = rng.choice(states[1:])
    lines = []
    for q in states:
        for sym in "01_":
            nxt = "qa" if q == end else rng.choice([s for s in states if s != q])
            lines.append(f"{q} {sym} -> {nxt} {rng.choice('01_')} {rng.choice('LRRS')}")
    tape = "".join(rng.choice("01_") for _ in range(rng.randint(1, 8)))
    return lag0.parse_tm0(f"blank _\nstart t0\naccept qa\nreject qr\ntape {tape}\n---\n" + "\n".join(lines) + "\n"), tape


rng = random.Random(5)
for i in range(T):
    S = rng.randint(64, 256)
    prog, tape = tour(rng, S)
    W, steps, halted = windows.collect(RULES, utm2.encode(prog, tape), max_steps=4_000_000)
    new = 0
    for w in W:
        if w not in first:
            first[w] = base + i; new += 1
    growth.append((base + i + 1, len(first)))
    if (i + 1) % 10 == 0 or i < 5:
        print(f"tour {i + 1} ({S} states, {steps:,} universal steps): +{new:,} new; {len(first):,} steps ({time.time() - t0:.0f}s)", flush=True)

tests = {k: set(tuple(S2["steps"][j].split("\u0001")) for j in v) for k, v in S2["tests"].items()}


def spread(rules, start, tape, S, seed):
    prog = check.source(rules + [f"z{j} 0 -> z{j} 0 R" for j in range(S - len({r.split()[0] for r in rules}))], start, tape)
    real = utm2.states_of(prog); rest = real[1:]; random.Random(seed).shuffle(rest)
    W, _, _ = windows.collect(RULES, utm2.encode(prog, tape, [real[0]] + rest), max_steps=10 ** 8)
    return W


dragon = ["s _ -> inc _ R", "inc 1 -> inc 0 R", "inc 0 -> back 1 L", "inc _ -> qa _ S", "back 0 -> back 0 L", "back 1 -> back 1 L",
          "back _ -> low _ R", "low 0 -> low 0 R", "low 1 -> nxt 1 R", "nxt 0 -> sayR 0 L", "nxt _ -> sayR _ L", "nxt 1 -> sayL 1 L",
          "sayR 1 -> home 1 L", "sayL 1 -> home 1 L", "home 0 -> home 0 L", "home _ -> inc _ R"]
collatz = ["s _ -> s _ R", "s 0 -> s _ R", "s 1 -> one 0 R", "one _ -> fix _ L", "one 0 -> c1 0 R", "one 1 -> c2 1 R", "fix 0 -> qa 1 S",
           "c0 0 -> c0 0 R", "c0 1 -> c1 1 R", "c0 _ -> back _ L", "c1 0 -> c0 1 R", "c1 1 -> c2 0 R", "c1 _ -> back 1 L",
           "c2 0 -> c1 0 R", "c2 1 -> c2 1 R", "c2 _ -> c1 0 R", "back 0 -> back 0 L", "back 1 -> back 1 L", "back _ -> s _ R"]
tests["held out: dragon among 128 states"] = spread(dragon, "s", "_" + "0" * 6, 128, 1)
tests["held out: collatz(27) among 200 states"] = spread(collatz, "s", "_11011", 200, 2)
tests["held out: rule 110 among 256 states"] = spread(ca.ca_rules(110), "g", "_1", 256, 3)
print(f"held-out programs collected ({time.time() - t0:.0f}s)", flush=True)

steps = sorted(set(first) | set().union(*tests.values()))
index = {w: k for k, w in enumerate(steps)}
labels = []
for w in steps:
    out = lag0.rule(UTM, *w)
    labels.append(" ".join(out) if out is not None else "<halt>")
for name, W in tests.items():
    print(f"  {name}: needs {len(W):,} steps; {sum(1 for w in W if w not in first):,} never produced by random machines or tours")
(HERE / "steps3.json").write_text(json.dumps({
    "machines": base, "tours": T, "growth": S2["growth"] + growth, "vin": sorted({t for w in steps for t in w}),
    "vout": sorted(set(labels)), "steps": ["\u0001".join(w) for w in steps], "labels": labels,
    "first": [first.get(w, -1) for w in steps], "tests": {k: sorted(index[w] for w in v) for k, v in tests.items()}},
    separators=(",", ":")))
print(f"{len(steps):,} steps ({len(first):,} from random machines and tours); wrote steps3.json ({time.time() - t0:.0f}s)")
