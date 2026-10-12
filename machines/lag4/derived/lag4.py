#!/usr/bin/env python3
"""lag4: lag3's trained network draws cellular automata.

The cellular-automaton machine (ca.py) is written on the universal machine's tape as data; lag3's trained network,
which never saw it, runs the universal machine on the lagging-window decoder; every time the described machine finishes
a generation, its row is read off the tape. Checked against the automaton computed directly:

    D-1  MACHINE     the 8-state, 18-rule Turing machine computes Rules 90, 30, and 110 correctly (direct run)
    D-2  EXACT       the universal machine on the lagging decoder, exact rule, draws the same rows
    D-3  COVERAGE    every step those runs need occurs in lag3's training data (a measurement)
    D-4  NETWORK     lag3's trained network, as the decoder's step, draws the same rows
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
L3 = HERE.parents[1] / "lag3" / "derived"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(L3))
import ca  # noqa: E402
import lag3  # noqa: E402  (sets up the paths to lag0 and lag1)
import lag0  # noqa: E402
import utm  # noqa: E402

B = lag0.BOUNDARY
RULES = (90, 30, 110)
GENERATIONS = 16


def draw(rule: int, step, generations: int, max_passes: int = 2_000_000_000):
    """Run the CA machine on the universal machine with the lagging-window decoder and `step(window) -> tokens`.
    Returns (rows, forward passes, universal steps, machine steps). The transcript is compacted as it is read."""
    prog = lag3.utm_program(ca.ca_source(rule))
    ring = lag0.initial_ring(prog)
    T = [ring[-1], *ring]
    k = 0
    rows, passes, usteps, msteps = ["1"], 0, 0, 0          # the first row is the input
    last_state, pending, ring_start = "find", False, None
    while len(rows) < generations and passes < max_passes:
        w = (T[k], T[k + 1], T[k + 2])
        out = step(w)
        if not out:
            break
        if w[1] == B:
            if ring_start is not None:
                usteps += 1
                if pending:                            # the ring just completed is the first of a new machine step
                    pending = False
                    msteps += 1
                    cells = [lag0.split_head(t)[1] if lag0.split_head(t) else t for t in T[ring_start:len(T)] if t != B]
                    program = "".join(cells).split("$")[0]
                    if program.lstrip("_").startswith("*"):   # the described machine is in g: a generation is done
                        tape = "".join(cells).split("$")[1]
                        tape = "".join({"O": "0", "I": "1", "U": "_"}.get(c, c) for c in tape)
                        rows.append(tape[1:].rstrip("_"))
            ring_start = len(T)
        for t in out:
            h = lag0.split_head(t)
            if h:
                if h[0] == "find" and last_state == "fin":
                    pending = True
                last_state = h[0]
        T.extend(out)
        k += 1
        passes += 1
        if k > 1_000_000:                              # drop what the window has already read
            drop = k - 4
            T = T[drop:]; k -= drop
            ring_start -= drop
    return rows, passes, usteps, msteps


def main() -> int:
    t0 = time.time()
    ok = True

    def say(cond, line):
        nonlocal ok
        ok &= bool(cond)
        print((line if cond else "FAILED " + line) + f"   ({time.time() - t0:.0f}s)", flush=True)

    for rule in RULES:
        rows, steps = ca.tm_rows(rule, GENERATIONS)
        say(rows == ca.reference_rows(rule, GENERATIONS),
            f"D-1 MACHINE rule={rule} states=8 rules={len(ca.ca_rules(rule))} generations={GENERATIONS} machine_steps={steps}")

    data = json.loads((HERE.parents[1] / "lag3" / "fixtures" / "lag3_training.json").read_text())
    train = {tuple(w.split("\u0001")) for w in data["train_windows"]}
    sys.argv = sys.argv[:1]
    import lag3w  # noqa: E402  (the trained network, in pure Python)
    net = lag3w.Trained(data["weights"])
    import lag1  # noqa: E402
    emits = {n: e for n, e, _ in lag1.Net(lag3.utm_program(ca.ca_source(90))).vocab_out}

    for rule in RULES:
        prog = lag3.utm_program(ca.ca_source(rule))
        exact = lag3.Exact(prog)
        rows, passes, usteps, msteps = draw(rule, exact, GENERATIONS)
        say(rows == ca.reference_rows(rule, GENERATIONS),
            f"D-2 EXACT rule={rule} rows={len(rows)} forward_passes={passes:,} universal_steps={usteps:,} machine_steps={msteps}")
        unseen = set(exact.cache) - train
        print(f"D-3 COVERAGE rule={rule} distinct_steps={len(exact.cache)} never_in_training={len(unseen)}"
              f"   ({time.time() - t0:.0f}s)", flush=True)
        step = lambda w: emits[net.predict(w)]
        rows, passes, _, _ = draw(rule, step, GENERATIONS)
        say(rows == ca.reference_rows(rule, GENERATIONS),
            f"D-4 NETWORK rule={rule} rows={len(rows)}/{GENERATIONS} correct forward_passes={passes:,}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
