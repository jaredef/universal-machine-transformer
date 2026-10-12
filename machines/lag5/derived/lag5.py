#!/usr/bin/env python3
"""lag5: lag3's trained network runs the busy beaver champions.

BB(4)'s champion (Brady, 1983) runs 107 steps from a blank tape and halts leaving 13 ones, the most any 4-state,
two-symbol machine can. It is written on the universal machine's tape as data (with a one-state prelude that gives it
room on the left of a one-way tape, bb.py); lag3's trained network, which never saw it, runs the universal machine on
the lagging-window decoder. Checked, configuration by configuration, against the champion run directly:

    E-1  MACHINE     on a one-way tape, the prelude and the champion halt with the champion's proven step and ones
                     counts, and the same final tape as the two-way run
    E-2  EXACT       the universal machine on the lagging decoder, exact rule, passes through every one of the
                     champion's configurations, in order, and halts with the same tape
    E-3  COVERAGE    every step that run needs occurs in lag3's training data (a measurement)
    E-4  NETWORK     lag3's trained network, as the decoder's step, does the same
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
L3 = HERE.parents[1] / "lag3" / "derived"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(L3))
import bb  # noqa: E402
import lag3  # noqa: E402  (sets up the paths to lag0 and lag1)
import lag0  # noqa: E402
import utm  # noqa: E402

B = lag0.BOUNDARY
UNMARK = {v: k for k, v in utm.MARK.items()}


def direct(src: str):
    """Run on a one-way tape. Returns the configurations (state, head, tape) after each step, the first being the start."""
    p = lag0.parse_tm0(src)
    tape = {i: s for i, s in enumerate(p.tape) if s != p.blank}
    head, state = 0, p.start
    out = [(state, head, show(tape))]
    while state not in (p.accept, p.reject):
        state, write, move = p.rules[(state, tape.get(head, p.blank))]
        if write == p.blank:
            tape.pop(head, None)
        else:
            tape[head] = write
        head = max(0, head + {"L": -1, "R": 1, "S": 0}[move])
        out.append((state, head, show(tape)))
    return out


def show(tape: dict) -> str:
    return "".join(tape.get(i, "_") for i in range(max(tape) + 1)) if tape else ""


def trace(src: str, step, max_passes: int = 500_000_000):
    """Run the universal machine on the lagging-window decoder with `step(window) -> tokens`, and read the described
    machine's configuration off the ring each time it completes a step. Returns (configurations, passes, universal steps)."""
    p = lag0.parse_tm0(src)
    named = {s for (s, _) in p.rules} | {n for (n, _, _) in p.rules.values()}
    order = [p.start] + sorted(named - {p.start, p.accept, p.reject})       # utm.encode's entry order
    prog = lag3.utm_program(src)
    ring = lag0.initial_ring(prog)
    T = [ring[-1], *ring]
    k = passes = usteps = 0
    configs = [(p.start, 0, "".join(p.tape).rstrip("_"))]
    last_state, pending, ring_start = "find", False, None

    def read(cells, state):
        program, tape = "".join(cells).split("$")
        head = next(i for i, c in enumerate(tape) if c in UNMARK)
        tape = "".join(UNMARK.get(c, c) for c in tape).rstrip("_")
        if state is None:
            state = order[program[:program.index("*")].count("|")]
        return state, head, tape

    while passes < max_passes:
        w = (T[k], T[k + 1], T[k + 2])
        out = step(w)
        if not out:                                    # the universal machine halted: so did the described one
            h = lag0.split_head(w[1])
            final = utm.decode_tape(lag0.read_tape(prog, T[k + 1:], 0)).rstrip("_")
            configs.append((p.accept if h and h[0] == prog.accept else p.reject, None, final))
            return configs, passes, usteps
        if w[1] == B:
            if ring_start is not None:
                usteps += 1
                if pending:
                    pending = False
                    cells = [lag0.split_head(t)[1] if lag0.split_head(t) else t for t in T[ring_start:] if t != B]
                    configs.append(read(cells, None))
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
    return configs, passes, usteps


def same(got, want) -> bool:
    """Every configuration in order; the final one compares state and tape (the head is not read at halt)."""
    if len(got) != len(want):
        return False
    return all(g == w for g, w in zip(got[:-1], want[:-1])) and got[-1][::2] == want[-1][::2]


def main() -> int:
    t0 = time.time()
    ok = True

    def say(cond, line):
        nonlocal ok
        ok &= bool(cond)
        print((line if cond else "FAILED " + line) + f"   ({time.time() - t0:.0f}s)", flush=True)

    sources = {}
    for name in bb.CHAMPIONS:
        steps, ones, lo, hi, hist = bb.run_two_way(name)
        src = sources[name] = bb.bb_source(name)
        run = direct(src)
        pad = bb.pad(name)
        champ = run[pad + 1:]                          # from the moment the prelude hands over
        two_way = "".join(hist[-1][2].get(i, "_") for i in range(lo, hi + 1)).strip("_")
        say((steps, ones) == bb.EXPECTED[name] and len(champ) - 1 == steps and run[-1][0] == "qa"
            and run[-1][2].count("1") == ones and run[-1][2].strip("_") == two_way,
            f"E-1 MACHINE {name} champion_steps={len(champ) - 1} ones={run[-1][2].count('1')} prelude_steps={pad + 1} "
            f"final_tape={run[-1][2].strip('_')}")

    data = json.loads((L3.parent / "fixtures" / "lag3_training.json").read_text())
    train = {tuple(w.split("\u0001")) for w in data["train_windows"]}
    sys.argv = sys.argv[:1]
    import lag3w  # noqa: E402  (the trained network, in pure Python)
    net = lag3w.Trained(data["weights"])
    import lag1  # noqa: E402

    for name, src in sources.items():
        want = direct(src)
        prog = lag3.utm_program(src)
        exact = lag3.Exact(prog)
        got, passes, usteps = trace(src, exact)
        say(same(got, want), f"E-2 EXACT {name} configurations={len(got)}/{len(want)} forward_passes={passes:,} "
                             f"universal_steps={usteps:,}")
        unseen = set(exact.cache) - train
        print(f"E-3 COVERAGE {name} distinct_steps={len(exact.cache)} never_in_training={len(unseen)}"
              f"   ({time.time() - t0:.0f}s)", flush=True)
        emits = {n: e for n, e, _ in lag1.Net(prog).vocab_out}
        got, passes, _ = trace(src, lambda w: emits[net.predict(w)])
        say(same(got, want), f"E-4 NETWORK {name} configurations={len(got)}/{len(want)} correct "
                             f"forward_passes={passes:,} final_tape={got[-1][2].strip('_')}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
