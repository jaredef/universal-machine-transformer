#!/usr/bin/env python3
"""lag3: one trained network runs any machine, by learning the steps of a universal Turing machine (torch-free witness).

The universal machine (utm.py) reads a described machine and its input from the tape and simulates it. lag0 turns it
into lagging-window steps. One ordinary transformer was trained on those steps while the universal machine ran random
small machines on random short tapes (fixtures/train_lag3.ref.py). This witness reloads the committed weights and
checks, in pure Python, on held-out machines it never saw, with inputs longer than any in training:

    U-1  UNIVERSAL       tm0 itself, running the universal machine, reproduces each held-out machine
    U-2  LAGGED          the universal machine on the lagging-window decoder (exact rule) reproduces every test run
    U-3  DERIVED         lag1's derived transformer weights equal the exact rule on sampled windows the tests meet
    U-4  FITS            the trained network, recomputed here, matches its committed predictions and targets on a sample
                         of its training windows
    U-5  COVERAGE        a measurement: how many of the windows the tests meet never occur in training
    U-6  ONE-NETWORK     the trained network, as the decoder's step, runs every test run of every held-out machine
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "tm0"))
import lag3  # noqa: E402
import lag0  # noqa: E402
import lag1  # noqa: E402
import utm  # noqa: E402
import tm0  # noqa: E402

DATA = json.loads((HERE.parent / "fixtures" / "lag3_training.json").read_text())
VIN, VOUT = DATA["vin"], DATA["vout"]
VIX = {t: i for i, t in enumerate(VIN)}


def suite() -> list[tuple[str, str]]:
    rng = random.Random(7)
    bits = lambda n: "".join(rng.choice("01") for _ in range(n))
    cases = [("increment", "0" + bits(rng.randint(3, 9))) for _ in range(12)]
    cases += [("decrement", "1" + bits(rng.randint(3, 9))) for _ in range(12)]
    cases += [("invert", bits(rng.randint(7, 12))) for _ in range(8)]
    cases += [("parity", bits(rng.randint(6, 12))) for _ in range(10)]
    cases += [("div3", "1" + bits(rng.randint(5, 9))) for _ in range(12)]
    cases += [("unary-add", "1" * rng.randint(1, 7) + "_" + "1" * rng.randint(1, 7)) for _ in range(6)]
    for i in range(10):
        n = rng.randint(4, 7)
        half = bits((n + 1) // 2)
        word = half + half[: n // 2][::-1] if i % 2 == 0 else bits(n)
        cases.append(("palindrome", word))
    return cases


class Trained:
    def __init__(self, w):
        self.E, self.P = w["E.weight"], w["P"]
        self.Wq, self.Wk, self.Wv, self.Wo = w["Wq.weight"], w["Wk.weight"], w["Wv.weight"], w["Wo.weight"]
        self.W1, self.b1, self.W2, self.b2 = w["W1.weight"], w["W1.bias"][0], w["W2.weight"], w["W2.bias"][0]
        self.U, self.bU = w["U.weight"], w["U.bias"][0]
        self.d, self.h = len(self.E[0]), DATA["architecture"]["heads"]
        self.cache: dict[tuple, str] = {}

    @staticmethod
    def lin(W, x, b=None):
        return [sum(a * c for a, c in zip(row, x)) + (b[i] if b else 0.0) for i, row in enumerate(W)]

    def predict(self, window) -> str:
        if window in self.cache:
            return self.cache[window]
        X = [[e + p for e, p in zip(self.E[VIX[t]], self.P[i])] for i, t in enumerate(window)]
        dh = self.d // self.h
        q = self.lin(self.Wq, X[-1]); K = [self.lin(self.Wk, x) for x in X]; V = [self.lin(self.Wv, x) for x in X]
        y = [0.0] * self.d
        for hd in range(self.h):
            lo, hi = hd * dh, (hd + 1) * dh
            sc = [sum(a * b for a, b in zip(q[lo:hi], k[lo:hi])) / math.sqrt(dh) for k in K]
            m = max(sc); ex = [math.exp(v - m) for v in sc]; z = sum(ex)
            for j in range(3):
                a = ex[j] / z
                for c in range(lo, hi):
                    y[c] += a * V[j][c]
        h = [a + b for a, b in zip(X[-1], self.lin(self.Wo, y))]
        hid = [max(0.0, v) for v in self.lin(self.W1, h, self.b1)]
        h = [a + b for a, b in zip(h, self.lin(self.W2, hid, self.b2))]
        z = self.lin(self.U, h, self.bU)
        self.cache[window] = VOUT[max(range(len(z)), key=z.__getitem__)]
        return self.cache[window]


def main() -> int:
    t0 = time.time()
    ok = True

    def say(cond, line):
        nonlocal ok
        ok &= bool(cond)
        print((line if cond else "FAILED " + line) + f"   ({time.time() - t0:.0f}s)", flush=True)

    cases = suite()
    ref = {c: lag3.reference(lag3.held_out_source(*c)) for c in cases}
    names = sorted({n for n, _ in cases})
    emits = {n: e for n, e, _ in lag1.Net(lag3.utm_program(lag3.held_out_source("invert", "0"))).vocab_out}

    # U-1: tm0 itself runs the universal machine (one small case per held-out machine)
    good = 0
    for name in names:
        t = next(t for n, t in cases if n == name)
        small = t[:4] if name != "unary-add" else "1_1"
        src = lag3.held_out_source(name, small)
        enc = utm.encode(lag0.parse_tm0(src), small)
        v, _, tape = tm0.run(tm0.parse_program(utm.utm_source(enc)), 50_000_000, False)
        final = "".join(tape.get(i, "_") for i in range(min(tape), max(tape) + 1))
        good += (v, utm.decode_tape(final)) == lag3.reference(src)
    rules = utm.utm_rules()
    say(good == len(names), f"U-1 UNIVERSAL machine={len({r[0] for r in rules} | {r[2] for r in rules})}states,"
                            f"{len(utm.ALL)}symbols,{len(rules)}rules tm0_runs_it_correctly={good}/{len(names)}")

    # U-2: the universal machine on the lagging decoder, exact rule; collect every window the tests meet
    met: set = set()
    good = steps = 0
    for c in cases:
        p = lag3.utm_program(lag3.held_out_source(*c))
        v, tape, n = lag3.lag_run(p, lag3.Exact(p), seen=met)
        good += (v, tape) == ref[c]; steps += n
    say(good == len(cases), f"U-2 LAGGED test_runs={len(cases)} correct={good} decode_steps={steps:,} windows_met={len(met)}")

    # U-3: lag1's derived weights on a sample of those windows
    p0 = lag3.utm_program(lag3.held_out_source("invert", "0"))
    net = lag1.Net(p0)
    sample = random.Random(3).sample(sorted(met), min(250, len(met)))
    same = sum((net.forward(list(w))[1]) == (lag0.rule(p0, *w) or []) for w in sample)
    say(same == len(sample), f"U-3 DERIVED sampled_windows={len(sample)} derived_equals_rule={same}")

    # U-4: the trained network, recomputed here, on a sample of its training windows
    tr = Trained(DATA["weights"])
    train = [tuple(w.split("\u0001")) for w in DATA["train_windows"]]
    pick = random.Random(4).sample(range(len(train)), 300)
    match = sum(tr.predict(train[i]) == DATA["train_predictions"][i] for i in pick)
    right = sum(tr.predict(train[i]) == {tuple(e): n for n, e in emits.items()}[tuple(lag0.rule(p0, *train[i]) or [])]
                for i in pick)
    run0 = DATA["runs"][0]
    say(match == len(pick) and right == len(pick),
        f"U-4 FITS train_windows={len(train)} seed0_train_acc={run0['final_train_acc']} errors={run0['train_errors']} "
        f"sample={len(pick)} recomputed_matches_committed={match} correct={right} "
        f"seeds_fit_at={[r['fit_at'] for r in DATA['runs']]}")

    # U-5: coverage
    unseen = met - set(train)
    print(f"U-5 COVERAGE windows_met_by_tests={len(met)} never_in_training={len(unseen)}   ({time.time() - t0:.0f}s)",
          flush=True)

    # U-6: one trained network runs every held-out machine
    for w in met:
        tr.predict(w)                                 # evaluate every window once, then decode by lookup
    step = lambda w: emits[tr.cache[w]] if w in tr.cache else emits[tr.predict(w)]
    per = {n: [0, 0] for n in names}
    for c in cases:
        p = lag3.utm_program(lag3.held_out_source(*c))
        got = lag3.lag_run(p, step)
        per[c[0]][0] += got[:2] == ref[c]; per[c[0]][1] += 1
    total = sum(a for a, _ in per.values())
    say(total == len(cases), "U-6 ONE-NETWORK " + " ".join(f"{n}={a}/{b}" for n, (a, b) in per.items()) +
        f" total={total}/{len(cases)} longest_input={max(len(t) for _, t in cases)} training_tapes<=6")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
