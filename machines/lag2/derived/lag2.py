#!/usr/bin/env python3
"""lag2: a trained transformer as the step of the lagging-window decoder (torch-free witness).

lag1 derived transformer weights for lag0's rule table by construction. lag2 asks whether training finds them. An
ordinary small transformer (learned embeddings, one layer of softmax attention with 2 heads, a ReLU MLP) was trained by
Adam on next-token prediction, only on the windows the decoder meets while running binary increment on inputs of 1 to 3
bits (fixtures/train_lag2.ref.py, needs torch). This witness reloads the committed weights and checks, in pure Python:

    TL-1  REPRODUCES        the pure-Python forward pass gives the committed prediction on every reachable window
    TL-2  FITS              the trained step is correct on all of its training windows
    TL-3  FORCED            every window met by every input of 4..12 bits is already a training window
    TL-4  LENGTH-GENERALIZES  decoding with the trained step equals tm0 on every input of 1..12 bits and on random
                            256-bit inputs, far longer than anything in training
    TL-5  SEEDS             in all five seeds, fitting the data and running long inputs happen at the same checkpoint
    TL-6  UNFORCED          trained only on 1..2-bit inputs (56 needed windows missing), every seed fits and fails on
                            every longer input
    TL-7  ATTENTION         a measurement: what the trained heads attend to, against lag1's derived heads
    TL-8  DERIVED           a measurement: agreement with lag1's derived network on the windows the decoder meets, and
                            on every reachable window

The decoder calls the network once per distinct window (the step is a fixed function of its window), then runs.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "lag0" / "derived", ROOT / "lag1" / "derived", ROOT / "tm0"):
    sys.path.insert(0, str(p))
import lag0  # noqa: E402
import lag1  # noqa: E402
import tm0  # noqa: E402

DATA = json.loads((HERE.parent / "fixtures" / "lag2_training.json").read_text())
SRC = (ROOT / "lag0" / "fixtures" / "binary-increment.tm0").read_text()
VIN, VOUT = DATA["vin"], DATA["vout"]


def prog_for(bits: str) -> lag0.Program:
    return lag0.parse_tm0(SRC.replace("tape 1011", "tape " + bits))


def inputs(lo: int, hi: int) -> list[str]:
    return ["".join(b) for n in range(lo, hi + 1) for b in itertools.product("01", repeat=n)]


# ---------------------------------------------------------------- the trained network, in pure Python


class Trained:
    def __init__(self, w: dict):
        g = lambda k: w[k]
        self.E, self.P = g("E.weight"), g("P")
        self.Wq, self.Wk, self.Wv, self.Wo = g("Wq.weight"), g("Wk.weight"), g("Wv.weight"), g("Wo.weight")
        self.W1, self.b1, self.W2, self.b2 = g("W1.weight"), g("W1.bias")[0], g("W2.weight"), g("W2.bias")[0]
        self.U, self.bU = g("U.weight"), g("U.bias")[0]
        self.d, self.h = len(self.E[0]), DATA["architecture"]["heads"]
        self.cache: dict[tuple, str] = {}
        self.last_attention = None

    @staticmethod
    def lin(W, x, b=None):                      # torch Linear: W is (out, in)
        return [sum(wi * xi for wi, xi in zip(row, x)) + (b[i] if b else 0.0) for i, row in enumerate(W)]

    def logits(self, window):
        X = [[e + p for e, p in zip(self.E[VIN.index(t)], self.P[i])] for i, t in enumerate(window)]
        dh = self.d // self.h
        q = self.lin(self.Wq, X[-1])
        K = [self.lin(self.Wk, x) for x in X]
        V = [self.lin(self.Wv, x) for x in X]
        y, attn = [0.0] * self.d, []
        for hd in range(self.h):
            s = slice(hd * dh, (hd + 1) * dh)
            sc = [sum(a * b for a, b in zip(q[s], k[s])) / math.sqrt(dh) for k in K]
            m = max(sc); ex = [math.exp(v - m) for v in sc]; z = sum(ex); a = [e / z for e in ex]
            attn.append(a)
            for j, aj in enumerate(a):
                for c in range(s.start, s.stop):
                    y[c] += aj * V[j][c]
        self.last_attention = attn
        h = [xi + oi for xi, oi in zip(X[-1], self.lin(self.Wo, y))]
        hid = [max(0.0, v) for v in self.lin(self.W1, h, self.b1)]
        h = [a + b for a, b in zip(h, self.lin(self.W2, hid, self.b2))]
        return self.lin(self.U, h, self.bU)

    def predict(self, window) -> str:
        key = tuple(window)
        if key not in self.cache:
            z = self.logits(list(window))
            self.cache[key] = VOUT[max(range(len(z)), key=lambda i: z[i])]
        return self.cache[key]

    def forward(self, window):                  # the interface lag1.decode expects
        name = self.predict(window)
        return name, EMITS[name]


EMITS = {name: emits for name, emits, _ in lag1.Net(prog_for("1011")).vocab_out}


def target(w) -> str:
    out = lag0.rule(prog_for("1011"), *w) or []
    return next(name for name, emits in EMITS.items() if emits == out)


def met_windows(bits: str) -> set:
    p = prog_for(bits); table = lag0.compile_rules(p)
    ring = lag0.initial_ring(p); T, k, seen = [ring[-1], *ring], 0, set()
    while True:
        w = tuple(T[k:k + 3]); seen.add(w); out = table.get(w)
        if out is None:
            return seen
        T.extend(out); k += 1


def tm0_line(bits: str) -> str:
    v, steps, tape = tm0.run(tm0.parse_program(SRC.replace("tape 1011", "tape " + bits)), 1_000_000, False)
    lo, hi = min(min(tape), 0), max(max(tape), 0)
    return f"{v} steps={steps} tape={''.join(tape.get(i, '_') for i in range(lo, hi + 1))}"


def decode_line(net, bits: str) -> str:
    verdict, steps, _, tape = lag1.decode(prog_for(bits), net, "lag", max_steps=10_000_000)
    return f"{verdict} steps={steps} tape={tape}"


# ---------------------------------------------------------------- the rungs


def main() -> int:
    rung = sys.argv[1] if len(sys.argv) > 1 else "all"
    net = Trained(DATA["weights"])
    reach = sorted(lag1.reachable_windows(prog_for("1011")))
    train = [tuple(w.split("|")) for w in DATA["train_windows"]]
    main_runs, control_runs = DATA["runs"]["main"], DATA["runs"]["control"]
    ok = True

    def say(cond, line):
        nonlocal ok
        ok &= bool(cond)
        print(line if cond else "FAILED " + line)

    if rung in ("all", "TL-1"):
        same = sum(net.predict(w) == DATA["predictions"]["|".join(w)] for w in reach)
        say(same == len(reach), f"TL-1 REPRODUCES windows={len(reach)} same_as_committed={same}")
    if rung in ("all", "TL-2"):
        right = sum(net.predict(w) == target(w) for w in train)
        say(right == len(train), f"TL-2 FITS train_windows={len(train)} correct={right} min_margin={main_runs[0]['min_train_margin']}")
    if rung in ("all", "TL-3"):
        long_windows = set().union(*(met_windows(b) for b in inputs(4, 12)))
        new = long_windows - set(train)
        say(not new, f"TL-3 FORCED inputs=4..12bits({len(inputs(4, 12))}) windows_met={len(long_windows)} "
                     f"not_in_training={len(new)}")
    if rung in ("all", "TL-4"):
        net = Trained(DATA["weights"])          # fresh: count only the windows these runs need
        cases = inputs(1, 12)
        rng = random.Random(0)
        big = ["1" + "".join(rng.choice("01") for _ in range(255)) for _ in range(4)] + ["1" * 256]
        agree = sum(decode_line(net, b) == tm0_line(b) for b in cases)
        agree_big = sum(decode_line(net, b) == tm0_line(b) for b in big)
        say(agree == len(cases) and agree_big == len(big),
            f"TL-4 LENGTH-GENERALIZES trained_on=1..3bits agree_1..12bits={agree}/{len(cases)} "
            f"agree_256bits={agree_big}/{len(big)} distinct_windows_needed={len(net.cache)}")
    if rung in ("all", "TL-5"):
        same_ckpt = all(r["fit_at"] is not None and r["fit_at"] == r["long_runs_all_correct_at"] for r in main_runs)
        say(same_ckpt, "TL-5 SEEDS seeds=%d fit_at=%s long_runs_correct_at=%s no_gap=%s" % (
            len(main_runs), [r["fit_at"] for r in main_runs], [r["long_runs_all_correct_at"] for r in main_runs], same_ckpt))
    if rung in ("all", "TL-6"):
        fails = all(r["final_train_acc"] == 1.0 and r["final_long_run_acc"] == 0.0 for r in control_runs)
        say(fails, "TL-6 UNFORCED trained_on=1..2bits seeds=%d fit=%s long_run_acc=%s missing_windows=%d "
                   "guessed_right=%s" % (len(control_runs), all(r["final_train_acc"] == 1.0 for r in control_runs),
                                         [r["final_long_run_acc"] for r in control_runs],
                                         control_runs[0]["missing_windows"],
                                         [r["missing_predicted_correctly"] for r in control_runs]))
    if rung in ("all", "TL-7"):
        heads = [r["attention_by_head"] for r in main_runs]
        prev = [max(h[1] for h in s) for s in heads]
        two_back = [max(h[0] for h in s) for s in heads]
        print("TL-7 ATTENTION mean_attention_from_last_position "
              f"strongest_head_on_previous_token={prev} strongest_head_on_two_back={two_back} "
              "derived_lag1_heads=previous-token:1.0,two-back:1.0")
    if rung in ("all", "TL-8"):
        derived = lag1.Net(prog_for("1011"))
        dname = lambda w: derived.forward(list(w))[0]
        met = set().union(*(met_windows(b) for b in inputs(1, 12)))
        on_met = sum(net.predict(w) == dname(w) for w in met)
        on_reach = sum(net.predict(w) == dname(w) for w in reach)
        print(f"TL-8 DERIVED agree_on_windows_met={on_met}/{len(met)} agree_on_all_reachable={on_reach}/{len(reach)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
