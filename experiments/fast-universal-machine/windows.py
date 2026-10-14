#!/usr/bin/env python3
"""The lagging-window steps (windows) a universal machine takes, read off a direct run.

On the lagging-window decoder the transcript is the universal machine's configurations, each a ring '#' c1 ... cn with
the head written as state:symbol, laid end to end (lag0). The windows the network reads are exactly the consecutive
triples of that sequence. From one configuration to the next only the cells around the head change, along with the
joins between rings, so every new triple can be found locally: collecting costs one universal-machine step per step,
not one forward pass per token. `check()` compares this with the actual decoder on small programs.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for d in ("machines/lag3/derived", "machines/lag0/derived"):
    sys.path.insert(0, str(ROOT / d))
import lag0  # noqa: E402

B = lag0.BOUNDARY


def collect(rules, tape, start="find", accept="qa", reject="qr", blank="_", max_steps=10 ** 6, into=None):
    """Run the universal machine (rules: {(state, symbol): (next, write, move)}) directly on `tape`, adding every window
    the lagging decoder would read to `into` (a set). Returns (windows, universal steps, halted)."""
    W = into if into is not None else set()
    t = list(tape) or [blank]
    h, q = 0, start
    hd = lag0.head

    def tok(i):                                              # cell i of the current ring (the head written state:symbol)
        return hd(q, t[i]) if i == h else t[i]

    n0 = len(t)
    seq = [tok(n0 - 1), B] + [tok(i) for i in range(n0)]       # the decoder starts by reading the ring's last cell, '#'
    for i in range(len(seq) - 2):
        W.add((seq[i], seq[i + 1], seq[i + 2]))
    steps, pending = 0, []                                    # the newest ring's triples near its head: (middle index, triple)
    while steps < max_steps:
        r = rules.get((q, t[h]))
        if r is None or q in (accept, reject):               # the decoder stops at the head: it never reads past it
            W.update(tr for mid, tr in pending if mid <= h)
            return W, steps, True
        W.update(tr for _, tr in pending)
        n, w, m = r
        L = len(t)
        last_old, prev_last_old = tok(L - 1), (tok(L - 2) if L > 1 else B)
        old_h = h
        t[h] = w
        h += {"L": -1, "R": 1, "S": 0}[m]
        if h < 0:
            t.insert(0, blank); h = 0; old_h += 1
        if h == len(t):
            t.append(blank)
        q = n
        L = len(t)
        # the joins: the old ring's last two cells, '#', the new ring's first two cells (always read)
        W.add((prev_last_old, last_old, B)); W.add((last_old, B, tok(0)))
        if L > 1:
            W.add((B, tok(0), tok(1)))
        # the cells around the head, before and after the step (a triple that is new must touch a changed cell),
        # and the ring's end; held until the next step shows how far the decoder reads this ring
        lo, hi = max(0, min(old_h, h) - 2), min(L, max(old_h, h) + 3)
        pending = [(i + 1, (tok(i), tok(i + 1), tok(i + 2))) for i in range(lo, hi - 2)]
        if L >= 2:
            pending.append((L - 1, (tok(L - 2), tok(L - 1), B)))
        steps += 1
    W.update(tr for _, tr in pending)
    return W, steps, False


def check(progs):
    """Compare collect() with the actual lagging-window decoder (lag3.lag_run) on universal-machine programs."""
    import lag3
    for name, prog in progs:
        seen = set()
        lag3.lag_run(prog, lag3.Exact(prog), max_decode=3_000_000, seen=seen)
        rules = {k: (v[0], v[1], v[2]) for k, v in prog.rules.items()}
        mine, _, _ = collect(rules, "".join(prog.tape))
        extra, missing = mine - seen, seen - mine
        print(f"{name}: decoder {len(seen)} windows, direct {len(mine)}; missing {len(missing)}, extra {len(extra)}"
              + (f"  e.g. missing {sorted(missing)[:3]} extra {sorted(extra)[:3]}" if missing or extra else ""))
